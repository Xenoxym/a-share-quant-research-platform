"""Fetch 龙虎榜 summary data — Step 2.

Key behaviours
--------------
- **One range call**: ``stock_lhb_detail_em`` supports arbitrary date ranges and
  returns full history.  We compute all missing trading dates, then make a
  single ``start_date / end_date`` API call covering the entire gap instead of
  one call per day.  This reduces ~252 calls/year → 1 call/run.
- **Retry**: 3 attempts on the range call with 1s / 2s / 4s back-off.
- **Fail collection**: a failed range is recorded in
  ``data/failures/failures_lhb_summary.csv`` as one row with key
  ``trade_date = "YYYYMMDD~YYYYMMDD"``.
- **Restore**: ``restore_lhb_summary()`` re-fetches any recorded failures.
- **Incremental / backfill**: uses date-set-difference so it handles both
  forward appends and historical gap-fills.
- **Idempotent merge**: deduped on ``(trade_date, stock_code)``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from src.ingestion.failure_tracker import with_retry, summary_tracker
from src.utils.calendar import get_trading_dates
from src.utils.io import load_parquet, save_parquet
from src.utils.logging import get_logger

logger = get_logger("fetch_lhb_summary")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SAVE = _PROJECT_ROOT / "data" / "raw" / "lhb_summary.parquet"
_DEDUP_KEYS = ["trade_date", "stock_code"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_lhb_summary(
    start_date: str,
    end_date: str,
    source: str = "akshare",
    save_path: str | None = None,
    incremental: bool = True,
) -> pd.DataFrame:
    """Fetch 龙虎榜 summary data for a date range (Step 2).

    One ``stock_lhb_detail_em`` call covers the entire missing date range.
    The API returns full history so there is no 90-day limitation.

    Parameters
    ----------
    start_date, end_date : str
        Requested range ``"YYYY-MM-DD"``.
    source : str
        ``"akshare"`` only (no fallback by design).
    save_path : str or None
        Destination parquet.  Default: ``data/raw/lhb_summary.parquet``.
    incremental : bool
        When True, skip dates already present in the saved parquet.

    Returns
    -------
    pd.DataFrame
        Combined (existing + new) LHB summary, deduped on
        ``(trade_date, stock_code)``.
    """
    out_path = Path(save_path) if save_path else _DEFAULT_SAVE
    tracker = summary_tracker(_load_failures_dir())
    client = _get_client(source)

    existing_df = _load_existing(out_path) if incremental else None

    missing_dates = _get_missing_trading_dates(start_date, end_date, existing_df)
    if not missing_dates:
        logger.info(
            f"LHB summary up-to-date for [{start_date}, {end_date}] — nothing to fetch"
        )
        result = existing_df if existing_df is not None else pd.DataFrame()
        result.attrs["update_report"] = {"status": "pass", "rows_added": 0, "unresolved_dates": []}
        return result

    fetch_start = missing_dates[0]
    fetch_end = missing_dates[-1]
    logger.info(
        f"Fetching LHB summary: {len(missing_dates)} missing dates, "
        f"range [{fetch_start}, {fetch_end}], source={source}"
    )

    # Failure key encodes the full date range as a single tracker entry.
    # format "YYYYMMDD~YYYYMMDD" — can be parsed back for restore.
    range_key = {
        "trade_date": f"{fetch_start.replace('-', '')}~{fetch_end.replace('-', '')}"
    }

    new_df: pd.DataFrame | None = None
    error = None
    try:
        raw = with_retry(client.fetch_lhb_summary, fetch_start, fetch_end)
        if raw is not None and not raw.empty:
            # Keep only dates we actually asked for (API may return slightly
            # outside the range in edge cases)
            missing_set = set(missing_dates)
            new_df = raw[raw["trade_date"].isin(missing_set)].copy()
            logger.info(f"Received {len(raw)} rows → {len(new_df)} within requested dates")
        received = set(new_df.trade_date) if new_df is not None else set()
        absent = sorted(set(missing_dates) - received)
        if absent:
            if new_df is not None and not new_df.empty:
                save_parquet(new_df, out_path.with_suffix(".partial.parquet"))
            new_df = None
            raise RuntimeError(f"Incomplete summary response; missing trading dates: {absent[:10]}")
        tracker.resolve(range_key)
    except Exception as exc:
        error = str(exc)
        tracker.record(range_key, str(exc))
        logger.warning(
            f"LHB summary range [{fetch_start}, {fetch_end}] failed after all retries: {exc}\n"
            f"Recorded in {tracker.summary_line()}\n"
            f"Run 'python main.py --step restore_lhb_summary' to retry."
        )

    new_chunks = [new_df] if new_df is not None and not new_df.empty else []
    result = (_merge_and_save(existing_df, new_chunks, out_path) if new_chunks
              else existing_df if existing_df is not None else pd.DataFrame())
    unresolved = _get_missing_trading_dates(start_date, end_date, result)
    result.attrs["update_report"] = {
        "status": "fail" if error or unresolved else "pass",
        "rows_added": len(result) - (len(existing_df) if existing_df is not None else 0),
        "unresolved_dates": unresolved,
        "error": error,
    }
    return result


def restore_lhb_summary(
    source: str = "akshare",
    save_path: str | None = None,
) -> pd.DataFrame:
    """Re-fetch all ranges recorded in ``failures_lhb_summary.csv``.

    Each failure record has ``trade_date = "YYYYMMDD~YYYYMMDD"`` encoding the
    original range.  The range is re-fetched as a single API call.

    Returns
    -------
    pd.DataFrame
        Updated combined LHB summary after restore.
    """
    out_path = Path(save_path) if save_path else _DEFAULT_SAVE
    tracker = summary_tracker(_load_failures_dir())
    pending = tracker.get_pending()

    if pending.empty:
        logger.info("No pending LHB summary failures to restore.")
        existing = _load_existing(out_path)
        return existing if existing is not None else pd.DataFrame()

    client = _get_client(source)
    existing_df = _load_existing(out_path)
    logger.info(f"Restoring {len(pending)} failed LHB summary range(s) (source={source})")

    new_chunks: list[pd.DataFrame] = []
    restored = 0

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="Restoring LHB Summary"):
        range_str = str(row["trade_date"])
        key = {"trade_date": range_str}

        # Parse "YYYYMMDD~YYYYMMDD" back into dates
        fetch_start, fetch_end = _parse_range_key(range_str)
        if fetch_start is None:
            logger.warning(f"Unparseable failure key '{range_str}' — skipping")
            continue

        try:
            raw = with_retry(client.fetch_lhb_summary, fetch_start, fetch_end)
            required = set(get_trading_dates(fetch_start, fetch_end))
            received = set(raw.trade_date) if raw is not None and not raw.empty else set()
            if not received or required - received:
                raise RuntimeError("Restore response is empty or misses requested trading dates")
            new_chunks.append(raw.loc[raw.trade_date.isin(required)].copy())
            tracker.resolve(key)
            restored += 1
        except Exception as exc:
            tracker.record(key, str(exc))
            logger.warning(f"Restore failed for {range_str}: {exc}")

    logger.info(
        f"Restore complete: {restored}/{len(pending)} ranges recovered. "
        f"{tracker.summary_line()}"
    )
    return _merge_and_save(existing_df, new_chunks, out_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_existing(out_path: Path) -> pd.DataFrame | None:
    if not out_path.exists():
        return None
    try:
        df = load_parquet(str(out_path))
        return df if not df.empty else None
    except Exception as exc:
        raise RuntimeError("Existing LHB summary is unreadable; refusing to overwrite it") from exc


def _get_missing_trading_dates(
    start_date: str,
    end_date: str,
    existing_df: pd.DataFrame | None,
) -> list[str]:
    """Trading dates in [start, end] not yet present in existing_df.

    Uses SimTradeData trade_days.parquet (via calendar module) for accuracy.
    Handles both forward-append and historical backfill.
    """
    all_trading = set(get_trading_dates(start_date, end_date))
    if not all_trading:
        return []

    if existing_df is None or "trade_date" not in existing_df.columns:
        return sorted(all_trading)

    existing_dates = set(existing_df["trade_date"].astype(str).unique())
    return sorted(all_trading - existing_dates)


def _merge_and_save(
    existing_df: pd.DataFrame | None,
    new_chunks: list[pd.DataFrame],
    out_path: Path,
) -> pd.DataFrame:
    # A refreshed event replaces its previous coherent report, including all
    # its reasons. Mixing a stale aggregate with fresh source rows can retain
    # old numeric values or invent a reporting window.
    fresh = [df for df in new_chunks if df is not None and not df.empty]
    if fresh and existing_df is not None and not existing_df.empty:
        refreshed = pd.MultiIndex.from_frame(pd.concat(fresh)[_DEDUP_KEYS])
        existing_df = existing_df.loc[~pd.MultiIndex.from_frame(existing_df[_DEDUP_KEYS]).isin(refreshed)]
    parts = [df for df in [existing_df] + fresh if df is not None and not df.empty]
    if not parts:
        logger.warning("No LHB summary data to save")
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    before = len(combined)

    # When the same (trade_date, stock_code) appears multiple times because the
    # stock qualified for multiple LHB criteria on the same day, concatenate all
    # reason strings rather than silently dropping any of them.
    combined = _aggregate_multi_reason_rows(combined)

    after = len(combined)
    if before > after:
        logger.info(
            f"Dedup: {before} → {after} rows "
            f"({before - after} multi-reason rows merged, lhb_reason concatenated with '；')"
        )

    if "trade_date" in combined.columns and "stock_code" in combined.columns:
        combined = combined.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)

    if fresh:
        # Preserve all source reporting windows for later research. The event
        # table selects one report; that selection must not erase alternatives.
        reports_path=out_path.with_name(out_path.stem+'_reports.parquet')
        source_rows=pd.concat(fresh,ignore_index=True)
        if reports_path.exists():
            prior=load_parquet(reports_path)
            refreshed=pd.MultiIndex.from_frame(source_rows[_DEDUP_KEYS])
            prior=prior.loc[~pd.MultiIndex.from_frame(prior[_DEDUP_KEYS]).isin(refreshed)]
            source_rows=pd.concat([prior,source_rows],ignore_index=True)
        save_parquet(source_rows.sort_values(_DEDUP_KEYS).reset_index(drop=True),reports_path)
    save_parquet(combined, str(out_path))
    logger.info(f"LHB summary saved: {out_path} ({len(combined):,} rows)")
    return combined


def _aggregate_multi_reason_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate (trade_date, stock_code) rows caused by multiple LHB
    criteria on the same day.

    Text columns (lhb_reason, lhb_interpret) are joined with '；' so that all
    reasons are preserved for downstream filtering.  All other columns use the
    one coherent source report, preferring a single-day window. Different reporting
    windows can have different numeric values; their cells must not be mixed.
    """
    if "trade_date" not in df.columns or "stock_code" not in df.columns:
        return df

    # Reports from different windows do NOT have identical numeric fields.
    # Select one whole source row, rather than groupby.first() independently
    # taking non-null cells from different reports. Prefer a single-day report.
    from src.data_sources.reporting_windows import reporting_window_days
    keys = ["trade_date", "stock_code"]
    work = df.copy()
    reason = work.get("lhb_reason", pd.Series("", index=work.index))
    work["summary_selected_reason"] = work.get("summary_selected_reason", reason).fillna(reason)
    work["summary_window_days"] = work.summary_selected_reason.map(reporting_window_days)
    if not work.duplicated(keys).any():
        return work
    work["_window_priority"] = work.summary_window_days.where(work.summary_window_days.gt(0),999)
    # Margin-monitoring disclosures may coexist with a normal LHB report but
    # expose only financing buys. Do not prefer them over a two-sided report.
    work['_special_report'] = work.summary_selected_reason.fillna('').str.contains('融资买入|融券卖出',regex=True).astype(int)
    result = work.sort_values(['_special_report',"_window_priority"],kind="stable").drop_duplicates(keys).drop(columns=['_special_report',"_window_priority"])
    # Only duplicate events need reason aggregation; untouched events keep the
    # exact original row and do not pay for per-group Python callbacks.
    duplicated = df[df.duplicated(keys,keep=False)]
    for col in ("lhb_reason", "lhb_interpret"):
        if col in df:
            text = duplicated.groupby(keys,sort=False)[col].agg(
                lambda values: "；".join(v for v in values.dropna().astype(str).unique() if v))
            idx = pd.MultiIndex.from_frame(result[keys])
            joined = text.reindex(idx)
            selected = idx.isin(text.index)
            result.loc[selected,col] = joined[selected].to_numpy()
    return result.reset_index(drop=True)


def _parse_range_key(range_str: str) -> tuple[str | None, str | None]:
    """Parse ``"YYYYMMDD~YYYYMMDD"`` → ``("YYYY-MM-DD", "YYYY-MM-DD")``."""
    try:
        parts = range_str.split("~")
        if len(parts) == 2:
            start = pd.Timestamp(parts[0]).strftime("%Y-%m-%d")
            end = pd.Timestamp(parts[1]).strftime("%Y-%m-%d")
            return start, end
    except Exception:
        pass
    return None, None


def _load_failures_dir() -> Path:
    config_path = _PROJECT_ROOT / "config" / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        rel = cfg.get("failures", {}).get("dir", "data/failures")
        return (_PROJECT_ROOT / rel).resolve()
    except Exception:
        return _PROJECT_ROOT / "data" / "failures"


def _get_client(source: str):
    if source == "akshare":
        from src.data_sources.akshare_client import AKShareClient
        return AKShareClient()
    raise ValueError(
        f"Unknown LHB summary source: '{source}'. "
        "Only 'akshare' is supported (no fallback by design)."
    )
