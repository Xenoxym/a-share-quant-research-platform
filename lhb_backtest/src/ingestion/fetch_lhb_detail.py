"""Fetch LHB broker detail with verified reporting windows.

The default route acquires all summary events from the same EastMoney reports
used by AKShare, with bounded monthly pagination, per-side counts, resumable
source partitions, and an explicit event-coverage report. It preserves direct
buy/sell amounts and source report types. A canonical complete report is chosen
per event without mixing daily and multi-day boards.

Explicit events_df/reason_keywords retain the selected-event adapter. Existing
partial buy/sell events are retried; unreadable existing files fail closed.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from src.cleaning.clean_lhb import filter_lhb_events
from src.ingestion.failure_tracker import with_retry, detail_tracker
from src.utils.io import load_parquet, save_parquet
from src.utils.logging import get_logger

logger = get_logger("fetch_lhb_detail")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SAVE = _PROJECT_ROOT / "data" / "raw" / "lhb_broker_detail.parquet"
_DEFAULT_SUMMARY = _PROJECT_ROOT / "data" / "raw" / "lhb_summary.parquet"
# Dedup key: one row = one broker on one side for a stock on a date
_DEDUP_KEYS = ["trade_date", "stock_code", "flag", "rank"]
_INTER_REQUEST_SLEEP = 0.3  # seconds between pair fetches (each pair = 2 API calls)
_CHECKPOINT_EVERY = 200  # pairs between incremental parquet saves (crash safety)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_lhb_detail(
    events_df: pd.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    source: str = "akshare",
    save_path: str | None = None,
    incremental: bool = True,
    reason_keywords: list[str] | None = None,
    summary_path: str | None = None,
) -> pd.DataFrame:
    """Fetch broker-seat buy/sell detail (Step 3); default scope is all events.

    For each qualifying (stock_code, trade_date) pair, two API calls are
    made — ``flag="买入"`` for the top-5 buyers and ``flag="卖出"`` for the
    top-5 sellers.  Both are stored together with a ``flag`` column.

    Parameters
    ----------
    events_df : pd.DataFrame or None
        Pre-filtered event DataFrame with ``stock_code`` and ``trade_date``.
        If None and reason_keywords is None, acquire the full summary universe.
    start_date, end_date : str or None
        Date bounds applied when loading the summary from disk.
    source : str
        ``"akshare"`` only (no fallback by design).
    save_path : str or None
        Destination parquet.  Default: ``data/raw/lhb_broker_detail.parquet``.
    incremental : bool
        Skip pairs already present in the saved parquet.
    reason_keywords : list[str] or None
        Optional opt-in selected-event route; None uses the full source archive.

    Returns
    -------
    pd.DataFrame
        Combined (existing + new) broker detail, deduped on
        ``(trade_date, stock_code, flag, rank)``.
    """
    out_path = Path(save_path) if save_path else _DEFAULT_SAVE
    if events_df is None and reason_keywords is None and source == 'akshare':
        from src.ingestion.lhb_detail_archive import refresh_detail_archive
        return refresh_detail_archive(Path(summary_path) if summary_path else _DEFAULT_SUMMARY,
                                      out_path,start_date,end_date)
    tracker = detail_tracker(_load_failures_dir())
    client = _get_client(source)

    # Resolve + filter the events universe
    if events_df is None and summary_path is not None:
        events_df = load_parquet(summary_path)
    events_df = _resolve_events(events_df, start_date, end_date, reason_keywords)
    if events_df is None or events_df.empty:
        logger.warning("No qualifying LHB events to fetch broker detail for")
        return pd.DataFrame()

    all_pairs = _to_pair_set(events_df)

    existing_df = _load_existing(out_path)
    if incremental and existing_df is not None and not existing_df.empty:
        existing_pairs = _complete_pair_set(existing_df)
        missing_pairs = all_pairs - existing_pairs
        logger.info(
            f"Incremental: {len(all_pairs)} filtered pairs, "
            f"{len(existing_pairs)} already fetched, "
            f"{len(missing_pairs)} to fetch"
        )
    else:
        missing_pairs = all_pairs
        logger.info(f"Fetching broker detail for {len(missing_pairs)} pairs, source={source}")

    if not missing_pairs:
        logger.info("All pairs already fetched — nothing to do.")
        return existing_df if existing_df is not None else pd.DataFrame()

    new_chunks: list[pd.DataFrame] = []
    fetch_failed = 0
    since_checkpoint = 0

    # Newest first: if a long backfill is interrupted, the most recent (and
    # most relevant) events are already covered; older ones resume next run.
    for trade_date, stock_code in tqdm(
        sorted(missing_pairs, reverse=True), desc="LHB Detail", unit="pair"
    ):
        key = {"trade_date": trade_date, "stock_code": stock_code}
        try:
            # One retry-wrapped call covers both buy and sell sides together
            df = with_retry(_fetch_both_sides, client, stock_code, trade_date)
            if df is not None and not df.empty:
                new_chunks.append(df)
            tracker.resolve(key)
        except Exception as exc:
            fetch_failed += 1
            tracker.record(key, str(exc))
            logger.warning(
                f"LHB detail {stock_code}/{trade_date} failed after all retries: {exc}"
            )

        # Periodic checkpoint so long multi-hour runs survive interruption:
        # everything saved so far is skipped by the incremental diff on rerun.
        since_checkpoint += 1
        if since_checkpoint >= _CHECKPOINT_EVERY and new_chunks:
            existing_df = _merge_and_save(existing_df, new_chunks, out_path)
            new_chunks = []
            since_checkpoint = 0

        time.sleep(_INTER_REQUEST_SLEEP)

    if fetch_failed:
        logger.warning(
            f"{fetch_failed} pairs failed — {tracker.summary_line()}\n"
            f"Run 'python main.py --step restore_lhb_detail' to retry."
        )

    result = _merge_and_save(existing_df, new_chunks, out_path)

    if tracker.count_pending() > 0:
        logger.info(
            f"Tip: {tracker.count_pending()} failures still pending."
        )

    return result


def restore_lhb_detail(
    source: str = "akshare",
    save_path: str | None = None,
) -> pd.DataFrame:
    """Re-fetch all pairs in ``failures_lhb_detail.csv`` with status='failed'.

    Successful fetches are merged into the saved parquet and marked 'resolved'.
    """
    out_path = Path(save_path) if save_path else _DEFAULT_SAVE
    tracker = detail_tracker(_load_failures_dir())
    pending = tracker.get_pending()

    if pending.empty:
        logger.info("No pending LHB detail failures to restore.")
        existing = _load_existing(out_path)
        return existing if existing is not None else pd.DataFrame()

    client = _get_client(source)
    existing_df = _load_existing(out_path)
    logger.info(f"Restoring {len(pending)} failed LHB detail pairs (source={source})")

    new_chunks: list[pd.DataFrame] = []
    restored = 0

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="Restoring LHB Detail"):
        trade_date = row["trade_date"]
        stock_code = row["stock_code"]
        key = {"trade_date": trade_date, "stock_code": stock_code}
        try:
            df = with_retry(_fetch_both_sides, client, stock_code, trade_date)
            if df is not None and not df.empty:
                new_chunks.append(df)
            tracker.resolve(key)
            restored += 1
        except Exception as exc:
            tracker.record(key, str(exc))
            logger.warning(f"Restore failed for {stock_code}/{trade_date}: {exc}")

        time.sleep(_INTER_REQUEST_SLEEP)

    logger.info(
        f"Restore complete: {restored}/{len(pending)} pairs recovered. "
        f"{tracker.summary_line()}"
    )
    return _merge_and_save(existing_df, new_chunks, out_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_both_sides(client, stock_code: str, trade_date: str) -> pd.DataFrame:
    """Fetch buy + sell broker sides for one (stock, date) pair."""
    frame = client.fetch_lhb_stock_detail_both(stock_code, trade_date)
    if (trade_date, stock_code) not in _complete_pair_set(frame):
        raise RuntimeError('Empty or incomplete buy/sell detail; source availability must be investigated')
    return frame


def _complete_pair_set(df):
    if df is None or df.empty or not {'trade_date','stock_code','flag','rank','broker_name'}.issubset(df.columns):
        return set()
    good = df.flag.isin(['买入','卖出']) & df.broker_name.notna() & df.broker_name.astype(str).str.strip().ne('') & df['rank'].between(1,5)
    valid = df[good]
    sides = valid.groupby(['trade_date','stock_code']).flag.nunique()
    result = set(sides[sides.eq(2)].index) - _to_pair_set(df[~good])
    if 'report_reason' in valid:
        reports=valid.groupby(['trade_date','stock_code']).agg(reason=('report_reason','first'),side=('flag','first'))
        single=(reports.reason.str.contains('融资买入',na=False)&reports.side.eq('买入') | reports.reason.str.contains('融券卖出',na=False)&reports.side.eq('卖出')) & sides.eq(1)
        result |= set(reports[single].index)
        result -= _to_pair_set(df[~good])
    duplicate = df.duplicated(['trade_date','stock_code','flag','rank'],keep=False)
    result -= _to_pair_set(df[duplicate])
    ranks=valid.groupby(['trade_date','stock_code','flag'])['rank'].agg(['min','max','count'])
    result -= {(code_date[0],code_date[1]) for code_date in ranks[(ranks['min']!=1)|(ranks['max']!=ranks['count'])].index}
    if 'report_reason' in df:
        reasons=df.groupby(['trade_date','stock_code']).report_reason.nunique()
        result -= set(reasons[reasons.gt(1)].index)
    return result


def _resolve_events(
    events_df: pd.DataFrame | None,
    start_date: str | None,
    end_date: str | None,
    reason_keywords: list[str] | None,
) -> pd.DataFrame | None:
    """Return the filtered event DataFrame, loading from summary if needed."""
    if events_df is not None:
        return events_df  # caller already filtered

    if not _DEFAULT_SUMMARY.exists():
        logger.error(
            "No events_df provided and raw LHB summary not found. "
            "Run Step 2 first: python main.py --step fetch"
        )
        return None

    # Load FULL summary history (not just start/end range) so that the
    # first-appearance filter works correctly across batch boundaries.
    full_df = load_parquet(str(_DEFAULT_SUMMARY))

    # Apply date range restriction AFTER the filter so that consecutive-day
    # detection can look back into history before start_date.
    filtered = filter_lhb_events(full_df, reason_keywords)

    if start_date and "trade_date" in filtered.columns:
        filtered = filtered[filtered["trade_date"] >= start_date]
    if end_date and "trade_date" in filtered.columns:
        filtered = filtered[filtered["trade_date"] <= end_date]

    logger.info(
        f"Events for broker detail: {len(full_df)} raw → "
        f"{len(filtered)} after filtering"
        + (f" + date range [{start_date}, {end_date}]" if start_date or end_date else "")
    )
    return filtered


def _to_pair_set(df: pd.DataFrame) -> set[tuple[str, str]]:
    if "trade_date" not in df.columns or "stock_code" not in df.columns:
        return set()
    pairs = df[["trade_date", "stock_code"]].drop_duplicates()
    return set(zip(pairs["trade_date"].astype(str), pairs["stock_code"].astype(str)))


def _load_existing(out_path: Path) -> pd.DataFrame | None:
    if not out_path.exists():
        return None
    try:
        return load_parquet(str(out_path))
    except Exception as exc:
        raise RuntimeError(f"Existing detail parquet cannot be read; refusing overwrite: {out_path}") from exc


def _merge_and_save(
    existing_df: pd.DataFrame | None,
    new_chunks: list[pd.DataFrame],
    out_path: Path,
) -> pd.DataFrame:
    replacement_pairs = set()
    for chunk in new_chunks:
        replacement_pairs.update(_complete_pair_set(chunk))
    if replacement_pairs and existing_df is not None and not existing_df.empty:
        # A refreshed report can legitimately have fewer than five seats.
        # Replace the whole event; rank-level upsert would retain stale tails.
        old_keys = pd.MultiIndex.from_frame(existing_df[['trade_date','stock_code']])
        existing_df = existing_df.loc[~old_keys.isin(replacement_pairs)]
    parts = [df for df in [existing_df] + new_chunks if df is not None and not df.empty]
    if not parts:
        logger.warning("No broker detail data to save")
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    before = len(combined)
    dedup_cols = [c for c in _DEDUP_KEYS if c in combined.columns]
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols, keep="last")

    sort_cols = [c for c in ["trade_date", "stock_code", "flag", "rank"]
                 if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols).reset_index(drop=True)

    if (dropped := before - len(combined)):
        logger.info(f"Dedup: dropped {dropped} duplicate rows")

    save_parquet(combined, str(out_path))
    logger.info(f"Broker detail saved: {out_path} ({len(combined):,} rows)")
    return combined


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
        f"Unknown LHB detail source: '{source}'. "
        "Only 'akshare' is supported (no fallback by design)."
    )
