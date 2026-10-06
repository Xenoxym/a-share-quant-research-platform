"""Load daily K-line from SimTradeData exported Parquet files.

Replaces the per-stock AKShare HTTP loop when ``--source simtradedata`` is used.
Reads directly from the SimTradeData export/cn/stocks/ directory (zero network calls).

Incremental logic detects missing/partial dates, including historical holes.
refresh_existing also re-reads the requested range to repair auxiliary fields.
Known old cells and keys are preserved when new source values are absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.utils.io import load_parquet, save_parquet
from src.utils.logging import get_logger

logger = get_logger("load_kline_simtradedata")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SAVE = _PROJECT_ROOT / "data" / "raw" / "daily_kline.parquet"


def load_kline_simtradedata(
    start_date: str,
    end_date: str,
    export_dir: str | Path | None = None,
    include_valuation: bool | None = None,
    save_path: str | None = None,
    incremental: bool = True,
    refresh_existing: bool = False,
) -> pd.DataFrame:
    """Load daily K-line from SimTradeData Parquet export into raw data dir.

    Parameters
    ----------
    start_date : str
        Requested start date ``"YYYY-MM-DD"``, including historical backfills.
    end_date : str
        End date ``"YYYY-MM-DD"`` (inclusive).
    export_dir : str, Path, or None
        Path to SimTradeData export/cn/ directory.  Falls back to config.yaml
        ``simtradedata.export_dir``, then to ``../../SimTradeData/data/export/cn``
        relative to the project root.
    include_valuation : bool or None
        Join ``turnover_rate`` from valuation/ parquet.  Falls back to
        ``config.yaml simtradedata.include_valuation`` (default True).
    save_path : str or None
        Destination parquet path.  Default: ``data/raw/daily_kline.parquet``.
    incremental : bool
        If True, merge the existing destination with missing/partial source dates.
    refresh_existing : bool
        Also reload already-present dates to refresh repaired auxiliary fields.

    Returns
    -------
    pd.DataFrame
        Combined (existing + new) K-line data saved to ``save_path``.
    """
    from src.data_sources.simtradedata_loader import load_stocks_parquet

    cfg = _load_config()
    resolved_dir = _resolve_export_dir(export_dir, cfg)
    if include_valuation is None:
        include_valuation = cfg.get("simtradedata", {}).get("include_valuation", True)

    out_path = Path(save_path) if save_path else _DEFAULT_SAVE

    # Incremental: compute missing trading dates via set-difference.
    # This handles both forward append (new days) and backfill (earlier range added).
    existing_present = incremental and out_path.exists()
    per_day = None
    old_rows = 0
    if existing_present:
        try:
            import duckdb
            with duckdb.connect() as connection:
                counts = connection.execute(
                    "SELECT CAST(CAST(trade_date AS DATE) AS VARCHAR) AS day, count(*) AS n "
                    "FROM read_parquet(?) GROUP BY day", [str(out_path)]
                ).fetchdf()
            per_day = counts.set_index('day')['n']
            old_rows = int(per_day.sum())
        except Exception as exc:
            raise RuntimeError('Existing K-line file is unreadable; refusing to overwrite it during incremental update') from exc

    missing_dates = _missing_dates_from_counts(start_date, end_date, per_day)
    requested_missing = list(missing_dates)

    if not missing_dates and not refresh_existing:
        logger.info(
            f"Incremental: all trading dates in [{start_date}, {end_date}] already present. "
            "Nothing to fetch."
        )
        result = load_parquet(str(out_path)) if existing_present else pd.DataFrame()
        result.attrs["update_report"] = {
            "status": "pass",
            "requested_missing_dates": 0,
            "rows_added": 0,
            "unresolved_dates": [],
        }
        return result

    # Load the contiguous range that covers all missing dates (fast PyArrow filter).
    # Dates already present in that range are removed by the dedup step below.
    load_start = start_date if refresh_existing else missing_dates[0]
    load_end = end_date if refresh_existing else missing_dates[-1]
    logger.info(
        f"Incremental: {len(missing_dates)} trading dates missing. "
        f"Loading SimTradeData [{load_start} ~ {load_end}] from {resolved_dir}"
    )

    new_df = load_stocks_parquet(
        export_dir=resolved_dir,
        start_date=load_start,
        end_date=load_end,
        include_valuation=include_valuation,
    )

    if new_df.empty:
        from src.utils.calendar import calendar_max_date
        snapshot_end = calendar_max_date()
        logger.warning(
            f"SimTradeData has no rows for [{load_start} ~ {load_end}]. "
            f"The local snapshot ends at {snapshot_end} — dates beyond that "
            "require re-exporting/updating the SimTradeData repository first. "
            "Existing kline data is unchanged."
        )
        result = load_parquet(str(out_path)) if existing_present else pd.DataFrame()
        result.attrs["update_report"] = {
            "status": "fail" if requested_missing else "pass",
            "requested_missing_dates": len(requested_missing),
            "rows_added": 0,
            "unresolved_dates": requested_missing,
        }
        return result

    # Merge on disk so a full auxiliary-field refresh does not hold old/new
    # pandas copies and multi-index intermediates simultaneously in memory.
    if existing_present:
        import gc
        from src.data_sources.parquet_merge import merge_kline_file
        source_rows = len(new_df)
        merge_kline_file(out_path, new_df)
        del new_df
        gc.collect()
        result = load_parquet(str(out_path))
        logger.info("Merged existing ({:,}) + source ({:,}) with bounded-memory file merge",
                    old_rows, source_rows)
    else:
        result = new_df
        save_parquet(result, str(out_path))

    logger.info(f"SimTradeData kline saved: {out_path} ({len(result):,} rows)")

    unresolved = _get_missing_trading_dates(start_date, end_date, result)
    rows_added = len(result) - old_rows
    result.attrs["update_report"] = {
        "status": "fail" if unresolved else "pass",
        "requested_missing_dates": len(requested_missing),
        "rows_added": rows_added,
        "unresolved_dates": unresolved,
    }
    if unresolved:
        logger.error(
            "K-line update remains incomplete: added {:,} net rows; {}/{} "
            "requested dates unresolved ({} ~ {}).",
            rows_added,
            len(unresolved),
            len(requested_missing),
            unresolved[0],
            unresolved[-1],
        )
    else:
        logger.info(
            "K-line update verified: added {:,} net rows; all {} requested "
            "missing/partial dates repaired.",
            rows_added,
            len(requested_missing),
        )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    config_path = _PROJECT_ROOT / "config" / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _resolve_export_dir(export_dir: str | Path | None, cfg: dict) -> Path:
    """Resolve the SimTradeData export/cn/ path with fallback chain.

    Priority:
    1. Explicit ``export_dir`` argument
    2. ``config.yaml`` → ``simtradedata.export_dir``
    3. ``../../SimTradeData/data/export/cn`` relative to project root
    """
    if export_dir:
        return Path(export_dir)

    configured = cfg.get("simtradedata", {}).get("export_dir", "")
    if configured:
        p = Path(configured)
        if not p.is_absolute():
            p = (_PROJECT_ROOT / p).resolve()
        return p

    # Default: D:\Projects\SimTradeData\data\export\cn  (parallel repo layout)
    default = (_PROJECT_ROOT / "../../SimTradeData/data/export/cn").resolve()
    logger.info(f"Using default SimTradeData export dir: {default}")
    return default


# A date is considered COMPLETE only when its row count reaches this fraction
# of the typical (median) per-day stock count.  Guards against partial loads
# (e.g. an interrupted SimTradeData export) being treated as "already present"
# forever.
_COMPLETE_DAY_FRACTION = 0.5


def _get_missing_trading_dates(
    start_date: str,
    end_date: str,
    existing_df: pd.DataFrame | None,
) -> list[str]:
    """Return sorted trading dates in [start_date, end_date] that are absent
    OR incomplete in existing_df.

    A date counts as *incomplete* when its stock count is below
    ``_COMPLETE_DAY_FRACTION`` × the median daily stock count — such dates are
    re-fetched so a partially-exported source heals automatically once the
    SimTradeData snapshot is updated.
    """
    per_day = None
    if existing_df is not None and not existing_df.empty and 'trade_date' in existing_df:
        per_day = pd.to_datetime(existing_df['trade_date']).dt.strftime('%Y-%m-%d').value_counts()
    return _missing_dates_from_counts(start_date, end_date, per_day)


def _missing_dates_from_counts(start_date, end_date, per_day):
    from src.utils.calendar import get_trading_dates
    all_trading = set(get_trading_dates(start_date, end_date))
    if not all_trading:
        return []
    if per_day is None or per_day.empty:
        return sorted(all_trading)
    complete_threshold = per_day.median() * _COMPLETE_DAY_FRACTION
    complete_dates = set(per_day[per_day >= complete_threshold].index)
    partial = (set(per_day.index) - complete_dates) & all_trading
    if partial:
        logger.warning("{} trading dates are PARTIAL (<{:,.0f} stocks) and will be re-fetched",
                       len(partial), complete_threshold)
    return sorted(all_trading - complete_dates)
