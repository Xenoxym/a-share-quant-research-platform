"""Trading calendar utilities for the Chinese A-share market.

Source priority (highest to lowest):
1. SimTradeData export/cn/metadata/trade_days.parquet  — accurate, offline, 1990-present
2. Local cache  data/raw/trading_dates.parquet          — written on first AKShare fetch
3. AKShare (Sina)                                        — requires network
4. Fallback: weekday filter + hardcoded holidays         — approximate
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import yaml

from src.utils.logging import get_logger

logger = get_logger("calendar")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_PATH = _PROJECT_ROOT / "data" / "raw" / "trading_dates.parquet"

_trading_dates_cache: Optional[list[str]] = None

_KNOWN_HOLIDAYS_2024_2026 = [
    # 2024
    "2024-01-01",
    "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12",
    "2024-02-13", "2024-02-14", "2024-02-15", "2024-02-16", "2024-02-17",
    "2024-04-04", "2024-04-05", "2024-04-06",
    "2024-05-01", "2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05",
    "2024-06-08", "2024-06-09", "2024-06-10",
    "2024-09-15", "2024-09-16", "2024-09-17",
    "2024-10-01", "2024-10-02", "2024-10-03", "2024-10-04", "2024-10-05",
    "2024-10-06", "2024-10-07",
    # 2025
    "2025-01-01",
    "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31",
    "2025-02-01", "2025-02-02", "2025-02-03", "2025-02-04",
    "2025-04-04", "2025-04-05", "2025-04-06",
    "2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05",
    "2025-05-31", "2025-06-01", "2025-06-02",
    "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-04", "2025-10-05",
    "2025-10-06", "2025-10-07",
    # 2026
    "2026-01-01", "2026-01-02",
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",
    "2026-02-21", "2026-02-22", "2026-02-23",
    "2026-04-05", "2026-04-06",
    "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
    "2026-06-19",
    "2026-09-25",
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05",
    "2026-10-06", "2026-10-07",
]


def _resolve_simtradedata_export_dir() -> Path | None:
    """Read simtradedata.export_dir from config.yaml and resolve to absolute path."""
    import os
    if os.environ.get('LHB_SIMTRADEDATA_EXPORT_DIR'):
        return Path(os.environ['LHB_SIMTRADEDATA_EXPORT_DIR'])
    config_path = _PROJECT_ROOT / "config" / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        rel = cfg.get("simtradedata", {}).get("export_dir", "")
        if not rel:
            return None
        p = Path(rel)
        if not p.is_absolute():
            p = (_PROJECT_ROOT / p).resolve()
        return p
    except Exception:
        return None


def _load_from_simtradedata() -> list[str] | None:
    """Load trading dates from SimTradeData metadata/trade_days.parquet.

    Returns sorted list of 'YYYY-MM-DD' strings, or None if file not available.
    """
    export_dir = _resolve_simtradedata_export_dir()
    if export_dir is None:
        return None

    trade_days_path = export_dir / "metadata" / "trade_days.parquet"
    if not trade_days_path.exists():
        return None

    try:
        df = pd.read_parquet(str(trade_days_path))
        # Column is named 'date' in SimTradeData
        col = "date" if "date" in df.columns else df.columns[0]
        dates = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .pipe(lambda s: s[s.str.match(r"^\d{4}-\d{2}-\d{2}$")])
            .sort_values()
            .tolist()
        )
        logger.debug(f"Loaded {len(dates)} trading dates from SimTradeData ({trade_days_path})")
        return dates
    except Exception as exc:
        logger.warning(f"Failed to read SimTradeData trade_days.parquet: {exc}")
        return None


def _fetch_trading_dates_akshare() -> list[str]:
    """Fetch historical trading dates from akshare (Sina source)."""
    import akshare as ak

    df = ak.tool_trade_date_hist_sina()
    dates = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d").tolist()
    dates.sort()
    return dates


def _generate_fallback_dates(start: str = "2000-01-01", end: str = "2027-12-31") -> list[str]:
    """Generate approximate trading dates by filtering weekends and known holidays."""
    all_days = pd.date_range(start=start, end=end, freq="B")
    holiday_set = set(_KNOWN_HOLIDAYS_2024_2026)
    return [d.strftime("%Y-%m-%d") for d in all_days if d.strftime("%Y-%m-%d") not in holiday_set]


def _load_trading_dates() -> list[str]:
    """Load trading dates.

    Priority:
    1. In-memory cache (fastest)
    2. SimTradeData trade_days.parquet (accurate, offline)
    3. Local parquet cache (data/raw/trading_dates.parquet)
    4. AKShare network fetch (cached to #3 on success)
    5. Fallback weekday filter
    """
    global _trading_dates_cache

    if _trading_dates_cache is not None:
        return _trading_dates_cache

    # 1. SimTradeData (preferred — accurate and offline)
    dates = _load_from_simtradedata()
    if dates:
        _trading_dates_cache = dates
        return _trading_dates_cache

    # 2. Local parquet cache
    if _CACHE_PATH.exists():
        try:
            df = pd.read_parquet(str(_CACHE_PATH))
            _trading_dates_cache = sorted(df["trade_date"].astype(str).tolist())
            logger.debug(f"Loaded {len(_trading_dates_cache)} trading dates from local cache")
            return _trading_dates_cache
        except Exception as exc:
            logger.warning(f"Failed to read cached trading dates: {exc}")

    # 3. AKShare (network)
    try:
        dates = _fetch_trading_dates_akshare()
        _trading_dates_cache = dates
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trade_date": dates}).to_parquet(str(_CACHE_PATH), index=False)
        logger.info(f"Fetched and cached {len(dates)} trading dates from akshare")
        return _trading_dates_cache
    except Exception as exc:
        logger.warning(f"AKShare calendar fetch failed, using fallback: {exc}")

    # 4. Fallback
    _trading_dates_cache = _generate_fallback_dates()
    logger.info(f"Using fallback calendar ({len(_trading_dates_cache)} dates, approximate)")
    return _trading_dates_cache


def invalidate_cache() -> None:
    """Force reload of trading dates on next call (useful after SimTradeData update)."""
    global _trading_dates_cache
    _trading_dates_cache = None


def _normalize_date(date: Union[str, datetime]) -> str:
    """Convert a date input to YYYY-MM-DD string."""
    if isinstance(date, datetime):
        return date.strftime("%Y-%m-%d")
    return str(date)[:10]


def calendar_max_date() -> str:
    """Last date covered by the loaded trading calendar.

    Dates after this are *unknown* to the calendar source (e.g. a SimTradeData
    snapshot ends at its export date) — not necessarily non-trading days.
    """
    dates = _load_trading_dates()
    return dates[-1] if dates else ""


def get_trading_dates(start_date: str, end_date: str) -> list[str]:
    """Get all trading dates within a date range (inclusive).

    If ``end_date`` lies beyond the calendar source coverage (e.g. the
    SimTradeData snapshot ends before it), the tail is extended with an
    approximate weekday+holiday calendar so that callers correctly see those
    dates as *missing* rather than *non-existent*.

    Parameters
    ----------
    start_date : str
        Start date in "YYYY-MM-DD" format.
    end_date : str
        End date in "YYYY-MM-DD" format.

    Returns
    -------
    list[str]
        Sorted list of trading dates as "YYYY-MM-DD" strings.
    """
    dates = _load_trading_dates()
    result = [d for d in dates if start_date <= d <= end_date]

    known_max = dates[-1] if dates else ""
    if end_date > known_max:
        approx_start = max(start_date, known_max)
        approx = [
            d for d in _generate_fallback_dates(approx_start, end_date)
            if d > known_max
        ]
        if approx:
            logger.warning(
                f"Trading calendar source only covers up to {known_max}; "
                f"extended [{approx[0]}, {approx[-1]}] with approximate "
                f"weekday+holiday dates ({len(approx)} days)"
            )
            result.extend(approx)
    return result


def get_next_n_trading_dates(date: str, n: int) -> list[str]:
    """Get the next n trading dates after the given date.

    Parameters
    ----------
    date : str
        Reference date in "YYYY-MM-DD" format.
    n : int
        Number of trading dates to return.

    Returns
    -------
    list[str]
        List of the next n trading dates (not including the reference date).
    """
    dates = _load_trading_dates()
    ref = _normalize_date(date)
    future = [d for d in dates if d > ref]
    return future[:n]


def get_prev_n_trading_dates(date: str, n: int) -> list[str]:
    """Get the previous n trading dates before the given date.

    Parameters
    ----------
    date : str
        Reference date in "YYYY-MM-DD" format.
    n : int
        Number of trading dates to return.

    Returns
    -------
    list[str]
        List of the previous n trading dates (not including the reference date),
        sorted from earliest to latest.
    """
    dates = _load_trading_dates()
    ref = _normalize_date(date)
    past = [d for d in dates if d < ref]
    return past[-n:]


def is_trading_date(date: str) -> bool:
    """Check whether a given date is a trading date.

    Parameters
    ----------
    date : str
        Date in "YYYY-MM-DD" format.

    Returns
    -------
    bool
        True if the date is a trading date.
    """
    dates = _load_trading_dates()
    return _normalize_date(date) in set(dates)


def shift_trading_date(date: str, n: int) -> str:
    """Shift a date by n trading days.

    Parameters
    ----------
    date : str
        Reference date in "YYYY-MM-DD" format.
    n : int
        Number of trading days to shift. Positive = forward, negative = backward.

    Returns
    -------
    str
        The resulting trading date in "YYYY-MM-DD" format.

    Raises
    ------
    IndexError
        If shifting beyond available calendar range.
    """
    if n == 0:
        ref = _normalize_date(date)
        if is_trading_date(ref):
            return ref
        return get_next_n_trading_dates(ref, 1)[0]

    if n > 0:
        result = get_next_n_trading_dates(date, n)
        return result[-1]
    else:
        result = get_prev_n_trading_dates(date, abs(n))
        return result[0]
