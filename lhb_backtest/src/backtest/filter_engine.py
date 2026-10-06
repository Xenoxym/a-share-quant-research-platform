"""Main backtest API for filtering and analyzing LHB events."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.statistics import compute_backtest_statistics
from src.utils.io import load_parquet, save_csv
from src.utils.logging import get_logger

logger = get_logger("filter_engine")

_DEFAULT_EVENT_PATH = "data/factor/lhb_event_labeled.parquet"
_DEFAULT_HORIZONS = [1, 2, 3, 5, 10]

_BOARD_TYPE_MAP = {
    "main_sh": "60",
    "main_sz": "00",
    "chinext": "30",
    "star": "68",
    "bse": "8",
}


@dataclass
class BacktestResult:
    """Container for backtest results."""

    filters: dict
    horizons: list[int]
    summary: pd.DataFrame
    detail: pd.DataFrame
    sample_count: int

    def to_csv(self, path: str) -> None:
        """Save summary and detail to CSV files.

        Parameters
        ----------
        path : str
            Base path for output. Creates {path}_summary.csv and {path}_detail.csv.
        """
        base = Path(path).with_suffix("")
        summary_path = f"{base}_summary.csv"
        detail_path = f"{base}_detail.csv"

        save_csv(self.summary, summary_path)
        save_csv(self.detail, detail_path)
        logger.info(f"Results saved to {summary_path} and {detail_path}")

    def __repr__(self) -> str:
        """Pretty print summary statistics."""
        lines = [
            f"BacktestResult(sample_count={self.sample_count})",
            f"  Filters: {self.filters}",
            f"  Horizons: {self.horizons}",
            "  Summary:",
        ]
        if not self.summary.empty:
            for col in self.summary.columns:
                val = self.summary[col].iloc[0]
                if isinstance(val, float):
                    if "rate" in col or "ratio" in col.lower():
                        lines.append(f"    {col}: {val:.4f} ({val*100:.2f}%)")
                    else:
                        lines.append(f"    {col}: {val:.4f}")
                else:
                    lines.append(f"    {col}: {val}")
        return "\n".join(lines)


def run_lhb_backtest(
    start_date: str | None = None,
    end_date: str | None = None,
    filters: dict | None = None,
    horizons: list[int] | None = None,
    event_path: str | None = None,
    event_df: pd.DataFrame | None = None,
    round_trip_cost: float = 0.0,
) -> BacktestResult:
    """Main backtest entry point.

    Parameters
    ----------
    start_date : str or None
        Start date filter (inclusive), format "YYYY-MM-DD" or "YYYYMMDD".
    end_date : str or None
        End date filter (inclusive), format "YYYY-MM-DD" or "YYYYMMDD".
    filters : dict or None
        Dictionary of filter conditions to apply. See apply_filters() for supported keys.
    horizons : list[int] or None
        Holding period horizons for return statistics. Default: [1, 2, 3, 5, 10].
    event_path : str or None
        Path to labeled event parquet file. Default: data/factor/lhb_event_labeled.parquet.
    event_df : pd.DataFrame or None
        Pre-loaded event DataFrame. If provided, event_path is ignored.
    round_trip_cost : float
        Total round-trip transaction cost as a decimal fraction
        (e.g. 0.002 = 0.2%: commission + stamp duty + slippage).
        Subtracted from every return-based statistic (net-of-cost view).

    Returns
    -------
    BacktestResult
        Container with summary statistics, detail data, and metadata.
    """
    if horizons is None:
        horizons = _DEFAULT_HORIZONS
    if filters is None:
        filters = {}

    if event_df is not None:
        df = event_df.copy()
        logger.info(f"Using provided event DataFrame with {len(df)} rows")
    else:
        path = event_path or _DEFAULT_EVENT_PATH
        logger.info(f"Loading event data from {path}")
        df = load_parquet(path)
        if "label_schema_version" not in df or not df.label_schema_version.eq(2).all():
            raise ValueError("Legacy labels detected: run clean, features, labels before backtesting")

    logger.info(f"Total events loaded: {len(df)}")

    # Identify consecutive appearances before cropping the requested period:
    # an event on its first day may follow an appearance just outside it.
    df = apply_filters(df, filters)
    logger.info(f"After applying filters: {len(df)} events")

    # Apply date range filter
    if start_date or end_date:
        df = _apply_date_filter(df, start_date, end_date)
        logger.info(f"After date filter: {len(df)} events")

    # Compute statistics
    summary = compute_backtest_statistics(
        df, horizons=horizons, round_trip_cost=round_trip_cost,
    )

    return BacktestResult(
        filters=filters,
        horizons=horizons,
        summary=summary,
        detail=df,
        sample_count=len(df),
    )


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply a dictionary of filters to the event DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Event DataFrame.
    filters : dict
        Filter conditions. Supported keys:
        - min_buy_sell_ratio: float
        - min_buy1_concentration: float
        - min_lhb_turnover_ratio: float
        - min_net_buy_ratio: float
        - min_net_buy_float_mcap_ratio: float
        - min_pct_chg: float
        - max_turnover_rate: float
        - min_daily_amount: float
        - include_only_limit_up_event: bool
        - include_only_first_board: bool  (event day is 首板: limit-up & prev day not)
        - exclude_consecutive_lhb_day: bool  (drop 2nd+ consecutive LHB appearances)
        - exclude_ST: bool  (uses historical is_st column when present,
          falls back to current stock_name matching)
        - exclude_untradable_next_day: bool  (drop events where T+1 entry was
          impossible: opened at limit-up, suspended, or no T+1 kline)
        - only_single_day_board: bool  (drop 三日榜 / mixed-window events)
        - board_type: str or list[str]
        - lhb_reason: str or list[str]  (keep only reasons containing any keyword)
        - exclude_lhb_reason: str or list[str]  (drop reasons containing any
          keyword, e.g. "无价格涨跌幅限制" for new listings without limits)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if not filters:
        return df
    filters = {key: value for key, value in filters.items() if value is not None}

    required = {
        "min_buy_sell_ratio": "buy_sell_ratio", "min_buy1_concentration": "buy1_concentration",
        "min_lhb_turnover_ratio": "lhb_turnover_ratio", "min_net_buy_ratio": "net_buy_ratio",
        "min_net_buy_float_mcap_ratio": "net_buy_float_mcap_ratio", "min_pct_chg": "pct_chg",
        "min_daily_amount": "amount", "max_turnover_rate": "turnover_rate",
        "include_only_first_board": "is_first_limit_up_board",
        "exclude_untradable_next_day": "next_day_untradable", "only_single_day_board": "lhb_window_days",
        "lhb_reason": "lhb_reason", "exclude_lhb_reason": "lhb_reason",
    }
    supported = set(required) | {"exclude_ST", "include_only_limit_up_event",
                                "exclude_consecutive_lhb_day", "board_type", "only_a_shares"}
    unknown = set(filters) - supported
    if unknown:
        raise ValueError(f"Unknown filter(s): {sorted(unknown)}")
    for key, col in required.items():
        if key in filters and filters[key] is not False and filters[key] is not None and col not in df:
            raise ValueError(f"Filter {key} requires column {col}; rebuild features/labels")

    mask = pd.Series(True, index=df.index)
    if filters.get("only_a_shares"):
        mask &= df.stock_code.str.match(r"^(?:60\d{4}\.SH|688\d{3}\.SH|00\d{4}\.SZ|30\d{4}\.SZ|(?:43|83|87|92)\d{4}\.BJ)$")

    # Minimum threshold filters
    _min_filters = {
        "min_buy_sell_ratio": "buy_sell_ratio",
        "min_buy1_concentration": "buy1_concentration",
        "min_lhb_turnover_ratio": "lhb_turnover_ratio",
        "min_net_buy_ratio": "net_buy_ratio",
        "min_net_buy_float_mcap_ratio": "net_buy_float_mcap_ratio",
        "min_pct_chg": "pct_chg",
        "min_daily_amount": "amount",
    }

    for filter_key, col_name in _min_filters.items():
        if filter_key in filters and col_name in df.columns:
            mask &= df[col_name] >= filters[filter_key]

    # Maximum threshold filters
    if "max_turnover_rate" in filters and "turnover_rate" in df.columns:
        mask &= df["turnover_rate"] <= filters["max_turnover_rate"]

    # Limit-up event filter — prefer the exact-price label from Step 6,
    # fall back to pct_chg threshold approximation
    if filters.get("include_only_limit_up_event"):
        if "event_day_limit_up" in df.columns:
            mask &= df["event_day_limit_up"] == 1.0
        else:
            raise ValueError("Limit-up filtering requires price-based event_day_limit_up labels")

    # First limit-up board only (首板): event day limit-up, previous trading day not
    if filters.get("include_only_first_board"):
        if "is_first_limit_up_board" in df.columns:
            mask &= df["is_first_limit_up_board"].eq(1)
        else:
            logger.warning(
                "include_only_first_board requested but column 'is_first_limit_up_board' "
                "not found — re-run Step 6 (labels) to generate it"
            )

    # Exclude consecutive LHB appearances (keep first day of each LHB streak per stock)
    if filters.get("exclude_consecutive_lhb_day"):
        from src.cleaning.clean_lhb import _filter_first_appearance
        allowed_idx = set(_filter_first_appearance(df).index)
        mask &= df.index.isin(allowed_idx)

    # Exclude ST stocks — prefer historical is_st (from stock_status metadata),
    # fall back to name matching (current name only, may misclassify past ST)
    if filters.get("exclude_ST"):
        if "is_st" in df.columns:
            mask &= df["is_st"].eq(0)
        else:
            raise ValueError("exclude_ST requires historical is_st; rebuild features")

    # Exclude events that could not actually be entered at T+1
    if filters.get("exclude_untradable_next_day") and "next_day_untradable" in df.columns:
        mask &= df["next_day_untradable"].eq(0)

    # Keep only single-day boards (drop 三日榜 / mixed-window events)
    if filters.get("only_single_day_board") and "lhb_window_days" in df.columns:
        mask &= df["lhb_window_days"] == 1

    # Board type filter
    if "board_type" in filters:
        mask &= _apply_board_type_filter(df, filters["board_type"])

    # LHB reason filter (substring match)
    if "lhb_reason" in filters and "lhb_reason" in df.columns:
        reasons = filters["lhb_reason"]
        if isinstance(reasons, str):
            reasons = [reasons]
        reason_mask = pd.Series(False, index=df.index)
        for reason in reasons:
            reason_mask |= df["lhb_reason"].str.contains(reason, case=False, na=False, regex=False)
        mask &= reason_mask

    # LHB reason exclusion (substring match) — e.g. drop "无价格涨跌幅限制"
    # events (new listings / first-day-unlimited stocks distort returns)
    if "exclude_lhb_reason" in filters and "lhb_reason" in df.columns:
        excludes = filters["exclude_lhb_reason"]
        if isinstance(excludes, str):
            excludes = [excludes]
        for reason in excludes:
            mask &= ~df["lhb_reason"].str.contains(reason, case=False, na=False, regex=False)

    return df[mask].copy()


def _apply_date_filter(
    df: pd.DataFrame, start_date: str | None, end_date: str | None
) -> pd.DataFrame:
    """Filter DataFrame by date range."""
    date_col = None
    for candidate in ["trade_date", "date", "dt"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        raise ValueError("Date filtering requires a date column")
    if start_date and end_date and pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError("start_date must not exceed end_date")

    dates = pd.to_datetime(df[date_col])

    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= dates >= pd.to_datetime(start_date)
    if end_date:
        mask &= dates <= pd.to_datetime(end_date)

    return df[mask].copy()


def _apply_board_type_filter(df: pd.DataFrame, board_types: str | list[str]) -> pd.Series:
    """Create mask for board type filter based on stock code prefix."""
    if isinstance(board_types, str):
        board_types = [board_types]
    unknown = set(board_types) - set(_BOARD_TYPE_MAP)
    if unknown:
        raise ValueError(f"Unknown board type(s): {sorted(unknown)}")

    # Try using board_type column directly if it exists
    if "board_type" in df.columns:
        return df["board_type"].isin(board_types)

    # Fall back to inferring from stock code
    code_col = None
    for candidate in ["stock_code", "ts_code", "code"]:
        if candidate in df.columns:
            code_col = candidate
            break

    if code_col is None:
        raise ValueError("Board filtering requires stock_code or board_type")

    codes = df[code_col].astype(str)
    patterns = {"main_sh": r"^60\d{4}\.SH$", "main_sz": r"^00\d{4}\.SZ$",
                "chinext": r"^30\d{4}\.SZ$", "star": r"^688\d{3}\.SH$",
                "bse": r"^(?:43|83|87|92)\d{4}\.BJ$"}
    mask = pd.Series(False, index=df.index)
    for board in board_types:
        mask |= codes.str.match(patterns[board])

    return mask
