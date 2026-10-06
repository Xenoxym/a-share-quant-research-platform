"""Statistical computation for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("statistics")


def compute_backtest_statistics(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    round_trip_cost: float = 0.0,
) -> pd.DataFrame:
    """Compute summary statistics for a filtered event dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered event DataFrame with labeled return columns.
    horizons : list[int] or None
        Holding period horizons. Default: [1, 2, 3, 5, 10].
    round_trip_cost : float
        Round-trip transaction cost as a decimal fraction (e.g. 0.002).
        Applied to the tradable ``entry_open_return_Nd`` statistics as
        ``net_entry_return_Nd`` / ``net_entry_win_rate_Nd``.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with all computed statistics.
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10]
    if not np.isfinite(round_trip_cost) or round_trip_cost < 0:
        raise ValueError("round_trip_cost must be finite and non-negative")

    stats: dict[str, float] = {}
    stats["sample_count"] = len(df)

    if len(df) == 0:
        return pd.DataFrame([stats])

    # Limit-up continuation rates
    stats["next_day_limit_up_rate"] = _safe_mean(df, "next_day_limit_up")
    stats["within_2d_limit_up_rate"] = _safe_mean(df, "within_2d_limit_up")
    stats["within_3d_limit_up_rate"] = _safe_mean(df, "within_3d_limit_up")
    stats["within_5d_limit_up_rate"] = _safe_mean(df, "within_5d_limit_up")
    for label in ("next_day_limit_up", "within_2d_limit_up", "within_3d_limit_up",
                  "within_5d_limit_up", "continue_board_2d", "continue_board_3d"):
        stats[f"valid_{label}_count"] = df[label].count() if label in df else 0

    # Per-horizon statistics
    for n in horizons:
        ret_col = f"future_return_{n}d"
        max_ret_col = f"future_max_return_{n}d"
        max_dd_col = f"future_max_drawdown_{n}d"

        if ret_col in df.columns:
            returns = df[ret_col].dropna()
            stats[f"valid_return_count_{n}d"] = len(returns)
            stats[f"avg_return_{n}d"] = returns.mean() if len(returns) > 0 else np.nan
            stats[f"median_return_{n}d"] = returns.median() if len(returns) > 0 else np.nan
            stats[f"std_return_{n}d"] = returns.std() if len(returns) > 0 else np.nan
            stats[f"win_rate_{n}d"] = (
                (returns > 0).sum() / len(returns) if len(returns) > 0 else np.nan
            )
        else:
            stats[f"avg_return_{n}d"] = np.nan
            stats[f"median_return_{n}d"] = np.nan
            stats[f"std_return_{n}d"] = np.nan
            stats[f"win_rate_{n}d"] = np.nan

        stats[f"avg_max_return_{n}d"] = _safe_mean(df, max_ret_col)
        stats[f"avg_max_drawdown_{n}d"] = _safe_mean(df, max_dd_col)

        # Tradable view: T+1 open entry, optionally net of round-trip cost
        entry_col = f"entry_open_return_{n}d"
        if n == 1 and entry_col in df.columns:
            stats["research_intraday_return_1d"] = _safe_mean(df, entry_col)
        if n >= 2 and entry_col in df.columns:
            entry_ret = df[entry_col].dropna()
            stats[f"valid_entry_count_{n}d"] = len(entry_ret)
            if len(entry_ret) > 0:
                stats[f"avg_entry_return_{n}d"] = entry_ret.mean()
                stats[f"entry_win_rate_{n}d"] = (entry_ret > 0).mean()
                net = entry_ret - round_trip_cost
                stats[f"net_entry_return_{n}d"] = net.mean()
                stats[f"net_entry_win_rate_{n}d"] = (net > 0).mean()

        # Benchmark excess (alpha)
        alpha_col = f"alpha_{n}d"
        if alpha_col in df.columns:
            stats[f"valid_close_alpha_count_{n}d"] = df[alpha_col].count()
            stats[f"avg_research_close_alpha_{n}d"] = _safe_mean(df, alpha_col)
        if n >= 2 and f"entry_alpha_{n}d" in df:
            stats[f"valid_entry_alpha_count_{n}d"] = df[f"entry_alpha_{n}d"].count()
            stats[f"avg_entry_alpha_{n}d"] = _safe_mean(df, f"entry_alpha_{n}d")
            stats[f"net_entry_alpha_{n}d"] = stats[f"avg_entry_alpha_{n}d"] - round_trip_cost

    # Shortest LEGAL round trip under the T+1 rule (buy T+1 open → sell T+2
    # open). entry_open_return_1d exits at T+1 close, which is not executable.
    oo_col = "entry_open_exit_open_1d"
    if oo_col in df.columns:
        oo = df[oo_col].dropna()
        stats["valid_oo_count_1d"] = len(oo)
        if len(oo) > 0:
            stats["avg_oo_return_1d"] = oo.mean()
            stats["oo_win_rate_1d"] = (oo > 0).mean()
            net_oo = oo - round_trip_cost
            stats["net_oo_return_1d"] = net_oo.mean()
            stats["net_oo_win_rate_1d"] = (net_oo > 0).mean()
    if "entry_oo_alpha_1d" in df:
        stats["valid_oo_alpha_count_1d"] = df["entry_oo_alpha_1d"].count()
        stats["net_oo_alpha_1d"] = _safe_mean(df, "entry_oo_alpha_1d") - round_trip_cost

    # Entry friction diagnostics
    stats["untradable_rate"] = _safe_mean(df, "next_day_untradable")
    stats["avg_entry_open_gap"] = _safe_mean(df, "entry_open_gap")

    # Continue board statistics
    stats["avg_continue_board_days"] = _safe_mean(df, "max_continue_board_days")
    stats["continue_board_1d_rate"] = _safe_mean(df, "continue_board_1d")
    stats["continue_board_2d_rate"] = _safe_mean(df, "continue_board_2d")
    stats["continue_board_3d_rate"] = _safe_mean(df, "continue_board_3d")

    # Profit-loss ratio
    stats["profit_loss_ratio_1d"] = _compute_profit_loss_ratio(df, "future_return_1d")
    stats["profit_loss_ratio_5d"] = _compute_profit_loss_ratio(df, "future_return_5d")

    return pd.DataFrame([stats])


def compute_grouped_statistics(
    df: pd.DataFrame,
    group_col: str,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Compute statistics grouped by a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        Event DataFrame.
    group_col : str
        Column name to group by (e.g., "lhb_reason", "board_type").
    horizons : list[int] or None
        Holding period horizons. Default: [1, 2, 3, 5, 10].

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per group, indexed by group value.
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10]

    if group_col not in df.columns:
        logger.warning(f"Group column '{group_col}' not found in DataFrame")
        return pd.DataFrame()

    results = []
    for group_val, group_df in df.groupby(group_col, dropna=False):
        row_stats = compute_backtest_statistics(group_df, horizons=horizons)
        row_stats.insert(0, group_col, group_val)
        results.append(row_stats)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def compute_quantile_statistics(
    df: pd.DataFrame,
    factor_col: str,
    n_quantiles: int = 5,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Split events into quantiles based on a factor column and compute stats.

    Parameters
    ----------
    df : pd.DataFrame
        Event DataFrame.
    factor_col : str
        Column to split into quantiles.
    n_quantiles : int
        Number of quantile bins. Default: 5.
    horizons : list[int] or None
        Holding period horizons. Default: [1, 2, 3, 5, 10].

    Returns
    -------
    pd.DataFrame
        DataFrame with n_quantiles rows, each containing statistics
        plus quantile, factor_min, factor_max, factor_mean columns.
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10]

    if factor_col not in df.columns:
        logger.warning(f"Factor column '{factor_col}' not found in DataFrame")
        return pd.DataFrame()

    work_df = df.dropna(subset=[factor_col]).copy()
    if len(work_df) == 0:
        return pd.DataFrame()

    work_df["_quantile"] = pd.qcut(
        work_df[factor_col], q=n_quantiles, labels=False, duplicates="drop"
    )

    results = []
    for q_val, q_df in work_df.groupby("_quantile"):
        row_stats = compute_backtest_statistics(q_df, horizons=horizons)
        row_stats.insert(0, "quantile", int(q_val) + 1)
        row_stats["factor_min"] = q_df[factor_col].min()
        row_stats["factor_max"] = q_df[factor_col].max()
        row_stats["factor_mean"] = q_df[factor_col].mean()
        results.append(row_stats)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def compute_rolling_period_statistics(
    df: pd.DataFrame,
    period_length: str = "3ME",
    metric_col: str = "next_day_limit_up",
    date_col: str = "trade_date",
    horizons: list[int] | None = None,
    top_n: int = 5,
) -> dict[str, pd.DataFrame]:
    """Find the best and worst time periods based on a target metric.

    Splits events into rolling calendar periods and ranks them.

    Parameters
    ----------
    df : pd.DataFrame
        Labeled event DataFrame.
    period_length : str
        Pandas offset alias for period grouping.
        Common values: ``"1ME"`` (monthly), ``"2ME"`` (bi-monthly), ``"3ME"`` (quarterly).
    metric_col : str
        Column to evaluate. For rate metrics use boolean label columns
        (``"next_day_limit_up"``, ``"continue_board_2d"``); for return metrics
        use ``"future_return_1d"`` etc.
    date_col : str
        Date column name.
    horizons : list[int] or None
        Horizons for full statistics per period.
    top_n : int
        Number of best/worst periods to return.

    Returns
    -------
    dict with keys:
        ``"all_periods"`` — DataFrame of all periods with stats, sorted by date.
        ``"best_periods"`` — Top N periods ranked by the target metric.
        ``"worst_periods"`` — Bottom N periods ranked by the target metric.
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10]

    if df.empty or date_col not in df.columns:
        empty = pd.DataFrame()
        return {"all_periods": empty, "best_periods": empty, "worst_periods": empty}

    work = df.copy()
    work["_dt"] = pd.to_datetime(work[date_col])
    # Datetime offsets use ME, Period uses M. Multi-month Period conversion
    # alone creates overlapping windows starting each month; bucket first.
    import re
    month_span = re.fullmatch(r"(\d*)M(?:E)?", period_length)
    if month_span:
        span = int(month_span.group(1) or 1)
        if span < 1:
            raise ValueError("period_length must be positive")
        months = work["_dt"].dt.to_period("M")
        mapping = {p: pd.Period(ordinal=(p.ordinal // span) * span, freq=f"{span}M")
                   for p in months.dropna().unique()}
        work["_period"] = months.map(mapping)
    else:
        work["_period"] = work["_dt"].dt.to_period(period_length)

    period_results = []
    for period_val, group in work.groupby("_period"):
        row_stats = compute_backtest_statistics(group, horizons=horizons)
        row_stats.insert(0, "period", str(period_val))
        row_stats["period_start"] = period_val.start_time.strftime("%Y-%m-%d")
        row_stats["period_end"] = period_val.end_time.strftime("%Y-%m-%d")

        if metric_col in group.columns:
            metric_values = group[metric_col].dropna()
            if len(metric_values) > 0:
                if metric_values.dtype == bool or set(metric_values.unique()).issubset({0, 1, True, False}):
                    row_stats["target_metric"] = metric_values.mean()
                else:
                    row_stats["target_metric"] = metric_values.mean()
            else:
                row_stats["target_metric"] = np.nan
        else:
            row_stats["target_metric"] = np.nan

        period_results.append(row_stats)

    if not period_results:
        empty = pd.DataFrame()
        return {"all_periods": empty, "best_periods": empty, "worst_periods": empty}

    all_periods = pd.concat(period_results, ignore_index=True)

    valid = all_periods[all_periods["sample_count"] >= 5].copy()

    if valid.empty:
        return {"all_periods": all_periods, "best_periods": pd.DataFrame(), "worst_periods": pd.DataFrame()}

    best = valid.nlargest(top_n, "target_metric")
    worst = valid.nsmallest(top_n, "target_metric")

    return {"all_periods": all_periods, "best_periods": best, "worst_periods": worst}


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    """Compute mean of a column, returning NaN if column doesn't exist."""
    if col not in df.columns:
        return np.nan
    values = df[col].dropna()
    return values.mean() if len(values) > 0 else np.nan


def _compute_profit_loss_ratio(df: pd.DataFrame, return_col: str) -> float:
    """Compute profit-loss ratio: mean(positive returns) / abs(mean(negative returns))."""
    if return_col not in df.columns:
        return np.nan

    returns = df[return_col].dropna()
    if len(returns) == 0:
        return np.nan

    positive = returns[returns > 0]
    negative = returns[returns < 0]

    if len(positive) == 0 or len(negative) == 0:
        return np.nan

    avg_profit = positive.mean()
    avg_loss = abs(negative.mean())

    if avg_loss == 0:
        return np.nan

    return avg_profit / avg_loss
