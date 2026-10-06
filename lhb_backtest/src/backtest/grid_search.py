"""Grid search over filter parameter combinations.

Supports in-sample / out-of-sample (IS/OOS) evaluation: when
``oos_split_date`` is set, every metric is computed twice — on events before
the split date (in-sample, plain column names) and on events on/after it
(out-of-sample, ``oos_`` prefix).  Rank on training metrics only. Held-out metrics are diagnostics, never
parameter-selection objectives. Training labels crossing the split are purged.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.backtest.filter_engine import apply_filters
from src.backtest.statistics import compute_backtest_statistics
from src.utils.io import load_parquet, save_csv
from src.utils.logging import get_logger

logger = get_logger("grid_search")

_DEFAULT_EVENT_PATH = "data/factor/lhb_event_labeled.parquet"

_DISPLAY_METRICS = [
    "sample_count",
    "next_day_limit_up_rate",
    "avg_oo_return_1d",
    "net_oo_return_1d",
    "avg_return_1d",
    "win_rate_1d",
    "avg_return_5d",
    "oos_sample_count",
    "oos_next_day_limit_up_rate",
    "oos_net_oo_return_1d",
]


def format_grid_search_summary(
    result_df: pd.DataFrame,
    param_names: list[str],
    sort_by: str,
    top_n: int = 10,
) -> str:
    """Format a readable text summary of grid-search results."""
    if result_df.empty:
        return "Grid search returned no results."

    display_cols = [c for c in param_names + _DISPLAY_METRICS if c in result_df.columns]
    top = result_df.head(top_n)[display_cols].copy()

    for col in top.columns:
        if col in param_names or col == "sample_count":
            continue
        if "rate" in col or "ratio" in col.lower():
            top[col] = top[col].map(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "—")
        elif col.startswith("avg_return"):
            top[col] = top[col].map(lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "—")

    lines = [
        f"Grid Search Summary (sorted by {sort_by}, top {top_n})",
        f"Total combinations: {len(result_df)}",
        "",
        top.to_string(index=False),
    ]

    valid = result_df[result_df.get("ranking_eligible", result_df["sample_count"].gt(0)).fillna(False)]
    if not valid.empty and sort_by in valid.columns:
        best = valid.iloc[0]
        param_str = ", ".join(f"{k}={best[k]}" for k in param_names if k in best)
        best_val = best[sort_by]
        if "rate" in sort_by or "ratio" in sort_by.lower():
            val_str = f"{best_val * 100:.2f}%"
        else:
            val_str = f"{best_val:.4f}"
        lines.extend([
            "",
            f"Best: {param_str}",
            f"  {sort_by} = {val_str}, sample_count = {int(best.get('sample_count', 0))}",
        ])

    return "\n".join(lines)


def _objective_count_column(objective: str) -> str:
    """Use observations of the selected label, not the total event count."""
    if objective in {"avg_oo_return_1d", "net_oo_return_1d", "oo_win_rate_1d", "net_oo_win_rate_1d"}:
        return "valid_oo_count_1d"
    if objective == "net_oo_alpha_1d":
        return "valid_oo_alpha_count_1d"
    if match := re.fullmatch(r"(?:avg|net)_entry_alpha_(\d+)d", objective):
        return f"valid_entry_alpha_count_{match[1]}d"
    if match := re.fullmatch(r"(?:avg_entry_return|net_entry_return|entry_win_rate|net_entry_win_rate)_(\d+)d", objective):
        return f"valid_entry_count_{match[1]}d"
    if match := re.fullmatch(r"avg_research_close_alpha_(\d+)d", objective):
        return f"valid_close_alpha_count_{match[1]}d"
    if match := re.fullmatch(r"(?:avg_return|median_return|std_return|win_rate)_(\d+)d", objective):
        return f"valid_return_count_{match[1]}d"
    if objective.endswith("_rate"):
        return "valid_" + objective.removesuffix("_rate") + "_count"
    raise ValueError(f"Unsupported ranking objective: {objective}")


def grid_search(
    param_grid: dict[str, list],
    event_path: str | None = None,
    event_df: pd.DataFrame | None = None,
    base_filters: dict | None = None,
    horizons: list[int] | None = None,
    sort_by: str = "net_oo_return_1d",
    ascending: bool = False,
    save_path: str | None = None,
    plot_dir: str | None = "data/output/grid_search_plots",
    generate_plots: bool = True,
    oos_split_date: str | None = None,
    min_sample_count: int = 30,
    round_trip_cost: float = 0.0,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Run grid search over threshold combinations.

    Parameters
    ----------
    param_grid : dict[str, list]
        Dictionary mapping filter keys to lists of values to try.
        Example::

            {
                "min_buy1_concentration": [0.2, 0.3, 0.4, 0.5],
                "min_net_buy_ratio": [0.01, 0.03, 0.05, 0.08],
                "min_lhb_turnover_ratio": [0.05, 0.10, 0.15, 0.20],
            }

    event_path : str or None
        Path to labeled event parquet file.
    event_df : pd.DataFrame or None
        Pre-loaded event DataFrame. If provided, event_path is ignored.
    base_filters : dict or None
        Filters that are always applied (e.g., exclude_ST, include_only_limit_up_event).
    horizons : list[int] or None
        Holding period horizons. Default: [1, 2, 3, 5, 10].
    sort_by : str
        Training metric to rank by. Default: "net_oo_return_1d".
    ascending : bool
        Sort order. Default: False (descending).
    save_path : str or None
        Path to save results CSV. None does not write a CSV.
    plot_dir : str or None
        Directory for PNG plots. Default: data/output/grid_search_plots.
        Set to None to skip plot generation.
    generate_plots : bool
        Whether to auto-generate visualization PNGs. Default: True.
    oos_split_date : str or None
        When set ("YYYY-MM-DD"), metrics are computed separately for events
        before the date (in-sample) and on/after it (out-of-sample, columns
        prefixed ``oos_``). Training labels crossing the split are excluded.
        OOS columns cannot be ranking objectives.
    min_sample_count : int
        Combinations with fewer in-sample events are pushed to the bottom of
        the ranking (their metrics are kept but not trusted).
    round_trip_cost : float
        Round-trip cost (decimal) forwarded to the statistics layer.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: all param columns + all stat columns,
        sorted by sort_by column.
    """
    if base_filters is None:
        base_filters = {}
    if horizons is None:
        horizons = [1, 2, 3, 5, 10]
    if sort_by.startswith("oos_"):
        raise ValueError("Held-out metrics cannot select parameters; rank on training metrics")
    if "entry_return_1d" in sort_by:
        raise ValueError("Same-day round trip violates A-share T+1; use net_oo_return_1d")
    count_col = _objective_count_column(sort_by)
    if not horizons or any(type(n) is not int or n < 1 for n in horizons):
        raise ValueError("horizons must contain positive integers")
    if min_sample_count < 1:
        raise ValueError("min_sample_count must be positive")

    # Load data once
    if event_df is not None:
        df = event_df.copy()
    else:
        path = event_path or _DEFAULT_EVENT_PATH
        logger.info(f"Loading event data from {path}")
        df = load_parquet(path)
        if "label_schema_version" not in df or not df.label_schema_version.eq(2).all():
            raise ValueError("Legacy labels detected: rebuild clean/features/labels")

    # Pre-apply base filters to reduce computation per iteration
    df = apply_filters(df, base_filters)
    if start_date or end_date:
        from src.backtest.filter_engine import _apply_date_filter
        df = _apply_date_filter(df, start_date, end_date)
    logger.info(f"After base filters: {len(df)} events")

    # IS/OOS split
    oos_df: pd.DataFrame | None = None
    if oos_split_date:
        if "trade_date" not in df:
            raise ValueError("Temporal splitting requires trade_date")
        oos_split_date = pd.Timestamp(oos_split_date).strftime("%Y-%m-%d")
        end_col = f"label_end_date_{max(2, max(horizons))}d"
        if end_col not in df:
            raise ValueError(f"Missing {end_col}; regenerate labels before purged temporal splitting")
        dates = df["trade_date"].astype(str)
        oos_df = df[dates >= oos_split_date]
        df = df[(dates < oos_split_date) & df[end_col].notna()
                & (df[end_col] < oos_split_date)]
        logger.info(
            f"IS/OOS split at {oos_split_date}: {len(df)} in-sample, "
            f"{len(oos_df)} out-of-sample events"
        )

    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = itertools.product(*param_values)
    total = math.prod(len(values) for values in param_values)
    logger.info(f"Grid search: {total} combinations over {param_names}")

    results: list[dict[str, Any]] = []

    for combo in tqdm(combinations, desc="Grid Search", total=total):
        combo_filters = dict(zip(param_names, combo))
        row: dict[str, Any] = {name: val for name, val in zip(param_names, combo)}

        # In-sample statistics
        filtered_df = apply_filters(df, combo_filters)
        if len(filtered_df) == 0:
            row["sample_count"] = 0
        else:
            summary = compute_backtest_statistics(
                filtered_df, horizons=horizons, round_trip_cost=round_trip_cost,
            )
            for col in summary.columns:
                row[col] = summary[col].iloc[0]

        # Out-of-sample statistics (same filters on held-out period)
        if oos_df is not None:
            oos_filtered = apply_filters(oos_df, combo_filters)
            row["oos_sample_count"] = len(oos_filtered)
            if len(oos_filtered) > 0:
                oos_summary = compute_backtest_statistics(
                    oos_filtered, horizons=horizons, round_trip_cost=round_trip_cost,
                )
                for col in oos_summary.columns:
                    if col != "sample_count":
                        row[f"oos_{col}"] = oos_summary[col].iloc[0]

        results.append(row)

    result_df = pd.DataFrame(results)
    if sort_by not in result_df and not result_df.empty and result_df.get("sample_count", pd.Series(dtype=float)).gt(0).any():
        raise ValueError(f"Objective {sort_by} is unavailable; no ranking was produced")

    # Sort by target metric; under-sampled combinations sink to the bottom
    if sort_by in result_df.columns:
        result_df["ranking_sample_count"] = result_df.get(count_col, pd.Series(0, index=result_df.index)).fillna(0)
        result_df["ranking_eligible"] = (result_df.ranking_sample_count.ge(min_sample_count)
                                         & result_df[sort_by].notna())
        result_df = result_df.sort_values(
            ["ranking_eligible", sort_by], ascending=[False, ascending],
        ).reset_index(drop=True)

    # Save results
    if save_path is not None:
        save_csv(result_df, save_path)
        logger.info(f"Grid search results saved to {save_path} ({len(result_df)} rows)")

    if generate_plots and plot_dir and not result_df.empty:
        from src.reports.plot_results import plot_grid_search_results

        # The requested metric may be absent (e.g. no OOS events matched any
        # combination) — fall back to a metric that actually exists.
        plot_metric = sort_by
        if plot_metric not in result_df.columns:
            candidates = ["next_day_limit_up_rate", "avg_return_1d", "sample_count"]
            plot_metric = next((c for c in candidates if c in result_df.columns), None)
            logger.warning(
                f"sort_by column '{sort_by}' missing from results — plotting {plot_metric}"
            )
        if plot_metric is not None:
            plot_paths = plot_grid_search_results(
                result_df,
                param_grid=param_grid,
                sort_by=plot_metric,
                output_dir=plot_dir,
            )
            logger.info(f"Grid search plots: {len(plot_paths)} files in {plot_dir}")

    return result_df
