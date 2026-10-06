"""Visualization functions for backtest results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging import get_logger

logger = get_logger("plot_results")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


def plot_return_distribution(
    df: pd.DataFrame,
    return_col: str = "future_return_1d",
    title: str = "Return Distribution",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot histogram of returns with mean/median lines.

    Parameters
    ----------
    df : pd.DataFrame
        Event DataFrame.
    return_col : str
        Column name for returns. Default: "future_return_1d".
    title : str
        Chart title.
    save_path : str or None
        Path to save figure. If None, figure is not saved.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    returns = df[return_col].dropna()

    ax.hist(returns, bins=50, alpha=0.7, color="#4C72B0", edgecolor="white", density=True)

    mean_val = returns.mean()
    median_val = returns.median()

    ax.axvline(mean_val, color="#C44E52", linestyle="--", linewidth=2,
               label=f"Mean: {mean_val:.4f} ({mean_val*100:.2f}%)")
    ax.axvline(median_val, color="#55A868", linestyle="-.", linewidth=2,
               label=f"Median: {median_val:.4f} ({median_val*100:.2f}%)")
    ax.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Return", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)

    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
    stats_text = (
        f"N={len(returns):,}  Win Rate={win_rate:.1%}\n"
        f"Std={returns.std():.4f}  Skew={returns.skew():.2f}"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_factor_vs_return(
    df: pd.DataFrame,
    factor_col: str,
    return_col: str = "future_return_1d",
    n_bins: int = 10,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Binned bar chart: factor quantiles on x-axis, avg return on y-axis.

    Parameters
    ----------
    df : pd.DataFrame
        Event DataFrame.
    factor_col : str
        Factor column for x-axis grouping.
    return_col : str
        Return column for y-axis. Default: "future_return_1d".
    n_bins : int
        Number of bins. Default: 10.
    title : str or None
        Chart title. Auto-generated if None.
    save_path : str or None
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    if title is None:
        title = f"{factor_col} vs {return_col}"

    work_df = df[[factor_col, return_col]].dropna().copy()
    if len(work_df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"{title} (no data)")
        return fig

    work_df["bin"] = pd.qcut(work_df[factor_col], q=n_bins, duplicates="drop")

    grouped = work_df.groupby("bin", observed=True).agg(
        avg_return=(return_col, "mean"),
        count=(return_col, "count"),
        factor_mean=(factor_col, "mean"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = range(len(grouped))
    colors = ["#C44E52" if r < 0 else "#55A868" for r in grouped["avg_return"]]
    bars = ax1.bar(x, grouped["avg_return"] * 100, color=colors, alpha=0.8, edgecolor="white")

    ax1.set_xlabel(f"{factor_col} (Quantile Bins)", fontsize=12)
    ax1.set_ylabel("Avg Return (%)", fontsize=12, color="#4C72B0")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Q{i+1}" for i in x], fontsize=10)
    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, grouped["count"], color="#DD8452", marker="o", linewidth=2, label="Sample Count")
    ax2.set_ylabel("Sample Count", fontsize=12, color="#DD8452")

    ax1.set_title(title, fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_grid_search_top_n(
    grid_df: pd.DataFrame,
    param_cols: list[str],
    value_col: str = "next_day_limit_up_rate",
    top_n: int = 15,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of the top-N parameter combinations."""
    if title is None:
        title = f"Top {top_n} by {value_col}"

    work = grid_df.dropna(subset=[value_col]).copy()
    if work.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"{title} (no data)")
        return fig

    top = work.nlargest(top_n, value_col)
    labels = [
        ", ".join(f"{col}={top.iloc[i][col]:g}" for col in param_cols)
        for i in range(len(top))
    ]

    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(top) + 1)))
    y = range(len(top))
    values = top[value_col].values * 100
    colors = ["#55A868" if v >= values.mean() else "#C44E52" for v in values]
    ax.barh(y, values, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"{value_col} (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    for i, (val, n) in enumerate(zip(values, top["sample_count"].values)):
        ax.text(val + 0.3, i, f"n={int(n)}", va="center", fontsize=8, color="#555555")

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_grid_search_metric_lines(
    grid_df: pd.DataFrame,
    param_col: str,
    metric_cols: list[str] | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Line chart: one param value on x-axis, metrics averaged over other params."""
    if metric_cols is None:
        metric_cols = ["next_day_limit_up_rate", "avg_return_1d", "win_rate_1d"]

    available = [c for c in metric_cols if c in grid_df.columns]
    if param_col not in grid_df.columns or not available:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_title(f"Grid Search by {param_col} (missing columns)")
        return fig

    if title is None:
        title = f"Metric vs {param_col} (other params averaged)"

    grouped = (
        grid_df.groupby(param_col, observed=True)[available]
        .mean()
        .reset_index()
        .sort_values(param_col)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = grouped[param_col].values
    for col in available:
        label = col.replace("_", " ")
        if "rate" in col or "ratio" in col.lower():
            ax.plot(x, grouped[col] * 100, marker="o", linewidth=2, label=f"{label} (%)")
        else:
            ax.plot(x, grouped[col] * 100, marker="o", linewidth=2, label=f"{label} (%)")

    ax.set_xlabel(param_col, fontsize=12)
    ax.set_ylabel("Value (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_grid_search_results(
    grid_df: pd.DataFrame,
    param_grid: dict[str, list],
    sort_by: str = "next_day_limit_up_rate",
    output_dir: str = "data/output/grid_search_plots",
) -> list[str]:
    """Generate a standard set of grid-search visualization files.

    Returns paths to saved PNG files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    param_names = list(param_grid.keys())
    saved: list[str] = []

    if len(param_names) >= 2:
        for i, x_param in enumerate(param_names):
            for y_param in param_names[i + 1:]:
                fname = out / f"heatmap_{sort_by}__{y_param}_vs_{x_param}.png"
                plot_grid_search_heatmap(
                    grid_df,
                    x_param=x_param,
                    y_param=y_param,
                    value_col=sort_by,
                    save_path=str(fname),
                )
                saved.append(str(fname))

                count_fname = out / f"heatmap_sample_count__{y_param}_vs_{x_param}.png"
                if "sample_count" in grid_df.columns:
                    plot_grid_search_heatmap(
                        grid_df,
                        x_param=x_param,
                        y_param=y_param,
                        value_col="sample_count",
                        title=f"Sample Count: {y_param} vs {x_param}",
                        save_path=str(count_fname),
                    )
                    saved.append(str(count_fname))

    if param_names:
        for param in param_names:
            line_fname = out / f"lines_{sort_by}_vs_{param}.png"
            plot_grid_search_metric_lines(
                grid_df,
                param_col=param,
                save_path=str(line_fname),
            )
            saved.append(str(line_fname))

    top_fname = out / f"top15_{sort_by}.png"
    plot_grid_search_top_n(
        grid_df,
        param_cols=param_names,
        value_col=sort_by,
        save_path=str(top_fname),
    )
    saved.append(str(top_fname))

    logger.info(f"Saved {len(saved)} grid-search plots to {out}")
    return saved


def plot_grid_search_heatmap(
    grid_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    value_col: str = "next_day_limit_up_rate",
    title: str | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap of grid search results.

    Parameters
    ----------
    grid_df : pd.DataFrame
        Grid search results DataFrame.
    x_param : str
        Parameter for x-axis.
    y_param : str
        Parameter for y-axis.
    value_col : str
        Value column for cell color. Default: "next_day_limit_up_rate".
    title : str or None
        Chart title.
    save_path : str or None
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    if title is None:
        title = f"Grid Search: {value_col}"

    # If more than 2 params, average over the others
    other_cols = [c for c in grid_df.columns
                  if c not in [x_param, y_param, value_col]
                  and c in grid_df.select_dtypes(include="number").columns]

    pivot_df = grid_df.groupby([y_param, x_param], observed=True)[value_col].mean().reset_index()
    pivot_table = pivot_df.pivot(index=y_param, columns=x_param, values=value_col)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": value_col},
    )

    ax.set_xlabel(x_param, fontsize=12)
    ax.set_ylabel(y_param, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_cumulative_returns(
    df: pd.DataFrame,
    return_col: str = "future_return_5d",
    date_col: str = "trade_date",
    title: str = "Expanding Mean Event Return (not portfolio NAV)",
    save_path: str | None = None,
) -> plt.Figure:
    """Legacy API: plot expanding mean event return, never compounded portfolio NAV.

    Parameters
    ----------
    df : pd.DataFrame
        Event DataFrame with dates and returns.
    return_col : str
        Return column. Default: "future_return_5d".
    date_col : str
        Date column. Default: "trade_date".
    title : str
        Chart title.
    save_path : str or None
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if date_col not in df.columns or return_col not in df.columns:
        ax.set_title(f"{title} (missing columns)")
        return fig

    work_df = df[[date_col, return_col]].dropna().copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col])
    work_df = work_df.sort_values(date_col)

    cum_returns = work_df[return_col].expanding().mean()

    ax.plot(work_df[date_col].values, cum_returns.values, color="#4C72B0", linewidth=1.5)
    ax.fill_between(
        work_df[date_col].values,
        cum_returns.values,
        alpha=0.15,
        color="#4C72B0",
    )
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Mean event return (not NAV)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    total_return = cum_returns.iloc[-1] if len(cum_returns) > 0 else 0
    n_events = len(work_df)
    stats_text = f"Mean event return: {total_return:.2%}  |  Events: {n_events:,}"
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def plot_monthly_stats(
    df: pd.DataFrame,
    return_col: str = "future_return_1d",
    date_col: str = "trade_date",
    title: str = "Monthly Performance",
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart: monthly average returns and sample counts.

    Parameters
    ----------
    df : pd.DataFrame
        Event DataFrame.
    return_col : str
        Return column. Default: "future_return_1d".
    date_col : str
        Date column. Default: "trade_date".
    title : str
        Chart title.
    save_path : str or None
        Path to save figure.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, ax1 = plt.subplots(figsize=(14, 6))

    if date_col not in df.columns or return_col not in df.columns:
        ax1.set_title(f"{title} (missing columns)")
        return fig

    work_df = df[[date_col, return_col]].dropna().copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col])
    work_df["year_month"] = work_df[date_col].dt.to_period("M")

    monthly = work_df.groupby("year_month").agg(
        avg_return=(return_col, "mean"),
        count=(return_col, "count"),
    ).reset_index()

    x = range(len(monthly))
    colors = ["#C44E52" if r < 0 else "#55A868" for r in monthly["avg_return"]]
    ax1.bar(x, monthly["avg_return"] * 100, color=colors, alpha=0.8, edgecolor="white")

    ax1.set_xlabel("Month", fontsize=12)
    ax1.set_ylabel("Avg Return (%)", fontsize=12)
    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, monthly["count"], color="#DD8452", marker="o", linewidth=1.5,
             markersize=4, label="Event Count")
    ax2.set_ylabel("Event Count", fontsize=12, color="#DD8452")
    ax2.legend(loc="upper right", fontsize=10)

    # X-axis labels (show every Nth label to avoid crowding)
    labels = [str(p) for p in monthly["year_month"]]
    step = max(1, len(labels) // 12)
    ax1.set_xticks([i for i in x if i % step == 0])
    ax1.set_xticklabels([labels[i] for i in x if i % step == 0], rotation=45, ha="right")

    ax1.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    _save_figure(fig, save_path)
    return fig


def _save_figure(fig: plt.Figure, save_path: str | None) -> None:
    """Save figure to disk if path is provided."""
    if save_path is None:
        return
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.debug(f"Figure saved to {p}")
