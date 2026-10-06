"""Report generation modules for the LHB Backtesting System."""

from src.reports.make_report import generate_html_report
from src.reports.plot_results import (
    plot_cumulative_returns,
    plot_factor_vs_return,
    plot_grid_search_heatmap,
    plot_grid_search_results,
    plot_monthly_stats,
    plot_return_distribution,
)

__all__ = [
    "generate_html_report",
    "plot_return_distribution",
    "plot_factor_vs_return",
    "plot_grid_search_heatmap",
    "plot_grid_search_results",
    "plot_cumulative_returns",
    "plot_monthly_stats",
]
