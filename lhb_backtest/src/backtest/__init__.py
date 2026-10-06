"""Backtest engine modules for the LHB Backtesting System."""

from src.backtest.filter_engine import BacktestResult, apply_filters, run_lhb_backtest
from src.backtest.grid_search import grid_search
from src.backtest.statistics import (
    compute_backtest_statistics,
    compute_grouped_statistics,
    compute_quantile_statistics,
    compute_rolling_period_statistics,
)

__all__ = [
    "BacktestResult",
    "apply_filters",
    "run_lhb_backtest",
    "grid_search",
    "compute_backtest_statistics",
    "compute_grouped_statistics",
    "compute_quantile_statistics",
    "compute_rolling_period_statistics",
]
