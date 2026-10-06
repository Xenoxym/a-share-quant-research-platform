"""Tests for the backtest filter engine (src.backtest.filter_engine)."""

import pytest
import pandas as pd
import numpy as np

from src.backtest.filter_engine import BacktestResult, apply_filters, run_lhb_backtest
from src.backtest.statistics import compute_backtest_statistics


def _build_labeled_df(n: int = 10) -> pd.DataFrame:
    """Build a realistic labeled DataFrame with *n* events."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-03", periods=n).strftime("%Y-%m-%d").tolist()

    stock_codes = [
        "000001.SZ", "600000.SH", "300001.SZ", "688001.SH", "000002.SZ",
        "600001.SH", "300002.SZ", "688002.SH", "000003.SZ", "600002.SH",
    ][:n]
    stock_names = [
        "平安银行", "浦发银行", "特锐德", "某科创A", "*ST某某",
        "招商银行", "创业B", "科创B", "万科A", "东方证券",
    ][:n]

    pct_chg_vals = [10.0, 10.5, 20.0, 20.5, 3.0, 9.9, 5.0, 19.8, 10.1, 7.5][:n]

    return pd.DataFrame({
        "trade_date": dates,
        "stock_code": stock_codes,
        "stock_name": stock_names,
        "is_st": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0][:n],
        "close": rng.uniform(8, 50, n).round(2).tolist(),
        "pct_chg": pct_chg_vals,
        "event_day_limit_up": [1, 1, 1, 1, 0, 1, 0, 1, 1, 0][:n],
        "lhb_reason": ["日涨幅达7%"] * (n - 2) + ["日跌幅达7%"] * min(2, n),
        "amount": rng.uniform(5_000_000, 100_000_000, n).round(0).tolist(),
        "turnover_rate": rng.uniform(1.0, 15.0, n).round(2).tolist(),
        "buy_sell_ratio": [1.7, 2.0, 1.2, 0.8, 1.5, 2.5, 1.1, 1.8, 1.3, 0.9][:n],
        "buy1_concentration": [0.45, 0.35, 0.50, 0.20, 0.30, 0.60, 0.25, 0.55, 0.40, 0.15][:n],
        "net_buy_ratio": [0.05, 0.08, 0.03, -0.02, 0.04, 0.10, 0.01, 0.06, 0.03, -0.01][:n],
        "net_buy_float_mcap_ratio": [0.006, 0.01, 0.003, -0.002, 0.004, 0.012, 0.001, 0.008, 0.003, -0.001][:n],
        "lhb_turnover_ratio": [0.10, 0.15, 0.08, 0.25, 0.12, 0.20, 0.06, 0.18, 0.09, 0.30][:n],
        "future_return_1d": rng.uniform(-0.05, 0.10, n).round(4).tolist(),
        "future_return_3d": rng.uniform(-0.08, 0.15, n).round(4).tolist(),
        "future_return_5d": rng.uniform(-0.10, 0.20, n).round(4).tolist(),
        "future_return_10d": rng.uniform(-0.15, 0.30, n).round(4).tolist(),
        "future_max_return_5d": rng.uniform(0.0, 0.25, n).round(4).tolist(),
        "future_max_drawdown_5d": rng.uniform(-0.20, 0.0, n).round(4).tolist(),
        "next_day_limit_up": [True, True, False, True, False, True, False, True, False, False][:n],
        "within_3d_limit_up": [True, True, True, True, False, True, False, True, True, False][:n],
        "within_5d_limit_up": [True, True, True, True, False, True, True, True, True, False][:n],
        "continue_board_1d": [True, True, False, True, False, True, False, True, False, False][:n],
        "continue_board_2d": [True, False, False, True, False, False, False, True, False, False][:n],
        "continue_board_3d": [False, False, False, True, False, False, False, False, False, False][:n],
        "max_continue_board_days": [2, 1, 0, 3, 0, 1, 0, 2, 0, 0][:n],
    })


class TestApplyFilters:
    """Test suite for apply_filters()."""

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return _build_labeled_df(10)

    def test_min_buy_sell_ratio(self, df) -> None:
        result = apply_filters(df, {"min_buy_sell_ratio": 1.5})
        assert (result["buy_sell_ratio"] >= 1.5).all()
        assert len(result) < len(df)

    def test_min_buy1_concentration(self, df) -> None:
        result = apply_filters(df, {"min_buy1_concentration": 0.4})
        assert (result["buy1_concentration"] >= 0.4).all()

    def test_max_turnover_rate(self, df) -> None:
        result = apply_filters(df, {"max_turnover_rate": 5.0})
        assert (result["turnover_rate"] <= 5.0).all()

    def test_multiple_filters(self, df) -> None:
        result = apply_filters(df, {
            "min_buy_sell_ratio": 1.0,
            "min_buy1_concentration": 0.3,
        })
        assert (result["buy_sell_ratio"] >= 1.0).all()
        assert (result["buy1_concentration"] >= 0.3).all()

    def test_exclude_st(self, df) -> None:
        result = apply_filters(df, {"exclude_ST": True})
        assert not result["stock_name"].str.contains("ST", case=False, na=False).any()

    def test_include_only_limit_up(self, df) -> None:
        result = apply_filters(df, {"include_only_limit_up_event": True})
        assert (result["pct_chg"] >= 9.8).all()

    def test_board_type_filter(self, df) -> None:
        result = apply_filters(df, {"board_type": ["main_sz", "main_sh"]})
        prefixes = result["stock_code"].str[:2]
        assert prefixes.isin(["00", "60"]).all()

    def test_lhb_reason_filter(self, df) -> None:
        result = apply_filters(df, {"lhb_reason": "涨幅"})
        assert result["lhb_reason"].str.contains("涨幅", na=False).all()

    def test_empty_result(self, df) -> None:
        result = apply_filters(df, {"min_buy_sell_ratio": 999})
        assert len(result) == 0

    def test_no_filters(self, df) -> None:
        result = apply_filters(df, {})
        assert len(result) == len(df)

    def test_preserves_columns(self, df) -> None:
        result = apply_filters(df, {"min_buy_sell_ratio": 1.0})
        assert set(df.columns).issubset(set(result.columns))


class TestBacktestStatistics:
    """Test compute_backtest_statistics()."""

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return _build_labeled_df(10)

    def test_sample_count(self, df) -> None:
        stats = compute_backtest_statistics(df, horizons=[1, 5])
        assert stats["sample_count"].iloc[0] == len(df)

    def test_return_stats(self, df) -> None:
        stats = compute_backtest_statistics(df, horizons=[1, 5])
        assert "avg_return_1d" in stats.columns
        assert "avg_return_5d" in stats.columns
        assert "win_rate_1d" in stats.columns

    def test_limit_up_rates(self, df) -> None:
        stats = compute_backtest_statistics(df)
        assert "next_day_limit_up_rate" in stats.columns
        rate = stats["next_day_limit_up_rate"].iloc[0]
        assert 0 <= rate <= 1

    def test_empty_df(self) -> None:
        empty = pd.DataFrame()
        stats = compute_backtest_statistics(empty)
        assert stats["sample_count"].iloc[0] == 0


class TestRunLhbBacktest:
    """Test the main run_lhb_backtest() entry point."""

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        return _build_labeled_df(10)

    def test_basic_run(self, df) -> None:
        result = run_lhb_backtest(event_df=df)
        assert isinstance(result, BacktestResult)
        assert result.sample_count == len(df)

    def test_with_filters(self, df) -> None:
        result = run_lhb_backtest(
            event_df=df,
            filters={"exclude_ST": True, "min_buy_sell_ratio": 1.5},
        )
        assert result.sample_count <= len(df)
        assert not result.detail["stock_name"].str.contains("ST", case=False, na=False).any()

    def test_with_date_range(self, df) -> None:
        dates = sorted(df["trade_date"].unique())
        result = run_lhb_backtest(
            event_df=df,
            start_date=dates[2],
            end_date=dates[6],
        )
        assert (result.detail["trade_date"] >= dates[2]).all()
        assert (result.detail["trade_date"] <= dates[6]).all()

    def test_summary_not_empty(self, df) -> None:
        result = run_lhb_backtest(event_df=df, horizons=[1, 5])
        assert not result.summary.empty
        assert "avg_return_1d" in result.summary.columns

    def test_to_csv(self, df, tmp_path) -> None:
        result = run_lhb_backtest(event_df=df)
        csv_path = str(tmp_path / "output.csv")
        result.to_csv(csv_path)
        summary_file = tmp_path / "output_summary.csv"
        detail_file = tmp_path / "output_detail.csv"
        assert summary_file.exists()
        assert detail_file.exists()

    def test_repr(self, df) -> None:
        result = run_lhb_backtest(event_df=df)
        text = repr(result)
        assert "BacktestResult" in text
        assert "sample_count" in text
