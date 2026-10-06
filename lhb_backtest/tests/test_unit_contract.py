"""Tests for the unit contract and the P0 correctness fixes.

Unit contract:
- ``pct_chg`` / ``turnover_rate`` are PERCENT values.
- Every other ratio / return column is a DECIMAL FRACTION.

Covers:
- clean_lhb_summary percent→decimal conversion
- lhb_window_days detection & masking of cross-window factors
- seat-internal buy1/sell1 concentration bounded in (0, 1]
- exact-price limit-up flags in label generation
- T+1 open entry labels and next_day_untradable
- transaction-cost statistics
- grid_search IS/OOS split
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cleaning.clean_lhb import clean_lhb_summary
from src.features.compute_lhb_factors import compute_lhb_factors
from src.labels.generate_future_labels import generate_future_labels
from src.backtest.statistics import compute_backtest_statistics
from src.backtest.grid_search import grid_search


# ---------------------------------------------------------------------------
# clean_lhb_summary: percent → decimal
# ---------------------------------------------------------------------------

class TestPercentToDecimal:
    def test_ratio_columns_converted(self, tmp_path):
        raw = pd.DataFrame({
            "trade_date": ["2024-01-05"],
            "stock_code": ["000001.SZ"],
            "stock_name": ["平安银行"],
            "lhb_reason": ["日涨幅偏离值达到7%"],
            "net_buy_ratio": [17.5],          # percent from API
            "lhb_turnover_ratio": [26.8],     # percent from API
            "turnover_rate": [15.5],          # stays percent
            "pct_chg": [10.0],                # stays percent
            "return_1d": [10.05],             # percent from API
        })
        out = clean_lhb_summary(raw_df=raw, save_path=str(tmp_path / "clean.parquet"))

        assert out["net_buy_ratio"].iloc[0] == pytest.approx(0.175)
        assert out["lhb_turnover_ratio"].iloc[0] == pytest.approx(0.268)
        assert out["return_1d"].iloc[0] == pytest.approx(0.1005)
        # percent columns unchanged
        assert out["turnover_rate"].iloc[0] == pytest.approx(15.5)
        assert out["pct_chg"].iloc[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# compute_lhb_factors: window detection + masking
# ---------------------------------------------------------------------------

def _factor_events() -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": ["2024-01-05", "2024-01-05"],
        "stock_code": ["000001.SZ", "300058.SZ"],
        "stock_name": ["平安银行", "蓝色光标"],
        "lhb_reason": [
            "日涨幅偏离值达到7%的前5只证券",
            "连续三个交易日内，涨幅偏离值累计达到30%的证券",
        ],
        "lhb_buy_total": [1e8, 2.8e9],
        "lhb_sell_total": [5e7, 3.1e9],
        "lhb_net_buy": [5e7, -2e8],
        "amount": [5e8, 2.3e10],
        "float_market_cap": [1e10, 5e10],
        # Seats: single-day board consistent; 3-day board cumulated (larger)
        "buy1_amount": [4e7, 8.5e10],
        "buy2_amount": [3e7, 4.5e10],
        "buy3_amount": [2e7, 4.0e10],
        "buy4_amount": [5e6, 2.5e10],
        "buy5_amount": [5e6, 5.0e9],
        "sell1_amount": [2e7, 3.0e10],
        "sell2_amount": [1e7, 2.0e10],
        "sell3_amount": [1e7, 1.0e10],
        "sell4_amount": [5e6, 5.0e9],
        "sell5_amount": [5e6, 5.0e9],
    })


class TestWindowSeparation:
    def test_window_days_detected(self):
        df = compute_lhb_factors(event_df=_factor_events(), save_path=None)
        assert df["lhb_window_days"].tolist() == [1, 3]

    def test_seat_internal_concentration_bounded(self):
        df = compute_lhb_factors(event_df=_factor_events(), save_path=None)
        # buy1_concentration = buy1 / Σ(buy1..5) — valid for both windows
        assert 0 < df["buy1_concentration"].iloc[0] <= 1
        assert 0 < df["buy1_concentration"].iloc[1] <= 1
        assert df["buy1_concentration"].iloc[0] == pytest.approx(4e7 / 1e8)

    def test_cross_window_factors_masked_for_3d_board(self):
        df = compute_lhb_factors(event_df=_factor_events(), save_path=None)
        # single-day row keeps cross-source factors
        assert not np.isnan(df["buy1_to_lhb_buy_ratio"].iloc[0])
        assert not np.isnan(df["buy1_to_daily_amount_ratio"].iloc[0])
        # 3-day row has them masked
        assert np.isnan(df["buy1_to_lhb_buy_ratio"].iloc[1])
        assert np.isnan(df["buy1_to_daily_amount_ratio"].iloc[1])

    def test_summary_only_factors_valid_for_both(self):
        df = compute_lhb_factors(event_df=_factor_events(), save_path=None)
        assert not df["buy_sell_ratio"].isna().any()
        assert not df["net_buy_float_mcap_ratio"].isna().any()


# ---------------------------------------------------------------------------
# Label generation: exact limit price, entry labels, untradable flag
# ---------------------------------------------------------------------------

def _kline_with_limits() -> pd.DataFrame:
    """4 trading days; day2 closes exactly at high_limit, day2 opens at limit."""
    dates = ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"]
    closes = [10.0, 11.0, 11.5, 11.0]
    opens = [9.8, 11.0, 11.3, 11.4]     # day2 opens at its limit (11.0)
    highs = [10.1, 11.0, 11.6, 11.5]
    lows = [9.7, 10.8, 11.2, 10.9]
    pre_closes = [9.5, 10.0, 11.0, 11.5]
    high_limits = [10.45, 11.0, 12.1, 12.65]
    pct = [(c / p - 1) * 100 for c, p in zip(closes, pre_closes)]
    return pd.DataFrame({
        "trade_date": dates,
        "stock_code": ["000001.SZ"] * 4,
        "open": opens, "high": highs, "low": lows, "close": closes,
        "pre_close": pre_closes, "pct_chg": pct,
        "volume": [1e6] * 4, "amount": [1e7] * 4,
        "turnover_rate": [2.0] * 4,
        "high_limit": high_limits,
        "low_limit": [round(p * 0.9, 2) for p in pre_closes],
    })


class TestExactLimitAndEntryLabels:
    @pytest.fixture
    def labeled(self, tmp_path):
        event = pd.DataFrame({
            "trade_date": ["2024-01-08"],
            "stock_code": ["000001.SZ"],
            "stock_name": ["平安银行"],
            "pct_chg": [5.26],
        })
        return generate_future_labels(
            event_df=event,
            kline_df=_kline_with_limits(),
            horizons=[1, 2, 3],
            save_path=str(tmp_path / "labeled.parquet"),
            include_alpha=False,
            include_suspension=False,
        )

    def test_next_day_limit_up_exact_price(self, labeled):
        # Day2 close 11.0 == high_limit 11.0 → limit-up even though
        # pct_chg = 10.0 > 9.8 would also pass; exact path is used.
        assert labeled["next_day_limit_up"].iloc[0] == 1.0

    def test_entry_open_return(self, labeled):
        # T+1 open = 11.0; 2d close (2024-01-10) = 11.5
        expected = (11.5 - 11.0) / 11.0
        assert labeled["entry_open_return_2d"].iloc[0] == pytest.approx(expected)

    def test_untradable_when_open_at_limit(self, labeled):
        # T+1 opened exactly at its high_limit (11.0) → cannot buy
        assert labeled["next_day_open_at_limit"].iloc[0] == 1.0
        assert labeled["next_day_untradable"].iloc[0] == 1.0

    def test_entry_open_gap(self, labeled):
        expected = (11.0 - 10.0) / 10.0
        assert labeled["entry_open_gap"].iloc[0] == pytest.approx(expected)

    def test_close_based_labels_unchanged(self, labeled):
        assert labeled["future_return_1d"].iloc[0] == pytest.approx(0.10)


class TestTradableEntry:
    def test_not_untradable_when_open_below_limit(self, tmp_path):
        kline = _kline_with_limits()
        kline.loc[1, "open"] = 10.5  # T+1 opens below limit → tradable
        event = pd.DataFrame({
            "trade_date": ["2024-01-08"],
            "stock_code": ["000001.SZ"],
            "stock_name": ["平安银行"],
            "pct_chg": [5.26],
        })
        out = generate_future_labels(
            event_df=event, kline_df=kline, horizons=[1],
            save_path=str(tmp_path / "labeled.parquet"),
            include_alpha=False, include_suspension=False,
        )
        assert out["next_day_open_at_limit"].iloc[0] == 0.0
        assert out["next_day_untradable"].iloc[0] == 0.0

    def test_untradable_when_no_next_kline(self, tmp_path):
        kline = _kline_with_limits().iloc[:1]  # only the event day exists
        event = pd.DataFrame({
            "trade_date": ["2024-01-08"],
            "stock_code": ["000001.SZ"],
            "stock_name": ["平安银行"],
            "pct_chg": [5.26],
        })
        out = generate_future_labels(
            event_df=event, kline_df=kline, horizons=[1],
            save_path=str(tmp_path / "labeled.parquet"),
            include_alpha=False, include_suspension=False,
        )
        assert out["next_day_untradable"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# Statistics: net-of-cost entry returns
# ---------------------------------------------------------------------------

class TestCostStatistics:
    def test_net_entry_return_subtracts_cost(self):
        df = pd.DataFrame({
            "entry_open_return_1d": [0.05, 0.01, -0.02],
            "entry_open_exit_open_1d": [0.02, -0.01, -0.03],
            "future_return_1d": [0.06, 0.02, -0.01],
        })
        stats = compute_backtest_statistics(df, horizons=[1], round_trip_cost=0.002)
        row = stats.iloc[0]
        assert "net_entry_return_1d" not in row
        assert row["research_intraday_return_1d"] == pytest.approx(np.mean([0.05, 0.01, -0.02]))
        assert row["net_oo_return_1d"] == pytest.approx(np.mean([0.02, -0.01, -0.03]) - 0.002)
        # 0.01 - 0.002 > 0 still a win; win rates equal here
        assert row["net_oo_win_rate_1d"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Grid search: IS/OOS split
# ---------------------------------------------------------------------------

class TestGridSearchOOS:
    def test_oos_columns_present_and_split_correct(self, tmp_path):
        rng = np.random.default_rng(7)
        n = 200
        dates = ["2024-01-05"] * 100 + ["2025-01-06"] * 100
        df = pd.DataFrame({
            "trade_date": dates,
            "label_end_date_10d": ["2024-01-19"] * 100 + ["2025-01-20"] * 100,
            "stock_code": ["000001.SZ"] * n,
            "buy1_concentration": rng.uniform(0, 1, n),
            "future_return_1d": rng.normal(0.01, 0.05, n),
            "next_day_limit_up": rng.integers(0, 2, n).astype(float),
        })
        result = grid_search(
            param_grid={"min_buy1_concentration": [0.0, 0.5]},
            event_df=df,
            sort_by="next_day_limit_up_rate",
            oos_split_date="2024-07-01",
            generate_plots=False,
            save_path=str(tmp_path / "grid.csv"),
            min_sample_count=10,
        )
        assert "oos_sample_count" in result.columns
        assert "oos_next_day_limit_up_rate" in result.columns
        base = result[result["min_buy1_concentration"] == 0.0].iloc[0]
        assert base["sample_count"] == 100      # in-sample half
        assert base["oos_sample_count"] == 100  # out-of-sample half
