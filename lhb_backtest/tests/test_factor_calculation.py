"""Tests for the LHB factor computation module (src.features.compute_lhb_factors)."""

import pytest
import pandas as pd
import numpy as np

from src.features.compute_lhb_factors import compute_lhb_factors


class TestFactorCalculation:
    """Test suite for LHB factor calculations."""

    @pytest.fixture
    def sample_event_df(self) -> pd.DataFrame:
        """Create a sample event DataFrame with known values for factor testing."""
        return pd.DataFrame(
            {
                "trade_date": ["2023-01-03", "2023-01-04", "2023-01-05"],
                "stock_code": ["000001.SZ", "600000.SH", "300001.SZ"],
                "stock_name": ["平安银行", "浦发银行", "特锐德"],
                "lhb_reason": ["日涨幅达7%", "日涨幅达7%", "日涨幅达15%"],
                "open": [10.0, 20.0, 30.0],
                "high": [11.0, 21.0, 33.0],
                "low": [9.5, 19.0, 29.0],
                "close": [10.5, 20.5, 32.0],
                "pre_close": [10.0, 20.0, 28.0],
                "pct_chg": [5.0, 2.5, 14.29],
                "volume": [1_000_000, 2_000_000, 500_000],
                "amount": [10_000_000.0, 40_000_000.0, 15_000_000.0],
                "turnover_rate": [2.5, 1.8, 5.0],
                "float_market_cap": [500_000_000.0, 1_000_000_000.0, 200_000_000.0],
                "buy1_amount": [3_000_000.0, 5_000_000.0, 4_000_000.0],
                "buy2_amount": [2_000_000.0, 3_000_000.0, 2_000_000.0],
                "buy3_amount": [1_000_000.0, 2_000_000.0, 1_000_000.0],
                "buy4_amount": [500_000.0, 1_000_000.0, 500_000.0],
                "buy5_amount": [300_000.0, 500_000.0, 300_000.0],
                "sell1_amount": [2_000_000.0, 4_000_000.0, 3_000_000.0],
                "sell2_amount": [1_000_000.0, 2_000_000.0, 1_500_000.0],
                "sell3_amount": [500_000.0, 1_000_000.0, 800_000.0],
                "sell4_amount": [300_000.0, 500_000.0, 400_000.0],
                "sell5_amount": [200_000.0, 300_000.0, 200_000.0],
                "lhb_buy_total": [6_800_000.0, 11_500_000.0, 7_800_000.0],
                "lhb_sell_total": [4_000_000.0, 7_800_000.0, 5_900_000.0],
                "lhb_net_buy": [2_800_000.0, 3_700_000.0, 1_900_000.0],
                "lhb_turnover": [10_800_000.0, 19_300_000.0, 13_700_000.0],
            }
        )

    # ------------------------------------------------------------------
    # Individual factor correctness
    # ------------------------------------------------------------------

    def test_seat_quality_counts(self, sample_event_df: pd.DataFrame) -> None:
        """inst_buy_count / northbound_buy derived from broker name columns."""
        df = sample_event_df.copy()
        df["buy1_broker"] = ["机构专用", "华泰证券深圳分公司", "机构专用"]
        df["buy2_broker"] = ["机构专用", "沪股通专用", None]
        df["sell1_broker"] = ["机构专用", "招商证券", None]

        result = compute_lhb_factors(df)

        assert result["inst_buy_count"].tolist() == [2.0, 0.0, 1.0]
        assert result["northbound_buy"].tolist() == [0.0, 1.0, 0.0]
        assert result["inst_sell_count"].iloc[0] == 1.0
        assert result["inst_net_seats"].iloc[0] == 1.0

    def test_seat_quality_nan_without_detail(self, sample_event_df: pd.DataFrame) -> None:
        """Events without broker names must yield NaN (not zero)."""
        df = sample_event_df.copy()
        df["buy1_broker"] = ["机构专用", None, ""]

        result = compute_lhb_factors(df)

        assert result["inst_buy_count"].iloc[0] == 1.0
        assert pd.isna(result["inst_buy_count"].iloc[1])
        assert pd.isna(result["inst_buy_count"].iloc[2])

    def test_buy_sell_ratio(self, sample_event_df: pd.DataFrame) -> None:
        """buy_sell_ratio = lhb_buy_total / lhb_sell_total."""
        result = compute_lhb_factors(sample_event_df)

        expected = sample_event_df["lhb_buy_total"] / sample_event_df["lhb_sell_total"]
        pd.testing.assert_series_equal(
            result["buy_sell_ratio"],
            expected,
            check_names=False,
            atol=1e-6,
        )
        # Spot-check first row: 6_800_000 / 4_000_000 = 1.7
        assert result["buy_sell_ratio"].iloc[0] == pytest.approx(1.7, abs=1e-6)

    def test_buy1_concentration(self, sample_event_df: pd.DataFrame) -> None:
        """buy1_concentration = buy1_amount / Σ(buy1..5_amount).

        (In this fixture Σ(buy1..5) == lhb_buy_total, so the expected value
        can be computed either way.)
        """
        result = compute_lhb_factors(sample_event_df)

        expected = sample_event_df["buy1_amount"] / sample_event_df["lhb_buy_total"]
        pd.testing.assert_series_equal(
            result["buy1_concentration"],
            expected,
            check_names=False,
            atol=1e-6,
        )
        # Row 0: 3_000_000 / 6_800_000 ≈ 0.4412
        assert result["buy1_concentration"].iloc[0] == pytest.approx(
            3_000_000 / 6_800_000, abs=1e-4
        )

    def test_sell1_concentration(self, sample_event_df: pd.DataFrame) -> None:
        """sell1_concentration = sell1_amount / Σ(sell1..5_amount).

        (Fixture Σ(sell1..5) == lhb_sell_total, values coincide.)
        """
        result = compute_lhb_factors(sample_event_df)

        expected = sample_event_df["sell1_amount"] / sample_event_df["lhb_sell_total"]
        pd.testing.assert_series_equal(
            result["sell1_concentration"],
            expected,
            check_names=False,
            atol=1e-6,
        )

    def test_net_buy_float_mcap_ratio(self, sample_event_df: pd.DataFrame) -> None:
        """net_buy_float_mcap_ratio = lhb_net_buy / float_market_cap."""
        result = compute_lhb_factors(sample_event_df)

        expected = sample_event_df["lhb_net_buy"] / sample_event_df["float_market_cap"]
        pd.testing.assert_series_equal(
            result["net_buy_float_mcap_ratio"],
            expected,
            check_names=False,
            atol=1e-10,
        )
        # Row 0: 2_800_000 / 500_000_000 = 0.0056
        assert result["net_buy_float_mcap_ratio"].iloc[0] == pytest.approx(0.0056, abs=1e-6)

    def test_lhb_turnover_ratio(self, sample_event_df: pd.DataFrame) -> None:
        """lhb_turnover_ratio = lhb_turnover / amount."""
        result = compute_lhb_factors(sample_event_df)

        expected = sample_event_df["lhb_turnover"] / sample_event_df["amount"]
        pd.testing.assert_series_equal(
            result["lhb_turnover_ratio"],
            expected,
            check_names=False,
            atol=1e-6,
        )

    def test_net_buy_ratio(self, sample_event_df: pd.DataFrame) -> None:
        """net_buy_ratio = lhb_net_buy / amount."""
        result = compute_lhb_factors(sample_event_df)

        expected = sample_event_df["lhb_net_buy"] / sample_event_df["amount"]
        pd.testing.assert_series_equal(
            result["net_buy_ratio"],
            expected,
            check_names=False,
            atol=1e-6,
        )

    def test_buy1_to_daily_amount_ratio(self, sample_event_df: pd.DataFrame) -> None:
        """buy1_to_daily_amount_ratio = buy1_amount / amount."""
        result = compute_lhb_factors(sample_event_df)

        expected = sample_event_df["buy1_amount"] / sample_event_df["amount"]
        pd.testing.assert_series_equal(
            result["buy1_to_daily_amount_ratio"],
            expected,
            check_names=False,
            atol=1e-6,
        )

    def test_top5_buy_avg(self, sample_event_df: pd.DataFrame) -> None:
        """top5_buy_avg = mean of buy1..buy5 amounts."""
        result = compute_lhb_factors(sample_event_df)

        buy_cols = [f"buy{i}_amount" for i in range(1, 6)]
        expected = sample_event_df[buy_cols].mean(axis=1)
        pd.testing.assert_series_equal(
            result["top5_buy_avg"],
            expected,
            check_names=False,
            atol=1e-2,
        )

    def test_top5_sell_avg(self, sample_event_df: pd.DataFrame) -> None:
        """top5_sell_avg = mean of sell1..sell5 amounts."""
        result = compute_lhb_factors(sample_event_df)

        sell_cols = [f"sell{i}_amount" for i in range(1, 6)]
        expected = sample_event_df[sell_cols].mean(axis=1)
        pd.testing.assert_series_equal(
            result["top5_sell_avg"],
            expected,
            check_names=False,
            atol=1e-2,
        )

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_zero_denominator_handling(self) -> None:
        """Zero denominators should produce NaN/inf, not raise exceptions."""
        df = pd.DataFrame(
            {
                "trade_date": ["2023-01-03"],
                "stock_code": ["000001.SZ"],
                "stock_name": ["平安银行"],
                "lhb_reason": ["日涨幅达7%"],
                "open": [10.0],
                "high": [11.0],
                "low": [9.5],
                "close": [10.5],
                "pre_close": [10.0],
                "pct_chg": [5.0],
                "volume": [0],
                "amount": [0.0],
                "turnover_rate": [0.0],
                "float_market_cap": [0.0],
                "buy1_amount": [0.0],
                "buy2_amount": [0.0],
                "buy3_amount": [0.0],
                "buy4_amount": [0.0],
                "buy5_amount": [0.0],
                "sell1_amount": [0.0],
                "sell2_amount": [0.0],
                "sell3_amount": [0.0],
                "sell4_amount": [0.0],
                "sell5_amount": [0.0],
                "lhb_buy_total": [0.0],
                "lhb_sell_total": [0.0],
                "lhb_net_buy": [0.0],
                "lhb_turnover": [0.0],
            }
        )
        result = compute_lhb_factors(df)

        # All ratio columns should be NaN or inf — no exception raised
        assert isinstance(result, pd.DataFrame), "Should return a DataFrame even with zeros"
        ratio_cols = [
            "buy_sell_ratio",
            "buy1_concentration",
            "net_buy_float_mcap_ratio",
            "lhb_turnover_ratio",
            "net_buy_ratio",
            "buy1_to_daily_amount_ratio",
            "sell1_concentration",
        ]
        for col in ratio_cols:
            val = result[col].iloc[0]
            assert np.isnan(val) or np.isinf(val), (
                f"Expected NaN or inf for {col} with zero denominator, got {val}"
            )

    def test_missing_values(self) -> None:
        """NaN inputs should propagate as NaN in outputs without errors."""
        df = pd.DataFrame(
            {
                "trade_date": ["2023-01-03", "2023-01-04"],
                "stock_code": ["000001.SZ", "600000.SH"],
                "stock_name": ["平安银行", "浦发银行"],
                "lhb_reason": ["日涨幅达7%", "日涨幅达7%"],
                "open": [10.0, np.nan],
                "high": [11.0, np.nan],
                "low": [9.5, np.nan],
                "close": [10.5, np.nan],
                "pre_close": [10.0, np.nan],
                "pct_chg": [5.0, np.nan],
                "volume": [1_000_000, np.nan],
                "amount": [10_000_000.0, np.nan],
                "turnover_rate": [2.5, np.nan],
                "float_market_cap": [500_000_000.0, np.nan],
                "buy1_amount": [3_000_000.0, np.nan],
                "buy2_amount": [2_000_000.0, np.nan],
                "buy3_amount": [1_000_000.0, np.nan],
                "buy4_amount": [500_000.0, np.nan],
                "buy5_amount": [300_000.0, np.nan],
                "sell1_amount": [2_000_000.0, np.nan],
                "sell2_amount": [1_000_000.0, np.nan],
                "sell3_amount": [500_000.0, np.nan],
                "sell4_amount": [300_000.0, np.nan],
                "sell5_amount": [200_000.0, np.nan],
                "lhb_buy_total": [6_800_000.0, np.nan],
                "lhb_sell_total": [4_000_000.0, np.nan],
                "lhb_net_buy": [2_800_000.0, np.nan],
                "lhb_turnover": [10_800_000.0, np.nan],
            }
        )
        result = compute_lhb_factors(df)

        # Row 0 should have valid factor values
        assert not np.isnan(result["buy_sell_ratio"].iloc[0]), "Row with data should produce value"

        # Row 1 (all-NaN inputs) should have NaN factors
        ratio_cols = [
            "buy_sell_ratio",
            "buy1_concentration",
            "net_buy_float_mcap_ratio",
        ]
        for col in ratio_cols:
            assert np.isnan(result[col].iloc[1]), (
                f"Expected NaN for {col} with NaN inputs, got {result[col].iloc[1]}"
            )

    def test_all_factors_present(self, sample_event_df: pd.DataFrame) -> None:
        """All expected factor columns must be present in the output."""
        result = compute_lhb_factors(sample_event_df)

        expected_factors = [
            "buy_sell_ratio",
            "buy1_concentration",
            "lhb_turnover_ratio",
            "net_buy_ratio",
            "net_buy_float_mcap_ratio",
            "buy1_to_daily_amount_ratio",
            "sell1_concentration",
            "top5_buy_avg",
            "top5_sell_avg",
        ]
        for factor in expected_factors:
            assert factor in result.columns, f"Missing expected factor column: {factor}"

    def test_original_columns_preserved(self, sample_event_df: pd.DataFrame) -> None:
        """Factor computation should preserve all original columns."""
        original_cols = set(sample_event_df.columns)
        result = compute_lhb_factors(sample_event_df)

        missing = original_cols - set(result.columns)
        assert not missing, f"Original columns lost after factor computation: {missing}"

    def test_row_count_unchanged(self, sample_event_df: pd.DataFrame) -> None:
        """Number of rows should remain unchanged after factor computation."""
        result = compute_lhb_factors(sample_event_df)
        assert len(result) == len(sample_event_df), (
            f"Row count changed: {len(sample_event_df)} → {len(result)}"
        )

    def test_single_row_input(self) -> None:
        """Factor computation should work with a single-row DataFrame."""
        df = pd.DataFrame(
            {
                "trade_date": ["2023-01-03"],
                "stock_code": ["000001.SZ"],
                "stock_name": ["平安银行"],
                "lhb_reason": ["日涨幅达7%"],
                "open": [10.0],
                "high": [11.0],
                "low": [9.5],
                "close": [10.5],
                "pre_close": [10.0],
                "pct_chg": [5.0],
                "volume": [1_000_000],
                "amount": [10_000_000.0],
                "turnover_rate": [2.5],
                "float_market_cap": [500_000_000.0],
                "buy1_amount": [3_000_000.0],
                "buy2_amount": [2_000_000.0],
                "buy3_amount": [1_000_000.0],
                "buy4_amount": [500_000.0],
                "buy5_amount": [300_000.0],
                "sell1_amount": [2_000_000.0],
                "sell2_amount": [1_000_000.0],
                "sell3_amount": [500_000.0],
                "sell4_amount": [300_000.0],
                "sell5_amount": [200_000.0],
                "lhb_buy_total": [6_800_000.0],
                "lhb_sell_total": [4_000_000.0],
                "lhb_net_buy": [2_800_000.0],
                "lhb_turnover": [10_800_000.0],
            }
        )
        result = compute_lhb_factors(df)

        assert len(result) == 1
        assert result["buy_sell_ratio"].iloc[0] == pytest.approx(1.7, abs=1e-6)

    def test_empty_dataframe(self) -> None:
        """Factor computation on an empty DataFrame should return an empty DataFrame."""
        cols = [
            "trade_date", "stock_code", "stock_name", "lhb_reason",
            "open", "high", "low", "close", "pre_close", "pct_chg",
            "volume", "amount", "turnover_rate", "float_market_cap",
            "buy1_amount", "buy2_amount", "buy3_amount", "buy4_amount", "buy5_amount",
            "sell1_amount", "sell2_amount", "sell3_amount", "sell4_amount", "sell5_amount",
            "lhb_buy_total", "lhb_sell_total", "lhb_net_buy", "lhb_turnover",
        ]
        df = pd.DataFrame(columns=cols)
        result = compute_lhb_factors(df)

        assert len(result) == 0
        assert "buy_sell_ratio" in result.columns
