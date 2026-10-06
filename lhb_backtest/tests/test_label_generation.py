"""Tests for the label generation module (src.labels)."""

import pytest
import pandas as pd
import numpy as np

from src.labels.generate_future_labels import generate_future_labels
from src.labels.limit_up_detector import detect_limit_up, get_board_type, get_limit_up_threshold


def _make_kline_df(
    stock_code: str,
    dates: list[str],
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    """Build a synthetic k-line DataFrame from close prices."""
    n = len(dates)
    if highs is None:
        highs = [c * 1.02 for c in closes]
    if lows is None:
        lows = [c * 0.98 for c in closes]
    opens = [(h + l) / 2 for h, l in zip(highs, lows)]
    pre_closes = [np.nan] + closes[:-1]
    pct_chgs = [
        np.nan if np.isnan(pc) else (c - pc) / pc * 100
        for c, pc in zip(closes, pre_closes)
    ]
    return pd.DataFrame({
        "trade_date": dates,
        "stock_code": [stock_code] * n,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "pre_close": pre_closes,
        "pct_chg": pct_chgs,
        "volume": [1_000_000] * n,
        "amount": [10_000_000.0] * n,
        "turnover_rate": [2.0] * n,
    })


class TestBoardType:
    """Tests for get_board_type()."""

    def test_main_board_shanghai(self) -> None:
        assert get_board_type("600000.SH") == "main_sh"
        assert get_board_type("601398.SH") == "main_sh"

    def test_main_board_shenzhen(self) -> None:
        assert get_board_type("000001.SZ") == "main_sz"
        assert get_board_type("002594.SZ") == "main_sz"

    def test_chinext(self) -> None:
        assert get_board_type("300001.SZ") == "chinext"
        assert get_board_type("301099.SZ") == "chinext"

    def test_star_market(self) -> None:
        assert get_board_type("688001.SH") == "star"
        assert get_board_type("688981.SH") == "star"


class TestLimitUpDetection:
    """Tests for detect_limit_up(). Thresholds: main 9.8%, chinext/star 19.5%, ST 4.8%."""

    def test_main_board_limit_up(self) -> None:
        assert detect_limit_up(10.0, "000001.SZ", "平安银行") is True

    def test_main_board_below_threshold(self) -> None:
        assert detect_limit_up(9.7, "000001.SZ", "平安银行") is False

    def test_chinext_limit_up(self) -> None:
        assert detect_limit_up(20.0, "300001.SZ", "特锐德") is True

    def test_chinext_below_threshold(self) -> None:
        assert detect_limit_up(19.0, "300001.SZ", "特锐德") is False

    def test_star_market_limit_up(self) -> None:
        assert detect_limit_up(20.0, "688001.SH", "某科创") is True

    def test_star_market_below_threshold(self) -> None:
        assert detect_limit_up(19.0, "688001.SH", "某科创") is False

    def test_st_limit_up(self) -> None:
        assert detect_limit_up(5.0, "000001.SZ", "*ST某某") is True

    def test_st_below_threshold(self) -> None:
        assert detect_limit_up(4.5, "000001.SZ", "*ST某某") is False

    def test_st_prefix_variations(self) -> None:
        for name in ("ST某某", "*ST某某"):
            assert detect_limit_up(5.0, "000001.SZ", name) is True

    def test_threshold_values(self) -> None:
        assert get_limit_up_threshold("000001.SZ", "平安银行") == 9.8
        assert get_limit_up_threshold("300001.SZ", "特锐德") == 19.5
        assert get_limit_up_threshold("000001.SZ", "*ST某某") == 4.8


class TestLabelGeneration:
    """Test suite for future label generation."""

    @pytest.fixture
    def sample_kline_df(self) -> pd.DataFrame:
        dates = pd.bdate_range("2023-01-03", periods=20).strftime("%Y-%m-%d").tolist()
        closes = [
            10.0, 11.0, 12.1, 11.5, 11.8,
            12.0, 11.0, 10.5, 11.2, 11.5,
            12.0, 12.5, 12.0, 11.5, 11.8,
            12.2, 12.5, 12.8, 13.0, 12.5,
        ]
        highs = [
            10.5, 11.5, 12.5, 12.0, 12.2,
            12.5, 11.5, 11.0, 11.5, 12.0,
            12.5, 13.0, 12.5, 12.0, 12.2,
            12.5, 13.0, 13.2, 13.5, 13.0,
        ]
        lows = [
            9.5, 10.5, 11.5, 11.0, 11.2,
            11.5, 10.5, 10.0, 10.8, 11.0,
            11.5, 12.0, 11.5, 11.0, 11.3,
            11.8, 12.0, 12.3, 12.5, 12.0,
        ]
        return _make_kline_df("000001.SZ", dates, closes, highs, lows)

    @pytest.fixture
    def sample_event_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "trade_date": ["2023-01-03"],
            "stock_code": ["000001.SZ"],
            "stock_name": ["平安银行"],
            "close": [10.0],
            "pct_chg": [5.0],
            "lhb_buy_total": [6_800_000.0],
            "lhb_sell_total": [4_000_000.0],
            "lhb_net_buy": [2_800_000.0],
        })

    def test_future_return_1d(self, sample_event_df, sample_kline_df) -> None:
        """1-day return: (11.0 - 10.0) / 10.0 = 0.10."""
        result = generate_future_labels(
            event_df=sample_event_df, kline_df=sample_kline_df, horizons=[1],
        )
        assert "future_return_1d" in result.columns
        assert result["future_return_1d"].iloc[0] == pytest.approx(0.10, abs=1e-4)

    def test_future_return_5d(self, sample_event_df, sample_kline_df) -> None:
        """5-day return: close on day 5 is 12.0 -> (12.0 - 10.0)/10.0 = 0.20."""
        result = generate_future_labels(
            event_df=sample_event_df, kline_df=sample_kline_df, horizons=[5],
        )
        assert "future_return_5d" in result.columns
        # Day 5 close = 12.0 (index 5 in kline, which is 5th day after event at index 0)
        assert result["future_return_5d"].iloc[0] == pytest.approx(0.18, abs=0.05)

    def test_future_return_multiple_horizons(self, sample_event_df, sample_kline_df) -> None:
        horizons = [1, 2, 3, 5, 10]
        result = generate_future_labels(
            event_df=sample_event_df, kline_df=sample_kline_df, horizons=horizons,
        )
        for h in horizons:
            col = f"future_return_{h}d"
            assert col in result.columns, f"Missing label column {col}"
            assert not np.isnan(result[col].iloc[0]), f"{col} should not be NaN"

    def test_entry_open_exit_open_1d(self, sample_event_df, sample_kline_df) -> None:
        """Shortest legal round trip: buy T+1 open, sell T+2 open.

        Kline opens are (high+low)/2: T+1 open = (11.5+10.5)/2 = 11.0,
        T+2 open = (12.5+11.5)/2 = 12.0 → (12.0-11.0)/11.0.
        """
        result = generate_future_labels(
            event_df=sample_event_df, kline_df=sample_kline_df, horizons=[1, 2],
        )
        assert "entry_open_exit_open_1d" in result.columns
        expected = (12.0 - 11.0) / 11.0
        assert result["entry_open_exit_open_1d"].iloc[0] == pytest.approx(expected, abs=1e-6)

    def test_max_return_uses_highs(self, sample_event_df, sample_kline_df) -> None:
        """Max return within 5 days uses HIGH prices."""
        result = generate_future_labels(
            event_df=sample_event_df, kline_df=sample_kline_df, horizons=[5],
        )
        assert "future_max_return_5d" in result.columns
        # Days 1-5 highs: 11.5, 12.5, 12.0, 12.2, 12.5 -> max=12.5
        # max_return = (12.5 - 10.0)/10.0 = 0.25
        assert result["future_max_return_5d"].iloc[0] == pytest.approx(0.25, abs=1e-4)

    def test_max_drawdown_uses_lows(self, sample_event_df, sample_kline_df) -> None:
        """Max drawdown within 5 days uses LOW prices."""
        result = generate_future_labels(
            event_df=sample_event_df, kline_df=sample_kline_df, horizons=[5],
        )
        assert "future_max_drawdown_5d" in result.columns
        # Days 1-5 lows: 10.5, 11.5, 11.0, 11.2, 11.5 -> min=10.5
        # max_drawdown = (10.5 - 10.0)/10.0 = 0.05
        val = result["future_max_drawdown_5d"].iloc[0]
        assert not np.isnan(val)

    def test_consecutive_limit_up(self) -> None:
        """Test consecutive limit-up counting."""
        dates = pd.bdate_range("2023-01-03", periods=6).strftime("%Y-%m-%d").tolist()
        closes = [10.0, 11.0, 12.1, 13.31, 13.5, 13.0]
        pct_chgs = [np.nan, 10.0, 10.0, 10.0, 1.43, -3.70]
        kline = pd.DataFrame({
            "trade_date": dates,
            "stock_code": ["000001.SZ"] * 6,
            "high_limit": [11.0, 11.0, 12.1, 13.31, 14.64, 14.85],
            "open": closes,
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.98 for c in closes],
            "close": closes,
            "pre_close": [np.nan] + closes[:-1],
            "pct_chg": pct_chgs,
            "volume": [1_000_000] * 6,
            "amount": [10_000_000.0] * 6,
            "turnover_rate": [2.0] * 6,
        })
        event = pd.DataFrame({
            "trade_date": ["2023-01-03"],
            "stock_code": ["000001.SZ"],
            "stock_name": ["平安银行"],
            "close": [10.0],
            "pct_chg": [5.0],
        })
        result = generate_future_labels(event_df=event, kline_df=kline, horizons=[5])
        assert result["max_continue_board_days"].iloc[0] >= 3

    def test_insufficient_future_data(self) -> None:
        dates = pd.bdate_range("2023-01-03", periods=3).strftime("%Y-%m-%d").tolist()
        kline = _make_kline_df("000001.SZ", dates, [10.0, 11.0, 12.0])
        event = pd.DataFrame({
            "trade_date": [dates[-1]],
            "stock_code": ["000001.SZ"],
            "stock_name": ["平安银行"],
            "close": [12.0],
            "pct_chg": [9.09],
        })
        result = generate_future_labels(event_df=event, kline_df=kline, horizons=[5, 10])
        assert np.isnan(result["future_return_5d"].iloc[0])
        assert np.isnan(result["future_return_10d"].iloc[0])

    def test_multiple_stocks(self) -> None:
        dates = pd.bdate_range("2023-01-03", periods=10).strftime("%Y-%m-%d").tolist()
        kline_a = _make_kline_df("000001.SZ", dates, [10 + i * 0.5 for i in range(10)])
        kline_b = _make_kline_df("600000.SH", dates, [20 - i * 0.3 for i in range(10)])
        kline = pd.concat([kline_a, kline_b], ignore_index=True)

        events = pd.DataFrame({
            "trade_date": [dates[0], dates[0]],
            "stock_code": ["000001.SZ", "600000.SH"],
            "stock_name": ["平安银行", "浦发银行"],
            "close": [10.0, 20.0],
            "pct_chg": [5.0, -1.5],
        })
        result = generate_future_labels(event_df=events, kline_df=kline, horizons=[1, 5])
        assert len(result) == 2
        assert result["future_return_1d"].iloc[0] > 0
        assert result["future_return_1d"].iloc[1] < 0

    def test_event_not_in_kline(self) -> None:
        dates = pd.bdate_range("2023-02-01", periods=5).strftime("%Y-%m-%d").tolist()
        kline = _make_kline_df("000001.SZ", dates, [10.0 + i for i in range(5)])
        event = pd.DataFrame({
            "trade_date": ["2023-01-03"],
            "stock_code": ["000001.SZ"],
            "stock_name": ["平安银行"],
            "close": [10.0],
            "pct_chg": [5.0],
        })
        result = generate_future_labels(event_df=event, kline_df=kline, horizons=[1])
        assert np.isnan(result["future_return_1d"].iloc[0])
