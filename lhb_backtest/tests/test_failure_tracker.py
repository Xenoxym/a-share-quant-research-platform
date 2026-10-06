"""Unit tests for failure_tracker: with_retry and FailureTracker.

All tests are offline and side-effect-free (tmp_path fixture for CSV files).
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from src.ingestion.failure_tracker import (
    FailureTracker,
    with_retry,
    summary_tracker,
    detail_tracker,
    DEFAULT_WAITS,
)


# ---------------------------------------------------------------------------
# with_retry
# ---------------------------------------------------------------------------

class TestWithRetry:
    def test_success_on_first_attempt(self):
        fn = MagicMock(return_value=42)
        result = with_retry(fn, "a", b=1)
        assert result == 42
        fn.assert_called_once_with("a", b=1)

    def test_success_on_second_attempt(self):
        fn = MagicMock(side_effect=[RuntimeError("fail"), "ok"])
        result = with_retry(fn, max_retries=3, waits=(0, 0, 0))
        assert result == "ok"
        assert fn.call_count == 2

    def test_success_on_third_attempt(self):
        fn = MagicMock(side_effect=[ValueError("a"), ValueError("b"), "done"])
        result = with_retry(fn, max_retries=3, waits=(0, 0))
        assert result == "done"
        assert fn.call_count == 3

    def test_raises_after_all_attempts_exhausted(self):
        fn = MagicMock(side_effect=ConnectionError("timeout"))
        with pytest.raises(ConnectionError, match="timeout"):
            with_retry(fn, max_retries=3, waits=(0, 0))
        assert fn.call_count == 3

    def test_waits_are_respected(self):
        """Verify sleep is called with correct durations."""
        fn = MagicMock(side_effect=[RuntimeError("1"), RuntimeError("2"), "ok"])
        with patch("src.ingestion.failure_tracker.time.sleep") as mock_sleep:
            with_retry(fn, max_retries=3, waits=(1.0, 2.0))
        assert mock_sleep.call_args_list == [call(1.0), call(2.0)]

    def test_no_sleep_on_first_success(self):
        fn = MagicMock(return_value="ok")
        with patch("src.ingestion.failure_tracker.time.sleep") as mock_sleep:
            with_retry(fn, max_retries=3, waits=(1, 2))
        mock_sleep.assert_not_called()

    def test_no_sleep_after_final_failure(self):
        """Sleep should NOT be called after the last (exhausting) attempt."""
        fn = MagicMock(side_effect=RuntimeError("boom"))
        sleep_calls: list[float] = []
        with patch("src.ingestion.failure_tracker.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            with pytest.raises(RuntimeError):
                with_retry(fn, max_retries=3, waits=(1.0, 2.0))
        # Only 2 sleeps for 3 attempts: before attempt 2 and before attempt 3
        assert sleep_calls == [1.0, 2.0]

    def test_max_retries_one_means_no_retry(self):
        fn = MagicMock(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError):
            with_retry(fn, max_retries=1, waits=(0,))
        assert fn.call_count == 1

    def test_kwargs_forwarded(self):
        fn = MagicMock(return_value="result")
        with_retry(fn, "pos", key="value", max_retries=1, waits=())
        fn.assert_called_once_with("pos", key="value")

    def test_return_value_preserved_through_retries(self):
        fn = MagicMock(side_effect=[Exception("x"), {"data": [1, 2, 3]}])
        result = with_retry(fn, max_retries=3, waits=(0,))
        assert result == {"data": [1, 2, 3]}


# ---------------------------------------------------------------------------
# FailureTracker
# ---------------------------------------------------------------------------

class TestFailureTracker:
    @pytest.fixture()
    def summary_csv(self, tmp_path: Path) -> Path:
        return tmp_path / "failures_lhb_summary.csv"

    @pytest.fixture()
    def detail_csv(self, tmp_path: Path) -> Path:
        return tmp_path / "failures_lhb_detail.csv"

    @pytest.fixture()
    def summary_ft(self, summary_csv: Path) -> FailureTracker:
        return FailureTracker(summary_csv, primary_keys=["trade_date"])

    @pytest.fixture()
    def detail_ft(self, detail_csv: Path) -> FailureTracker:
        return FailureTracker(detail_csv, primary_keys=["trade_date", "stock_code"])

    # --- record ---

    def test_record_creates_file(self, summary_ft: FailureTracker, summary_csv: Path):
        summary_ft.record({"trade_date": "2024-01-02"}, "timeout")
        assert summary_csv.exists()

    def test_record_writes_status_failed(self, summary_ft: FailureTracker):
        summary_ft.record({"trade_date": "2024-01-02"}, "err")
        df = pd.read_csv(summary_ft._path)
        assert df.iloc[0]["status"] == "failed"

    def test_record_increments_retry_count(self, summary_ft: FailureTracker):
        key = {"trade_date": "2024-01-02"}
        summary_ft.record(key, "first")
        summary_ft.record(key, "second")
        df = pd.read_csv(summary_ft._path)
        assert int(df.iloc[0]["retry_count"]) == 2

    def test_record_deduplicates_by_primary_key(self, summary_ft: FailureTracker):
        key = {"trade_date": "2024-01-03"}
        for _ in range(5):
            summary_ft.record(key, "x")
        df = pd.read_csv(summary_ft._path)
        assert len(df[df["trade_date"] == "2024-01-03"]) == 1

    def test_record_stores_error_message(self, summary_ft: FailureTracker):
        summary_ft.record({"trade_date": "2024-01-04"}, "connection refused")
        df = pd.read_csv(summary_ft._path)
        assert "connection refused" in df.iloc[0]["error_msg"]

    def test_record_detail_two_key_columns(self, detail_ft: FailureTracker):
        detail_ft.record({"trade_date": "2024-01-02", "stock_code": "000001.SZ"}, "err")
        df = pd.read_csv(detail_ft._path)
        assert len(df) == 1
        assert df.iloc[0]["stock_code"] == "000001.SZ"

    # --- resolve ---

    def test_resolve_sets_status_resolved(self, summary_ft: FailureTracker):
        key = {"trade_date": "2024-01-05"}
        summary_ft.record(key, "err")
        summary_ft.resolve(key)
        df = pd.read_csv(summary_ft._path)
        assert df.iloc[0]["status"] == "resolved"

    def test_resolve_nonexistent_key_is_noop(self, summary_ft: FailureTracker):
        summary_ft.record({"trade_date": "2024-01-06"}, "err")
        summary_ft.resolve({"trade_date": "9999-99-99"})  # no-op
        df = pd.read_csv(summary_ft._path)
        assert len(df) == 1
        assert df.iloc[0]["status"] == "failed"

    # --- get_pending ---

    def test_get_pending_returns_only_failed(self, summary_ft: FailureTracker):
        summary_ft.record({"trade_date": "2024-01-07"}, "e")
        summary_ft.record({"trade_date": "2024-01-08"}, "e")
        summary_ft.resolve({"trade_date": "2024-01-07"})
        pending = summary_ft.get_pending()
        assert len(pending) == 1
        assert pending.iloc[0]["trade_date"] == "2024-01-08"

    def test_get_pending_empty_when_no_file(self, summary_ft: FailureTracker):
        assert summary_ft.get_pending().empty

    def test_get_pending_empty_when_all_resolved(self, summary_ft: FailureTracker):
        key = {"trade_date": "2024-01-09"}
        summary_ft.record(key, "e")
        summary_ft.resolve(key)
        assert summary_ft.get_pending().empty

    # --- count_pending ---

    def test_count_pending_zero_initially(self, summary_ft: FailureTracker):
        assert summary_ft.count_pending() == 0

    def test_count_pending_increases_with_records(self, summary_ft: FailureTracker):
        summary_ft.record({"trade_date": "2024-01-10"}, "e")
        summary_ft.record({"trade_date": "2024-01-11"}, "e")
        assert summary_ft.count_pending() == 2

    def test_count_pending_decreases_after_resolve(self, summary_ft: FailureTracker):
        key = {"trade_date": "2024-01-12"}
        summary_ft.record(key, "e")
        assert summary_ft.count_pending() == 1
        summary_ft.resolve(key)
        assert summary_ft.count_pending() == 0

    # --- multi-record isolation ---

    def test_two_detail_records_with_same_date_different_codes(self, detail_ft: FailureTracker):
        detail_ft.record({"trade_date": "2024-01-02", "stock_code": "000001.SZ"}, "e")
        detail_ft.record({"trade_date": "2024-01-02", "stock_code": "600519.SH"}, "e")
        assert detail_ft.count_pending() == 2

    def test_resolve_one_does_not_affect_other(self, detail_ft: FailureTracker):
        k1 = {"trade_date": "2024-01-02", "stock_code": "000001.SZ"}
        k2 = {"trade_date": "2024-01-02", "stock_code": "600519.SH"}
        detail_ft.record(k1, "e")
        detail_ft.record(k2, "e")
        detail_ft.resolve(k1)
        pending = detail_ft.get_pending()
        assert len(pending) == 1
        assert pending.iloc[0]["stock_code"] == "600519.SH"


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

class TestConvenienceConstructors:
    def test_summary_tracker(self, tmp_path: Path):
        t = summary_tracker(tmp_path)
        assert t._pk == ["trade_date"]
        assert "failures_lhb_summary.csv" in str(t._path)

    def test_detail_tracker(self, tmp_path: Path):
        t = detail_tracker(tmp_path)
        assert t._pk == ["trade_date", "stock_code"]
        assert "failures_lhb_detail.csv" in str(t._path)
