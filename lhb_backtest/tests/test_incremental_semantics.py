"""Regression tests for incremental-update semantics.

Covers the 2026-07 incident where the updater claimed
"all trading dates already present" for a range the local snapshot did not
cover, silently swallowing a month of data.
"""

import numpy as np
import pandas as pd
import pytest

import src.utils.calendar as cal
from src.ingestion.load_kline_simtradedata import (
    _get_missing_trading_dates,
    load_kline_simtradedata,
)


@pytest.fixture
def small_calendar(monkeypatch):
    """Calendar snapshot that ends on 2026-05-26 (a Tuesday)."""
    dates = [
        "2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22",
        "2026-05-25", "2026-05-26",
    ]
    monkeypatch.setattr(cal, "_trading_dates_cache", dates)
    yield dates
    monkeypatch.setattr(cal, "_trading_dates_cache", None)


class TestCalendarExtension:
    def test_within_snapshot_unchanged(self, small_calendar):
        got = cal.get_trading_dates("2026-05-18", "2026-05-26")
        assert got == small_calendar

    def test_extends_beyond_snapshot_end(self, small_calendar):
        """Dates after the snapshot end must be reported as (approximate)
        trading days, NOT silently dropped — otherwise incremental updaters
        believe they are already up to date."""
        got = cal.get_trading_dates("2026-05-18", "2026-06-03")
        assert got[: len(small_calendar)] == small_calendar
        extension = got[len(small_calendar):]
        assert extension, "extension beyond snapshot must not be empty"
        assert extension[0] > "2026-05-26"
        assert "2026-06-01" in extension  # a Monday
        # weekends are never included
        assert all(pd.Timestamp(d).dayofweek < 5 for d in extension)

    def test_extension_excludes_known_holidays(self, small_calendar):
        got = cal.get_trading_dates("2026-06-15", "2026-06-22")
        assert "2026-06-19" not in got  # Dragon Boat Festival 2026

    def test_calendar_max_date(self, small_calendar):
        assert cal.calendar_max_date() == "2026-05-26"


class TestMissingDateDetection:
    def _existing(self, dates: list[str], n_stocks: int) -> pd.DataFrame:
        rows = [
            {"stock_code": f"{i:06d}.SZ", "trade_date": d}
            for d in dates for i in range(n_stocks)
        ]
        return pd.DataFrame(rows)

    def test_full_days_not_refetched(self, small_calendar):
        existing = self._existing(small_calendar, 100)
        missing = _get_missing_trading_dates("2026-05-18", "2026-05-26", existing)
        assert missing == []

    def test_absent_days_detected(self, small_calendar):
        existing = self._existing(small_calendar[:5], 100)  # missing 05-25/26
        missing = _get_missing_trading_dates("2026-05-18", "2026-05-26", existing)
        assert missing == ["2026-05-25", "2026-05-26"]

    def test_partial_days_refetched(self, small_calendar):
        """A day with far fewer stocks than the median must be treated as
        missing (partial source export), not as complete."""
        full = self._existing(small_calendar[:6], 100)
        partial = self._existing([small_calendar[6]], 3)  # 3 stocks vs 100
        existing = pd.concat([full, partial], ignore_index=True)
        missing = _get_missing_trading_dates("2026-05-18", "2026-05-26", existing)
        assert missing == ["2026-05-26"]

    def test_range_beyond_snapshot_reports_missing(self, small_calendar):
        """The incident scenario: data present through snapshot end, but the
        requested range extends past it — those dates must surface as missing."""
        existing = self._existing(small_calendar, 100)
        missing = _get_missing_trading_dates("2026-05-18", "2026-06-03", existing)
        assert missing, "dates beyond snapshot must be reported missing"
        assert all(d > "2026-05-26" for d in missing)


class TestUpdateReporting:
    def test_empty_source_reports_every_requested_date_unresolved(
        self, small_calendar, tmp_path, monkeypatch,
    ):
        existing = pd.DataFrame([
            {"stock_code": f"{i:06d}.SZ", "trade_date": date}
            for date in small_calendar for i in range(100)
        ])
        output = tmp_path / "daily_kline.parquet"
        existing.to_parquet(output, index=False)

        monkeypatch.setattr(
            "src.data_sources.simtradedata_loader.load_stocks_parquet",
            lambda **kwargs: pd.DataFrame(),
        )

        result = load_kline_simtradedata(
            "2026-05-18",
            "2026-06-03",
            export_dir=tmp_path,
            save_path=str(output),
            incremental=True,
        )
        report = result.attrs["update_report"]
        assert report["rows_added"] == 0
        assert report["requested_missing_dates"] > 0
        assert report["unresolved_dates"]
