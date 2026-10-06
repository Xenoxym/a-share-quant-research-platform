"""Economic counterexamples: these must not pass as executable profits."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest.filter_engine import apply_filters
from src.backtest.grid_search import grid_search
from src.cleaning.price_limits import validate_price_limits
from src.data_sources.snapshot_quality import require_snapshot_quality
from src.labels.generate_future_labels import generate_future_labels


@pytest.fixture
def market(monkeypatch):
    from src.utils import calendar
    from src.data_sources import simtradedata_metadata as meta
    dates = pd.bdate_range("2024-01-08", periods=7).strftime("%Y-%m-%d").tolist()
    monkeypatch.setattr(calendar, "_trading_dates_cache", dates)
    monkeypatch.setattr(meta, "load_status_lookup", lambda kind: {d: set() for d in dates})
    monkeypatch.setattr(meta, "load_benchmark", lambda: pd.DataFrame({
        "trade_date": dates, "open": [100, 105, 105, 105, 105, 105, 105],
        "close": [100, 105, 105, 105, 105, 105, 105]}))
    k = pd.DataFrame({"stock_code": "000001.SZ", "trade_date": dates,
                      "open": [10, 11, 11, 11, 11, 11, 11],
                      "close": [10, 11, 11, 11, 11, 11, 11],
                      "high": [10, 11, 11, 11, 11, 11, 11],
                      "low": [10, 11, 11, 11, 11, 11, 11],
                      "pct_chg": [0, 10, 0, 0, 0, 0, 0],
                      "volume": 1000, "high_limit": 12.1, "low_limit": 9.9})
    e = pd.DataFrame({"stock_code": ["000001.SZ"], "trade_date": [dates[0]], "pct_chg": [0]})
    return e, k


def test_pre_entry_gap_is_not_strategy_alpha(market):
    e, k = market
    r = generate_future_labels(event_df=e, kline_df=k, horizons=[5])
    assert r.alpha_5d.iloc[0] == pytest.approx(.05)  # +10% stock -5% index
    assert r.entry_alpha_5d.iloc[0] == pytest.approx(0)
    assert r.entry_oo_alpha_1d.iloc[0] == pytest.approx(0)


def test_horizon_one_still_has_legal_overnight_label(market):
    e, k = market
    r = generate_future_labels(event_df=e, kline_df=k, horizons=[1], include_alpha=False)
    assert r.entry_open_exit_open_1d.iloc[0] == 0
    assert r.label_end_date_2d.iloc[0] == k.trade_date.iloc[2]


def test_missing_benchmark_entry_day_is_not_replaced(market, monkeypatch):
    from src.data_sources import simtradedata_metadata as meta
    e, k = market
    benchmark = meta.load_benchmark().drop(index=1)
    monkeypatch.setattr(meta, "load_benchmark", lambda: benchmark)
    r = generate_future_labels(event_df=e, kline_df=k, horizons=[5])
    assert pd.isna(r.entry_alpha_5d.iloc[0])
    assert pd.isna(r.entry_oo_alpha_1d.iloc[0])


def test_zero_volume_cannot_enter(market):
    e, k = market
    k.loc[1, "volume"] = 0
    r = generate_future_labels(event_df=e, kline_df=k, horizons=[1], include_alpha=False)
    assert r.next_day_untradable.iloc[0] == 1


def test_unknown_halt_is_not_false(market, monkeypatch):
    e, k = market
    monkeypatch.setattr("src.data_sources.simtradedata_metadata.load_status_lookup", lambda kind: {})
    r = generate_future_labels(event_df=e, kline_df=k, horizons=[1], include_alpha=False)
    assert pd.isna(r.suspended_next_day.iloc[0])
    assert pd.isna(r.next_day_untradable.iloc[0])


def test_missing_market_day_cannot_shift_to_later_quote(market):
    e, k = market
    r = generate_future_labels(event_df=e, kline_df=k.drop(index=1), horizons=[1], include_alpha=False)
    assert pd.isna(r.future_return_1d.iloc[0])
    assert r.next_day_untradable.iloc[0] == 1


def test_incomplete_extreme_label_is_nan(market):
    e, k = market
    r = generate_future_labels(event_df=e, kline_df=k.iloc[:3], horizons=[5], include_alpha=False)
    assert pd.isna(r.future_max_return_5d.iloc[0])
    assert pd.isna(r.future_max_drawdown_5d.iloc[0])


def test_no_percentage_fallback(market):
    e, k = market
    k = k.drop(columns=["high_limit"])
    r = generate_future_labels(event_df=e, kline_df=k, horizons=[1], include_alpha=False)
    assert pd.isna(r.next_day_limit_up.iloc[0])


def test_bad_star_source_boundary_is_quarantined():
    df = pd.DataFrame({"stock_code": ["688011.SH"], "trade_date": ["2019-09-02"],
                       "pre_close": [65.0], "high_limit": [71.5], "low_limit": [58.5],
                       "high": [72.78], "low": [65]})
    meta = pd.DataFrame({"symbol": ["688011.SS"], "listed_date": ["2019-07-22"]})
    out = validate_price_limits(df, meta, {"2019-09-02": set()})
    assert out.source_high_limit.iloc[0] == 71.5
    assert pd.isna(out.high_limit.iloc[0])
    assert out.limit_price_status.iloc[0] == "source_rule_mismatch"


def test_half_cent_rounding_and_unknown_st():
    df = pd.DataFrame({"stock_code": ["000001.SZ"] * 2, "trade_date": ["2024-01-08", "2024-01-09"],
                       "pre_close": [10.05]*2, "high_limit": [11.06]*2,
                       "low_limit": [9.05]*2, "high": [10.5]*2, "low": [10]*2})
    meta = pd.DataFrame({"symbol": ["000001.SZ"], "listed_date": ["1991-04-03"]})
    out = validate_price_limits(df, meta, {"2024-01-08": set()})
    assert out.limit_price_valid.tolist() == [True, False]
    assert pd.isna(out.is_st.iloc[1])


def test_filters_fail_on_missing_or_misspelled_column():
    with pytest.raises(ValueError, match="requires column"):
        apply_filters(pd.DataFrame({"stock_code": ["000001.SZ"]}), {"min_buy1_concentration": .9})
    with pytest.raises(ValueError, match="Unknown filter"):
        apply_filters(pd.DataFrame({"x": [1]}), {"min_buy1_concentraton": .9})


def test_unknown_st_and_entry_are_excluded():
    df = pd.DataFrame({"is_st": [0., np.nan, 1.], "next_day_untradable": [np.nan, 0., 0.]})
    assert apply_filters(df, {"exclude_ST": True, "exclude_untradable_next_day": True}).empty


def test_grid_purges_crossing_labels_and_has_no_default_file_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    def unexpected_write(*args, **kwargs):
        pytest.fail("save_path=None must not invoke the CSV writer")
    import importlib
    monkeypatch.setattr(importlib.import_module("src.backtest.grid_search"), "save_csv", unexpected_write)
    df = pd.DataFrame({"trade_date": ["2024-06-20", "2024-06-28", "2024-07-02"],
                       "label_end_date_2d": ["2024-06-24", "2024-07-02", "2024-07-04"],
                       "net_buy_ratio": [.1]*3, "entry_open_exit_open_1d": [.01, 10., .02]})
    out = grid_search({"min_net_buy_ratio": [0.]}, event_df=df, horizons=[1],
                      oos_split_date="2024-07-01", generate_plots=False, min_sample_count=1)
    assert out.sample_count.iloc[0] == 1
    assert out.net_oo_return_1d.iloc[0] == pytest.approx(.01)
    assert out.oos_sample_count.iloc[0] == 1
    assert not list(tmp_path.rglob("*.csv"))


@pytest.mark.parametrize("target", ["oos_net_oo_return_1d", "net_entry_return_1d"])
def test_grid_refuses_invalid_objectives(target):
    with pytest.raises(ValueError):
        grid_search({}, event_df=pd.DataFrame(), sort_by=target, generate_plots=False)


def test_publication_gate_refuses_stale_status(tmp_path):
    meta = tmp_path / "metadata"
    meta.mkdir()
    for name in ("trade_days", "benchmark"):
        pd.DataFrame({"date": ["2024-01-08", "2024-01-09"]}).to_parquet(meta/f"{name}.parquet")
    pd.DataFrame({"date": ["20240108"]*2, "status_type": ["ST", "HALT"],
                  "symbols": [[], []]}).to_parquet(meta/"stock_status.parquet")
    for name in ("valuation", "exrights"):
        (tmp_path/name).mkdir()
        pd.DataFrame({"date": ["2024-01-09"]}).to_parquet(tmp_path/name/"000001.SZ.parquet")
    with pytest.raises(RuntimeError, match="ST status unknown"):
        require_snapshot_quality(tmp_path, "2024-01-09", "2024-01-08")
    assert (tmp_path/"lhb_quality_report.json").exists()


@pytest.mark.parametrize("name", ["fetch_lhb_summary", "fetch_lhb_detail"])
def test_restore_without_failures_returns_existing_frame(name, monkeypatch, tmp_path):
    import importlib
    module = importlib.import_module(f"src.ingestion.{name}")
    existing = pd.DataFrame({"stock_code": ["000001.SZ"]})
    monkeypatch.setattr(module, "_load_failures_dir", lambda: tmp_path)
    monkeypatch.setattr(module, "_load_existing", lambda path: existing)
    restore = getattr(module, name.replace("fetch_", "restore_"))
    assert restore() is existing


def test_atomic_parquet_failure_preserves_old_file(tmp_path, monkeypatch):
    from src.utils import io
    destination = tmp_path / "result.parquet"
    io.save_parquet(pd.DataFrame({"x": [1]}), destination)
    before = destination.read_bytes()
    def fail_write(table, path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise OSError("simulated disk failure")
    monkeypatch.setattr(io.pq, "write_table", fail_write)
    with pytest.raises(OSError, match="simulated"):
        io.save_parquet(pd.DataFrame({"x": [2]}), destination)
    assert destination.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
