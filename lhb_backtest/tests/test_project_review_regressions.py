"""Counterexamples found during the complete project review."""
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.backtest.filter_engine import apply_filters, run_lhb_backtest
from src.backtest.grid_search import grid_search, format_grid_search_summary
from src.backtest.statistics import compute_rolling_period_statistics
from src.features.build_lhb_features import _pivot_broker_detail
from src.labels.generate_future_labels import generate_future_labels


@pytest.fixture
def bars(monkeypatch):
    from src.utils import calendar
    dates = pd.bdate_range("2024-01-08", periods=7).strftime("%Y-%m-%d").tolist()
    monkeypatch.setattr(calendar, "_trading_dates_cache", dates)
    k = pd.DataFrame({"stock_code": "000001.SZ", "trade_date": dates,
                      "open": 10., "close": 10., "high": 10., "low": 10.,
                      "high_limit": 11., "low_limit": 9., "volume": 100., "pct_chg": 0.})
    return k


def label(k, events=None, **kwargs):
    return generate_future_labels(event_df=k.iloc[[0]][["stock_code", "trade_date"]]
                                  if events is None else events,
                                  kline_df=k, include_alpha=False, include_suspension=False,
                                  horizons=[1, 2, 3, 5], **kwargs)


def test_gap_does_not_hide_valid_later_market_date(bars):
    bars.loc[2, "close"] = 12.
    out = label(bars.drop(index=1))
    assert pd.isna(out.future_return_1d.iloc[0])
    assert out.future_return_2d.iloc[0] == pytest.approx(.2)
    assert pd.isna(out.future_max_return_2d.iloc[0])
    assert pd.isna(out.entry_open_exit_open_1d.iloc[0])


def test_first_board_requires_previous_market_day(bars):
    bars.loc[2, ["close", "high_limit"]] = 11.
    out = label(bars.drop(index=1), bars.iloc[[2]][["stock_code", "trade_date"]])
    assert pd.isna(out.is_first_limit_up_board.iloc[0])


def test_unfinished_board_streak_is_censored(bars):
    bars.loc[1, "high_limit"] = 10.
    out = label(bars.iloc[:2])
    assert out.observed_continue_board_days.iloc[0] == 1
    assert out.continue_board_censored.iloc[0]
    assert pd.isna(out.max_continue_board_days.iloc[0])


def test_empty_events_have_label_schema(bars):
    out = label(bars, bars.iloc[:0][["stock_code", "trade_date"]])
    assert out.empty and "label_end_date_2d" in out and "future_return_5d" in out


def test_missing_all_quotes_keeps_unknown_labels(bars):
    out = label(bars.iloc[:0], bars.iloc[[0]][["stock_code", "trade_date"]])
    assert pd.isna(out.future_return_1d.iloc[0])


def test_duplicate_dates_after_normalization_rejected(bars):
    bars = pd.concat([bars, bars.iloc[[0]].assign(trade_date="20240108")], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        label(bars)


def test_board_filter_unknown_and_bse_prefixes():
    frame = pd.DataFrame({"stock_code": ["430001.BJ", "830001.BJ", "920001.BJ", "688001.SH", "689009.SH"]})
    assert len(apply_filters(frame, {"board_type": "bse"})) == 3
    assert apply_filters(frame, {"board_type": "star"}).stock_code.tolist() == ["688001.SH"]
    with pytest.raises(ValueError, match="Unknown board"):
        apply_filters(frame, {"board_type": "starr"})
    with pytest.raises(ValueError, match="requires"):
        apply_filters(pd.DataFrame({"x": [1]}), {"board_type": "star"})
    assert apply_filters(frame, {"board_type": []}).empty


def test_reason_filter_is_literal_and_none_is_disabled():
    frame = pd.DataFrame({"lhb_reason": ["涨幅(7%)", "涨幅7%"], "buy_sell_ratio": [1., 2.]})
    assert len(apply_filters(frame, {"lhb_reason": "(7%)", "min_buy_sell_ratio": None})) == 1


def test_date_boundary_preserves_consecutive_history(bars):
    frame = bars.iloc[:2].assign(entry_open_exit_open_1d=.1)
    out = run_lhb_backtest(event_df=frame, start_date=frame.trade_date.iloc[1],
                           filters={"exclude_consecutive_lhb_day": True})
    assert out.sample_count == 0


def test_multi_month_statistics_partition_instead_of_overlap():
    frame = pd.DataFrame({"trade_date": ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15"],
                          "future_return_1d": [.1, .2, .3, .4]})
    out = compute_rolling_period_statistics(frame)["all_periods"]
    assert out.sample_count.tolist() == [3, 1]
    assert out.period_start.tolist() == ["2024-01-01", "2024-04-01"]
    assert out.period_end.iloc[0] == "2024-03-31"


def test_grid_counts_observed_objective_not_all_events():
    frame = pd.DataFrame({"net_buy_ratio": [.1] * 10, "entry_oo_alpha_1d": [1.] + [np.nan] * 9})
    out = grid_search({}, event_df=frame, sort_by="net_oo_alpha_1d", min_sample_count=5,
                      generate_plots=False)
    assert out.ranking_sample_count.iloc[0] == 1
    assert not out.ranking_eligible.iloc[0]
    assert "Best:" not in format_grid_search_summary(out, [], "net_oo_alpha_1d")


def test_grid_temporal_split_requires_dates():
    with pytest.raises(ValueError, match="trade_date"):
        grid_search({}, event_df=pd.DataFrame({"entry_open_exit_open_1d": [.1]}),
                    oos_split_date="2024-07-01", generate_plots=False)


def test_pivot_preserves_anonymous_seats_and_zero_rank_one(tmp_path):
    frame = pd.DataFrame({"stock_code": "600000.SH", "trade_date": "2024-01-08",
                          "direction": ["buy"] * 3, "rank": [3, 1, 2],
                          "broker_name": ["机构专用"] * 3, "buy_amount": [10., 0., 10.]})
    path = tmp_path / "seats.parquet"
    frame.to_parquet(path)
    out = _pivot_broker_detail(str(path))
    assert out.buy1_amount.iloc[0] == 0
    assert out.buy2_amount.iloc[0] == out.buy3_amount.iloc[0] == 10
    assert out.buy1_concentration.iloc[0] == 0
    frame.assign(report_reason=["日涨幅", "三日涨幅", "日涨幅"]).to_parquet(path)
    with pytest.raises(ValueError, match="Mixed"):
        _pivot_broker_detail(str(path))


@pytest.fixture
def live(monkeypatch):
    module = importlib.import_module("src.ingestion.fetch_daily_kline")
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return module


def quote(code="600000.SH", day="2024-01-08"):
    return pd.DataFrame({"stock_code": [code], "trade_date": [day], "open": [10.],
                         "close": [10.], "high": [10.], "low": [10.],
                         "volume": [100.], "amount": [1000.]})


def test_live_failure_never_publishes_partial_batch(live, monkeypatch, tmp_path):
    path = tmp_path / "quotes.parquet"
    quote().assign(data_source="baostock", quote_contract="raw-shares-cny-v1").to_parquet(path)
    before = path.read_bytes()
    client = SimpleNamespace(fetch_daily_kline=lambda code, *args: quote() if code == "600000.SH" else pd.DataFrame())
    monkeypatch.setattr(live, "_get_client", lambda _: client)
    with pytest.raises(RuntimeError, match="Incomplete"):
        live.fetch_daily_kline("2024-01-08", "2024-01-08", ["600000.SH", "000001.SZ"],
                               source="baostock", save_path=path)
    assert path.read_bytes() == before
    assert path.with_suffix(".update.json").exists()


def test_live_merge_preserves_history_and_dotted_code(live, monkeypatch, tmp_path):
    path = tmp_path / "quotes.parquet"
    received = []
    def fetch(code, start, end):
        received.append(code)
        return quote(code, start)
    monkeypatch.setattr(live, "_get_client", lambda _: SimpleNamespace(fetch_daily_kline=fetch))
    for day in ["2024-01-08", "2024-01-09"]:
        live.fetch_daily_kline(day, day, ["600000.SH"], source="tushare", save_path=path)
    assert received == ["600000.SH", "600000.SH"]
    assert len(pd.read_parquet(path)) == 2


def test_live_refuses_unverified_or_mixed_source(live, tmp_path):
    path = tmp_path / "quotes.parquet"
    quote().to_parquet(path)
    with pytest.raises(ValueError, match="contract"):
        live.fetch_daily_kline("2024-01-08", "2024-01-08", [], save_path=path)


def test_akshare_default_raw_and_volume_shares(monkeypatch):
    import src.data_sources.akshare_client as module
    calls = []
    def fetch(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"日期": ["2024-01-08"], "成交量": [12.], "成交额": [1000.]})
    monkeypatch.setattr(module.ak, "stock_zh_a_hist", fetch)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    out = module.AKShareClient().fetch_daily_kline("600000", "2024-01-08", "2024-01-08")
    assert calls[0]["adjust"] == ""
    assert out.volume.iloc[0] == 1200
    assert out.amount.iloc[0] == 1000


def test_baostock_raw_keeps_preclose(monkeypatch):
    import src.data_sources.baostock_client as module
    calls = []
    client = module.BaostockClient()
    client._logged_in = True
    steps = iter([True, False])
    def fetch(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(error_code="0", fields=module._KLINE_FIELDS.split(","),
                               next=lambda: next(steps),
                               get_data=lambda: pd.DataFrame([["2024-01-08", "sh.600000", "10", "10", "10", "10", "9", "100", "1000", "1", "11.11"]], columns=module._KLINE_FIELDS.split(",")),
                               get_row_data=lambda: ["2024-01-08", "sh.600000", "10", "10", "10", "10", "9", "100", "1000", "1", "11.11"])
    monkeypatch.setattr(module.bs, "query_history_k_data_plus", fetch)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    out = client.fetch_daily_kline("600000.SH", "2024-01-08", "2024-01-08")
    assert calls[0]["adjustflag"] == "3"
    assert out.pre_close.iloc[0] == 9 and out.volume.iloc[0] == 100


def test_tushare_units_and_preclose(monkeypatch):
    import src.data_sources.tushare_client as module
    client = object.__new__(module.TushareClient)
    client._api = SimpleNamespace(daily=lambda **kwargs: pd.DataFrame({
        "ts_code": ["600000.SH"], "trade_date": ["20240108"], "vol": [12.],
        "amount": [5.], "pre_close": [9.]}))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    out = client.fetch_daily_kline("600000.SH", "2024-01-08", "2024-01-08")
    assert out.volume.iloc[0] == 1200 and out.amount.iloc[0] == 5000
    assert out.pre_close.iloc[0] == 9


def test_partial_summary_is_not_published_or_resolved(tmp_path, monkeypatch):
    module = importlib.import_module("src.ingestion.fetch_lhb_summary")
    path = tmp_path / "summary.parquet"
    quote().to_parquet(path)
    before = path.read_bytes()
    monkeypatch.setattr(module, "get_trading_dates", lambda *a: ["2024-01-08", "2024-01-09", "2024-01-10"])
    monkeypatch.setattr(module, "_load_failures_dir", lambda: tmp_path / "failures")
    monkeypatch.setattr(module, "_get_client", lambda _: SimpleNamespace(fetch_lhb_summary=lambda *a: quote(day="2024-01-09")))
    monkeypatch.setattr(module, "with_retry", lambda fn, *args: fn(*args))
    out = module.fetch_lhb_summary("2024-01-08", "2024-01-10", save_path=str(path))
    assert out.attrs["update_report"]["status"] == "fail"
    assert path.read_bytes() == before
    assert not module.summary_tracker(tmp_path / "failures").get_pending().empty


def test_cli_fetch_respects_config_and_stops_on_failed_summary(tmp_path, monkeypatch):
    import main
    captures = {}
    def response(name, status):
        def call(**kwargs):
            captures[name] = kwargs
            frame = pd.DataFrame()
            frame.attrs["update_report"] = {"status": status}
            return frame
        return call
    monkeypatch.setattr("src.ingestion.load_kline_simtradedata.load_kline_simtradedata", response("quotes", "pass"))
    monkeypatch.setattr("src.ingestion.fetch_lhb_summary.fetch_lhb_summary", response("summary", "fail"))
    monkeypatch.setattr("src.ingestion.fetch_lhb_detail.fetch_lhb_detail", response("detail", "pass"))
    config = {"paths": {"raw_dir": str(tmp_path)}, "simtradedata": {"export_dir": str(tmp_path / "vendor")}}
    with pytest.raises(RuntimeError, match="LHB summary"):
        main.step_fetch_data(config, "2024-01-08", "2024-01-09")
    assert captures["quotes"]["save_path"] == str(tmp_path / "daily_kline.parquet")
    assert captures["quotes"]["export_dir"] == str(tmp_path / "vendor")
    assert "detail" not in captures
    monkeypatch.setattr("src.ingestion.fetch_lhb_summary.restore_lhb_summary", response("restore_summary", "pass"))
    monkeypatch.setattr("src.ingestion.fetch_lhb_detail.restore_lhb_detail", response("restore_detail", "pass"))
    main.step_restore_all(config)
    assert captures["restore_summary"]["save_path"] == str(tmp_path / "lhb_summary.parquet")
    assert captures["restore_detail"]["save_path"] == str(tmp_path / "lhb_broker_detail.parquet")


def test_duckdb_path_with_quote_and_partitioned_parquet(tmp_path):
    from src.utils.io import save_parquet, query_parquet, load_parquet
    path = tmp_path / "owner's.parquet"
    save_parquet(pd.DataFrame({"x": [1, 2]}), path)
    assert query_parquet("select sum(x) AS total from data", path).total.iloc[0] == 3
    path = tmp_path / "partitioned"
    save_parquet(pd.DataFrame({"year": [2024, 2025], "x": [1, 2]}), path, partition_cols=["year"])
    assert len(load_parquet(path)) == 2


def test_sell_only_disclosure_counts_its_institutional_seats():
    from src.features.compute_lhb_factors import compute_lhb_factors
    frame = pd.DataFrame({"stock_code": ["600000.SH"], "lhb_reason": ["日涨幅"],
                          "sell1_broker": ["机构专用"], "sell1_amount": [100.]})
    out = compute_lhb_factors(event_df=frame)
    assert out.inst_sell_count.iloc[0] == 1
    assert pd.isna(out.inst_buy_count.iloc[0]) and pd.isna(out.inst_net_seats.iloc[0])


def test_html_escapes_source_text(tmp_path):
    from src.reports.make_report import generate_html_report
    result = run_lhb_backtest(event_df=pd.DataFrame())
    result.filters = {"reason": '<script>alert("source")</script>'}
    path = generate_html_report(result, output_dir=str(tmp_path), title="<script>bad</script>")
    from pathlib import Path
    html = Path(path).read_text(encoding="utf-8")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "未模拟跌停或停牌退出延期" in html
