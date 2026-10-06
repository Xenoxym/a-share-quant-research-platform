from dataclasses import replace
import json
import threading
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pytest

from src.technical.contracts import ResearchSpec, Strategy, Experiment, describe
from src.technical.foundation_data import attach_reports, normalize_filings
from src.technical.foundation_lab import plan
from src.technical.foundation_review import joint_block_intervals
from src.technical.signals import decisions, features
from src.technical.server import make_server
from tests.test_technical import bars, master


def reports():
    return pd.DataFrame([
        dict(stock_code=c, date=period, publ_date=pub, total_shares=shares,
             np_parent_company_owners=profit, total_shareholder_equity=5e8, total_assets=1e9)
        for c, period, pub, shares, profit in [
            ("600001.SH", "2019-12-31", "20200115", 1e8, 5e7),
            ("600002.SH", "2019-12-31", "20200115", 2e8, -1e7),
            ("600001.SH", "2020-03-31", "20200420", 3e8, 2e7),
        ]])


def test_filings_strict_publication_date_and_no_future_shares():
    f, _ = normalize_filings(reports())
    points = pd.DataFrame(dict(stock_code=["600001.SH"]*4,
        trade_date=["2020-01-14", "2020-01-15", "2020-01-16", "2020-04-21"], close=[10.]*4))
    a = attach_reports(points, f)
    assert a.estimated_market_cap.iloc[:2].isna().all()
    assert a.estimated_market_cap.iloc[2] == 1e9
    assert a.estimated_market_cap.iloc[3] == 3e9
    assert a.a_report_date.iloc[3] == "2019-12-31"
    changed = f.copy()
    changed.loc[changed.report_date.eq("2020-03-31"), "total_shares"] = 1e15
    pd.testing.assert_frame_equal(a.iloc[:3], attach_reports(points.iloc[:3], changed))


def test_late_old_report_does_not_replace_new_fiscal_period():
    raw = reports()
    raw.loc[len(raw)] = ["600001.SH", "2019-09-30", "20200501", 8e8, 9e7, 8e8, 2e9]
    f, _ = normalize_filings(raw)
    p = pd.DataFrame([dict(stock_code="600001.SH", trade_date="2020-05-04", close=10.)])
    out = attach_reports(p, f).iloc[0]
    assert out.q_report_date == "2020-03-31" and out.q_total_shares == 3e8


def test_share_bridge_uses_only_known_post_report_share_actions():
    f, _ = normalize_filings(reports())
    points=pd.DataFrame(dict(stock_code=['600001.SH']*4,
        trade_date=['2020-01-20','2020-02-20','2020-03-20','2020-04-21'],close=[10.]*4))
    actions=pd.DataFrame(dict(stock_code=['600001.SH']*4,
        trade_date=['2019-12-10','2020-02-01','2020-03-01','2020-05-01'],
        allotted_ps=[9.,.3,0.,10.], rationed_ps=[0.,0.,.1,0.],bonus_ps=[0.,0.,9.,0.]))
    a=attach_reports(points,f,actions,True)
    assert a.estimated_market_cap.iloc[0]==1e9  # pre-period action ignored
    assert a.estimated_market_cap.iloc[1]==1.3e9
    assert np.isnan(a.estimated_market_cap.iloc[2])  # rights subscriptions unknown
    assert a.estimated_market_cap.iloc[3]==3e9  # new quarter resets the bridge
    assert a.q_total_shares.iloc[1]==1e8  # original report stays intact
    changed=actions.copy();changed.loc[3,'allotted_ps']=1000.
    pd.testing.assert_frame_equal(a,attach_reports(points,f,changed,True))


def test_invalid_publication_and_duplicates_are_not_silently_used():
    raw = reports()
    raw.loc[0, "publ_date"] = "20190101"
    valid, bad = normalize_filings(raw)
    assert len(bad) == 1 and len(valid) == 2
    with pytest.raises(ValueError, match="重复"):
        normalize_filings(pd.concat([reports(), reports().iloc[:1]]))


def fixture(family="size", **kw):
    cal = pd.bdate_range("2020-01-01", "2020-03-10").strftime("%Y-%m-%d").tolist()
    spec = ResearchSpec(strategy=Strategy(family=family, lookback=20, skip=0, top_k=1,
        min_avg_amount=0, require_fundamentals=True, **kw), experiment=Experiment("2020-01-01", "2020-03-10"))
    quotes = bars(cal, ("600001.SH", "600002.SH"))
    f, _ = normalize_filings(reports())
    return cal, spec, features(quotes, cal, spec.strategy), f


def test_size_definition_and_opposite_control_use_past_reported_cap():
    cal, spec, b, f = fixture()
    d, _ = decisions(b, cal, master(["600001.SH", "600002.SH"]), spec, f)
    assert set(d.loc[d.selected, "stock_code"]) == {"600001.SH"}
    big, _ = decisions(b, cal, master(["600001.SH", "600002.SH"]), replace(spec, strategy=replace(spec.strategy, family="large_size")), f)
    assert set(big.loc[big.selected, "stock_code"]) == {"600002.SH"}
    assert "已披露" in describe(spec)["formula"]
    assert d.loc[d.selected, "q_publication_date"].lt(d.loc[d.selected, "trade_date"]).all()


def test_size_band_and_profit_filter_do_not_redefine_common_percentiles():
    cal, spec, b, f = fixture(size_floor_quantile=.5, positive_profit=True)
    d, _ = decisions(b, cal, master(["600001.SH", "600002.SH"]), spec, f)
    assert not d.selected.any()
    assert set(d.loc[d.stock_code.eq("600001.SH"), "reason"]) == {"size_band"}
    assert set(d.loc[d.stock_code.eq("600002.SH"), "reason"]) == {"nonpositive_annual_profit"}


def test_stale_share_reports_and_missing_financials_do_not_pass():
    cal, spec, b, f = fixture()
    f.loc[f.report_date.eq("2019-12-31"), "report_date"] = "2018-12-31"
    d, _ = decisions(b, cal, master(["600001.SH", "600002.SH"]), spec, f)
    assert not d.selected.any()
    assert "missing_or_stale_share_report" in set(d.reason)
    with pytest.raises(ValueError, match="历史财报"):
        decisions(b, cal, master(["600001.SH", "600002.SH"]), spec)


def test_negative_earnings_are_not_high_value_or_quality():
    for family in ["earnings_yield", "profitability"]:
        cal, spec, b, f = fixture(family)
        d, _ = decisions(b, cal, master(["600001.SH", "600002.SH"]), spec, f)
        assert set(d.loc[d.selected, "stock_code"]) == {"600001.SH"}
        assert set(d.loc[d.stock_code.eq("600002.SH"), "reason"]) == {"invalid_family_fundamental"}


def test_market_gate_uses_only_current_and_past_reference():
    cal, spec, b, f = fixture("market_trend", trend_window=5)
    bench = pd.DataFrame(dict(trade_date=cal, close=np.arange(len(cal))+100.))
    d, _ = decisions(b, cal, master(["600001.SH", "600002.SH"]), spec, f, bench)
    changed = bench.copy()
    changed.loc[changed.trade_date.gt("2020-01-31"), "close"] = 1
    d2, _ = decisions(b, cal, master(["600001.SH", "600002.SH"]), spec, f, changed)
    pd.testing.assert_frame_equal(d.loc[d.trade_date.le("2020-01-31")], d2.loc[d2.trade_date.le("2020-01-31")])
    assert not d2.loc[d2.trade_date.eq("2020-02-28"), "selected"].any()


def test_plan_contains_distinct_mechanisms_and_explicit_controls():
    p = plan()
    assert len(p["trials"]) == 15
    assert len({r["key"] for r in p["trials"]}) == 15
    assert sum(r["role"] == "candidate" for r in p["trials"]) == 10
    for r in p["trials"]:
        spec = ResearchSpec.from_dict(r["spec"])
        assert spec.strategy.require_fundamentals and spec.strategy.min_history == 252
        assert r["definition"]["formula"]


def test_joint_bootstrap_reproducible_paired_and_constant_difference():
    x = np.c_[np.full(200, .001), np.zeros(200), np.linspace(-.002, .002, 200)]
    a = joint_block_intervals(x, samples=100)
    assert a == joint_block_intervals(x, samples=100)
    assert a[0]["lower"][0] == pytest.approx(.252)
    assert a[0]["upper"][1] == 0
    assert a[0]["lower"][2] < 0 < a[0]["upper"][2]


def test_foundation_routes_do_not_shadow_run_folder_lookup(tmp_path):
    server = make_server(tmp_path, port=0, root=tmp_path/"technical")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        origin = f"http://127.0.0.1:{server.server_port}"
        assert len(json.load(urlopen(origin+"/api/foundations/plan"))["trials"]) == 15
        assert json.load(urlopen(origin+"/api/foundations")) == []
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as e:
            urlopen(origin+"/api/runs/20260101T000000-00000000")
        assert e.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
