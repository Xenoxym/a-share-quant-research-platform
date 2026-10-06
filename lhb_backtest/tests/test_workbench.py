"""Economic counterexamples and research-contract regression tests."""
from dataclasses import replace
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from src.workbench.artifacts import digest, verify_artifacts, write_json
from src.workbench.contracts import StrategyCard
from src.workbench.data import attach_features, institutional_features
from src.workbench.portfolio import Market, fees, simulate
from src.workbench.research import block_interval, select_cohorts


@pytest.fixture
def scenario():
    days = pd.bdate_range("2024-01-02", periods=35).strftime("%Y-%m-%d").tolist()
    rows = []
    for code in ["600001.SH", "000001.SZ", "600002.SH"]:
        for day in days:
            rows.append(dict(stock_code=code, trade_date=day, open=10., high=10.5, low=9.5,
                             close=10., pre_close=10., volume=1_000_000., amount=10_000_000.,
                             high_limit=11., low_limit=9., limit_price_valid=True, is_st=0.))
    bars = pd.DataFrame(rows)
    signals = pd.DataFrame([dict(stock_code="600001.SH", trade_date=days[20], entry_date=days[21],
                                 event_id=days[20]+"_600001.SH", close=10., avg_volume_20=1_000_000., inst_ratio=.1)])
    card = StrategyCard(start_date=days[20], end_date=days[-1], initial_cash=10000,
                        holding_days=1, max_positions=1, position_fraction=1., min_avg_amount=0,
                        commission_rate=0, minimum_commission=0, other_fee_rate=0,
                        slippage_bps=0, max_participation=.05)
    return days, bars, signals, card


def run_case(scenario, bars=None, signals=None, card=None, actions=None, status=None, metadata=None):
    days, base, sig, default = scenario
    return simulate(sig if signals is None else signals, Market(base if bars is None else bars, days, actions, status, metadata),
                    days[20:], default if card is None else card)


def test_cash_shares_and_t_plus_one(scenario):
    days, _, _, card = scenario
    r = run_case(scenario)
    assert list(r["fills"].side) == ["buy", "sell"]
    assert list(r["fills"].date) == [days[21], days[22]]
    assert r["fills"].iloc[0].filled_shares == 900
    assert r["metrics"]["final_equity"] == pytest.approx(card.initial_cash - 9000 * .0005)
    flows = r["ledger"].groupby("date").cash_flow.sum().reindex(r["equity"].date, fill_value=0).cumsum()
    assert np.allclose(r["equity"].cash, flows.to_numpy() + card.initial_cash)


def test_order_sized_before_open_not_using_future_price(scenario):
    days, bars, _, _ = scenario
    changed = bars.copy()
    changed.loc[changed.trade_date.eq(days[21]) & changed.stock_code.eq("600001.SH"), "open"] = 10.8
    first, second = run_case(scenario), run_case(scenario, bars=changed)
    assert first["orders"].iloc[0].requested_shares == second["orders"].iloc[0].requested_shares


def test_limit_buy_is_recorded_not_filtered(scenario):
    days, bars, _, _ = scenario
    bars.loc[bars.trade_date.eq(days[21]), "open"] = 11.
    result = run_case(scenario, bars=bars)
    assert result["metrics"]["buy_fills"] == 0
    assert result["orders"].iloc[0].reason == "open_at_upper_limit"
    assert result["metrics"]["final_equity"] == 10000


def test_limit_exit_retries_and_keeps_loss(scenario):
    days, bars, _, _ = scenario
    bars.loc[bars.trade_date.eq(days[22]), ["open", "close"]] = 9.
    bars.loc[bars.trade_date.ge(days[23]), ["open", "close"]] = 9.5
    r = run_case(scenario, bars=bars)
    assert r["orders"].iloc[1].status == "deferred"
    assert r["trades"].iloc[0].exit_date == days[23]
    assert r["trades"].iloc[0].pnl < 0


def test_no_fabricated_final_liquidation(scenario):
    days, _, _, card = scenario
    r = run_case(scenario, card=replace(card, holding_days=60))
    assert r["metrics"]["open_positions"] == 1
    assert len(r["trades"]) == 0
    assert r["metrics"]["final_equity"] == 10000


def test_missing_exit_quote_is_unknown_and_retained(scenario):
    days, bars, _, card = scenario
    bars = bars.loc[~(bars.stock_code.eq("600001.SH") & bars.trade_date.ge(days[22]))]
    r = run_case(scenario, bars=bars)
    assert r["metrics"]["open_positions"] == 1
    assert r["metrics"]["warnings"]["missing_mark"] > 0
    assert r["orders"].iloc[-1].reason == "missing_quote"
    assert r["metrics"]["unresolved_market_value"] == 9000
    assert r["metrics"]["return_zero_unresolved"] == pytest.approx(-.9)


def test_position_becoming_st_can_exit_under_dated_rule(scenario):
    days, bars, _, _ = scenario
    mask = bars.stock_code.eq("600001.SH") & bars.trade_date.ge(days[22])
    bars.loc[mask, "is_st"] = 1.
    bars.loc[mask, "limit_price_valid"] = False
    bars.loc[mask, ["high_limit", "low_limit"]] = np.nan
    r = run_case(scenario, bars=bars)
    assert r["metrics"]["closed_trades"] == 1
    assert r["fills"].iloc[-1].limit_origin == "dated_st_rule_model"
    assert r["metrics"]["warnings"]["dated_st_exit_model"] == 1


def test_st_rule_switch_and_special_session_rejection():
    from src.workbench.rules import st_exit_limits
    row = dict(is_st=1, pre_close=10., high=10.3, low=9.7)
    assert st_exit_limits("600001.SH", "2026-07-03", row) == (10.5, 9.5)
    assert st_exit_limits("600001.SH", "2026-07-06", row) == (11., 9.)
    assert st_exit_limits("600001.SH", "2026-07-03", {**row, "high": 15.}) is None
    assert st_exit_limits("600001.SH", "2027-01-01", row) is None


def test_unknown_non_st_exit_is_flagged(scenario):
    days, bars, _, _ = scenario
    bars.loc[bars.stock_code.eq("600001.SH") & bars.trade_date.ge(days[22]), "limit_price_valid"] = False
    r = run_case(scenario, bars=bars)
    assert r["metrics"]["open_positions"] == 1
    assert r["metrics"]["warnings"]["unresolved_exit_regime"] > 0


def test_dividend_entitlement_not_double_counted_or_reinvested(scenario):
    days, bars, _, _ = scenario
    bars.loc[bars.trade_date.ge(days[22]), ["open", "close", "pre_close"]] = 9.8
    actions = pd.DataFrame([dict(stock_code="600001.SH", trade_date=days[22], dividend=.2,
                                allotted_ps=0., rationed_ps=0., rationed_px=0.)])
    r = run_case(scenario, bars=bars, actions=actions)
    assert r["metrics"]["receivable"] == pytest.approx(900 * .2 * .8)
    assert r["metrics"]["final_equity"] == pytest.approx(10000 - 900 * .2 * .2 - 900 * 9.8 * .0005)
    assert r["metrics"]["accounting"]["pnl_error"] < 1e-6


def test_ex_date_buyer_has_no_dividend(scenario):
    days, _, _, _ = scenario
    actions = pd.DataFrame([dict(stock_code="600001.SH", trade_date=days[21], dividend=.2,
                                allotted_ps=0., rationed_ps=0., rationed_px=0.)])
    assert run_case(scenario, actions=actions)["metrics"]["receivable"] == 0


def test_bonus_shares_and_odd_lot_exit(scenario):
    days, bars, _, _ = scenario
    bars.loc[bars.trade_date.ge(days[22]), ["open", "close", "pre_close"]] = 10 / 1.13
    bars.loc[bars.trade_date.ge(days[22]), "low_limit"] = 7.
    actions = pd.DataFrame([dict(stock_code="600001.SH", trade_date=days[22], dividend=0.,
                                allotted_ps=.13, rationed_ps=0., rationed_px=0.)])
    r = run_case(scenario, bars=bars, actions=actions)
    assert r["fills"].iloc[-1].filled_shares == 1017
    assert r["metrics"]["final_equity"] == pytest.approx(10000 - 9000 * .0005)


def test_partial_exit_and_remaining_quantity(scenario):
    days, bars, _, _ = scenario
    bars.loc[bars.trade_date.eq(days[22]), "volume"] = 5000
    r = run_case(scenario, bars=bars)
    assert r["orders"].iloc[1].filled_shares == 200
    assert r["orders"].iloc[1].status == "partial"
    assert r["orders"].iloc[2].filled_shares == 700
    assert r["metrics"]["closed_trades"] == 1
    assert r["trades"].iloc[0].deferred_days == 1


def test_no_same_open_sale_proceeds_for_planned_buys(scenario):
    days, _, signals, _ = scenario
    second = signals.iloc[0].to_dict()
    second.update(stock_code="000001.SZ", trade_date=days[21], entry_date=days[22], event_id=days[21]+"_000001.SZ")
    r = run_case(scenario, signals=pd.concat([signals, pd.DataFrame([second])], ignore_index=True))
    assert r["orders"].loc[r["orders"].event_id.eq(second["event_id"])].iloc[0].reason == "position_limit"


def test_empty_signal_set_has_valid_flat_account(scenario):
    _, _, signals, _ = scenario
    r = run_case(scenario, signals=signals.iloc[:0])
    assert r["metrics"]["total_return"] == 0
    assert "date" in r["positions"] and "date" in r["orders"]


def test_future_signal_is_pending_without_fabricated_order_date(scenario):
    _, _, signals, _ = scenario
    signals = signals.copy()
    signals["entry_date"] = None
    r = run_case(scenario, signals=signals)
    assert r["orders"].iloc[0].status == "pending"
    assert pd.isna(r["orders"].iloc[0].date)


def test_stamp_tax_boundary_and_minimum_fee(scenario):
    card = replace(scenario[-1], minimum_commission=5, commission_rate=.0003)
    assert fees(1000, "sell", "2023-08-27", card)["stamp_tax"] == 1
    assert fees(1000, "sell", "2023-08-28", card)["stamp_tax"] == .5
    assert fees(1000, "buy", "2023-08-27", card)["stamp_tax"] == 0
    assert fees(1000, "buy", "2023-08-27", card)["commission"] == 5


@pytest.mark.parametrize("values", [{"unknown": 2}, {"holding_days": 0}, {"seed": True},
                                    {"initial_cash": float("nan")}, {"slippage_bps": -1},
                                    {"end_date": "2018-01-01"}, {"max_positions": 1.5}])
def test_bad_card_rejected(values):
    with pytest.raises(ValueError):
        StrategyCard.from_dict(values)


def test_rolling_features_exclude_future_and_require_market_contiguity(scenario):
    days, bars, _, _ = scenario
    events = pd.DataFrame([dict(stock_code="600001.SH", trade_date=days[20], summary_window_days=1,
                                seat_window_days=1, disclosure_kind="ordinary", seats_known=True, inst_disclosed_net=1000)])
    valuations = pd.DataFrame([dict(stock_code="600001.SH", trade_date=days[20], float_value=1e9)])
    first = attach_features(events, bars, valuations, days)
    bars.loc[bars.trade_date.gt(days[20]), "close"] = 100000
    second = attach_features(events, bars, valuations, days)
    pd.testing.assert_frame_equal(first, second)
    broken = bars.loc[~bars.trade_date.eq(days[10])]
    assert attach_features(events, broken, valuations, days).iloc[0].base_reason == "missing_20_sessions"


def test_anonymous_institutions_stay_separate():
    key = dict(stock_code="600001.SH", trade_date="2024-01-02")
    summary = pd.DataFrame([key])
    rows = []
    for side, amount in [("buy", 100.), ("sell", 40.)]:
        for rank in [1, 2]:
            rows.append({**key, "direction": side, "rank": rank, "broker_name": "机构专用",
                         "buy_amount": amount if side == "buy" else 0., "sell_amount": amount if side == "sell" else 0.,
                         "disclosure_kind": "ordinary", "window_days": 1, "report_reason": "单日榜"})
    assert institutional_features(summary, pd.DataFrame(rows)).iloc[0].inst_disclosed_net == 120


def test_matching_does_not_use_outcomes_and_does_not_reuse_controls():
    rows = []
    for i in range(8):
        rows.append(dict(event_id=f"e{i}", trade_date="2024-01-02", stock_code=f"60000{i}.SH",
                         base_reason="eligible", float_value=1e9+i, momentum_20=i*.001,
                         avg_amount_20=1e8, inst_ratio=.1 if i % 2 else -.1, label_return=i))
    frame = pd.DataFrame(rows)
    card = StrategyCard(match_caliper=3)
    first = select_cohorts(frame, card)[1]
    frame.label_return = frame.label_return * -100
    second = select_cohorts(frame, card)[1]
    pd.testing.assert_frame_equal(first, second)
    assert first.control_event_id.is_unique


def test_small_samples_do_not_get_confidence_interval():
    interval = block_interval(pd.DataFrame([dict(trade_date="2024-01-02", difference=.01)]), ["2024-01-02"], 100, 1, 10)
    assert interval["mean"] == .01 and interval["low"] is None


def test_artifact_tampering_is_detected(tmp_path):
    p = tmp_path / "card.json"
    write_json(p, {"x": 1})
    manifest = {"artifacts": {p.name: digest(p)}}
    verify_artifacts(tmp_path, manifest)
    write_json(p, {"x": 2})
    with pytest.raises(ValueError, match="完整性"):
        verify_artifacts(tmp_path, manifest)


def test_server_requires_local_host_and_csrf_token(tmp_path):
    from src.workbench.server import make_server, serve
    server = make_server(Path(__file__).parents[1], tmp_path, port=0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url) as response:
            html = response.read().decode()
            assert '龙虎榜研究工作台' in html
        serve(Path(__file__).parents[1], tmp_path, port=server.server_port, open_browser=False)
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url + '/api/run', data=b'{}', headers={'Content-Type': 'application/json'}))
        assert error.value.code == 403
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url, headers={'Host': 'external.invalid'}))
        assert error.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
