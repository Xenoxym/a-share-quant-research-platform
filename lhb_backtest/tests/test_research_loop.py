from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from src.technical.contracts import ResearchSpec, Experiment
from src.technical.diagnostics import contributions, period_stats, market_states, drawdown_episodes, summarize, execution_delays
from src.technical.laboratory import plan, score_trial, select


def test_stock_attribution_includes_closed_positions_fees_and_entitlements():
    eq = pd.DataFrame({"date": ["2020-01-02", "2020-01-03", "2020-01-06"], "equity": [995., 1015., 1009.]})
    pos = pd.DataFrame({"date": eq.date[:2], "stock_code": ["A", "A"], "market_value": [500., 500.]})
    # Buy 500 + fee 5; bonus/dividend preserves mark value and adds receivable;
    # full sale 499 - fee 5 closes the position, which must still appear in P&L.
    ledger = pd.DataFrame({"date": eq.date, "stock_code": ["A"]*3,
        "cash_flow": [-505., 0., 494.], "receivable_flow": [0., 20., 0.], "fee": [5., 0., 5.]})
    a = contributions(eq, pos, ledger, 1000.)
    assert a.net_pnl.tolist() == [-5., 20., -6.]
    assert a.fee.sum() == 10.
    assert a.market_value.iloc[-1] == 0
    with pytest.raises(ValueError, match="对账"):
        contributions(eq.assign(equity=[995., 1016., 1009.]), pos, ledger, 1000.)


def test_period_uses_carry_in_equity_and_initial_peak():
    eq = pd.DataFrame({"date": ["2020-01-02", "2020-01-03", "2020-01-06"], "equity": [110., 88., 99.]})
    s = period_stats(eq, 100, "2020-01-03", "2020-01-06")
    assert s["return"] == pytest.approx(-.1)
    assert s["max_drawdown"] == pytest.approx(-.2)
    assert s["opening_equity"] == 110.
    assert period_stats(eq, 100)["return"] == pytest.approx(-.01)
    with pytest.raises(ValueError):
        period_stats(eq, 100, "2021-01-01", "2021-12-31")


def test_market_state_is_lagged_and_future_independent():
    days = pd.bdate_range("2020-01-01", periods=200).strftime("%Y-%m-%d").tolist()
    b = {"rows": [{"date": d, "nav": 1+i*.001} for i, d in enumerate(days)]}
    original = market_states(b, days)
    mutated = {"rows": [r if i < 150 else dict(r, nav=r["nav"]*2) for i, r in enumerate(b["rows"])]}
    pd.testing.assert_frame_equal(original.iloc[:151], market_states(mutated, days).iloc[:151])
    assert original.state.iloc[126] == "历史不足"
    assert original.state.iloc[127] == "上行 · 常态波动"


def test_drawdown_episode_tracks_recovery_and_open_episode():
    eq = pd.DataFrame({"date": list("abcdef"), "equity": [100, 80, 90, 100, 90, 95]})
    episodes = drawdown_episodes(eq, 100.)
    assert episodes[0]["trough"] == "b" and episodes[0]["recovery"] == "d"
    assert episodes[1]["recovery"] is None and episodes[1]["end"] == "f"


def test_batch_declares_24_trials_and_selection_cannot_read_validation():
    p = plan(replace(ResearchSpec(), experiment=Experiment(end_date="2026-09-24")), "2024-01-01")
    assert len(p["trials"]) == 24
    assert len({r["spec"]["strategy"]["name"] for r in p["trials"]}) == 24
    rows = [{"trial": 1, "screen_score": 1., "validation": {"return": -.9}},
            {"trial": 2, "screen_score": .5, "validation": {"return": 99.}}]
    assert select(rows)["trial"] == 1
    rows[0]["validation"]["return"] = -1
    assert select(rows)["trial"] == 1
    assert select([{ "trial": 1, "screen_score": None}]) is None


def test_batch_validation_keeps_boundary_position_and_screen_score_fixed():
    days = pd.bdate_range("2020-01-01", periods=300).strftime("%Y-%m-%d")
    eq = pd.DataFrame({"date": days, "equity": 100*np.cumprod(1+np.sin(np.arange(300))*.01+.001)})
    fills = pd.DataFrame({"date": [days[1]]})
    old = score_trial(eq, fills, 100., days[180])
    changed = eq.copy()
    changed.loc[180:, "equity"] *= .1
    new = score_trial(changed, fills, 100., days[180])
    assert old["screen_score"] == new["screen_score"]
    assert old["validation"]["opening_equity"] == eq.equity.iloc[179]
    assert new["validation"]["return"] != old["validation"]["return"]


def test_all_cash_diagnostics_do_not_invent_trades():
    eq = pd.DataFrame({"date": ["2020-01-02", "2020-01-03"], "equity": [100., 100.], "exposure": [0., 0.]})
    pos = pd.DataFrame(columns=["date", "stock_code", "market_value"])
    ledger = pd.DataFrame(columns=["date", "stock_code", "cash_flow", "receivable_flow", "fee"])
    attr = contributions(eq, pos, ledger, 100.)
    summary = summarize(eq, attr, pd.DataFrame(columns=["date", "stock_code", "kind"]), 100., {"rows": []})
    assert summary["attribution_error"] == 0.
    assert not summary["episodes"]
    assert summary["months"][0]["return"] == 0.


def test_execution_delay_matches_latest_known_target_without_future_target():
    sessions = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    orders = pd.DataFrame({"date": [sessions[0], sessions[2], sessions[2]], "stock_code": ["A", "A", "B"],
        "side": ["buy"]*3, "filled_shares": [100, 100, 0], "price": [10., 20., 20.]})
    targets = pd.DataFrame({"stock_code": ["A", "A"], "execution_date": [sessions[0], sessions[3]],
                            "signal_date": ["2020-01-01", sessions[2]]})
    d = execution_delays(orders, targets, sessions)
    assert d["same_session_fraction"] == .5
    assert d["median_delay"] == 1.
    assert d["notional_delayed_fraction"] == pytest.approx(2/3)
    assert d["examples"][0]["execution_date"] == sessions[0]


def test_batch_plan_rejects_unfrozen_end_date():
    with pytest.raises(ValueError, match="具体日期"):
        plan(ResearchSpec(), "2024-01-01")
