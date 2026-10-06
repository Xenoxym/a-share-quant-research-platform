"""Synthetic score/account evidence; score stubs never fit a model."""
from dataclasses import replace
import json
import shutil

import pandas as pd
import pytest

from .test_alpha_portfolio import build, resign_scores, scored_example
from .test_technical import bars, master
from src.alpharesearch.account import prepare_score_account
from src.alpharesearch.audit import audit_score_account_run, audit_score_portfolio
from src.technical.artifacts import digest, write_json
from src.technical.contracts import Execution, Experiment, ResearchSpec, Strategy
from src.technical.runner import run


def inputs(root):
    data = list(scored_example())
    data[2] = pd.bdate_range("2023-11-01", data[2][-1]).strftime("%Y-%m-%d").tolist()
    second = data[0].predictions.trade_date.eq("2024-01-12")
    data[0].predictions.loc[second, "score"] = data[0].predictions.loc[second, "stock_code"].map(
        {"600000.SH":4., "600001.SH":3., "600002.SH":2., "600003.SH":1.}).to_numpy()
    resign_scores(data[0])
    codes = sorted(data[0].predictions.stock_code.unique())
    quotes = bars(data[2], codes).drop(columns="avg_volume_20")
    down = quotes.trade_date.ge("2024-01-16")
    quotes.loc[down, ["open", "high", "low", "close", "pre_close"]] = 9.
    adjusted = quotes.stock_code.eq("600003.SH") & quotes.trade_date.between("2024-01-12", "2024-01-15")
    quotes.loc[adjusted, ["open", "high", "low", "close", "pre_close"]] = 8.91
    quotes.loc[adjusted, "low_limit"] = 8.
    data[1].loc[data[1].stock_code.eq("600003.SH") & data[1].trade_date.eq("2024-01-12"), "reference_close"] = 8.91
    snapshot = root/"snapshots"/("a"*24)
    snapshot.mkdir(parents=True)
    frames = dict(bars=quotes, metadata=master(codes),
        actions=pd.DataFrame([dict(stock_code="600003.SH", trade_date="2024-01-10",
            allotted_ps=.1, rationed_ps=0., rationed_px=0., bonus_ps=.2, dividend=.2)]),
        status=pd.DataFrame([dict(trade_date=d, status_type=k,
            symbols=["600003.SH"] if k=="HALT" and d in ["2024-01-10", "2024-01-11"] else [])
                             for d in data[2] for k in ["ST", "HALT"]]),
        benchmark=pd.DataFrame(dict(trade_date=data[2], close=10.)))
    for key, frame in frames.items():
        frame.to_parquet(snapshot/(key+".parquet"), index=False)
    write_json(snapshot/"calendar.json", data[2])
    write_json(snapshot/"manifest.json", dict(snapshot_id=snapshot.name, start=data[2][0],
        end=data[2][-1], sources=[], missing_action_files=[], quote_codes=len(codes),
        master_codes_without_quotes_in_window=[], counts={k:len(v) for k,v in frames.items()},
        artifacts={p.name:digest(p) for p in snapshot.iterdir()}))
    cfg = data[3].to_dict()
    account = ResearchSpec(strategy=replace(Strategy(), name=cfg["name"], top_k=cfg["top_k"],
                              allocation=cfg["allocation"], rebalance=cfg["rebalance"]),
        execution=Execution(initial_cash=100000., max_participation=.05),
        experiment=Experiment(cfg["start"], cfg["end"]))
    return data, snapshot, account


def prepare(values):
    data, snapshot, account = values
    return prepare_score_account(*data, snapshot, account)


def test_prepared_inputs_have_snapshot_relative_prices_code_and_clear_definition(tmp_path):
    data = inputs(tmp_path)
    prepared = prepare(data)
    assert prepared["market"].quote("600000.SH", "2024-01-05")["close"] == 10.
    assert prepared["market"].quote("600000.SH", "2024-01-05")["avg_volume_20"] == 1000000.
    assert "模型" in prepared["signal_definition"]["momentum"]
    assert "eligible" in prepared["signal_definition"]["filters"]
    assert prepared["extras"]["alpha_account_receipt.json"]["numeric_model_reproduction"] is False
    original = prepared["extras"]["alpha_predictions.parquet"].score.iloc[0]
    data[0][0].predictions.loc[0, "score"] = 99.
    assert prepared["extras"]["alpha_predictions.parquet"].score.iloc[0] == original


@pytest.mark.parametrize("change", ["interval", "top_k", "allocation", "frequency", "family",
                                  "hidden_filter", "calendar", "missing_artifact",
                                  "modified_raw_price", "snapshot_count", "unsafe_artifact"])
def test_prepared_contract_rejects_before_any_account(change, tmp_path):
    values = list(inputs(tmp_path))
    data, snapshot, spec = values
    if change == "interval":
        values[2] = replace(spec, experiment=Experiment("2024-01-03", "2024-01-18"))
    elif change in {"top_k", "allocation", "frequency", "family", "hidden_filter"}:
        alter = {"top_k":{"top_k":3}, "allocation":{"allocation":.5},
                 "frequency":{"rebalance":"monthly"}, "family":{"family":"size"},
                 "hidden_filter":{"min_history":60}}[change]
        values[2] = replace(spec, strategy=replace(spec.strategy, **alter))
    elif change == "calendar":
        data[2] = data[2][1:]
    else:
        sm = json.loads((snapshot/"manifest.json").read_text())
        if change == "missing_artifact":
            del sm["artifacts"]["actions.parquet"]
        elif change == "unsafe_artifact":
            sm["artifacts"]["../escape.parquet"] = "0"*64
        elif change == "snapshot_count":
            sm["counts"]["bars"] += 1
        else:
            quotes = pd.read_parquet(snapshot/"bars.parquet")
            quotes.loc[quotes.trade_date.eq("2024-01-05"), "close"] = 11.
            quotes.to_parquet(snapshot/"bars.parquet", index=False)
            sm["artifacts"]["bars.parquet"] = digest(snapshot/"bars.parquet")
        write_json(snapshot/"manifest.json", sm)
    with pytest.raises(ValueError):
        prepare(values)


@pytest.mark.parametrize("change", ["selected", "weight", "rank", "score", "schedule", "source", "rows"])
def test_independent_selection_rejects_coherently_resigned_outputs(change):
    data = scored_example()
    portfolio = build(data)
    if change == "schedule":
        portfolio.schedule["2024-01-05"] = "2024-01-09"
    elif change == "source":
        data[0].receipt["numeric_source"] = "unbound replacement"
    elif change == "rows":
        portfolio.decisions = portfolio.decisions.iloc[1:].copy()
    else:
        index = portfolio.decisions.index[0]
        key = {"weight":"target_weight"}.get(change, change)
        portfolio.decisions.loc[index, key] = False if key=="selected" else 123.
    from src.alpharesearch.learning import _fingerprint
    portfolio.receipt["decision_outputs"] = _fingerprint(portfolio.decisions)
    with pytest.raises(ValueError):
        audit_score_portfolio(*data[:3], portfolio)


@pytest.fixture(scope="module")
def archived_account(tmp_path_factory):
    # One native run with two cost scenarios per pytest invocation. All tamper
    # checks reuse copies of this evidence; they never rerun an account.
    root = tmp_path_factory.mktemp("alpha_account")
    values = inputs(root)
    prepared = prepare(values)
    folder = run(root, values[2], root=root, prepared=prepared,
                 scenarios=["zero_transaction_cost", "configured"], publish=False,
                 research_context={"role":"screen", "source":"synthetic_e18_interface_stub"})
    return root, folder


def test_native_archive_has_audited_returns_drawdown_and_rotation(archived_account):
    root, folder = archived_account
    result = audit_score_account_run(folder)
    assert set(result["scenarios"]) == {"zero_transaction_cost", "configured"}
    assert result["selection"]["selected_rows"] == 4
    assert result["numeric_model_reproduction"] is False
    for audit in result["scenarios"].values():
        metrics = audit["recalculated_metrics"]
        assert metrics["buy_fills"] > 0 and metrics["sell_fills"] > 0
        assert metrics["max_drawdown"] < 0
        assert metrics["total_return"] < 0
        assert audit["annualized_return"] is None
        assert max(audit["daily_errors"].values()) < .001
    eq = pd.read_parquet(folder/"configured"/"equity.parquet")
    holdings = pd.read_parquet(folder/"configured"/"positions.parquet")
    halted = holdings.stock_code.eq("600003.SH") & holdings.date.eq("2024-01-10")
    assert holdings.loc[halted, "mark"].iloc[0] == pytest.approx((10.-.2)/1.1)
    assert holdings.loc[halted, "stale_reason"].iloc[0] == "halt_mark"
    assert eq.receivable.max() > 0
    sold = pd.read_parquet(folder/"configured"/"fills.parquet")
    assert (sold.loc[sold.side.eq("sell"), "filled_shares"] % 100 != 0).any()
    assert not (root/"latest.json").exists()


def resign_archive(folder):
    manifest = json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"] = {name:digest(folder/name) for name in manifest["artifacts"]}
    write_json(folder/"manifest.json", manifest)


@pytest.mark.parametrize("change", ["cash", "shares", "fill_price", "fees", "targets", "metric",
                                  "drawdown", "definition", "source_code", "score_member", "nonfinite_ledger", "embedded_equity"])
def test_archive_audit_detects_resigned_financial_or_source_tampering(change, archived_account, tmp_path):
    root, baseline = archived_account
    local = tmp_path/"r"
    shutil.copytree(root/"snapshots", local/"snapshots")
    folder = local/"runs"/baseline.name
    shutil.copytree(baseline, folder)
    if change in {"metric", "definition", "embedded_equity"}:
        name = "definition.json" if change=="definition" else "result.json"
        document = json.loads((folder/name).read_text(encoding="utf-8"))
        if change=="metric":
            document["portfolios"][0]["metrics"]["total_return"] += .25
        elif change=="embedded_equity":
            document["portfolios"][0]["equity"][-1]["nav"] = 10.
        else:
            document["filters"] = "旧动量资格规则"
        write_json(folder/name, document)
    elif change=="source_code":
        name = "alpha_scoring_code.json"
        bundle = json.loads((folder/name).read_text(encoding="utf-8"))
        key = "src/alpharesearch/models.py"
        bundle["files"][key]["text"] += "\n# mutation\n"
        import hashlib
        bundle["files"][key]["sha256"] = hashlib.sha256(bundle["files"][key]["text"].encode()).hexdigest()
        write_json(folder/name, bundle)
        from src.technical.artifacts import content_id
        receipt = json.loads((folder/"alpha_account_receipt.json").read_text())
        receipt["code_bundle_id"] = content_id(bundle)
        write_json(folder/"alpha_account_receipt.json", receipt)
    else:
        which, column = {"cash":("equity", "cash"), "shares":("positions", "shares"),
                         "fill_price":("fills", "price"), "fees":("fills", "total"),
                         "targets":("targets", "target_shares"), "drawdown":("equity", "drawdown"),
                         "score_member":("alpha_predictions", "sample_id"),
                         "nonfinite_ledger":("ledger", "cash_flow")}[change]
        path = folder/(which+".parquet") if change=="score_member" else folder/"configured"/(which+".parquet")
        frame = pd.read_parquet(path)
        if change=="score_member":
            frame.loc[0,column] = "replaced-source-member"
        elif change=="nonfinite_ledger":
            frame.loc[0,column] = float("nan")
        else:
            frame.loc[0,column] += 100. if change not in {"drawdown", "fill_price"} else .01
        frame.to_parquet(path, index=False)
    resign_archive(folder)
    with pytest.raises(ValueError):
        audit_score_account_run(folder)


def test_resigned_learning_summary_cannot_change_bound_fit_identity():
    from src.technical.artifacts import content_id
    data = scored_example()
    portfolio = build(data)
    data[0].receipt['fitted_model']['spec']['family'] = 'huber'
    portfolio.receipt['source_learning_receipt_id'] = content_id(data[0].receipt)
    with pytest.raises(ValueError, match='metadata identity'):
        audit_score_portfolio(*data[:3], portfolio)

def manual_balanced_output(*, quantity=4900, late_mark=10., early_exit=False):
    """Hand-built internally balanced counterexamples; never an engine run."""
    from src.alpharesearch.audit import audit_account_tables
    days = ["2024-01-08", "2024-01-09"]; code="600001.SH"
    account = ResearchSpec(strategy=Strategy(top_k=2, allocation=.98, rebalance="weekly"),
        execution=Execution(initial_cash=100000., max_participation=.05),
        experiment=Experiment(days[0], days[-1]))
    quote = bars(days, [code])
    fills = [dict(date=days[0], stock_code=code, side="buy", requested_shares=4900,
        submitted_shares=quantity, filled_shares=quantity, price=10., commission=0.,
        other_fee=0., stamp_tax=0., slippage=0., total=0.)]
    if early_exit:
        fills.append(dict(fills[0], date=days[1], side="sell", requested_shares=quantity))
    fills = pd.DataFrame(fills)
    ledger = pd.DataFrame([dict(date=r.date, stock_code=code, kind=r.side,
        cash_flow=(-1 if r.side=="buy" else 1)*r.price*r.filled_shares,
        receivable_flow=0., shares_delta=(1 if r.side=="buy" else -1)*r.filled_shares,
        fee=0.) for r in fills.itertuples()])
    positions, equity = [], []
    for i,day in enumerate(days):
        held = not (i==1 and early_exit)
        mark = late_mark if i==1 else 10.
        mv = quantity*mark if held else 0.
        cash = 100000.-quantity*10. if held else 100000.
        unrl = quantity*(mark-10.) if held else 0.
        if held:
            positions.append(dict(date=day, stock_code=code, shares=quantity, mark=mark,
                market_value=mv, cost_basis=quantity*10., unrealized_pnl=unrl,
                target_shares=4900, stale_reason=""))
        equity.append(dict(date=day, cash=cash, receivable=0., market_value=mv,
            equity=cash+mv, nav=(cash+mv)/100000., positions=int(held),
            exposure=mv/(cash+mv), cumulative_realized_pnl=0., unrealized_pnl=unrl,
            unresolved_market_value=0., equity_zero_unresolved=cash+mv, accounting_error=0.))
    equity = pd.DataFrame(equity)
    equity["drawdown"] = equity.nav/equity.nav.cummax().clip(lower=1)-1
    final = equity.iloc[-1]
    metrics = dict(scenario="zero_transaction_cost", final_equity=final.equity,
        total_return=final.nav-1, max_drawdown=equity.drawdown.min(), total_cost=0.,
        average_exposure=equity.exposure.mean(), turnover_notional=float((fills.price*fills.filled_shares).sum()),
        receivable=0., buy_fills=1, sell_fills=int(early_exit), final_positions=int(not early_exit),
        unresolved_market_value=0., realized_pnl=0., unrealized_pnl=final.unrealized_pnl,
        max_accounting_error=0., cash_error=0., receivable_error=0., annualized_return=None,
        cost_components={k:0. for k in ["commission","other_fee","stamp_tax","slippage"]})
    targets = pd.DataFrame([dict(signal_date="2024-01-05", execution_date=days[0], stock_code=code,
        equity_used=100000., reference_close=10., target_weight=.49, target_shares=4900., rank=1.)])
    decisions = pd.DataFrame([dict(trade_date="2024-01-05", stock_code=code, selected=True,
                                  target_weight=.49, close=10., rank=1.)])
    output = dict(equity=equity, fills=fills, ledger=ledger, positions=pd.DataFrame(positions),
                  targets=targets, metrics=metrics)
    kwargs = dict(decisions=decisions, schedule={"2024-01-05":days[0]},
        actions=pd.DataFrame(columns=["stock_code", "trade_date", "allotted_ps", "rationed_ps", "rationed_px", "dividend"]),
        status=pd.DataFrame(columns=["trade_date", "status_type", "symbols"]), metadata=master([code]))
    return audit_account_tables, (output, quote, account, "zero_transaction_cost", days), kwargs


@pytest.mark.parametrize("changes,message", [
    ({"quantity":9000}, "target gap"),
    ({"late_mark":20.}, "holdings/mark"),
    ({"early_exit":True}, "target gap"),
])
def test_balanced_fake_profit_overbuy_or_premature_exit_is_rejected(changes, message):
    check, args, kwargs = manual_balanced_output(**changes)
    with pytest.raises(ValueError, match=message):
        check(*args, **kwargs)


@pytest.mark.parametrize("quantity", [4500, 4900])
def test_legitimate_partial_or_full_target_can_leave_cash(quantity):
    check, args, kwargs = manual_balanced_output(quantity=quantity)
    assert check(*args, **kwargs)["recalculated_metrics"]["total_return"] == 0.


def test_wrong_fee_component_cannot_hide_under_correct_total():
    check, args, kwargs = manual_balanced_output()
    args[0]["metrics"]["cost_components"]["commission"] = 999999.
    with pytest.raises(ValueError, match="cost component"):
        check(*args, **kwargs)
