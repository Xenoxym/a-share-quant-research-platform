"""Score/account interface fixtures; stub score receipts are not actual model fits."""
import numpy as np
import pandas as pd
import pytest

from .test_alpha_learning import example
from src.alpharesearch.learning import LearningResult, LearningSpec, _prepare, _fingerprint
from src.alpharesearch.portfolio import (
    ScorePortfolioSpec, build_score_portfolio, score_portfolio_spec,
)
from src.technical.contracts import Execution, Experiment, ResearchSpec, Strategy
from src.technical.market import Market
from src.technical.portfolio import simulate


def scored_example(*, broad=False, frequency="weekly", only_first_prediction=False):
    protocol, model, decisions, assembly, labels = example()
    # Change the prediction decisions to actual weekly close -> next-session open.
    mapping = {"2024-01-08": ("2024-01-05", "2024-01-08", "2024-01-15"),
               "2024-01-09": ("2024-01-12", "2024-01-15", "2024-01-22")}
    if frequency == "monthly":
        mapping = {"2024-01-08": ("2024-01-31", "2024-02-01", "2024-03-01"),
                   "2024-01-09": ("2024-02-29", "2024-03-01", "2024-04-01")}
    for old, (day, entry, end) in mapping.items():
        selected = decisions.trade_date.eq(old)
        decisions.loc[selected, "trade_date"] = day
        decisions.loc[selected, "decision_at"] = f"{day}T15:30:00+08:00"
        decisions.loc[selected, "execution_at"] = f"{entry}T09:30:00+08:00"
        decisions.loc[selected, "target_end_at"] = f"{end}T09:30:00+08:00"
        for attr in ("values", "missing"):
            frame = getattr(assembly.block, attr)
            changed = frame.trade_date.eq(old)
            frame.loc[changed, "trade_date"] = day
            if attr == "values":
                frame.loc[changed, ["observed_end", "known_at"]] = f"{day}T15:30:00+08:00"
    if broad:
        for frame in (decisions, assembly.block.values, assembly.block.missing, labels):
            frame.loc[frame.stock_code.eq("600003.SH"), "stock_code"] = "300001.SZ"
    doc = protocol.to_dict()
    doc.update(train_end="2024-01-03", predict_start=mapping["2024-01-08"][0],
               predict_end=mapping["2024-01-09"][0], fit_cutoff="2024-01-04T23:00:00+08:00")
    if only_first_prediction:
        doc["predict_end"] = doc["predict_start"]
    protocol = LearningSpec.from_dict(doc)
    _, _, _, P, p, receipt = _prepare(protocol, model, decisions, assembly, labels)
    # Deliberate interface-only stub: no model.fit or LearningRunner.run occurs here.
    p["score"] = P["test.signal"].to_numpy() / 100
    p["fit_id"] = receipt["fit_id"]
    p["prediction_id"] = receipt["prediction_id"]
    receipt["prediction_outputs"] = _fingerprint(p)
    receipt["fitted_model"] = {
        "schema": "fitted-tabular-model-v1", "model_id": model.model_id,
        "training_rows": receipt["mature_training_rows"], "fit_attempts": 1,
        "account_results": False, "scope": "synthetic interface stub; never fitted",
        "spec": model.to_dict(), "feature_names": receipt["retained_features"],
        "transformed_features": 2 * len(receipt["retained_features"]),
        "environment": receipt["fit_identity"]["environment"],
        "implementation_hashes": {"src/alpharesearch/models.py":
            receipt["fit_identity"]["implementation_hashes"]["src/alpharesearch/models.py"]},
    }
    result = LearningResult(p, receipt)
    references = p[["sample_id", "trade_date", "stock_code"]].copy()
    references["reference_close"] = 10.
    references["reference_close_observed_at"] = references.trade_date + "T15:00:00+08:00"
    references["reference_close_known_at"] = references.trade_date + "T15:01:00+08:00"
    references["eligible"] = True
    references["eligibility_known_at"] = references.trade_date + "T15:01:00+08:00"
    last = "2024-04-10" if frequency == "monthly" else "2024-02-05"
    end = "2024-03-28" if frequency == "monthly" else "2024-01-18"
    calendar = pd.bdate_range("2024-01-02", last).strftime("%Y-%m-%d").tolist()
    spec = score_portfolio_spec(start="2024-01-02", end=end, top_k=2, rebalance=frequency)
    return result, references, calendar, spec


def build(data):
    return build_score_portfolio(*data)


def change_spec(data, **changes):
    data = list(data)
    doc = data[-1].to_dict()
    doc.update(changes)
    data[-1] = ScorePortfolioSpec.from_dict(doc)
    return data


def resign_scores(result, *, membership=False):
    # Only fixtures use this to isolate timing/key rejection from hash rejection.
    if membership:
        from src.technical.artifacts import content_id
        result.receipt["prediction_membership"] = _fingerprint(result.predictions[
            ["sample_id", "trade_date", "stock_code", "decision_at", "execution_at", "target_end_at"]])
        receipt = result.receipt
        receipt["prediction_id"] = content_id({"fit_id": receipt["fit_id"],
            "protocol_id": receipt["protocol_id"], "prediction_inputs": receipt["prediction_inputs"],
            "prediction_missing": receipt["prediction_missing"],
            "prediction_membership": receipt["prediction_membership"]})
        result.predictions["prediction_id"] = receipt["prediction_id"]
    result.receipt["prediction_outputs"] = _fingerprint(result.predictions)


def test_full_pool_deterministic_ranking_allocation_and_horizons():
    portfolio = build(scored_example())
    selected = portfolio.decisions.loc[portfolio.decisions.selected]
    assert len(portfolio.decisions) == 8 and len(selected) == 4
    assert selected.stock_code.tolist() == ["600003.SH", "600002.SH"] * 2
    assert selected.target_weight.eq(.49).all()
    assert portfolio.schedule == {"2024-01-05": "2024-01-08", "2024-01-12": "2024-01-15"}
    assert portfolio.receipt["target_mismatch_eligible_rows"] == 0
    assert portfolio.receipt["account_results"] is False
    assert portfolio.receipt["terminal_policy"].startswith("mark holdings")


def test_ties_stock_order_and_unfilled_slots_leave_declared_cash():
    data = scored_example()
    data[0].predictions["score"] = 0.
    resign_scores(data[0])
    portfolio = build(change_spec(data, top_k=6))
    assert portfolio.decisions.stock_code.tolist() == [
        "600000.SH", "600001.SH", "600002.SH", "600003.SH"] * 2
    sums = portfolio.decisions.groupby("trade_date").target_weight.sum()
    np.testing.assert_allclose(sums, .98 * 4 / 6)


def test_reference_row_order_cannot_move_scores_between_stocks():
    data = list(scored_example())
    one = build(data)
    data[1] = data[1].sample(frac=1, random_state=12).reset_index(drop=True)
    two = build(data)
    pd.testing.assert_frame_equal(one.decisions, two.decisions)
    assert one.receipt["decision_outputs"] == two.receipt["decision_outputs"]


def test_broad_learning_requires_explicit_account_subset_and_keeps_reasons():
    data = scored_example(broad=True)
    with pytest.raises(ValueError, match="explicit main-board"):
        build(data)
    result = build(change_spec(data, account_universe="main_board_subset"))
    assert result.receipt["reason_counts"]["outside_native_main_board"] == 2
    assert not result.decisions.loc[result.decisions.stock_code.eq("300001.SZ"), "selected"].any()
    assert len(result.decisions) == 8


def test_eligibility_is_explicit_and_missing_price_of_ineligible_row_does_not_remove_audit_row():
    data = scored_example()
    inactive = data[1].stock_code.eq("600003.SH")
    data[1].loc[inactive, "eligible"] = False
    data[1].loc[inactive, "reference_close"] = np.nan
    result = build(data)
    assert result.receipt["reason_counts"]["ineligible_declared"] == 2
    assert len(result.decisions) == 8
    assert not result.decisions.loc[result.decisions.stock_code.eq("600003.SH"), "selected"].any()


@pytest.mark.parametrize("mutation", [
    "score_changed", "missing_score", "reference_missing", "reference_wrong_id",
    "future_close", "before_observation", "close_wrong_day", "future_eligibility",
    "nonboolean_eligibility", "zero_close", "entry_same_day", "entry_wrong_time",
    "naive_clock", "decision_wrong_day", "duplicate_calendar", "missing_calendar_day",
    "missing_scheduled_pool", "different_fit_id", "wrong_fit_receipt", "no_fit_metadata",
])
def test_bad_score_account_inputs_reject(mutation):
    data = list(scored_example())
    result, references, calendar, spec = data
    if mutation == "score_changed":
        result.predictions.loc[0, "score"] += 1.
    elif mutation == "missing_score":
        result.predictions.loc[0, "score"] = np.nan
        resign_scores(result)
    elif mutation == "reference_missing":
        data[1] = references.drop(index=0)
    elif mutation == "reference_wrong_id":
        references.loc[0, "sample_id"] = "wrong-sample"
    elif mutation == "future_close":
        references.loc[0, "reference_close_known_at"] = "2024-01-08T10:00:00+08:00"
    elif mutation == "before_observation":
        references.loc[0, "reference_close_known_at"] = "2024-01-05T14:00:00+08:00"
    elif mutation == "close_wrong_day":
        references.loc[0, "reference_close_observed_at"] = "2024-01-04T15:00:00+08:00"
    elif mutation == "future_eligibility":
        references.loc[0, "eligibility_known_at"] = "2024-01-08T10:00:00+08:00"
    elif mutation == "nonboolean_eligibility":
        references["eligible"] = references.eligible.astype(object)
        references.loc[0, "eligible"] = 1
    elif mutation == "zero_close":
        references.loc[0, "reference_close"] = 0.
    elif mutation in ("entry_same_day", "entry_wrong_time", "naive_clock", "decision_wrong_day"):
        result.predictions = result.predictions.copy()
        for key in ("execution_at", "decision_at"):
            result.predictions[key] = result.predictions[key].astype(object)
        if mutation == "entry_same_day":
            result.predictions.loc[0, "execution_at"] = "2024-01-05T16:00:00+08:00"
        elif mutation == "entry_wrong_time":
            result.predictions.loc[0, "execution_at"] = "2024-01-08T10:00:00+08:00"
        elif mutation == "naive_clock":
            result.predictions.loc[0, "decision_at"] = "2024-01-05T15:30:00"
        else:
            result.predictions.loc[0, "decision_at"] = "2024-01-04T15:30:00+08:00"
        resign_scores(result, membership=True)
    elif mutation == "duplicate_calendar":
        calendar.append(calendar[-1])
    elif mutation == "missing_calendar_day":
        calendar.remove("2024-01-05")
    elif mutation == "missing_scheduled_pool":
        data = list(scored_example(only_first_prediction=True))
    elif mutation == "different_fit_id":
        result.predictions.loc[0, "fit_id"] = "z" * 64
        resign_scores(result)
    elif mutation == "wrong_fit_receipt":
        result.receipt["fit_identity"]["settings"]["fit_cutoff"] = "2024-01-08T00:00:00+08:00"
    elif mutation == "no_fit_metadata":
        result.receipt.pop("fitted_model")
    with pytest.raises((ValueError, KeyError)):
        build(data)


def test_different_forecast_horizon_requires_declaration_and_is_not_forced_exit():
    data = scored_example()
    result = data[0]
    result.predictions["target_end_at"] = pd.to_datetime(
        result.predictions.trade_date.map({"2024-01-05": "2024-01-12", "2024-01-12": "2024-01-19"})
        + "T15:00:00+08:00", utc=True)
    resign_scores(result, membership=True)
    with pytest.raises(ValueError, match="Learning target differs"):
        build(data)
    bridge = build(change_spec(data, horizon_policy="allow_forecast_mismatch"))
    assert bridge.receipt["target_mismatch_eligible_rows"] == 8
    assert bridge.receipt["terminal_policy"].startswith("mark holdings")


@pytest.mark.parametrize("field,value", [
    ("top_k", True), ("top_k", 101), ("allocation", np.nan), ("rebalance", "daily"),
    ("account_universe", "all_boards"), ("horizon_policy", "auto"),
    ("max_prediction_rows", 0), ("end", "2023-01-01"), ("schema", "unknown"),
])
def test_portfolio_spec_rejects_unregistered_choices(field, value):
    document = scored_example()[-1].to_dict()
    document[field] = value
    with pytest.raises(ValueError):
        ScorePortfolioSpec.from_dict(document)


def test_declared_prediction_row_budget_rejects():
    with pytest.raises(ValueError, match="row cap"):
        build(change_spec(scored_example(), max_prediction_rows=4))


@pytest.mark.parametrize("scenario", ["zero_transaction_cost", "configured"])
def test_bridge_to_existing_synthetic_account_funds_and_terminal_holdings(scenario):
    data = scored_example()
    bridge = build(data)
    _, _, calendar, policy = data
    codes = sorted(data[0].predictions.stock_code.unique())
    bars = pd.DataFrame([
        dict(stock_code=code, trade_date=day, open=10., high=10., low=10., close=10.,
             pre_close=10., volume=1000000., amount=10000000., high_limit=11.,
             low_limit=9., limit_price_valid=1, is_st=0, avg_volume_20=1000000.)
        for code in codes for day in calendar
    ])
    market = Market(bars, calendar)
    account = ResearchSpec(
        strategy=Strategy(top_k=policy.to_dict()["top_k"], allocation=.98, rebalance="weekly"),
        execution=Execution(initial_cash=100000., max_participation=.05),
        experiment=Experiment(start_date="2024-01-02", end_date="2024-01-18"),
    )
    sessions = [d for d in calendar if "2024-01-02" <= d <= "2024-01-18"]
    output = simulate(bridge.decisions, bridge.schedule, market, sessions, account, scenario)
    equity, ledger = output["equity"], output["ledger"]
    assert equity.cash.ge(0).all()
    np.testing.assert_allclose(equity.equity, equity.cash + equity.market_value + equity.receivable)
    assert equity.iloc[-1].positions == 2
    assert set(output["fills"].stock_code) == {"600002.SH", "600003.SH"}
    assert output["fills"].date.min() == "2024-01-08"
    assert output["metrics"]["buy_fills"] > 0
    assert equity.accounting_error.max() < .001
    assert equity.iloc[-1].cash == pytest.approx(100000. + ledger.cash_flow.sum())
    if scenario == "zero_transaction_cost":
        assert equity.iloc[-1].equity == pytest.approx(100000.)
    else:
        assert equity.iloc[-1].equity < 100000.



def test_monthly_schedule_uses_explicit_next_month_session_and_horizon():
    bridge = build(scored_example(frequency="monthly"))
    assert bridge.schedule == {"2024-01-31": "2024-02-01", "2024-02-29": "2024-03-01"}
    assert bridge.receipt['target_mismatch_eligible_rows'] == 0
    assert bridge.receipt['selected_rows'] == 4



def test_dropped_single_candidate_cannot_keep_original_prediction_identity():
    data = list(scored_example())
    original_id = data[0].receipt['prediction_id']
    data[0].predictions = data[0].predictions.drop(index=0).copy()
    data[1] = data[1].drop(index=0).copy()
    data[0].receipt['prediction_rows'] = 7
    resign_scores(data[0])
    assert data[0].receipt['prediction_id'] == original_id
    with pytest.raises(ValueError, match='complete output'):
        build(data)


def test_same_row_count_key_substitution_cannot_keep_source_membership_identity():
    data = scored_example()
    data[0].predictions.loc[0, 'sample_id'] = 'substituted-sample'
    data[1].loc[0, 'sample_id'] = 'substituted-sample'
    resign_scores(data[0])
    with pytest.raises(ValueError, match='membership'):
        build(data)


@pytest.mark.parametrize('mutation', ['empty_sample', 'future_mature_label', 'model_params', 'model_features'])
def test_inconsistent_source_contract_rejects_without_model_or_account(mutation):
    data = scored_example()
    if mutation == 'empty_sample':
        data[0].predictions.loc[0, 'sample_id'] = ''
        data[1].loc[0, 'sample_id'] = ''
        resign_scores(data[0])
    elif mutation == 'future_mature_label':
        data[0].receipt['latest_training_label_known_at'] = '2024-01-15T15:00:00+08:00'
    elif mutation == 'model_params':
        data[0].receipt['fitted_model']['spec']['family'] = 'huber'
    else:
        data[0].receipt['fitted_model']['feature_names'] = ['unrelated_feature']
    with pytest.raises(ValueError):
        build(data)



def test_full_source_receipt_content_is_bound_even_when_score_content_is_equal():
    data = scored_example()
    one = build(data)
    data[0].receipt['input_artifact_verification'] = 'changed caller provenance statement'
    two = build(data)
    pd.testing.assert_frame_equal(one.decisions, two.decisions)
    assert one.receipt['source_learning_receipt_id'] != two.receipt['source_learning_receipt_id']


def test_old_receipt_without_v2_membership_cannot_be_silently_upgraded():
    data = scored_example()
    data[0].receipt['schema'] = 'causal-learning-receipt-v1'
    data[0].receipt.pop('prediction_membership')
    with pytest.raises(ValueError, match='Causal learning receipt'):
        build(data)


def test_excluded_feature_summary_cannot_change_bound_training_features():
    data = scored_example()
    result = data[0]
    assert result.receipt['fit_identity']['excluded_train_empty'] == []
    original_id = result.receipt['fit_id']
    result.receipt['excluded_train_empty'] = ['test.event']
    result.receipt['retained_features'] = ['test.signal']
    result.receipt['fitted_model']['feature_names'] = ['test.signal']
    result.receipt['fitted_model']['transformed_features'] = 2
    assert result.receipt['fit_id'] == original_id
    with pytest.raises(ValueError, match='excluded features'):
        build(data)
