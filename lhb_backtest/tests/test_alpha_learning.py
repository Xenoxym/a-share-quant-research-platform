"""Synthetic causal learning counterexamples; no market data or account returns."""
import numpy as np
import pandas as pd
import pytest

from src.alpharesearch.assembly import FeatureAssembly
from src.alpharesearch.contracts import MissingReason as M
from src.alpharesearch.features.base import FeatureBlock
from src.alpharesearch.learning import LearningRunner, LearningSpec, learning_spec
from src.alpharesearch.models import model_spec
from src.alpharesearch.registry import FeatureDefinition, FeatureRegistry


def example():
    rows = []
    for day in ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-08", "2024-01-09"]:
        for i in range(4):
            rows.append(dict(sample_id=f"{day}-{i}", trade_date=day,
                             stock_code=f"{600000 + i:06}.SH",
                             decision_at=f"{day}T09:00:00+08:00",
                             execution_at=f"{day}T09:30:00+08:00",
                             target_end_at=f"{day}T15:00:00+08:00"))
    decisions = pd.DataFrame(rows)
    names = ["test.signal", "test.event"]
    registry = FeatureRegistry([
        FeatureDefinition(key, "ratio", "stock_day", "synthetic", ("synthetic",), "{}",
                          (("synthetic.py", "a" * 64),), "by decision", "explicit")
        for key in names
    ])
    values = decisions[["sample_id", "trade_date", "stock_code"]].copy()
    values[names[0]] = np.tile([-2., -1., 1., 2.], 5)
    values[names[1]] = np.tile([1., np.nan, 0., 1.], 5)
    values["observed_end"] = decisions.decision_at
    values["known_at"] = decisions.decision_at
    missing = values[["sample_id", "trade_date", "stock_code"]].copy()
    for key in names:
        missing[key] = np.where(values[key].isna(), M.UNCOVERED.value, M.PRESENT.value)
    bindings = ({
        "namespace": "test", "materialization_id": "b" * 64,
        "source_id": "synthetic", "source_version": "synthetic-v1",
        "vintage": "market_observation", "definition_ids": [
            d.definition_id for d in registry.definitions
        ],
    },)
    block = FeatureBlock(values, missing, {k: "ratio" for k in names}, {
        "schema": "assembled-feature-matrix-v1",
        "key_columns": ["sample_id", "trade_date", "stock_code"],
        "source_id": "synthetic", "version_id": "c" * 64,
        "vintage": "market_observation", "universe_id": "synthetic_universe",
        "registry_version": registry.version_id, "source_bindings": list(bindings),
    })
    assembly = FeatureAssembly(block, registry, bindings)
    labels = decisions[["sample_id", "trade_date", "stock_code"]].copy()
    labels["target_return"] = values[names[0]] * 0.03
    labels["label_start"] = decisions.execution_at
    labels["label_end"] = decisions.target_end_at
    labels["label_known_at"] = decisions.target_end_at
    labels["label_observed"] = True
    # Last training day has a future-ending outcome, unavailable at cutoff.
    late = decisions.trade_date.eq("2024-01-04")
    decisions.loc[late, "target_end_at"] = "2024-01-10T15:00:00+08:00"
    labels.loc[late, "label_end"] = "2024-01-10T15:00:00+08:00"
    labels.loc[late, "label_known_at"] = "2024-01-10T15:01:00+08:00"
    protocol = learning_spec(
        dataset_id="synthetic_dataset", universe_id="synthetic_universe",
        features=names, train=("2024-01-02", "2024-01-04"),
        predict=("2024-01-08", "2024-01-09"),
        fit_cutoff="2024-01-07T23:00:00+08:00",
    )
    return protocol, model_spec("ridge"), decisions, assembly, labels


def run(data, runner=None):
    return (runner or LearningRunner(max_calls=8, max_fit_intents=4)).run(*data)


def test_mature_labels_prediction_keys_and_actual_fit_receipt():
    result = run(example())
    assert result.receipt["training_decisions"] == 12
    assert result.receipt["mature_training_rows"] == 8
    assert result.receipt["observed_but_immature_labels"] == 4
    assert len(result.predictions) == 8
    assert result.predictions.score.notna().all()
    assert result.receipt["actual_fit_attempts_this_call"] == 1
    assert result.receipt["account_results"] is False
    assert result.receipt["fitted_model"]["training_rows"] == 8


def test_immature_target_perturbation_neither_refits_nor_changes_scores():
    data = example()
    runner = LearningRunner(max_calls=2, max_fit_intents=1)
    one = run(data, runner)
    data[-1].loc[data[-1].trade_date.eq("2024-01-04"), "target_return"] = 999999.
    two = run(data, runner)
    pd.testing.assert_frame_equal(one.predictions, two.predictions)
    assert two.receipt["fit_reused"] and runner.fit_intents == 1
    assert two.receipt["actual_fit_attempts_this_call"] == 0


def test_prediction_needs_no_future_labels_and_future_label_values_are_ignored():
    data = list(example())
    data[-1] = data[-1].loc[data[-1].trade_date < "2024-01-08"].copy()
    runner = LearningRunner(max_calls=2, max_fit_intents=1)
    without = run(data, runner)
    other = list(example())
    future = other[-1].trade_date >= "2024-01-08"
    other[-1]["target_return"] = other[-1].target_return.astype(object)
    other[-1].loc[future, "target_return"] = "unread future outcome"
    other[-1].loc[future, "label_known_at"] = "unread future time"
    with_future = run(other, runner)
    pd.testing.assert_frame_equal(without.predictions, with_future.predictions)
    assert with_future.receipt["fit_reused"]


def test_appended_future_data_is_not_training_or_prediction_input():
    data = list(example())
    runner = LearningRunner(max_calls=2, max_fit_intents=1)
    one = run(data, runner)
    d = data[2].iloc[:4].copy()
    d["sample_id"] = [f"future-{i}" for i in range(4)]
    d["trade_date"] = "2024-02-01"
    for key, clock in [("decision_at", "09:00"), ("execution_at", "09:30"), ("target_end_at", "15:00")]:
        d[key] = f"2024-02-01T{clock}:00+08:00"
    data[2] = pd.concat([data[2], d], ignore_index=True)
    block = data[3].block
    for attr in ("values", "missing"):
        f = getattr(block, attr).iloc[:4].copy()
        f["sample_id"] = d.sample_id.to_numpy()
        f["trade_date"] = "2024-02-01"
        if attr == "values":
            f["test.signal"] = 999999.
            f["observed_end"] = f["known_at"] = "2024-02-01T09:00:00+08:00"
        setattr(block, attr, pd.concat([getattr(block, attr), f], ignore_index=True))
    block.metadata["version_id"] = "d" * 64
    two = run(data, runner)
    pd.testing.assert_frame_equal(one.predictions, two.predictions)
    assert two.receipt["fit_reused"]


def test_predict_values_change_scores_without_refitting_or_training_stats():
    data = example()
    runner = LearningRunner(max_calls=2, max_fit_intents=1)
    one = run(data, runner)
    pred = data[3].block.values.trade_date >= "2024-01-08"
    data[3].block.values.loc[pred, "test.signal"] = 1000.
    two = run(data, runner)
    assert two.receipt["fit_reused"] and one.receipt["fit_id"] == two.receipt["fit_id"]
    assert one.receipt["prediction_id"] != two.receipt["prediction_id"]
    assert not one.predictions.score.equals(two.predictions.score)
    assert one.receipt["fitted_model"] == two.receipt["fitted_model"]


def test_label_rows_are_key_joined_rather_than_position_aligned():
    data = list(example())
    one = run(data)
    data[-1] = data[-1].sample(frac=1, random_state=5).reset_index(drop=True)
    two = run(data)
    pd.testing.assert_frame_equal(one.predictions, two.predictions)


@pytest.mark.parametrize("mutation", [
    "late_feature", "late_observation", "naive_clock", "entry_before_decision",
    "end_before_entry", "wrong_day", "feature_wrong_sample", "duplicate_stock_day",
    "bad_label_window", "premature_known", "unobserved_known", "unknown_flag",
    "wrong_universe", "registry_binding", "weak_vintage", "target_inf",
])
def test_bad_information_contracts_are_rejected_before_fit(mutation):
    data = list(example())
    protocol, model, decisions, assembly, labels = data
    if mutation == "late_feature":
        assembly.block.values.loc[0, "known_at"] = "2024-01-02T10:00:00+08:00"
    elif mutation == "late_observation":
        assembly.block.values.loc[0, ["known_at", "observed_end"]] = "2024-01-02T10:00:00+08:00"
    elif mutation == "naive_clock":
        decisions.loc[0, "decision_at"] = "2024-01-02T09:00:00"
    elif mutation == "entry_before_decision":
        decisions.loc[0, "execution_at"] = "2024-01-02T08:59:00+08:00"
    elif mutation == "end_before_entry":
        decisions.loc[0, "target_end_at"] = "2024-01-02T09:29:00+08:00"
    elif mutation == "wrong_day":
        decisions.loc[0, "decision_at"] = "2024-01-01T09:00:00+08:00"
    elif mutation == "feature_wrong_sample":
        assembly.block.values.loc[0, "sample_id"] = "unknown"
    elif mutation == "duplicate_stock_day":
        decisions.loc[1, "stock_code"] = decisions.loc[0, "stock_code"]
    elif mutation == "bad_label_window":
        labels.loc[0, "label_start"] = "2024-01-02T09:29:00+08:00"
    elif mutation == "premature_known":
        labels.loc[0, "label_known_at"] = "2024-01-02T14:00:00+08:00"
    elif mutation == "unobserved_known":
        labels.loc[0, "label_observed"] = False
    elif mutation == "unknown_flag":
        labels["label_observed"] = labels.label_observed.astype(object)
        labels.loc[0, "label_observed"] = 1
    elif mutation == "wrong_universe":
        assembly.block.metadata["universe_id"] = "other"
    elif mutation == "registry_binding":
        assembly.block.metadata["registry_version"] = "fake"
    elif mutation == "weak_vintage":
        assembly.block.metadata["vintage"] = "latest_snapshot_only"
    elif mutation == "target_inf":
        labels.loc[0, "target_return"] = np.inf
    runner = LearningRunner(max_calls=1, max_fit_intents=1)
    with pytest.raises((ValueError, TypeError)):
        run(data, runner)
    assert runner.fit_intents == 0 and runner.calls == 1
    assert runner.ledger[-1]["state"] == "failed"


def test_train_empty_policy_cannot_depend_on_prediction_coverage():
    data = list(example())
    block = data[3].block
    train = block.values.trade_date < "2024-01-08"
    block.values.loc[train, "test.event"] = np.nan
    block.missing.loc[train, "test.event"] = M.UNCOVERED.value
    with pytest.raises(ValueError, match="explicit exclusion"):
        run(data)
    doc = data[0].to_dict()
    doc["empty_feature_policy"] = "exclude_train_empty"
    data[0] = LearningSpec.from_dict(doc)
    result = run(data)
    assert result.receipt["excluded_train_empty"] == ["test.event"]
    assert result.receipt["retained_features"] == ["test.signal"]
    assert result.receipt["prediction_rows"] == 8


def test_unobserved_and_absent_training_labels_are_counted_and_not_used():
    data = list(example())
    data[-1].loc[0, ["target_return", "label_known_at"]] = [np.nan, None]
    data[-1].loc[0, "label_observed"] = False
    data[-1] = data[-1].drop(index=1)
    result = run(data)
    assert result.receipt["mature_training_rows"] == 6
    assert result.receipt["omitted_or_unobserved_labels"] == 2


def test_equal_day_weights_and_negative_control_are_explicit_distinct_fits():
    data = list(example())
    data[-1] = data[-1].drop(index=[0, 1])
    runner = LearningRunner(max_calls=3, max_fit_intents=3)
    baseline = run(data, runner)
    doc = data[0].to_dict()
    doc["weighting"] = "equal_decision_date"
    data[0] = LearningSpec.from_dict(doc)
    weighted = run(data, runner)
    doc["negative_control"] = "within_date_label_permutation"
    data[0] = LearningSpec.from_dict(doc)
    control = run(data, runner)
    assert len({r.receipt["fit_id"] for r in [baseline, weighted, control]}) == 3
    assert not baseline.receipt["fitted_model"]["sample_weight_used"]
    assert weighted.receipt["fitted_model"]["sample_weight_used"]
    assert runner.fit_intents == 3
    # The caller's labels remain unshuffled.
    assert data[-1].loc[2, "target_return"] == 0.03


@pytest.mark.parametrize("mutation", ["overlap", "cutoff_late", "cutoff_early", "naive", "unknown", "duplicate_feature"])
def test_strict_protocol(mutation):
    doc = example()[0].to_dict()
    if mutation == "overlap":
        doc["predict_start"] = "2024-01-04"
    elif mutation == "cutoff_late":
        doc["fit_cutoff"] = "2024-01-08T10:00:00+08:00"
    elif mutation == "cutoff_early":
        doc["fit_cutoff"] = "2024-01-03T08:00:00+08:00"
    elif mutation == "naive":
        doc["fit_cutoff"] = "2024-01-07T23:00:00"
    elif mutation == "unknown":
        doc["auto_optimize"] = True
    else:
        doc["features"].append(doc["features"][0])
    with pytest.raises(ValueError):
        spec = LearningSpec.from_dict(doc)
        data = list(example())
        data[0] = spec
        run(data)


def test_maturity_boundary_includes_outcomes_known_exactly_at_cutoff():
    data = list(example())
    doc = data[0].to_dict()
    doc["fit_cutoff"] = "2024-01-04T09:00:00+08:00"
    data[0] = LearningSpec.from_dict(doc)
    data[-1].loc[0, "label_known_at"] = doc["fit_cutoff"]
    result = run(data)
    assert result.receipt["mature_training_rows"] == 8


def test_fit_failure_and_call_budget_never_implicitly_retry(monkeypatch):
    data = example()
    runner = LearningRunner(max_calls=3, max_fit_intents=1)
    from src.alpharesearch import models
    attempts = []
    def fail(spec):
        attempts.append(1)
        raise RuntimeError("synthetic initialization failure")
    monkeypatch.setattr(models, "_pipeline", fail)
    with pytest.raises(RuntimeError, match="initialization"):
        run(data, runner)
    assert runner.fit_intents == 1 and len(attempts) == 1
    assert runner.ledger[-1]["fit_attempts"] == 0
    with pytest.raises(ValueError, match="retry"):
        run(data, runner)
    assert len(attempts) == 1 and runner.fit_intents == 1
    bad = list(data)
    bad[2] = bad[2].drop(columns="decision_at")
    with pytest.raises(ValueError):
        run(bad, runner)
    with pytest.raises(ValueError, match="call budget"):
        run(data, runner)
    assert runner.calls == 3


def test_successful_fit_can_predict_another_range_without_refitting():
    data = list(example())
    runner = LearningRunner(max_calls=2, max_fit_intents=1)
    first = run(data, runner)
    doc = data[0].to_dict()
    doc["predict_start"] = doc["predict_end"] = "2024-01-09"
    data[0] = LearningSpec.from_dict(doc)
    second = run(data, runner)
    assert second.receipt["fit_reused"] and len(second.predictions) == 4
    assert first.receipt["fit_id"] == second.receipt["fit_id"]


def test_prediction_failure_does_not_discard_successful_fit(monkeypatch):
    data = example()
    runner = LearningRunner(max_calls=2, max_fit_intents=1)
    from src.alpharesearch.models import TabularRegressor
    original = TabularRegressor.predict
    def fail(self, matrix):
        raise RuntimeError("synthetic prediction failure")
    monkeypatch.setattr(TabularRegressor, "predict", fail)
    with pytest.raises(RuntimeError, match="prediction failure"):
        run(data, runner)
    assert runner.fit_intents == 1
    monkeypatch.setattr(TabularRegressor, "predict", original)
    result = run(data, runner)
    assert result.receipt["fit_reused"]
    assert result.receipt["actual_fit_attempts_this_call"] == 0


def test_changed_mature_target_uses_new_fit_intent_and_budget_cannot_refund():
    data = example()
    runner = LearningRunner(max_calls=2, max_fit_intents=1)
    run(data, runner)
    data[-1].loc[0, "target_return"] += 0.01
    with pytest.raises(ValueError, match="fit intent budget"):
        run(data, runner)
    assert runner.fit_intents == 1


@pytest.mark.parametrize("limit", ["rows", "cells"])
def test_input_and_matrix_caps_reject_before_fit(limit):
    data = list(example())
    if limit == "rows":
        doc = data[0].to_dict()
        doc["max_input_rows"] = 10
        data[0] = LearningSpec.from_dict(doc)
    else:
        data[1] = model_spec("ridge", max_matrix_cells=1)
    runner = LearningRunner(max_calls=1, max_fit_intents=1)
    with pytest.raises(ValueError, match="cap|budget"):
        run(data, runner)
    assert runner.fit_intents == 0


def test_equal_day_weights_and_permutation_values_without_model_fit():
    from src.alpharesearch.learning import _prepare
    data = list(example())
    data[-1] = data[-1].drop(index=[0, 1])
    doc = data[0].to_dict()
    doc['weighting'] = 'equal_decision_date'
    data[0] = LearningSpec.from_dict(doc)
    X, y, weight, _, _, _ = _prepare(*data)
    sums = weight.groupby(level='trade_date').sum()
    np.testing.assert_allclose(sums, sums.iloc[0])
    assert weight.sum() == pytest.approx(len(X))
    doc['negative_control'] = 'within_date_label_permutation'
    data[0] = LearningSpec.from_dict(doc)
    _, shuffled, _, _, _, receipt = _prepare(*data)
    assert not shuffled.equals(y)
    for day in y.index.get_level_values('trade_date').unique():
        np.testing.assert_array_equal(np.sort(y.xs(day,level='trade_date')),
                                      np.sort(shuffled.xs(day,level='trade_date')))
    assert receipt['fit_identity']['settings']['negative_control'] == doc['negative_control']


def test_clock_content_changes_fit_identity_without_model_fit():
    from src.alpharesearch.learning import _prepare
    data = list(example())
    one = _prepare(*data)[-1]
    data[3].block.values.loc[0, 'observed_end'] = '2024-01-02T08:59:00+08:00'
    two = _prepare(*data)[-1]
    assert one['fit_id'] != two['fit_id']


@pytest.mark.parametrize("mutation", ["sample", "stock", "day"])
def test_conflicting_training_label_keys_are_not_treated_as_missing(mutation):
    from src.alpharesearch.learning import _prepare
    data = list(example())
    column, value = {"sample": ("sample_id", "wrong-sample"),
                     "stock": ("stock_code", "600999.SH"),
                     "day": ("trade_date", "2024-02-01")}[mutation]
    data[-1].loc[0, column] = value
    with pytest.raises(ValueError, match="identity|keys"):
        _prepare(*data)


def test_receipt_bindings_are_detached_and_prediction_clock_is_identified():
    from src.alpharesearch.learning import _prepare
    data = list(example())
    one = _prepare(*data)[-1]
    data[3].source_bindings[0]['source_version'] = 'mutated-later'
    assert one['full_assembly_source_bindings'][0]['source_version'] == 'synthetic-v1'
    pred = data[3].block.values.trade_date.eq('2024-01-08')
    data[3].block.values.loc[pred, 'observed_end'] = '2024-01-08T08:59:00+08:00'
    two = _prepare(*data)[-1]
    assert one['fit_id'] == two['fit_id']
    assert one['prediction_id'] != two['prediction_id']
