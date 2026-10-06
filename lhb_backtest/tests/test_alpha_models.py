"""Model data-boundary counterexamples and a learnable synthetic signal."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.alpharesearch.models import DEFAULTS, ModelSpec, TabularRegressor, model_spec


def sample(n=360):
    rng = np.random.default_rng(901)
    x = pd.DataFrame(rng.normal(size=(n, 3)), columns=["price_move", "volume", "event"])
    y = pd.Series(0.03 * x.price_move - 0.02 * x.volume, index=x.index)
    x.loc[x.index[::9], "event"] = np.nan
    return x, y


@pytest.mark.parametrize("family", list(DEFAULTS))
def test_each_family_learns_and_preserves_prediction_keys(family):
    x, y = sample()
    params = {"min_samples_leaf": 4} if family not in {"ridge", "elastic_net", "huber"} else {}
    adapter = TabularRegressor(model_spec(family, params=params))
    adapter.fit(x.iloc[:260], y.iloc[:260])
    score = adapter.predict(x.iloc[260:])
    assert score.index.equals(x.iloc[260:].index) and score.notna().all()
    mse = np.mean((score - y.iloc[260:]) ** 2)
    reference = np.mean((y.iloc[:260].mean() - y.iloc[260:]) ** 2)
    assert mse < reference * 0.5
    meta = adapter.metadata()
    assert meta["transformed_features"] == 6
    assert meta["fit_attempts"] == adapter.fit_attempts == adapter.successful_fits == 1
    assert meta["account_results"] is False


def test_prediction_outliers_and_new_missingness_do_not_fit_preprocessing():
    x, y = sample(60)
    adapter = TabularRegressor(model_spec("ridge")).fit(x.iloc[:40], y.iloc[:40])
    before = adapter.metadata()
    imputer = dict(adapter._model.named_steps["inputs"].transformer_list)["values"]
    actual_medians = imputer.statistics_.copy()
    actual_scale = adapter._model.named_steps["scale"].mean_.copy()
    predicted = x.iloc[40:].copy()
    predicted["price_move"] = 1000000.0
    predicted["volume"] = np.nan
    adapter.predict(predicted)
    assert adapter.metadata() == before
    np.testing.assert_array_equal(imputer.statistics_, actual_medians)
    np.testing.assert_array_equal(adapter._model.named_steps["scale"].mean_, actual_scale)
    np.testing.assert_allclose(before["imputation_medians"], np.nanmedian(x.iloc[:40], axis=0))
    # Flags exist even for columns without training NaNs.
    assert before["transformed_features"] == 2 * x.shape[1]


@pytest.mark.parametrize("mutation", ["reorder", "rename", "extra", "missing"])
def test_wrong_prediction_columns_are_rejected(mutation):
    x, y = sample(40)
    adapter = TabularRegressor(model_spec("ridge")).fit(x, y)
    other = x.copy()
    if mutation == "reorder":
        other = other[other.columns[::-1]]
    elif mutation == "rename":
        other = other.rename(columns={"event": "future_return"})
    elif mutation == "extra":
        other["unknown"] = 1.0
    else:
        other = other.drop(columns="event")
    with pytest.raises(ValueError, match="names/order"):
        adapter.predict(other)


@pytest.mark.parametrize("mutation", ["all_missing", "inf", "object", "complex", "duplicate_column", "duplicate_index"])
def test_invalid_training_matrix_is_rejected_before_fit(mutation):
    x, y = sample(30)
    if mutation == "all_missing":
        x["event"] = np.nan
    elif mutation == "inf":
        x.loc[0, "event"] = np.inf
    elif mutation == "object":
        x["event"] = "1.0"
    elif mutation == "complex":
        x["event"] = 1 + 2j
    elif mutation == "duplicate_column":
        x.columns = ["price_move", "volume", "volume"]
    else:
        x.index = [0] * len(x)
    adapter = TabularRegressor(model_spec("ridge"))
    with pytest.raises(ValueError):
        adapter.fit(x, y)
    assert adapter.fit_attempts == 0 and adapter.state == "created"


@pytest.mark.parametrize("bad_y", ["wrong_order", "missing", "infinite"])
def test_bad_target_is_rejected(bad_y):
    x, y = sample(30)
    if bad_y == "wrong_order":
        y = y.iloc[::-1]
    elif bad_y == "missing":
        y.iloc[0] = np.nan
    else:
        y.iloc[0] = np.inf
    with pytest.raises(ValueError):
        TabularRegressor(model_spec("ridge")).fit(x, y)


def test_not_fitted_and_refitting_are_rejected():
    x, y = sample(30)
    adapter = TabularRegressor(model_spec("ridge"))
    with pytest.raises(ValueError, match="successfully fitted"):
        adapter.predict(x)
    adapter.fit(x, y)
    with pytest.raises(ValueError, match="refitted"):
        adapter.fit(x, y)


def test_model_identity_ignores_display_name_and_explicit_default_parameter():
    one = model_spec("ridge", name="first")
    two = model_spec("ridge", name="second", params={"alpha": 10.0})
    assert one.model_id == two.model_id
    assert model_spec("ridge", params={"alpha": 1}).model_id != one.model_id
    assert ModelSpec.from_dict(one.to_dict()).to_dict() == one.to_dict()


@pytest.mark.parametrize("field,value", [
    ("family", "eval"), ("threads", 3), ("seed", True), ("max_features", 0),
    ("params", {"early_stopping": True}), ("params", {"alpha": float("nan")}),
    ("params", {"max_iter": True}),
])
def test_bad_model_configuration_is_rejected(field, value):
    doc = copy.deepcopy(model_spec("ridge").to_dict())
    doc[field] = value
    with pytest.raises(ValueError):
        ModelSpec.from_dict(doc)


def test_directly_constructed_spec_does_not_bypass_validation():
    with pytest.raises(ValueError):
        TabularRegressor(ModelSpec('{"family":"eval"}'))


def test_matrix_and_row_budgets():
    x, y = sample(30)
    for kwargs in ({"max_train_rows": 20}, {"max_features": 2}, {"max_matrix_cells": 20}):
        with pytest.raises(ValueError, match="budget"):
            TabularRegressor(model_spec("ridge", **kwargs)).fit(x, y)


def test_convergence_failure_is_counted_and_cannot_retry():
    x, y = sample(100)
    model = TabularRegressor(model_spec("huber", params={"max_iter": 1, "tol": 1e-10}))
    with pytest.raises(Exception):
        model.fit(x, y)
    assert model.state == "failed" and model.fit_attempts == 1 and model.successful_fits == 0
    with pytest.raises(ValueError, match="retried"):
        model.fit(x, y)


def test_positive_weights_apply_and_misalignment_is_rejected():
    x, y = sample(30)
    w = pd.Series(np.linspace(1, 2, len(x)), index=x.index)
    adapter = TabularRegressor(model_spec("ridge")).fit(x, y, sample_weight=w)
    assert adapter.metadata()["sample_weight_used"]
    for invalid in (w.iloc[::-1], w * 0):
        with pytest.raises(ValueError):
            TabularRegressor(model_spec("ridge")).fit(x, y, sample_weight=invalid)


def test_histogram_model_has_no_hidden_random_validation_split():
    x, y = sample(110)
    adapter = TabularRegressor(model_spec("hist_gbdt", params={"max_iter": 5})).fit(x, y)
    estimator = adapter._model.named_steps["regressor"]
    assert estimator.early_stopping is False and estimator.do_early_stopping_ is False
    assert estimator.n_iter_ == 5


def test_composite_sample_index_is_preserved_and_misaligned_target_rejected():
    x,y=sample(40)
    idx=pd.MultiIndex.from_arrays([["2024-01-02"]*20+["2024-01-03"]*20,
                                   [f"{i:06d}.SH" for i in range(20)]*2],
                                  names=["trade_date","stock_code"])
    x.index=idx;y.index=idx
    adapter=TabularRegressor(model_spec("ridge")).fit(x,y)
    assert adapter.predict(x).index.equals(idx)
    with pytest.raises(ValueError,match="index/order"):
        TabularRegressor(model_spec("ridge")).fit(x,y.iloc[::-1])


@pytest.mark.parametrize("bad_key",["missing","duplicate"])
def test_invalid_composite_keys_rejected(bad_key):
    x,y=sample(30)
    keys=[("2024-01-02",f"{i:06d}.SH") for i in range(30)]
    keys[1]=(None,"000001.SH") if bad_key=="missing" else keys[0]
    x.index=pd.MultiIndex.from_tuples(keys);y.index=x.index
    with pytest.raises(ValueError,match="index"):
        TabularRegressor(model_spec("ridge")).fit(x,y)


def test_continuous_parameters_have_canonical_identity_and_fraction_semantics():
    assert model_spec("ridge",params={"alpha":10}).model_id==model_spec("ridge").model_id
    from src.alpharesearch.models import _pipeline
    one=model_spec("random_forest",params={"max_features":1})
    two=model_spec("random_forest",params={"max_features":1.0})
    assert one.model_id==two.model_id
    value=_pipeline(one).named_steps["regressor"].max_features
    assert type(value) is float and value==1.0


def test_initialization_failure_is_terminal_but_not_a_fake_fit(monkeypatch):
    from src.alpharesearch import models
    x,y=sample(30)
    def fail(spec):
        raise RuntimeError("fixture initialization failure")
    monkeypatch.setattr(models,"_pipeline",fail)
    adapter=TabularRegressor(model_spec("ridge"))
    with pytest.raises(RuntimeError,match="fixture"):adapter.fit(x,y)
    assert adapter.state=="failed" and adapter.fit_attempts==0
    assert adapter.initialization_attempts==1 and "fixture" in adapter.error
    with pytest.raises(ValueError,match="retried"):adapter.fit(x,y)
