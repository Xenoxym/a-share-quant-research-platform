"""Engineering fixtures for single preparation; not market alpha evidence.

Prospective: apply alongside candidate sources after their own SOURCE review.
No test in this draft has been run yet. At most six tiny actual fits in this
new file; negative fixtures must reject before estimator fit. Legacy and worker
fixtures use independent frozen packages, not modified __file__ identities.
"""
import copy
from contextlib import contextmanager
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType
import uuid
import weakref

import numpy as np
import pandas as pd
import pytest

from src.alpharesearch import learning as module
from src.alpharesearch.learning import LearningRunner, LearningSpec
from src.alpharesearch.models import model_spec
from tests.test_alpha_learning import example


def weighted_example():
    data = list(example())
    cfg = data[0].to_dict()
    cfg["weighting"] = "equal_decision_date"
    data[0] = LearningSpec.from_dict(cfg)
    return data


def test_internal_consumption_prepares_once_and_keeps_mature_capture(monkeypatch):
    original = module._prepare
    calls = []
    def counted(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(module, "_prepare", counted)
    data = weighted_example()
    r = LearningRunner(max_calls=1, max_fit_intents=1)
    token = r._prepare_once(*data, capture_fit=True)
    receipt = r._prepared_receipt(token)
    assert len(calls) == 1 and r.fit_intents == 0
    assert receipt["mature_training_rows"] == 8
    result = r._consume_prepared(token)
    assert len(calls) == 1 and r.fit_intents == 1
    assert result.receipt["actual_fit_attempts_this_call"] == 1
    assert result.receipt["observed_but_immature_labels"] == 4
    assert len(result.fit_artifacts["X"]) == 8
    assert len(result.predictions) == 8
    assert r._prepared == {}
    with pytest.raises(ValueError, match="another runner or was consumed"):
        r._consume_prepared(token)
    assert r.fit_intents == 1


@pytest.mark.parametrize("mutation", ["X", "y", "weights", "P", "X_index", "clock", "receipt", "spec", "model", "capture"])
def test_changed_preparation_rejected_before_any_fit(monkeypatch, mutation):
    def forbidden(*args, **kwargs):
        raise AssertionError("Mutated preparation reached estimator fit")
    monkeypatch.setattr(module.TabularRegressor, "fit", forbidden)
    r = LearningRunner(max_calls=1, max_fit_intents=1)
    token = r._prepare_once(*weighted_example(), capture_fit=True)
    state = r._prepared[token]
    X, y, weights, P, decisions, receipt = state["prepared"]
    if mutation in ["X", "P"]:
        frame = X if mutation == "X" else P
        frame.iloc[0, 0] += 1.
    elif mutation in ["y", "weights"]:
        vector = y if mutation == "y" else weights
        vector.iloc[0] += 1.
    elif mutation == "X_index":
        X.index = X.index[::-1]
    elif mutation == "clock":
        decisions.loc[0, "decision_at"] += pd.Timedelta(seconds=1)
    elif mutation == "receipt":
        receipt["mature_training_rows"] += 1
    elif mutation == "spec":
        cfg = state["spec"].to_dict(); cfg["name"] = "Changed_scope"
        state["spec"] = LearningSpec.from_dict(cfg)
    elif mutation == "model":
        cfg = state["model"].to_dict(); cfg["params"]["alpha"] += 1.
        state["model"] = module.ModelSpec.from_dict(cfg)
    else:
        state["capture_fit"] = False
    with pytest.raises(ValueError, match="preparation changed before fit"):
        r._consume_prepared(token)
    assert r.fit_intents == 0 and r.ledger[0]["fit_attempts"] == 0
    assert r.ledger[0]["state"] == "failed" and r._prepared == {}
    with pytest.raises(ValueError, match="another runner or was consumed"):
        r._consume_prepared(token)


def test_fabricated_or_cross_owner_token_cannot_bypass_validation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Foreign token reached estimator fit")
    monkeypatch.setattr(module.TabularRegressor, "fit", forbidden)
    owner = LearningRunner(max_calls=1, max_fit_intents=1)
    stranger = LearningRunner(max_calls=1, max_fit_intents=1)
    token = owner._prepare_once(*example())
    for value in [object(), {}, module._PreparedLearning(), token]:
        with pytest.raises(ValueError, match="another runner or was consumed"):
            stranger._consume_prepared(value)
    assert owner.fit_intents == stranger.fit_intents == 0
    assert token in owner._prepared


def test_declared_call_budget_also_limits_unconsumed_preparations():
    r = LearningRunner(max_calls=1, max_fit_intents=1)
    token = r._prepare_once(*example())
    with pytest.raises(ValueError, match="call budget exhausted"):
        r._prepare_once(*example())
    assert len(r._prepared) == 1 and token in r._prepared
    assert r.calls == 1 and r.fit_intents == 0


@contextmanager
def frozen_module(code_root, relative_module):
    """Use the same verified-file importer as audit, without bypassing __file__."""
    from src.researchops.screen_output_audit import _FrozenFinder
    prefix = "_preparation_fixture_" + uuid.uuid4().hex
    package = ModuleType(prefix); package.__path__ = [str(code_root)]
    sys.modules[prefix] = package
    finder = _FrozenFinder(prefix, code_root); sys.meta_path.insert(0, finder)
    try:
        yield importlib.import_module(prefix + "." + relative_module)
    finally:
        sys.meta_path.remove(finder)
        for key in list(sys.modules):
            if key == prefix or key.startswith(prefix + "."):
                del sys.modules[key]


def uneven_weighted_example():
    data = weighted_example()
    sample = "2024-01-03-3"
    data[2] = data[2].loc[data[2].sample_id.ne(sample)].copy()
    for attr in ("values", "missing"):
        f = getattr(data[3].block, attr)
        setattr(data[3].block, attr, f.loc[f.sample_id.ne(sample)].copy())
    data[4] = data[4].loc[data[4].sample_id.ne(sample)].copy()
    return data


def legacy_data(legacy, data):
    """Convert types into the independent package; numerical frames stay equal."""
    registry_module = importlib.import_module(legacy.__package__ + ".registry")
    spec, model, decisions, assembly, labels = data
    block = legacy.FeatureBlock(assembly.block.values.copy(), assembly.block.missing.copy(),
        copy.deepcopy(assembly.block.units), copy.deepcopy(assembly.block.metadata))
    reg = registry_module.FeatureRegistry.from_dict(assembly.registry.to_dict())
    old_assembly = legacy.FeatureAssembly(block, reg, copy.deepcopy(assembly.source_bindings))
    return (legacy.LearningSpec.from_dict(spec.to_dict()), legacy.ModelSpec.from_dict(model.to_dict()),
            decisions.copy(), old_assembly, labels.copy())


def test_independent_legacy_preparation_and_scores_match(tmp_path):
    project = Path(__file__).resolve().parents[1]
    code = tmp_path / "legacy_code"; shutil.copytree(project / "src", code / "src")
    fixture = project / "tests/fixtures/learning_preparation_legacy_v2.py"
    assert hashlib.sha256(fixture.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == "1382f69e84b44ab01f3b6e7a81bd84cabf4ef7e3e6d135ffe22a07f3fa0d729b"
    shutil.copyfile(fixture, code / "src/alpharesearch/learning.py")
    data = uneven_weighted_example()
    with frozen_module(code, "src.alpharesearch.learning") as legacy:
        old_data = legacy_data(legacy, data)
        old_prepared = legacy._prepare(*old_data)
        runner = LearningRunner(max_calls=1, max_fit_intents=1)
        token = runner._prepare_once(*data, capture_fit=True)
        new_prepared = runner._prepared[token]["prepared"]
        for a, b in zip(old_prepared[:5], new_prepared[:5]):
            if a is None:
                assert b is None
            elif isinstance(a, pd.Series):
                pd.testing.assert_series_equal(a, b, check_exact=True)
            else:
                pd.testing.assert_frame_equal(a, b, check_exact=True)
        X, y, weights, P, clocks, receipt = new_prepared
        assert len(X) == 7 and len(P) == 8
        np.testing.assert_allclose(y.to_numpy(), [-.06, -.03, .03, .06, -.06, -.03, .03], rtol=0, atol=1e-15)
        np.testing.assert_allclose(weights.to_numpy(), [.875]*4 + [7/6]*3, rtol=0, atol=1e-15)
        assert weights.mean() == pytest.approx(1.)
        old = legacy.LearningRunner(max_calls=1, max_fit_intents=1).run(*old_data, capture_fit=True)
        new = runner._consume_prepared(token)
        pd.testing.assert_frame_equal(old.predictions.drop(columns=["fit_id", "prediction_id"]),
                                      new.predictions.drop(columns=["fit_id", "prediction_id"]), check_exact=True)
        for key in ("X", "y", "weights"):
            a, b = old.fit_artifacts[key], new.fit_artifacts[key]
            if isinstance(a, pd.DataFrame): pd.testing.assert_frame_equal(a, b, check_exact=True)
            else: pd.testing.assert_series_equal(a, b, check_exact=True)
        for key in ("mature_training_rows", "observed_but_immature_labels", "prediction_rows",
                    "prediction_membership", "prediction_missing", "prediction_inputs"):
            assert old.receipt[key] == new.receipt[key]
        # Implementation identity changes legitimately; it is not a score change.
        assert old.receipt["fit_identity"]["implementation_hashes"] != new.receipt["fit_identity"]["implementation_hashes"]
        assert old.receipt["fit_id"] != new.receipt["fit_id"]


@contextmanager
def native_worker_fixture(tmp_path, monkeypatch):
    from tests.test_alpha_learn_registration import setup
    from src.researchops.alpha_execution import THREAD_ENV
    import pyarrow as pa
    ops, session, proposal, _ = setup(tmp_path)
    ex = ops.register(session, proposal)
    job = ops.root / "worker_jobs" / ex["id"]
    assert not (job / "learn_result").exists()
    cfg = json.loads((job / "input.json").read_text(encoding="utf8"))
    for key in THREAD_ENV:
        monkeypatch.setenv(key, str(cfg["resources"]["library_threads"]))
    cpu, io = pa.cpu_count(), pa.io_thread_count()
    try:
        with frozen_module(job / "code", "src.researchops.learn_worker") as worker:
            learn = importlib.import_module(worker.__package__.rsplit(".", 1)[0] + ".alpharesearch.learning")
            assert Path(worker.__file__).resolve().parents[2] == (job / "code").resolve()
            yield worker, learn, job
    finally:
        pa.set_cpu_count(cpu); pa.set_io_thread_count(io)


def test_native_worker_prepares_once_and_releases_raw_before_fit(tmp_path, monkeypatch):
    with native_worker_fixture(tmp_path, monkeypatch) as (worker, learn, job):
        calls = {"prepare": 0, "fit": 0}; refs = []
        prepare, inspect, fit = learn._prepare, worker.inspect_inputs, learn.TabularRegressor.fit
        def count_prepare(*args, **kwargs):
            calls["prepare"] += 1
            return prepare(*args, **kwargs)
        def retain_only_weakrefs(*args, **kwargs):
            answer = inspect(*args, **kwargs)
            decisions, assembly, labels = answer[-1]["data"]
            refs.extend(weakref.ref(x) for x in (decisions, assembly, assembly.block.values, assembly.block.missing, labels))
            return answer
        def count_fit(adapter, *args, **kwargs):
            calls["fit"] += 1
            assert refs and all(ref() is None for ref in refs)
            return fit(adapter, *args, **kwargs)
        monkeypatch.setattr(learn, "_prepare", count_prepare)
        monkeypatch.setattr(worker, "inspect_inputs", retain_only_weakrefs)
        monkeypatch.setattr(learn.TabularRegressor, "fit", count_fit)
        folder = worker.compute(job)
        assert calls == {"prepare": 1, "fit": 1}
        attempts = json.loads((folder / "attempts.json").read_text(encoding="utf8"))
        assert attempts["ledger"][0]["fit_attempts"] == 1
        assert attempts["ledger"][0]["state"] == "succeeded"
        assert (folder / "manifest.json").is_file()
        # Weakrefs establish raw Python-object release, not a peak-RSS guarantee.


@pytest.mark.parametrize("failure", ["inspection", "manifest", "preparation"])
def test_native_prefit_rejection_persists_known_zero_fits(tmp_path, monkeypatch, failure):
    with native_worker_fixture(tmp_path, monkeypatch) as (worker, learn, job):
        calls = {"prepare": 0}; prepare = learn._prepare
        def count_prepare(*args, **kwargs):
            calls["prepare"] += 1
            if failure == "preparation": raise ValueError("Fixture preparation rejected")
            return prepare(*args, **kwargs)
        monkeypatch.setattr(learn, "_prepare", count_prepare)
        def forbidden(*args, **kwargs):
            raise AssertionError("Rejected inspection reached consume or fit")
        monkeypatch.setattr(learn.TabularRegressor, "fit", forbidden)
        monkeypatch.setattr(learn.LearningRunner, "_consume_prepared", forbidden)
        if failure == "inspection":
            original = worker.inspect_inputs
            def changed(*args, **kwargs):
                spec, model, registry, info = original(*args, **kwargs)
                info["mature_training_rows"] += 1
                return spec, model, registry, info
            monkeypatch.setattr(worker, "inspect_inputs", changed)
        elif failure == "manifest":
            original = worker.planned_attempts
            def changed(*args, **kwargs):
                plan = original(*args, **kwargs); plan["fit_intents"] += 1
                return plan
            monkeypatch.setattr(worker, "planned_attempts", changed)
        with pytest.raises(ValueError, match="scope changed|intent changed|Fixture preparation rejected"):
            worker.compute(job)
        evidence = json.loads((job / "learn_result/preparation_failure.json").read_text(encoding="utf8"))
        assert calls["prepare"] == evidence["preparation_calls"] == 1
        assert evidence["actual_fit_calls"] == evidence["fit_intents_used"] == 0
        assert len(evidence["ledger"]) == 1 and evidence["ledger"][0]["state"] == "failed"
        assert evidence["ledger"][0]["fit_attempts"] == 0
        assert not (job / "learn_result/manifest.json").exists()


def test_failed_fit_cannot_be_retried_with_fresh_token(monkeypatch):
    calls = []
    def fail(adapter, *args, **kwargs):
        adapter.fit_attempts += 1; calls.append(True)
        raise ValueError("Fixture estimator failed")
    monkeypatch.setattr(module.TabularRegressor, "fit", fail)
    runner = LearningRunner(max_calls=2, max_fit_intents=2)
    token = runner._prepare_once(*example())
    with pytest.raises(ValueError, match="Fixture estimator failed"):
        runner._consume_prepared(token)
    other = runner._prepare_once(*example())
    with pytest.raises(ValueError, match="implicit retry is forbidden"):
        runner._consume_prepared(other)
    assert calls == [True] and runner.fit_intents == 1
    assert [x["fit_attempts"] for x in runner.ledger] == [1, 0]


def test_out_of_order_tokens_keep_call_identity_and_share_successful_fit():
    runner = LearningRunner(max_calls=2, max_fit_intents=1)
    first = runner._prepare_once(*example())
    second = runner._prepare_once(*example())
    b = runner._consume_prepared(second)
    a = runner._consume_prepared(first)
    assert b.receipt["call"] == 2 and a.receipt["call"] == 1
    assert not b.receipt["fit_reused"] and a.receipt["fit_reused"]
    assert b.receipt["actual_fit_attempts_this_call"] == 1
    assert a.receipt["actual_fit_attempts_this_call"] == 0
    assert runner.fit_intents == 1 and runner._prepared == {}


def test_predict_failure_keeps_successful_fit_for_later_call(monkeypatch):
    original = module.TabularRegressor.predict
    attempts = []
    def once(adapter, *args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1: raise ValueError("Fixture prediction failed")
        return original(adapter, *args, **kwargs)
    monkeypatch.setattr(module.TabularRegressor, "predict", once)
    runner = LearningRunner(max_calls=2, max_fit_intents=1)
    with pytest.raises(ValueError, match="Fixture prediction failed"):
        runner.run(*example())
    result = runner.run(*example())
    assert result.receipt["fit_reused"] and result.receipt["actual_fit_attempts_this_call"] == 0
    assert runner.fit_intents == 1 and [x["fit_attempts"] for x in runner.ledger] == [1, 0]
