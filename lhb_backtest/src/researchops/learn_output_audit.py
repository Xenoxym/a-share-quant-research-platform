"""Frozen input/output/transform audit without any model refit or score reproduction."""
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
import uuid

import numpy as np
import pandas as pd

from .alpha_worker import verify_frozen_job
from .alpha_experiments import verify_inputs
from .learn_experiments import inspect_inputs, planned_attempts, verify_registered, same_json
from .learn_worker import FILES
from .resources import WorkerResources
from .screen_output_audit import _FrozenFinder
from ..alpharesearch.learning import _fingerprint, _environment
from ..alpharesearch.models import LINEAR
from ..alpharesearch.registry import implementation_hashes
from ..technical.artifacts import content_id, digest, verify_artifacts


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _vector(given, expected, name):
    if (not isinstance(given, list) or any(type(x) is not float for x in given)
            or len(given) != len(expected) or not np.isfinite(given).all()
            or not np.allclose(given, expected, rtol=1e-10, atol=1e-12)):
        raise ValueError("Training-only transform metadata differs: " + name)


def _model_metadata(given, model, X, weights):
    cfg = model.to_dict()
    expected = dict(schema="fitted-tabular-model-v1", model_id=model.model_id,
        spec=cfg, feature_names=list(X.columns), training_rows=len(X), transformed_features=2*len(X.columns),
        preprocessing="train median plus missing flags for every feature",
        scaling="train-only mean/std" if cfg["family"] in LINEAR else "none",
        preprocessing_weighting="unweighted; optional weights apply to regressor",
        sample_weight_used=weights is not None, early_stopping=False, forest_jobs=1,
        initialization_attempts=1, fit_attempts=1,
        implementation_hashes=implementation_hashes(["src/alpharesearch/models.py"]), environment=_environment(),
        scope="numeric adapter only; E16 must validate features and mature labels", account_results=False)
    arrays = {"imputation_medians"}
    if cfg["family"] in LINEAR:
        arrays |= {"scaler_mean", "scaler_scale"}
    if (not isinstance(given, dict) or set(given) != set(expected)|arrays
            or not same_json({k:given[k] for k in expected}, expected)):
        raise ValueError("Fitted model metadata declaration differs")
    raw = X.to_numpy(dtype=float); medians = np.nanmedian(raw, axis=0)
    _vector(given["imputation_medians"], medians, "median")
    if cfg["family"] in LINEAR:
        values = np.column_stack((np.where(np.isnan(raw), medians, raw), np.isnan(raw)))
        mean, variance = values.mean(axis=0), values.var(axis=0)
        eps = np.finfo(float).eps; n = len(X)
        constant = variance <= n*eps*variance + (n*mean*eps)**2
        scale = np.sqrt(variance); scale[constant] = 1.
        _vector(given["scaler_mean"], mean, "mean")
        _vector(given["scaler_scale"], scale, "scale")


def audit_frozen(cfg, job):
    if Path(__file__).resolve().parents[2] != (job / "code").resolve():
        raise ValueError("Learning audit must execute registered frozen source")
    folder = job / "learn_result"; paths = [folder, *folder.rglob("*")]
    if any(p.is_symlink() or (hasattr(p,"is_junction") and p.is_junction()) for p in paths):
        raise ValueError("Linked learning output refused")
    manifest_hash = digest(folder / "manifest.json"); manifest = _read(folder / "manifest.json")
    header = dict(schema="registered-alpha-learn-manifest-v1", experiment_id=job.name,
        task_id=cfg["task_id"], input_hash=digest(job / "input.json"), engine_hash=cfg["engine_hash"],
        candidate_manifest_hash=digest(job / "candidate_manifest.json"))
    if (set(manifest) != set(header)|{"artifacts"} or not same_json({k:manifest.get(k) for k in header},header)
            or set(manifest["artifacts"]) != FILES
            or {p.relative_to(folder).as_posix() for p in paths[1:] if p.is_file()} != FILES|{"manifest.json"}):
        raise ValueError("Strict learning output file set/manifest differs")
    verify_artifacts(folder, manifest)
    if sum(p.stat().st_size for p in paths if p.is_file()) > WorkerResources.from_dict(cfg["resources"]).max_output_bytes:
        raise ValueError("Learning output exceeds byte budget")
    spec, model, _, info = inspect_inputs(cfg["project"], cfg, include_data=True)
    info.pop("data"); X, _, weights, _, decision_rows, base = info.pop("prepared")
    if not same_json(info, cfg["inspection"]):
        raise ValueError("Frozen learning input scope differs")
    predictions = pd.read_parquet(folder / "predictions.parquet")
    expected_columns = list(decision_rows.columns)+["score", "fit_id", "prediction_id"]
    if (list(predictions.columns) != expected_columns
            or not same_json(_fingerprint(predictions[list(decision_rows.columns)]), _fingerprint(decision_rows))
            or not pd.api.types.is_float_dtype(predictions.score.dtype)
            or not np.isfinite(predictions.score.to_numpy()).all()
            or not predictions.fit_id.eq(base["fit_id"]).all()
            or not predictions.prediction_id.eq(base["prediction_id"]).all()):
        raise ValueError("Prediction keys/clocks/finite scores or identities differ")
    receipt = _read(folder / "learning_receipt.json")
    extra = dict(fit_reused=False, call=1, fit_intents_used=1, actual_fit_attempts_this_call=1,
        prediction_outputs=_fingerprint(predictions))
    if (set(receipt) != set(base)|set(extra)|{"fitted_model"}
            or not same_json({k:receipt.get(k) for k in base},base)
            or not same_json({k:receipt.get(k) for k in extra},extra)):
        raise ValueError("Learning receipt inputs/one-fit/output identity differ")
    metadata = _read(folder / "model_metadata.json")
    if not same_json(metadata, receipt["fitted_model"]):
        raise ValueError("Fitted metadata files differ")
    _model_metadata(metadata, model, X, weights)
    plan = planned_attempts(spec, model)
    expected_attempts = dict(plan, attempts=[dict(plan["attempts"][0], status="succeeded")],
        ledger=[dict(call=1,state="succeeded",fit_attempts=1,fit_id=base["fit_id"],new_fit_intent=1)])
    if not same_json(_read(folder / "attempts.json"), expected_attempts):
        raise ValueError("Learning attempt ledger differs")
    verify_inputs(cfg["project"], cfg["inputs"], cfg["spec"]["budget"]["max_input_bytes"])
    if not same_json(verify_frozen_job(job,"alpha_learn"), cfg) or digest(folder/"manifest.json") != manifest_hash:
        raise ValueError("Learning identities changed during audit")
    verify_artifacts(folder, manifest)
    value = dict(schema="registered-alpha-learn-audit-v1", verified=True, experiment_id=job.name,
        task_id=cfg["task_id"], candidate_count=1, model_fits=1, fit_intents=1, additional_audit_fits=0, accounts=0,
        input_hash=digest(job/"input.json"), output_manifest_hash=manifest_hash, engine_hash=cfg["engine_hash"],
        audit_code_hash=cfg["engine_inventory"]["src/researchops/learn_output_audit.py"],
        fit_id=base["fit_id"], prediction_id=base["prediction_id"], prediction_rows=base["prediction_rows"],
        replay_scope="frozen causal preparation, keys/clocks/finite scores, transform metadata and hashes only; no model refit",
        model_scores_independently_reproduced=False, vendor_history_certified=False, account_results=False)
    return dict(value, audit_id=content_id(value))


def audit_output(research, experiment):
    verify_registered(research, experiment)
    job = research.root/"worker_jobs"/experiment["id"]; cfg = verify_frozen_job(job,"alpha_learn")
    prefix = "_frozen_learn_audit_"+uuid.uuid4().hex
    package = ModuleType(prefix); package.__path__ = [str(job/"code")]; sys.modules[prefix] = package
    finder = _FrozenFinder(prefix, job/"code"); sys.meta_path.insert(0,finder)
    try:
        module = importlib.import_module(prefix+".src.researchops.learn_output_audit")
        return module.audit_frozen(cfg,job)
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix+"."):
                del sys.modules[name]
