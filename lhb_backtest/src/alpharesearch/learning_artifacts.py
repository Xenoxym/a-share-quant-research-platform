"""Actual local fitted artifacts for the proposed durable learning v2 worker.

Private candidate: not integrated or executed yet. Local checkpoints only;
verification reproduces predictions without another training attempt.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .learning import KEYS, _fingerprint
from .checkpoints import save_checkpoint, load_checkpoint
from ..technical.artifacts import digest, content_id, write_json

def _same_json(left, right):
    """JSON equality preserving scalar types; reject nonfinite JSON."""
    options = dict(sort_keys=True, separators=(",", ":"), allow_nan=False)
    return json.dumps(left, **options) == json.dumps(right, **options)


FILES = {"training_trace.parquet", "training_artifact_receipt.json",
         "model_checkpoint/model.joblib", "model_checkpoint/manifest.json"}


def _weights(X, weights):
    if weights is None:
        return np.ones(len(X), dtype=float)
    if not weights.index.equals(X.index):
        raise ValueError("Actual training weights and feature membership differ")
    return weights.to_numpy(dtype=float)


def _trace(X, y, weights):
    if not X.index.equals(y.index) or list(X.index.names) != KEYS:
        raise ValueError("Actual target/feature membership differs")
    if set(X.columns) & set(KEYS + ["target_return_used", "weight_used", "train_score"]):
        raise ValueError("Feature names overlap persistence fields")
    trace = X.reset_index()
    trace["target_return_used"] = y.to_numpy(dtype=float)
    trace["weight_used"] = _weights(X, weights)
    if (not np.isfinite(trace[["target_return_used", "weight_used"]].to_numpy()).all()
            or not trace.weight_used.gt(0).all()):
        raise ValueError("Actual targets/weights must be finite and weights positive")
    return trace


def _context(trace, receipt):
    names = receipt["retained_features"]
    settings = receipt["fit_identity"]["settings"]
    return dict(fit_id=receipt["fit_id"], fit_cutoff=settings["fit_cutoff"],
        training_inputs_sha256=_fingerprint(trace[KEYS + names])["sha256"],
        training_targets_sha256=_fingerprint(trace[KEYS + ["target_return_used"]])["sha256"],
        training_weights_sha256=_fingerprint(trace[KEYS + ["weight_used"]])["sha256"],
        target_spec_id=content_id(dict(schema="used-learning-target-v1",
            target=settings["target"], negative_control=settings["negative_control"],
            control_seed=settings["control_seed"])))


def _receipt(trace, context, checkpoint, receipt):
    return dict(schema="actual-learning-artifacts-v1", fit_id=receipt["fit_id"],
        rows=len(trace), feature_names=receipt["retained_features"],
        actual_training_trace=_fingerprint(trace), training_context=context,
        checkpoint=checkpoint, additional_model_fits=0,
        train_scores_saved=True, actual_used_targets_saved=True,
        effective_row_weights_saved=True, preprocessing_saved=True,
        row_weight_passed_to_regressor=receipt["fitted_model"]["sample_weight_used"],
        account_results=False, scope="actual local fitted pipeline and used training values")


def save_learning_artifacts(result, folder, *, max_checkpoint_bytes):
    artifact = result.fit_artifacts
    if not isinstance(artifact, dict) or set(artifact) != {"fit_id", "adapter", "X", "y", "weights"}:
        raise ValueError("Captured actual fitted artifact required")
    if artifact["fit_id"] != result.receipt["fit_id"]:
        raise ValueError("Captured adapter belongs to another fit")
    folder = Path(folder)
    if any((folder/name).exists() for name in FILES):
        raise ValueError("Persistence evidence already exists; preserve partial output")
    X, y, weights = artifact["X"], artifact["y"], artifact["weights"]
    adapter = artifact["adapter"]
    if not _same_json(adapter.metadata(), result.receipt["fitted_model"]):
        raise ValueError("Captured actual model metadata differs")
    trace = _trace(X, y, weights)
    trace["train_score"] = adapter.predict(X).to_numpy()
    if not np.isfinite(trace.train_score.to_numpy()).all():
        raise ValueError("Training predictions must be finite")
    trace.to_parquet(folder/"training_trace.parquet", index=False)
    context = _context(trace, result.receipt)
    checkpoint = save_checkpoint(adapter, folder/"model_checkpoint", context,
                                 max_bytes=max_checkpoint_bytes)
    receipt = _receipt(trace, context, checkpoint, result.receipt)
    write_json(folder/"training_artifact_receipt.json", receipt)
    return receipt


def audit_learning_artifacts(folder, X, y, weights, P, predictions, receipt, *, max_checkpoint_bytes, trusted_local=False):
    if trusted_local is not True:
        raise ValueError("Explicit trusted_local=True required for caller-generated local checkpoint")
    folder = Path(folder)
    actual = pd.read_parquet(folder/"training_trace.parquet")
    expected = _trace(X, y, weights)
    if list(actual.columns) != list(expected.columns) + ["train_score"]:
        raise ValueError("Strict actual training trace fields differ")
    pd.testing.assert_frame_equal(actual[list(expected)], expected, check_exact=True)
    context = _context(expected, receipt)
    artifact = json.loads((folder/"training_artifact_receipt.json").read_text(encoding="utf8"))
    checkpoint = artifact["checkpoint"]
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"manifest_sha256", "model_sha256", "model_bytes"}:
        raise ValueError("Strict checkpoint receipt fields required")
    if type(checkpoint["model_bytes"]) is not int or checkpoint["model_bytes"] < 1:
        raise ValueError("Checkpoint model bytes must be a positive integer")
    expected_receipt = _receipt(actual, context, checkpoint, receipt)
    if not _same_json(artifact, expected_receipt):
        raise ValueError("Strict actual artifact receipt differs")
    if checkpoint["model_sha256"] != digest(folder/"model_checkpoint/model.joblib") or checkpoint["model_bytes"] != (folder/"model_checkpoint/model.joblib").stat().st_size:
        raise ValueError("Actual checkpoint byte identity differs")
    if checkpoint["manifest_sha256"] != digest(folder/"model_checkpoint/manifest.json"):
        raise ValueError("Local checkpoint manifest differs")
    modeldoc = json.loads((folder/"model_checkpoint/manifest.json").read_text(encoding="utf8"))
    if not _same_json(modeldoc["training_context"], context):
        raise ValueError("Checkpoint actual training context differs")
    loaded = load_checkpoint(folder/"model_checkpoint",
        expected_manifest_sha256=checkpoint["manifest_sha256"], trusted_local=trusted_local,
        max_bytes=max_checkpoint_bytes)
    if not _same_json(loaded.metadata(), receipt["fitted_model"]):
        raise ValueError("Reloaded actual fitted metadata differs")
    if not np.array_equal(loaded.predict(X).to_numpy(), actual.train_score.to_numpy()):
        raise ValueError("Reloaded model does not reproduce saved training scores")
    if not np.array_equal(loaded.predict(P).to_numpy(), predictions.score.to_numpy()):
        raise ValueError("Reloaded model does not reproduce saved forecast scores")
    if type(loaded.fit_attempts) is not int or loaded.fit_attempts != 0:
        raise ValueError("Unexpected fit occurred while auditing artifacts")
    return dict(additional_audit_fits=0, model_scores_independently_reproduced=True,
                training_values_independently_reconstructed=True,
                scope="reload caller-generated local checkpoint; not vendor data certification")
