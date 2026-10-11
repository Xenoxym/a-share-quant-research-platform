"""Reuse a completed local learning audit by identity, without loading a model.

This verifies existing evidence; it does not reproduce scores or certify vendors.
Only persisted v2/v3 learning jobs with a successful supervised launch qualify.
"""
import json
import hashlib
from pathlib import Path
import re

from .alpha_experiments import verify_inputs
from .alpha_execution import _alive, _launch
from .alpha_worker import verify_frozen_job
from ..technical.artifacts import content_id, digest, verify_artifacts

MODE = "reuse_completed_identity_v1"
VERSIONS = {"registered-alpha-learn-v2": "v2", "registered-alpha-learn-v3": "v3"}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _ordinary_tree(folder):
    for path in (folder, *folder.parents, *folder.rglob("*")):
        _require(not path.is_symlink() and not (hasattr(path, "is_junction") and path.is_junction()),
                 "Linked learning evidence refused")


def verify_completed_learning_identity(research, experiment_id):
    """Bind DB, frozen source/inputs, saved audit, launch and all output bytes.

    The result is local provenance, not a new numerical audit. No checkpoint is
    deserialized; hashes may read its bytes. Database reads remain read-only.
    """
    _require(isinstance(experiment_id, str) and re.fullmatch(r"exp-[a-f0-9]{12}", experiment_id),
             "Registered learning identity required")
    ex = research.store.experiment(experiment_id)
    proposal, result = ex["proposal"], ex["result"]
    _require(ex["status"] == "completed" and proposal.get("kind") == "alpha_learn"
             and isinstance(result, dict) and result.get("verified") is True,
             "Completed audited registered learning required")
    version = VERSIONS.get(proposal["spec"].get("schema"))
    _require(version is not None, "Unsupported completed learning producer schema")
    job = research.root / "worker_jobs" / experiment_id
    _ordinary_tree(job)
    _require(not any((job / name).exists() for name in ("supervision_failure.json", "cancel_request.json")),
             "Cancellation or supervision failure invalidates score reuse")
    before = content_id(ex)
    cfg = verify_frozen_job(job, "alpha_learn")
    expected = dict(proposal, project=str(research.project), root=str(research.root),
                    task_id=ex["task_id"], experiment_id=experiment_id)
    _require(cfg == expected, "Completed producer differs from registered proposal")
    verify_inputs(research.project, proposal["inputs"], proposal["spec"]["budget"]["max_input_bytes"])
    hashes = {name: digest(job / name) for name in
              ("input.json", "code_manifest.json", "candidate_manifest.json", "alpha_audit.json",
               "receipt.json", "launch.json", "execution_stats.json")}
    for key, name in (("input_hash", "input.json"), ("audit_hash", "alpha_audit.json"),
                      ("receipt_hash", "receipt.json"), ("launch_hash", "launch.json"),
                      ("observation_hash", "execution_stats.json")):
        _require(result.get(key) == hashes[name], "Completed producer metadata changed: " + name)
    _require(result.get("engine_hash") == proposal["engine_hash"], "Producer engine identity differs")
    audit = _read(job / "alpha_audit.json")
    _require(audit.get("schema") == "registered-alpha-learn-audit-" + version
             and audit.get("verified") is True
             and audit.get("audit_id") == content_id({k: v for k, v in audit.items() if k != "audit_id"})
             and {k: result.get(k) for k in audit} == audit,
             "Saved numerical audit differs from completed database result")
    _require(audit.get("model_scores_independently_reproduced") is True
             and audit.get("additional_audit_fits") == 0
             and audit.get("audit_code_hash") == proposal["engine_inventory"]["src/researchops/learn_output_audit.py"],
             "Prior numerical reproduction and its exact frozen audit source required")
    launch = _launch(research, ex)
    observation = _read(job / "execution_stats.json")
    _require(launch is not None and not _alive(launch), "Completed producer process remains live")
    _require(observation.get("schema") == "alpha-worker-observation-v1"
             and observation.get("experiment_id") == experiment_id and observation.get("launch") == launch
             and type(observation.get("child_exit_code")) is int and observation["child_exit_code"] == 0
             and observation.get("parent_observed_to_exit") is True,
             "Successfully supervised producer terminal observation required")
    out = job / "learn_result"
    manifest = _read(out / "manifest.json")
    _require(manifest.get("schema") == "registered-alpha-learn-manifest-" + version
             and manifest.get("experiment_id") == experiment_id and manifest.get("task_id") == ex["task_id"]
             and manifest.get("input_hash") == result["input_hash"]
             and manifest.get("engine_hash") == proposal["engine_hash"]
             and manifest.get("candidate_manifest_hash") == hashes["candidate_manifest.json"],
             "Completed producer output manifest differs")
    names = manifest["artifacts"]
    for name in names:
        path = Path(name)
        _require(isinstance(name, str) and not path.is_absolute() and ".." not in path.parts
                 and "\\" not in name and path.as_posix() == name,
                 "Unsafe learning output artifact")
    _require({q.relative_to(out).as_posix() for q in out.rglob("*") if q.is_file()} == set(names) | {"manifest.json"},
             "Completed learning output file set differs")
    verify_artifacts(out, manifest)
    manifest_hash = digest(out / "manifest.json")
    _require(_read(job / "receipt.json") == dict(status="completed", kind="alpha_learn",
             experiment_id=experiment_id, result_manifest_hash=manifest_hash)
             and result.get("output_manifest_hash") == manifest_hash,
             "Completed producer receipt differs")
    learning = _read(out / "learning_receipt.json")
    _require(type(learning.get("actual_fit_attempts_this_call")) is int
             and learning["actual_fit_attempts_this_call"] == learning.get("fit_intents_used") == 1
             and type(result.get("model_fits")) is int and result["model_fits"] == 1
             and type(proposal.get("planned_model_fits")) is int and proposal["planned_model_fits"] == 1
             and type(proposal.get("planned_accounts")) is int and proposal["planned_accounts"] == 0
             and learning["fit_id"] == audit["fit_id"] and learning["prediction_id"] == audit["prediction_id"],
             "Original one-fit and saved prediction identities differ")
    # The original audit proved these times; repeat the cheap maturity boundary.
    import pandas as pd
    latest = pd.Timestamp(learning["latest_training_label_known_at"])
    cutoff = pd.Timestamp(learning["fit_identity"]["settings"]["fit_cutoff"])
    _require(latest.tzinfo is not None and cutoff.tzinfo is not None and latest <= cutoff,
             "Learning labels were not mature at fit cutoff")
    files = {}
    for name, sha in learning["fit_identity"]["implementation_hashes"].items():
        _require(name.startswith("src/") and ".." not in Path(name).parts and "\\" not in name,
                 "Unsafe declared producer source")
        raw = (job / "code" / name).read_bytes()
        text = raw.replace(b"\r\n", b"\n").decode("utf-8")
        _require(hashlib.sha256(text.encode("utf-8")).hexdigest() == sha,
                 "Producer declared implementation differs from its frozen source")
        files[name] = dict(sha256=sha, text=text)
    identity = dict(schema="completed-learning-identity-v1", verification_mode=MODE,
        experiment_id=experiment_id, producer_schema=proposal["spec"]["schema"],
        audit_id=audit["audit_id"], fit_id=learning["fit_id"], prediction_id=learning["prediction_id"],
        output_manifest_hash=manifest_hash, learning_receipt_sha256=names["learning_receipt.json"],
        predictions_sha256=names["predictions.parquet"], learning_receipt_id=content_id(learning),
        prediction_outputs=learning["prediction_outputs"], metadata_hashes=hashes,
        registered_experiment_id=before, source_environment=proposal["environment"], producer_files=files,
        additional_model_fits=0, additional_predict_calls=0, numeric_model_reproduction=False,
        scope="Identity reuse of actual completed local numerical audit; no new prediction, refit or vendor certification")
    # Detect changes while evidence was being inspected, without repeating numerics.
    _require(content_id(research.store.experiment(experiment_id)) == before,
             "Completed producer database record changed during reuse")
    _require(all(digest(job / name) == sha for name, sha in hashes.items())
             and digest(out / "manifest.json") == manifest_hash,
             "Producer metadata changed during reuse")
    verify_inputs(research.project, proposal["inputs"], proposal["spec"]["budget"]["max_input_bytes"])
    verify_artifacts(out, manifest)
    _require(verify_frozen_job(job, "alpha_learn") == cfg
             and not any((job / n).exists() for n in ("supervision_failure.json", "cancel_request.json")),
             "Producer source or supervision changed during reuse")
    return dict(identity, identity_id=content_id(identity))
