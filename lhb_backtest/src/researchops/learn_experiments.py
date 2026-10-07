"""Typed durable learning. Referenced assembly files are not upstream certification."""
import json
import importlib.metadata
import platform
import shutil
import time

import pandas as pd

from .alpha_worker import verify_frozen_job
from .resources import WorkerResources
from .store import text
from .alpha_experiments import verify_inputs
from .screen_experiments import ROLES, assembly_reference, same_json
from ..alpharesearch.assembly import FeatureAssembly
from ..alpharesearch.features.base import FeatureBlock
from ..alpharesearch.registry import FeatureRegistry
from ..alpharesearch.learning import LearningSpec, _prepare
from ..alpharesearch.models import ModelSpec
from ..technical.artifacts import content_id, digest, write_json


def protocol(value):
    fields = {"schema", "learning", "model", "budget"}
    if not isinstance(value, dict) or set(value) != fields or value["schema"] != "registered-alpha-learn-v1":
        raise ValueError("Strict registered learning protocol required")
    budget = value["budget"]
    if not isinstance(budget, dict) or set(budget) != {"max_input_bytes", "max_task_fit_intents"}:
        raise ValueError("Strict learning input/task fit budgets required")
    for key, cap in (("max_input_bytes", 100_000_000_000), ("max_task_fit_intents", 1000)):
        if type(budget[key]) is not int or not 1 <= budget[key] <= cap:
            raise ValueError("Bounded positive learning budget required: " + key)
    spec, model = LearningSpec.from_dict(value["learning"]), ModelSpec.from_dict(value["model"])
    return spec, model, dict(schema=value["schema"], learning=spec.to_dict(), model=model.to_dict(), budget=budget)


def inspect_inputs(project, proposal, *, include_data=False):
    spec, model, doc = protocol(proposal["spec"])
    roles = verify_inputs(project, proposal["inputs"], doc["budget"]["max_input_bytes"])
    if set(roles) != ROLES:
        raise ValueError("Exactly seven declared learning input roles required")
    registry = FeatureRegistry.from_dict(json.loads(roles["registry"].read_text(encoding="utf-8")))
    meta = json.loads(roles["assembly_definition"].read_text(encoding="utf-8")); units = meta.pop("units")
    bindings = meta.get("source_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("Explicit learning assembly source bindings required")
    block = FeatureBlock(pd.read_parquet(roles["assembly_values"]), pd.read_parquet(roles["assembly_missing"]), units, meta)
    assembly = FeatureAssembly(block, registry, tuple(bindings))
    hashes = {role: digest(roles[role]) for role in ("assembly_values", "assembly_missing", "assembly_definition")}
    if not same_json(json.loads(roles["assembly_receipt"].read_text(encoding="utf-8")), assembly_reference(assembly, hashes)):
        raise ValueError("Learning assembly reference differs from files/declarations")
    decisions, labels = pd.read_parquet(roles["decisions"]), pd.read_parquet(roles["labels"])
    prepared = _prepare(spec, model, decisions, assembly, labels)
    base = prepared[-1]
    info = {k: base[k] for k in ("fit_id", "prediction_id", "mature_training_rows", "prediction_rows")}
    info.update(source_files_by_reference=True, historical_source_certified=False,
        target_scope="mature training only; prediction labels ignored; metadata not vendor history")
    if include_data:
        info["data"] = (decisions, assembly, labels)
        info["prepared"] = prepared
    return spec, model, registry, info


def planned_attempts(spec, model):
    return dict(schema="planned-alpha-learn-attempts-v1", candidate_count=1, fit_intents=1, accounts=0,
        attempts=[dict(protocol_id=spec.protocol_id, model_id=model.model_id, status="planned")],
        counting_rule="One persisted fit intent per distinct job, including failure/cancel; no cross-job fitted cache")


def register(research, session, proposal):
    required = {"kind", "hypothesis", "expected_observation", "falsification", "spec", "inputs"}
    if (not isinstance(proposal, dict) or not required <= set(proposal)
            or set(proposal) - required - {"timeout_seconds", "resources"}
            or proposal["kind"] != "alpha_learn"):
        raise ValueError("Strict alpha_learn registration fields required")
    with research.store.connection() as db:
        research.store.owned(db, session)
    resources = WorkerResources.from_dict(proposal.get("resources", WorkerResources().to_dict())).to_dict()
    spec, model, registry, inspection = inspect_inputs(research.project, proposal)
    if model.to_dict()["threads"] > resources["library_threads"]:
        raise ValueError("Model threads exceed worker thread budget")
    _, _, doc = protocol(proposal["spec"])
    timeout = proposal.get("timeout_seconds", 900)
    if type(timeout) is not int or not 30 <= timeout <= 7200:
        raise ValueError("Bounded learning execution timeout required")
    sources = sorted(p for p in (research.project / "src").rglob("*.py") if "__pycache__" not in p.parts)
    if not (research.project / "src/researchops/learn_worker.py").is_file():
        raise ValueError("Missing learning worker implementation")
    for source in sources:
        if not source.resolve().is_relative_to(research.project):
            raise ValueError("Source implementation escapes project")
        for item in (source, *source.parents):
            if item == research.project:
                break
            if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
                raise ValueError("Linked learning source refused")
    inventory = {p.relative_to(research.project).as_posix(): digest(p) for p in sources}
    environment = dict(python=platform.python_version(), packages={n: importlib.metadata.version(n)
        for n in ("numpy", "pandas", "pyarrow", "tzdata", "psutil", "scikit-learn", "scipy", "joblib", "threadpoolctl")})
    p = {k: text(proposal[k], k) for k in ("hypothesis", "expected_observation", "falsification")}
    p.update(kind="alpha_learn", spec=doc, inputs=proposal["inputs"], timeout_seconds=timeout,
        resources=resources, registry_version=registry.version_id, engine_inventory=inventory,
        engine_hash=content_id(inventory), environment=environment, inspection=inspection,
        candidate_count=1, planned_model_fits=1, planned_accounts=0,
        submitted_by=session["worker"], evaluation_scope="retrospective_time_split")
    canonical = json.loads(json.dumps(p)); canonical["spec"]["learning"].pop("name"); canonical["spec"]["model"].pop("name")
    for key in ("hypothesis", "expected_observation", "falsification", "timeout_seconds", "submitted_by"):
        canonical.pop(key)
    canonical["inputs"].sort(key=lambda x: x["role"])
    ex = research.store.register(session, p, content_id(canonical),
        fit_budget=doc["budget"]["max_task_fit_intents"])
    job = research.root / "worker_jobs" / ex["id"]
    if (job / "ready.json").is_file():
        verify_registered(research, ex)
        return ex
    if ex["status"] != "planned" or job.exists():
        raise ValueError("Previous learning freeze incomplete or failed; preserve failed try")
    p = ex["proposal"]; code = job / "code"
    try:
        code.mkdir(parents=True)
        for name, sha in p["engine_inventory"].items():
            target = code / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(research.project / name, target)
            if digest(target) != sha:
                raise ValueError("Source changed during learning freeze")
        write_json(job / "code_manifest.json", dict(artifacts=p["engine_inventory"]))
        write_json(job / "input.json", dict(p, project=str(research.project), root=str(research.root),
            task_id=ex["task_id"], experiment_id=ex["id"]))
        write_json(job / "candidate_manifest.json", planned_attempts(LearningSpec.from_dict(p["spec"]["learning"]), ModelSpec.from_dict(p["spec"]["model"])))
        verify_inputs(research.project, p["inputs"], p["spec"]["budget"]["max_input_bytes"])
        write_json(job / "ready.json", dict(kind="alpha_learn", input_hash=digest(job / "input.json"),
            code_manifest_hash=digest(job / "code_manifest.json"),
            candidate_manifest_hash=digest(job / "candidate_manifest.json")))
    except Exception as exc:
        with research.store.connection(True) as db:
            db.execute("UPDATE experiments SET status='failed',error=?,updated=? WHERE id=?",
                (str(exc), time.time(), ex["id"]))
            research.store.event(db, ex["task_id"], session["worker"], "freeze_failed",
                dict(experiment_id=ex["id"], error=str(exc)))
        raise
    return ex


def verify_registered(research, experiment):
    p = experiment["proposal"]
    if p.get("kind") != "alpha_learn":
        raise ValueError("Not a registered learning job")
    job = research.root / "worker_jobs" / experiment["id"]
    cfg = verify_frozen_job(job, "alpha_learn")
    expected = dict(p, project=str(research.project), root=str(research.root),
        task_id=experiment["task_id"], experiment_id=experiment["id"])
    if not same_json(cfg, expected):
        raise ValueError("Frozen learning job differs from database registration")
    spec, model, doc = protocol(p["spec"])
    roles = verify_inputs(research.project, p["inputs"], doc["budget"]["max_input_bytes"])
    if set(roles) != ROLES:
        raise ValueError("Learn input roles changed")
    registry = FeatureRegistry.from_dict(json.loads(roles["registry"].read_text(encoding="utf-8")))
    plan = json.loads((job / "candidate_manifest.json").read_text(encoding="utf-8"))
    if (not same_json(plan, planned_attempts(spec, model)) or p["candidate_count"] != 1
            or p["registry_version"] != registry.version_id
            or p["planned_model_fits"] != 1 or p["planned_accounts"] != 0):
        raise ValueError("Learn registry/candidate plan changed")
    return dict(verified=True, kind="alpha_learn_registration", candidate_count=p["candidate_count"],
        executed_candidates=0, fits=0, planned_fit_intents=1, accounts=0,
        scope="registered input/source/plan identity; not upstream algorithm or strategy validity")
