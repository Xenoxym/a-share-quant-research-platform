"""Typed durable screening; reference integrity does not certify upstream algorithms."""
import importlib.metadata
import json
import platform
import shutil
import time

import pandas as pd

from .alpha_experiments import verify_inputs
from .alpha_worker import verify_frozen_job
from .resources import WorkerResources
from .store import text
from ..alpharesearch.assembly import FeatureAssembly
from ..alpharesearch.features.base import FeatureBlock
from ..alpharesearch.registry import FeatureRegistry
from ..alpharesearch.screening import ScreenSpec, _prepare
from ..technical.artifacts import content_id, digest, write_json

ROLES = {"registry", "assembly_values", "assembly_missing", "assembly_definition",
         "assembly_receipt", "decisions", "labels"}


def same_json(left, right):
    """Preserve JSON scalar types, including bool vs numbers and int vs float."""
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return canonical(left) == canonical(right)


def protocol(value):
    if (not isinstance(value, dict) or set(value) != {"schema", "screen", "budget"}
            or value["schema"] != "registered-alpha-screen-v1"):
        raise ValueError("Strict registered screening protocol required")
    budget = value["budget"]
    if not isinstance(budget, dict) or set(budget) != {"max_input_bytes", "max_task_channel_intents"}:
        raise ValueError("Strict screening input/task budgets required")
    for key, cap in (("max_input_bytes", 100_000_000_000), ("max_task_channel_intents", 100000)):
        if type(budget[key]) is not int or not 1 <= budget[key] <= cap:
            raise ValueError("Bounded positive screening budget required: " + key)
    spec = ScreenSpec.from_dict(value["screen"])
    if len(spec.to_dict()["channels"]) > budget["max_task_channel_intents"]:
        raise ValueError("Screen channel intents exceed task budget")
    return spec, dict(schema=value["schema"], screen=spec.to_dict(), budget=budget)


def assembly_reference(assembly, artifacts):
    """Bind supplied files/declarations, without granting materialization authenticity."""
    payload = dict(schema="screen-assembly-reference-v1", artifacts=artifacts,
        registry_version=assembly.registry.version_id,
        assembly_version=assembly.block.metadata["version_id"],
        source_bindings=list(assembly.source_bindings), rows=len(assembly.block.values),
        upstream_execution_certified=False)
    return dict(payload, reference_id=content_id(payload))


def inspect_inputs(project, proposal, *, include_data=False):
    spec, doc = protocol(proposal["spec"])
    roles = verify_inputs(project, proposal["inputs"], doc["budget"]["max_input_bytes"])
    if set(roles) != ROLES:
        raise ValueError("Exactly seven declared screening input roles required")
    registry = FeatureRegistry.from_dict(json.loads(roles["registry"].read_text(encoding="utf-8")))
    meta = json.loads(roles["assembly_definition"].read_text(encoding="utf-8"))
    units = meta.pop("units")
    bindings = meta.get("source_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("Explicit assembly source bindings required")
    block = FeatureBlock(pd.read_parquet(roles["assembly_values"]),
        pd.read_parquet(roles["assembly_missing"]), units, meta)
    assembly = FeatureAssembly(block, registry, tuple(bindings))
    artifacts = {role: digest(roles[role]) for role in ("assembly_values", "assembly_missing", "assembly_definition")}
    receipt = json.loads(roles["assembly_receipt"].read_text(encoding="utf-8"))
    if not same_json(receipt, assembly_reference(assembly, artifacts)):
        raise ValueError("Assembly reference receipt differs from supplied files/declarations")
    decisions = pd.read_parquet(roles["decisions"])
    labels = pd.read_parquet(roles["labels"])
    _, scoped, _, _, _ = _prepare(spec, decisions, assembly, labels)
    info = dict(development_decisions=len(scoped), holdout_rows_executed=0,
        source_files_by_reference=True, historical_source_certified=False,
        target_scope="mature development only; declared label clocks are not vendor authentication")
    if include_data:
        info["data"] = (decisions, assembly, labels)
    return spec, registry, info


def planned_attempts(spec):
    return dict(schema="planned-alpha-screen-attempts-v1",
        attempts=[dict(c, status="planned") for c in spec.to_dict()["channels"]],
        candidate_count=len(spec.to_dict()["channels"]), fits=0, accounts=0,
        counting_rule="All persisted direction/condition intents count, including failed and cancelled jobs")


def register(research, session, proposal):
    required = {"kind", "hypothesis", "expected_observation", "falsification", "spec", "inputs"}
    if (not isinstance(proposal, dict) or not required <= set(proposal)
            or set(proposal) - required - {"timeout_seconds", "resources"}
            or proposal["kind"] != "alpha_screen"):
        raise ValueError("Strict alpha_screen registration fields required")
    with research.store.connection() as db:
        research.store.owned(db, session)
    resources = WorkerResources.from_dict(proposal.get("resources", WorkerResources().to_dict())).to_dict()
    spec, registry, inspection = inspect_inputs(research.project, proposal)
    _, doc = protocol(proposal["spec"])
    timeout = proposal.get("timeout_seconds", 900)
    if type(timeout) is not int or not 30 <= timeout <= 7200:
        raise ValueError("Bounded screening execution timeout required")
    sources = sorted(p for p in (research.project / "src").rglob("*.py") if "__pycache__" not in p.parts)
    if not (research.project / "src/researchops/screen_worker.py").is_file():
        raise ValueError("Missing screening worker implementation")
    for source in sources:
        if not source.resolve().is_relative_to(research.project):
            raise ValueError("Source implementation escapes project")
        for item in (source, *source.parents):
            if item == research.project:
                break
            if item.is_symlink() or (hasattr(item, "is_junction") and item.is_junction()):
                raise ValueError("Linked screening source refused")
    inventory = {p.relative_to(research.project).as_posix(): digest(p) for p in sources}
    environment = dict(python=platform.python_version(), packages={n: importlib.metadata.version(n)
        for n in ("numpy", "pandas", "pyarrow", "tzdata", "psutil", "scikit-learn", "scipy", "joblib", "threadpoolctl")})
    p = {k: text(proposal[k], k) for k in ("hypothesis", "expected_observation", "falsification")}
    p.update(kind="alpha_screen", spec=doc, inputs=proposal["inputs"], timeout_seconds=timeout,
        resources=resources, registry_version=registry.version_id, engine_inventory=inventory,
        engine_hash=content_id(inventory), environment=environment, inspection=inspection,
        candidate_count=len(spec.to_dict()["channels"]), planned_model_fits=0, planned_accounts=0,
        submitted_by=session["worker"], evaluation_scope="retrospective_time_split")
    canonical = json.loads(json.dumps(p)); canonical["spec"]["screen"].pop("name")
    for key in ("hypothesis", "expected_observation", "falsification", "timeout_seconds", "submitted_by"):
        canonical.pop(key)
    canonical["inputs"].sort(key=lambda x: x["role"])
    for channel in canonical["spec"]["screen"]["channels"]:
        channel.pop("hypothesis")
    canonical["spec"]["screen"]["channels"].sort(key=lambda c: c["channel_id"])
    ex = research.store.register(session, p, content_id(canonical),
        channel_budget=doc["budget"]["max_task_channel_intents"])
    job = research.root / "worker_jobs" / ex["id"]
    if (job / "ready.json").is_file():
        verify_registered(research, ex)
        return ex
    if ex["status"] != "planned" or job.exists():
        raise ValueError("Previous screening freeze incomplete or failed; preserve failed try")
    p = ex["proposal"]; code = job / "code"
    try:
        code.mkdir(parents=True)
        for name, sha in p["engine_inventory"].items():
            target = code / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(research.project / name, target)
            if digest(target) != sha:
                raise ValueError("Source changed during screening freeze")
        write_json(job / "code_manifest.json", dict(artifacts=p["engine_inventory"]))
        write_json(job / "input.json", dict(p, project=str(research.project), root=str(research.root),
            task_id=ex["task_id"], experiment_id=ex["id"]))
        write_json(job / "candidate_manifest.json", planned_attempts(ScreenSpec.from_dict(p["spec"]["screen"])))
        verify_inputs(research.project, p["inputs"], p["spec"]["budget"]["max_input_bytes"])
        write_json(job / "ready.json", dict(kind="alpha_screen", input_hash=digest(job / "input.json"),
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
    if p.get("kind") != "alpha_screen":
        raise ValueError("Not a registered screening job")
    job = research.root / "worker_jobs" / experiment["id"]
    cfg = verify_frozen_job(job, "alpha_screen")
    expected = dict(p, project=str(research.project), root=str(research.root),
        task_id=experiment["task_id"], experiment_id=experiment["id"])
    if not same_json(cfg, expected):
        raise ValueError("Frozen screening job differs from database registration")
    spec, doc = protocol(p["spec"])
    roles = verify_inputs(research.project, p["inputs"], doc["budget"]["max_input_bytes"])
    if set(roles) != ROLES:
        raise ValueError("Screen input roles changed")
    registry = FeatureRegistry.from_dict(json.loads(roles["registry"].read_text(encoding="utf-8")))
    plan = json.loads((job / "candidate_manifest.json").read_text(encoding="utf-8"))
    if (not same_json(plan, planned_attempts(spec)) or p["candidate_count"] != len(spec.to_dict()["channels"])
            or p["registry_version"] != registry.version_id
            or p["planned_model_fits"] != 0 or p["planned_accounts"] != 0):
        raise ValueError("Screen registry/candidate plan changed")
    return dict(verified=True, kind="alpha_screen_registration", candidate_count=p["candidate_count"],
        executed_candidates=0, fits=0, accounts=0,
        scope="registered input/source/plan identity; not upstream algorithm or strategy validity")
