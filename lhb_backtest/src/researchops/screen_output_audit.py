"""Full same-frozen-algorithm output replay. Not an independent algorithm/vendor audit."""
import importlib
from importlib.abc import MetaPathFinder
from importlib.machinery import SourceFileLoader
from importlib.util import spec_from_file_location
import json
from pathlib import Path
import sys
from types import ModuleType
import uuid

from .alpha_worker import verify_frozen_job
from .resources import WorkerResources
from .screen_experiments import inspect_inputs, planned_attempts, verify_registered, same_json
from .screen_worker import TABLES
from ..alpharesearch.screening import screen
from ..technical.artifacts import content_id, digest, jsonable, verify_artifacts


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_frozen(cfg, job):
    """Executed from verified frozen files; replay all tables, not new training."""
    if Path(__file__).resolve().parents[2] != (job / "code").resolve():
        raise ValueError("Screen audit must execute the registered frozen source")
    folder = job / "screen_result"
    paths = [folder, *folder.rglob("*")]
    if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()) for p in paths):
        raise ValueError("Linked screen output refused")
    manifest_hash = digest(folder / "manifest.json")
    manifest = _read(folder / "manifest.json")
    expected = dict(schema="registered-alpha-screen-manifest-v1", experiment_id=job.name,
        task_id=cfg["task_id"], input_hash=digest(job / "input.json"), engine_hash=cfg["engine_hash"],
        candidate_manifest_hash=digest(job / "candidate_manifest.json"))
    files = {name + ".json" for name in TABLES} | {"attempts.json", "diagnostic_receipt.json"}
    if (set(manifest) != set(expected) | {"artifacts"}
            or any(manifest.get(k) != v for k, v in expected.items())
            or set(manifest["artifacts"]) != files
            or {p.relative_to(folder).as_posix() for p in paths[1:] if p.is_file()} != files | {"manifest.json"}):
        raise ValueError("Strict registered screen manifest/file set differs")
    verify_artifacts(folder, manifest)
    resources = WorkerResources.from_dict(cfg["resources"])
    if sum(p.stat().st_size for p in paths if p.is_file()) > resources.max_output_bytes:
        raise ValueError("Audited screen output exceeds byte budget")
    spec, registry, info = inspect_inputs(cfg["project"], cfg, include_data=True)
    result = screen(spec, *info.pop("data"))
    if not same_json(info, cfg["inspection"]):
        raise ValueError("Screen development input scope changed")
    for name in TABLES:
        frame = getattr(result, name)
        expected_table = dict(columns=list(frame.columns), rows=jsonable(frame.to_dict("records")))
        if not same_json(_read(folder / (name + ".json")), expected_table):
            raise ValueError("Frozen numerical screen replay differs: " + name)
    if not same_json(_read(folder / "diagnostic_receipt.json"), jsonable(result.receipt)):
        raise ValueError("Frozen screen diagnostic receipt differs")
    attempts = planned_attempts(spec)
    attempts["attempts"] = [dict(c, status="evaluated") for c in spec.to_dict()["channels"]]
    if not same_json(_read(folder / "attempts.json"), attempts):
        raise ValueError("Screen evaluated intent ledger differs")
    # Recheck sources after full replay; mutable referenced files cannot be accepted.
    from .alpha_experiments import verify_inputs
    verify_inputs(cfg["project"], cfg["inputs"], cfg["spec"]["budget"]["max_input_bytes"])
    if not same_json(verify_frozen_job(job, "alpha_screen"), cfg) or digest(folder / "manifest.json") != manifest_hash:
        raise ValueError("Frozen screening identities changed during replay")
    verify_artifacts(folder, manifest)
    value = dict(schema="registered-alpha-screen-audit-v1", verified=True,
        experiment_id=job.name, task_id=cfg["task_id"], candidate_count=cfg["candidate_count"],
        development_decisions=info["development_decisions"], holdout_rows_executed=0,
        registry_version=registry.version_id, model_fits=0, accounts=0,
        input_hash=digest(job / "input.json"), output_manifest_hash=digest(folder / "manifest.json"),
        engine_hash=cfg["engine_hash"], audit_code_hash=cfg["engine_inventory"]["src/researchops/screen_output_audit.py"],
        replay_scope="every output table and receipt, same frozen algorithm; no independent algorithm or vendor certification",
        account_results=False, annualized_results=False, complete_intraperiod_drawdown=False)
    return dict(value, audit_id=content_id(value))


class _FrozenSourceLoader(SourceFileLoader):
    def get_code(self, fullname):
        # Frozen source inventories exclude bytecode: neither read nor create it.
        return self.source_to_code(self.get_data(self.path), self.path)


class _FrozenFinder(MetaPathFinder):
    def __init__(self, prefix, root):
        self.prefix, self.root = prefix, root

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.prefix + "."):
            return None
        relative = fullname[len(self.prefix) + 1:].replace(".", "/")
        source = self.root / (relative + ".py")
        if not source.is_file():
            source = self.root / relative / "__init__.py"
        if not source.is_file():
            return None
        if not source.resolve().is_relative_to(self.root):
            raise ValueError("Frozen import escaped code root")
        return spec_from_file_location(fullname, source,
            loader=_FrozenSourceLoader(fullname, str(source)))


def audit_output(research, experiment):
    verify_registered(research, experiment)
    job = research.root / "worker_jobs" / experiment["id"]
    cfg = verify_frozen_job(job, "alpha_screen")
    # The namespace is unique per call; current editable src is never substituted.
    prefix = "_frozen_screen_audit_" + uuid.uuid4().hex
    package = ModuleType(prefix); package.__path__ = [str(job / "code")]
    sys.modules[prefix] = package
    finder = _FrozenFinder(prefix, job / "code"); sys.meta_path.insert(0, finder)
    try:
        module = importlib.import_module(prefix + ".src.researchops.screen_output_audit")
        return module.audit_frozen(cfg, job)
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
