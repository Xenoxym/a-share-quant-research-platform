"""Independent frozen score-selection and native account audit, without fitting."""
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType
import uuid
from .alpha_experiments import verify_inputs
from .alpha_worker import verify_frozen_job
from .account_experiments import verify_registered, inspect_inputs, native_folder, same_json
from .account_worker import FILES, output_bytes
from .resources import WorkerResources
from .screen_output_audit import _FrozenFinder
from ..alpharesearch.audit import audit_score_account_run
from ..technical.artifacts import content_id, digest, verify_artifacts
from ..technical.runner import code_inventory

def read(p): return json.loads(p.read_text(encoding="utf-8"))

def audit_frozen(cfg,job):
    if Path(__file__).resolve().parents[2]!=(job/"code").resolve(): raise ValueError("Account audit requires frozen source")
    folder=job/"account_result";native=native_folder(cfg)
    for directory in (folder,native):
        if any(p.is_symlink() or (hasattr(p,"is_junction") and p.is_junction()) for p in [directory,*directory.rglob("*")]):
            raise ValueError("Linked account evidence refused")
    manifest_hash=digest(folder/"manifest.json"); nm_hash=digest(native/"manifest.json"); manifest=read(folder/"manifest.json")
    expected=dict(schema="registered-alpha-account-manifest-v1",experiment_id=job.name,task_id=cfg["task_id"],
        input_hash=digest(job/"input.json"),engine_hash=cfg["engine_hash"],candidate_manifest_hash=digest(job/"candidate_manifest.json"),native_manifest_hash=nm_hash)
    if (set(manifest)!=set(expected)|{"artifacts"} or not same_json({k:manifest.get(k) for k in expected},expected)
            or set(manifest["artifacts"])!=FILES or {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}!=FILES|{"manifest.json"}):
        raise ValueError("Strict scored-account output manifest differs")
    verify_artifacts(folder,manifest)
    if not same_json(read(folder/"native_pointer.json"),dict(run_id=native.name,manifest_hash=nm_hash)): raise ValueError("Native pointer changed")
    if output_bytes(job,cfg)>WorkerResources.from_dict(cfg["resources"]).max_output_bytes: raise ValueError("Account output exceeds byte budget")
    _,spec,info=inspect_inputs(cfg["project"],cfg,cfg["root"],include_data=True); info.pop("data")
    if not same_json(info,cfg["inspection"]): raise ValueError("Account inputs changed")
    nm=read(native/"manifest.json");result=read(native/"result.json")
    if (not same_json(nm["spec"],spec.to_dict()) or nm["snapshot_id"]!=cfg["spec"]["snapshot_id"]
            or nm["snapshot_manifest_sha256"]!=cfg["spec"]["snapshot_manifest_hash"] or nm["code_hash"]!=content_id(code_inventory())
            ):
        raise ValueError("Native run spec/source/scenarios differ")
    for role,name in (("predictions","alpha_predictions.parquet"),("learning_receipt","alpha_learning_receipt.json"),("references","alpha_references.parquet")):
        paths=verify_inputs(cfg["project"],cfg["inputs"],cfg["spec"]["budget"]["max_input_bytes"])
        if role.endswith("receipt"):
            if not same_json(read(paths[role]),read(native/name)):raise ValueError("Archived source receipt differs")
        else:
            import pandas as pd
            pd.testing.assert_frame_equal(pd.read_parquet(paths[role]),pd.read_parquet(native/name),check_exact=True)
    attempts=dict(schema="alpha-account-attempts-v1",planned_accounts=cfg["planned_accounts"],model_fits=0,
        scenarios=[dict(scenario=s,status="completed") for s in cfg["spec"]["scenarios"]])
    if not same_json(read(folder/"attempts.json"),attempts): raise ValueError("Account attempt ledger differs")
    audit=audit_score_account_run(native)
    listed=[p["metrics"]["scenario"] for p in result["portfolios"]]
    if set(listed)!=set(cfg["spec"]["scenarios"]) or len(listed)!=len(cfg["spec"]["scenarios"]):raise ValueError("Result scenario set differs")
    # Files may change after they were read; bind all evidence again at completion.
    verify_inputs(cfg["project"],cfg["inputs"],cfg["spec"]["budget"]["max_input_bytes"])
    snapshot=Path(cfg["root"])/"snapshots"/cfg["spec"]["snapshot_id"]
    if digest(snapshot/"manifest.json")!=cfg["spec"]["snapshot_manifest_hash"]:
        raise ValueError("Snapshot changed during account audit")
    verify_artifacts(snapshot,read(snapshot/"manifest.json"))
    verify_artifacts(native,nm)
    verify_artifacts(folder,manifest)
    if output_bytes(job,cfg)>WorkerResources.from_dict(cfg["resources"]).max_output_bytes:
        raise ValueError("Account output changed beyond byte budget during audit")
    if (digest(folder/"manifest.json")!=manifest_hash or digest(native/"manifest.json")!=nm_hash
            or not same_json(verify_frozen_job(job,"alpha_account"),cfg)):raise ValueError("Account changed during audit")
    value=dict(schema="registered-alpha-account-audit-v1",verified=True,experiment_id=job.name,task_id=cfg["task_id"],
        candidate_count=1,accounts=cfg["planned_accounts"],model_fits=0,additional_audit_fits=0,run_id=native.name,
        input_hash=digest(job/"input.json"),output_manifest_hash=manifest_hash,native_manifest_hash=nm_hash,engine_hash=cfg["engine_hash"],
        audit_code_hash=cfg["engine_inventory"]["src/researchops/account_output_audit.py"],account_audit=audit,
        model_scores_independently_reproduced=False,vendor_history_certified=False,account_results=True,
        score_source=cfg["spec"]["score_source"]["kind"],scope="independent selection/daily account checks; never refit or reproduce model scores")
    return dict(value,audit_id=content_id(value))

def audit_output(research,experiment):
    verify_registered(research,experiment);job=research.root/"worker_jobs"/experiment["id"];cfg=verify_frozen_job(job,"alpha_account")
    prefix="_frozen_account_audit_"+uuid.uuid4().hex;package=ModuleType(prefix);package.__path__=[str(job/"code")];sys.modules[prefix]=package
    finder=_FrozenFinder(prefix,job/"code");sys.meta_path.insert(0,finder)
    try:return importlib.import_module(prefix+".src.researchops.account_output_audit").audit_frozen(cfg,job)
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name==prefix or name.startswith(prefix+"."):del sys.modules[name]
