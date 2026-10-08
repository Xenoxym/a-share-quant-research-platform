"""Frozen scored-account jobs; registered models or explicit synthetic score stubs."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import time
import pandas as pd
from .alpha_experiments import verify_inputs
from .alpha_worker import verify_frozen_job
from .account_intents import declared
from .resources import WorkerResources
from .screen_experiments import same_json
from .store import text
from ..alpharesearch.learning import LearningResult
from ..alpharesearch.portfolio import ScorePortfolioSpec, build_score_portfolio
from ..alpharesearch.account import _account_spec, SNAPSHOT_FILES
from ..technical.contracts import ResearchSpec
from ..technical.artifacts import content_id, digest, write_json, verify_artifacts
ROLES={"predictions","learning_receipt","references","calendar"}

def protocol(value):
    fields={"schema","portfolio","account","scenarios","snapshot_id","snapshot_manifest_hash","score_source","budget"}
    if not isinstance(value,dict) or set(value)!=fields or value["schema"]!="registered-alpha-account-v1":
        raise ValueError("Strict scored-account protocol required")
    policy=ScorePortfolioSpec.from_dict(value["portfolio"])
    spec=_account_spec(ResearchSpec.from_dict(value["account"]),policy.to_dict())
    if value["scenarios"] != ["zero_transaction_cost","configured"]:
        raise ValueError("Scored account requires fixed zero/configured scenarios")
    for k,pattern in (("snapshot_id",r"[a-f0-9]{24}"),("snapshot_manifest_hash",r"[a-f0-9]{64}")):
        if not isinstance(value[k],str) or not re.fullmatch(pattern,value[k]): raise ValueError("Invalid snapshot identity")
    source=value["score_source"]
    if not isinstance(source,dict): raise ValueError("Explicit score provenance required")
    if source.get("kind")=="synthetic_stub":
        if set(source)!={"kind","statement"} or source["statement"]!="engineering scores only; no fitted strategy":
            raise ValueError("Explicit synthetic no-fit statement required")
    elif source.get("kind")=="registered_learning":
        if set(source)!={"kind","experiment_id"} or not re.fullmatch(r"exp-[a-f0-9]{12}",str(source["experiment_id"])):
            raise ValueError("Registered learning identity required")
    else: raise ValueError("Unsupported score provenance")
    b=value["budget"]
    if not isinstance(b,dict) or set(b)!={"max_input_bytes","max_task_account_intents","max_real_account_intents","max_synthetic_account_intents"}:
        raise ValueError("Strict account input/intent budgets required")
    if type(b["max_input_bytes"]) is not int or not 1<=b["max_input_bytes"]<=100_000_000_000:
        raise ValueError("Bounded account input bytes required")
    declared({"spec":value})
    doc=dict(value,portfolio=policy.to_dict(),account=spec.to_dict())
    return policy,spec,doc

def inspect_inputs(project,proposal,root,*,include_data=False):
    policy,spec,doc=protocol(proposal["spec"])
    paths=verify_inputs(project,proposal["inputs"],doc["budget"]["max_input_bytes"])
    if set(paths)!=ROLES: raise ValueError("Exactly four scored input roles required")
    result=LearningResult(pd.read_parquet(paths["predictions"]),json.loads(paths["learning_receipt"].read_text(encoding="utf-8")))
    refs=pd.read_parquet(paths["references"]); calendar=json.loads(paths["calendar"].read_text(encoding="utf-8"))
    portfolio=build_score_portfolio(result,refs,calendar,policy)
    snapshot=Path(root)/"snapshots"/doc["snapshot_id"]
    if snapshot.is_symlink() or (hasattr(snapshot,"is_junction") and snapshot.is_junction()): raise ValueError("Linked snapshot refused")
    if digest(snapshot/"manifest.json")!=doc["snapshot_manifest_hash"]: raise ValueError("Registered snapshot changed")
    sm=json.loads((snapshot/"manifest.json").read_text(encoding="utf-8"))
    if (sm["snapshot_id"]!=snapshot.name or not SNAPSHOT_FILES<=set(sm["artifacts"])
            or not sm["start"]<=policy.to_dict()["start"]<=policy.to_dict()["end"]<=sm["end"]):
        raise ValueError("Native snapshot identity/coverage differs")
    for name in sm["artifacts"]:
        if Path(name).name!=name or "\\" in name or (snapshot/name).is_symlink(): raise ValueError("Unsafe snapshot artifact")
    verify_artifacts(snapshot,sm)
    if digest(paths["calendar"])!=digest(snapshot/"calendar.json"): raise ValueError("Score calendar differs from snapshot")
    info=dict(prediction_rows=len(result.predictions),fit_id=result.receipt["fit_id"],prediction_id=result.receipt["prediction_id"],
        portfolio_id=policy.portfolio_id,portfolio_receipt_id=content_id(portfolio.receipt),
        score_provenance=doc["score_source"]["kind"],numeric_model_reproduction=False)
    if include_data: info["data"]=(result,refs,calendar,policy,snapshot,spec)
    return policy,spec,info

def upstream(research,proposal):
    source=proposal["spec"]["score_source"]
    if source["kind"]=="synthetic_stub": return dict(kind="synthetic_stub",actual_model_fits=0)
    from .alpha_execution import verify_completed
    ex=research.store.experiment(source["experiment_id"])
    if ex["proposal"].get("kind")!="alpha_learn" or ex["status"]!="completed": raise ValueError("Completed registered learning required")
    checked=verify_completed(research,ex)
    folder=research.root/"worker_jobs"/ex["id"]/"learn_result"
    paths=verify_inputs(research.project,proposal["inputs"],proposal["spec"]["budget"]["max_input_bytes"])
    for role,name in (("predictions","predictions.parquet"),("learning_receipt","learning_receipt.json")):
        if digest(paths[role])!=digest(folder/name): raise ValueError("Scores differ from registered learning output")
    return dict(kind="registered_learning",experiment_id=ex["id"],output_manifest_hash=digest(folder/"manifest.json"),
        audit_id=checked["audit_id"],additional_fits=0,numeric_model_reproduction=False)

def run_id(experiment):
    return datetime.fromtimestamp(experiment["created"],timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+experiment["id"][-8:]

def native_folder(cfg):
    value=cfg.get("native_run_id")
    if not isinstance(value,str) or not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}",value) or value[-8:]!=cfg["experiment_id"][-8:]:
        raise ValueError("Fixed native run identity differs")
    folder=Path(cfg["root"])/"runs"/value
    for p in (folder,folder.parent,folder.parent.parent):
        if p.is_symlink() or (hasattr(p,"is_junction") and p.is_junction()): raise ValueError("Linked native account path refused")
    return folder

def register(research,session,proposal):
    required={"kind","hypothesis","expected_observation","falsification","spec","inputs"}
    if (not isinstance(proposal,dict) or not required<=set(proposal) or set(proposal)-required-{"timeout_seconds","resources"}
            or proposal["kind"]!="alpha_account"): raise ValueError("Strict scored-account proposal required")
    with research.store.connection() as db: research.store.owned(db,session)
    resources=WorkerResources.from_dict(proposal.get("resources",WorkerResources(output_scope="scored_account_result").to_dict())).to_dict()
    if resources["output_scope"]!="scored_account_result": raise ValueError("Scored-account resource output scope required")
    _,_,doc=protocol(proposal["spec"]); _,_,inspection=inspect_inputs(research.project,proposal,research.root)
    source=upstream(research,proposal); timeout=proposal.get("timeout_seconds",900)
    if type(timeout) is not int or not 30<=timeout<=7200: raise ValueError("Bounded scored-account timeout required")
    files=sorted(p for p in (research.project/"src").rglob("*") if p.is_file() and p.suffix in {".py",".html",".js",".css"} and "__pycache__" not in p.parts)
    for p in files:
        if not p.resolve().is_relative_to(research.project): raise ValueError("Source escaped project")
        for item in (p,*p.parents):
            if item==research.project: break
            if item.is_symlink() or (hasattr(item,"is_junction") and item.is_junction()): raise ValueError("Linked source refused")
    inventory={p.relative_to(research.project).as_posix():digest(p) for p in files}
    environment=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in
        ("numpy","pandas","pyarrow","tzdata","psutil","scikit-learn","scipy","joblib","threadpoolctl","duckdb","pyyaml")})
    p={k:text(proposal[k],k) for k in ("hypothesis","expected_observation","falsification")}
    p.update(kind="alpha_account",spec=doc,inputs=proposal["inputs"],resources=resources,timeout_seconds=timeout,
        engine_inventory=inventory,engine_hash=content_id(inventory),environment=environment,inspection=inspection,
        upstream=source,candidate_count=1,planned_model_fits=0,planned_accounts=len(doc["scenarios"]),
        submitted_by=session["worker"],evaluation_scope="retrospective_account")
    canonical=json.loads(json.dumps(p))
    canonical["spec"]["portfolio"].pop("name"); canonical["spec"]["account"]["strategy"].pop("name")
    for k in ("hypothesis","expected_observation","falsification","submitted_by","timeout_seconds"): canonical.pop(k)
    canonical["inspection"].pop("portfolio_receipt_id")
    canonical["inputs"].sort(key=lambda x:x["role"])
    ex=research.store.register(session,p,content_id(canonical)); job=research.root/"worker_jobs"/ex["id"]
    if (job/"ready.json").is_file(): verify_registered(research,ex); return ex
    if ex["status"]!="planned" or job.exists(): raise ValueError("Previous account freeze failed/incomplete; never retry")
    p=ex["proposal"]
    try:
        code=job/"code";code.mkdir(parents=True)
        for name,h in p["engine_inventory"].items():
            dest=code/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(research.project/name,dest)
            if digest(dest)!=h: raise ValueError("Source changed during account freeze")
        write_json(job/"code_manifest.json",dict(artifacts=p["engine_inventory"]))
        write_json(job/"input.json",dict(p,project=str(research.project),root=str(research.root),
            task_id=ex["task_id"],experiment_id=ex["id"],native_run_id=run_id(ex)))
        write_json(job/"candidate_manifest.json",dict(schema="planned-alpha-account-v1",scenarios=doc["scenarios"],
            accounts=len(doc["scenarios"]),model_fits=0))
        verify_inputs(research.project,p["inputs"],doc["budget"]["max_input_bytes"])
        write_json(job/"ready.json",dict(kind="alpha_account",input_hash=digest(job/"input.json"),
            code_manifest_hash=digest(job/"code_manifest.json"),candidate_manifest_hash=digest(job/"candidate_manifest.json")))
    except Exception as exc:
        with research.store.connection(True) as db:
            db.execute("UPDATE experiments SET status='failed',error=?,updated=? WHERE id=?",(str(exc),time.time(),ex["id"]))
            research.store.event(db,ex["task_id"],session["worker"],"freeze_failed",dict(experiment_id=ex["id"],error=str(exc)))
        raise
    return ex

def verify_registered(research,experiment):
    p=experiment["proposal"]; job=research.root/"worker_jobs"/experiment["id"]
    if p.get("kind")!="alpha_account": raise ValueError("Registered scored account required")
    cfg=verify_frozen_job(job,"alpha_account")
    expected=dict(p,project=str(research.project),root=str(research.root),task_id=experiment["task_id"],
        experiment_id=experiment["id"],native_run_id=run_id(experiment))
    if not same_json(cfg,expected): raise ValueError("Frozen account differs from registration")
    _,_,doc=protocol(p["spec"]); verify_inputs(research.project,p["inputs"],doc["budget"]["max_input_bytes"])
    plan=dict(schema="planned-alpha-account-v1",scenarios=doc["scenarios"],accounts=len(doc["scenarios"]),model_fits=0)
    if (not same_json(json.loads((job/"candidate_manifest.json").read_text(encoding="utf-8")),plan)
            or type(p["planned_accounts"]) is not int or p["planned_accounts"]!=len(doc["scenarios"])
            or type(p["planned_model_fits"]) is not int or p["planned_model_fits"]!=0
            or type(p["candidate_count"]) is not int or p["candidate_count"]!=1 or not same_json(p["upstream"],upstream(research,p))):
        raise ValueError("Account plan/upstream source changed")
    native_folder(cfg)
    return dict(verified=True,kind="alpha_account_registration",planned_accounts=len(doc["scenarios"]),additional_fits=0,
        numeric_model_reproduction=False,scope="input/source/intent identity; no account results yet")
