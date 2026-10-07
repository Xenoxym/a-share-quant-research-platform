"""Durable learning mechanics; fixture fits are verification, not market strategies."""
import copy
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from src.alpharesearch.learning import _fingerprint
from src.researchops.learn_experiments import verify_registered
from src.researchops.learn_output_audit import audit_output
from src.researchops.screen_experiments import assembly_reference
from src.researchops.service import Research
from src.technical.artifacts import digest, write_json
from tests.test_alpha_learning import example
from tests.test_researchops import TASK

PROJECT = Path(__file__).resolve().parents[1]


def setup(tmp_path, intent_budget=3):
    project = tmp_path/"project"; project.mkdir()
    for source in (PROJECT/"src").rglob("*.py"):
        if "__pycache__" in source.parts:
            continue
        dest = project/source.relative_to(PROJECT); dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    folder = project/"data/learn_input"; folder.mkdir(parents=True)
    spec, model, decisions, assembly, labels = example(); paths={}
    for role, frame in (("assembly_values",assembly.block.values),("assembly_missing",assembly.block.missing),
            ("decisions",decisions),("labels",labels)):
        paths[role] = folder/(role+".parquet"); frame.to_parquet(paths[role],index=False)
    paths["registry"] = folder/"registry.json"; write_json(paths["registry"],assembly.registry.to_dict())
    paths["assembly_definition"] = folder/"definition.json"
    write_json(paths["assembly_definition"],dict(assembly.block.metadata,units=assembly.block.units))
    paths["assembly_receipt"] = folder/"receipt.json"
    write_json(paths["assembly_receipt"],assembly_reference(assembly,
        {r:digest(paths[r]) for r in ("assembly_values","assembly_missing","assembly_definition")}))
    inputs=[dict(role=r,path=p.relative_to(project).as_posix(),sha256=digest(p),bytes=p.stat().st_size) for r,p in paths.items()]
    proposal=dict(kind="alpha_learn",hypothesis="Synthetic single-fit mechanics",expected_observation="Retained immutable predictions",
        falsification="Any refit/clock/budget mismatch",inputs=inputs,timeout_seconds=30,
        spec=dict(schema="registered-alpha-learn-v1",learning=spec.to_dict(),model=model.to_dict(),
            budget=dict(max_input_bytes=1000000,max_task_fit_intents=intent_budget)))
    ops=Research(project); task=ops.create(dict(TASK,max_experiments=4)); session=ops.store.claim(task["id"],"learn-test-worker")
    return ops,session,proposal,folder


def test_duplicate_registration_preserves_original_single_fit_intent(tmp_path):
    ops,s,p,_=setup(tmp_path); ex=ops.register(s,p); q=copy.deepcopy(p)
    q["hypothesis"]="Another display"; q["inputs"].reverse()
    q["spec"]["learning"]["name"]="Another_protocol"; q["spec"]["model"]["name"]="Another_model"
    assert ops.register(s,q)["id"]==ex["id"]
    assert verify_registered(ops,ex)["planned_fit_intents"]==1
    assert len(Research(ops.project).store.get(s["task_id"])["experiments"])==1


def test_fit_budget_survives_failure_cancel_restart_and_cannot_reset(tmp_path):
    ops,s,p,_=setup(tmp_path,intent_budget=2); first=ops.register(s,p)
    ops.store.start(s,first["id"]); ops.store.finish(first["id"],"failed",error="Fixture retained failure")
    q=copy.deepcopy(p); q["spec"]["model"]["params"]["alpha"]=20.
    second=ops.register(s,q); ops.cancel(s,second["id"],"Fixture retained cancellation")
    r=copy.deepcopy(p); r["spec"]["model"]["params"]["alpha"]=30.
    with pytest.raises(ValueError,match="fit budget exhausted"):
        Research(ops.project).register(s,r)
    r["spec"]["budget"]["max_task_fit_intents"]=3
    with pytest.raises(ValueError,match="budget already frozen"):
        Research(ops.project).register(s,r)
    assert [e["status"] for e in ops.store.get(s["task_id"])["experiments"]]==["failed","cancelled"]


def test_earliest_freeze_failure_cannot_implicitly_retry(tmp_path,monkeypatch):
    ops,s,p,_=setup(tmp_path); original=Path.mkdir
    def fail(path,*args,**kwargs):
        if path.name=="code" and path.parent.name.startswith("exp-"):
            raise OSError("Earliest freeze failed")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"mkdir",fail)
    with pytest.raises(OSError,match="Earliest freeze"):
        ops.register(s,p)
    monkeypatch.setattr(Path,"mkdir",original)
    ex=ops.store.get(s["task_id"])["experiments"][0]; assert ex["status"]=="failed"
    assert not (ops.root/"worker_jobs"/ex["id"]).exists()
    with pytest.raises(ValueError,match="freeze incomplete or failed"):
        ops.register(s,p)


@pytest.mark.parametrize("mutation",["budget_bool","unknown_model","worker_threads","extra_field"])
def test_strict_learning_registration_rejects_before_intent(tmp_path,mutation):
    ops,s,p,_=setup(tmp_path)
    if mutation=="budget_bool": p["spec"]["budget"]["max_task_fit_intents"]=True
    if mutation=="unknown_model": p["spec"]["model"]["family"]="free_eval"
    if mutation=="worker_threads":
        from src.researchops.resources import WorkerResources
        p["resources"]=WorkerResources(library_threads=1).to_dict()
    if mutation=="extra_field": p["spec"]["arbitrary_module"]="example"
    with pytest.raises(ValueError): ops.register(s,p)
    assert not ops.store.get(s["task_id"])["experiments"]


def test_recovery_submit_and_review_never_call_estimator_fit(tmp_path,monkeypatch):
    ops,s,p,_=setup(tmp_path); ex=ops.register(s,p); done=ops.execute(s,ex["id"])
    assert done["status"]=="completed",done.get("error")
    assert done["result"]["model_fits"]==1 and done["result"]["additional_audit_fits"]==0
    from sklearn.linear_model import Ridge,ElasticNet,HuberRegressor
    from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
    def prohibited(*args,**kwargs): raise AssertionError("Audit refitted a model")
    for cls in (Ridge,ElasticNet,HuberRegressor,RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor):
        monkeypatch.setattr(cls,"fit",prohibited)
    assert Research(ops.project).recover(ex["id"])["status"]=="completed"
    report=dict(summary="Fixture verification only",findings=["Single fit"],limitations=["No account"],next_steps=["Continue finite protocol"])
    ops.submit(s,report); ops.review(s["task_id"],"different-reviewer","defer","Fixture verification complete")
    assert ops.store.get(s["task_id"])["status"]=="deferred"



def _reseal(job):
    folder=job/"learn_result"; manifest=json.loads((folder/"manifest.json").read_text())
    manifest["artifacts"]={name:digest(folder/name) for name in manifest["artifacts"]}
    write_json(folder/"manifest.json",manifest)
    receipt=json.loads((job/"receipt.json").read_text()); receipt["result_manifest_hash"]=digest(folder/"manifest.json")
    write_json(job/"receipt.json",receipt)


@pytest.mark.parametrize("mutation",["clock","nan_score","bool_score","median","fit_bool","ledger_float","metadata_int"])
def test_coherently_resealed_bad_output_semantics_rejected(tmp_path,mutation):
    ops,s,p,_=setup(tmp_path); ex=ops.register(s,p); done=ops.execute(s,ex["id"])
    assert done["status"]=="completed",done.get("error")
    job=ops.root/"worker_jobs"/ex["id"]; folder=job/"learn_result"
    receipt=json.loads((folder/"learning_receipt.json").read_text())
    if mutation in {"clock","nan_score","bool_score"}:
        frame=pd.read_parquet(folder/"predictions.parquet")
        if mutation=="clock": frame.loc[0,"execution_at"]+=pd.Timedelta(seconds=1)
        if mutation=="nan_score": frame.loc[0,"score"]=float("nan")
        if mutation=="bool_score": frame["score"]=True
        frame.to_parquet(folder/"predictions.parquet",index=False); receipt["prediction_outputs"]=_fingerprint(frame)
    if mutation in {"median","metadata_int"}:
        metadata=json.loads((folder/"model_metadata.json").read_text())
        if mutation=="median": metadata["imputation_medians"][0]+=1.
        else: metadata["imputation_medians"][0]=int(metadata["imputation_medians"][0])
        write_json(folder/"model_metadata.json",metadata); receipt["fitted_model"]=metadata
    if mutation=="fit_bool": receipt["actual_fit_attempts_this_call"]=True
    if mutation=="ledger_float":
        attempts=json.loads((folder/"attempts.json").read_text()); attempts["ledger"][0]["fit_attempts"]=1.
        write_json(folder/"attempts.json",attempts)
    write_json(folder/"learning_receipt.json",receipt); _reseal(job)
    with pytest.raises(ValueError): audit_output(ops,done)


def test_structural_score_audit_limit_and_postcollection_immutable_hash(tmp_path):
    ops,s,p,_=setup(tmp_path); ex=ops.register(s,p); done=ops.execute(s,ex["id"])
    assert done["status"]=="completed",done.get("error")
    job=ops.root/"worker_jobs"/ex["id"]; folder=job/"learn_result"
    frame=pd.read_parquet(folder/"predictions.parquet"); frame.loc[0,"score"]+=.5
    frame.to_parquet(folder/"predictions.parquet",index=False)
    receipt=json.loads((folder/"learning_receipt.json").read_text()); receipt["prediction_outputs"]=_fingerprint(frame)
    write_json(folder/"learning_receipt.json",receipt); _reseal(job)
    # A coherently replaced finite score cannot be mathematically authenticated by
    # a structural-only initial audit. Never claim independent score reproduction.
    result=audit_output(ops,done)
    assert result["verified"] and not result["model_scores_independently_reproduced"]
    # Already collected evidence additionally binds the original result in DB.
    with pytest.raises(ValueError,match="Stored alpha audit differs"):
        ops.recover(ex["id"])


def test_store_auto_enforces_declared_fit_budget_without_optional_keyword(tmp_path):
    ops,s,p,_=setup(tmp_path,intent_budget=1); ex=ops.register(s,p)
    retained=copy.deepcopy(ex["proposal"])
    with pytest.raises(ValueError,match="fit budget exhausted"):
        ops.store.register(s,retained,"f"*64)
