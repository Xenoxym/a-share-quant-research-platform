"""Persistent scored accounts use synthetic stubs, never fit market models."""
import copy
import json
from pathlib import Path
import shutil
import pandas as pd
import pytest
from .test_alpha_account_audit import inputs
from .test_researchops import TASK
from src.researchops.account_experiments import verify_registered, native_folder
from src.researchops.account_output_audit import audit_output
from src.researchops.service import Research
from src.researchops.resources import WorkerResources
from src.technical.artifacts import digest, write_json
PROJECT=Path(__file__).resolve().parents[1]

def setup(tmp_path,*,real_budget=4,synthetic_budget=4):
    project=tmp_path/"project";project.mkdir()
    for source in (PROJECT/"src").rglob("*"):
        if not source.is_file() or source.suffix not in {".py",".html",".js",".css"} or "__pycache__" in source.parts:continue
        dest=project/source.relative_to(PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    ops=Research(project); data,snapshot,account=inputs(ops.root)
    result,refs,calendar,policy=data;folder=project/"data/scored_inputs";folder.mkdir()
    paths={role:folder/name for role,name in (("predictions","predictions.parquet"),("learning_receipt","receipt.json"),("references","references.parquet"),("calendar","calendar.json"))}
    result.predictions.to_parquet(paths["predictions"],index=False);refs.to_parquet(paths["references"],index=False)
    write_json(paths["learning_receipt"],result.receipt);write_json(paths["calendar"],calendar)
    p=dict(kind="alpha_account",hypothesis="Synthetic durable scored account",expected_observation="two retained independently audited scenarios",falsification="clock/account/budget mismatch",timeout_seconds=60,
        spec=dict(schema="registered-alpha-account-v1",portfolio=policy.to_dict(),account=account.to_dict(),
            scenarios=["zero_transaction_cost","configured"],snapshot_id=snapshot.name,snapshot_manifest_hash=digest(snapshot/"manifest.json"),
            score_source=dict(kind="synthetic_stub",statement="engineering scores only; no fitted strategy"),
            budget=dict(max_input_bytes=1000000,max_task_account_intents=real_budget+synthetic_budget,max_real_account_intents=real_budget,max_synthetic_account_intents=synthetic_budget)),
        inputs=[dict(role=r,path=x.relative_to(project).as_posix(),sha256=digest(x),bytes=x.stat().st_size) for r,x in paths.items()])
    task=ops.create(dict(TASK,max_experiments=8));s=ops.store.claim(task["id"],"account-fixture-worker")
    return ops,s,p,folder

def test_duplicate_names_input_order_preserve_intent(tmp_path):
    ops,s,p,_=setup(tmp_path);ex=ops.register(s,p);q=copy.deepcopy(p)
    q["hypothesis"]="different display";q["inputs"].reverse();q["spec"]["portfolio"]["name"]="different_name";q["spec"]["account"]["strategy"]["name"]="different_name"
    assert ops.register(s,q)["id"]==ex["id"]
    assert verify_registered(ops,ex)["planned_accounts"]==2
    assert len(ops.store.get(s["task_id"])["experiments"])==1

@pytest.mark.parametrize("change",["scenario","budget_bool","provenance","output_scope","clock"])
def test_invalid_definitions_never_register(change,tmp_path):
    ops,s,p,folder=setup(tmp_path)
    if change=="scenario":p["spec"]["scenarios"]=["configured"]
    elif change=="budget_bool":p["spec"]["budget"]["max_task_account_intents"]=True
    elif change=="provenance":p["spec"]["score_source"]["kind"]="unverified_model"
    elif change=="output_scope":p["resources"]=WorkerResources().to_dict()
    else:
        path=folder/"predictions.parquet";f=pd.read_parquet(path);f.loc[0,"execution_at"]=f.loc[0,"decision_at"];f.to_parquet(path,index=False)
        for r in p["inputs"]:
            if r["role"]=="predictions":r.update(sha256=digest(path),bytes=path.stat().st_size)
    with pytest.raises(ValueError):ops.register(s,p)
    assert not ops.store.get(s["task_id"])["experiments"]

def test_unregistered_real_model_scores_rejected(tmp_path):
    ops,s,p,_=setup(tmp_path);p["spec"]["score_source"]=dict(kind="registered_learning",experiment_id="exp-"+"a"*12)
    with pytest.raises(ValueError):ops.register(s,p)
    assert not ops.store.get(s["task_id"])["experiments"]

def test_synthetic_budget_failure_cancel_restart_no_refund(tmp_path):
    ops,s,p,_=setup(tmp_path,synthetic_budget=4);ex=ops.register(s,p)
    ops.store.start(s,ex["id"]);ops.store.finish(ex["id"],"failed",error="retained fixture failure")
    q=copy.deepcopy(p);q["spec"]["portfolio"]["allocation"]=.5;q["spec"]["account"]["strategy"]["allocation"]=.5
    second=ops.register(s,q);ops.cancel(s,second["id"],"retained fixture cancellation")
    q["spec"]["portfolio"]["allocation"]=.6;q["spec"]["account"]["strategy"]["allocation"]=.6
    with pytest.raises(ValueError,match="intent budget exhausted"):Research(ops.project).register(s,q)
    q["spec"]["budget"].update(max_task_account_intents=9,max_synthetic_account_intents=5)
    with pytest.raises(ValueError,match="budget already frozen"):ops.register(s,q)
    assert [e["status"] for e in ops.store.get(s["task_id"])["experiments"]]==["failed","cancelled"]

def test_transaction_also_counts_plain_native_controls(tmp_path):
    ops,s,p,_=setup(tmp_path,real_budget=2);ops.register(s,p)
    control=dict(kind=None,scenarios=["zero_transaction_cost","configured"])
    first=ops.store.register(s,control,"fixture_control_1");ops.cancel(s,first["id"],"counts despite cancel")
    with pytest.raises(ValueError,match="intent budget exhausted"):ops.store.register(s,control,"fixture_control_2")

def test_direct_store_cannot_bypass_declared_budget(tmp_path):
    ops,s,p,_=setup(tmp_path,synthetic_budget=2);ops.register(s,p)
    direct=dict(kind="alpha_account",spec=p["spec"],planned_accounts=2)
    with pytest.raises(ValueError,match="intent budget exhausted"):ops.store.register(s,direct,"fixture_bypass")
    direct["planned_accounts"]=True
    with pytest.raises(ValueError,match="persisted account count"):ops.store.register(s,direct,"fixture_bool")

def test_earliest_freeze_failure_retained_without_retry(tmp_path,monkeypatch):
    ops,s,p,_=setup(tmp_path,synthetic_budget=2);original=Path.mkdir
    def fail(path,*args,**kwargs):
        if path.name=="code" and path.parent.name.startswith("exp-"):raise OSError("earliest fixture failure")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"mkdir",fail)
    with pytest.raises(OSError):ops.register(s,p)
    monkeypatch.setattr(Path,"mkdir",original)
    with pytest.raises(ValueError,match="never retry"):ops.register(s,p)
    ex=ops.store.get(s["task_id"])["experiments"][0];assert ex["status"]=="failed"
    assert not (ops.root/"worker_jobs"/ex["id"]).exists()

def test_existing_native_artifacts_never_start(tmp_path):
    ops,s,p,_=setup(tmp_path);ex=ops.register(s,p);cfg=json.loads((ops.root/"worker_jobs"/ex["id"]/"input.json").read_text())
    native=native_folder(cfg);native.mkdir(parents=True)
    with pytest.raises(ValueError,match="never repeat"):ops.execute(s,ex["id"])
    assert ops.store.experiment(ex["id"])["status"]=="planned"

def test_completed_recovery_submit_review_without_model_fit(tmp_path,monkeypatch):
    ops,s,p,_=setup(tmp_path);ex=ops.register(s,p);done=ops.execute(s,ex["id"])
    assert done["status"]=="completed",done.get("error")
    for module,classes in [("sklearn.linear_model",["Ridge","HuberRegressor","ElasticNet"]),("sklearn.ensemble",["HistGradientBoostingRegressor","RandomForestRegressor","ExtraTreesRegressor"])]:
        imported=__import__(module,fromlist=classes)
        for name in classes:monkeypatch.setattr(getattr(imported,name),"fit",lambda *a,**k:pytest.fail("account recovery must never fit"))
    other=Research(ops.project);assert other.recover(ex["id"])["id"]==ex["id"]
    audit=audit_output(other,done);assert audit["accounts"]==2 and audit["model_fits"]==0
    assert done["run_id"]==audit["run_id"] and (ops.root/"runs"/done["run_id"]/"report.html").is_file()
    assert audit["score_source"]=="synthetic_stub" and audit["model_scores_independently_reproduced"] is False
    assert set(audit["account_audit"]["scenarios"])=={"zero_transaction_cost","configured"}
    report=dict(summary="synthetic account only",findings=["No new fits"],limitations=["Not market performance"],next_steps=["finite real batch"])
    other.submit(s,report);other.review(s["task_id"],"fixture-separate-reviewer","defer","verification done")
    assert other.store.get(s["task_id"])["status"]=="deferred"

@pytest.mark.parametrize("mutation",["metric","source","ledger_bool","pointer","extra"])
def test_resealed_outputs_still_rejected(tmp_path,mutation):
    ops,s,p,_=setup(tmp_path);ex=ops.register(s,p);done=ops.execute(s,ex["id"]);assert done["status"]=="completed",done.get("error")
    job=ops.root/"worker_jobs"/ex["id"];folder=job/"account_result";cfg=json.loads((job/"input.json").read_text());native=native_folder(cfg)
    if mutation=="metric":
        x=json.loads((native/"result.json").read_text());x["portfolios"][0]["metrics"]["total_return"]+=.25;write_json(native/"result.json",x)
    elif mutation=="source":
        f=pd.read_parquet(native/"alpha_predictions.parquet");f.loc[0,"score"]+=1.;f.to_parquet(native/"alpha_predictions.parquet",index=False)
    elif mutation=="ledger_bool":
        x=json.loads((folder/"attempts.json").read_text());x["model_fits"]=False;write_json(folder/"attempts.json",x)
    elif mutation=="pointer":write_json(folder/"native_pointer.json",dict(run_id="20240101T000000-12345678",manifest_hash=digest(native/"manifest.json")))
    else:(folder/"unexpected.json").write_text("{}")
    nm=json.loads((native/"manifest.json").read_text());nm["artifacts"]={k:digest(native/k) for k in nm["artifacts"]};write_json(native/"manifest.json",nm)
    if mutation!="pointer":write_json(folder/"native_pointer.json",dict(run_id=native.name,manifest_hash=digest(native/"manifest.json")))
    m=json.loads((folder/"manifest.json").read_text());m["native_manifest_hash"]=digest(native/"manifest.json");m["artifacts"]={k:digest(folder/k) for k in m["artifacts"]};write_json(folder/"manifest.json",m)
    receipt=json.loads((job/"receipt.json").read_text());receipt["result_manifest_hash"]=digest(folder/"manifest.json");write_json(job/"receipt.json",receipt)
    with pytest.raises((ValueError,AssertionError)):audit_output(ops,done)

def test_fixed_run_identity_cannot_escape_or_reuse(tmp_path):
    from src.technical.runner import run
    with pytest.raises(ValueError,match="Registered run identity"):
        run(PROJECT,root=tmp_path,registered_run_id="../escape")
    ops,s,p,_=setup(tmp_path);ex=ops.register(s,p);done=ops.execute(s,ex["id"]);assert done["status"]=="completed",done.get("error")
    with pytest.raises(ValueError,match="already started"):ops.execute(s,ex["id"])


def test_external_native_outputs_obey_sampled_byte_budget(tmp_path):
    ops,s,p,_=setup(tmp_path)
    asset=ops.project/"src/technical/assets/index.html"
    asset.write_bytes(asset.read_bytes()+b" "*2000000)
    p["resources"]=WorkerResources(max_output_bytes=1048576,output_scope="scored_account_result").to_dict()
    ex=ops.register(s,p)
    done=ops.execute(s,ex["id"])
    assert done["status"]=="failed" and "output" in done["error"].lower()
    cfg=json.loads((ops.root/"worker_jobs"/ex["id"]/"input.json").read_text())
    assert native_folder(cfg).exists()
    assert len(ops.store.get(s["task_id"])["experiments"])==1
    with pytest.raises(ValueError,match="already started"):ops.execute(s,ex["id"])


@pytest.mark.parametrize("evidence",["input","native","snapshot"])
def test_changes_during_audit_never_first_collect_completed(tmp_path,monkeypatch,evidence):
    import src.researchops.account_output_audit as dispatch
    ops,s,p,folder=setup(tmp_path);ex=ops.register(s,p)
    original=dispatch.importlib.import_module;changed=[]
    def intercept(name,*args,**kwargs):
        module=original(name,*args,**kwargs)
        if name.startswith("_frozen_account_audit_") and name.endswith(".account_output_audit"):
            audit=module.audit_score_account_run
            def mutate_after_read(native):
                result=audit(native)
                target={"input":folder/"predictions.parquet","native":native/"configured/equity.parquet",
                    "snapshot":ops.root/"snapshots"/p["spec"]["snapshot_id"]/"bars.parquet"}[evidence]
                target.write_bytes(target.read_bytes()+b"modified_after_audit")
                changed.append(str(target))
                return result
            module.audit_score_account_run=mutate_after_read
        return module
    monkeypatch.setattr(dispatch.importlib,"import_module",intercept)
    try:done=ops.execute(s,ex["id"])
    except (ValueError,AssertionError):done=ops.store.experiment(ex["id"])
    assert changed, "Mutation must occur after successful independent numerical audit"
    assert done["status"]=="failed" and not done.get("run_id")
    assert not (ops.root/"worker_jobs"/ex["id"]/"alpha_audit.json").exists()


def test_registered_fitted_learning_scores_bind_account_without_refit(tmp_path,monkeypatch):
    from .test_alpha_learning import example
    from src.alpharesearch.learning import LearningSpec
    from src.researchops.screen_experiments import assembly_reference
    ops,s,p,folder=setup(tmp_path)
    spec,model,decisions,assembly,labels=example()
    for old,day,entry,end in [("2024-01-08","2024-01-05","2024-01-08","2024-01-15"),("2024-01-09","2024-01-12","2024-01-15","2024-01-22")]:
        mask=decisions.trade_date.eq(old)
        decisions.loc[mask,"trade_date"]=day;decisions.loc[mask,"decision_at"]=day+"T15:30:00+08:00"
        decisions.loc[mask,"execution_at"]=entry+"T09:30:00+08:00";decisions.loc[mask,"target_end_at"]=end+"T09:30:00+08:00"
        for frame in (assembly.block.values,assembly.block.missing):
            selected=frame.trade_date.eq(old);frame.loc[selected,"trade_date"]=day
            if "known_at" in frame:
                frame.loc[selected,["observed_end","known_at"]]=day+"T15:30:00+08:00"
    doc=spec.to_dict();doc.update(train_end="2024-01-03",predict_start="2024-01-05",predict_end="2024-01-12",fit_cutoff="2024-01-04T23:00:00+08:00")
    spec=LearningSpec.from_dict(doc);raw=folder/"learning";raw.mkdir();paths={}
    for role,frame in [("assembly_values",assembly.block.values),("assembly_missing",assembly.block.missing),("decisions",decisions),("labels",labels)]:
        paths[role]=raw/(role+".parquet");frame.to_parquet(paths[role],index=False)
    for role,value in [("registry",assembly.registry.to_dict()),("assembly_definition",dict(assembly.block.metadata,units=assembly.block.units))]:
        paths[role]=raw/(role+".json");write_json(paths[role],value)
    paths["assembly_receipt"]=raw/"assembly_receipt.json";write_json(paths["assembly_receipt"],assembly_reference(assembly,{r:digest(paths[r]) for r in ("assembly_values","assembly_missing","assembly_definition")}))
    proposal=dict(kind="alpha_learn",hypothesis="Synthetic software actual-fit bridge",expected_observation="Reuse persisted original model output",falsification="Any refit/source mismatch",
        inputs=[dict(role=r,path=x.relative_to(ops.project).as_posix(),sha256=digest(x),bytes=x.stat().st_size) for r,x in paths.items()],timeout_seconds=60,
        spec=dict(schema="registered-alpha-learn-v1",learning=spec.to_dict(),model=model.to_dict(),budget=dict(max_input_bytes=1000000,max_task_fit_intents=1)))
    learn=ops.register(s,proposal);completed=ops.execute(s,learn["id"]);assert completed["status"]=="completed",completed.get("error")
    source=ops.root/"worker_jobs"/learn["id"]/"learn_result"
    for role,name in (("predictions","predictions.parquet"),("learning_receipt","learning_receipt.json")):
        path=source/name
        for entry in p["inputs"]:
            if entry["role"]==role:entry.update(path=path.relative_to(ops.project).as_posix(),sha256=digest(path),bytes=path.stat().st_size)
    p["spec"]["score_source"]=dict(kind="registered_learning",experiment_id=learn["id"])
    for module,classes in [("sklearn.linear_model",["Ridge","HuberRegressor","ElasticNet"]),("sklearn.ensemble",["HistGradientBoostingRegressor","RandomForestRegressor","ExtraTreesRegressor"])]:
        imported=__import__(module,fromlist=classes)
        for name in classes:monkeypatch.setattr(getattr(imported,name),"fit",lambda *a,**k:pytest.fail("bridge must not refit"))
    account=ops.register(s,p);done=ops.execute(s,account["id"]);assert done["status"]=="completed",done.get("error")
    assert done["result"]["score_source"]=="registered_learning" and done["result"]["additional_audit_fits"]==0
    assert Research(ops.project).recover(account["id"])["run_id"]==done["run_id"]
    assert len(ops.store.get(s["task_id"])["experiments"])==2
