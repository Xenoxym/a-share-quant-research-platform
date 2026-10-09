"""Proposed v2 persistence counterexamples: exactly eight synthetic fit intentions.

Unexecuted candidate. The module fixture owns five library fits and three registered
fixture fits; mutations reuse the fitted artifacts, not repeated model training.
"""
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pytest
from src.alpharesearch.assembly import FeatureAssembly
from src.alpharesearch.features.base import FeatureBlock
from src.alpharesearch.learning import LearningRunner, LearningSpec, _prepare
from src.alpharesearch.learning_artifacts import save_learning_artifacts, audit_learning_artifacts
from src.alpharesearch.models import model_spec, TabularRegressor
from src.researchops.learn_experiments import protocol
from src.researchops.learn_output_audit import audit_output
from src.technical.artifacts import digest, write_json
from tests.test_alpha_learning import example
from tests.test_alpha_learn_registration import setup


def unequal_example():
    spec, model, p, assembly, labels = example()
    remove = p.trade_date.eq("2024-01-03") & p.stock_code.eq("600003.SH")
    ids = set(p.loc[remove, "sample_id"])
    p = p.loc[~remove].reset_index(drop=True)
    labels = labels.loc[~labels.sample_id.isin(ids)].reset_index(drop=True)
    block = FeatureBlock(assembly.block.values.loc[~assembly.block.values.sample_id.isin(ids)].reset_index(drop=True),
        assembly.block.missing.loc[~assembly.block.missing.sample_id.isin(ids)].reset_index(drop=True),
        assembly.block.units, assembly.block.metadata)
    return spec, model, p, FeatureAssembly(block, assembly.registry, assembly.source_bindings), labels


@pytest.fixture(scope="module")
def saved_cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("eight_fits")
    cases = {}; library_records = []
    for name in ["uniform", "weighted", "permuted", "forest", "legacy"]:
        data = list(unequal_example())
        cfg = data[0].to_dict()
        if name == "weighted": cfg["weighting"] = "equal_decision_date"
        if name == "permuted": cfg["negative_control"] = "within_date_label_permutation"
        data[0] = LearningSpec.from_dict(cfg)
        if name == "forest": data[1] = model_spec("random_forest", params={"n_estimators":3,"max_depth":2,"min_samples_leaf":1})
        runner = LearningRunner(max_calls=1, max_fit_intents=1)
        record=dict(name=name,status="running",declared_fit_intents=1,charged_fit_intents=1,actual_fit_intents=None,finalized=False,ledger=[])
        library_records.append(record)
        write_json(root/"library_fit_ledger.json",dict(schema="synthetic-library-fit-ledger-v1",records=library_records))
        try:
            result = runner.run(*data, capture_fit=name != "legacy")
            record["status"]="completed"
        except BaseException:
            record["status"]="failed"
            raise
        finally:
            record.update(actual_fit_intents=runner.fit_intents,finalized=True,ledger=runner.ledger)
            write_json(root/"library_fit_ledger.json",dict(schema="synthetic-library-fit-ledger-v1",records=library_records))
        assert runner.fit_intents == 1 and runner.ledger[0]["fit_attempts"] == 1
        folder = root/name; folder.mkdir()
        if name != "legacy": save_learning_artifacts(result, folder, max_checkpoint_bytes=1000000)
        cases[name] = dict(result=result, folder=folder, data=data, runner=runner)
    for name in ["registered", "registered_v1", "registered_failure"]:
        registered=root/name;registered.mkdir()
        ops,session,proposal,_=setup(registered,intent_budget=1)
        if name!="registered_v1":
            proposal["spec"]["schema"]="registered-alpha-learn-v2"
            proposal["spec"]["budget"]["max_checkpoint_bytes"]=1024 if name=="registered_failure" else 1000000
        ex=ops.register(session,proposal)
        terminal=ops.execute(session,ex["id"]);ex=ops.store.experiment(ex["id"])
        expected="failed" if name=="registered_failure" else "completed"
        assert terminal["status"]==expected,terminal.get("error")
        assert ex["status"]==expected,ex.get("error")
        cases[name]=dict(ops=ops,session=session,proposal=proposal,experiment=ex,
            folder=ops.root/"worker_jobs"/ex["id"]/"learn_result")
    yield cases
    assert sum(c["runner"].fit_intents for c in cases.values() if "runner" in c)==5
    for name in ["registered","registered_v1","registered_failure"]:
        c=cases[name];assert len(c["ops"].store.get(c["session"]["task_id"])["experiments"])==1



def prepared(case):
    X,y,w,P,_,_ = _prepare(*case["data"])
    return X,y,w,P


def check(case, *, folder=None, receipt=None, predictions=None):
    X,y,w,P = prepared(case)
    return audit_learning_artifacts(folder or case["folder"], X,y,w,P,
        case["result"].predictions if predictions is None else predictions,
        case["result"].receipt if receipt is None else receipt,
        max_checkpoint_bytes=1000000,trusted_local=True)


@pytest.mark.parametrize("name", ["uniform", "weighted", "permuted", "forest"])
def test_local_reload_reproduces_actual_train_and_forecast_without_fit(saved_cases, name, monkeypatch):
    prohibit_fits(monkeypatch)
    assert check(saved_cases[name])["model_scores_independently_reproduced"]


def test_uniform_weights_and_immature_labels_are_visible(saved_cases):
    c=saved_cases["uniform"]; t=pd.read_parquet(c["folder"]/"training_trace.parquet")
    assert len(t)==7 and t.weight_used.eq(1.).all()
    assert "2024-01-04" not in set(t.trade_date)
    assert c["result"].receipt["observed_but_immature_labels"]==4


def test_equal_date_weights_are_actual_numbers_not_metadata_only(saved_cases):
    t=pd.read_parquet(saved_cases["weighted"]["folder"]/"training_trace.parquet")
    a=t.loc[t.trade_date.eq("2024-01-02"),"weight_used"]
    b=t.loc[t.trade_date.eq("2024-01-03"),"weight_used"]
    np.testing.assert_array_equal(a.to_numpy(),np.full(4, .875))
    np.testing.assert_allclose(b.to_numpy(),np.full(3, 7/6),rtol=0,atol=1e-15)
    assert a.sum()==pytest.approx(b.sum())


def test_permutation_saves_used_targets_and_preserves_within_date_multisets(saved_cases):
    original=pd.read_parquet(saved_cases["uniform"]["folder"]/"training_trace.parquet")
    permuted=pd.read_parquet(saved_cases["permuted"]["folder"]/"training_trace.parquet")
    assert not original.target_return_used.equals(permuted.target_return_used)
    for day in set(original.trade_date):
        np.testing.assert_array_equal(np.sort(original.loc[original.trade_date.eq(day),"target_return_used"]),
                                      np.sort(permuted.loc[permuted.trade_date.eq(day),"target_return_used"]))


def test_future_outcomes_do_not_change_training_or_prediction_inputs(saved_cases):
    c=saved_cases["uniform"]; data=list(c["data"]); lab=data[4].copy()
    pred=lab.trade_date.ge(data[0].to_dict()["predict_start"])
    lab.loc[pred,"target_return"]=99999.;lab.loc[pred,"label_known_at"]="2099-01-01T00:00:00Z"
    data[4]=lab
    left=_prepare(*c["data"]);right=_prepare(*data)
    for a,b in zip(left[:4],right[:4]):
        if a is None:assert b is None
        elif isinstance(a,pd.Series):pd.testing.assert_series_equal(a,b,check_exact=True)
        else:pd.testing.assert_frame_equal(a,b,check_exact=True)


def test_default_api_preserves_legacy_result_and_no_fit_artifact(saved_cases):
    c=saved_cases["legacy"];assert c["result"].fit_artifacts is None
    assert not list(c["folder"].iterdir())
    pd.testing.assert_frame_equal(c["result"].predictions,saved_cases["uniform"]["result"].predictions,check_exact=True)


def test_capture_type_rejected_before_fit():
    runner=LearningRunner(max_calls=1,max_fit_intents=1)
    with pytest.raises(ValueError,match="explicit boolean"):runner.run(*unequal_example(),capture_fit=1)
    assert runner.fit_intents==0 and runner.ledger[0]["fit_attempts"]==0


@pytest.mark.parametrize("mutation",["feature","target","weight","train_score","forecast_score",
    "payload","payload_receipt","checkpoint_extra","fit_id","model_metadata","artifact_rows","artifact_hash"])
def test_tampered_values_or_model_cannot_pass_reproduction(saved_cases,tmp_path,mutation):
    c=saved_cases["uniform"];folder=tmp_path/"copy";shutil.copytree(c["folder"],folder)
    receipt=copy.deepcopy(c["result"].receipt);pred=c["result"].predictions.copy()
    if mutation in {"feature","target","weight","train_score"}:
        t=pd.read_parquet(folder/"training_trace.parquet")
        field={"feature":"test.signal","target":"target_return_used","weight":"weight_used","train_score":"train_score"}[mutation]
        t.loc[0,field]+=1.;t.to_parquet(folder/"training_trace.parquet",index=False)
    elif mutation=="forecast_score":pred.loc[0,"score"]+=1.
    elif mutation=="payload":
        p=folder/"model_checkpoint/model.joblib";raw=p.read_bytes();p.write_bytes(bytes([raw[0]^1])+raw[1:])
    elif mutation in {"payload_receipt","checkpoint_extra","artifact_rows","artifact_hash"}:
        p=folder/"training_artifact_receipt.json";v=json.loads(p.read_bytes())
        if mutation=="payload_receipt":v["checkpoint"]["model_sha256"]="0"*64
        if mutation=="checkpoint_extra":v["checkpoint"]["arbitrary"]="not allowed"
        if mutation=="artifact_rows":v["rows"]+=1
        if mutation=="artifact_hash":v["actual_training_trace"]["sha256"]="0"*64
        write_json(p,v)
    elif mutation=="fit_id":receipt["fit_id"]="0"*64
    elif mutation=="model_metadata":receipt["fitted_model"]["training_rows"]+=1
    with pytest.raises((ValueError,AssertionError)):
        check(c,folder=folder,receipt=receipt,predictions=pred)


def test_existing_partial_output_is_preserved_without_overwrite(saved_cases):
    c=saved_cases["uniform"];before={p.relative_to(c["folder"]).as_posix():digest(p) for p in c["folder"].rglob("*") if p.is_file()}
    with pytest.raises(ValueError,match="already exists"):
        save_learning_artifacts(c["result"],c["folder"],max_checkpoint_bytes=1000000)
    assert before=={p.relative_to(c["folder"]).as_posix():digest(p) for p in c["folder"].rglob("*") if p.is_file()}


def test_checkpoint_bytecap_leaves_partial_evidence_and_no_additional_fit(saved_cases,tmp_path):
    c=saved_cases["uniform"];folder=tmp_path/"small";folder.mkdir()
    with pytest.raises(ValueError,match="byte budget"):
        save_learning_artifacts(c["result"],folder,max_checkpoint_bytes=1024)
    assert (folder/"training_trace.parquet").is_file() and (folder/"model_checkpoint/model.joblib").is_file()
    assert c["runner"].fit_intents==1


@pytest.mark.parametrize("mode",["v1_extra","v2_missing","v2_bool","v2_small","v2_large"])
def test_artifact_budgets_strict_before_fit(mode):
    spec,model,*_=unequal_example()
    v=dict(schema="registered-alpha-learn-v2",learning=spec.to_dict(),model=model.to_dict(),budget=dict(max_input_bytes=1000000,max_task_fit_intents=1,max_checkpoint_bytes=1000000))
    if mode=="v1_extra":v["schema"]="registered-alpha-learn-v1"
    if mode=="v2_missing":v["budget"].pop("max_checkpoint_bytes")
    if mode=="v2_bool":v["budget"]["max_checkpoint_bytes"]=True
    if mode=="v2_small":v["budget"]["max_checkpoint_bytes"]=1023
    if mode=="v2_large":v["budget"]["max_checkpoint_bytes"]=50000001
    with pytest.raises(ValueError):protocol(v)


def test_registered_v2_freezes_real_artifacts_and_reloads_without_fit(saved_cases,monkeypatch):
    prohibit_fits(monkeypatch)
    c=saved_cases["registered"];audit=audit_output(c["ops"],c["experiment"])
    assert audit["schema"]=="registered-alpha-learn-audit-v2"
    assert audit["additional_audit_fits"]==0 and audit["model_scores_independently_reproduced"]
    assert audit["training_values_independently_reconstructed"]
    attempts=json.loads((c["folder"]/"attempts.json").read_bytes())
    assert attempts["ledger"][0]["fit_attempts"]==1
    trace=pd.read_parquet(c["folder"]/"training_trace.parquet");assert len(trace)==8


def prohibit_fits(monkeypatch):
    from sklearn.linear_model import Ridge,ElasticNet,HuberRegressor
    from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
    def prohibited(*args,**kwargs):raise AssertionError("Audit must not train any adapter or estimator")
    for cls in (TabularRegressor,Ridge,ElasticNet,HuberRegressor,RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor):
        monkeypatch.setattr(cls,"fit",prohibited)


@pytest.mark.parametrize("mutation",["rows_float","fits_bool","saved_int","bytes_float"])
def test_receipt_scalar_type_substitution_is_rejected_without_fit(saved_cases,tmp_path,mutation,monkeypatch):
    prohibit_fits(monkeypatch)
    c=saved_cases["uniform"];folder=tmp_path/"typed";shutil.copytree(c["folder"],folder)
    p=folder/"training_artifact_receipt.json";v=json.loads(p.read_bytes())
    if mutation=="rows_float":v["rows"]=float(v["rows"])
    if mutation=="fits_bool":v["additional_model_fits"]=False
    if mutation=="saved_int":v["train_scores_saved"]=1
    if mutation=="bytes_float":v["checkpoint"]["model_bytes"]=float(v["checkpoint"]["model_bytes"])
    write_json(p,v)
    with pytest.raises(ValueError):check(c,folder=folder)


def test_untrusted_caller_rejected_before_read_or_deserialization(saved_cases,monkeypatch):
    from src.alpharesearch import learning_artifacts
    def prohibited(*args,**kwargs):raise AssertionError("Must refuse before loading")
    monkeypatch.setattr(learning_artifacts,"load_checkpoint",prohibited)
    c=saved_cases["uniform"];X,y,w,P=prepared(c)
    monkeypatch.setattr(learning_artifacts.pd,"read_parquet",prohibited)
    for trusted in [False,None,1]:
        with pytest.raises(ValueError,match="Explicit trusted_local"):
            audit_learning_artifacts(c["folder"],X,y,w,P,c["result"].predictions,c["result"].receipt,
                max_checkpoint_bytes=1000000,trusted_local=trusted)


@pytest.mark.parametrize("name",["registered","registered_v1"])
def test_registered_recovery_submit_review_do_not_refit(saved_cases,name,monkeypatch):
    prohibit_fits(monkeypatch)
    c=saved_cases[name];ops=c["ops"];s=c["session"]
    assert ops.recover(c["experiment"]["id"])["status"]=="completed"
    report=dict(summary="Synthetic persistence verification only",findings=["Actual one fit retained"],
        limitations=["No real data or account"],next_steps=["Continue bounded research"])
    ops.submit(s,report)
    ops.review(s["task_id"],"independent-software-fixture-role","defer","Completed synthetic compatibility check")
    assert ops.store.get(s["task_id"])["status"]=="deferred"


def test_registered_v1_preserves_strict_legacy_output_and_audit(saved_cases,monkeypatch):
    prohibit_fits(monkeypatch)
    c=saved_cases["registered_v1"]
    assert {p.relative_to(c["folder"]).as_posix() for p in c["folder"].rglob("*") if p.is_file()}=={
        "predictions.parquet","learning_receipt.json","model_metadata.json","attempts.json","manifest.json"}
    audit=audit_output(c["ops"],c["experiment"])
    assert audit["schema"]=="registered-alpha-learn-audit-v1"
    assert audit["additional_audit_fits"]==0
    assert not audit["model_scores_independently_reproduced"]


def test_registered_post_fit_byte_failure_retains_actual_ledger_and_partial_output(saved_cases,monkeypatch):
    prohibit_fits(monkeypatch)
    c=saved_cases["registered_failure"];ops=c["ops"];s=c["session"]
    attempt=json.loads((c["folder"]/"attempts.json").read_bytes())
    assert c["experiment"]["status"]=="failed"
    assert attempt["failure_phase"]=="post_fit_output"
    assert attempt["attempts"][0]["status"]=="failed"
    assert len(attempt["ledger"])==1 and attempt["ledger"][0]["fit_attempts"]==1
    assert (c["folder"]/"training_trace.parquet").is_file()
    assert (c["folder"]/"model_checkpoint/model.joblib").is_file()
    assert not (c["folder"]/"manifest.json").exists()
    before={p.relative_to(c["folder"]).as_posix():digest(p) for p in c["folder"].rglob("*") if p.is_file()}
    p=copy.deepcopy(c["proposal"]);p["spec"]["model"]["params"]["alpha"]=20.
    with pytest.raises(ValueError,match="fit budget exhausted"):ops.register(s,p)
    assert before=={p.relative_to(c["folder"]).as_posix():digest(p) for p in c["folder"].rglob("*") if p.is_file()}
