"""Native target engineering, ten synthetic fits; zero market/account runs.

Five explicit transformed-target library fits, one independently frozen legacy fit,
and four native old/new protocol jobs. Mutations reuse saved models or reject
before fit. All attempt ledgers remain in the short pytest output tree.
"""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from src.alpharesearch.learning import LearningRunner, LearningSpec, LearningResult, KEYS, CLOCKS, _prepare, _prepared_parts, _fingerprint
from src.alpharesearch.learning_artifacts import save_learning_artifacts, audit_learning_artifacts
from src.alpharesearch.portfolio import build_score_portfolio, score_portfolio_spec
from src.alpharesearch.audit import audit_score_portfolio
from src.researchops.learn_experiments import protocol
from src.researchops.service import Research
from src.technical.artifacts import content_id, write_json
from tests.test_alpha_learning_artifacts import unequal_example, prohibit_fits
from tests.test_alpha_learning_preparation import frozen_module, legacy_data
from tests.test_alpha_learn_registration import setup

KINDS = ["raw_return", "date_centered", "date_zscore", "date_rank", "permuted_rank"]


def data_for(kind):
    data = list(unequal_example())
    cfg = data[0].to_dict()
    cfg.update(weighting="equal_decision_date", fit_cutoff="2024-01-04T23:00:00+08:00", predict_start="2024-01-05", predict_end="2024-01-12")
    # Forecasts use genuine weekly-open clock intervals; training remains tiny synthetic data.
    mapping = {"2024-01-08": ("2024-01-05", "2024-01-08", "2024-01-15"),
               "2024-01-09": ("2024-01-12", "2024-01-15", "2024-01-22")}
    for old, (day, entry, exit_) in mapping.items():
        mask = data[2].trade_date.eq(old)
        ids = set(data[2].loc[mask, "sample_id"])
        data[2].loc[mask, ["trade_date", "decision_at", "execution_at", "target_end_at"]] = [day, day+"T15:00:00+08:00", entry+"T09:30:00+08:00", exit_+"T09:30:00+08:00"]
        for attr in ("values", "missing"):
            frame = getattr(data[3].block, attr)
            selected = frame.sample_id.isin(ids)
            frame.loc[selected, "trade_date"] = day
            if attr == "values":
                frame.loc[selected, ["observed_end", "known_at"]] = day+"T15:00:00+08:00"
    # Forecast labels are deliberately absent: no target values are needed there.
    data[4] = data[4].loc[data[4].trade_date.le("2024-01-04")].copy()
    if kind != "legacy":
        cfg.update(schema="causal-learning-v2", training_target=dict(
            kind="date_rank" if kind == "permuted_rank" else kind,
            min_date_rows=2, scale_floor=1e-12, unusable_policy="reject"))
    if kind == "permuted_rank":
        cfg["negative_control"] = "within_date_label_permutation"
    data[0] = LearningSpec.from_dict(cfg)
    return data


def parts(case):
    return _prepared_parts(_prepare(*case["data"]))


def check(case, *, folder=None, receipt=None):
    X, y, w, P, _, trace, _ = parts(case)
    return audit_learning_artifacts(folder or case["folder"], X, y, w, P,
        case["result"].predictions, receipt or case["result"].receipt,
        target_trace=trace, max_checkpoint_bytes=1000000, trusted_local=True)


@pytest.fixture(scope="module")
def saved_cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("n0")
    cases = {}; ledger = []
    for name in KINDS + ["legacy"]:
        data = data_for(name)
        record = dict(name=name, status="running", declared_fit_intents=1, actual_fit_intents=None)
        ledger.append(record); write_json(root/"library_fit_ledger.json", ledger)
        runner = None
        try:
            if name == "legacy":
                project = Path(__file__).resolve().parents[1]
                code = root/"lg"
                shutil.copytree(project/"src", code/"src")
                fixture = project/"tests/fixtures/learning_preparation_legacy_v2.py"
                assert hashlib.sha256(fixture.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == "1382f69e84b44ab01f3b6e7a81bd84cabf4ef7e3e6d135ffe22a07f3fa0d729b"
                shutil.copyfile(fixture, code/"src/alpharesearch/learning.py")
                with frozen_module(code, "src.alpharesearch.learning") as old:
                    old_data = legacy_data(old, data)
                    runner = old.LearningRunner(max_calls=1, max_fit_intents=1)
                    result = runner.run(*old_data)
            else:
                runner = LearningRunner(max_calls=1, max_fit_intents=1)
                result = runner.run(*data, capture_fit=True)
            record["status"] = "completed"
        except BaseException:
            record["status"] = "failed"; raise
        finally:
            record.update(actual_fit_intents=None if runner is None else runner.fit_intents,
                ledger=[] if runner is None else runner.ledger)
            write_json(root/"library_fit_ledger.json", ledger)
        assert runner.fit_intents == 1 and runner.ledger[0]["fit_attempts"] == 1
        folder = root/("p"+str(len(cases))); folder.mkdir()
        if name != "legacy": save_learning_artifacts(result, folder, max_checkpoint_bytes=1000000)
        cases[name] = dict(result=result, folder=folder, data=data, runner=runner)
    native = []
    for name in ["v1", "v2", "center", "rank"]:
        folder = root/name; folder.mkdir()
        ops, session, proposal, _ = setup(folder, intent_budget=1)
        if name != "v1":
            proposal["spec"]["schema"] = "registered-alpha-learn-v2" if name == "v2" else "registered-alpha-learn-v3"
            proposal["spec"]["budget"]["max_checkpoint_bytes"] = 1000000
        if name in ("center", "rank"):
            proposal["spec"]["learning"].update(schema="causal-learning-v2", training_target=dict(
                kind="date_centered" if name == "center" else "date_rank",
                min_date_rows=2, scale_floor=1e-12, unusable_policy="reject"))
        record = dict(name=name, status="registering", experiment_id=None)
        native.append(record); write_json(root/"native_fit_ledger.json", native)
        ex = ops.register(session, proposal)
        record.update(status="executing", experiment_id=ex["id"]);write_json(root/"native_fit_ledger.json", native)
        terminal = ops.execute(session, ex["id"])
        record.update(status=terminal["status"], error=terminal.get("error"));write_json(root/"native_fit_ledger.json", native)
        assert terminal["status"] == "completed", terminal.get("error")
        cases[name] = dict(ops=ops, session=session, experiment=ops.store.experiment(ex["id"]),
            folder=ops.root/"worker_jobs"/ex["id"]/"learn_result")
    assert sum(row["actual_fit_intents"] for row in ledger) == 6
    assert len(native) == 4 and all(row["status"] == "completed" for row in native)
    write_json(root/"fit_total.json", dict(library_fit_intents=6, native_fit_intents=4,
        real_synthetic_fits=10, market_fits=0, account_runs=0))
    return cases


@pytest.mark.parametrize("name", KINDS)
def test_saved_pipeline_reproduces_raw_used_targets_and_scores_without_fit(saved_cases, name, monkeypatch):
    prohibit_fits(monkeypatch)
    case = saved_cases[name]; outcome = check(case)
    assert outcome["additional_audit_fits"] == 0 and outcome["raw_and_used_targets_independently_reconstructed"]
    trace = pd.read_parquet(case["folder"]/"training_trace.parquet")
    assert len(trace) == 7 and trace.mature_date_rows.tolist() == [4]*4+[3]*3
    np.testing.assert_allclose(trace.weight_used, [.875]*4+[7/6]*3, rtol=0, atol=1e-15)
    assert not trace.trade_date.eq("2024-01-04").any()
    np.testing.assert_allclose(trace.raw_target_return, [-.06,-.03,.03,.06,-.06,-.03,.03], rtol=0, atol=1e-15)
    receipt = case["result"].receipt
    assert receipt["score_unit"] == ("decimal_return" if name in ("raw_return", "date_centered") else "dimensionless")
    assert receipt["predicted_score_is_raw_return"] is (name == "raw_return")


def test_independent_arithmetic_ties_centering_and_rank_denominator(saved_cases):
    rank = parts(saved_cases["date_rank"])[1].to_numpy()
    np.testing.assert_allclose(rank, [-.5,-1/6,1/6,.5,-.5,0,.5], rtol=0, atol=1e-15)
    centered = parts(saved_cases["date_centered"])[1].to_numpy()
    np.testing.assert_allclose(centered, [-.06,-.03,.03,.06,-.04,-.01,.05], rtol=0, atol=1e-15)
    data = data_for("date_rank"); data[-1].loc[data[-1].sample_id.eq("2024-01-02-1"), "target_return"] = -.06
    used = _prepared_parts(_prepare(*data))[1].to_numpy()
    np.testing.assert_allclose(used[:4], [-1/3,-1/3,1/6,.5], rtol=0, atol=1e-15)
    z = parts(saved_cases["date_zscore"])[1]
    for _, group in z.groupby(level="trade_date"):
        assert group.mean() == pytest.approx(0, abs=1e-15) and group.std(ddof=0) == pytest.approx(1)


def test_raw_variant_matches_independently_frozen_legacy_numeric_predictions(saved_cases):
    new = saved_cases["raw_return"]["result"]; old = saved_cases["legacy"]["result"]
    pd.testing.assert_frame_equal(new.predictions.drop(columns=["fit_id", "prediction_id"]),
        old.predictions.drop(columns=["fit_id", "prediction_id"]), check_exact=True)
    assert new.receipt["fit_identity"]["schema"] == "causal-fit-identity-v2"
    assert old.receipt["fit_identity"]["schema"] == "causal-fit-identity-v1"


@pytest.mark.parametrize("name", KINDS)
def test_immature_and_future_labels_cannot_change_mature_targets_or_scores(saved_cases, name, monkeypatch):
    prohibit_fits(monkeypatch)
    case = saved_cases[name]; before = parts(case)
    data = copy.deepcopy(case["data"])
    data[-1].loc[data[-1].trade_date.eq("2024-01-04"), "target_return"] = 999999.
    future = data[2].loc[data[2].trade_date.ge("2024-01-05"), KEYS].copy()
    future["target_return"] = "future_value_not_parsed";future["label_start"]="invalid_future_clock"
    future["label_end"]="invalid_future_clock";future["label_known_at"]="invalid_future_clock";future["label_observed"]=False
    data[-1] = pd.concat([data[-1],future], ignore_index=True)
    after = _prepared_parts(_prepare(*data))
    for position in [0,1,2,3,4,5]:
        if isinstance(before[position],pd.Series):pd.testing.assert_series_equal(before[position],after[position],check_exact=True)
        else:pd.testing.assert_frame_equal(before[position],after[position],check_exact=True)
    assert before[-1] == after[-1]
    adapter = case["result"].fit_artifacts["adapter"]
    np.testing.assert_array_equal(adapter.predict(after[3]).to_numpy(),case["result"].predictions.score.to_numpy())


def test_raw_changes_with_same_rank_order_change_fit_identity(saved_cases):
    case=saved_cases["date_rank"];before=parts(case);data=copy.deepcopy(case["data"])
    mask=data[-1].trade_date.le("2024-01-03")
    data[-1].loc[mask,"target_return"]*=2
    after=_prepared_parts(_prepare(*data))
    pd.testing.assert_series_equal(before[1],after[1],check_exact=True)
    assert before[-1]["fit_id"] != after[-1]["fit_id"]
    assert before[-1]["fit_identity"]["target_trace"] != after[-1]["fit_identity"]["target_trace"]


def test_negative_control_permutes_only_transformed_values_within_mature_date(saved_cases):
    original=parts(saved_cases["date_rank"]);permuted=parts(saved_cases["permuted_rank"])
    pd.testing.assert_frame_equal(original[5],permuted[5],check_exact=True)
    for date,group in original[1].groupby(level="trade_date"):
        np.testing.assert_array_equal(np.sort(group),np.sort(permuted[1].xs(date,level="trade_date")))
    assert not original[1].equals(permuted[1])


@pytest.mark.parametrize("kind,reason",[("date_rank","constant"),("date_zscore","constant"),("date_rank","thin"),("date_centered","thin")])
def test_unusable_mature_groups_reject_before_any_fit(kind,reason,monkeypatch):
    prohibit_fits(monkeypatch);data=data_for(kind)
    if reason == "constant":data[-1].loc[data[-1].trade_date.eq("2024-01-02"),"target_return"] = .01
    else:
        cfg=data[0].to_dict();cfg["training_target"]["min_date_rows"]=4;data[0]=LearningSpec.from_dict(cfg)
    runner=LearningRunner(max_calls=1,max_fit_intents=1)
    with pytest.raises(ValueError,match="Unusable mature target group"):
        runner.run(*data,capture_fit=True)
    assert runner.fit_intents==0 and runner.ledger[0]["fit_attempts"]==0


def test_constant_centered_group_retains_zero_targets_without_omission():
    data=data_for("date_centered");data[-1].loc[data[-1].trade_date.eq("2024-01-02"),"target_return"] = .01
    X,y,*_= _prepared_parts(_prepare(*data))
    assert len(X)==7;np.testing.assert_array_equal(y.iloc[:4],np.zeros(4))


@pytest.mark.parametrize("mutation",["raw","transformed","count","clock"])
def test_sealed_mature_trace_mutation_rejects_before_fit(mutation,monkeypatch):
    prohibit_fits(monkeypatch);runner=LearningRunner(max_calls=1,max_fit_intents=1)
    token=runner._prepare_once(*data_for("date_rank"),capture_fit=True)
    trace=runner._prepared[token]["prepared"][5]
    column={"raw":"raw_target","transformed":"training_target","count":"mature_date_rows","clock":"label_known_at"}[mutation]
    trace.loc[0,column]+= pd.Timedelta(seconds=1) if mutation=="clock" else 1
    with pytest.raises(ValueError,match="preparation changed before fit"):runner._consume_prepared(token)
    assert runner.fit_intents==0


@pytest.mark.parametrize("field",["raw_target_return","transformed_target_before_control","target_return_used","mature_date_rows","label_known_at"])
def test_saved_target_mutation_fails_zero_fit_independent_audit(saved_cases,tmp_path,field,monkeypatch):
    prohibit_fits(monkeypatch);case=saved_cases["date_rank"]
    folder=tmp_path/"m";shutil.copytree(case["folder"],folder)
    trace=pd.read_parquet(folder/"training_trace.parquet")
    trace.loc[0,field]+=pd.Timedelta(seconds=1) if field=="label_known_at" else 1
    trace.to_parquet(folder/"training_trace.parquet",index=False)
    with pytest.raises((ValueError,AssertionError)):check(case,folder=folder)


@pytest.mark.parametrize("name",KINDS)
def test_real_saved_scores_pass_weekly_portfolio_and_independent_audit(saved_cases,name,monkeypatch):
    prohibit_fits(monkeypatch);result=saved_cases[name]["result"]
    refs=result.predictions[KEYS].copy();refs["reference_close"]=10.
    refs["reference_close_observed_at"]=result.predictions.decision_at
    refs["reference_close_known_at"]=result.predictions.decision_at
    refs["eligible"]=True;refs["eligibility_known_at"]=result.predictions.decision_at
    calendar=pd.bdate_range("2024-01-02","2024-01-22").strftime("%Y-%m-%d").tolist()
    spec=score_portfolio_spec(start="2024-01-05",end="2024-01-12",top_k=2)
    portfolio=build_score_portfolio(result,refs,calendar,spec)
    audit=audit_score_portfolio(result,refs,calendar,portfolio)
    assert audit["numeric_model_reproduction"] is False and audit["selected_rows"] == 4 and len(portfolio.decisions)==8


@pytest.mark.parametrize("mutation",["score_unit","raw_flag","target_unit","target_raw_flag","date_count","version"])
def test_resigned_semantic_target_forgery_rejected(saved_cases,mutation,monkeypatch):
    prohibit_fits(monkeypatch);case=saved_cases["date_rank"];result=copy.deepcopy(case["result"])
    if mutation=="score_unit":result.receipt["score_unit"]="decimal_return"
    if mutation=="raw_flag":result.receipt["predicted_score_is_raw_return"]=True
    if mutation=="target_unit":result.receipt["training_target_receipt"]["unit"]="decimal_return"
    if mutation=="target_raw_flag":result.receipt["training_target_receipt"]["predicted_score_is_raw_return"]=True
    if mutation=="date_count":result.receipt["training_target_receipt"]["usable_rows"]-=1
    if mutation=="version":result.receipt["fit_identity"]["schema"]="causal-fit-identity-v1"
    # Re-sign outer identities: semantic rejection cannot rely on a stale hash alone.
    result.receipt["fit_identity"]["training_target_receipt"]=copy.deepcopy(result.receipt["training_target_receipt"])
    result.receipt["fit_id"]=content_id(result.receipt["fit_identity"])
    result.receipt["prediction_id"]=content_id({k:result.receipt[k] for k in ("fit_id","protocol_id","prediction_inputs","prediction_missing","prediction_membership")})
    result.predictions["fit_id"]=result.receipt["fit_id"];result.predictions["prediction_id"]=result.receipt["prediction_id"]
    result.receipt["prediction_outputs"]=_fingerprint(result.predictions)
    refs=result.predictions[KEYS].copy();refs["reference_close"]=10.
    refs["reference_close_observed_at"]=refs["reference_close_known_at"]=result.predictions.decision_at
    refs["eligible"]=True;refs["eligibility_known_at"]=result.predictions.decision_at
    calendar=pd.bdate_range("2024-01-02","2024-01-22").strftime("%Y-%m-%d").tolist()
    policy=score_portfolio_spec(start="2024-01-05",end="2024-01-12",top_k=2)
    with pytest.raises(ValueError,match="target|units|versions"):build_score_portfolio(result,refs,calendar,policy)
    # Independent audit receives an otherwise valid original portfolio, plus the forged source.
    original=case["result"];original_refs=refs.copy()
    valid=build_score_portfolio(original,original_refs,calendar,policy)
    with pytest.raises(ValueError,match="target|units|versions|Target"):audit_score_portfolio(result,refs,calendar,valid)


@pytest.mark.parametrize("name",["v1","v2","center","rank"])
def test_native_frozen_recovery_and_checkpoint_audit_add_no_fits(saved_cases,name,monkeypatch):
    prohibit_fits(monkeypatch);case=saved_cases[name]
    recovered=Research(case["ops"].project,readonly=True).recover(case["experiment"]["id"])
    assert recovered["status"]=="completed" and recovered["result"]["additional_audit_fits"]==0
    if name in ("center","rank"):
        receipt=json.loads((case["folder"]/"learning_receipt.json").read_text())
        assert receipt["schema"]=="causal-learning-receipt-v3"
        assert (case["folder"]/"model_checkpoint/model.joblib").is_file()
        assert recovered["result"]["raw_and_used_targets_independently_reconstructed"]


@pytest.mark.parametrize("registered,native",[("registered-alpha-learn-v1","causal-learning-v2"),("registered-alpha-learn-v2","causal-learning-v2"),("registered-alpha-learn-v3","causal-learning-v1")])
def test_protocol_version_mismatch_rejected_without_registration(registered,native):
    data=data_for("raw_return" if native.endswith("v2") else "legacy")
    budget=dict(max_input_bytes=1000000,max_task_fit_intents=1)
    if registered!="registered-alpha-learn-v1":budget["max_checkpoint_bytes"]=1000000
    with pytest.raises(ValueError,match="protocol versions differ"):
        protocol(dict(schema=registered,learning=data[0].to_dict(),model=data[1].to_dict(),budget=budget))
