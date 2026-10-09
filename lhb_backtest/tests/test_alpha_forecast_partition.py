"""Proposed forecast boundary counterexamples, with eight toy formula intentions.

Only the module fixture executes formulas: four native batches, two candidates
each. All other tests inspect protocols or mutate retained output, with no fits
or accounts. This file is an unexecuted private candidate until SOURCE review.
"""
import copy
import json

import pandas as pd
import pytest

from src.alpharesearch.batch import AlphaBatchSpec
from src.alpharesearch.batch_partition import execution_partition, validate_forecast_selection
from src.alpharesearch.dsl import Expr as E, compile_expression
from src.alpharesearch.features.primitives import daily_primitives
from src.alpharesearch.feature_storage import read_shared_index, read_compact_candidate
from src.alpharesearch.registry import definitions_for_block, materialization_receipt
from src.researchops.alpha_experiments import inspect_inputs
from src.researchops.alpha_output_audit import audit_output
from src.technical.artifacts import digest, write_json
from tests.test_alpha_batch_registration import setup
from tests.test_alpha_dsl import bars


def reseal_input(proposal, folder, name):
    path = folder / name
    record = next(r for r in proposal["inputs"] if r["path"].endswith("/" + name))
    record.update(sha256=digest(path), bytes=path.stat().st_size)


def forecast_fixture(root, *, lag=False):
    ops, session, proposal, registry, folder = setup(root, max_experiments=1)
    spec = proposal["spec"]
    spec.update(schema="alpha-feature-batch-v3", storage_layout="shared_index_v1",
                execution_partition="forecast_features", forecast_warmup_start="2024-01-05",
                training_selection=dict(receipt_role="selection_receipt", summary_role="selection_summary",
                    library_role="selection_library", source_receipt_role="selection_source"))
    second = copy.deepcopy(spec["candidates"][0])
    second["candidate_id"] = "second_counted"
    if lag:
        second["expression"] = compile_expression(E.call("cs_rank", E.call("lag", E.ref("daily.close"), periods=1)), registry).to_dict()
    spec["candidates"].append(second)
    if lag: spec["forecast_warmup_start"]="2024-01-03"
    extra = []
    for day in ("2024-01-08", "2024-01-09", "2024-01-10"):
        for code, close in [("000001.SZ", 10.), ("600001.SH", 11.)]:
            extra.append(dict(trade_date=day, stock_code=code, open=10., high=12., low=8., close=close,
                pre_close=10., volume=1000., amount=close*1000., high_limit=12., low_limit=8., limit_price_valid=True))
    block = daily_primitives(pd.concat([bars(), pd.DataFrame(extra)], ignore_index=True), source_id="fixture_daily", version_id="q1")
    definitions = definitions_for_block(block)
    assert {d.definition_id for d in definitions} == {d.definition_id for d in registry.definitions}
    block.values.to_parquet(folder/"values.parquet", index=False)
    block.missing.to_parquet(folder/"missing.parquet", index=False)
    receipt = materialization_receipt(block, definitions,
        artifact_hashes={n:digest(folder/n) for n in ("values.parquet", "missing.parquet")})
    write_json(folder/"receipt.json", receipt)
    for binding in spec["bindings"].values():
        binding["materialization_id"] = receipt["materialization_id"]
    p = block.values[["trade_date", "stock_code"]].copy()
    p["sample_id"] = p.trade_date+p.stock_code
    p["decision_at"] = p.trade_date+"T15:10:00+08:00"
    p.to_parquet(folder/"membership.parquet", index=False)
    for name in ("values.parquet", "missing.parquet", "receipt.json", "membership.parquet"):
        reseal_input(proposal, folder, name)
    library = copy.deepcopy(spec["candidates"])
    rejected = copy.deepcopy(library[0]); rejected["candidate_id"] = "rejected_kept"
    library.append(rejected)
    summary = dict(at="2024-02-01T00:00:00+08:00", new_fits=0, new_accounts=0,
        all_trials=[dict(channel_id=c["candidate_id"], family=c["family"], promoted=i<2,
            quality_passed=i<2, proxy_complete=True) for i,c in enumerate(library)])
    source = dict(schema="formula-training-leaves-v1", universe_id=spec["universe_id"],
        development_start=spec["ranges"]["development_start"], development_end=spec["ranges"]["development_end"],
        fit_cutoff="2024-01-07T15:30:00+08:00", label_filter_for_membership=False)
    for name, data in [("summary.json", summary), ("library.json", library), ("source.json", source)]:
        write_json(folder/name, data)
    selection = dict(schema="train-window-formula-selection-v1", dataset_id=spec["dataset_id"],
        universe_id=spec["universe_id"], development_start=spec["ranges"]["development_start"],
        development_end=spec["ranges"]["development_end"], training_label_cutoff=source["fit_cutoff"],
        selected_at="2024-02-01T00:01:00+08:00", historical_selection_certified=False,
        summary_sha256=digest(folder/"summary.json"), library_sha256=digest(folder/"library.json"),
        source_receipt_sha256=digest(folder/"source.json"), selected_candidate_ids=sorted(c["candidate_id"] for c in library[:2]))
    write_json(folder/"selection.json", selection)
    for role, name in [("selection_receipt","selection.json"), ("selection_summary","summary.json"),
                       ("selection_library","library.json"), ("selection_source","source.json")]:
        path=folder/name
        proposal["inputs"].append(dict(role=role, path=path.relative_to(ops.project).as_posix(), sha256=digest(path), bytes=path.stat().st_size))
    return ops, session, proposal, registry, folder


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    root=tmp_path_factory.mktemp("eight_formula_intentions")
    cases={}; ledger=[]
    for mode in ("v1", "v2", "v3", "v3_lag"):
        target=root/mode; target.mkdir()
        if mode.startswith("v3"):
            ops, session, proposal, registry, folder=forecast_fixture(target, lag=mode=="v3_lag")
        else:
            ops, session, proposal, registry, folder=setup(target, max_experiments=1)
            proposal["spec"]["candidates"].append(dict(copy.deepcopy(proposal["spec"]["candidates"][0]), candidate_id="second_counted"))
            if mode=="v2": proposal["spec"].update(schema="alpha-feature-batch-v2", storage_layout="shared_index_v1")
        record=dict(mode=mode, declared_formula_intents=2, charged_formula_intents=2,
                    observed_attempts=None, status="running", finalized=False)
        ledger.append(record); write_json(root/"formula_intent_ledger.json", dict(records=ledger))
        ex=None
        try:
            ex=ops.register(session, proposal)
            terminal=ops.execute(session, ex["id"])
            ex=ops.store.experiment(ex["id"])
            assert terminal["status"]==ex["status"]=="completed", ex.get("error")
            record["status"]="completed"
        except BaseException:
            record["status"]="failed"; raise
        finally:
            if ex is not None:
                path=ops.root/"worker_jobs"/ex["id"]/"alpha_result/attempts.json"
                if path.exists(): record["observed_attempts"]=len(json.loads(path.read_bytes()))
            record["finalized"]=True
            write_json(root/"formula_intent_ledger.json", dict(records=ledger))
        assert record["observed_attempts"]==2
        cases[mode]=dict(ops=ops, session=session, proposal=proposal, registry=registry,
            input_folder=folder, experiment=ex, output=ops.root/"worker_jobs"/ex["id"]/"alpha_result")
    yield cases
    assert sum(r["charged_formula_intents"] for r in ledger)==8
    assert all(r["observed_attempts"]==2 and r["status"]=="completed" for r in ledger)


@pytest.mark.parametrize("mode", ["v1", "v2"])
def test_old_protocol_retains_training_rows_zero_holdout_and_hand_values(executed, mode):
    c=executed[mode]; out=c["output"]
    result=json.loads((out/"result.json").read_bytes())
    assert result["holdout_rows_executed"]==0 and "forecast_rows_executed" not in result
    if mode=="v1": values=pd.read_parquet(out/"relative_close_rank/values.parquet")
    else: values=read_compact_candidate(read_shared_index(out,max_bytes=10_000_000),"relative_close_rank",max_bytes=10_000_000).values
    assert values.trade_date.tolist()==["2024-01-03"]*2+["2024-01-04"]*2
    assert values.score.tolist()==[.75]*4
    assert audit_output(c["ops"],c["experiment"])["holdout_rows_executed"]==0


def test_forecast_registration_is_only_planned_then_worker_records_actual_rows(executed):
    c=executed["v3"]; inspection=c["experiment"]["proposal"]["inspection"]
    assert inspection["forecast_rows_planned"]==6
    assert inspection["holdout_rows_executed"]==inspection["forecast_rows_executed"]==0
    result=json.loads((c["output"]/"result.json").read_bytes())
    assert result["schema"]=="registered-alpha-feature-result-v3"
    assert result["holdout_rows_executed"]==result["forecast_rows_executed"]==6
    assert result["training_rows_executed"]==result["model_fits"]==result["accounts"]==0
    assert result["unseen_holdout_claim"] is False
    values=read_compact_candidate(read_shared_index(c["output"],max_bytes=10_000_000),"relative_close_rank",max_bytes=10_000_000).values
    assert values.trade_date.tolist()==["2024-01-08"]*2+["2024-01-09"]*2+["2024-01-10"]*2
    assert values.score.tolist()==[.5,1.]*3
    audit=audit_output(c["ops"],c["experiment"])
    assert audit["holdout_rows_executed"]==audit["rows_per_completed_candidate"]==6


def test_forecast_lag_keeps_missing_warmup_and_causal_later_values(executed):
    c=executed["v3_lag"]
    block=read_compact_candidate(read_shared_index(c["output"],max_bytes=10_000_000),"second_counted",max_bytes=10_000_000)
    assert block.values.score.iloc[:2].isna().all()
    assert block.values.score.iloc[2:].tolist()==[.5,1.]*2
    assert block.missing.score.iloc[:2].ne("present").all()
    assert audit_output(c["ops"],c["experiment"])["rows_per_completed_candidate"]==6


def selection_objects(case):
    folder=case["input_folder"]
    read=lambda n:json.loads((folder/n).read_bytes())
    return [copy.deepcopy(case["proposal"]["spec"]), read("selection.json"),read("summary.json"),read("library.json"),read("source.json")]


def selection_check(objects):
    doc,receipt,summary,library,source=objects
    return validate_forecast_selection(doc,receipt,summary,library,source,input_hashes={
        k:receipt[k] for k in ("summary_sha256","library_sha256","source_receipt_sha256")})


@pytest.mark.parametrize("mutation", ["dataset","universe","train_dates","cutoff_overlap","selected_before_summary",
    "historical_claim","missing_attempt","duplicate_attempt","false_quality","false_complete","fits_bool",
    "account_count","missing_selected","duplicate_selected","unselected_candidate","candidate_change",
    "omit_selected_in_family","label_filtered_pool","wrong_source_cutoff","wrong_source_universe"])
def test_train_selection_semantics_reject_even_if_identity_hashes_are_resealed(executed, mutation):
    objects=selection_objects(executed["v3"]);doc,r,summary,library,source=objects
    if mutation=="dataset":r["dataset_id"]="other"
    if mutation=="universe":r["universe_id"]="other"
    if mutation=="train_dates":r["development_end"]="2024-01-04"
    if mutation=="cutoff_overlap":r["training_label_cutoff"]="2024-01-08T00:00:00+08:00"
    if mutation=="selected_before_summary":r["selected_at"]="2024-01-08T00:00:00+08:00"
    if mutation=="historical_claim":r["historical_selection_certified"]=True
    if mutation=="missing_attempt":summary["all_trials"].pop()
    if mutation=="duplicate_attempt":summary["all_trials"][1]=copy.deepcopy(summary["all_trials"][0])
    if mutation=="false_quality":summary["all_trials"][0]["quality_passed"]=False
    if mutation=="false_complete":summary["all_trials"][0]["proxy_complete"]=False
    if mutation=="fits_bool":summary["new_fits"]=False
    if mutation=="account_count":summary["new_accounts"]=1
    if mutation=="missing_selected":r["selected_candidate_ids"].pop()
    if mutation=="duplicate_selected":r["selected_candidate_ids"]*=2
    if mutation=="unselected_candidate":doc["candidates"].append(copy.deepcopy(library[2]))
    if mutation=="candidate_change":doc["candidates"][0]["hypothesis"]="changed after selection"
    if mutation=="omit_selected_in_family":doc["candidates"].pop()
    if mutation=="label_filtered_pool":source["label_filter_for_membership"]=True
    if mutation=="wrong_source_cutoff":source["fit_cutoff"]="2024-01-06T15:30:00+08:00"
    if mutation=="wrong_source_universe":source["universe_id"]="other"
    with pytest.raises(ValueError):selection_check(objects)


def test_retrospective_selection_can_happen_later_without_claiming_unseen_test(executed):
    info=selection_check(selection_objects(executed["v3"]))
    assert not info["historical_selection_certified"] and not info["future_labels_read"]
    assert not info["numerical_selection_reproduction"] and info["selected_candidates"]==2


@pytest.mark.parametrize("mutation", ["partition_missing","partition_wrong","warmup_after","overlap",
    "roles_missing","roles_alias","roles_reserved","roles_block_collision","v2_forecast_extra","v1_layout_extra"])
def test_explicit_forecast_schema_rejects_ambiguous_or_legacy_mixed_protocol(executed, mutation):
    c=executed["v3"]; spec=copy.deepcopy(c["proposal"]["spec"])
    if mutation=="partition_missing":spec.pop("execution_partition")
    if mutation=="partition_wrong":spec["execution_partition"]="development"
    if mutation=="warmup_after":spec["forecast_warmup_start"]="2024-01-09"
    if mutation=="overlap":spec["ranges"]["holdout_start"]="2024-01-05"
    if mutation=="roles_missing":spec["training_selection"].pop("summary_role")
    if mutation=="roles_alias":spec["training_selection"]["summary_role"]=spec["training_selection"]["receipt_role"]
    if mutation=="roles_reserved":spec["training_selection"]["summary_role"]="membership"
    if mutation=="roles_block_collision":spec["training_selection"]["summary_role"]="daily_values"
    if mutation=="v2_forecast_extra":spec["schema"]="alpha-feature-batch-v2"
    if mutation=="v1_layout_extra":spec["schema"]="alpha-feature-batch-v1"
    with pytest.raises(ValueError):AlphaBatchSpec.from_dict(spec,c["registry"])


@pytest.mark.parametrize("mutation", ["count_zero","count_bool","training_count","partition_wrong","unseen_claim",
    "selection_hash","legacy_schema","metadata_count","metadata_unseen","missing_reason"])
def test_independent_forecast_output_audit_rejects_resealed_false_claims(executed, mutation):
    c=executed["v3"]; out=c["output"]
    originals={p.relative_to(out):p.read_bytes() for p in out.rglob("*") if p.is_file()}
    try:
        p=out/"result.json"; result=json.loads(p.read_bytes())
        if mutation=="count_zero":result["forecast_rows_executed"]=0
        if mutation=="count_bool":result["training_rows_executed"]=False
        if mutation=="training_count":result["training_rows_executed"]=6
        if mutation=="partition_wrong":result["execution_partition"]="development"
        if mutation=="unseen_claim":result["unseen_holdout_claim"]=True
        if mutation=="selection_hash":result["training_selection_sha256"]="0"*64
        if mutation=="legacy_schema":result["schema"]="registered-alpha-feature-result-v2"
        write_json(p,result)
        if mutation.startswith("metadata_"):
            p=out/"relative_close_rank/definition.json"; meta=json.loads(p.read_bytes())
            if mutation=="metadata_count":meta["holdout_rows_executed"]=0
            else:meta["unseen_holdout_claim"]=True
            write_json(p,meta)
        if mutation=="missing_reason":
            p=out/"relative_close_rank/missing.parquet"; frame=pd.read_parquet(p)
            frame.loc[0,"score"]="source_missing";frame.to_parquet(p,index=False)
        manifest=json.loads((out/"manifest.json").read_bytes())
        manifest["artifacts"]={name:digest(out/name) for name in manifest["artifacts"]}
        write_json(out/"manifest.json",manifest)
        with pytest.raises(ValueError):audit_output(c["ops"],c["experiment"])
    finally:
        for name,raw in originals.items():(out/name).write_bytes(raw)


def test_future_target_role_cannot_be_added_to_formula_inputs(executed):
    c=executed["v3"];p=copy.deepcopy(c["proposal"])
    # Use a distinct existing file so the exact-role guard is what rejects it.
    source=c["ops"].project/"data/alpharesearch/test_input/unused_target.json"
    write_json(source,dict(future_return=99999))
    p["inputs"].append(dict(role="future_target",path=source.relative_to(c["ops"].project).as_posix(),sha256=digest(source),bytes=source.stat().st_size))
    with pytest.raises(ValueError,match="exactly cover"):inspect_inputs(c["ops"].project,p)


@pytest.mark.parametrize("mode", ["v1","v2","v3","v3_lag"])
def test_submit_review_and_recovery_do_not_evaluate_formulas_again(executed, mode, monkeypatch):
    from src.alpharesearch.operators import ExpressionEvaluator
    def forbidden(*args,**kwargs):raise AssertionError("Audit cannot evaluate formulas again")
    monkeypatch.setattr(ExpressionEvaluator,"evaluate",forbidden)
    c=executed[mode];ops=c["ops"];session=c["session"]
    assert ops.recover(c["experiment"]["id"])["status"]=="completed"
    ops.submit(session,dict(summary="Synthetic forecast boundary only",findings=["Eight total fixture formula intentions"],
        limitations=["No actual market fit or account; retrospective selection"],next_steps=["Bounded real feature verification later"]))
    ops.review(session["task_id"],"independent-forecast-fixture-role","defer","Synthetic protocol checks completed")
    assert ops.store.get(session["task_id"])["status"]=="deferred"


def test_forecast_warmup_training_compute_is_visible_but_never_output(executed):
    c=executed["v3_lag"];inspection=c["experiment"]["proposal"]["inspection"]
    assert inspection["training_rows_planned"]==4 and inspection["training_rows_executed"]==0
    result=json.loads((c["output"]/"result.json").read_bytes())
    assert result["training_rows_executed"]==4 and result["training_rows_output"]==0
    block=read_compact_candidate(read_shared_index(c["output"],max_bytes=10_000_000),"second_counted",max_bytes=10_000_000)
    assert block.metadata["training_rows_executed"]==4 and block.metadata["training_rows_output"]==0
    assert block.values.trade_date.ge("2024-01-08").all()
    assert audit_output(c["ops"],c["experiment"])["holdout_rows_executed"]==6
