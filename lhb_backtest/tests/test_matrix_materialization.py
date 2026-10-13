"""Registered formula outputs to reusable matrix: no labels or extra formula work."""
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import pytest
from tests.test_all_candidate_forecast_integration import setup
from tests.test_prepared_assembly import case
from src.alpharesearch.assembly import FeatureAssembly
from src.alpharesearch.forecast_handoff import prepare_registered_forecast
from src.alpharesearch.operators import ExpressionEvaluator
from src.researchops.matrix_materialization import matrix_from_registered_expressions,save_learning_matrix,_output
from src.researchops.learning_proposal import learning_from_matrix
from tests.test_alpha_account_registration import learning_proposal_for_account


def assembly():
    block,registry,*_=case()
    return FeatureAssembly(block,registry,tuple(block.metadata["source_bindings"]))


def test_registered_partitions_save_all_candidates_and_learning_handoff(tmp_path,monkeypatch):
    ops,session,proposal,p,block,registry,*_=setup(tmp_path)
    dev=ops.execute(session,ops.register(session,proposal)["id"])
    assert dev["status"]=="completed",dev.get("error")
    future_proposal=prepare_registered_forecast(ops,dev["id"],name="matrix_forecast",timeout_seconds=60)
    future=ops.execute(session,ops.register(session,future_proposal)["id"])
    assert future["status"]=="completed",future.get("error")
    def no_formula(*args,**kwargs):raise AssertionError("Matrix handoff recomputed formula")
    monkeypatch.setattr(ExpressionEvaluator,"evaluate",no_formula)
    out=matrix_from_registered_expressions(ops,dev["id"],future["id"],"research/matrix",max_rows=4,max_matrix_cells=20,max_output_bytes=1000000)
    assert out["rows"]==4 and out["features"]==5
    assert out["formula_evaluations"]==out["model_fits"]==out["accounts"]==0
    folder=ops.project/"research/matrix"
    assert {x.name for x in folder.iterdir()}=={"assembly_values.parquet","assembly_missing.parquet","definition.json","registry.json","receipt.json"}
    values=pd.read_parquet(folder/"assembly_values.parquet");missing=pd.read_parquet(folder/"assembly_missing.parquet")
    pd.testing.assert_frame_equal(values[list(block.values)],block.values)
    pd.testing.assert_frame_equal(missing[list(block.missing)],block.missing)
    assert values["expr.tiny_candidate"].tolist()==[1.,1.,3.,4.]
    assert missing["expr.second_candidate"].iloc[3]=="source_uncovered"
    original_counts=ops.brief(session["task_id"])["budget"]["registered_experiments"]
    # Existing learning adapter binds the newly written matrix. No label-based
    # membership filtering, ranking screen, fit, prediction or new registration.
    fixture_folder=ops.project/"research/template";fixture_folder.mkdir()
    template=learning_proposal_for_account(ops,fixture_folder)
    template["spec"]["learning"]["features"]=["stock.p","expr.second_candidate"]
    derived=learning_from_matrix(ops,"research/matrix",template)
    assert derived["spec"]==template["spec"]
    assert ops.brief(session["task_id"])["budget"]["registered_experiments"]==original_counts
    for role in ("decisions","labels"):
        assert next(r for r in derived["inputs"] if r["role"]==role)==next(r for r in template["inputs"] if r["role"]==role)
    with pytest.raises(ValueError,match="Existing matrix"):
        matrix_from_registered_expressions(ops,dev["id"],future["id"],"research/matrix",max_rows=4,max_matrix_cells=20,max_output_bytes=1000000)
    # Cap rejection precedes reading either full input frame.
    with monkeypatch.context() as m:
        m.setattr(pd,"read_parquet",lambda *a,**k: (_ for _ in ()).throw(AssertionError("Unbounded read")))
        with pytest.raises(ValueError,match="before loading"):
            matrix_from_registered_expressions(ops,dev["id"],future["id"],"research/too-many",max_rows=3,max_matrix_cells=20,max_output_bytes=1000000)
    # CLI reads the actual original DB; no new task/session or registration.
    command=[sys.executable,str(Path(__file__).parents[1]/"tools/execution.py"),"--project",str(ops.project),"expression-matrix","--development",dev["id"],"--forecast",future["id"],"--output","research/cli-matrix","--max-rows","4","--max-cells","20","--max-bytes","1000000"]
    result=subprocess.run(command,text=True,capture_output=True)
    assert result.returncode==0,result.stderr+result.stdout
    assert json.loads(result.stdout)["data"]["reference_id"]==out["reference_id"]


def test_writer_budget_includes_last_receipt_and_preserves_failure(tmp_path):
    original=assembly();first=save_learning_matrix(original,tmp_path/"first",max_output_bytes=1000000)
    assert first["bytes"]==sum(p.stat().st_size for p in (tmp_path/"first").iterdir())
    with pytest.raises(ValueError,match="byte budget"):
        save_learning_matrix(original,tmp_path/"partial",max_output_bytes=first["bytes"]-1)
    assert (tmp_path/"partial").exists() and not (tmp_path/"partial/receipt.json").exists()
    before={p.name:p.read_bytes() for p in (tmp_path/"first").iterdir()}
    with pytest.raises(FileExistsError):save_learning_matrix(original,tmp_path/"first",max_output_bytes=1000000)
    assert before=={p.name:p.read_bytes() for p in (tmp_path/"first").iterdir()}


def test_interrupted_writer_has_no_receipt(tmp_path,monkeypatch):
    original=pd.DataFrame.to_parquet;calls=[]
    def interrupted(frame,*args,**kwargs):
        calls.append(1)
        if len(calls)==2:raise OSError("interrupted second output")
        return original(frame,*args,**kwargs)
    monkeypatch.setattr(pd.DataFrame,"to_parquet",interrupted)
    with pytest.raises(OSError,match="interrupted"):
        save_learning_matrix(assembly(),tmp_path/"partial",max_output_bytes=1000000)
    assert (tmp_path/"partial/assembly_values.parquet").exists()
    assert not (tmp_path/"partial/receipt.json").exists()


@pytest.mark.parametrize("name",["../escape","research/../escape",".local/foo","C:/escape","research/hidden/.x","data\\bad"])
def test_output_escape_rejected(tmp_path,name):
    with pytest.raises(ValueError):_output(tmp_path,name)


@pytest.mark.parametrize("cap",[0,-1,True,1.5])
def test_invalid_writer_budget_before_creation(tmp_path,cap):
    with pytest.raises(ValueError,match="Positive integer"):
        save_learning_matrix(assembly(),tmp_path/"not-created",max_output_bytes=cap)
    assert not (tmp_path/"not-created").exists()
