"""Small registered train-to-forecast path, not a market experiment."""
from copy import deepcopy
import shutil
from pathlib import Path

import pandas as pd
import pytest

from tests.test_prepared_assembly import case, document, reference
from src.alpharesearch.assembly import FeatureAssembly
from src.alpharesearch.dsl import Expr, compile_expression
from src.alpharesearch.expression_assembly import (
    assemble_registered_expression_batch, merge_prepared_expression_assemblies, iter_prepared_expression_assemblies,
)
from src.alpharesearch.forecast_handoff import prepare_registered_forecast
from src.alpharesearch.prepared_assembly import prepared_assembly_receipt
from src.researchops.service import Research
from src.technical.artifacts import digest, write_json


def setup(tmp_path):
    project = tmp_path / "project"
    shutil.copytree(Path(__file__).parents[1] / "src", project / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    block, registry, membership, _, _, _ = case()
    block.values.loc[3, "stock.q"] = float("nan")
    block.missing.loc[3, "stock.q"] = "source_uncovered"
    folder = project / "research" / "tiny"
    folder.mkdir(parents=True)
    paths = {r: folder / (r + ".json") for r in
             ("definition", "registry", "source_receipt", "receipt", "calendar")}
    paths.update({r: folder / (r + ".parquet") for r in
                  ("values", "missing", "membership", "decision_clocks")})
    block.values.to_parquet(paths["values"], index=False)
    block.missing.to_parquet(paths["missing"], index=False)
    membership.to_parquet(paths["membership"], index=False)
    write_json(paths["registry"], registry.to_dict())
    write_json(paths["definition"], dict(block.metadata, units=block.units))
    days = pd.bdate_range("2024-01-02", "2024-01-12").strftime("%Y-%m-%d").tolist()
    write_json(paths["calendar"], dict(trade_dates=days))
    pd.DataFrame(dict(trade_date=days, decision_at=[d+"T15:30:00+08:00" for d in days])).to_parquet(paths["decision_clocks"], index=False)
    hashes = {k: digest(paths[k]) for k in ("values", "missing", "definition", "registry", "membership")}
    source = reference(block, registry, hashes)
    write_json(paths["source_receipt"], source)
    hashes["source_receipt"] = digest(paths["source_receipt"])
    receipt = prepared_assembly_receipt(block, registry, membership, source, artifact_hashes=hashes)
    write_json(paths["receipt"], receipt)
    doc = document("alpha-feature-batch-v4", registry, receipt["adapter_id"])
    doc["ranges"].update(development_end="2024-01-05", holdout_start="2024-01-12", holdout_end="2024-01-12")
    compiled = compile_expression(Expr.call("cs_rank", Expr.ref("stock.q")), registry)
    doc["candidates"].append(dict(candidate_id="second_candidate", hypothesis="Missing future q stays missing", family="tiny_family", expression=compiled.to_dict()))
    doc["bindings"]["stock.q"] = dict(doc["bindings"]["stock.p"], column="stock.q", definition_id=registry.resolve("stock.q").definition_id)
    inputs = [dict(role=r, path=p.relative_to(project).as_posix(), sha256=digest(p), bytes=p.stat().st_size) for r,p in paths.items()]
    proposal = dict(kind="alpha_batch", spec=doc, inputs=inputs, timeout_seconds=60,
                    hypothesis="Two fixed synthetic formulas", expected_observation="Exact small partition", falsification="Key or clock changes")
    ops = Research(project)
    task = ops.create(dict(title="tiny full candidate forecast", question="two fixed formulas", rationale="hand check only",
                          success_criteria=["Preserve four original keys"], stop_criteria=["Two tiny jobs only"], max_experiments=2))
    session = ops.store.claim(task["id"], "test-forecast")
    return ops, session, proposal, membership, block, registry, source, receipt, hashes


def test_registered_all_candidate_forecast_and_exact_merge(tmp_path):
    ops, session, proposal, p, block, registry, source, receipt, hashes = setup(tmp_path)
    training = ops.register(session, proposal)
    done = ops.execute(session, training["id"])
    assert done["status"] == "completed", done.get("error")
    forecast_proposal = prepare_registered_forecast(ops, done["id"], name="tiny_forecast", timeout_seconds=60)
    assert forecast_proposal["spec"]["candidates"] == done["proposal"]["spec"]["candidates"]
    # DB identity must be checked, independently of the syntactically valid fingerprint.
    changed = deepcopy(forecast_proposal)
    changed["spec"]["forecast_source"]["producer_fingerprint"] = "f"*64
    with pytest.raises(ValueError, match="DB identity"):
        ops.register(session, changed)
    for filename in ("prepared_assembly.py", "registry.py"):
        implementation = ops.project / "src" / "alpharesearch" / filename
        original = implementation.read_bytes()
        try:
            implementation.write_bytes(original + b"\n# changed after training producer\n")
            with pytest.raises(ValueError, match="operator implementation"):
                ops.register(session, forecast_proposal)
        finally:
            implementation.write_bytes(original)
    forecast = ops.register(session, forecast_proposal)
    assert forecast["proposal"]["inspection"]["training_rows_planned"] == 0
    assert forecast["proposal"]["inspection"]["forecast_rows_planned"] == 2
    finished = ops.execute(session, forecast["id"])
    assert finished["status"] == "completed", finished.get("error")
    assert finished["result"]["holdout_rows_executed"] == 2
    dev = assemble_registered_expression_batch(ops, done, p, max_rows=4, max_matrix_cells=100)
    future = assemble_registered_expression_batch(ops, finished, p, max_rows=4, max_matrix_cells=100)
    assert dev.registry.to_dict() == future.registry.to_dict()
    assert future.block.values["expr.tiny_candidate"].tolist() == [3.,4.]
    assert future.block.missing["expr.second_candidate"].iloc[1] != "present"
    base = FeatureAssembly(block, registry, tuple(block.metadata["source_bindings"]))
    keys = ["expr."+c["candidate_id"] for c in proposal["spec"]["candidates"]]
    merged = merge_prepared_expression_assemblies(base, dev, future, p, source, receipt,
        artifact_hashes=hashes, expected_expression_keys=keys, max_rows=4, max_matrix_cells=20)
    assert len(merged.block.units) == 5
    pd.testing.assert_frame_equal(merged.block.values[list(block.values)], block.values)
    pd.testing.assert_frame_equal(merged.block.missing[list(block.missing)], block.missing)
    assert merged.block.values["expr.tiny_candidate"].tolist() == [1.,1.,3.,4.]
    for batch_rows in (1,3):
        parts=list(iter_prepared_expression_assemblies(base,dev,future,p,source,receipt,
            artifact_hashes=hashes,expected_expression_keys=keys,max_rows=4,max_matrix_cells=20,batch_rows=batch_rows))
        assert max(len(x.block.values) for x in parts)<=batch_rows
        assert [v for x in parts for v in x.block.values['expr.tiny_candidate']]==[1.,1.,3.,4.]
        assert parts[-1].block.missing['expr.second_candidate'].iloc[-1]=='source_uncovered'
        pd.testing.assert_frame_equal(pd.concat([x.block.values for x in parts],ignore_index=True),merged.block.values)
        pd.testing.assert_frame_equal(pd.concat([x.block.missing for x in parts],ignore_index=True),merged.block.missing)
        assert all(x.block.metadata==merged.block.metadata and x.registry.to_dict()==merged.registry.to_dict() for x in parts)
    with pytest.raises(ValueError, match="All declared"):
        merge_prepared_expression_assemblies(base, dev, future, p, source, receipt,
            artifact_hashes=hashes, expected_expression_keys=keys[:1], max_rows=4, max_matrix_cells=20)
    with pytest.raises(ValueError, match="cell budget"):
        merge_prepared_expression_assemblies(base, dev, future, p, source, receipt,
            artifact_hashes=hashes, expected_expression_keys=keys, max_rows=4, max_matrix_cells=19)
    overlapping = deepcopy(future)
    overlapping.block.values["sample_id"] = dev.block.values.sample_id
    overlapping.block.missing["sample_id"] = dev.block.values.sample_id
    with pytest.raises(ValueError, match="overlap"):
        merge_prepared_expression_assemblies(base, dev, overlapping, p, source, receipt,
            artifact_hashes=hashes, expected_expression_keys=keys, max_rows=4, max_matrix_cells=20)

    other_producer = deepcopy(future)
    other_producer.block.metadata["source_provenance"]["forecast_producer"]["producer_experiment_id"] = "exp-another-source"
    with pytest.raises(ValueError, match="exact development producer"):
        merge_prepared_expression_assemblies(base, dev, other_producer, p, source, receipt,
            artifact_hashes=hashes, expected_expression_keys=keys, max_rows=4, max_matrix_cells=20)
    other_deps, other_future_deps = deepcopy(dev), deepcopy(future)
    for assembly in (other_deps, other_future_deps):
        assembly.source_bindings[0]["original_dependencies"][0]["materialization_id"] = "f"*64
    with pytest.raises(ValueError, match="original prepared base"):
        merge_prepared_expression_assemblies(base, other_deps, other_future_deps, p, source, receipt,
            artifact_hashes=hashes, expected_expression_keys=keys, max_rows=4, max_matrix_cells=20)
    future_clock = deepcopy(future)
    future_clock.block.values["known_at"] = pd.to_datetime(future_clock.block.values.known_at) + pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="clock differs"):
        merge_prepared_expression_assemblies(base, dev, future_clock, p, source, receipt,
            artifact_hashes=hashes, expected_expression_keys=keys, max_rows=4, max_matrix_cells=20)
