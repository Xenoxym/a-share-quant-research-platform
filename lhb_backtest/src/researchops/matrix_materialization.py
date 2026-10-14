"""Save completed expression outputs for the existing learning-matrix adapter.

Reuse the original declared candidates, exact partitions and missing reasons.
No formula evaluation, outcome selection, label reads or model execution.
"""
import json
import os
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .alpha_experiments import verify_inputs
from .screen_experiments import assembly_reference
from .service import Research
from ..alpharesearch.assembly import FeatureAssembly
from ..alpharesearch.features.base import FeatureBlock
from ..alpharesearch.registry import FeatureRegistry
from ..alpharesearch.expression_assembly import assemble_registered_expression_batch, iter_prepared_expression_assemblies
from ..technical.artifacts import content_id, digest, write_json


def _output(project, name, *, allow_existing=False):
    if not isinstance(name,str) or "\\" in name:
        raise ValueError("Portable project-relative output required")
    relative=Path(name)
    if (relative.is_absolute() or not relative.parts or relative.parts[0] not in {"research","data"}
            or any(x.startswith(".") for x in relative.parts)):
        raise ValueError("Matrix output must stay in project research/data")
    root=Path(project).resolve(); path=root/relative
    for item in (path,*path.parents):
        if item.is_symlink() or (hasattr(item,"is_junction") and item.is_junction()):
            raise ValueError("Linked matrix output refused")
        if item==root:break
    if not path.resolve().is_relative_to(root):raise ValueError("Matrix output escaped project")
    if path.exists() and not allow_existing:raise ValueError("Existing matrix directory refused; retain and reuse original files")
    return path


def _budget(**limits):
    if any(type(v) is not int or v<1 for v in limits.values()):
        raise ValueError("Positive integer matrix budgets required")


def save_learning_matrix(assembly,folder,*,max_output_bytes):
    """Compatibility writer for one validated complete assembly."""
    return save_learning_matrix_parts(iter([assembly]),folder,max_output_bytes=max_output_bytes)


def save_learning_matrix_parts(parts,folder,*,max_output_bytes,expected_rows=None):
    """Write bounded output blocks with Arrow; publish the receipt only at end.

    Input blocks must share the complete declaration. Exact global key sets are
    retained for duplicate detection, but full value/reason frames are not held.
    """
    _budget(max_output_bytes=max_output_bytes)
    if expected_rows is not None:_budget(expected_rows=expected_rows)
    parts=iter(parts)
    try:first=next(parts)
    except StopIteration:raise ValueError("At least one matrix block required") from None
    if not isinstance(first,FeatureAssembly):raise TypeError("Typed assembly required")
    first.block.validate()
    first.registry.resolve_bound_definitions(first.source_bindings,first.block.units)
    if (first.block.metadata.get("registry_version")!=first.registry.version_id
            or first.block.metadata.get("source_bindings")!=list(first.source_bindings)):
        raise ValueError("Matrix registry or source bindings differ")
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    writers={};seen_ids=set();seen_pairs=set();rows=0
    from itertools import chain
    try:
        for part in chain([first],parts):
            if (not isinstance(part,FeatureAssembly) or part.block.units!=first.block.units
                    or part.block.metadata!=first.block.metadata or part.registry.to_dict()!=first.registry.to_dict()
                    or part.source_bindings!=first.source_bindings):raise ValueError("Matrix block declaration changed")
            part.block.validate()
            if part.block.values.empty:raise ValueError("Nonempty matrix block required")
            for sample,day,stock in part.block.values[['sample_id','trade_date','stock_code']].itertuples(index=False,name=None):
                if sample in seen_ids or (day,stock) in seen_pairs:raise ValueError("Duplicate matrix keys across blocks")
                seen_ids.add(sample);seen_pairs.add((day,stock))
            rows+=len(part.block.values)
            if expected_rows is not None and rows>expected_rows:raise ValueError("Matrix row count exceeded")
            for frame,name in ((part.block.values,"assembly_values.parquet"),(part.block.missing,"assembly_missing.parquet")):
                table=pa.Table.from_pandas(frame,preserve_index=False)
                if name not in writers:writers[name]=pq.ParquetWriter(folder/name,table.schema)
                writers[name].write_table(table)
            if sum(p.stat().st_size for p in folder.iterdir())>max_output_bytes:
                raise ValueError("Matrix output byte budget exceeded")
    finally:
        for writer in writers.values():writer.close()
    if expected_rows is not None and rows!=expected_rows:raise ValueError("Matrix row count incomplete")
    write_json(folder/"definition.json",dict(first.block.metadata,units=first.block.units))
    write_json(folder/"registry.json",first.registry.to_dict())
    roles={"assembly_values":"assembly_values.parquet","assembly_missing":"assembly_missing.parquet","assembly_definition":"definition.json"}
    receipt=assembly_reference(first,{r:digest(folder/n) for r,n in roles.items()},source_rows=rows)
    current=sum(p.stat().st_size for p in folder.iterdir())
    receipt_bytes=len(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False).replace("\n",os.linesep).encode("utf-8"))
    if current+receipt_bytes>max_output_bytes:raise ValueError("Matrix output byte budget exceeded")
    write_json(folder/"receipt.json",receipt)
    return dict(rows=rows,features=len(first.block.units),bytes=sum(p.stat().st_size for p in folder.iterdir()),reference_id=receipt["reference_id"])


def _registered_expression_parts(research,development_id,forecast_id,*,max_rows,max_matrix_cells):
    """Join the complete original prepared base and both actual batch partitions."""
    if not isinstance(research,Research):raise TypeError("Actual Research interface required")
    _budget(max_rows=max_rows,max_matrix_cells=max_matrix_cells)
    development=research.store.experiment(development_id); forecast=research.store.experiment(forecast_id)
    if any(x["status"]!="completed" or x["proposal"].get("kind")!="alpha_batch" for x in (development,forecast)):
        raise ValueError("Completed development and forecast batches required")
    proposal=development["proposal"];spec=proposal["spec"];future=forecast["proposal"]["spec"]
    if (spec["schema"]!="alpha-feature-batch-v4" or future["schema"]!="alpha-feature-batch-v6"
            or len(spec["blocks"])!=1 or future["blocks"]!=spec["blocks"] or future["candidates"]!=spec["candidates"]):
        raise ValueError("Exact prepared development/forecast candidates and single base required")
    roles=verify_inputs(research.project,proposal["inputs"],spec["budget"]["max_input_bytes"])
    record=next(iter(spec["blocks"].values()))
    base_roles={record[k] for k in ("values","missing","definition","receipt","source_receipt")}|{"registry","membership"}
    dev_refs={r["role"]:(r["sha256"],r["bytes"]) for r in proposal["inputs"]}
    future_refs={r["role"]:(r["sha256"],r["bytes"]) for r in forecast["proposal"]["inputs"]}
    if any(dev_refs[r]!=future_refs.get(r) for r in base_roles):raise ValueError("Original prepared input identities differ")
    rows=pq.ParquetFile(roles["membership"]).metadata.num_rows
    definition=json.loads(roles[record["definition"]].read_text(encoding="utf-8"))
    if rows>max_rows or rows*(len(definition["units"])+len(spec["candidates"]))>max_matrix_cells:
        raise ValueError("Matrix row/cell budget exceeded before loading")
    registry=FeatureRegistry.from_dict(json.loads(roles["registry"].read_text(encoding="utf-8")))
    meta=dict(definition);units=meta.pop("units")
    membership=pd.read_parquet(roles["membership"])
    source=json.loads(roles[record["source_receipt"]].read_text(encoding="utf-8"))
    prepared=json.loads(roles[record["receipt"]].read_text(encoding="utf-8"))
    hashes={k:digest(roles[record[k]]) for k in ("values","missing","definition","source_receipt")}
    hashes.update(registry=digest(roles["registry"]),membership=digest(roles["membership"]))
    dev=assemble_registered_expression_batch(research,development,membership,max_rows=max_rows,max_matrix_cells=max_matrix_cells)
    pred=assemble_registered_expression_batch(research,forecast,membership,max_rows=max_rows,max_matrix_cells=max_matrix_cells)
    base=FeatureAssembly(FeatureBlock(pd.read_parquet(roles[record["values"]]),pd.read_parquet(roles[record["missing"]]),units,meta),registry,tuple(meta["source_bindings"]))
    merged=iter_prepared_expression_assemblies(base,dev,pred,membership,source,prepared,artifact_hashes=hashes,
        expected_expression_keys=["expr."+c["candidate_id"] for c in spec["candidates"]],max_rows=max_rows,max_matrix_cells=max_matrix_cells)
    def verified_parts():
        yield from merged
        # Run after consumption, before the writer publishes its final receipt.
        verify_inputs(research.project,proposal["inputs"],spec["budget"]["max_input_bytes"])
        verify_inputs(research.project,forecast["proposal"]["inputs"],future["budget"]["max_input_bytes"])
        if research.store.experiment(development_id)!=development or research.store.experiment(forecast_id)!=forecast:
            raise ValueError("Registered batch changed during matrix assembly")
    return verified_parts(),rows


def matrix_from_registered_expressions(research,development_id,forecast_id,matrix_folder,*,max_rows,max_matrix_cells,max_output_bytes):
    """Write once from original, verified completed producers; no formula rerun."""
    _budget(max_output_bytes=max_output_bytes)
    output=_output(research.project,matrix_folder)
    parts,rows=_registered_expression_parts(research,development_id,forecast_id,max_rows=max_rows,max_matrix_cells=max_matrix_cells)
    result=save_learning_matrix_parts(parts,output,max_output_bytes=max_output_bytes,expected_rows=rows)
    return dict(result,matrix_folder=matrix_folder,development=development_id,forecast=forecast_id,
        formula_evaluations=0,model_fits=0,predict_calls=0,accounts=0)


def verify_expression_matrix_content(research,development_id,forecast_id,matrix_folder,*,max_rows,max_matrix_cells):
    """Compare actual output blocks to original producers, even after re-sealing.

    This is for the pre-registration interruption window. Once registered, the
    ordinary frozen input hashes bind the matrix; no new cache/approval layer.
    """
    folder=_output(research.project,matrix_folder,allow_existing=True)
    parts,rows=_registered_expression_parts(research,development_id,forecast_id,max_rows=max_rows,max_matrix_cells=max_matrix_cells)
    streams={name:iter(pq.ParquetFile(folder/file).iter_batches(batch_size=8192))
        for name,file in [('values','assembly_values.parquet'),('missing','assembly_missing.parquet')]}
    definition_doc=json.loads((folder/'definition.json').read_text(encoding='utf-8-sig'))
    registry_doc=json.loads((folder/'registry.json').read_text(encoding='utf-8-sig'))
    count=0
    for part in parts:
        definition=dict(part.block.metadata,units=part.block.units)
        if (definition_doc!=definition
                or content_id(registry_doc)!=content_id(part.registry.to_dict())):
            raise ValueError('Matrix declaration differs from original expression producers')
        for name,stream in streams.items():
            actual=next(stream,None)
            if actual is None:raise ValueError('Matrix content ended before original producer rows')
            try:
                pd.testing.assert_frame_equal(actual.to_pandas().reset_index(drop=True),
                    getattr(part.block,name).reset_index(drop=True),check_dtype=False,check_exact=True)
            except AssertionError as exc:
                raise ValueError('Matrix content differs from original expression producers: '+name) from exc
        count+=len(part.block.values)
    if count!=rows or any(next(stream,None) is not None for stream in streams.values()):
        raise ValueError('Matrix contains rows beyond original producers')
