"""Save completed expression outputs for the existing learning-matrix adapter.

Reuse the original declared candidates, exact partitions and missing reasons.
No formula evaluation, outcome selection, label reads or model execution.
"""
import json
import os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from .alpha_experiments import verify_inputs
from .screen_experiments import assembly_reference
from .service import Research
from ..alpharesearch.assembly import FeatureAssembly
from ..alpharesearch.features.base import FeatureBlock
from ..alpharesearch.registry import FeatureRegistry
from ..alpharesearch.expression_assembly import assemble_registered_expression_batch, merge_prepared_expression_assemblies
from ..technical.artifacts import digest, write_json


def _output(project, name):
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
    if path.exists():raise ValueError("Existing matrix directory refused; retain and reuse original files")
    return path


def _budget(**limits):
    if any(type(v) is not int or v<1 for v in limits.values()):
        raise ValueError("Positive integer matrix budgets required")


def save_learning_matrix(assembly, folder, *, max_output_bytes):
    """Write the ordinary five-file matrix; receipt is written last.

    Caller owns the authorized path. Failed partial directories remain, with
    no valid receipt; never overwrite an existing matrix or change its scope.
    """
    _budget(max_output_bytes=max_output_bytes)
    if not isinstance(assembly,FeatureAssembly):raise TypeError("Typed assembly required")
    assembly.block.validate()
    assembly.registry.resolve_bound_definitions(assembly.source_bindings,assembly.block.units)
    if assembly.block.metadata.get("registry_version")!=assembly.registry.version_id:
        raise ValueError("Matrix registry differs")
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    for frame,name in ((assembly.block.values,"assembly_values.parquet"),(assembly.block.missing,"assembly_missing.parquet")):
        frame.to_parquet(folder/name,index=False)
        if sum(p.stat().st_size for p in folder.iterdir())>max_output_bytes:
            raise ValueError("Matrix output byte budget exceeded")
    write_json(folder/"definition.json",dict(assembly.block.metadata,units=assembly.block.units))
    write_json(folder/"registry.json",assembly.registry.to_dict())
    roles={"assembly_values":"assembly_values.parquet","assembly_missing":"assembly_missing.parquet","assembly_definition":"definition.json"}
    receipt=assembly_reference(assembly,{r:digest(folder/n) for r,n in roles.items()})
    current=sum(p.stat().st_size for p in folder.iterdir())
    receipt_bytes=len(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False).replace("\n",os.linesep).encode("utf-8"))
    if current+receipt_bytes>max_output_bytes:raise ValueError("Matrix output byte budget exceeded")
    write_json(folder/"receipt.json",receipt)
    return dict(rows=len(assembly.block.values),features=len(assembly.block.units),
        bytes=sum(p.stat().st_size for p in folder.iterdir()),reference_id=receipt["reference_id"])


def matrix_from_registered_expressions(research,development_id,forecast_id,matrix_folder,*,max_rows,max_matrix_cells,max_output_bytes):
    """Join the complete original prepared base and both actual batch partitions."""
    if not isinstance(research,Research):raise TypeError("Actual Research interface required")
    _budget(max_rows=max_rows,max_matrix_cells=max_matrix_cells,max_output_bytes=max_output_bytes)
    output=_output(research.project,matrix_folder)
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
    base=FeatureAssembly(FeatureBlock(pd.read_parquet(roles[record["values"]]),pd.read_parquet(roles[record["missing"]]),units,meta),registry,tuple(meta["source_bindings"]))
    membership=pd.read_parquet(roles["membership"])
    source=json.loads(roles[record["source_receipt"]].read_text(encoding="utf-8"))
    prepared=json.loads(roles[record["receipt"]].read_text(encoding="utf-8"))
    hashes={k:digest(roles[record[k]]) for k in ("values","missing","definition","source_receipt")}
    hashes.update(registry=digest(roles["registry"]),membership=digest(roles["membership"]))
    dev=assemble_registered_expression_batch(research,development,membership,max_rows=max_rows,max_matrix_cells=max_matrix_cells)
    pred=assemble_registered_expression_batch(research,forecast,membership,max_rows=max_rows,max_matrix_cells=max_matrix_cells)
    merged=merge_prepared_expression_assemblies(base,dev,pred,membership,source,prepared,artifact_hashes=hashes,
        expected_expression_keys=["expr."+c["candidate_id"] for c in spec["candidates"]],max_rows=max_rows,max_matrix_cells=max_matrix_cells)
    verify_inputs(research.project,proposal["inputs"],spec["budget"]["max_input_bytes"])
    verify_inputs(research.project,forecast["proposal"]["inputs"],future["budget"]["max_input_bytes"])
    if research.store.experiment(development_id)!=development or research.store.experiment(forecast_id)!=forecast:
        raise ValueError("Registered batch changed during matrix assembly")
    result=save_learning_matrix(merged,output,max_output_bytes=max_output_bytes)
    return dict(result,matrix_folder=matrix_folder,development=development_id,forecast=forecast_id,
        formula_evaluations=0,model_fits=0,predict_calls=0,accounts=0)
