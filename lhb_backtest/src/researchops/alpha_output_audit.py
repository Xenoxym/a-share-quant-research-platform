"""Independent output identities/keys/clock/reason audit; no formula reevaluation."""
import json
import hashlib
import math
from pathlib import Path

import pandas as pd

from .alpha_experiments import verify_inputs, verify_registered
from .resources import WorkerResources
from ..alpharesearch.features.base import FeatureBlock
from ..alpharesearch.feature_storage import read_shared_index, read_compact_candidate, INDEX_FILES, CANDIDATE_FILES
from ..technical.artifacts import content_id, digest, verify_artifacts


def _expected_definitions(proposal,roles,job):
    """Derive provenance/clock identities without evaluating a candidate formula."""
    from ..alpharesearch.registry import FeatureRegistry
    from ..alpharesearch.dsl import verify_compiled
    from ..alpharesearch.contracts import timestamps
    registry=FeatureRegistry.from_dict(json.loads(roles["registry"].read_text(encoding="utf-8")))
    spec=proposal["spec"];r=spec["ranges"];keys=["sample_id","trade_date","stock_code"]
    days=[d for d in json.loads(roles["calendar"].read_text(encoding="utf-8"))["trade_dates"]
          if r["warmup_start"]<=d<=r["development_end"]]
    members=pd.read_parquet(roles["membership"])
    members=members.loc[members.trade_date.between(r["warmup_start"],r["development_end"])].reset_index(drop=True)
    clocks=pd.read_parquet(roles["decision_clocks"])
    clocks=clocks.loc[clocks.trade_date.isin(days)].copy();clocks["decision_at"]=timestamps(clocks.decision_at)
    clocks=clocks.set_index("trade_date").reindex(days).decision_at
    context_id=content_id(dict(schema="expression-context-v1",calendar=days,
        membership=members[keys].to_dict("records"),
        decision_clocks=[(d,t.isoformat()) for d,t in clocks.items()],
        universe_id=spec["universe_id"],allow_weak_vintage=spec["allow_weak_vintage"]))
    paths=("src/alpharesearch/operators.py","src/alpharesearch/dsl.py",
           "src/alpharesearch/contracts.py","src/alpharesearch/features/base.py")
    implementation=sorted((name,hashlib.sha256((job/"code"/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest()) for name in paths)
    metas={name:json.loads(roles[record["definition"]].read_text(encoding="utf-8")) for name,record in spec["blocks"].items()}
    expected={}
    stocks=members.stock_code.nunique();cells=len(days)*stocks
    for candidate in spec["candidates"]:
        compiled=verify_compiled(candidate["expression"],registry);deps=[]
        for did in compiled.dependencies:
            definition=next(d for d in registry.definitions if d.definition_id==did)
            binding=spec["bindings"][definition.key];meta=metas[binding["block"]]
            deps.append(dict(key=definition.key,definition_id=did,materialization_id=binding["materialization_id"],
                             source_id=meta["source_id"],version_id=meta["version_id"],vintage=meta["vintage"]))
        identity=dict(expression_id=compiled.expression_id,context_id=context_id,dependencies=deps,implementation=implementation)
        weak=any(d["vintage"] in ("latest_snapshot_only","unknown") for d in deps)
        expected[candidate["candidate_id"]]=dict(schema="alpha-expression-values-v1",key_columns=keys,
            source_id="typed_alpha_expression",version_id=content_id(identity),
            vintage="latest_snapshot_only" if weak else "market_observation",
            expression_id=compiled.expression_id,context_id=context_id,dependencies=deps,
            implementation_hashes=[list(p) for p in implementation],universe_id=spec["universe_id"],
            calendar_sessions=len(days),grid_stocks=int(stocks),grid_cells=int(cells),
            estimated_buffer_bytes=int(cells*9*(compiled.nodes+len(compiled.dependencies)+4)),
            registered_numerical_execution=True,historical_execution_certified=False,
            independent_reproduction=False,holdout_rows_executed=0)
    return expected


def audit_output(research,experiment):
    verify_registered(research,experiment)
    compact=experiment["proposal"]["spec"].get("storage_layout")=="shared_index_v1"
    proposal=experiment["proposal"];job=research.root/"worker_jobs"/experiment["id"];folder=job/"alpha_result"
    cfg=json.loads((job/"input.json").read_text(encoding="utf-8"))
    if folder.is_symlink() or (hasattr(folder,"is_junction") and folder.is_junction()):
        raise ValueError("Linked result folder refused")
    paths=list(folder.rglob("*"))
    if any(p.is_symlink() or (hasattr(p,"is_junction") and p.is_junction()) for p in paths):
        raise ValueError("Linked output artifact paths refused")
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    expected=dict(schema="registered-alpha-feature-manifest-v2" if compact else "registered-alpha-feature-manifest-v1",experiment_id=experiment["id"],
        task_id=experiment["task_id"],input_hash=digest(job/"input.json"),
        engine_hash=proposal["engine_hash"],candidate_manifest_hash=digest(job/"candidate_manifest.json"))
    if any(manifest.get(k)!=v for k,v in expected.items()) or set(manifest)!=set(expected)|{"artifacts"}:
        raise ValueError("Numerical output manifest differs from registration")
    actual={p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p!=folder/"manifest.json"}
    if actual!=set(manifest["artifacts"]):
        raise ValueError("Unregistered/missing numerical output files")
    for name in actual:
        path=folder/name
        if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(folder):
            raise ValueError("Numerical output path escaped result folder")
    verify_artifacts(folder,manifest)
    resources=WorkerResources.from_dict(proposal["resources"])
    if sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())>resources.max_output_bytes:
        raise ValueError("Final numerical output exceeds byte budget")
    result=json.loads((folder/"result.json").read_text(encoding="utf-8"))
    for key in ("task_id","experiment_id","spec","engine_hash","environment","resources","registry_version"):
        expected_value=experiment["task_id"] if key=="task_id" else experiment["id"] if key=="experiment_id" else proposal[key]
        if result.get(key)!=expected_value:
            raise ValueError("Numerical result definition/ownership differs: "+key)
    if (result.get("schema")!=("registered-alpha-feature-result-v2" if compact else "registered-alpha-feature-result-v1") or result.get("model_fits")!=0
            or result.get("accounts")!=0 or result.get("holdout_rows_executed")!=0
            or result.get("selection_rule")!="all_candidates_no_selection"
            or result.get("evaluation_scope")!="retrospective_time_split"):
        raise ValueError("Feature-only execution must not claim fits/accounts/holdout selection")
    if compact and result.get("storage_layout")!="shared_index_v1":raise ValueError("Compact result layout differs")
    threads=resources.library_threads
    expected_runtime={"environment_threads":{name:str(threads) for name in (
        "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS","ARROW_NUM_THREADS")},
        "arrow_cpu_threads":threads,"arrow_io_threads":1}
    if result.get("runtime_threads")!=expected_runtime:raise ValueError("Worker thread controls differ")
    for key in ("candidate_count","completed_candidates","failed_candidates","model_fits","accounts","holdout_rows_executed"):
        if type(result.get(key)) is not int or result[key]<0:raise ValueError("Integer result counters required")
    if type(result.get("elapsed_seconds")) not in (int,float) or not math.isfinite(result["elapsed_seconds"]) or result["elapsed_seconds"]<0:
        raise ValueError("Finite observed numerical time required")
    attempts=json.loads((folder/"attempts.json").read_text(encoding="utf-8"))
    candidates=proposal["spec"]["candidates"]
    if len(attempts)!=len(candidates) or result["candidate_count"]!=len(candidates):
        raise ValueError("Numerical candidate count changed")
    roles=verify_inputs(cfg["project"],proposal["inputs"],proposal["spec"]["budget"]["max_input_bytes"])
    expected_definitions=_expected_definitions(proposal,roles,job)
    membership=pd.read_parquet(roles["membership"])
    r=proposal["spec"]["ranges"]
    membership=membership.loc[membership.trade_date.between(r["development_start"],r["development_end"])].copy()
    keys=["sample_id","trade_date","stock_code"]
    sorted_keys=membership[keys].sort_values(keys).reset_index(drop=True)
    # No hidden failed candidate may be promoted as a successful complete batch.
    if any(a.get("status")!="completed" for a in attempts):
        raise ValueError("Successful feature batch requires every registered candidate completed")
    completed=0
    allowed={"result.json","attempts.json"}
    index=read_shared_index(folder,max_bytes=resources.max_output_bytes) if compact else None
    if compact:
        allowed|=INDEX_FILES
        try:
            pd.testing.assert_frame_equal(index.keys,membership[keys].reset_index(drop=True),check_dtype=False)
        except AssertionError as exc:
            raise ValueError("Shared index row order differs from frozen membership") from exc
    for attempt,candidate in zip(attempts,candidates):
        cid=candidate["candidate_id"]
        if (attempt["candidate_id"]!=cid or attempt["expression_id"]!=candidate["expression"]["expression_id"]
                or attempt["family"]!=candidate["family"] or attempt["status"] not in {"completed","failed"}):
            raise ValueError("Numerical attempt identity/order/terminal status differs")
        allowed|={cid+"/"+name for name in (CANDIDATE_FILES if compact else ("values.parquet","missing.parquet","definition.json"))}
        if attempt["status"]=="failed":
            allowed|={cid+"/"+name+".partial" for name in ("values.parquet","missing.parquet")}
            if not isinstance(attempt.get("error"),str) or not attempt["error"]:
                raise ValueError("Failed candidate requires retained cause")
            continue
        completed+=1
        if attempt.get("output_folder")!=cid:
            raise ValueError("Completed candidate output folder differs")
        metadata=json.loads((folder/cid/"definition.json").read_text(encoding="utf-8"));units=metadata.pop("units")
        block=(read_compact_candidate(index,cid,max_bytes=resources.max_output_bytes) if compact else
               FeatureBlock(pd.read_parquet(folder/cid/"values.parquet"),
                            pd.read_parquet(folder/cid/"missing.parquet"),units,metadata).validate())
        try:
            pd.testing.assert_frame_equal(block.values[keys].sort_values(keys).reset_index(drop=True),
                                          sorted_keys,check_dtype=False)
        except AssertionError as exc:
            raise ValueError("Numerical output membership differs from full development pool") from exc
        expected=expected_definitions[cid]
        if any(type(metadata.get(k)) is not type(v) or metadata.get(k)!=v for k,v in expected.items()):
            raise ValueError("Numerical feature provenance/context/source/code identity differs")
        if (metadata["expression_id"]!=attempt["expression_id"] or metadata["experiment_id"]!=experiment["id"]
                or metadata["task_id"]!=experiment["task_id"] or metadata["universe_id"]!=proposal["spec"]["universe_id"]
                or metadata.get("registered_numerical_execution") is not True
                or metadata.get("holdout_rows_executed")!=0 or units!={"score":"ratio"}
                or type(attempt.get("rows")) is not int or type(attempt.get("present")) is not int
                or attempt["rows"]!=len(block.values) or attempt["present"]!=int(block.values.score.notna().sum())):
            raise ValueError("Numerical feature definition/summary differs")
        for name in ("known_at","observed_end"):
            check=block.values[keys+[name]].merge(membership[keys+["decision_at"]],on=keys,validate="one_to_one")
            if not pd.to_datetime(check[name],utc=True).equals(pd.to_datetime(check.decision_at,utc=True)):
                raise ValueError("Numerical output clocks differ from declared development decisions")
    if not actual<=allowed or not completed or result["completed_candidates"]!=completed or result["failed_candidates"]!=len(candidates)-completed:
        raise ValueError("Numerical completion/failed counts or files differ")
    return dict(schema="alpha-frozen-output-audit-v1",verified=True,experiment_id=experiment["id"],
        candidate_count=len(candidates),completed_candidates=completed,
        failed_candidates=len(candidates)-completed,rows_per_completed_candidate=len(membership),
        fits=0,accounts=0,holdout_rows_executed=0,numeric_output_reproduction=False,
        manifest_hash=digest(folder/"manifest.json"),
        verification_scope="registered source/output file identities, full development keys, clocks and value/missing consistency; not vendor authentication, formula math reexecution or strategy performance")
