"""Frozen feature-only numerical worker. No model fits, accounts or free eval."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import threading
import traceback

from .alpha_experiments import inspect_inputs
import psutil
import pyarrow as pa

from .resources import WorkerResources, hidden_console_record
from ..alpharesearch.dsl import verify_compiled
from ..alpharesearch.features.base import FeatureBlock
from ..technical.artifacts import content_id, digest, verify_artifacts, write_json


def verify_job(job):
    job = Path(job).resolve();code = job/"code"
    cfg = json.loads((job/"input.json").read_text(encoding="utf-8"))
    ready = json.loads((job/"ready.json").read_text(encoding="utf-8"))
    expected = dict(kind="alpha_batch", input_hash=digest(job/"input.json"),
        code_manifest_hash=digest(job/"code_manifest.json"),
        candidate_manifest_hash=digest(job/"candidate_manifest.json"))
    if ready!=expected or cfg["experiment_id"]!=job.name or Path(cfg["root"]).resolve()!=job.parent.parent:
        raise ValueError("Frozen batch job identity differs")
    inventory = json.loads((job/"code_manifest.json").read_text(encoding="utf-8"))
    actual = {p.relative_to(code).as_posix() for p in code.rglob("*") if p.is_file()}
    if set(inventory)!={"artifacts"} or actual!=set(inventory["artifacts"]):
        raise ValueError("Frozen import file set differs")
    if any(p.is_symlink() or (hasattr(p,"is_junction") and p.is_junction()) for p in code.rglob("*")):
        raise ValueError("Frozen source contains linked paths")
    for name in actual:
        path=code/name
        if path.is_symlink() or not path.resolve().is_relative_to(code):
            raise ValueError("Frozen source path escaped worker code root")
    verify_artifacts(code, inventory)
    if inventory["artifacts"]!=cfg["engine_inventory"] or content_id(inventory["artifacts"])!=cfg["engine_hash"]:
        raise ValueError("Worker source differs from frozen registration")
    environment = dict(python=platform.python_version(), packages={
        name:importlib.metadata.version(name) for name in cfg["environment"]["packages"]})
    if environment!=cfg["environment"]:
        raise ValueError("Worker environment differs from registration")
    WorkerResources.from_dict(cfg["resources"])
    return cfg


def compute(job):
    job=Path(job).resolve()
    # Never execute current editable source instead of the frozen module.
    if Path(__file__).resolve().parents[2]!=(job/"code").resolve():
        raise ValueError("Numerical worker must execute the registered frozen module")
    cfg=verify_job(job)
    resources=WorkerResources.from_dict(cfg["resources"])
    pa.set_cpu_count(resources.library_threads);pa.set_io_thread_count(1)
    spec,registry,info=inspect_inputs(cfg["project"],cfg,include_evaluator=True)
    evaluator=info.pop("_evaluator");doc=spec.to_dict()
    runtime={"environment_threads":{name:os.environ.get(name) for name in (
        "OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS","ARROW_NUM_THREADS")},
        "arrow_cpu_threads":pa.cpu_count(),"arrow_io_threads":pa.io_thread_count()}
    if any(v!=str(resources.library_threads) for v in runtime["environment_threads"].values()):
        raise ValueError("Worker numerical thread environment differs")
    planned=json.loads((job/"candidate_manifest.json").read_text(encoding="utf-8"))
    if planned["attempts"]!=spec.planned_attempts or planned["candidate_count"]!=len(doc["candidates"]):
        raise ValueError("Candidate execution plan differs")
    folder=job/"alpha_result";folder.mkdir(exist_ok=False)
    attempts=[dict(item) for item in spec.planned_attempts]
    write_json(folder/"attempts.json",attempts)
    began=time.monotonic()
    from .alpha_execution import cancel_reason
    for index,candidate in enumerate(doc["candidates"]):
        reason=cancel_reason(job)
        if reason is not None:raise RuntimeError("Cancellation requested: "+reason)
        attempt=attempts[index];attempt["status"]="running";write_json(folder/"attempts.json",attempts)
        try:
            block=evaluator.evaluate(verify_compiled(candidate["expression"],registry))
            dates=block.values.trade_date
            keep=dates.between(doc["ranges"]["development_start"],doc["ranges"]["development_end"])
            values=block.values.loc[keep].reset_index(drop=True)
            missing=block.missing.loc[keep].reset_index(drop=True)
            meta=dict(block.metadata,registered_numerical_execution=True,
                task_id=cfg["task_id"],experiment_id=cfg["experiment_id"],
                source_file_authentication="registered role files verified by SHA; vendor history not authenticated",
                independent_reproduction=False,holdout_rows_executed=0)
            block=FeatureBlock(values,missing,block.units,meta).validate()
            used=sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            # A conservative frame estimate prevents beginning an output that
            # plainly exceeds the finite disk budget. Actual bytes checked too.
            projected=int(values.memory_usage(deep=True).sum()+missing.memory_usage(deep=True).sum()+20000)
            if used+projected>resources.max_output_bytes:
                raise ValueError("Projected batch output byte budget exceeded")
            target=folder/candidate["candidate_id"];target.mkdir(exist_ok=False)
            for name,frame in (("values.parquet",values),("missing.parquet",missing)):
                temporary=target/(name+".partial")
                frame.to_parquet(temporary,index=False)
                os.replace(temporary,target/name)
            write_json(target/"definition.json",dict(meta,units=block.units))
            actual_bytes=sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            if actual_bytes>resources.max_output_bytes:
                raise ValueError("Actual batch output byte budget exceeded")
            attempt.update(status="completed",rows=len(values),present=int(values.score.notna().sum()),
                           output_folder=candidate["candidate_id"])
        except Exception as exc:
            attempt.update(status="failed",error=str(exc))
            traceback.print_exc()
        write_json(folder/"attempts.json",attempts)
    # Source references can be changed by an external refresh; require the
    # original input identities still to hold before a terminal success.
    verify_job(job)
    from .alpha_experiments import verify_inputs
    verify_inputs(cfg["project"],cfg["inputs"],doc["budget"]["max_input_bytes"])
    completed=sum(a["status"]=="completed" for a in attempts)
    result=dict(schema="registered-alpha-feature-result-v1",task_id=cfg["task_id"],
        experiment_id=cfg["experiment_id"],spec=doc,engine_hash=cfg["engine_hash"],
        environment=cfg["environment"],resources=cfg["resources"],runtime_threads=runtime,registry_version=registry.version_id,
        candidate_count=len(attempts),completed_candidates=completed,
        failed_candidates=len(attempts)-completed,model_fits=0,accounts=0,
        holdout_rows_executed=0,selection_rule="all_candidates_no_selection",
        elapsed_seconds=time.monotonic()-began,evaluation_scope="retrospective_time_split",
        verification_scope="frozen numerical execution/output identity; not independent full numerical reproduction or account evidence")
    write_json(folder/"result.json",result)
    manifest=dict(schema="registered-alpha-feature-manifest-v1",experiment_id=cfg["experiment_id"],
        task_id=cfg["task_id"],input_hash=digest(job/"input.json"),engine_hash=cfg["engine_hash"],
        candidate_manifest_hash=digest(job/"candidate_manifest.json"),
        artifacts={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob("*") if p.is_file()})
    write_json(folder/"manifest.json",manifest)
    if sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())>resources.max_output_bytes:
        raise ValueError("Final numerical output byte budget exceeded; artifacts retained")
    if completed!=len(attempts):
        raise ValueError("Numerical batch contains failed candidates; entire batch failed, partial outputs and attempts retained")
    return folder


def _watchdog_memory(current, accepted_helpers):
    """Freeze first observed helper identities; count RSS and reject later children."""
    children=current.children(recursive=True)
    if len(children)>1:
        raise ValueError("Worker spawned unregistered processes")
    observed=[];helper_rss=0
    for child in children:
        try:
            record=hidden_console_record(child,current.pid,windows=os.name=="nt",
                                         system_root=os.environ.get("SystemRoot",""))
            observed.append(record)
            helper_rss+=child.memory_info().rss
        except psutil.NoSuchProcess:pass
    if accepted_helpers is None:
        accepted_helpers=tuple(observed)
    elif any(record not in accepted_helpers for record in observed):
        raise ValueError("Worker hidden-console identity changed or appeared after first sample")
    return current.memory_info().rss+helper_rss,accepted_helpers


def _watchdog(job, cfg, done):
    # Survives loss of the supervising parent. A sampled soft limit can overshoot;
    # this is not an OS memory sandbox. Startup imports precede this thread.
    from .alpha_execution import cancel_reason
    resources=WorkerResources.from_dict(cfg["resources"]);began=time.monotonic();helpers=None
    while not done.wait(.2):
        try:
            reason=cancel_reason(job)
            if reason is not None:raise RuntimeError("Cancellation requested: "+reason)
            if time.monotonic()-began>cfg["timeout_seconds"]:raise TimeoutError("Worker self-watchdog timeout exceeded")
            current=psutil.Process()
            rss,helpers=_watchdog_memory(current,helpers)
            if rss>resources.max_process_rss_bytes:raise MemoryError("Worker self-watchdog RSS limit exceeded")
            if psutil.virtual_memory().available<resources.min_available_memory_bytes:
                raise MemoryError("Worker self-watchdog available-memory floor crossed")
        except BaseException as exc:
            if done.is_set():return
            try:
                print("Worker watchdog stopped: "+str(exc),flush=True)
                path=job/"receipt.json"
                if not path.exists():
                    write_json(path,dict(status="failed",kind="alpha_batch",experiment_id=job.name,error=str(exc)))
            finally:
                # Record-write failure must not leave an unmonitored worker alive.
                os._exit(1)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("job",type=Path)
    args=parser.parse_args();job=args.job.resolve();done=threading.Event();watchdog=None
    try:
        cfg=verify_job(job)
        startup=dict(schema="alpha-worker-startup-v1",experiment_id=job.name,
                     input_hash=digest(job/"input.json"),pid=os.getpid(),
                     created=psutil.Process().create_time())
        # Atomic visibility: parent never reads a partially written handshake.
        temp=job/"worker_started.tmp"
        write_json(temp,startup)
        temp.replace(job/"worker_started.json")
        watchdog=threading.Thread(target=_watchdog,args=(job,cfg,done),daemon=True);watchdog.start()
        folder=compute(job)
        done.set();watchdog.join()
        write_json(job/"receipt.json",dict(status="completed",kind="alpha_batch",
            experiment_id=job.name,result_manifest_hash=digest(folder/"manifest.json")))
    except BaseException as exc:
        done.set()
        if watchdog is not None:watchdog.join()
        if not (job/"receipt.json").exists():
            write_json(job/"receipt.json",dict(status="failed",kind="alpha_batch",
                experiment_id=job.name,error=str(exc)))
        traceback.print_exc()
        raise SystemExit(1)


if __name__=="__main__":
    main()
