"""Single frozen native scored account with explicit zero/configured scenarios."""
import argparse
import os
import re
import stat
from pathlib import Path
import threading
import traceback
import psutil
import pyarrow as pa
from .alpha_execution import THREAD_ENV, cancel_reason
from .alpha_worker import verify_frozen_job, _watchdog
from .account_experiments import inspect_inputs, native_folder, same_json
from .resources import WorkerResources
from ..alpharesearch.account import prepare_score_account
from ..technical.artifacts import digest, write_json
from ..technical.contracts import SCENARIOS
from ..technical.runner import run
FILES={"attempts.json","native_pointer.json"}

def verify_job(job): return verify_frozen_job(job,"alpha_account")

def output_bytes(job,cfg):
    total=0
    for folder in (job/"account_result",native_folder(cfg)):
        if folder.exists():
            for p in folder.rglob("*"):
                try:
                    file_stat=p.lstat()
                except FileNotFoundError:
                    # A writer can atomically rename its JSON UUID temp after the
                    # listing. Only a truly absent temp entry is harmless. lstat
                    # observes broken links too; canonical disappearance is fatal.
                    if not re.fullmatch(r".+\.json\.[0-9a-f]{32}\.tmp", p.name):
                        raise
                    try:
                        p.lstat()
                    except FileNotFoundError:
                        continue
                    raise
                if stat.S_ISLNK(file_stat.st_mode) or (hasattr(p,"is_junction") and p.is_junction()):
                    raise ValueError("Linked account output refused")
                if stat.S_ISREG(file_stat.st_mode):
                    total+=file_stat.st_size
    return total

def compute(job):
    if Path(__file__).resolve().parents[2]!=(job/"code").resolve(): raise ValueError("Account must run frozen modules")
    cfg=verify_job(job); resources=WorkerResources.from_dict(cfg["resources"])
    pa.set_cpu_count(resources.library_threads);pa.set_io_thread_count(1)
    if any(os.environ.get(k)!=str(resources.library_threads) for k in THREAD_ENV): raise ValueError("Account thread budget differs")
    if cancel_reason(job) is not None: raise RuntimeError("Cancellation requested")
    native=native_folder(cfg)
    if native.exists(): raise ValueError("Native account artifacts already exist; never restart")
    _,_,info=inspect_inputs(cfg["project"],cfg,cfg["root"],include_data=True); data=info.pop("data")
    if not same_json(info,cfg["inspection"]): raise ValueError("Frozen account input scope differs")
    folder=job/"account_result";folder.mkdir(exist_ok=False)
    attempts=dict(schema="alpha-account-attempts-v1",planned_accounts=cfg["planned_accounts"],model_fits=0,
        scenarios=[dict(scenario=s,status="planned") for s in cfg["spec"]["scenarios"]])
    write_json(folder/"attempts.json",attempts)
    def progress(message):
        print(message,flush=True)
        for entry in attempts["scenarios"]:
            if message=="重算账户："+SCENARIOS[entry["scenario"]]:
                for prior in attempts["scenarios"]:
                    if prior["status"]=="running": prior["status"]="completed"
                entry["status"]="running";write_json(folder/"attempts.json",attempts)
    try:
        prepared=prepare_score_account(*data)
        spec=data[-1]
        result=run(cfg["project"],spec,root=cfg["root"],prepared=prepared,
            frozen_snapshot=data[-2],lock_root=job,scenarios=cfg["spec"]["scenarios"],publish=False,
            registered_run_id=cfg["native_run_id"],progress=progress,
            research_context=dict(role="screen",source="registered_alpha_account",task_id=cfg["task_id"],experiment_id=job.name))
        if result!=native: raise ValueError("Native account location differs")
        for entry in attempts["scenarios"]:
            if entry["status"]=="running": entry["status"]="completed"
        if any(e["status"]!="completed" for e in attempts["scenarios"]): raise ValueError("Scenario attempts incomplete")
        write_json(folder/"attempts.json",attempts)
    except BaseException as exc:
        for entry in attempts["scenarios"]:
            if entry["status"]=="running":entry["status"]="failed"
        attempts["error"]=str(exc);write_json(folder/"attempts.json",attempts);raise
    write_json(folder/"native_pointer.json",dict(run_id=native.name,manifest_hash=digest(native/"manifest.json")))
    verify_job(job)
    write_json(folder/"manifest.json",dict(schema="registered-alpha-account-manifest-v1",experiment_id=job.name,
        task_id=cfg["task_id"],input_hash=digest(job/"input.json"),engine_hash=cfg["engine_hash"],
        candidate_manifest_hash=digest(job/"candidate_manifest.json"),native_manifest_hash=digest(native/"manifest.json"),
        artifacts={name:digest(folder/name) for name in FILES}))
    if output_bytes(job,cfg)>resources.max_output_bytes: raise ValueError("Account output byte budget exceeded; retain artifacts")
    return folder

def main():
    parser=argparse.ArgumentParser();parser.add_argument("job",type=Path);job=parser.parse_args().job.resolve()
    done=threading.Event();watchdog=None
    try:
        cfg=verify_job(job)
        temp=job/"worker_started.tmp";write_json(temp,dict(schema="alpha-worker-startup-v1",experiment_id=job.name,
            input_hash=digest(job/"input.json"),pid=os.getpid(),created=psutil.Process().create_time()))
        temp.replace(job/"worker_started.json")
        watchdog=threading.Thread(target=_watchdog,args=(job,cfg,done),daemon=True);watchdog.start()
        folder=compute(job);done.set();watchdog.join()
        write_json(job/"receipt.json",dict(status="completed",kind="alpha_account",experiment_id=job.name,
            result_manifest_hash=digest(folder/"manifest.json")))
    except BaseException as exc:
        done.set()
        if watchdog is not None:watchdog.join()
        if not (job/"receipt.json").exists():write_json(job/"receipt.json",dict(status="failed",kind="alpha_account",experiment_id=job.name,error=str(exc) or type(exc).__name__))
        traceback.print_exc();raise SystemExit(1)
if __name__=="__main__": main()
