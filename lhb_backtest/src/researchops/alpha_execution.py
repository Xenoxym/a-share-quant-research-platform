"""Supervised frozen alpha batches. Completion is feature evidence, never account evidence."""
import json
import os
import subprocess
import sys
import time

import psutil

from .resources import WorkerResources, process_alive, reserve, release_terminal_reservation, check_terminal_reservation, _guard, GuardBusy, hidden_console_record
from ..technical.artifacts import digest, write_json

THREAD_ENV=("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS","ARROW_NUM_THREADS")


class CollectionCommitPending(RuntimeError):
    """Collection storage unavailable; retry collection, never execution."""


def _collection_experiment(research, eid):
    """Only use after exit/inside collection, never for prelaunch authorization."""
    try:
        return research.store.experiment(eid)
    except Exception as exc:
        raise CollectionCommitPending(
            "Terminal collection database read pending for "+eid) from exc


def _commit_outcome(research, eid, status, **fields):
    try:
        return research.store.finish(eid,status,**fields)
    except Exception as exc:
        raise CollectionCommitPending(
            "Terminal collection database commit pending for "+eid+" ("+status+")") from exc


def _read_json(path):
    if path.is_symlink() or (hasattr(path,"is_junction") and path.is_junction()):
        raise ValueError("Linked execution metadata refused")
    return json.loads(path.read_text(encoding="utf-8"))


def _launch(research, experiment):
    job=research.root/"worker_jobs"/experiment["id"];path=job/"launch.json"
    if not path.exists():return None
    launch=_read_json(path)
    expected=dict(schema="alpha-worker-launch-v1",experiment_id=experiment["id"],
                  input_hash=digest(job/"input.json"),resources=experiment["proposal"]["resources"])
    if set(launch)!=set(expected)|{"pid","created","interpreter_chain"} or any(launch.get(k)!=v for k,v in expected.items()):
        raise ValueError("Launched process differs from frozen job")
    if experiment.get("pid")!=launch["pid"]:
        raise ValueError("Launched process differs from registered PID")
    process_alive(launch["pid"],launch["created"])  # validates identity; no process is killed here
    chain=launch["interpreter_chain"]
    if not isinstance(chain,list) or len(chain)>2 or len({c.get("role") for c in chain})!=len(chain):
        raise ValueError("Bounded interpreter/console helper launch chain required")
    for child in chain:
        if not isinstance(child,dict) or set(child)!={"pid","created","role"} or child["role"] not in {"interpreter","console_host"}:
            raise ValueError("Strict interpreter/helper child identity required")
        process_alive(child["pid"],child["created"])
    return launch


def _alive(launch):
    return process_alive(launch["pid"],launch["created"]) or any(
        process_alive(c["pid"],c["created"]) for c in launch["interpreter_chain"])



def _interpreter_chain(proc, created, command):
    # The Windows venv launcher and hidden console host are bookkeeping,
    # alongside one actual Python interpreter, not extra numerical workers.
    launcher=(os.name=="nt" and os.path.normcase(sys.executable)!=os.path.normcase(sys._base_executable))
    deadline=time.monotonic()+2
    while proc.poll() is None:
        parent=psutil.Process(proc.pid)
        if parent.create_time()!=created:raise ValueError("Launcher PID identity changed")
        children=parent.children(recursive=True);chain=[]
        if len(children)>2:raise ValueError("Unexpected worker child process tree")
        for child in children:
            try:
                executable=os.path.normcase(child.exe());args=child.cmdline()
                if child.ppid()!=proc.pid:raise ValueError("Unexpected worker child ancestry")
                if launcher and executable==os.path.normcase(sys._base_executable) and args[1:]==command[1:]:
                    record=dict(pid=child.pid,created=child.create_time(),role="interpreter")
                else:
                    record=hidden_console_record(child,proc.pid,windows=os.name=="nt",
                                                 system_root=os.environ.get("SystemRoot",""))
                chain.append(record)
            except psutil.NoSuchProcess:pass
        if len({c["role"] for c in chain})!=len(chain):raise ValueError("Duplicate interpreter/helper process")
        if any(c["role"]=="interpreter" for c in chain):return sorted(chain,key=lambda c:c["role"])
        if not launcher:return sorted(chain,key=lambda c:c["role"])
        if time.monotonic()>deadline:raise ValueError("Windows interpreter launch identity not observed")
        time.sleep(.02)
    raise ValueError("Worker exited before exact interpreter identity was recorded")


def _sample(launch):
    total=0;alive=[]
    for record in [launch,*launch["interpreter_chain"]]:
        try:
            current=psutil.Process(record["pid"])
            if current.create_time()!=record["created"]:continue
            if current.is_running():
                total+=current.memory_info().rss;alive.append(current)
        except psutil.NoSuchProcess:pass
    declared={(c["pid"],c["created"]) for c in launch["interpreter_chain"]}
    for process in alive:
        try:
            for child in process.children(recursive=True):
                try:
                    if (child.pid,child.create_time()) not in declared:
                        raise ValueError("Alpha feature worker spawned unregistered child processes")
                except psutil.NoSuchProcess:pass
        except psutil.NoSuchProcess:pass
    return total


def _receipt(research, experiment, receipt):
    if not isinstance(receipt,dict) or receipt.get("kind")!="alpha_batch" or receipt.get("experiment_id")!=experiment["id"]:
        raise ValueError("Execution receipt ownership differs")
    if receipt.get("status")=="completed":
        if set(receipt)!={"status","kind","experiment_id","result_manifest_hash"}:
            raise ValueError("Strict completed alpha receipt required")
        if receipt["result_manifest_hash"]!=digest(research.root/"worker_jobs"/experiment["id"]/"alpha_result/manifest.json"):
            raise ValueError("Execution receipt result manifest changed")
    elif receipt.get("status")=="failed":
        if set(receipt)!={"status","kind","experiment_id","error"} or not isinstance(receipt["error"],str) or not receipt["error"]:
            raise ValueError("Strict failed alpha receipt required")
    else:
        raise ValueError("Unknown alpha receipt status")


def _write_once(path, value):
    if path.exists():
        if _read_json(path)!=value:raise ValueError("Existing execution metadata differs; preserve it")
    else:write_json(path,value)


def _record_observation(job, value):
    """Keep the first valid observation, including a recovered coverage gap."""
    with _guard(job):
        path=job/"execution_stats.json"
        if path.exists():
            saved=_read_json(path)
            if (not isinstance(saved,dict) or any(saved.get(k)!=value.get(k)
                    for k in ("schema","experiment_id","launch"))):
                raise ValueError("Existing observation ownership differs; preserve it")
            return saved
        write_json(path,value)
        return value


def request_cancel(research, session, eid, reason):
    from .store import text
    reason=text(reason,"Cancellation reason")
    with research.store.connection(True) as db:
        task=research.store.owned(db,session);ex=research.store.experiment(eid)
        if ex["task_id"]!=task["id"] or ex["status"]!="running" or ex["proposal"].get("kind")!="alpha_batch":
            raise ValueError("Only an owned running alpha batch supports cancellation requests")
        job=research.root/"worker_jobs"/eid
        value=dict(schema="alpha-cancel-request-v1",experiment_id=eid,
                   input_hash=digest(job/"input.json"),reason=reason,worker=session["worker"])
        path=job/"cancel_request.json"
        if path.exists():
            old=_read_json(path)
            if old["experiment_id"]!=eid or old["input_hash"]!=value["input_hash"]:
                raise ValueError("Existing cancellation belongs to another input")
            return dict(ex,cancellation_requested=True,cancellation_reason=old["reason"])
        write_json(path,value)
        research.store.event(db,task["id"],session["worker"],"alpha_cancellation_requested",
                             {"experiment_id":eid,"reason":reason})
    return dict(ex,cancellation_requested=True,cancellation_reason=reason)


def cancel_reason(job):
    path=job/"cancel_request.json"
    if not path.exists():return None
    request=_read_json(path)
    if (set(request)!={"schema","experiment_id","input_hash","reason","worker"}
            or request["schema"]!="alpha-cancel-request-v1" or request["experiment_id"]!=job.name
            or request["input_hash"]!=digest(job/"input.json")
            or not isinstance(request["reason"],str) or not request["reason"].strip()):
        raise ValueError("Cancellation request identity differs")
    return request["reason"]


def _supervision_failure(job, eid, error=None):
    """Immutable parent failure, serialized with collectors; child never writes it."""
    path=job/"supervision_failure.json"
    expected=dict(schema="alpha-supervision-failure-v1",experiment_id=eid,
                  input_hash=digest(job/"input.json"),
                  launch_hash=digest(job/"launch.json") if (job/"launch.json").exists() else None)
    if path.exists():
        saved=_read_json(path)
        if (not isinstance(saved,dict) or set(saved)!=set(expected)|{"error"}
                or any(saved.get(k)!=v for k,v in expected.items())
                or not isinstance(saved["error"],str) or not saved["error"].strip()):
            raise ValueError("Supervision failure identity differs; preserve it")
        return saved
    if error is None:return None
    if not isinstance(error,str) or not error.strip():raise ValueError("Supervision failure cause required")
    value=dict(expected,error=error)
    write_json(path,value)
    return value


def _record_supervision_failure(job, eid, error):
    with _guard(job):
        return _supervision_failure(job,eid,error)


def _failure_receipt(job, eid, error):
    path=job/"receipt.json"
    # Only called after stopping the worker; collection uses this same job lock.
    with _guard(job):
        if not path.exists():
            write_json(path,dict(status="failed",kind="alpha_batch",experiment_id=eid,error=error))


def execute(research, session, eid):
    from .alpha_experiments import verify_registered
    from .alpha_worker import verify_job
    with research.store.connection() as db:research.store.owned(db,session)
    experiment=research.store.experiment(eid)
    if experiment["task_id"]!=session["task_id"] or experiment["status"]!="planned":
        raise ValueError("Batch belongs to another task or already started; never repeat execution")
    verify_registered(research,experiment)
    job=research.root/"worker_jobs"/eid
    if "resources" not in experiment["proposal"] or not (job/"code/src/researchops/alpha_worker.py").is_file():
        raise ValueError("Old registration has no frozen E12 worker/resources; register a new version without rewriting history")
    if any((job/name).exists() for name in ("launch.json","receipt.json","alpha_result","execution_stats.json","cancel_request.json","supervision_failure.json")):
        raise ValueError("Unexpected existing execution artifacts; preserve and inspect")
    verify_job(job)
    resources=WorkerResources.from_dict(experiment["proposal"]["resources"])
    # Resource denial occurs before Store.start, so no running job is stranded.
    with reserve(research.root,eid,resources) as reservation:
        research.store.start(session,eid)
        proc=None;began=time.monotonic();last=began;peak=0;samples=0;minimum=None
        try:
            minimum=psutil.virtual_memory().available
            research.store.heartbeat(session)
            with (job/"execution.log").open("x",encoding="utf-8") as log:
                env=dict(os.environ,PYTHONUTF8="1",PYTHONIOENCODING="utf-8",
                         PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(job/"code"))
                env.update({key:str(resources.library_threads) for key in THREAD_ENV})
                flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
                reservation.launching()
                try:
                    command=[sys.executable,"-B","-m","src.researchops.alpha_worker",str(job)]
                    proc=subprocess.Popen(command,
                                          cwd=job/"code",env=env,stdout=log,stderr=subprocess.STDOUT,
                                          creationflags=flags)
                    created=reservation.bind(proc)
                    chain=_interpreter_chain(proc,created,command)
                    reservation.bind_chain(chain)
                except BaseException:
                    if proc is not None and proc.poll() is None:research._stop(proc)
                    if reservation.record["child_pid"] is None:reservation.abort_unbound_launch(proc)
                    raise
                launch=dict(schema="alpha-worker-launch-v1",experiment_id=eid,pid=proc.pid,
                            created=created,interpreter_chain=chain,
                            input_hash=digest(job/"input.json"),resources=resources.to_dict())
                write_json(job/"launch.json",launch)
                with research.store.connection(True) as db:
                    db.execute("UPDATE experiments SET pid=? WHERE id=?",(proc.pid,eid))
                while proc.poll() is None:
                    elapsed=time.monotonic()-began
                    if elapsed>experiment["proposal"]["timeout_seconds"]:
                        raise TimeoutError("Registered alpha worker timeout exceeded")
                    reason=cancel_reason(job)
                    if reason is not None:raise RuntimeError("Cancellation requested: "+reason)
                    rss=_sample(launch)
                    available=psutil.virtual_memory().available
                    peak=max(peak,rss);minimum=min(minimum,available);samples+=1
                    if rss>resources.max_process_rss_bytes:raise MemoryError("Sampled worker RSS limit exceeded")
                    if available<resources.min_available_memory_bytes:raise MemoryError("Available-memory reserve crossed")
                    if time.monotonic()-last>15:
                        try:research.store.heartbeat(session)
                        except ValueError:pass  # a new owner can collect this exact job
                        last=time.monotonic()
                    time.sleep(.2)
            proc.wait(timeout=15)
            while _alive(launch):
                if time.monotonic()-began>experiment["proposal"]["timeout_seconds"]:
                    raise TimeoutError("Registered launch helper exit timeout exceeded")
                time.sleep(.02)
            stats=dict(schema="alpha-worker-observation-v1",experiment_id=eid,launch=launch,
                       elapsed_seconds=time.monotonic()-began,peak_sampled_rss_bytes=peak,
                       minimum_available_memory_bytes=minimum,samples=samples,sample_interval_seconds=.2,
                       child_exit_code=proc.returncode,parent_observed_to_exit=True,
                       policy="sampled_soft_stop; not an OS hard sandbox")
            try:
                _record_observation(job,stats)
                if not (job/"receipt.json").is_file():
                    raise RuntimeError("Worker exited without terminal receipt; inspect retained log/outputs")
                return recover(research,_collection_experiment(research,eid))
            except (GuardBusy,CollectionCommitPending) as pending:
                # Another collector or the terminal commit can still be pending.
                # Preserve its slot and receipt so retrying collection remains safe.
                reservation.retain_on_exit=True
                latest=_collection_experiment(research,eid)
                if latest["status"]=="completed":
                    verify_completed(research,latest)
                    return latest
                if latest["status"]!="running":
                    return latest
                note=("Collection guard busy" if isinstance(pending,GuardBusy)
                      else "Collection database commit unavailable")
                return dict(latest,
                            recovery_note=note+"; terminal collection pending. Exact reservation retained; retry recover, never execute again.")
        except CollectionCommitPending:
            reservation.retain_on_exit=True
            raise
        except BaseException as exc:
            # A metadata write failure must not skip the original failed state
            # and then release the only independent process identity.
            reservation.retain_on_exit=True
            gaps=[];stop_uncertain=False;cause=str(exc) or type(exc).__name__
            try:
                if proc is not None and proc.poll() is None:research._stop(proc)
            except Exception as stop_error:
                stop_uncertain=True;gaps.append("process stop: "+str(stop_error))
            try:
                # Persist the primary cause independently of a worker receipt.
                # Collection takes the same guard and cannot consume child
                # success after this record has been committed.
                _record_supervision_failure(job,eid,cause)
            except Exception as failure_error:
                gaps.append("supervision failure record: "+str(failure_error))
            try:
                research.store.finish(eid,"failed",error=cause)
            except Exception as state_error:
                gaps.append("terminal database commit: "+str(state_error))
            else:
                reservation.retain_on_exit=stop_uncertain
            try:
                if proc is not None and reservation.record["child_pid"] is not None and not (job/"execution_stats.json").exists():
                    _record_observation(job,dict(schema="alpha-worker-observation-v1",
                        experiment_id=eid,launch=_read_json(job/"launch.json") if (job/"launch.json").exists() else None,
                        elapsed_seconds=time.monotonic()-began,peak_sampled_rss_bytes=peak,
                        minimum_available_memory_bytes=minimum,samples=samples,sample_interval_seconds=.2,
                        child_exit_code=proc.returncode,parent_observed_to_exit=not stop_uncertain,
                        stop_cause=cause,policy="sampled_soft_stop; not an OS hard sandbox"))
            except Exception as observation_error:
                gaps.append("observation: "+str(observation_error))
            try:
                _failure_receipt(job,eid,cause)
            except Exception as receipt_error:
                gaps.append("failure receipt: "+str(receipt_error))
            if gaps:
                try:
                    with research.store.connection(True) as db:
                        research.store.event(db,experiment["task_id"],session["worker"],
                            "alpha_parent_failure_record_gap",dict(experiment_id=eid,original_error=cause,record_gaps=gaps))
                except Exception:
                    pass  # simultaneous storage failure cannot guarantee a second record
            raise


def recover(research, experiment, receipt=None):
    """Serialize terminal collection against all parent observation writers."""
    job=research.root/"worker_jobs"/experiment["id"]
    with _guard(job):
        return _recover_locked(research,_collection_experiment(research,experiment["id"]),receipt)


def _recover_locked(research, experiment, receipt=None):
    """Collect an exited exact worker only; never restart or infer a missing receipt."""
    job=research.root/"worker_jobs"/experiment["id"]
    launch=_launch(research,experiment)
    if launch is None:
        return dict(experiment,recovery_note="No exact launch identity; inspect retained artifacts. No automatic restart/release.")
    if _alive(launch):
        return dict(experiment,recovery_note="Exact worker still alive; no result collected or resource released.")
    if experiment["status"] in {"completed","failed"}:
        # Terminal evidence remains immutable, but a busy pool lock can have
        # delayed release. Public recovery must retry only that exact cleanup.
        if experiment["status"]=="completed":
            verify_completed(research,experiment)
        release_terminal_reservation(research.root,experiment["id"],launch)
        return experiment
    supervision=_supervision_failure(job,experiment["id"])
    if receipt is None and supervision is None:
        if not (job/"receipt.json").is_file():
            return dict(experiment,recovery_note="Exited worker has no terminal receipt; inspect artifacts. No automatic restart/release.")
        receipt=_read_json(job/"receipt.json")
    # Check the independent slot before a forged old creation time or empty
    # interpreter chain can turn a live worker into a committed completion.
    try:
        terminal=check_terminal_reservation(research.root,experiment["id"],launch)
    except ValueError:
        latest=_collection_experiment(research,experiment["id"])
        if latest["status"]=="completed":
            verify_completed(research,latest)
            return latest
        if latest["status"]=="failed":return latest
        raise
    if not terminal:
        return dict(experiment,recovery_note="Registered resource owner still alive; no terminal state committed.")
    # Identify/validate the outcome separately from committing it. A transient
    # database error must not become a second computation failure transaction.
    result=None;error=None
    try:
        if supervision is not None:
            status="failed";error=supervision["error"]
        else:
            _receipt(research,experiment,receipt)
            status=receipt["status"]
            if status=="failed":
                error=receipt["error"]
            else:
                from .alpha_output_audit import audit_output
                audit=audit_output(research,experiment)
                _write_once(job/"alpha_audit.json",audit)
                if not (job/"execution_stats.json").exists():
                    write_json(job/"execution_stats.json",dict(schema="alpha-worker-observation-v1",
                        experiment_id=experiment["id"],launch=launch,parent_observed_to_exit=False,
                        peak_sampled_rss_bytes=None,elapsed_seconds=None,
                        policy="recovered terminal worker; parent observation coverage unavailable"))
                result=dict(kind="registered_alpha_feature_execution",**audit,
                            audit_hash=digest(job/"alpha_audit.json"),receipt_hash=digest(job/"receipt.json"),
                            launch_hash=digest(job/"launch.json"),observation_hash=digest(job/"execution_stats.json"))
    except Exception as exc:
        # Preserve an actual validation failure even if its commit is unavailable.
        # The surrounding collector already owns the job guard.
        original=_supervision_failure(job,experiment["id"],str(exc) or type(exc).__name__)
        _commit_outcome(research,experiment["id"],"failed",error=original["error"])
        release_terminal_reservation(research.root,experiment["id"],launch)
        raise
    final=_commit_outcome(research,experiment["id"],status,result=result,error=error)
    release_terminal_reservation(research.root,experiment["id"],launch)
    return final



def verify_completed(research, experiment):
    if experiment["status"]!="completed" or not isinstance(experiment.get("result"),dict):
        raise ValueError("Completed alpha execution evidence required")
    from .alpha_output_audit import audit_output
    job=research.root/"worker_jobs"/experiment["id"];launch=_launch(research,experiment)
    if launch is None or _alive(launch):
        raise ValueError("Completed worker launch not terminal")
    _receipt(research,experiment,_read_json(job/"receipt.json"))
    audit=audit_output(research,experiment);saved=_read_json(job/"alpha_audit.json");stored=experiment["result"]
    if saved!=audit or any(stored.get(k)!=v for k,v in audit.items()):
        raise ValueError("Stored alpha audit differs from current output identities")
    if stored.get("kind")!="registered_alpha_feature_execution":
        raise ValueError("Wrong completed result kind")
    for key,name in (("audit_hash","alpha_audit.json"),("receipt_hash","receipt.json"),
                     ("launch_hash","launch.json"),("observation_hash","execution_stats.json")):
        if stored.get(key)!=digest(job/name):raise ValueError("Completed alpha evidence changed: "+name)
    return audit
