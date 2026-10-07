"""E12 executable freeze, complete keys/provenance, finite reservations and recovery."""
import copy
import json
import os
from types import SimpleNamespace

import pandas as pd
import psutil
import pytest

from src.researchops.alpha_experiments import verify_completed
from src.researchops.alpha_execution import cancel_reason
from src.researchops.alpha_output_audit import audit_output
from src.researchops.alpha_worker import _watchdog
from src.researchops.resources import WorkerResources, process_alive, reserve, release_terminal_reservation
from src.technical.artifacts import digest, write_json
from tests.test_alpha_batch_registration import setup


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    ops,session,p,_,_=setup(tmp_path_factory.mktemp("ex"))
    p["spec"]["candidates"].append(dict(copy.deepcopy(p["spec"]["candidates"][0]),
                                        candidate_id="identical_counted"))
    ex=ops.register(session,p)
    # Current editable code is not the source used by the numerical child.
    (ops.project/"src/alpharesearch/cache.py").write_text("not valid current python",encoding="utf-8")
    final=ops.execute(session,ex["id"])
    assert final["status"]=="completed"
    return ops,session,final,ops.root/"worker_jobs"/ex["id"]


def test_frozen_process_hand_values_development_only_and_no_rerun(completed):
    ops,session,ex,job=completed
    audit=verify_completed(ops,ex)
    assert audit["candidate_count"]==2 and audit["completed_candidates"]==2
    assert audit["fits"]==audit["accounts"]==audit["holdout_rows_executed"]==0
    assert not audit["numeric_output_reproduction"]
    result=json.loads((job/"alpha_result/result.json").read_text(encoding="utf-8"))
    assert result["runtime_threads"]["arrow_cpu_threads"]==2 and result["runtime_threads"]["arrow_io_threads"]==1
    for cid in ("relative_close_rank","identical_counted"):
        values=pd.read_parquet(job/"alpha_result"/cid/"values.parquet")
        assert values.trade_date.tolist()==["2024-01-03"]*2+["2024-01-04"]*2
        assert values.score.tolist()==[.75]*4  # tied rank (1+2)/2 divided by two stocks
    stats=json.loads((job/"execution_stats.json").read_text(encoding="utf-8"))
    assert stats["samples"]>0 and stats["parent_observed_to_exit"] and stats["child_exit_code"]==0
    assert stats["peak_sampled_rss_bytes"]>0
    assert not list((ops.root/".alpha_slots").glob("slot*.json"))
    assert ops.recover(ex["id"])["id"]==ex["id"]
    with pytest.raises(ValueError,match="already started"):
        ops.execute(session,ex["id"])


@pytest.mark.parametrize("mutation",["dropped_member","holdout_row","earlier_clock","future_clock",
    "wrong_source","wrong_context","wrong_dependencies","wrong_impl","false_vintage","wrong_owner",
    "extra_file","nested_manifest","missing_reason","attempt_order","false_fit_count","bool_counter",
    "result_env","result_engine","manifest_owner","hidden_as_failed"])
def test_independent_audit_rejects_coherently_resealed_wrong_outputs(completed,mutation):
    ops,_,ex,job=completed;folder=job/"alpha_result";target=folder/"relative_close_rank"
    originals={p.relative_to(folder):p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    try:
        if mutation in {"dropped_member","holdout_row","earlier_clock","future_clock","missing_reason"}:
            values=pd.read_parquet(target/"values.parquet");missing=pd.read_parquet(target/"missing.parquet")
            if mutation=="dropped_member":
                values=values.iloc[:1];missing=missing.iloc[:1]
                attempts=json.loads((folder/"attempts.json").read_text(encoding="utf-8"))
                attempts[0].update(rows=1,present=1);write_json(folder/"attempts.json",attempts)
            if mutation=="holdout_row":
                values["trade_date"]="2024-01-08";missing["trade_date"]="2024-01-08"
            if mutation in {"earlier_clock","future_clock"}:
                values["known_at"]="2024-01-03T15:11:00+08:00" if mutation=="future_clock" else "2024-01-03T15:09:00+08:00"
                if mutation=="earlier_clock":values["observed_end"]=values.known_at
            if mutation=="missing_reason":missing["score"]="source_missing"
            values.to_parquet(target/"values.parquet",index=False)
            missing.to_parquet(target/"missing.parquet",index=False)
        elif mutation in {"wrong_source","wrong_context","wrong_dependencies","wrong_impl","false_vintage","wrong_owner"}:
            meta=json.loads((target/"definition.json").read_text(encoding="utf-8"))
            key,value={"wrong_source":("source_id","another_source"),"wrong_context":("context_id","f"*64),
                "wrong_dependencies":("dependencies",[]),"wrong_impl":("implementation_hashes",[]),
                "false_vintage":("vintage","historical_versions"),"wrong_owner":("experiment_id","exp-ffffffffffff")}[mutation]
            meta[key]=value;write_json(target/"definition.json",meta)
        elif mutation=="hidden_as_failed":
            for name in ("values.parquet","missing.parquet","definition.json"):(target/name).unlink()
            attempts=json.loads((folder/"attempts.json").read_text(encoding="utf-8"))
            attempts[0].update(status="failed",error="fabricated failure")
            write_json(folder/"attempts.json",attempts)
            result=json.loads((folder/"result.json").read_text(encoding="utf-8"))
            result.update(completed_candidates=1,failed_candidates=1);write_json(folder/"result.json",result)
        elif mutation in {"extra_file","nested_manifest"}:
            (target/("foreign.txt" if mutation=="extra_file" else "manifest.json")).write_text("{}",encoding="utf-8")
        elif mutation=="attempt_order":
            attempts=json.loads((folder/"attempts.json").read_text(encoding="utf-8"))
            write_json(folder/"attempts.json",attempts[::-1])
        elif mutation in {"false_fit_count","bool_counter","result_env","result_engine"}:
            result=json.loads((folder/"result.json").read_text(encoding="utf-8"))
            if mutation=="false_fit_count":result["model_fits"]=1
            if mutation=="bool_counter":result["completed_candidates"]=True
            if mutation=="result_env":result["environment"]={}
            if mutation=="result_engine":result["engine_hash"]="f"*64
            write_json(folder/"result.json",result)
        manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        manifest["artifacts"]={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob("*") if p.is_file() and p!=folder/"manifest.json"}
        if mutation=="manifest_owner":manifest["experiment_id"]="exp-ffffffffffff"
        write_json(folder/"manifest.json",manifest)
        with pytest.raises((ValueError,KeyError)):audit_output(ops,ex)
    finally:
        for p in folder.rglob("*"):
            if p.is_file() and p.relative_to(folder) not in originals:p.unlink()
        for name,value in originals.items():(folder/name).write_bytes(value)


def test_failed_numerical_budget_retained_and_cannot_restart(tmp_path):
    ops,session,p,_,_=setup(tmp_path,max_experiments=1)
    # Referenced leaves fit; the deeper formula's preflight buffers do not.
    p["spec"]["budget"]["max_buffer_bytes"]=200
    ex=ops.register(session,p);final=ops.execute(session,ex["id"])
    assert final["status"]=="failed" and "contains failed candidates" in final["error"]
    folder=ops.root/"worker_jobs"/ex["id"]/"alpha_result"
    attempts=json.loads((folder/"attempts.json").read_text(encoding="utf-8"))
    assert len(attempts)==1 and attempts[0]["status"]=="failed"
    assert "buffer budget" in attempts[0]["error"]
    assert not list((ops.root/".alpha_slots").glob("slot*.json"))
    with pytest.raises(ValueError,match="already started"):ops.execute(session,ex["id"])
    p["spec"]["candidates"][0]["candidate_id"]="second_try"
    with pytest.raises(ValueError,match="预算"):ops.register(session,p)


def test_resource_denial_leaves_experiment_planned(tmp_path,monkeypatch):
    ops,session,p,_,_=setup(tmp_path);ex=ops.register(session,p)
    monkeypatch.setattr(psutil,"virtual_memory",lambda:SimpleNamespace(available=1))
    with pytest.raises(RuntimeError,match="Insufficient"):ops.execute(session,ex["id"])
    assert ops.store.experiment(ex["id"])["status"]=="planned"
    assert not (ops.root/"worker_jobs"/ex["id"]/"launch.json").exists()


def test_shared_two_slots_no_refund_and_release_exact_owner(tmp_path,monkeypatch):
    monkeypatch.setattr(psutil,"virtual_memory",lambda:SimpleNamespace(available=1<<40))
    resources=WorkerResources(max_process_rss_bytes=134_217_728,min_available_memory_bytes=268_435_456)
    with reserve(tmp_path,"exp-111111111111",resources),reserve(tmp_path,"exp-222222222222",resources):
        with pytest.raises(RuntimeError,match="Both"):
            with reserve(tmp_path,"exp-333333333333",resources):pass
        assert len(list((tmp_path/".alpha_slots").glob("slot*.json")))==2
    assert not list((tmp_path/".alpha_slots").glob("slot*.json"))


def test_no_release_live_worker_or_different_launch(tmp_path,monkeypatch):
    monkeypatch.setattr(psutil,"virtual_memory",lambda:SimpleNamespace(available=1<<40))
    with reserve(tmp_path,"exp-111111111111",WorkerResources()) as reservation:
        created=reservation.bind(SimpleNamespace(pid=os.getpid()))
        launch=dict(pid=os.getpid(),created=created)
        assert not release_terminal_reservation(tmp_path,"exp-111111111111",launch)
        with pytest.raises(ValueError,match="differs"):
            release_terminal_reservation(tmp_path,"exp-111111111111",dict(launch,created=created-1))
        with pytest.raises(ValueError,match="still alive"):reservation.release()
        # Isolated fixture: model a proven exit, without killing this pytest process.
        monkeypatch.setattr("src.researchops.resources.process_alive",lambda pid,created:False)


def test_no_pid_reuse_confusion():
    created=psutil.Process().create_time()
    assert process_alive(os.getpid(),created)
    assert not process_alive(os.getpid(),created-1)
    with pytest.raises(ValueError):process_alive(True,created)


def test_terminal_recovery_does_not_restart_and_collects_missing_observation(completed):
    ops,_,ex,job=completed
    stats=(job/"execution_stats.json").read_bytes()
    try:
        (job/"execution_stats.json").unlink()
        launch=json.loads((job/"launch.json").read_text(encoding="utf-8"))
        created=psutil.Process().create_time()
        write_json(ops.root/".alpha_slots/slot0.json",dict(
            schema="alpha-resource-reservation-v1",experiment_id=ex["id"],token="b"*32,
            parent_pid=os.getpid(),parent_created=created,child_pid=launch["pid"],
            child_created=launch["created"],child_chain=launch["interpreter_chain"],launching=False,
            max_process_rss_bytes=ex["proposal"]["resources"]["max_process_rss_bytes"]))
        with ops.store.connection(True) as db:
            db.execute("UPDATE experiments SET status='running',result=NULL WHERE id=?",(ex["id"],))
        final=ops.recover(ex["id"])
        assert final["status"]=="completed" and verify_completed(ops,final)["verified"]
        observation=json.loads((job/"execution_stats.json").read_text(encoding="utf-8"))
        assert not observation["parent_observed_to_exit"] and observation["peak_sampled_rss_bytes"] is None
        assert len(ops.store.get(ex["task_id"])["experiments"])==1
    finally:
        (job/"execution_stats.json").write_bytes(stats)
        with ops.store.connection(True) as db:
            db.execute("UPDATE experiments SET status='completed',result=? WHERE id=?",(json.dumps(ex["result"]),ex["id"]))


def test_alive_or_missing_receipt_never_collected(tmp_path):
    ops,session,p,_,_=setup(tmp_path);ex=ops.register(session,p);ops.store.start(session,ex["id"])
    job=ops.root/"worker_jobs"/ex["id"];created=psutil.Process().create_time()
    with ops.store.connection(True) as db:db.execute("UPDATE experiments SET pid=? WHERE id=?",(os.getpid(),ex["id"]))
    launch=dict(schema="alpha-worker-launch-v1",experiment_id=ex["id"],pid=os.getpid(),created=created,interpreter_chain=[],
                input_hash=digest(job/"input.json"),resources=ex["proposal"]["resources"])
    write_json(job/"launch.json",launch)
    write_json(job/"receipt.json",dict(status="failed",kind="alpha_batch",experiment_id=ex["id"],error="not terminal while alive"))
    alive=ops.recover(ex["id"]);assert alive["status"]=="running" and "still alive" in alive["recovery_note"]
    (job/"receipt.json").unlink();write_json(job/"launch.json",dict(launch,created=created-1))
    lost=ops.recover(ex["id"]);assert lost["status"]=="running" and "no terminal receipt" in lost["recovery_note"]


def test_owned_running_cancellation_is_request_and_not_budget_refund(tmp_path):
    ops,session,p,_,_=setup(tmp_path);ex=ops.register(session,p);ops.store.start(session,ex["id"])
    requested=ops.cancel(session,ex["id"],"User requested bounded stop")
    assert requested["status"]=="running" and requested["cancellation_requested"]
    job=ops.root/"worker_jobs"/ex["id"]
    assert cancel_reason(job)=="User requested bounded stop"
    with pytest.raises(ValueError):ops.cancel(dict(session,token="foreign"),ex["id"],"foreign stop")
    doc=json.loads((job/"cancel_request.json").read_text(encoding="utf-8"))
    write_json(job/"cancel_request.json",dict(doc,input_hash="f"*64))
    with pytest.raises(ValueError,match="identity"):cancel_reason(job)


@pytest.mark.parametrize("stop",["timeout","rss","available","cancel"])
def test_child_self_watchdog_stops_with_retained_failed_receipt(tmp_path,monkeypatch,stop):
    job=tmp_path/"exp-111111111111";job.mkdir();write_json(job/"input.json",{})
    cfg=dict(resources=WorkerResources().to_dict(),timeout_seconds=60)
    if stop=="timeout":cfg["timeout_seconds"]=-1  # isolate elapsed-time branch, no registered job
    if stop=="rss":monkeypatch.setattr(psutil,"Process",lambda:SimpleNamespace(children=lambda **k:[],memory_info=lambda:SimpleNamespace(rss=1<<50)))
    monkeypatch.setattr(psutil,"virtual_memory",lambda:SimpleNamespace(available=1 if stop=="available" else 1<<40))
    if stop=="cancel":
        write_json(job/"cancel_request.json",dict(schema="alpha-cancel-request-v1",experiment_id=job.name,
                   input_hash=digest(job/"input.json"),worker="test",reason="stop"))
    def stopped(code):raise SystemExit(code)
    monkeypatch.setattr(os,"_exit",stopped)
    with pytest.raises(SystemExit):_watchdog(job,cfg,SimpleNamespace(wait=lambda seconds:False,is_set=lambda:False))
    receipt=json.loads((job/"receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"]=="failed" and receipt["experiment_id"]==job.name


@pytest.mark.parametrize("mutation",["threads","schema","rss","output","unknown","boolean"])
def test_resource_limits_are_strict(mutation):
    value=WorkerResources().to_dict()
    if mutation=="threads":value["library_threads"]=3
    if mutation=="schema":value["schema"]="unbounded"
    if mutation=="rss":value["max_process_rss_bytes"]=0
    if mutation=="output":value["max_output_bytes"]=0
    if mutation=="unknown":value["unbounded"]=True
    if mutation=="boolean":value["library_threads"]=True
    with pytest.raises(ValueError):WorkerResources.from_dict(value)


def test_partial_failure_is_retained_as_failed_whole_batch(tmp_path):
    from src.alpharesearch.dsl import Expr,compile_expression
    ops,session,p,registry,_=setup(tmp_path)
    p["spec"]["budget"]["max_buffer_bytes"]=680
    good=copy.deepcopy(p["spec"]["candidates"][0])
    good.update(candidate_id="ratio_without_rank",expression=compile_expression(
        Expr.call("div",Expr.ref("daily.close"),Expr.ref("daily.pre_close")),registry).to_dict())
    p["spec"]["candidates"].append(good)
    ex=ops.register(session,p);final=ops.execute(session,ex["id"])
    assert final["status"]=="failed" and "entire batch failed" in final["error"]
    folder=ops.root/"worker_jobs"/ex["id"]/"alpha_result"
    attempts=json.loads((folder/"attempts.json").read_text(encoding="utf-8"))
    assert [a["status"] for a in attempts]==["failed","completed"]
    assert (folder/"ratio_without_rank/values.parquet").is_file()
    with pytest.raises(ValueError,match="every registered"):audit_output(ops,final)


@pytest.mark.parametrize("write_failure",["receipt","log"])
def test_watchdog_exit_does_not_depend_on_record_or_log_write(tmp_path,monkeypatch,write_failure):
    from src.researchops import alpha_worker
    job=tmp_path/"exp-111111111111";job.mkdir()
    events=[]
    def disk_full(*args,**kwargs):
        events.append("record_failure");raise OSError("isolated disk full")
    def log_full(*args,**kwargs):
        events.append("log_failure");raise OSError("isolated log full")
    def stopped(code):
        events.append("exit");raise SystemExit(code)
    monkeypatch.setattr(alpha_worker,"write_json",disk_full)
    if write_failure=="log":monkeypatch.setattr("builtins.print",log_full)
    monkeypatch.setattr(os,"_exit",stopped)
    cfg=dict(resources=WorkerResources().to_dict(),timeout_seconds=-1)
    with pytest.raises(SystemExit):
        _watchdog(job,cfg,SimpleNamespace(wait=lambda seconds:False,is_set=lambda:False))
    assert events[-1]=="exit" and ("record_failure" in events if write_failure=="receipt" else "log_failure" in events)


def test_arrow_limits_are_set_before_first_input_read(tmp_path,monkeypatch):
    from src.researchops import alpha_worker
    import pyarrow as pa
    job=tmp_path/"exp-111111111111";job.mkdir()
    old=(pa.cpu_count(),pa.io_thread_count());seen=[]
    monkeypatch.setattr(alpha_worker,"__file__",str(job/"code/src/researchops/alpha_worker.py"))
    monkeypatch.setattr(alpha_worker,"verify_job",lambda job:dict(
        resources=WorkerResources().to_dict(),project="unused-isolated-fixture"))
    def inspect(*args,**kwargs):
        seen.append((pa.cpu_count(),pa.io_thread_count()))
        raise RuntimeError("stop before any source read or formula")
    monkeypatch.setattr(alpha_worker,"inspect_inputs",inspect)
    try:
        pa.set_cpu_count(2);pa.set_io_thread_count(8)
        with pytest.raises(RuntimeError,match="before any source"):
            alpha_worker.compute(job)
        assert seen==[(2,1)]
    finally:
        pa.set_cpu_count(old[0]);pa.set_io_thread_count(old[1])


def test_forged_creation_and_omitted_chain_cannot_commit_live_reservation(tmp_path):
    ops,session,p,_,_=setup(tmp_path);ex=ops.register(session,p);ops.store.start(session,ex["id"])
    job=ops.root/"worker_jobs"/ex["id"];created=psutil.Process().create_time()
    with ops.store.connection(True) as db:db.execute("UPDATE experiments SET pid=? WHERE id=?",(os.getpid(),ex["id"]))
    pool=ops.root/".alpha_slots";pool.mkdir()
    owner=dict(schema="alpha-resource-reservation-v1",experiment_id=ex["id"],token="a"*32,
               parent_pid=os.getpid(),parent_created=created,child_pid=os.getpid(),
               child_created=created,child_chain=[dict(pid=os.getpid(),created=created,role="interpreter")],
               launching=False,max_process_rss_bytes=ex["proposal"]["resources"]["max_process_rss_bytes"])
    write_json(pool/"slot0.json",owner)
    launch=dict(schema="alpha-worker-launch-v1",experiment_id=ex["id"],pid=os.getpid(),
                created=created-1,interpreter_chain=[],input_hash=digest(job/"input.json"),
                resources=ex["proposal"]["resources"])
    write_json(job/"launch.json",launch)
    write_json(job/"receipt.json",dict(status="completed",kind="alpha_batch",
        experiment_id=ex["id"],result_manifest_hash="f"*64))
    with pytest.raises(ValueError,match="resource owner"):ops.recover(ex["id"])
    assert ops.store.experiment(ex["id"])["status"]=="running"
    assert (pool/"slot0.json").is_file()


@pytest.mark.parametrize("first",["parent","collector"])
def test_parent_observation_and_collector_share_first_writer_guard(completed,monkeypatch,first):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from src.researchops import alpha_execution, alpha_output_audit
    ops,_,ex,job=completed
    original=(job/"execution_stats.json").read_bytes()
    launch=json.loads((job/"launch.json").read_text(encoding="utf-8"))
    normal=json.loads(original)
    entered=threading.Event();writer_started=threading.Event()
    real_audit=alpha_output_audit.audit_output
    def controlled_audit(*args):
        if first=="collector":
            entered.set()
            assert writer_started.wait(1)
        return real_audit(*args)
    def parent():
        if first=="collector":
            assert entered.wait(1)
            writer_started.set()
        return alpha_execution._record_observation(job,normal)
    try:
        (job/"execution_stats.json").unlink()
        created=psutil.Process().create_time()
        write_json(ops.root/".alpha_slots/slot0.json",dict(
            schema="alpha-resource-reservation-v1",experiment_id=ex["id"],token="c"*32,
            parent_pid=os.getpid(),parent_created=created,child_pid=launch["pid"],
            child_created=launch["created"],child_chain=launch["interpreter_chain"],launching=False,
            max_process_rss_bytes=ex["proposal"]["resources"]["max_process_rss_bytes"]))
        with ops.store.connection(True) as db:
            db.execute("UPDATE experiments SET status='running',result=NULL WHERE id=?",(ex["id"],))
        monkeypatch.setattr(alpha_output_audit,"audit_output",controlled_audit)
        if first=="parent":
            parent()
            final=ops.recover(ex["id"])
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                future_parent=pool.submit(parent)
                future_collector=pool.submit(ops.recover,ex["id"])
                final=future_collector.result(timeout=10)
                future_parent.result(timeout=10)
        saved=json.loads((job/"execution_stats.json").read_text(encoding="utf-8"))
        assert saved["parent_observed_to_exit"]==(first=="parent")
        assert final["status"]=="completed" and verify_completed(ops,final)["verified"]
        assert final["result"]["observation_hash"]==digest(job/"execution_stats.json")
        assert not list((ops.root/".alpha_slots").glob("slot*.json"))
    finally:
        (job/"execution_stats.json").write_bytes(original)
        with ops.store.connection(True) as db:
            db.execute("UPDATE experiments SET status='completed',result=? WHERE id=?",(json.dumps(ex["result"]),ex["id"]))


@pytest.mark.parametrize("busy_at,collected",[("observation",False),("collection",False),("collection",True)])
def test_execute_collection_busy_preserves_running_receipt_and_owner(tmp_path,monkeypatch,busy_at,collected):
    from src.researchops import alpha_execution
    from src.researchops.resources import GuardBusy
    ops,session,p,_,_=setup(tmp_path);ex=ops.register(session,p)
    real_record=alpha_execution._record_observation;real_recover=alpha_execution.recover
    def busy(*args,**kwargs):
        if collected:real_recover(ops,ops.store.experiment(ex["id"]))
        raise GuardBusy("isolated collection busy")
    if busy_at=="observation":
        monkeypatch.setattr(alpha_execution,"_record_observation",busy)
    else:
        monkeypatch.setattr(alpha_execution,"recover",busy)
    final=ops.execute(session,ex["id"])
    job=ops.root/"worker_jobs"/ex["id"]
    if collected:
        assert final["status"]=="completed" and "recovery_note" not in final
        assert verify_completed(ops,final)["verified"]
    else:
        assert final["status"]=="running" and "guard busy" in final["recovery_note"]
    assert ops.store.experiment(ex["id"])["error"] is None
    receipt=(job/"receipt.json").read_bytes()
    assert json.loads(receipt)["status"]=="completed"
    assert len(list((ops.root/".alpha_slots").glob("slot*.json")))==(0 if collected else 1)
    launch=(job/"launch.json").read_bytes()
    monkeypatch.setattr(alpha_execution,"_record_observation",real_record)
    monkeypatch.setattr(alpha_execution,"recover",real_recover)
    recovered=ops.recover(ex["id"])
    assert recovered["status"]=="completed" and verify_completed(ops,recovered)["verified"]
    assert (job/"launch.json").read_bytes()==launch and (job/"receipt.json").read_bytes()==receipt
    assert len(ops.store.get(ex["task_id"])["experiments"])==1
    assert not list((ops.root/".alpha_slots").glob("slot*.json"))


def test_startup_guard_busy_is_real_failure_not_pending_collection(tmp_path,monkeypatch):
    from src.researchops.resources import GuardBusy,Reservation
    ops,session,p,_,_=setup(tmp_path);ex=ops.register(session,p)
    def busy(self):raise GuardBusy("isolated prelaunch pool busy")
    monkeypatch.setattr(Reservation,"launching",busy)
    with pytest.raises(GuardBusy,match="prelaunch"):ops.execute(session,ex["id"])
    final=ops.store.experiment(ex["id"]);job=ops.root/"worker_jobs"/ex["id"]
    assert final["status"]=="failed" and "prelaunch" in final["error"]
    assert not (job/"launch.json").exists()
    assert json.loads((job/"receipt.json").read_text(encoding="utf-8"))["status"]=="failed"
    assert not list((ops.root/".alpha_slots").glob("slot*.json"))
    assert len(ops.store.get(ex["task_id"])["experiments"])==1


@pytest.mark.parametrize("outcome",["completed","failed"])
def test_terminal_pool_cleanup_busy_retries_through_public_recovery(tmp_path,monkeypatch,outcome):
    from src.researchops import alpha_execution
    from src.researchops.resources import GuardBusy
    ops,session,p,_,_=setup(tmp_path)
    if outcome=="failed":p["spec"]["budget"]["max_buffer_bytes"]=200
    ex=ops.register(session,p);real_release=alpha_execution.release_terminal_reservation;calls=[]
    def once_busy(*args):
        calls.append("release")
        if len(calls)==1:raise GuardBusy("isolated cleanup pool busy")
        return real_release(*args)
    monkeypatch.setattr(alpha_execution,"release_terminal_reservation",once_busy)
    final=ops.execute(session,ex["id"]);job=ops.root/"worker_jobs"/ex["id"]
    assert final["status"]==outcome
    assert len(list((ops.root/".alpha_slots").glob("slot*.json")))==1
    before=ops.store.experiment(ex["id"])
    identities={name:(job/name).read_bytes() for name in ["launch.json","receipt.json","execution_stats.json"]}
    recovered=ops.recover(ex["id"])
    assert recovered==before
    assert calls==["release","release"] and not list((ops.root/".alpha_slots").glob("slot*.json"))
    assert all((job/name).read_bytes()==value for name,value in identities.items())
    assert len(ops.store.get(ex["task_id"])["experiments"])==1
    if outcome=="completed":assert verify_completed(ops,recovered)["verified"]


@pytest.mark.parametrize("record_failure",["observation_busy","receipt_disk","database_commit"])
def test_real_supervision_failure_survives_metadata_fault_and_keeps_recovery_owner(tmp_path,monkeypatch,record_failure):
    from src.researchops import alpha_execution
    from src.researchops.resources import GuardBusy
    ops,session,p,_,_=setup(tmp_path);ex=ops.register(session,p)
    real_sample=alpha_execution._sample;real_record=alpha_execution._record_observation
    real_receipt=alpha_execution._failure_receipt;real_finish=ops.store.finish
    def original_failure(*args):raise MemoryError("isolated original supervision failure")
    def busy(*args,**kwargs):raise GuardBusy("isolated failure-observation busy")
    def disk_error(*args,**kwargs):raise OSError("isolated failed receipt disk error")
    def db_error(*args,**kwargs):raise OSError("isolated terminal database write error")
    monkeypatch.setattr(alpha_execution,"_sample",original_failure)
    if record_failure=="observation_busy":monkeypatch.setattr(alpha_execution,"_record_observation",busy)
    if record_failure=="receipt_disk":monkeypatch.setattr(alpha_execution,"_failure_receipt",disk_error)
    if record_failure=="database_commit":monkeypatch.setattr(ops.store,"finish",db_error)
    with pytest.raises(MemoryError,match="original supervision"):ops.execute(session,ex["id"])
    current=ops.store.experiment(ex["id"]);job=ops.root/"worker_jobs"/ex["id"]
    if record_failure=="database_commit":
        assert current["status"]=="running"
        assert len(list((ops.root/".alpha_slots").glob("slot*.json")))==1
        assert json.loads((job/"receipt.json").read_text(encoding="utf-8"))["status"]=="failed"
    else:
        assert current["status"]=="failed" and current["error"]=="isolated original supervision failure"
        assert not list((ops.root/".alpha_slots").glob("slot*.json"))
    monkeypatch.setattr(alpha_execution,"_sample",real_sample)
    monkeypatch.setattr(alpha_execution,"_record_observation",real_record)
    monkeypatch.setattr(alpha_execution,"_failure_receipt",real_receipt)
    monkeypatch.setattr(ops.store,"finish",real_finish)
    final=ops.recover(ex["id"])
    assert final["status"]=="failed" and final["error"]=="isolated original supervision failure"
    assert not list((ops.root/".alpha_slots").glob("slot*.json"))
    assert len(ops.store.get(ex["task_id"])["experiments"])==1


@pytest.mark.parametrize("child_outcome,existing_observation",[
    ("completed",False),("completed",True),("failed",False),("failed",True),("missing",False)])
def test_parent_failure_survives_db_outage_after_child_terminal_receipt(
        tmp_path,monkeypatch,child_outcome,existing_observation):
    from src.researchops import alpha_execution
    ops,session,p,_,_=setup(tmp_path)
    if child_outcome=="failed":p["spec"]["budget"]["max_buffer_bytes"]=200
    ex=ops.register(session,p);job=ops.root/"worker_jobs"/ex["id"]
    real_record=alpha_execution._record_observation;real_finish=ops.store.finish
    preserved={}
    def failed_after_exit(job,stats):
        # Actual frozen numerical child already exited and wrote its receipt.
        assert (job/"receipt.json").is_file()
        if child_outcome=="missing":(job/"receipt.json").unlink()
        if existing_observation:real_record(job,stats)
        for name in ("launch.json","receipt.json","execution_stats.json"):
            if (job/name).exists():preserved[name]=(job/name).read_bytes()
        monkeypatch.setattr(ops.store,"finish",lambda *a,**k:(_ for _ in ()).throw(
            OSError("isolated database outage after receipt")))
        raise MemoryError("isolated original parent failure after child exit")
    monkeypatch.setattr(alpha_execution,"_record_observation",failed_after_exit)
    with pytest.raises(MemoryError,match="original parent failure"):ops.execute(session,ex["id"])
    assert ops.store.experiment(ex["id"])["status"]=="running"
    assert len(list((ops.root/".alpha_slots").glob("slot*.json")))==1
    marker=(job/"supervision_failure.json").read_bytes()
    failure=json.loads(marker)
    assert failure["error"]=="isolated original parent failure after child exit"
    assert failure["input_hash"]==digest(job/"input.json")
    assert failure["launch_hash"]==digest(job/"launch.json")
    monkeypatch.setattr(ops.store,"finish",real_finish)
    monkeypatch.setattr(alpha_execution,"_record_observation",real_record)
    final=ops.recover(ex["id"])
    assert final["status"]=="failed" and final["error"]==failure["error"]
    assert ops.recover(ex["id"])==final
    assert all((job/name).read_bytes()==value for name,value in preserved.items())
    assert (job/"supervision_failure.json").read_bytes()==marker
    assert not list((ops.root/".alpha_slots").glob("slot*.json"))
    assert len(ops.store.get(ex["task_id"])["experiments"])==1


@pytest.mark.parametrize("field",["input_hash","launch_hash","experiment_id","error","schema","extra"])
def test_recovery_refuses_forged_supervision_failure_before_terminal_commit(completed,field):
    from src.researchops import alpha_execution
    ops,_,ex,job=completed
    launch=json.loads((job/"launch.json").read_text(encoding="utf-8"))
    saved_stats=(job/"execution_stats.json").read_bytes()
    marker=job/"supervision_failure.json"
    try:
        with ops.store.connection(True) as db:
            db.execute("UPDATE experiments SET status='running',result=NULL WHERE id=?",(ex["id"],))
        write_json(ops.root/".alpha_slots/slot0.json",dict(
            schema="alpha-resource-reservation-v1",experiment_id=ex["id"],token="d"*32,
            parent_pid=os.getpid(),parent_created=psutil.Process().create_time(),
            child_pid=launch["pid"],child_created=launch["created"],child_chain=launch["interpreter_chain"],
            launching=False,max_process_rss_bytes=ex["proposal"]["resources"]["max_process_rss_bytes"]))
        alpha_execution._record_supervision_failure(job,ex["id"],"original retained cause")
        doc=json.loads(marker.read_text(encoding="utf-8"))
        doc[field]="" if field=="error" else "forged"
        write_json(marker,doc)
        with pytest.raises(ValueError,match="failure identity"):ops.recover(ex["id"])
        assert ops.store.experiment(ex["id"])["status"]=="running"
        assert (ops.root/".alpha_slots/slot0.json").exists()
        marker.unlink()
        final=ops.recover(ex["id"])
        assert final["status"]=="completed"
    finally:
        if marker.exists():marker.unlink()
        (job/"execution_stats.json").write_bytes(saved_stats)
        with ops.store.connection(True) as db:
            db.execute("UPDATE experiments SET status='completed',result=? WHERE id=?",(json.dumps(ex["result"]),ex["id"]))


def test_late_parent_failure_preserves_already_committed_completion(completed):
    from src.researchops import alpha_execution
    ops,_,ex,job=completed;marker=job/"supervision_failure.json"
    before=ops.store.experiment(ex["id"])
    try:
        original=alpha_execution._record_supervision_failure(job,ex["id"],"late parent record")
        assert alpha_execution._record_supervision_failure(job,ex["id"],"another cause")==original
        assert ops.recover(ex["id"])==before
        assert verify_completed(ops,before)["verified"]
    finally:
        marker.unlink()


@pytest.mark.parametrize("outcome",["completed","child_failed","validation_failed"])
def test_execute_terminal_db_fault_is_pending_collection_not_new_supervision_failure(
        tmp_path,monkeypatch,outcome):
    from src.researchops import alpha_output_audit
    ops,session,p,_,_=setup(tmp_path)
    if outcome=="child_failed":p["spec"]["budget"]["max_buffer_bytes"]=200
    ex=ops.register(session,p);real_finish=ops.store.finish;calls=[]
    def fail_once(eid,status,**fields):
        calls.append((status,fields.get("error")))
        if len(calls)==1:raise OSError("isolated collection-only database outage")
        return real_finish(eid,status,**fields)
    monkeypatch.setattr(ops.store,"finish",fail_once)
    if outcome=="validation_failed":
        def invalid_output(*args):raise ValueError("isolated original output audit failure")
        monkeypatch.setattr(alpha_output_audit,"audit_output",invalid_output)
    pending=ops.execute(session,ex["id"]);job=ops.root/"worker_jobs"/ex["id"]
    assert pending["status"]=="running" and "database commit unavailable" in pending["recovery_note"]
    assert len(calls)==1 and ops.store.experiment(ex["id"])["error"] is None
    assert len(list((ops.root/".alpha_slots").glob("slot*.json")))==1
    assert (job/"supervision_failure.json").exists()==(outcome=="validation_failed")
    saved={name:(job/name).read_bytes() for name in ("receipt.json","launch.json","execution_stats.json")}
    final=ops.recover(ex["id"])
    assert final["status"]==("completed" if outcome=="completed" else "failed")
    if outcome=="completed":assert verify_completed(ops,final)["verified"]
    if outcome=="child_failed":assert "contains failed candidates" in final["error"]
    if outcome=="validation_failed":assert final["error"]=="isolated original output audit failure"
    assert all((job/name).read_bytes()==value for name,value in saved.items())
    assert calls[0]==calls[1]
    assert not list((ops.root/".alpha_slots").glob("slot*.json"))
    assert len(ops.store.get(ex["task_id"])["experiments"])==1


@pytest.mark.parametrize("outcome",["completed","child_failed","supervision_failed"])
def test_public_recovery_terminal_db_fault_keeps_original_cause_and_slot(completed,monkeypatch,outcome):
    from src.researchops import alpha_execution
    ops,_,ex,job=completed;marker=job/"supervision_failure.json"
    original_receipt=(job/"receipt.json").read_bytes()
    launch=json.loads((job/"launch.json").read_text(encoding="utf-8"))
    before=ops.store.experiment(ex["id"]);real_finish=ops.store.finish;calls=[]
    try:
        with ops.store.connection(True) as db:
            db.execute("UPDATE experiments SET status='running',result=NULL WHERE id=?",(ex["id"],))
        write_json(ops.root/".alpha_slots/slot0.json",dict(
            schema="alpha-resource-reservation-v1",experiment_id=ex["id"],token="e"*32,
            parent_pid=os.getpid(),parent_created=psutil.Process().create_time(),
            child_pid=launch["pid"],child_created=launch["created"],child_chain=launch["interpreter_chain"],
            launching=False,max_process_rss_bytes=ex["proposal"]["resources"]["max_process_rss_bytes"]))
        if outcome=="supervision_failed":
            alpha_execution._record_supervision_failure(job,ex["id"],"original sampled RSS failure")
        if outcome=="child_failed":
            write_json(job/"receipt.json",dict(status="failed",kind="alpha_batch",
                experiment_id=ex["id"],error="original child buffer failure"))
        saved={name:p.read_bytes() for name in ("launch.json","receipt.json","execution_stats.json","supervision_failure.json")
               if (p:=job/name).exists()}
        def fail_once(eid,status,**fields):
            calls.append((status,fields.get("error")))
            if len(calls)==1:raise OSError("isolated recovery database outage")
            return real_finish(eid,status,**fields)
        monkeypatch.setattr(ops.store,"finish",fail_once)
        with pytest.raises(alpha_execution.CollectionCommitPending,match="commit pending"):
            ops.recover(ex["id"])
        assert len(calls)==1
        assert ops.store.experiment(ex["id"])["status"]=="running"
        assert (ops.root/".alpha_slots/slot0.json").exists()
        assert all((job/name).read_bytes()==value for name,value in saved.items())
        final=ops.recover(ex["id"])
        assert calls[0]==calls[1]
        assert final["status"]==("completed" if outcome=="completed" else "failed")
        if outcome=="supervision_failed":assert final["error"]=="original sampled RSS failure"
        if outcome=="child_failed":assert final["error"]=="original child buffer failure"
        assert ops.recover(ex["id"])==final
        assert not list((ops.root/".alpha_slots").glob("slot*.json"))
        assert all((job/name).read_bytes()==value for name,value in saved.items())
    finally:
        monkeypatch.setattr(ops.store,"finish",real_finish)
        (job/"receipt.json").write_bytes(original_receipt)
        if marker.exists():marker.unlink()
        with ops.store.connection(True) as db:
            db.execute("UPDATE experiments SET status='completed',result=?,error=NULL,updated=? WHERE id=?",
                       (json.dumps(before["result"]),before["updated"],ex["id"]))


def test_execute_collection_db_unreadable_preserves_pending_identity(tmp_path,monkeypatch):
    from src.researchops import alpha_execution
    ops,session,p,_,_=setup(tmp_path);ex=ops.register(session,p)
    real_finish=ops.store.finish;real_read=ops.store.experiment;unavailable=[False];calls=[]
    def offline_finish(*args,**kwargs):
        calls.append("commit");unavailable[0]=True;raise OSError("isolated collection database offline")
    def offline_read(*args,**kwargs):
        if unavailable[0]:raise OSError("isolated database read also offline")
        return real_read(*args,**kwargs)
    monkeypatch.setattr(ops.store,"finish",offline_finish)
    monkeypatch.setattr(ops.store,"experiment",offline_read)
    with pytest.raises(alpha_execution.CollectionCommitPending):ops.execute(session,ex["id"])
    job=ops.root/"worker_jobs"/ex["id"]
    assert calls==["commit"] and not (job/"supervision_failure.json").exists()
    assert len(list((ops.root/".alpha_slots").glob("slot*.json")))==1
    monkeypatch.setattr(ops.store,"finish",real_finish)
    monkeypatch.setattr(ops.store,"experiment",real_read)
    final=ops.recover(ex["id"])
    assert final["status"]=="completed" and verify_completed(ops,final)["verified"]
    assert not list((ops.root/".alpha_slots").glob("slot*.json"))
    assert len(ops.store.get(ex["task_id"])["experiments"])==1


@pytest.mark.parametrize("fault_at",["before_first_collection","after_collection_guard_busy"])
def test_execute_all_postexit_db_reads_preserve_pending_collection(tmp_path,monkeypatch,fault_at):
    from src.researchops import alpha_execution
    from src.researchops.resources import GuardBusy
    ops,session,p,_,_=setup(tmp_path);ex=ops.register(session,p)
    real_read=ops.store.experiment;real_record=alpha_execution._record_observation;real_finish=ops.store.finish
    reads=[];finishes=[]
    def offline_read(*args,**kwargs):
        reads.append("offline");raise OSError("isolated postexit database read unavailable")
    def recorded_then_offline(job,stats):
        if fault_at=="before_first_collection":real_record(job,stats)
        monkeypatch.setattr(ops.store,"experiment",offline_read)
        if fault_at=="after_collection_guard_busy":raise GuardBusy("isolated observation busy before DBread")
    def finish(*args,**kwargs):
        finishes.append("commit");return real_finish(*args,**kwargs)
    monkeypatch.setattr(alpha_execution,"_record_observation",recorded_then_offline)
    monkeypatch.setattr(ops.store,"finish",finish)
    with pytest.raises(alpha_execution.CollectionCommitPending,match="database read pending"):
        ops.execute(session,ex["id"])
    job=ops.root/"worker_jobs"/ex["id"]
    assert reads and finishes==[]
    assert not (job/"supervision_failure.json").exists()
    assert len(list((ops.root/".alpha_slots").glob("slot*.json")))==1
    saved={name:(job/name).read_bytes() for name in ("receipt.json","launch.json")}
    monkeypatch.setattr(ops.store,"experiment",real_read)
    monkeypatch.setattr(alpha_execution,"_record_observation",real_record)
    assert ops.store.experiment(ex["id"])["status"]=="running"
    final=ops.recover(ex["id"])
    assert final["status"]=="completed" and verify_completed(ops,final)["verified"]
    assert all((job/name).read_bytes()==value for name,value in saved.items())
    assert not list((ops.root/".alpha_slots").glob("slot*.json"))
    assert len(ops.store.get(ex["task_id"])["experiments"])==1


def test_public_collector_locked_db_read_failure_does_not_change_state_or_owner(completed,monkeypatch):
    from src.researchops import alpha_execution
    ops,_,ex,job=completed;before=ops.store.experiment(ex["id"]);real_read=ops.store.experiment
    launch=json.loads((job/"launch.json").read_text(encoding="utf-8"))
    saved={name:(job/name).read_bytes() for name in ("receipt.json","launch.json","execution_stats.json")}
    try:
        with ops.store.connection(True) as db:
            db.execute("UPDATE experiments SET status='running',result=NULL WHERE id=?",(ex["id"],))
        write_json(ops.root/".alpha_slots/slot0.json",dict(
            schema="alpha-resource-reservation-v1",experiment_id=ex["id"],token="f"*32,
            parent_pid=os.getpid(),parent_created=psutil.Process().create_time(),
            child_pid=launch["pid"],child_created=launch["created"],child_chain=launch["interpreter_chain"],
            launching=False,max_process_rss_bytes=ex["proposal"]["resources"]["max_process_rss_bytes"]))
        def offline(*args,**kwargs):raise OSError("isolated locked collector read unavailable")
        monkeypatch.setattr(ops.store,"experiment",offline)
        with pytest.raises(alpha_execution.CollectionCommitPending,match="database read pending"):
            alpha_execution.recover(ops,ex)
        assert (ops.root/".alpha_slots/slot0.json").exists()
        assert not (job/"supervision_failure.json").exists()
        assert all((job/name).read_bytes()==value for name,value in saved.items())
        monkeypatch.setattr(ops.store,"experiment",real_read)
        assert ops.store.experiment(ex["id"])["status"]=="running"
        final=ops.recover(ex["id"])
        assert final["status"]=="completed" and verify_completed(ops,final)["verified"]
        assert not list((ops.root/".alpha_slots").glob("slot*.json"))
    finally:
        monkeypatch.setattr(ops.store,"experiment",real_read)
        with ops.store.connection(True) as db:
            db.execute("UPDATE experiments SET status='completed',result=?,error=NULL,updated=? WHERE id=?",
                       (json.dumps(before["result"]),before["updated"],ex["id"]))
