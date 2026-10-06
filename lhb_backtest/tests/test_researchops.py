"""Task concurrency, stale-worker fencing and real frozen executor acceptance."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pandas as pd
import pytest
import yaml

from src.researchops.store import Store
from src.researchops.service import Research
from src.technical.contracts import ResearchSpec, Strategy, Experiment, Execution
from src.technical.data import prepare_snapshot
from src.technical.server import make_server
from tests.test_technical import bars

PROJECT = Path(__file__).resolve().parents[1]
TASK = dict(
    title="状态与反弹",
    question="状态是否解释分支收益差？",
    rationale="已有反弹窗口存在收益损失",
    success_criteria=["用共同对照解释状态条件收益"],
    stop_criteria=["收益差不稳定则停止"],
    max_experiments=2,
)
REPORT = dict(
    summary="证据及边界",
    findings=["实验记录见登记结果"],
    limitations=["回顾性"],
    next_steps=["交接后复核"],
)


def active(tmp_path):
    store = Store(tmp_path)
    task = store.create(TASK)
    return store, task["id"], store.claim(task["id"], "a")


def test_atomic_claim_across_database_connections(tmp_path):
    s = Store(tmp_path)
    t = s.create(TASK)

    def claim(i):
        try:
            return Store(tmp_path).claim(t["id"], str(i))
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(claim, range(8)))
    assert sum(v is not None for v in results) == 1
    token = next(v["token"] for v in results if v)
    assert token not in json.dumps(s.get(t["id"]))
    assert "token_hash" not in json.dumps(s.get(t["id"]))


def test_lease_expiry_and_handoff_fence_stale_worker(tmp_path, monkeypatch):
    s, tid, a = active(tmp_path)
    with s.connection(True) as db:
        db.execute("UPDATE tasks SET lease_until=0 WHERE id=?", (tid,))
    b = Store(tmp_path).claim(tid, "b")
    with pytest.raises(ValueError):
        s.checkpoint(a, "stale write")
    s.checkpoint(b, "已登记但未启动，下一位继续", True)
    with pytest.raises(ValueError):
        s.heartbeat(b)
    c = Store(tmp_path).claim(tid, "c")
    s.heartbeat(c)
    assert s.get(tid)["checkpoint"]["worker"] == "b"
    assert s.get(tid)["owner"] == "c"


def test_budget_dedup_cancel_and_review_transition(tmp_path):
    s, tid, a = active(tmp_path)
    first = s.register(a, {}, "one")
    assert s.register(a, {}, "one")["id"] == first["id"]
    s.start(a, first["id"])
    s.finish(first["id"], "completed", run_id="audit-fixture", result={"verified": True})
    with pytest.raises(ValueError):
        s.start(a, first["id"])
    second = s.register(a, {}, "two")
    with pytest.raises(ValueError):
        s.submit(a, REPORT)
    s.cancel(a, second["id"], "不再需要的对照，保留记录")
    with pytest.raises(ValueError):
        s.register(a, {}, "three")
    task = s.submit(a, REPORT)
    assert task["status"] == "review"
    with pytest.raises(ValueError):
        s.review(tid, "a", "stop", "self review")
    with pytest.raises(ValueError):
        s.review(tid, "reviewer", "continue", "missing next question")
    assert s.get(tid)["status"] == "review"
    out = s.review(tid, "reviewer", "continue", "下一项有具体依据", TASK)
    assert out["status"] == "completed" and len(out["children"]) == 1
    assert s.get(out["children"][0])["spec"]["parent_id"] == tid
    memory = Store(tmp_path).knowledge()
    assert memory[0]["task_id"] == tid
    assert memory[0]["review"]["decision"] == "continue"
    assert memory[0]["run_ids"] == ["audit-fixture"]


def test_failed_only_submission_labeled_and_deferred_reopens(tmp_path):
    s, tid, a = active(tmp_path)
    with pytest.raises(ValueError):
        s.submit(a, REPORT)
    e = s.register(a, {}, "one")
    s.start(a, e["id"])
    s.finish(e["id"], "failed", error="坏数据")
    assert s.submit(a, REPORT)["submission"]["evidence_kind"] == "execution_failure_only"
    s.review(tid, "reviewer", "defer", "等新数据")
    s.reopen(tid, "user", "数据已修复")
    assert s.get(tid)["status"] == "queued"
    with pytest.raises(ValueError):
        s.reopen(tid, "user", "不能重复")


@pytest.fixture
def research(tmp_path):
    source = tmp_path / "source"
    clean = source / "clean"
    export = source / "export"
    clean.mkdir(parents=True)
    (export / "metadata").mkdir(parents=True)
    (export / "exrights").mkdir()
    (source / "config").mkdir()
    (source / "config/config.yaml").write_text(
        yaml.safe_dump({"paths": {"clean_dir": "clean"}, "simtradedata": {"export_dir": "export"}})
    )
    cal = pd.bdate_range("2020-01-01", "2020-03-06").strftime("%Y-%m-%d").tolist()
    bars(cal).drop(columns=["avg_volume_20"]).to_parquet(clean / "daily_kline.parquet")
    pd.DataFrame({"date": cal}).to_parquet(export / "metadata/trade_days.parquet")
    pd.DataFrame(
        [
            dict(
                symbol="600001.SS",
                stock_name="test",
                listed_date="2000-01-01",
                de_listed_date="2900-01-01",
                security_type="1",
            )
        ]
    ).to_parquet(export / "metadata/stock_metadata.parquet")
    pd.DataFrame(
        [dict(date=d, status_type=k, symbols=[]) for d in cal for k in ["ST", "HALT"]]
    ).to_parquet(export / "metadata/stock_status.parquet")
    pd.DataFrame({"date": cal, "close": 10.0}).to_parquet(export / "metadata/benchmark.parquet")
    (export / ".source_state.json").write_text(
        json.dumps(
            dict(
                status_policy="dated-flags-v1",
                status_history_start=cal[0],
                status_history_end=cal[-1],
            )
        )
    )
    conf = ResearchSpec(
        Strategy(lookback=5, skip=1, top_k=1, min_avg_amount=0),
        Execution(initial_cash=10000, max_participation=0.05),
        Experiment("2020-01-01", "latest"),
    )
    project = tmp_path / "project"
    shutil.copytree(
        PROJECT / "src/technical",
        project / "src/technical",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (project / "tools").mkdir()
    shutil.copy2(PROJECT / "tools/verify_technical.py", project / "tools/verify_technical.py")
    root = tmp_path / "store"
    snap, sm = prepare_snapshot(source, root, conf, lambda m: None)
    r = Research(project, root)
    t = r.create(TASK)
    session = r.store.claim(t["id"], "worker-a")
    proposal = dict(
        hypothesis="代码冻结与跨进程交接不改变登记实验",
        expected_observation="两个费用账户均完成独立核查",
        falsification="区间或代码与登记不同即失败",
        snapshot_id=sm["snapshot_id"],
        spec=replace(conf, experiment=Experiment("2020-01-01", "2020-02-14")).to_dict(),
    )
    return r, session, proposal


def test_frozen_executor_handoff_subperiod_audit_and_review(research):
    r, a, p = research
    e = r.register(a, p)
    renamed = json.loads(json.dumps(p))
    renamed["spec"]["strategy"]["name"] = "名称不同不应重跑"
    assert r.register(a, renamed)["id"] == e["id"]
    r.store.checkpoint(a, "冻结完毕，接力运行", True)
    b = Research(r.project, r.root).store.claim(a["task_id"], "worker-b")
    # Current branch can change, but the frozen executor must still use its copy.
    (r.project / "src/technical/portfolio.py").write_text(
        'raise RuntimeError("new broken development branch")'
    )
    out = r.execute(b, e["id"])
    assert out["status"] == "completed" and out["result"]["verified"]
    assert out["result"]["data_end"] == "2020-02-14"
    assert len(out["result"]["metrics"]) == 2
    assert out["result"]["metrics"][0]["total_cost"] == 0
    assert out["result"]["metrics"][1]["total_cost"] > 0
    assert out["result"]["audit"]["no_lhb_inputs"]
    assert r.recover(e["id"])["id"] == e["id"]
    r.submit(b, REPORT)
    final = r.review(a["task_id"], "qa-role", "stop", "仅证明流程；不证明有效策略")
    assert final["status"] == "completed"
    with pytest.raises(ValueError):
        r.execute(b, e["id"])


def test_parent_crash_receipt_is_recoverable(research):
    r, a, p = research
    e = r.register(a, p)
    r.store.start(a, e["id"])
    job = r.root / "worker_jobs" / e["id"]
    subprocess.run(
        [sys.executable, str(job / "code/execute.py")],
        cwd=job / "code",
        capture_output=True,
        check=True,
    )
    # No parent execute() call finalizes SQLite; a fresh Research instance collects it.
    assert r.store.experiment(e["id"])["status"] == "running"
    assert Research(r.project, r.root).recover(e["id"])["status"] == "completed"


def test_tampered_frozen_input_fails_before_process_start(research):
    r, a, p = research
    e = r.register(a, p)
    job = r.root / "worker_jobs" / e["id"]
    (job / "input.json").write_text("{}")
    with pytest.raises(ValueError, match="冻结输入"):
        r.execute(a, e["id"])
    assert r.store.experiment(e["id"])["status"] == "failed"
    assert r.store.experiment(e["id"])["pid"] is None


def test_timeout_stops_owned_child_and_preserves_failure(research, monkeypatch):
    r, a, p = research
    e = r.register(a, p)

    class Process:
        pid = 99999999

        def poll(self):
            return None

    monkeypatch.setattr("src.researchops.service.subprocess.Popen", lambda *a, **k: Process())
    ticks = iter([0, 1201])
    monkeypatch.setattr("src.researchops.service.time.monotonic", lambda: next(ticks))
    stopped = []
    monkeypatch.setattr(r, "_stop", lambda proc: stopped.append(proc.pid))
    with pytest.raises(TimeoutError):
        r.execute(a, e["id"])
    assert stopped and r.store.experiment(e["id"])["status"] == "failed"


def test_registration_requires_explicit_date_snapshot_and_costs(research):
    r, a, p = research
    bad = json.loads(json.dumps(p))
    bad["spec"]["experiment"]["end_date"] = "latest"
    with pytest.raises(ValueError, match="明确结束"):
        r.register(a, bad)
    with pytest.raises(ValueError, match="成本"):
        r.register(a, {**p, "scenarios": ["configured"]})
    with pytest.raises(ValueError):
        r.register(a, {**p, "snapshot_id": "../../invalid"})
    assert r.store.get(a["task_id"])["experiments"] == []


def test_http_create_context_auth_and_no_secrets(tmp_path):
    server = make_server(tmp_path, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base) as response:
            page = response.read().decode()
        token = json.loads(re.search(r"window.TECHNICAL=(.*?);</script>", page).group(1))["token"]
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base + "/api/research/tasks", data=json.dumps(TASK).encode()))
        assert exc.value.code == 403
        with urlopen(
            Request(
                base + "/api/research/tasks",
                data=json.dumps(TASK).encode(),
                headers={"X-Research-Token": token},
            )
        ) as response:
            t = json.load(response)
        session = Store(tmp_path / "data/technical").claim(t["id"], "other-process")
        with urlopen(base + "/api/research/tasks/" + t["id"]) as response:
            context = json.load(response)
        assert context["task"]["owner"] == "other-process"
        assert session["token"] not in json.dumps(context)
        assert "tasks.js" not in page and "window.taskWorkbench" in page
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
