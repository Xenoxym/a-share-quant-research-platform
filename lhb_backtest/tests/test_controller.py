"""Exercise workflow invariants with deterministic agents, never paid calls in CI."""

import json
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from src.researchops.controller import Campaigns, Controller
from src.researchops.service import Research
from tests.test_researchops import TASK, REPORT


@pytest.fixture
def setup(tmp_path):
    project = tmp_path / "project"
    source = project / "research/evidence"
    source.mkdir(parents=True)
    (source / "REPORT.md").write_text("Negative research finding, illustrative test evidence.")
    (source / "results.json").write_text('{"failed_attempts":1,"numerical_reproduction":false}')
    r = Research(project)
    t = r.create(TASK)
    session = r.store.claim(t["id"], "original-researcher")
    r.attach(session, dict(path=str(source), kind="feature_analysis", summary="回顾性分析证据"))
    r.submit(session, REPORT)
    return r, t["id"], source


def test_analysis_import_without_dummy_backtest_and_tamper_blocks_review(setup):
    r, tid, source = setup
    t = r.store.get(tid)
    assert not t["experiments"] and t["submission"]["evidence_kind"] == "analysis_artifacts"
    assert t["evidence"][0]["payload"]["registration_timing"] == "post_hoc_import"
    (source / "REPORT.md").write_text("Source later changes; frozen copy remains.")
    r.verify_evidence(tid)
    from pathlib import Path

    frozen = Path(t["evidence"][0]["payload"]["folder"])
    (frozen / "REPORT.md").write_text("tampered")
    with pytest.raises(ValueError):
        r.review(tid, "independent", "stop", "cannot accept corrupt evidence")
    assert r.store.get(tid)["status"] == "review"


def test_readonly_reviewer_reads_live_state_without_schema_writes(setup):
    r, tid, _ = setup
    from src.technical.artifacts import digest

    cid = Campaigns(r).create(dict(title="Readonly inspection", task_ids=[tid]))["id"]

    before = digest(r.store.path)
    readonly = Research(r.project, r.root, readonly=True)
    assert Campaigns(readonly).get(cid)["state"]["status"] == "queued"
    assert readonly.context(tid)["task"]["submission"]["evidence_kind"] == "analysis_artifacts"
    assert digest(r.store.path) == before
    with pytest.raises(ValueError, match="只读"):
        readonly.store.review(tid, "reader", "stop", "cannot mutate")
    missing = r.project / "missing"
    with pytest.raises(ValueError, match="不存在"):
        Research(r.project, missing, readonly=True)
    assert not missing.exists()


def fake_transport(decision="stop", review="accept", fail=False):
    def invoke(job, folder, prompt, schema, research, cancelled, heartbeat, record, timeout):
        record(thread_id=job["id"], usage=dict(input_tokens=10, output_tokens=20))
        if fail:
            raise RuntimeError("provider unavailable")
        if job["role"] == "researcher":
            # Revisions retain old frozen evidence; this fake tests protocol, not research quality.
            heartbeat()
            return REPORT
        if job["role"] == "synthesis":
            context = json.loads((folder / "context.json").read_text(encoding="utf-8"))
            return dict(
                summary="Evidence-led decision",
                decisions=[
                    dict(
                        task_id=t["task"]["id"],
                        decision=decision if job["round"] == 1 else "stop",
                        rationale="Evidence constraints respected",
                        next_task=None,
                    )
                    for t in context["tasks"]
                ],
            )
        return dict(
            verdict=review if job["round"] == 1 else "accept",
            summary="Targeted review",
            findings=["Checked frozen evidence"],
            blockers=["Requires correction"] if review == "revise" and job["round"] == 1 else [],
            limitations=["Not source vendor verification"],
        )

    return invoke


def campaign(r, tid, **kwargs):
    return Campaigns(r).create(dict(title="Research campaign", task_ids=[tid], **kwargs))["id"]


def test_independent_parallel_reviews_and_synthesis_close_loop(setup):
    r, tid, _ = setup
    cid = campaign(r, tid)
    barrier = threading.Barrier(2)
    agent = fake_transport()

    def invoke(job, *args):
        if job["role"].endswith("reviewer"):
            barrier.wait(timeout=5)
        return agent(job, *args)

    out = Controller(r, invoke).run(cid)
    assert out["state"]["status"] == "completed"
    assert len(out["jobs"]) == 3
    assert r.store.get(tid)["review"]["decision"] == "stop"
    assert r.store.knowledge()[0]["evidence_kind"] == "analysis_artifacts"
    with pytest.raises(ValueError):
        Controller(r, invoke).run(cid)


def test_revision_automatically_researches_then_rechecks(setup):
    r, tid, _ = setup
    cid = campaign(r, tid, max_calls=7)
    out = Controller(r, fake_transport("revise", "revise")).run(cid)
    assert out["state"]["status"] == "completed"
    assert len(out["state"]["history"]) == 2 and len(out["jobs"]) == 7
    assert [j["role"] for j in out["jobs"]].count("researcher") == 1
    assert r.store.get(tid)["status"] == "completed"


def test_veto_prevents_optimistic_controller_promotion(setup):
    r, tid, _ = setup
    cid = campaign(r, tid)
    out = Controller(r, fake_transport("stop", "revise")).run(cid)
    assert out["state"]["status"] == "needs_attention"
    assert "阻断" in out["state"]["error"]
    assert r.store.get(tid)["status"] == "review"


def test_budget_preserves_pending_revision_without_extra_call(setup):
    r, tid, _ = setup
    cid = campaign(r, tid, max_calls=3)
    out = Controller(r, fake_transport("revise", "revise")).run(cid)
    assert out["state"]["status"] == "budget_exhausted"
    assert len(out["jobs"]) == 3 and out["state"]["pending"] == [tid]
    assert r.store.get(tid)["status"] == "queued"


def test_failed_agents_retained_no_retry_no_review(setup):
    r, tid, _ = setup
    cid = campaign(r, tid)
    out = Controller(r, fake_transport(fail=True)).run(cid)
    assert out["state"]["status"] == "needs_attention"
    assert all(j["state"]["status"] == "failed" for j in out["jobs"])
    assert r.store.get(tid)["status"] == "review"


def test_stop_before_dispatch_and_atomic_call_reservation(setup):
    r, tid, _ = setup
    db = Campaigns(r)
    cid = campaign(r, tid, max_calls=3)
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = list(pool.map(lambda _: db.reserve(cid, 1, tid, "data_reviewer"), range(8)))
    assert len({j["id"] for j in jobs}) == 1
    db.stop(cid)
    with pytest.raises(ValueError):
        Controller(r, fake_transport()).run(cid)
    out = db.get(cid)
    assert out["state"]["status"] == "stopped"
    assert len(out["jobs"]) == 1


def test_two_new_researchers_parallel_then_automatic_child_research(setup):
    r, _, source = setup
    tasks = [r.create({**TASK, "question": f"Independent hypothesis {i}"})["id"] for i in range(2)]
    cid = Campaigns(r).create(dict(title="Parallel research", task_ids=tasks, max_calls=11))["id"]
    barrier = threading.Barrier(2)
    base = fake_transport()

    def transport(job, folder, prompt, schema, research, *args):
        if job["role"] == "researcher":
            if job["round"] == 1:
                barrier.wait(timeout=5)
            session = json.loads((folder / "session.json").read_text())
            research.attach(
                session,
                dict(
                    path=str(source),
                    kind="reused_evidence",
                    summary="Fixture with separate task ownership",
                ),
            )
        out = base(job, folder, prompt, schema, research, *args)
        if job["role"] == "synthesis" and job["round"] == 1:
            out["decisions"][0].update(
                decision="continue",
                next_task={
                    **TASK,
                    "question": "Different followup mechanism",
                    "scope": "Within original research scope",
                    "baseline_runs": [],
                    "tags": [],
                },
            )
        return out

    out = Controller(r, transport).run(cid)
    assert out["state"]["status"] == "completed"
    assert len(out["jobs"]) == 11
    assert len([j for j in out["jobs"] if j["role"] == "researcher"]) == 3
    assert len(r.store.get(tasks[0])["children"]) == 1
    assert r.store.get(r.store.get(tasks[0])["children"][0])["status"] == "completed"


def test_same_session_reuse_and_concurrent_campaign_are_rejected(setup):
    r, tid, _ = setup
    cid = campaign(r, tid)
    with pytest.raises(ValueError, match="其他"):
        campaign(r, tid)
    base = fake_transport()

    def transport(job, folder, prompt, schema, research, cancelled, heartbeat, record, timeout):
        out = base(job, folder, prompt, schema, research, cancelled, heartbeat, record, timeout)
        record(thread_id="same-session-is-not-independent")
        return out

    out = Controller(r, transport).run(cid)
    assert out["state"]["status"] == "needs_attention"
    assert "独立会话" in out["state"]["error"]
    assert r.store.get(tid)["status"] == "review"


def test_campaign_http_auth_atomic_launch_stop_and_public_state(setup, monkeypatch):
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    import re
    from src.technical.server import make_server

    r, tid, _ = setup
    server = make_server(r.project, port=0, root=r.root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    launched = []

    def spawn(args, **kwargs):
        launched.append(args)
        # A fast child can reach running before Popen returns to the HTTP handler.
        Campaigns(r).update(args[-1], status="running", stage="child already running")
        return type("Process", (), {"pid": 123456})()

    monkeypatch.setattr("src.technical.server.subprocess.Popen", spawn)
    try:
        with urlopen(base) as response:
            page = response.read().decode()
        token = json.loads(re.search(r"window.TECHNICAL=(.*?);</script>", page).group(1))["token"]
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base + "/api/research/campaigns", data=b"{}"))
        assert exc.value.code == 403

        def post(path, value):
            with urlopen(
                Request(
                    base + path,
                    data=json.dumps(value).encode(),
                    headers={"X-Research-Token": token},
                )
            ) as response:
                return json.load(response)

        c = post("/api/research/campaigns", dict(title="HTTP dispatch", task_ids=[tid]))
        path = "/api/research/campaigns/" + c["id"]
        out = post(path + "/run", {})
        assert out["state"]["status"] == "running"
        assert out["state"]["stage"] == "child already running"
        with pytest.raises(HTTPError):
            post(path + "/run", {})
        assert len(launched) == 1
        assert post(path + "/stop", {})["state"]["stop_requested"]
        with urlopen(base + path) as response:
            text = response.read().decode()
        assert "token_hash" not in text and "session.json" not in text
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
