"""Real failure/continuation invariants; no paid calls or credential access."""
import json
import time

import pytest

from src.researchops.controller import Campaigns, Controller, DispatchDeferred
from src.researchops.loop import Loops, Supervisor, quota_failure
from src.researchops.quota import quota_status
from tests.test_controller import setup, fake_transport
from tests.test_researchops import TASK, REPORT
from src.technical.artifacts import content_id, digest, write_json


def payload(pct=7, reset=500, weekly=20, week_reset=1000):
    return dict(ordinaryUsageAllowed=True, rateLimits=dict(limitId="codex", primary=dict(usedPercent=pct, windowDurationMins=300, resetsAt=reset),
        secondary=dict(usedPercent=weekly, windowDurationMins=10080, resetsAt=week_reset), rateLimitReachedType=None))


def test_server_weekly_reset_dominates_five_hour_window():
    assert quota_status(payload(100, 500, 100, 1000), now=100)["reset_at"] == 1000
    assert quota_status(payload(100, None), now=100)["reset_at"] is None
    assert quota_status(payload(100, 99), now=100)["blocked"]
    assert not quota_status(payload(), now=100)["blocked"]
    assert not quota_status({})["known"]
    assert not quota_status(dict(rateLimits=dict(primary=dict(usedPercent=float("nan")))))["known"]


def test_credit_or_owner_limit_has_no_guessed_reset():
    p = payload()
    p["rateLimits"]["rateLimitReachedType"] = "workspace_owner_usage_limit_reached"
    s = quota_status(p, now=100)
    assert s["blocked"] and s["reset_at"] is None


@pytest.mark.parametrize("permission", [False,None,"missing"])
def test_backend_denied_or_unknown_permission_cannot_be_inferred_from_percent(permission):
    p=payload()
    if permission == "missing":
        p.pop("ordinaryUsageAllowed")
    else:
        p["ordinaryUsageAllowed"]=permission
    s=quota_status(p,now=100)
    assert s["blocked"]
    assert s["known"] is (permission is False)


def test_guard_respects_backend_denial_and_waits_for_explicit_permission(setup,monkeypatch):
    r,tid,_=setup
    cfg=config(tid);cfg["resume_after_reset"]=True
    loops=Loops(r);lid=loops.create(cfg)["state"]["id"]
    denied=payload();denied["ordinaryUsageAllowed"]=False
    sup=Supervisor(r,probe=lambda:denied)
    with pytest.raises(DispatchDeferred):sup.guard(lid)
    calls=[]
    def probe():
        calls.append(1)
        return denied if len(calls)==1 else payload()
    sup=Supervisor(r,probe=probe)
    monkeypatch.setattr("src.researchops.loop.time.sleep",lambda seconds:None)
    assert sup.wait_quota(lid,quota_status(payload(100),now=100))
    assert len(calls)==2
    assert not Campaigns(r).list()


def test_wait_does_not_report_recovery_when_permission_is_unavailable(setup,monkeypatch):
    r,tid,_=setup
    cfg=config(tid);cfg["resume_after_reset"]=True
    loops=Loops(r);lid=loops.create(cfg)["state"]["id"]
    unknown=payload();unknown["ordinaryUsageAllowed"]=None
    sup=Supervisor(r,probe=lambda:unknown)
    monkeypatch.setattr("src.researchops.loop.time.sleep",lambda seconds:None)
    assert not sup.wait_quota(lid,quota_status(payload(100),now=100))
    assert loops.get(lid)["state"]["status"]=="needs_attention"
    assert 'quota_available_again' not in (loops.folder(lid)/'events.jsonl').read_text(encoding='utf-8')


def test_partial_round_quota_pause_reuses_completed_reviewer_and_budget(setup):
    r, tid, _ = setup
    db = Campaigns(r)
    cid = db.create(dict(title="Quota continuation", task_ids=[tid], max_parallel=1, max_calls=3))["id"]
    count = 0
    def guard():
        nonlocal count
        count += 1
        if count >= 2:
            raise DispatchDeferred(quota_status(payload(100), now=100))
    out = Controller(r, fake_transport(), before_job=guard).run(cid)
    assert out["state"]["status"] == "waiting_quota"
    assert len(out["jobs"]) == 1 and out["jobs"][0]["state"]["status"] == "completed"
    first_id = out["jobs"][0]["id"]
    with pytest.raises(ValueError, match="其他"):
        db.create(dict(title="Duplicate", task_ids=[tid]))
    db.update(cid, quota_wait_started=time.time()-18000)
    out = Controller(r, fake_transport(), before_job=lambda: None).run(cid)
    assert out["state"]["status"] == "completed" and len(out["jobs"]) == 3
    assert out["jobs"][0]["id"] == first_id
    assert out["state"]["quota_paused_seconds"] >= 18000
    assert out["config"]["max_calls"] == 3 and out["state"]["round"] == 2


def config(tid, **changes):
    return dict(title="ML loop", task_ids=[tid], campaign_budget=dict(max_parallel=1, max_rounds=1, max_calls=3,
        job_timeout_seconds=60, wall_seconds=60), agenda=[], guidance="Preserve negative findings.",
        resume_after_reset=False, **changes)


def notice_transport(tid, kind="mechanism", decision="continue", review="accept", refs=None):
    base = fake_transport("revise" if decision == "revise" else "stop", review)
    def invoke(job, folder, prompt, schema, research, *args):
        out = base(job, folder, prompt, schema, research, *args)
        if job["role"] == "synthesis" and "notice" in schema["properties"]:
            if decision == "continue":
                out["decisions"][0].update(decision="continue", next_task={**TASK,
                    "question":"Distinct unstarted mechanism", "scope":"No new computation in this fixture",
                    "baseline_runs":[], "tags":[]})
            evidence = refs if refs is not None else [research.store.get(tid)["evidence"][0]["id"]]
            out["notice"] = None if kind is None else dict(kind=kind, title="Important illustrative finding",
                explanation="A fixture checks stopping before a new child starts, not financial effectiveness.",
                task_ids=[tid], evidence_refs=evidence, limitations=["Synthetic protocol fixture"])
        return out
    return invoke


def install_notice(loops, lid, refs=None):
    return loops.enable_noteworthy(lid, dict(guidance="Stop only for material new evidence.",
        reported_evidence_refs=refs or []), "User explicitly added a conditional stop")


def test_noteworthy_stops_before_child_and_never_resumes_on_quota_reset(setup):
    r, tid, _ = setup
    loops=Loops(r);lid=loops.create(config(tid))["state"]["id"]
    before=digest(loops.folder(lid)/"config.json")
    install_notice(loops,lid)
    def notice_factory(research,**kwargs):
        return Controller(research,notice_transport(tid),**kwargs)
    sup=Supervisor(r,probe=lambda:payload(),controller_factory=notice_factory)
    out=sup.run(lid);assert out["state"]["status"]=="noteworthy_result"
    cid=out["state"]["campaign_id"];c=Campaigns(r).get(cid)
    child=c["state"]["pending"][0]
    assert len(c["jobs"])==3 and r.store.get(child)["status"]=="queued"
    assert out["state"]["pending"]==[child]
    archived=json.loads((loops.folder(lid)/"checkpoints"/cid/"tasks.json").read_text(encoding='utf-8'))
    assert child in archived
    assert [e["kind"] for e in r.store.get(child)["events"]]==["created"]
    assert digest(loops.folder(lid)/"config.json")==before
    assert (loops.folder(lid)/"notices"/f"{cid}-round-1"/"receipt.json").exists()
    sup.probe=lambda:pytest.fail("An important-result stop must not even probe for restart")
    assert sup.run(lid)["state"]["status"]=="noteworthy_result"
    assert len(Campaigns(r).get(cid)["jobs"])==3
    with pytest.raises(ValueError,match="其他"):
        Campaigns(r).create(dict(title="Bypass stop",task_ids=[child]))
    with pytest.raises(RuntimeError,match="等待用户"):
        sup.guard(lid)
    loops.acknowledge(lid,"User explicitly said continue after reading this report")
    resumed=Campaigns(r).get(cid)
    assert loops.get(lid)["state"]["status"]=="ready"
    assert resumed["state"]["round"]==2 and resumed["config"]==c["config"]
    assert len(resumed["jobs"])==3 and resumed["state"]["history"]==c["state"]["history"]
    assert loops.get(lid)["notice_policy"]["reported_evidence_refs"]


@pytest.mark.parametrize("kind,refs,error",[
    ("learning_gain",None,"两名复核"),
    ("mechanism",["evidence-does-not-exist"],"结构"),
])
def test_noteworthy_cannot_present_unaccepted_research_as_learning_success(setup,kind,refs,error):
    r,tid,_=setup;cid=Campaigns(r).create(dict(title="Blocked finding",task_ids=[tid]))["id"]
    c=Controller(r,notice_transport(tid,kind=kind,decision="revise",review="revise",refs=refs),
                 notice_policy={"guidance":"Material evidence", "reported_evidence_refs":[]}).run(cid)
    assert c["state"]["status"]=="needs_attention" and error in c["state"]["error"]
    assert r.store.get(tid)["status"]=="review"


def test_unknown_notice_reference_is_rejected_by_schema_before_business_transition(setup):
    r,tid,_=setup;cid=Campaigns(r).create(dict(title="Unknown reference",task_ids=[tid]))["id"]
    c=Controller(r,notice_transport(tid,refs=["evidence-does-not-exist"]),
                 notice_policy={"guidance":"Material evidence", "reported_evidence_refs":[]}).run(cid)
    assert c["state"]["status"]=="needs_attention" and "结构" in c["state"]["error"]
    assert r.store.get(tid)["status"]=="review"
    assert not c["state"].get("notice")
    assert not c["state"]["history"]


def test_schema_valid_evidence_from_another_task_cannot_support_notice(setup):
    """The schema allows this round's IDs; the business gate checks task linkage."""
    r, tid, source = setup
    other = r.create({**TASK, "question": "Different task with its own attached evidence"})
    session = r.store.claim(other["id"], "other-fixture-researcher")
    r.attach(session, dict(path=str(source), kind="feature_analysis", summary="Other task evidence"))
    r.submit(session, REPORT)
    wrong_ref = r.store.get(other["id"])["evidence"][0]["id"]
    cid = Campaigns(r).create(dict(title="Mislinked finding", task_ids=[tid, other["id"]]))["id"]
    c = Controller(r, notice_transport(tid, decision="stop", refs=[wrong_ref]),
                   notice_policy={"guidance": "Material evidence", "reported_evidence_refs": []}).run(cid)
    assert c["state"]["status"] == "needs_attention"
    assert "关联证据" in c["state"]["error"]
    assert all(r.store.get(t)["status"] == "review" for t in (tid, other["id"]))
    assert not c["state"].get("notice")
    assert not c["state"]["history"]


def test_integrity_notice_preserves_blocker_and_does_not_accept_strategy(setup):
    r,tid,_=setup;loops=Loops(r);lid=loops.create(config(tid))["state"]["id"];install_notice(loops,lid)
    def notice_factory(research,**kwargs):
        return Controller(research,notice_transport(tid,kind="integrity_issue",decision="revise",review="revise"),**kwargs)
    out=Supervisor(r,probe=lambda:payload(),controller_factory=notice_factory).run(lid)
    assert out["state"]["status"]=="noteworthy_result"
    assert r.store.get(tid)["status"]=="queued" and r.store.get(tid)["review"]["decision"]=="revise"
    assert "不表示策略研究通过" in (__import__('pathlib').Path(out["state"]["notice_folder"])/"REPORT.md").read_text(encoding='utf-8')


@pytest.mark.parametrize("kind",[None,"mechanism"])
def test_ordinary_negative_or_already_reported_findings_do_not_pause(setup,kind):
    r,tid,_=setup;loops=Loops(r);lid=loops.create(config(tid))["state"]["id"]
    install_notice(loops,lid)
    notice=dict(kind="mechanism",title="Important illustrative finding",
        explanation="A fixture checks stopping before a new child starts, not financial effectiveness.",
        task_ids=[tid],evidence_refs=[r.store.get(tid)["evidence"][0]["id"]],limitations=["Synthetic protocol fixture"])
    loops.update(lid,notice_acknowledgements=[dict(evidence_refs=notice["evidence_refs"],
        notice=notice,notice_signature=content_id(notice))])
    def notice_factory(research,**kwargs):
        return Controller(research,notice_transport(tid,kind=kind,decision="stop"),**kwargs)
    out=Supervisor(r,probe=lambda:payload(),controller_factory=notice_factory).run(lid)
    assert out["state"]["status"]=="needs_new_evidence"
    assert not (loops.folder(lid)/"notices").exists()


def test_policy_and_notice_archive_integrity_block_tampered_resume(setup):
    r,tid,_=setup;loops=Loops(r);lid=loops.create(config(tid))["state"]["id"]
    entry=install_notice(loops,lid)
    path=loops.folder(lid)/entry["state"]["notice_policy_file"]
    path.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match="政策"):
        loops.get(lid)


def test_corrupt_important_report_cannot_be_acknowledged_and_original_stop_stays(setup):
    r,tid,_=setup;loops=Loops(r);lid=loops.create(config(tid))["state"]["id"];install_notice(loops,lid)
    def notice_factory(research,**kwargs):
        return Controller(research,notice_transport(tid),**kwargs)
    out=Supervisor(r,probe=lambda:payload(),controller_factory=notice_factory).run(lid)
    from pathlib import Path
    (Path(out["state"]["notice_folder"])/"REPORT.md").write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match="完整性"):
        loops.acknowledge(lid,"User asked to continue")
    assert loops.get(lid)["state"]["status"]=="noteworthy_result"
    assert Campaigns(r).get(out["state"]["campaign_id"])["state"]["status"]=="noteworthy_result"


def test_new_integrity_issue_using_old_reported_evidence_still_stops(setup):
    r,tid,_=setup;loops=Loops(r);lid=loops.create(config(tid))["state"]["id"]
    install_notice(loops,lid,[r.store.get(tid)["evidence"][0]["id"]])
    def notice_factory(research,**kwargs):
        return Controller(research,notice_transport(tid,kind="integrity_issue",decision="revise",review="revise"),**kwargs)
    out=Supervisor(r,probe=lambda:payload(),controller_factory=notice_factory).run(lid)
    assert out["state"]["status"]=="noteworthy_result"
    assert len(Campaigns(r).get(out["state"]["campaign_id"])["jobs"])==3


@pytest.mark.parametrize("required_file",["REPORT.md","notice.json"])
def test_receipt_missing_artifact_cannot_clear_important_stop(setup,required_file):
    from pathlib import Path
    r,tid,_=setup;loops=Loops(r);lid=loops.create(config(tid))["state"]["id"];install_notice(loops,lid)
    def notice_factory(research,**kwargs):
        return Controller(research,notice_transport(tid),**kwargs)
    out=Supervisor(r,probe=lambda:payload(),controller_factory=notice_factory).run(lid)
    path=Path(out["state"]["notice_folder"])/"receipt.json";receipt=json.loads(path.read_text(encoding='utf-8'))
    del receipt["artifacts"][required_file];write_json(path,receipt)
    with pytest.raises(ValueError,match="收据"):
        loops.acknowledge(lid,"User asked to continue")
    assert loops.get(lid)["state"]["status"]=="noteworthy_result"


def test_failed_author_session_cannot_be_reused_for_independent_reviewer(setup):
    r,tid,_=setup;db=Campaigns(r);cid=db.create(dict(title="Source identity",task_ids=[tid],max_calls=4))["id"]
    failed=db.reserve(cid,1,tid,"researcher")
    db.job_update(failed["id"],status="failed",ended=time.time(),thread_id="author-session")
    base=fake_transport()
    def reused(job,folder,prompt,schema,research,cancelled,heartbeat,record,timeout):
        out=base(job,folder,prompt,schema,research,cancelled,heartbeat,record,timeout)
        if job["role"]=="method_reviewer":record(thread_id="author-session")
        return out
    out=Controller(r,reused).run(cid)
    assert out["state"]["status"]=="needs_attention" and "独立会话" in out["state"]["error"]
    assert not [j for j in out["jobs"] if j["role"]=="synthesis"]


def test_failed_research_handoff_can_resume_review_in_last_round_without_reset(setup):
    r,tid,_=setup;db=Campaigns(r)
    cid=db.create(dict(title="Last-round submission handoff",task_ids=[tid],max_rounds=1,max_calls=4))["id"]
    # Simulate a genuinely interrupted researcher whose evidence already exists.
    failed=db.reserve(cid,1,tid,"researcher")
    db.job_update(failed["id"],status="failed",ended=time.time(),error="Quota error",thread_id="failed-session")
    db.update(cid,status="needs_attention",started=time.time(),error="Quota error")
    with pytest.raises(ValueError,match="新提交"):
        db.resume_review(cid,"Old submission alone is insufficient")
    r.review(tid,"previous-role","revise","Complete the already attached correction")
    session=r.store.claim(tid,"handoff-worker")
    r.submit(session,REPORT)
    before=db.get(cid);db.resume_review(cid,"Explicit continuation after new handoff submission")
    resumed=Controller(r,fake_transport()).run(cid)
    assert resumed["state"]["status"]=="completed" and len(resumed["jobs"])==4
    assert resumed["jobs"][0]==before["jobs"][0] and resumed["jobs"][0]["state"]["status"]=="failed"
    assert resumed["config"]==before["config"] and resumed["state"]["started"]==before["state"]["started"]
    assert resumed["state"]["submission_recoveries"][0]["failed_jobs"]==[failed["id"]]
    assert all(j["role"]!="researcher" for j in resumed["jobs"][1:])


def test_checkpoint_preserves_original_failure_and_versions_later_recovery(setup):
    r,tid,_=setup;loops=Loops(r);lid=loops.create(config(tid))["state"]["id"]
    db=Campaigns(r);cid=db.create(dict(title="Versioned checkpoint",task_ids=[tid]))["id"]
    db.update(cid,status="needs_attention",error="Original failure")
    sup=Supervisor(r,probe=lambda:payload());original=sup.checkpoint(lid,db.get(cid))
    original_hash=digest(original/"campaign.json")
    db.update(cid,status="completed",error=None)
    recovered=sup.checkpoint(lid,db.get(cid))
    assert recovered!=original and recovered.parent==original/"versions"
    assert digest(original/"campaign.json")==original_hash
    assert json.loads((recovered/"campaign.json").read_text(encoding='utf-8'))["state"]["status"]=="completed"


@pytest.mark.parametrize("message,expected",[
    ("You’ve hit your usage limit. Try again at 8:25 PM.",True),
    ("Connection reset by peer",False),
])
def test_real_curly_apostrophe_quota_error_is_recognized_without_guessing(setup,message,expected):
    r,tid,_=setup;db=Campaigns(r);cid=db.create(dict(title="Provider error",task_ids=[tid]))["id"]
    j=db.reserve(cid,1,tid,"researcher");folder=db.root/cid/j["id"];folder.mkdir(parents=True)
    (folder/"events.jsonl").write_text(json.dumps(dict(type="error",message=message)),encoding='utf-8')
    db.job_update(j["id"],status="failed",folder=str(folder))
    assert quota_failure(db.get(cid)) is expected


def factory(research, **kwargs):
    return Controller(research, fake_transport(), **kwargs)


def test_supervisor_archives_before_advance_and_does_not_restart_finished_stage(setup):
    r, tid, _ = setup
    loops = Loops(r)
    lid = loops.create(config(tid))["state"]["id"]
    sup = Supervisor(r, probe=lambda: payload(), controller_factory=factory)
    out = sup.run(lid)
    assert out["state"]["status"] == "needs_new_evidence"
    cid = out["state"]["archived"][0]
    folder = loops.folder(lid) / "checkpoints" / cid
    assert (folder / "receipt.json").exists() and (folder / "next_plan.json").exists()
    assert sup.run(lid)["state"]["archived"] == [cid]
    assert len(Campaigns(r).list()) == 1


def test_auth_failure_is_not_a_quota_wait_and_no_agent_is_reserved(setup):
    r, tid, _ = setup
    lid = Loops(r).create(config(tid))["state"]["id"]
    def failed():
        raise RuntimeError("authentication required")
    out = Supervisor(r, probe=failed, controller_factory=factory).run(lid)
    assert out["state"]["status"] == "needs_attention"
    assert not Campaigns(r).list()


def test_quota_wait_can_stop_without_starting_campaign(setup):
    r, tid, _ = setup
    loops = Loops(r)
    lid = loops.create(config(tid))["state"]["id"]
    out = Supervisor(r, probe=lambda: payload(100, time.time()+300), controller_factory=factory).run(lid)
    assert out["state"]["status"] == "waiting_quota" and not Campaigns(r).list()
    loops.stop(lid)
    assert Supervisor(r, probe=lambda: payload(), controller_factory=factory).run(lid)["state"]["status"] == "stopped"


def test_stage_revision_budget_is_not_reset_by_supervisor(setup):
    r, tid, _ = setup
    lid = Loops(r).create(config(tid))["state"]["id"]
    def revisions(research, **kwargs):
        return Controller(research, fake_transport("revise", "revise"), **kwargs)
    out = Supervisor(r, probe=lambda: payload(), controller_factory=revisions).run(lid)
    assert out["state"]["status"] == "needs_attention"
    assert "清零预算" in out["state"]["error"]
    assert len(Campaigns(r).list()) == 1 and r.store.get(tid)["status"] == "queued"


def child_transport(source, revise_child):
    base = fake_transport()
    def invoke(job, folder, prompt, schema, research, *args):
        if job["role"] == "researcher":
            session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
            research.attach(session, dict(path=str(source), kind="reused_evidence", summary="Controlled child fixture"))
        out = base(job, folder, prompt, schema, research, *args)
        if job["role"] == "synthesis" and job["round"] == 1:
            out["decisions"][0].update(decision="continue", next_task={**TASK, "question":"A distinct child mechanism", "scope":"Controlled child analysis", "baseline_runs":[], "tags":[]})
        elif job["role"] == "synthesis" and revise_child:
            out["decisions"][0].update(decision="revise", next_task=None)
        return out
    return invoke


def test_parent_continue_then_child_revise_cannot_reset_campaign_budget(setup):
    r, tid, source = setup
    loops = Loops(r)
    lid = loops.create(config(tid))["state"]["id"]
    db = Campaigns(r)
    cid = db.create(dict(title="Executed child revision", task_ids=[tid], max_rounds=2, max_calls=7))["id"]
    c = Controller(r, child_transport(source, revise_child=True)).run(cid)
    child = c["state"]["pending"][0]
    assert c["state"]["status"] == "budget_exhausted" and child != tid
    assert len(c["jobs"]) == 7 and r.store.get(child)["status"] == "queued"
    sup = Supervisor(r, probe=lambda: payload())
    with pytest.raises(RuntimeError, match="清零预算"):
        sup.advance(lid, c)
    assert loops.get(lid)["state"]["phase"] == 1
    assert c["config"]["max_calls"] == 7 and len(db.list()) == 1


def test_fresh_unexecuted_child_can_roll_into_next_stage(setup):
    r, tid, source = setup
    loops = Loops(r)
    lid = loops.create(config(tid))["state"]["id"]
    db = Campaigns(r)
    cid = db.create(dict(title="Fresh next question", task_ids=[tid], max_rounds=1, max_calls=3))["id"]
    c = Controller(r, child_transport(source, revise_child=False)).run(cid)
    child = c["state"]["pending"][0]
    assert not any(j["task_id"] == child for j in c["jobs"])
    Supervisor(r, probe=lambda: payload()).advance(lid,c)
    assert loops.get(lid)["state"]["pending"] == [child]
    assert loops.get(lid)["state"]["phase"] == 2


def test_child_started_outside_current_campaign_cannot_receive_fresh_budget(setup):
    r, tid, source = setup
    loops=Loops(r);lid=loops.create(config(tid))["state"]["id"]
    cid=Campaigns(r).create(dict(title="Outside child work",task_ids=[tid],max_rounds=1,max_calls=3))["id"]
    c=Controller(r,child_transport(source,revise_child=False)).run(cid)
    child=c["state"]["pending"][0]
    session=r.store.claim(child,"other-session")
    r.store.checkpoint(session,"Work began outside this campaign; preserve history.",True)
    assert not any(j["task_id"]==child for j in c["jobs"])
    with pytest.raises(RuntimeError,match="清零预算"):
        Supervisor(r,probe=lambda:payload()).advance(lid,c)
    assert loops.get(lid)["state"]["phase"]==1


def completed_parent_with_recovered_agenda(setup):
    r, tid, source = setup
    spec = {**TASK, "question": "Predeclared distinct agenda mechanism",
            "scope": "Controlled agenda-only fixture", "baseline_runs": [], "tags": []}
    loops = Loops(r)
    cfg = config(tid)
    cfg["agenda"] = [spec]
    lid = loops.create(cfg)["state"]["id"]
    cid = Campaigns(r).create(dict(title="Completed parent before agenda recovery", task_ids=[tid],
                                  max_rounds=1, max_calls=3))["id"]
    c = Controller(r, fake_transport()).run(cid)
    assert c["state"]["status"] == "completed" and c["state"]["pending"] == []
    agenda_id = "task-" + content_id(dict(loop=lid, agenda=0))[:12]
    r.create(dict(spec, parent_id=tid), task_id=agenda_id)
    return r, loops, lid, c, agenda_id, source


@pytest.mark.parametrize("past_work", ["handoff", "submitted_and_revised"])
def test_recovered_agenda_with_prior_work_cannot_receive_fresh_budget(setup, past_work):
    r, loops, lid, c, agenda_id, source = completed_parent_with_recovered_agenda(setup)
    session = r.store.claim(agenda_id, "outside-agenda-worker")
    if past_work == "handoff":
        r.store.checkpoint(session, "Previously started work must retain its budget.", True)
    else:
        r.attach(session, dict(path=str(source), kind="reused_evidence", summary="Previous agenda attempt"))
        r.submit(session, REPORT)
        r.review(agenda_id, "outside-agenda-reviewer", "revise", "Keep this task's already used budget.")
    assert r.store.get(agenda_id)["status"] == "queued"
    assert not any(j["task_id"] == agenda_id for j in c["jobs"])
    with pytest.raises(RuntimeError, match="清零预算"):
        Supervisor(r, probe=lambda: payload()).advance(lid, c)
    state = loops.get(lid)["state"]
    assert state["phase"] == 1 and state["agenda_cursor"] == 0
    assert not (loops.folder(lid) / "checkpoints" / c["id"] / "next_plan.json").exists()
    assert len(Campaigns(r).list()) == 1


def test_pristine_created_only_agenda_recovery_can_receive_first_budget(setup):
    r, loops, lid, c, agenda_id, _ = completed_parent_with_recovered_agenda(setup)
    assert [event["kind"] for event in r.store.get(agenda_id)["events"]] == ["created"]
    Supervisor(r, probe=lambda: payload()).advance(lid, c)
    state = loops.get(lid)["state"]
    assert state["pending"] == [agenda_id] and state["phase"] == 2
    assert state["agenda_cursor"] == 1


def test_mid_call_quota_requires_explicit_provider_error_and_retains_failure(setup):
    r, tid, _ = setup
    db = Campaigns(r)
    cid = db.create(dict(title="Interrupted provider call", task_ids=[tid], max_parallel=1))["id"]
    c = Controller(r, fake_transport(fail=True)).run(cid)
    job = c["jobs"][0]
    from pathlib import Path
    events = Path(job["state"]["folder"]) / "events.jsonl"
    events.write_text(json.dumps(dict(type="turn.failed", error=dict(message="network unavailable")))+"\n", encoding="utf-8")
    assert not quota_failure(c)
    events.write_text(json.dumps(dict(type="turn.failed", error=dict(message="You've hit your usage limit")))+"\n", encoding="utf-8")
    for other in c["jobs"][1:]:
        (Path(other["state"]["folder"]) / "events.jsonl").write_text(events.read_text(encoding="utf-8"), encoding="utf-8")
    assert quota_failure(c)
    lid = Loops(r).create(config(tid))["state"]["id"]
    sup = Supervisor(r, probe=lambda: payload())
    sup.park_failed_quota(lid, c, quota_status(payload(100), now=100))
    db.update(cid, quota_wait_started=time.time()-18000)
    sup.resume_failed_quota(cid)
    resumed = db.get(cid)
    assert resumed["state"]["round"] == 2 and resumed["state"]["quota_paused_seconds"] >= 18000
    assert resumed["config"] == c["config"] and resumed["jobs"][0]["state"]["status"] == "failed"
    assert events.is_file() and r.store.get(tid)["status"] == "review"
