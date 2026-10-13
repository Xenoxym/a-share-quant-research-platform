"""Real frozen-worker acceptance and interruption/cache counterexamples."""

import copy
import json

import pytest

from src.researchops.audit_cache import numerical_audit
from src.researchops.claims import claim_to_file
from src.researchops.service import Research
from src.researchops.store import Store
from src.researchops.workflow import ExecutionPlans, validate_plan
from tests.test_alpha_batch_registration import setup
from tests.test_researchops import TASK
from tests.test_researchops import research as research


def plan_for(session, steps, **changes):
    return dict(
        schema="execution-plan-v1",
        id="acceptance-plan",
        task_id=session["task_id"],
        steps=steps,
        max_starts=2,
        timeout_seconds=3600,
        **changes,
    )


def proposal_step(proposal, name="account", needs=None):
    return dict(id=name, needs=needs or [], proposal=proposal)


def test_real_account_plan_runs_once_and_reuses_after_deadline(research, monkeypatch):
    ops, session, proposal = research
    plans = ExecutionPlans(ops)
    plan = plan_for(session, [proposal_step(proposal)])
    first = plans.run(session, plan)
    assert first["completed"] and first["starts"] == 1
    eid = first["steps"][0]["experiment_id"]
    ex = ops.store.experiment(eid)
    assert len(ex["result"]["metrics"]) == 2
    assert ex["result"]["audit"]["no_lhb_inputs"]
    with ops.store.connection(True) as db:
        db.execute("UPDATE execution_plans SET deadline=0 WHERE id=?", (plan["id"],))

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed steps must not register or execute again")

    monkeypatch.setattr(ops, "register", forbidden)
    monkeypatch.setattr(ops, "execute", forbidden)
    second = plans.run(session, plan)
    assert second["completed"] and second["starts"] == 1
    assert second["steps"][0]["experiment_id"] == eid
    assert len(ops.store.get(session["task_id"])["experiments"]) == 1
    assert second["usage"] == {"llm_calls": 0, "desktop_tokens": None}
    assert ExecutionPlans(Research(ops.project, ops.root, readonly=True)).status(plan["id"])


def test_crash_between_registration_and_checkpoint_uses_original_freeze(research, monkeypatch):
    ops, session, proposal = research
    plans = ExecutionPlans(ops)
    plan = plan_for(session, [proposal_step(proposal)])
    original = plans._save

    def crash(*args, **kwargs):
        if kwargs.get("status") == "registered":
            raise SystemExit("parent interrupted before step ID saved")
        return original(*args, **kwargs)

    monkeypatch.setattr(plans, "_save", crash)
    with pytest.raises(SystemExit):
        plans.run(session, plan)
    ex = ops.store.get(session["task_id"])["experiments"][0]
    # A changed working tree must not create a second registration on recovery.
    (ops.project / "src/technical/portfolio.py").write_text("raise RuntimeError('changed')")
    monkeypatch.setattr(plans, "_save", original)
    monkeypatch.setattr(ops, "register", lambda *a: pytest.fail("must recover operation"))
    final = plans.run(session, plan)
    assert final["completed"] and final["steps"][0]["experiment_id"] == ex["id"]
    assert len(ops.store.get(session["task_id"])["experiments"]) == 1


def test_failed_dependency_preserved_without_downstream_or_retry(research, monkeypatch):
    ops, session, proposal = research
    ex = ops.register(session, proposal)
    ops.store.cancel(session, ex["id"], "retained failed attempt")
    plan = plan_for(
        session,
        [
            dict(id="failed", needs=[], experiment_id=ex["id"]),
            proposal_step(proposal, "later", ["failed"]),
        ],
    )
    monkeypatch.setattr(ops, "execute", lambda *a: pytest.fail("no automatic retry"))
    result = ExecutionPlans(ops).run(session, plan)
    assert result["steps"][0]["status"] == "blocked"
    assert result["steps"][1]["status"] == "pending"
    assert result["starts"] == 0
    assert len(ops.store.get(session["task_id"])["experiments"]) == 1


def test_plan_budget_and_definition_cannot_reset(research):
    ops, session, proposal = research
    plan = plan_for(session, [proposal_step(proposal)])
    plan["max_starts"] = 0
    plans = ExecutionPlans(ops)
    result = plans.run(session, plan)
    assert result["starts"] == 0 and not ops.store.get(session["task_id"])["experiments"]
    changed = dict(plan, max_starts=2)
    with pytest.raises(ValueError, match="immutable"):
        plans.run(session, changed)


def test_short_deadline_does_not_dispatch_long_worker(research):
    ops, session, proposal = research
    plan = plan_for(session, [proposal_step(proposal)])
    plan["timeout_seconds"] = 30
    result = ExecutionPlans(ops).run(session, plan)
    assert result["starts"] == 0 and result["steps"][0]["status"] == "blocked"


def test_claim_write_failure_rolls_back_database_and_preserves_existing_file(tmp_path, monkeypatch):
    store = Store(tmp_path / "store")
    task = store.create(TASK)
    session_path = tmp_path / "session.json"

    def fail(*args, **kwargs):
        raise OSError("disk write failed")

    with monkeypatch.context() as patch:
        patch.setattr("src.researchops.claims.json.dump", fail)
        with pytest.raises(OSError):
            claim_to_file(store, task["id"], "owner", session_path)
    assert store.get(task["id"])["status"] == "queued" and not session_path.exists()
    session = claim_to_file(store, task["id"], "owner", session_path)
    assert json.loads(session_path.read_text()) == session
    with pytest.raises(FileExistsError):
        claim_to_file(store, task["id"], "other", session_path)
    assert json.loads(session_path.read_text()) == session


def test_atomic_operation_rejects_changed_request_and_deduplicates(tmp_path):
    store = Store(tmp_path)
    task = store.create(TASK)
    session = dict(
        store.claim(task["id"], "owner"),
        registration_key="plan:one:first",
        registration_hash="a" * 64,
    )
    first = store.register(session, {"spec": {}}, "first-fingerprint")
    assert store.register(session, {"spec": {}}, "changed-engine")["id"] == first["id"]
    with pytest.raises(ValueError, match="different intent"):
        store.register(dict(session, registration_hash="b" * 64), {}, "second")
    assert (
        store.registration_operation(session, session["registration_key"], "a" * 64)["id"]
        == first["id"]
    )


def test_real_alpha_collection_recovery_and_submission_share_one_audit(tmp_path, monkeypatch):
    ops, session, proposal, _, _ = setup(tmp_path)
    from src.researchops import alpha_output_audit
    from src.researchops.alpha_execution import verify_completed

    original = alpha_output_audit.audit_output
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(alpha_output_audit, "audit_output", counted)
    ex = ops.register(session, proposal)
    done = ops.execute(session, ex["id"])
    assert done["status"] == "completed" and calls == [1]
    verify_completed(ops, done)
    ops.recover(done["id"])
    ops.submit(
        session,
        dict(
            summary="engineering fixture",
            findings=["shared audit"],
            limitations=["not strategy performance"],
            next_steps=["review"],
        ),
    )
    assert calls == [1]


def test_audit_cache_invalidates_for_input_output_code_and_environment(tmp_path, monkeypatch):
    project = tmp_path / "app"
    project.mkdir()
    ops = Research(project, tmp_path / "store")
    (project / "research").mkdir()
    inp = project / "research/input.json"
    inp.write_text("first")
    job = ops.root / "worker_jobs/ex-fixture"
    for name in ("code", "alpha_result"):
        (job / name).mkdir(parents=True)
    code = job / "code/worker.py"
    code.write_text("original")
    output = job / "alpha_result/values.json"
    output.write_text("original")
    ex = dict(
        id="ex-fixture",
        proposal=dict(kind="alpha_batch", inputs=[dict(path="research/input.json")]),
    )
    calls = []

    def audit(*args):
        calls.append(1)
        return {"verified": True, "generation": len(calls)}

    assert numerical_audit(ops, ex, audit)["generation"] == 1
    assert numerical_audit(ops, ex, audit)["generation"] == 1
    for path in (inp, output, code):
        path.write_text(path.read_text() + "changed")
        numerical_audit(ops, ex, audit)
    assert len(calls) == 4
    monkeypatch.setattr(
        "src.researchops.audit_cache.importlib.metadata.version", lambda n: "changed"
    )
    numerical_audit(ops, ex, audit)
    assert len(calls) == 5
    numerical_audit(ops, ex, audit, force=True)
    assert len(calls) == 6


def test_cache_refuses_evidence_changed_during_audit(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    ops = Research(project, tmp_path / "store")
    job = ops.root / "worker_jobs/ex-fixture"
    for name in ("code", "alpha_result"):
        (job / name).mkdir(parents=True)
    output = job / "alpha_result/values.json"
    output.write_text("original")
    ex = dict(id="ex-fixture", proposal=dict(kind="alpha_batch", inputs=[]))

    def mutating(*args):
        output.write_text("changed")
        return {"verified": True}

    with pytest.raises(ValueError, match="changed during"):
        numerical_audit(ops, ex, mutating)
    with ops.store.connection() as db:
        assert not db.execute("SELECT 1 FROM execution_audits").fetchone()


def test_dependency_cycle_and_forward_reference_are_rejected():
    p = dict(
        schema="execution-plan-v1",
        id="x",
        task_id="task-x",
        max_starts=1,
        timeout_seconds=60,
        steps=[dict(id="a", needs=["b"], proposal={}), dict(id="b", needs=["a"], proposal={})],
    )
    with pytest.raises(ValueError, match="Dependencies"):
        validate_plan(p)


def test_one_step_calls_advance_beyond_completed_prefix(research):
    ops, session, proposal = research
    second = copy.deepcopy(proposal)
    second["spec"]["strategy"]["lookback"] = 4
    plan = plan_for(
        session, [proposal_step(proposal, "one"), proposal_step(second, "two", ["one"])]
    )
    plans = ExecutionPlans(ops)
    first = plans.run(session, plan, max_steps=1)
    assert first["steps"][0]["status"] == "completed"
    assert first["steps"][1]["status"] == "pending"
    final = plans.run(session, plan, max_steps=1)
    assert final["completed"] and final["starts"] == 2


def test_readonly_cache_hit_and_miss_never_write(tmp_path):
    project = tmp_path / "app"
    project.mkdir()
    ops = Research(project, tmp_path / "store")
    job = ops.root / "worker_jobs/ex-fixture"
    for name in ("code", "alpha_result"):
        (job / name).mkdir(parents=True)
    output = job / "alpha_result/values.json"
    output.write_text("original")
    ex = dict(id="ex-fixture", proposal=dict(kind="alpha_batch", inputs=[]))
    calls = []

    def audit(*args):
        calls.append(1)
        return {"verified": True}

    numerical_audit(ops, ex, audit)
    reader = Research(project, ops.root, readonly=True)
    numerical_audit(reader, ex, audit)
    assert len(calls) == 1
    output.write_text("changed")
    numerical_audit(reader, ex, audit)
    assert len(calls) == 2
    with ops.store.connection(True) as db:
        db.execute("DROP TABLE execution_audits")
    numerical_audit(reader, ex, audit)
    assert len(calls) == 3
    with ops.store.connection() as db:
        assert not db.execute(
            "SELECT 1 FROM sqlite_master WHERE name='execution_audits'"
        ).fetchone()


def test_cache_hit_rejects_mutation_after_first_binding(tmp_path, monkeypatch):
    project = tmp_path / "app"
    project.mkdir()
    ops = Research(project, tmp_path / "store")
    job = ops.root / "worker_jobs/ex-fixture"
    for name in ("code", "alpha_result"):
        (job / name).mkdir(parents=True)
    output = job / "alpha_result/values.json"
    output.write_text("original")
    ex = dict(id="ex-fixture", proposal=dict(kind="alpha_batch", inputs=[]))

    def audit(*args):
        return {"verified": True}

    numerical_audit(ops, ex, audit)
    from src.researchops import audit_cache

    original = audit_cache._binding
    calls = []

    def mutate_after_binding(*args, **kwargs):
        value = original(*args, **kwargs)
        calls.append(1)
        if len(calls) == 1:
            output.write_text("mutated after first digest")
        return value

    monkeypatch.setattr(audit_cache, "_binding", mutate_after_binding)
    with pytest.raises(ValueError, match="changed during"):
        numerical_audit(ops, ex, audit)


def test_retirement_keeps_active_scripts_and_preserves_original_bytes(tmp_path, monkeypatch):
    from tools import retire_stage_scripts

    app = tmp_path / "app"
    folder = app / "research/autonomous_delivery_test"
    folder.mkdir(parents=True)
    originals = {"old_prepare.py": b"print('prepare')\n", "old_close.py": b"print('close')\n"}
    for name, data in originals.items():
        (folder / name).write_bytes(data)
    (folder / "active.py").write_text("active")
    (folder / "r32sm_repair.py").write_text("queued repair")
    (folder / "CHECKPOINT.json").write_text(json.dumps({"script": "active.py"}))
    monkeypatch.setattr(retire_stage_scripts, "APP", app)
    assert retire_stage_scripts.retire(folder)["candidates"] == 2
    result = retire_stage_scripts.retire(folder, apply=True)
    assert result["retired"] == 2
    assert (folder / "active.py").exists() and (folder / "r32sm_repair.py").exists()
    import zipfile

    with zipfile.ZipFile(result["archive"]) as archive:
        for name, data in originals.items():
            assert archive.read(name) == data and not (folder / name).exists()
    with pytest.raises(ValueError, match="already exists"):
        retire_stage_scripts.retire(folder, apply=True)


def test_cached_audit_does_not_hide_unregistered_bytecode(tmp_path):
    from src.researchops.alpha_execution import verify_completed

    ops, session, proposal, _, _ = setup(tmp_path)
    ex = ops.register(session, proposal)
    completed = ops.execute(session, ex["id"])
    job = ops.root / "worker_jobs" / ex["id"]
    for relative in ("injected.pyc", "__pycache__/injected.pyc"):
        extra = job / "code" / relative
        extra.parent.mkdir(exist_ok=True)
        extra.write_bytes(b"unregistered")
        try:
            with pytest.raises(ValueError):
                verify_completed(ops, completed)
        finally:
            extra.unlink()
            if extra.parent.name == "__pycache__":
                extra.parent.rmdir()
    assert verify_completed(ops, completed)["verified"]


def test_atomic_json_short_temporary_name_supports_long_parent(tmp_path):
    from src.technical.artifacts import write_json

    folder = tmp_path
    while len(str(folder)) < 215:
        folder = folder / ("nested" * 3)
    path = folder / "campaign.json"
    assert len(str(path)) < 260
    write_json(path, {"completed": True})
    assert json.loads(path.read_text()) == {"completed": True}
    assert list(folder.iterdir()) == [path]


def test_atomic_json_failed_replace_preserves_original_and_removes_temp(tmp_path, monkeypatch):
    from src.technical.artifacts import write_json

    path = tmp_path / "original.json"
    write_json(path, {"original": True})
    original = path.read_bytes()

    def fail(*args):
        raise OSError("replace failed")

    monkeypatch.setattr("src.technical.artifacts.os.replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        write_json(path, {"new": True})
    assert path.read_bytes() == original and list(tmp_path.iterdir()) == [path]


def test_derived_registration_survives_interruption_without_rebuilding(research,monkeypatch):
    # Coordination fixture uses the existing tiny native-account executor. The
    # real saved-learning identity builder is tested separately, with no fits.
    from src.researchops import account_proposal
    from tests.test_account_proposal import example_template
    ops,session,proposal=research
    source=ops.register(session,proposal);source=ops.execute(session,source['id'])
    template=example_template(ops)
    plan=plan_for(session,[dict(id='learn',needs=[],experiment_id=source['id']),
        dict(id='account',needs=['learn'],account_from_learning=dict(learning_step='learn',template=template))])
    plan['schema']='execution-plan-v2'
    derived=copy.deepcopy(proposal);derived['spec']['strategy']['top_k']+=1
    calls=[]
    def build(research,eid,value):
        assert eid==source['id'] and value==template
        calls.append(eid);return derived
    monkeypatch.setattr(account_proposal,'account_from_learning',build)
    plans=ExecutionPlans(ops);original=plans._save
    def crash(*args,**kwargs):
        if args[2]=='account' and kwargs.get('status')=='registered':raise SystemExit('registered before checkpoint')
        return original(*args,**kwargs)
    monkeypatch.setattr(plans,'_save',crash)
    with pytest.raises(SystemExit):plans.run(session,plan)
    assert calls==[source['id']]
    assert len(ops.store.get(session['task_id'])['experiments'])==2
    monkeypatch.setattr(plans,'_save',original)
    monkeypatch.setattr(account_proposal,'account_from_learning',lambda *a:pytest.fail('registered derivation must not rebuild'))
    final=plans.run(session,plan)
    assert final['completed'] and final['starts']==1
    again=plans.run(session,plan)
    assert again['completed'] and again['starts']==1
    assert len(ops.store.get(session['task_id'])['experiments'])==2
