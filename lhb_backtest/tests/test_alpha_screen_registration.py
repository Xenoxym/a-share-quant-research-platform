"""Durability/clock/semantic counterexamples, not newly discovered trading strategies."""
import copy
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from src.researchops.screen_experiments import assembly_reference, verify_registered
from src.researchops.screen_output_audit import audit_output
from src.researchops.service import Research
from src.technical.artifacts import content_id, digest, write_json
from tests.test_alpha_screening import example
from tests.test_researchops import TASK

PROJECT = Path(__file__).resolve().parents[1]


def setup(tmp_path, max_experiments=3, intent_budget=12):
    project = tmp_path / "project"; project.mkdir()
    for source in (PROJECT / "src").rglob("*.py"):
        if "__pycache__" in source.parts:
            continue
        dest = project / source.relative_to(PROJECT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    folder = project / "data/screen_input"; folder.mkdir(parents=True)
    spec, decisions, assembly, labels = example()
    paths = {}
    for role, frame in (("assembly_values", assembly.block.values),
                        ("assembly_missing", assembly.block.missing), ("decisions", decisions), ("labels", labels)):
        paths[role] = folder / (role + ".parquet"); frame.to_parquet(paths[role], index=False)
    paths["registry"] = folder / "registry.json"; write_json(paths["registry"], assembly.registry.to_dict())
    paths["assembly_definition"] = folder / "definition.json"
    write_json(paths["assembly_definition"], dict(assembly.block.metadata, units=assembly.block.units))
    paths["assembly_receipt"] = folder / "receipt.json"
    write_json(paths["assembly_receipt"], assembly_reference(assembly,
        {r: digest(paths[r]) for r in ("assembly_values", "assembly_missing", "assembly_definition")}))
    inputs = [dict(role=r, path=p.relative_to(project).as_posix(), sha256=digest(p), bytes=p.stat().st_size)
              for r, p in paths.items()]
    proposal = dict(kind="alpha_screen", hypothesis="Synthetic mechanics", expected_observation="Saved XOR selections",
        falsification="Any clock/selection/recovery mismatch", inputs=inputs, timeout_seconds=30,
        spec=dict(schema="registered-alpha-screen-v1", screen=spec.to_dict(),
            budget=dict(max_input_bytes=1000000, max_task_channel_intents=intent_budget)))
    ops = Research(project); task = ops.create(dict(TASK, max_experiments=max_experiments))
    session = ops.store.claim(task["id"], "screen-test-worker")
    return ops, session, proposal, folder


def test_register_full_intents_and_duplicate_is_original(tmp_path):
    ops, session, p, _ = setup(tmp_path)
    first = ops.register(session, p); q = copy.deepcopy(p)
    q["hypothesis"] = "New wording"; q["inputs"].reverse(); q["spec"]["screen"]["name"] = "Renamed"
    q["spec"]["screen"]["channels"].reverse()
    for c in q["spec"]["screen"]["channels"]:
        c["hypothesis"] = "Descriptive replacement"
    assert ops.register(session, q)["id"] == first["id"]
    check = verify_registered(ops, first)
    assert check["candidate_count"] == 4 and check["fits"] == check["accounts"] == 0
    assert first["proposal"]["inspection"]["development_decisions"] == 12
    assert len(Research(ops.project).store.get(session["task_id"])["experiments"]) == 1


def test_intents_survive_failed_cancelled_restart_and_budget_cannot_reset(tmp_path):
    ops, session, p, _ = setup(tmp_path, intent_budget=8)
    first = ops.register(session, p); ops.store.start(session, first["id"])
    ops.store.finish(first["id"], "failed", error="Retained fixture failure")
    q = copy.deepcopy(p); q["spec"]["screen"]["channels"][0]["direction"] = -1
    second = ops.register(session, q); ops.cancel(session, second["id"], "Retained fixture cancellation")
    restarted = Research(ops.project); third = copy.deepcopy(p)
    third["spec"]["screen"]["top_k"] = 2
    with pytest.raises(ValueError, match="intent budget exhausted"):
        restarted.register(session, third)
    third["spec"]["budget"]["max_task_channel_intents"] = 12
    with pytest.raises(ValueError, match="budget already frozen"):
        restarted.register(session, third)
    assert [e["status"] for e in restarted.store.get(session["task_id"])["experiments"]] == ["failed", "cancelled"]


@pytest.mark.parametrize("role", ["labels", "decisions", "assembly_values", "registry"])
def test_reference_damage_rejected_before_running(tmp_path, role):
    ops, session, p, _ = setup(tmp_path); ex = ops.register(session, p)
    record = next(r for r in p["inputs"] if r["role"] == role)
    path = ops.project / record["path"]; path.write_bytes(path.read_bytes() + b"damaged")
    with pytest.raises(ValueError, match="Frozen input content changed"):
        ops.execute(session, ex["id"])
    assert ops.store.experiment(ex["id"])["status"] == "planned"


def test_wrong_clock_never_creates_registered_job(tmp_path):
    ops, session, p, folder = setup(tmp_path)
    path = folder / "decisions.parquet"; frame = pd.read_parquet(path)
    frame.loc[0, "execution_at"] = frame.loc[0, "decision_at"]; frame.to_parquet(path, index=False)
    for record in p["inputs"]:
        if record["role"] == "decisions":
            record.update(sha256=digest(path), bytes=path.stat().st_size)
    with pytest.raises(ValueError, match="Execution/target clocks"):
        ops.register(session, p)
    assert not ops.store.get(session["task_id"])["experiments"]


def test_completed_recovery_full_tables_hand_result_and_no_rerun(tmp_path, monkeypatch):
    ops, session, p, _ = setup(tmp_path); ex = ops.register(session, p)
    completed = ops.execute(session, ex["id"])
    assert completed["status"] == "completed" and completed["result"]["kind"] == "registered_alpha_screen_execution"
    job = ops.root / "worker_jobs" / ex["id"]; folder = job / "screen_result"
    trial = {row["channel_id"]: row for row in json.loads((folder / "trials.json").read_text())["rows"]}
    assert trial["interaction_merit"]["proxy_total_gross_return"] == pytest.approx(1.1**3 - 1)
    selections = json.loads((folder / "selections.json").read_text())["rows"]
    assert {r["stock_code"] for r in selections if r["channel_id"] == "interaction_merit"} == {"600000.SH"}
    assert not completed["result"]["account_results"]
    original = {p.name: digest(p) for p in job.iterdir() if p.is_file()}
    # A current pure algorithm edit must not substitute for the frozen replay.
    import src.alpharesearch.screening as current
    monkeypatch.setattr(current, "screen", lambda *a: (_ for _ in ()).throw(AssertionError("Current algorithm used")))
    assert audit_output(ops, completed)["verified"]
    assert Research(ops.project).recover(ex["id"])["status"] == "completed"
    assert original == {p.name: digest(p) for p in job.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="already started"):
        ops.execute(session, ex["id"])
    # Updating a manifest alone cannot bless a altered numeric result.
    data = json.loads((folder / "selections.json").read_text()); data["rows"][0]["score"] = 999.
    write_json(folder / "selections.json", data)
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["artifacts"]["selections.json"] = digest(folder / "selections.json"); write_json(folder / "manifest.json", manifest)
    with pytest.raises(ValueError, match="replay differs: selections"):
        audit_output(ops, completed)


def test_unknown_kind_cannot_pick_module_and_current_compute_refused(tmp_path):
    ops, session, p, _ = setup(tmp_path); p["kind"] = "screen_worker_hacked"
    with pytest.raises(ValueError):
        ops.register(session, p)
    p["kind"] = "alpha_screen"; ex = ops.register(session, p)
    from src.researchops.screen_worker import compute
    with pytest.raises(ValueError, match="registered frozen modules"):
        compute(ops.root / "worker_jobs" / ex["id"])


@pytest.mark.parametrize("field, value", [("max_input_bytes", True), ("max_task_channel_intents", 0)])
def test_invalid_budget_is_not_registered(tmp_path, field, value):
    ops, session, p, _ = setup(tmp_path); p["spec"]["budget"][field] = value
    with pytest.raises(ValueError, match="budget"):
        ops.register(session, p)
    assert not ops.store.get(session["task_id"])["experiments"]
def test_pending_collector_recovered_by_new_instance_without_execution(tmp_path, monkeypatch):
    from src.researchops import alpha_execution
    from src.researchops.resources import GuardBusy
    ops, session, p, _ = setup(tmp_path); ex = ops.register(session, p)
    original = alpha_execution._record_observation
    def busy(*args):
        raise GuardBusy("Fixture collector busy")
    monkeypatch.setattr(alpha_execution, "_record_observation", busy)
    pending = ops.execute(session, ex["id"])
    assert pending["status"] == "running" and "collection pending" in pending["recovery_note"]
    job = ops.root / "worker_jobs" / ex["id"]
    receipt_hash = digest(job / "receipt.json"); launch_hash = digest(job / "launch.json")
    monkeypatch.setattr(alpha_execution, "_record_observation", original)
    completed = Research(ops.project).recover(ex["id"])
    assert completed["status"] == "completed"
    assert receipt_hash == digest(job / "receipt.json") and launch_hash == digest(job / "launch.json")
    stats = json.loads((job / "execution_stats.json").read_text())
    assert not stats["parent_observed_to_exit"] and stats["peak_sampled_rss_bytes"] is None


def test_resource_denial_does_not_release_intent_budget(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.researchops import resources
    ops, session, p, _ = setup(tmp_path, intent_budget=4); ex = ops.register(session, p)
    monkeypatch.setattr(resources.psutil, "virtual_memory", lambda: SimpleNamespace(available=0))
    with pytest.raises(RuntimeError, match="Insufficient unreserved"):
        ops.execute(session, ex["id"])
    assert ops.store.experiment(ex["id"])["status"] == "planned"
    ops.cancel(session, ex["id"], "Resource denial retained")
    q = copy.deepcopy(p); q["spec"]["screen"]["top_k"] = 2
    with pytest.raises(ValueError, match="intent budget exhausted"):
        ops.register(session, q)


def test_strict_receipt_type_and_unknown_dispatch(tmp_path):
    from src.researchops import alpha_execution
    ops, session, p, _ = setup(tmp_path); ex = ops.register(session, p)
    with pytest.raises(ValueError, match="ownership"):
        alpha_execution._receipt(ops, ex, dict(kind="alpha_batch", experiment_id=ex["id"], status="failed", error="bad"))
    ex["proposal"]["kind"] = "arbitrary_module"
    with pytest.raises(ValueError, match="Unsupported typed"):
        alpha_execution._protocol(ex)


def test_freeze_failure_is_persisted_and_never_implicitly_retried(tmp_path, monkeypatch):
    from src.researchops import screen_experiments
    ops, session, p, _ = setup(tmp_path, max_experiments=1)
    def failed(*args):
        raise OSError("Fixture disk failure")
    monkeypatch.setattr(screen_experiments.shutil, "copy2", failed)
    with pytest.raises(OSError, match="disk failure"):
        ops.register(session, p)
    recorded = ops.store.get(session["task_id"])["experiments"]
    assert len(recorded) == 1 and recorded[0]["status"] == "failed"
    with pytest.raises(ValueError, match="freeze incomplete"):
        ops.register(session, p)
    q = copy.deepcopy(p); q["spec"]["screen"]["top_k"] = 2
    with pytest.raises(ValueError, match="任务实验预算"):
        ops.register(session, q)


@pytest.mark.parametrize("mode", ["unit", "reference", "source_late"])
def test_assembly_identity_and_clock_reject_before_registration(tmp_path, mode):
    ops, session, p, folder = setup(tmp_path)
    if mode == "reference":
        path = folder / "receipt.json"; obj = json.loads(path.read_text()); obj["rows"] += 1; write_json(path, obj)
    elif mode == "unit":
        path = folder / "definition.json"; obj = json.loads(path.read_text()); obj["units"]["test.first"] = "currency"; write_json(path, obj)
    else:
        path = folder / "assembly_values.parquet"; frame = pd.read_parquet(path)
        frame.loc[0, "known_at"] = "2024-01-02T09:30:00+08:00"; frame.to_parquet(path, index=False)
    for rec in p["inputs"]:
        if ops.project / rec["path"] == path:
            rec.update(sha256=digest(path), bytes=path.stat().st_size)
    if mode != "reference":
        receipt_path = folder / "receipt.json"
        receipt = json.loads(receipt_path.read_text()); receipt.pop("reference_id")
        for role in receipt["artifacts"]:
            rec = next(r for r in p["inputs"] if r["role"] == role)
            receipt["artifacts"][role] = digest(ops.project / rec["path"])
        write_json(receipt_path, dict(receipt, reference_id=content_id(receipt)))
        rec = next(r for r in p["inputs"] if r["role"] == "assembly_receipt")
        rec.update(sha256=digest(receipt_path), bytes=receipt_path.stat().st_size)
    with pytest.raises(ValueError, match="receipt differs|unit and causal|not known|not a valid Unit"):
        ops.register(session, p)
    assert not ops.store.get(session["task_id"])["experiments"]
@pytest.fixture(scope="module")
def resealed_completed(tmp_path_factory):
    ops, session, p, _ = setup(tmp_path_factory.mktemp("screen_resealed"))
    ex = ops.register(session, p)
    return ops, ops.execute(session, ex["id"])


@pytest.mark.parametrize("mutation", ["weight_bool", "rank_bool", "quality_number", "receipt_number", "attempts_float"])
def test_fully_resealed_json_type_changes_are_rejected(resealed_completed, mutation):
    ops, ex = resealed_completed; job = ops.root / "worker_jobs" / ex["id"]; folder = job / "screen_result"
    name = {"weight_bool":"selections.json", "rank_bool":"selections.json", "quality_number":"trials.json",
            "receipt_number":"diagnostic_receipt.json", "attempts_float":"attempts.json"}[mutation]
    paths = [folder/name, folder/"manifest.json", job/"receipt.json"]
    originals = {path:path.read_bytes() for path in paths}
    try:
        obj = json.loads(paths[0].read_text())
        if mutation == "weight_bool": obj["rows"][0]["weight"] = True
        if mutation == "rank_bool": obj["rows"][0]["rank"] = True
        if mutation == "quality_number": obj["rows"][0]["quality_passed"] = 1
        if mutation == "receipt_number": obj["durable_executor"] = 0
        if mutation == "attempts_float": obj["candidate_count"] = 4.0
        write_json(paths[0], obj)
        manifest = json.loads(paths[1].read_text()); manifest["artifacts"][name] = digest(paths[0]); write_json(paths[1], manifest)
        receipt = json.loads(paths[2].read_text()); receipt["result_manifest_hash"] = digest(paths[1]); write_json(paths[2], receipt)
        with pytest.raises(ValueError, match="replay differs|diagnostic receipt differs|intent ledger differs"):
            audit_output(ops, ex)
    finally:
        for path, raw in originals.items(): path.write_bytes(raw)


def test_resealed_assembly_counter_type_is_rejected(tmp_path):
    ops, session, p, folder = setup(tmp_path); path=folder/"receipt.json"
    obj=json.loads(path.read_text()); obj.pop("reference_id"); obj["rows"]=float(obj["rows"])
    write_json(path, dict(obj, reference_id=content_id(obj)))
    rec=next(r for r in p["inputs"] if r["role"]=="assembly_receipt")
    rec.update(sha256=digest(path), bytes=path.stat().st_size)
    with pytest.raises(ValueError, match="reference receipt differs"):
        ops.register(session, p)


def test_freeze_failure_before_job_exists_cannot_retry(tmp_path, monkeypatch):
    ops, session, p, _ = setup(tmp_path, max_experiments=1); original=Path.mkdir
    def early_failure(path, *args, **kwargs):
        if path.name=="code" and "worker_jobs" in path.parts:
            raise OSError("Fixture earliest directory failure")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "mkdir", early_failure)
    with pytest.raises(OSError, match="earliest"):
        ops.register(session, p)
    ex=ops.store.get(session["task_id"])["experiments"][0]; job=ops.root/"worker_jobs"/ex["id"]
    assert ex["status"]=="failed" and not job.exists()
    monkeypatch.setattr(Path, "mkdir", original)
    with pytest.raises(ValueError, match="freeze incomplete or failed"):
        ops.register(session, p)
    assert not job.exists() and ops.store.experiment(ex["id"])["status"]=="failed"
@pytest.mark.parametrize("field, replacement", [("candidate_count", 4.0), ("verified", 1)])
def test_pending_commit_audit_type_change_cannot_complete(tmp_path, monkeypatch, field, replacement):
    ops, session, p, _ = setup(tmp_path); ex=ops.register(session, p); original_finish=ops.store.finish
    def pending(eid, status, **fields):
        if status=="completed":
            raise OSError("Fixture terminal database temporarily unavailable")
        return original_finish(eid, status, **fields)
    monkeypatch.setattr(ops.store, "finish", pending)
    pending_result = ops.execute(session, ex["id"])
    assert pending_result["status"] == "running" and "collection pending" in pending_result["recovery_note"]
    assert ops.store.experiment(ex["id"])["status"]=="running"
    job=ops.root/"worker_jobs"/ex["id"]; path=job/"alpha_audit.json"
    value=json.loads(path.read_text()); value[field]=replacement; write_json(path, value)
    damaged=path.read_bytes(); receipt_hash=digest(job/"receipt.json")
    monkeypatch.setattr(ops.store, "finish", original_finish)
    with pytest.raises(ValueError, match="Existing execution metadata differs"):
        Research(ops.project).recover(ex["id"])
    assert ops.store.experiment(ex["id"])["status"]=="failed"
    assert path.read_bytes()==damaged and receipt_hash==digest(job/"receipt.json")
    assert len(ops.store.get(session["task_id"])["experiments"])==1
