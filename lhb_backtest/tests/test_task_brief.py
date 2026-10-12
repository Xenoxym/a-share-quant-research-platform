"""Task briefs stay bounded, read-only and separate from scientific acceptance."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.researchops.service import Research
from src.researchops.store import Store, encode
from tests.test_researchops import TASK


def populated(tmp_path):
    store = Store(tmp_path / "research")
    task = store.create({**TASK, "max_experiments": 4})
    session = store.claim(task["id"], "engineer")
    with store.connection(True) as db:
        for i, status in enumerate(["complete", "failed", "registered"]):
            db.execute(
                "INSERT INTO experiments(id,task_id,fingerprint,proposal,status,created,updated,"
                "run_id,result,error) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (f"exp-{i}", task["id"], str(i), encode({"unused": "x" * 10000}),
                 status, i, i, f"run-{i}", encode({"huge": "x" * 10000}) if i == 0 else None,
                 "failure" if i == 1 else None),
            )
    return store, task["id"], session


def test_brief_counts_failures_without_writes_or_loading_full_history(tmp_path, monkeypatch):
    store, task, session = populated(tmp_path)
    ops = Research(tmp_path, store.root, readonly=True)
    before = store.get(task)

    def forbidden(*args, **kwargs):
        raise AssertionError("Brief must not load catalog or the complete task history")

    monkeypatch.setattr(ops, "catalog", forbidden)
    monkeypatch.setattr(ops.store, "get", forbidden)
    value = ops.brief(task)
    assert value["budget"] == {
        "max_registered_experiments": 4, "registered_experiments": 3, "unregistered_slots": 1,
    }
    assert value["experiment_counts"] == {"complete": 1, "failed": 1, "registered": 1}
    assert [x["id"] for x in value["latest_experiments"]] == ["exp-2", "exp-1", "exp-0"]
    assert value["latest_experiments"][-1]["has_result"] is True
    assert value["latest_experiments"][1]["error"] == "failure"
    assert store.get(task) == before
    output = json.dumps(value)
    assert session["token"] not in output and "token_hash" not in output
    assert "huge" not in output and "unused" not in output
    with pytest.raises(ValueError, match="任务不存在"):
        ops.brief("unknown")


def test_brief_marks_long_contract_as_partial_and_limits_recent_trials(tmp_path):
    store, task, _ = populated(tmp_path)
    with store.connection(True) as db:
        spec = json.loads(db.execute("SELECT spec FROM tasks WHERE id=?", (task,)).fetchone()[0])
        spec.update(question="长" * 8000, rationale="长" * 8000, scope="长" * 8000,
                    success_criteria=["长" * 8000] * 40, stop_criteria=["长" * 8000] * 40)
        db.execute("UPDATE tasks SET spec=? WHERE id=?", (encode(spec), task))
        db.execute("UPDATE experiments SET error=?", ("bad" * 8000,))
        for i in range(3, 10):
            db.execute(
                "INSERT INTO experiments(id,task_id,fingerprint,proposal,status,created,updated) "
                "VALUES(?,?,?,?,?,?,?)",
                (f"exp-{i}", task, str(i), "{}", "failed", i, i),
            )
    value = Research(tmp_path, store.root, readonly=True).brief(task)
    assert len(value["latest_experiments"]) == 3
    assert value["budget"]["registered_experiments"] == 10
    assert value["budget"]["unregistered_slots"] == 0
    assert {"question", "scope", "success_criteria", "stop_criteria"} <= set(value["truncated_fields"])
    assert len(json.dumps(value, ensure_ascii=False)) < 4000


def test_brief_cli_reads_existing_database_and_does_not_create_missing_one(tmp_path):
    store, task, _ = populated(tmp_path)
    cli = Path(__file__).resolve().parents[1] / "tools/execution.py"
    args = [sys.executable, str(cli), "--project", str(tmp_path), "--root", str(store.root),
            "brief", task]
    env = {**os.environ, "PYTHONUTF8": "1"}
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", env=env, check=True)
    assert json.loads(result.stdout)["data"]["task_id"] == task
    missing = tmp_path / "missing"
    args[args.index("--root") + 1] = str(missing)
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", env=env)
    assert result.returncode == 1
    assert not missing.exists()
    assert json.loads(result.stdout)["ok"] is False
