"""Bounded declarative execution over the existing Research workers.

No model calls, shell templates, new experiment database, or automatic retries.
The database records registration requests atomically with their experiment.
"""

import json
import re
import time

from ..technical.artifacts import content_id, digest
from ..technical.runner import verify
from .resources import _guard
from .service import Research
from .store import encode

NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")
ALPHA = {"alpha_batch", "alpha_screen", "alpha_learn", "alpha_account"}


def validate_plan(value):
    required = {"schema", "id", "task_id", "steps", "max_starts", "timeout_seconds"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Strict execution plan fields required")
    if (
        value["schema"] != "execution-plan-v1"
        or not isinstance(value["id"], str)
        or not NAME.fullmatch(value["id"])
    ):
        raise ValueError("Invalid plan schema or identifier")
    if not isinstance(value["task_id"], str) or not NAME.fullmatch(value["task_id"]):
        raise ValueError("Invalid task identifier")
    if type(value["max_starts"]) is not int or not 0 <= value["max_starts"] <= 40:
        raise ValueError("Plan start budget must be 0..40")
    if type(value["timeout_seconds"]) is not int or not 30 <= value["timeout_seconds"] <= 86400:
        raise ValueError("Plan lifetime budget must be 30..86400 seconds")
    steps = value["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 40:
        raise ValueError("A plan needs 1..40 ordered steps")
    seen = set()
    for step in steps:
        if not isinstance(step, dict) or set(step) not in (
            {"id", "needs", "proposal"},
            {"id", "needs", "experiment_id"},
        ):
            raise ValueError("Step needs exactly one proposal or experiment reference")
        if not isinstance(step["id"], str) or not NAME.fullmatch(step["id"]) or step["id"] in seen:
            raise ValueError("Unique valid step names required")
        needs = step["needs"]
        if not isinstance(needs, list) or len(set(needs)) != len(needs) or not set(needs) <= seen:
            raise ValueError("Dependencies must name earlier steps, without duplicates")
        if "proposal" in step and not isinstance(step["proposal"], dict):
            raise ValueError("Inline experiment proposal required")
        if "experiment_id" in step and (
            not isinstance(step["experiment_id"], str) or not NAME.fullmatch(step["experiment_id"])
        ):
            raise ValueError("Valid experiment reference required")
        seen.add(step["id"])
    return json.loads(encode(value))


class ExecutionPlans:
    def __init__(self, research):
        if not isinstance(research, Research):
            raise TypeError("Existing Research interface required")
        self.research = research
        self.store = research.store
        if self.store.readonly:
            return
        with self.store.connection(True) as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS execution_plans(
                id TEXT PRIMARY KEY, task_id TEXT NOT NULL, definition TEXT NOT NULL,
                definition_hash TEXT NOT NULL, began REAL NOT NULL, deadline REAL NOT NULL,
                starts INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS execution_steps(
                plan_id TEXT NOT NULL, name TEXT NOT NULL, experiment_id TEXT,
                dispatched INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending',
                error TEXT, seconds REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(plan_id,name));
            """)

    def _bind(self, session, plan):
        now = time.time()
        with self.store.connection(True) as db:
            self.store.owned(db, session)
            if plan["task_id"] != session["task_id"]:
                raise ValueError("Plan and claimed task differ")
            old = db.execute("SELECT * FROM execution_plans WHERE id=?", (plan["id"],)).fetchone()
            if old:
                if old["definition_hash"] != content_id(plan):
                    raise ValueError("Persisted plan is immutable; resume cannot reset its budget")
                return
            db.execute(
                "INSERT INTO execution_plans VALUES(?,?,?,?,?,?,0)",
                (
                    plan["id"],
                    plan["task_id"],
                    encode(plan),
                    content_id(plan),
                    now,
                    now + plan["timeout_seconds"],
                ),
            )
            db.executemany(
                "INSERT INTO execution_steps(plan_id,name) VALUES(?,?)",
                [(plan["id"], s["id"]) for s in plan["steps"]],
            )
            self.store.event(
                db,
                plan["task_id"],
                session["worker"],
                "execution_plan_bound",
                {"plan_id": plan["id"], "definition_hash": content_id(plan)},
            )

    def status(self, plan_id):
        if not isinstance(plan_id, str) or not NAME.fullmatch(plan_id):
            raise ValueError("Invalid plan identifier")
        with self.store.connection() as db:
            row = db.execute("SELECT * FROM execution_plans WHERE id=?", (plan_id,)).fetchone()
            if row is None:
                raise ValueError("Execution plan not found")
            definition = json.loads(row["definition"])
            steps = {
                r["name"]: dict(r)
                for r in db.execute("SELECT * FROM execution_steps WHERE plan_id=?", (plan_id,))
            }
        return dict(
            plan_id=plan_id,
            task_id=row["task_id"],
            starts=row["starts"],
            max_starts=definition["max_starts"],
            began=row["began"],
            deadline=row["deadline"],
            steps=[
                {
                    k: steps[s["id"]][k]
                    for k in ("name", "experiment_id", "dispatched", "status", "error", "seconds")
                }
                for s in definition["steps"]
            ],
            usage={"llm_calls": 0, "desktop_tokens": None},
        )

    def _save(self, session, plan_id, name, *, eid=None, status, error=None, seconds=0):
        with self.store.connection(True) as db:
            self.store.owned(db, session)
            db.execute(
                """UPDATE execution_steps SET experiment_id=COALESCE(?,experiment_id),
                       status=?,error=?,seconds=seconds+? WHERE plan_id=? AND name=?""",
                (eid, status, error, seconds, plan_id, name),
            )
            self.store.event(
                db,
                session["task_id"],
                session["worker"],
                "execution_step",
                dict(
                    plan_id=plan_id,
                    step=name,
                    experiment_id=eid,
                    status=status,
                    error=error,
                    seconds=seconds,
                ),
            )

    def _budget(self, session, plan, name, dispatch=False, timeout=0):
        with self.store.connection(True) as db:
            self.store.owned(db, session)
            row = db.execute("SELECT * FROM execution_plans WHERE id=?", (plan["id"],)).fetchone()
            step = db.execute(
                "SELECT * FROM execution_steps WHERE plan_id=? AND name=?", (plan["id"], name)
            ).fetchone()
            if time.time() + timeout >= row["deadline"]:
                raise ValueError(
                    "Persisted plan deadline cannot cover this worker timeout; no new work"
                )
            if not step["dispatched"] and row["starts"] >= plan["max_starts"]:
                raise ValueError("Persisted plan start budget exhausted")
            if dispatch and not step["dispatched"]:
                db.execute(
                    "UPDATE execution_steps SET dispatched=1 WHERE plan_id=? AND name=?",
                    (plan["id"], name),
                )
                db.execute("UPDATE execution_plans SET starts=starts+1 WHERE id=?", (plan["id"],))

    def _verify(self, ex):
        if ex["status"] != "completed" or not ex.get("result", {}).get("verified"):
            raise ValueError("Completed verified experiment required")
        kind = ex["proposal"].get("kind")
        if kind in ALPHA:
            from .alpha_execution import verify_completed

            verify_completed(self.research, ex)
        elif kind == "ml":
            from .ml_experiments import verify_completed

            verify_completed(self.research, ex)
        else:
            folder = self.research.run_folder(ex["run_id"])
            if digest(folder / "manifest.json") != ex["result"]["manifest_hash"]:
                raise ValueError("Registered result manifest changed")
            verify(folder)
            job = self.research.root / "worker_jobs" / ex["id"]
            if digest(job / "account_audit.json") != ex["result"]["audit_hash"]:
                raise ValueError("Original account audit changed")

    def run(self, session, definition, *, max_steps=40):
        plan = validate_plan(definition)
        if type(max_steps) is not int or not 1 <= max_steps <= 40:
            raise ValueError("Per-call step limit must be 1..40")
        folder = self.research.root / "execution_plans" / plan["id"]
        if folder.is_symlink() or not folder.resolve().is_relative_to(self.research.root):
            raise ValueError("Plan lock path escaped research root")
        folder.mkdir(parents=True, exist_ok=True)
        # Kernel lock releases on parent death. It never resets persisted budgets.
        with _guard(folder):
            self._bind(session, plan)
            visited = 0
            for step in plan["steps"]:
                started = time.monotonic()
                state = {s["name"]: s for s in self.status(plan["id"])["steps"]}
                current = state[step["id"]]
                if current["status"] != "completed":
                    if visited >= max_steps:
                        break
                    visited += 1
                eid = current["experiment_id"] or step.get("experiment_id")
                try:
                    self.store.heartbeat(session)
                    if any(state[n]["status"] != "completed" for n in step["needs"]):
                        raise ValueError("Dependency has not completed verification")
                    if not eid:
                        operation = f"plan:{plan['id']}:{step['id']}"
                        request_hash = content_id(step["proposal"])
                        registered = self.store.registration_operation(
                            session, operation, request_hash
                        )
                        if registered:
                            eid = registered["id"]
                        else:
                            self._budget(session, plan, step["id"])
                            registered = self.research.register(
                                dict(
                                    session,
                                    registration_key=operation,
                                    registration_hash=request_hash,
                                ),
                                step["proposal"],
                            )
                            eid = registered["id"]
                        self._save(session, plan["id"], step["id"], eid=eid, status="registered")
                    ex = self.store.experiment(eid)
                    if ex["task_id"] != session["task_id"] and ex["status"] != "completed":
                        raise ValueError(
                            "Only completed experiments from other tasks may be reused"
                        )
                    if ex["status"] == "completed":
                        self._verify(ex)
                        action = "reused"
                    elif ex["status"] == "running":
                        ex = self.research.recover(eid)
                        action = "recovered"
                        if ex["status"] != "completed":
                            raise ValueError(
                                ex.get("recovery_note")
                                or ex.get("error")
                                or "Original attempt remains unfinished; no automatic restart"
                            )
                    elif ex["status"] == "planned":
                        self._budget(
                            session,
                            plan,
                            step["id"],
                            dispatch=True,
                            timeout=ex["proposal"]["timeout_seconds"],
                        )
                        ex = self.research.execute(session, eid)
                        action = "executed"
                        if ex["status"] != "completed":
                            raise ValueError(ex.get("error") or "Worker did not complete")
                    else:
                        raise ValueError("Retained failed/cancelled attempt; no automatic retry")
                    self._save(
                        session,
                        plan["id"],
                        step["id"],
                        eid=eid,
                        status="completed",
                        seconds=time.monotonic() - started,
                    )
                    self.store.checkpoint(
                        session, f"Plan {plan['id']}: {step['id']} {action}; {eid}"
                    )
                except Exception as exc:
                    self._save(
                        session,
                        plan["id"],
                        step["id"],
                        eid=eid,
                        status="blocked",
                        error=str(exc),
                        seconds=time.monotonic() - started,
                    )
                    break
            out = self.status(plan["id"])
            out["completed"] = all(s["status"] == "completed" for s in out["steps"])
            for s in out["steps"]:
                if s["experiment_id"]:
                    s["log_directory"] = str(
                        self.research.root / "worker_jobs" / s["experiment_id"]
                    )
            return out
