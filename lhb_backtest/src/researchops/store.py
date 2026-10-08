"""SQLite transaction boundaries, expiring claims and an append-only event journal.

Worker names are provenance labels, not authenticated identities. Claim secrets
fence stale sessions; they are never returned by read APIs or written to events.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import secrets
import sqlite3
import time
import uuid


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def text(value, name, limit=8000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name}必须为非空文本，最多{limit}字")
    return value.strip()


def items(value, name):
    if not isinstance(value, list) or not value or len(value) > 40:
        raise ValueError(f"{name}需要1至40个条目")
    return [text(v, name) for v in value]


def task_spec(value):
    required = {"title", "question", "rationale", "success_criteria", "stop_criteria"}
    allowed = required | {"max_experiments", "scope", "baseline_runs", "parent_id", "tags"}
    if not isinstance(value, dict) or not required <= value.keys() or value.keys() - allowed:
        raise ValueError("任务需要标题、问题、依据、成功与停止标准；不接受未知字段")
    out = {
        k: text(value[k], k, 200 if k == "title" else 8000)
        for k in ["title", "question", "rationale"]
    }
    out.update({k: items(value[k], k) for k in ["success_criteria", "stop_criteria"]})
    budget = value.get("max_experiments", 4)
    if type(budget) is not int or not 1 <= budget <= 40:
        raise ValueError("每个任务实验预算必须为1至40")
    out.update(
        max_experiments=budget,
        scope=text(
            value.get("scope", "仅修改策略、特征和研究配置；不修改公共成交与会计规则"), "scope"
        ),
    )
    for name in ["baseline_runs", "tags"]:
        val = value.get(name, [])
        if not isinstance(val, list) or len(val) > 40:
            raise ValueError(name + "必须为列表")
        out[name] = [text(v, name, 120) for v in val]
    out["parent_id"] = value.get("parent_id")
    if out["parent_id"] is not None:
        text(out["parent_id"], "parent_id", 80)
    out["evidence_scope"] = "retrospective"  # A fresh worker does not restore unseen history.
    return out


class Store:
    def __init__(self, root, readonly=False):
        self.root = Path(root).resolve()
        self.readonly = readonly
        self.path = self.root / "research.sqlite3"
        if readonly:
            if not self.path.is_file():
                raise ValueError("研究库不存在；只读操作不会初始化新库")
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, spec TEXT NOT NULL, status TEXT NOT NULL,
                owner TEXT, token_hash TEXT, lease_until REAL, created REAL NOT NULL, updated REAL NOT NULL,
                checkpoint TEXT, submission TEXT, review TEXT);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                at REAL NOT NULL, actor TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY, task_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                proposal TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                pid INTEGER, run_id TEXT, result TEXT, error TEXT);
            CREATE UNIQUE INDEX IF NOT EXISTS unique_trial ON experiments(task_id,fingerprint);
            CREATE TABLE IF NOT EXISTS knowledge(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL);
            CREATE UNIQUE INDEX IF NOT EXISTS unique_evidence ON evidence(task_id,fingerprint);
            PRAGMA user_version=1;
            """)

    @contextmanager
    def connection(self, write=False):
        if write and self.readonly:
            raise ValueError("只读研究库不可写入")
        db = sqlite3.connect(
            self.path.as_uri() + "?mode=ro" if self.readonly else self.path,
            timeout=30,
            uri=self.readonly,
        )
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def event(self, db, task, actor, kind, payload):
        db.execute(
            "INSERT INTO events(task_id,at,actor,kind,payload) VALUES(?,?,?,?,?)",
            (task, time.time(), actor, kind, encode(payload)),
        )

    @staticmethod
    def public(row):
        d = dict(row)
        for k in ["spec", "checkpoint", "submission", "review", "proposal", "result", "payload"]:
            if k in d and d[k] is not None:
                d[k] = json.loads(d[k])
        d.pop("token_hash", None)
        if "lease_until" in d:
            d["lease_expired"] = bool(d["lease_until"] and d["lease_until"] < time.time())
        return d

    def _task(self, db, task):
        row = db.execute("SELECT * FROM tasks WHERE id=?", (task,)).fetchone()
        if row is None:
            raise ValueError("任务不存在")
        return row

    def owned(self, db, session):
        if not isinstance(session, dict):
            raise ValueError("需要领取任务的session")
        row = self._task(db, session.get("task_id"))
        expected = hashlib.sha256(str(session.get("token", "")).encode()).hexdigest()
        if (
            row["status"] != "active"
            or row["owner"] != session.get("worker")
            or row["token_hash"] != expected
            or (row["lease_until"] or 0) < time.time()
        ):
            raise ValueError("领取已失效、已交接或不属于此Worker；先重新领取")
        return row

    def create(self, spec, actor="user", task_id=None):
        spec = task_spec(spec)
        actor = text(actor, "actor", 120)
        tid = task_id or "task-" + uuid.uuid4().hex[:12]
        with self.connection(True) as db:
            if spec["parent_id"]:
                self._task(db, spec["parent_id"])
            now = time.time()
            db.execute(
                "INSERT INTO tasks(id,spec,status,created,updated) VALUES(?,?,?,?,?)",
                (tid, encode(spec), "queued", now, now),
            )
            self.event(db, tid, actor, "created", spec)
        return self.get(tid)

    def list(self):
        with self.connection() as db:
            return [
                self.public(r) for r in db.execute("SELECT * FROM tasks ORDER BY created DESC,id")
            ]

    def get(self, task):
        with self.connection() as db:
            out = self.public(self._task(db, task))
            out["experiments"] = [
                self.public(r)
                for r in db.execute(
                    "SELECT * FROM experiments WHERE task_id=? ORDER BY created", (task,)
                )
            ]
            out["events"] = [
                self.public(r)
                for r in db.execute("SELECT * FROM events WHERE task_id=? ORDER BY seq", (task,))
            ]
            out["evidence"] = [
                self.public(r)
                for r in db.execute(
                    "SELECT * FROM evidence WHERE task_id=? ORDER BY created", (task,)
                )
            ]
            out["children"] = [
                r[0]
                for r in db.execute(
                    "SELECT id FROM tasks WHERE json_extract(spec,'$.parent_id')=?", (task,)
                )
            ]
            return out

    def claim(self, task, worker, ttl=1800):
        worker = text(worker, "worker", 120)
        if type(ttl) is not int or not 60 <= ttl <= 7200:
            raise ValueError("租约必须为60至7200秒")
        with self.connection(True) as db:
            row = self._task(db, task)
            now = time.time()
            if row["status"] not in {"queued", "active"}:
                raise ValueError("当前任务不在可领取状态")
            if row["status"] == "active" and (row["lease_until"] or 0) >= now:
                raise ValueError("任务已由Worker领取，需交接或等待租约到期")
            token = secrets.token_urlsafe(32)
            db.execute(
                "UPDATE tasks SET status='active',owner=?,token_hash=?,lease_until=?,updated=? WHERE id=?",
                (worker, hashlib.sha256(token.encode()).hexdigest(), now + ttl, now, task),
            )
            self.event(
                db, task, worker, "claimed", {"previous_owner": row["owner"], "ttl_seconds": ttl}
            )
        return {"task_id": task, "worker": worker, "token": token}

    def heartbeat(self, session, ttl=1800):
        if type(ttl) is not int or not 60 <= ttl <= 7200:
            raise ValueError("租约必须为60至7200秒")
        with self.connection(True) as db:
            row = self.owned(db, session)
            db.execute(
                "UPDATE tasks SET lease_until=?,updated=? WHERE id=?",
                (time.time() + ttl, time.time(), row["id"]),
            )
        return {"task_id": row["id"], "renewed": True}

    def checkpoint(self, session, note, handoff=False):
        note = text(note, "交接记录")
        with self.connection(True) as db:
            row = self.owned(db, session)
            now = time.time()
            point = {"at": now, "worker": session["worker"], "note": note}
            db.execute(
                "UPDATE tasks SET checkpoint=?,updated=? WHERE id=?",
                (encode(point), now, row["id"]),
            )
            if handoff:
                db.execute(
                    "UPDATE tasks SET status='queued',owner=NULL,token_hash=NULL,lease_until=NULL WHERE id=?",
                    (row["id"],),
                )
            self.event(
                db, row["id"], session["worker"], "handoff" if handoff else "checkpoint", point
            )
        return self.get(row["id"])

    def register(self, session, proposal, fingerprint, *, channel_budget=None, fit_budget=None):
        if proposal.get("kind") == "alpha_screen":
            declared = proposal.get("spec", {}).get("budget", {}).get("max_task_channel_intents")
            if channel_budget is None:
                channel_budget = declared
            if (type(declared) is not int or channel_budget != declared
                    or type(channel_budget) is not int or not 1 <= channel_budget <= 100000):
                raise ValueError("Strict persisted screen intent budget required")
        if proposal.get("kind") == "alpha_learn":
            declared = proposal.get("spec", {}).get("budget", {}).get("max_task_fit_intents")
            if fit_budget is None:
                fit_budget = declared
            if (type(declared) is not int or type(fit_budget) is not int
                    or fit_budget != declared or not 1 <= fit_budget <= 1000):
                raise ValueError("Strict persisted learning fit budget required")
        with self.connection(True) as db:
            row = self.owned(db, session)
            spec = json.loads(row["spec"])
            old = db.execute(
                "SELECT * FROM experiments WHERE task_id=? AND fingerprint=?",
                (row["id"], fingerprint),
            ).fetchone()
            if old:
                return self.public(old)
            count = db.execute(
                "SELECT count(*) FROM experiments WHERE task_id=?", (row["id"],)
            ).fetchone()[0]
            if count >= spec["max_experiments"]:
                raise ValueError("已达到任务实验预算，失败尝试也计入预算")
            from .account_intents import enforce_account_budget
            histories = [json.loads(r["proposal"]) for r in db.execute(
                "SELECT proposal FROM experiments WHERE task_id=?", (row["id"],))]
            enforce_account_budget(proposal, histories)
            if channel_budget is not None:
                if (proposal.get("kind") != "alpha_screen" or type(channel_budget) is not int
                        or not 1 <= channel_budget <= 100000
                        or type(proposal.get("candidate_count")) is not int
                        or proposal["candidate_count"] < 1):
                    raise ValueError("Strict persisted screen intent budget required")
                histories = [json.loads(r["proposal"]) for r in db.execute(
                    "SELECT proposal FROM experiments WHERE task_id=?", (row["id"],))]
                screens = [p for p in histories if p.get("kind") == "alpha_screen"]
                if any(p["spec"]["budget"]["max_task_channel_intents"] != channel_budget for p in screens):
                    raise ValueError("Task screen intent budget already frozen; cannot reset it")
                if sum(p["candidate_count"] for p in screens) + proposal["candidate_count"] > channel_budget:
                    raise ValueError("Persisted screen intent budget exhausted; failures are counted")
            if fit_budget is not None:
                if (proposal.get("kind") != "alpha_learn" or type(fit_budget) is not int
                        or not 1 <= fit_budget <= 1000 or type(proposal.get("planned_model_fits")) is not int
                        or proposal["planned_model_fits"] != 1):
                    raise ValueError("Strict persisted learning fit budget required")
                histories = [json.loads(r["proposal"]) for r in db.execute(
                    "SELECT proposal FROM experiments WHERE task_id=?", (row["id"],))]
                learns = [p for p in histories if p.get("kind") == "alpha_learn"]
                if any(p["spec"]["budget"]["max_task_fit_intents"] != fit_budget for p in learns):
                    raise ValueError("Task learning fit budget already frozen; cannot reset it")
                if sum(p["planned_model_fits"] for p in learns) + proposal["planned_model_fits"] > fit_budget:
                    raise ValueError("Persisted learning fit budget exhausted; failures are counted")
            eid = "exp-" + uuid.uuid4().hex[:12]
            now = time.time()
            db.execute(
                "INSERT INTO experiments(id,task_id,fingerprint,proposal,status,created,updated) VALUES(?,?,?,?,?,?,?)",
                (eid, row["id"], fingerprint, encode(proposal), "planned", now, now),
            )
            self.event(
                db,
                row["id"],
                session["worker"],
                "experiment_registered",
                {"experiment_id": eid, "proposal": proposal},
            )
        return self.experiment(eid)

    def experiment(self, eid):
        with self.connection() as db:
            row = db.execute("SELECT * FROM experiments WHERE id=?", (eid,)).fetchone()
            if row is None:
                raise ValueError("实验不存在")
            return self.public(row)

    def start(self, session, eid):
        with self.connection(True) as db:
            task = self.owned(db, session)
            row = db.execute(
                "SELECT * FROM experiments WHERE id=? AND task_id=?", (eid, task["id"])
            ).fetchone()
            if row is None or row["status"] != "planned":
                raise ValueError("实验不属于此任务或已执行；不能重复启动")
            if db.execute(
                "SELECT 1 FROM experiments WHERE task_id=? AND status='running'", (task["id"],)
            ).fetchone():
                raise ValueError("此任务仍有实验在运行；先恢复或收取其结果")
            db.execute(
                "UPDATE experiments SET status='running',updated=? WHERE id=?", (time.time(), eid)
            )
            self.event(
                db, task["id"], session["worker"], "experiment_started", {"experiment_id": eid}
            )
        return self.experiment(eid)

    def finish(self, eid, status, *, run_id=None, result=None, error=None):
        if status not in {"completed", "failed"}:
            raise ValueError("实验终态无效")
        with self.connection(True) as db:
            row = db.execute("SELECT * FROM experiments WHERE id=?", (eid,)).fetchone()
            if row is None:
                raise ValueError("实验不存在")
            if row["status"] != "running":
                return self.public(row)
            db.execute(
                "UPDATE experiments SET status=?,updated=?,run_id=?,result=?,error=? WHERE id=?",
                (status, time.time(), run_id, encode(result) if result else None, error, eid),
            )
            self.event(
                db,
                row["task_id"],
                "executor",
                "experiment_" + status,
                {"experiment_id": eid, "run_id": run_id, "error": error},
            )
        return self.experiment(eid)

    def cancel(self, session, eid, reason):
        reason = text(reason, "取消依据")
        with self.connection(True) as db:
            task = self.owned(db, session)
            row = db.execute(
                "SELECT * FROM experiments WHERE id=? AND task_id=?", (eid, task["id"])
            ).fetchone()
            if row is None or row["status"] != "planned":
                raise ValueError("只可取消本任务尚未启动的实验")
            db.execute(
                "UPDATE experiments SET status='cancelled',error=?,updated=? WHERE id=?",
                (reason, time.time(), eid),
            )
            self.event(
                db,
                task["id"],
                session["worker"],
                "experiment_cancelled",
                {"experiment_id": eid, "reason": reason},
            )
        return self.experiment(eid)

    def reopen(self, task, actor, reason):
        actor = text(actor, "actor", 120)
        reason = text(reason, "重新开启的依据")
        with self.connection(True) as db:
            row = self._task(db, task)
            if row["status"] != "deferred":
                raise ValueError("只有暂缓任务可以重新开启；已完成的研究应创建后续任务")
            db.execute("UPDATE tasks SET status='queued',updated=? WHERE id=?", (time.time(), task))
            self.event(db, task, actor, "reopened", {"reason": reason})
        return self.get(task)

    def attach(self, session, fingerprint, payload):
        with self.connection(True) as db:
            task = self.owned(db, session)
            old = db.execute(
                "SELECT * FROM evidence WHERE task_id=? AND fingerprint=?",
                (task["id"], fingerprint),
            ).fetchone()
            if old:
                return self.public(old)
            count = db.execute(
                "SELECT count(*) FROM evidence WHERE task_id=?", (task["id"],)
            ).fetchone()[0]
            if count >= json.loads(task["spec"])["max_experiments"]:
                raise ValueError("分析证据包上限已达到；导入上限不等同统计试验预算")
            eid = "evidence-" + uuid.uuid4().hex[:12]
            db.execute(
                "INSERT INTO evidence VALUES(?,?,?,?,?)",
                (eid, task["id"], fingerprint, encode(payload), time.time()),
            )
            self.event(
                db,
                task["id"],
                session["worker"],
                "evidence_attached",
                {"evidence_id": eid, "kind": payload["kind"], "fingerprint": fingerprint},
            )
        return {"id": eid, "task_id": task["id"], "payload": payload, "fingerprint": fingerprint}

    def submit(self, session, report):
        if not isinstance(report, dict) or set(report) != {
            "summary",
            "findings",
            "limitations",
            "next_steps",
        }:
            raise ValueError("结论需要summary、findings、limitations、next_steps")
        report = {
            "summary": text(report["summary"], "summary"),
            **{k: items(report[k], k) for k in ["findings", "limitations", "next_steps"]},
        }
        with self.connection(True) as db:
            row = self.owned(db, session)
            rows = db.execute(
                "SELECT status,result FROM experiments WHERE task_id=?", (row["id"],)
            ).fetchall()
            evidence = db.execute(
                "SELECT id FROM evidence WHERE task_id=?", (row["id"],)
            ).fetchall()
            if (not rows and not evidence) or any(
                r["status"] in {"planned", "running"} for r in rows
            ):
                raise ValueError("须先完成所有已登记实验")
            completed = [r for r in rows if r["status"] == "completed"]
            if any(
                not r["result"] or not json.loads(r["result"]).get("verified") for r in completed
            ):
                raise ValueError("缺少通过核查的实验；不能提交为策略证据")
            report["evidence_kind"] = (
                "audited_experiment"
                if completed
                else "analysis_artifacts"
                if evidence
                else "execution_failure_only"
            )
            if completed:
                kinds = {json.loads(r["result"]).get("kind", "account") for r in completed}
                if kinds == {"ml"}:
                    report["evidence_kind"] = "registered_ml_analysis"
                elif "ml" in kinds:
                    report["evidence_kind"] = "mixed_registered_experiments"
            report["evidence_ids"] = [r[0] for r in evidence]
            report["author"] = session["worker"]
            report["at"] = time.time()
            db.execute(
                "UPDATE tasks SET status='review',submission=?,token_hash=NULL,lease_until=NULL,updated=? WHERE id=?",
                (encode(report), time.time(), row["id"]),
            )
            self.event(db, row["id"], session["worker"], "submitted", report)
        return self.get(row["id"])

    def review(self, task, reviewer, decision, rationale, next_task=None):
        reviewer = text(reviewer, "reviewer", 120)
        rationale = text(rationale, "复核依据")
        if decision not in {"continue", "defer", "stop", "revise"}:
            raise ValueError("复核决定无效")
        if decision == "continue" and next_task is None:
            raise ValueError("继续研究必须登记下一项具体任务")
        if decision != "continue" and next_task is not None:
            raise ValueError("只有继续决定可创建后续任务")
        child = task_spec({**next_task, "parent_id": task}) if next_task is not None else None
        with self.connection(True) as db:
            row = self._task(db, task)
            if row["status"] != "review":
                raise ValueError("任务尚未提交复核或已复核")
            if json.loads(row["submission"])["author"] == reviewer:
                raise ValueError("复核者标识须与提交者不同；角色标识不等同独立模型认证")
            review = {
                "reviewer": reviewer,
                "decision": decision,
                "rationale": rationale,
                "at": time.time(),
            }
            if child:
                tid = "task-" + uuid.uuid4().hex[:12]
                review["next_task_id"] = tid
                now = time.time()
                db.execute(
                    "INSERT INTO tasks(id,spec,status,created,updated) VALUES(?,?,?,?,?)",
                    (tid, encode(child), "queued", now, now),
                )
                self.event(db, tid, reviewer, "created", child)
            status = {
                "continue": "completed",
                "defer": "deferred",
                "stop": "completed",
                "revise": "queued",
            }[decision]
            db.execute(
                "UPDATE tasks SET status=?,review=?,owner=NULL,token_hash=NULL,lease_until=NULL,updated=? WHERE id=?",
                (status, encode(review), time.time(), task),
            )
            self.event(db, task, reviewer, "reviewed", review)
        return self.get(task)

    def knowledge(self):
        with self.connection() as db:
            records = [
                json.loads(r[0]) for r in db.execute("SELECT payload FROM knowledge ORDER BY id")
            ]
            # Include negative findings, with the review decision and scope.
            for row in db.execute(
                "SELECT * FROM tasks WHERE status IN ('completed','deferred') AND review IS NOT NULL ORDER BY updated"
            ):
                task = self.public(row)
                report = task["submission"]
                records.append(
                    dict(
                        id="review-" + task["id"],
                        task_id=task["id"],
                        summary=report["summary"],
                        findings=report["findings"],
                        limits=report["limitations"],
                        source="research.sqlite3 / " + task["id"],
                        review=task["review"],
                        evidence_kind=report["evidence_kind"],
                        next_steps=report["next_steps"],
                        run_ids=[
                            r[0]
                            for r in db.execute(
                                "SELECT run_id FROM experiments WHERE task_id=? AND status='completed'",
                                (task["id"],),
                            )
                        ],
                    )
                )
            return records

    def add_knowledge(self, key, value):
        with self.connection(True) as db:
            db.execute("INSERT OR IGNORE INTO knowledge VALUES(?,?)", (key, encode(value)))
