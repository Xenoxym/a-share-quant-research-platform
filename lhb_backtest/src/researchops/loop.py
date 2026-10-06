"""Durable stage supervision over bounded campaigns, using real account quota."""
import json
import html
import os
from pathlib import Path
import threading
import time
import uuid

from .controller import Campaigns, Controller, DispatchDeferred, config as campaign_config
from .quota import read_limits, quota_status
from .store import task_spec
from ..technical.artifacts import content_id, digest, exclusive_run, utc_now, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quota_failure(campaign):
    """Require explicit provider quota errors, not merely a nonzero CLI exit."""
    failed = [j for j in campaign["jobs"] if j["round"] == campaign["state"]["round"] and j["state"]["status"] == "failed"]
    if not failed:
        return False
    for job in failed:
        path = Path(job["state"].get("folder", "")) / "events.jsonl"
        if not path.is_file():
            return False
        found = False
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") not in {"error", "turn.failed"}:
                continue
            message = json.dumps(event, ensure_ascii=False).lower().replace("\u2019", "'")
            if any(term in message for term in ["you've hit your usage limit", "usage_limit_reached", "usage_limit_exceeded", "usage limit reached", "ratelimitreached"]):
                found = True
        if not found:
            return False
    return True


class Loops:
    def __init__(self, research):
        self.research = research
        self.root = research.root / "loops"

    def folder(self, lid):
        if not isinstance(lid, str) or not lid.startswith("loop-") or not lid[5:].isalnum():
            raise ValueError("循环编号无效")
        return self.root / lid

    def create(self, value):
        required = {"title", "task_ids", "campaign_budget", "agenda", "guidance", "resume_after_reset"}
        if set(value) != required or type(value["resume_after_reset"]) is not bool:
            raise ValueError("循环配置字段无效")
        campaign_config(dict(title=value["title"], task_ids=value["task_ids"], **value["campaign_budget"]))
        if not isinstance(value["agenda"], list) or len(value["agenda"]) > 12:
            raise ValueError("后续问题清单无效")
        for spec in value["agenda"]:
            task_spec(spec)
        for tid in value["task_ids"]:
            if self.research.store.get(tid)["status"] not in {"queued", "review"}:
                raise ValueError("起点任务须为待研究或待复核")
        for p in self.root.glob("*/state.json"):
            prior = read(p)
            if prior["status"] in {"ready", "running", "waiting_quota", "noteworthy_result"} and set(prior["pending"]) & set(value["task_ids"]):
                raise ValueError("起点已被另一循环监督")
        lid = "loop-" + uuid.uuid4().hex[:12]
        folder = self.folder(lid)
        folder.mkdir(parents=True)
        write_json(folder / "config.json", value)
        write_json(folder / "state.json", dict(id=lid, status="ready", phase=1, pending=value["task_ids"],
            agenda_cursor=0, campaign_id=None, archived=[], config_hash=digest(folder / "config.json"),
            error=None, updated=utc_now(), pid=None))
        self.event(lid, "created", task_ids=value["task_ids"])
        return self.get(lid)

    def get(self, lid):
        folder = self.folder(lid)
        state = read(folder / "state.json")
        cfg = read(folder / "config.json")
        if digest(folder / "config.json") != state["config_hash"]:
            raise ValueError("循环设计在建立后发生变化")
        policy = None
        if state.get("notice_policy_file"):
            name = state["notice_policy_file"]
            if Path(name).name != name or digest(folder / name) != state["notice_policy_hash"]:
                raise ValueError("停止政策附录发生变化")
            policy = read(folder / name)
            # Acknowledgements are append-only history, not edits of frozen policy.
            policy = {**policy, "reported_evidence_refs": policy["reported_evidence_refs"] + [
                ref for ack in state.get("notice_acknowledgements", []) for ref in ack["evidence_refs"]],
                "reported_notice_signatures": [ack["notice_signature"] for ack in state.get("notice_acknowledgements", [])],
                "reported_notices": [ack["notice"] for ack in state.get("notice_acknowledgements", [])]}
        return dict(config=cfg, state=state, notice_policy=policy, folder=str(folder),
                    stop_requested=(folder / "stop.requested").exists())

    def enable_noteworthy(self, lid, value, reason):
        """Record new user authorization as an immutable policy addendum."""
        from .store import text
        reason = text(reason, "新增停止条件的用户依据")
        if (not isinstance(value, dict) or set(value) != {"guidance", "reported_evidence_refs"}
            or not isinstance(value["guidance"], str) or not value["guidance"].strip()
            or not isinstance(value["reported_evidence_refs"], list)
            or any(not isinstance(ref, str) for ref in value["reported_evidence_refs"])):
            raise ValueError("停止政策需要guidance与已报告的证据ID列表")
        folder = self.folder(lid)
        with exclusive_run(folder):
            entry = self.get(lid)
            if entry["state"].get("pid"):
                raise ValueError("需等现有循环收取后安装政策，不修改运行中冻结输入")
            name = "notice_policy_" + uuid.uuid4().hex[:12] + ".json"
            write_json(folder / name, dict(version=1, authorized_at=utc_now(), authorization=reason, **value))
            self.update(lid, notice_policy_file=name, notice_policy_hash=digest(folder / name))
            self.event(lid, "noteworthy_policy_enabled", policy=name, authorization=reason)
        return self.get(lid)

    def acknowledge(self, lid, reason):
        """Only an explicit user continuation can clear an important-result stop."""
        from .store import text
        reason = text(reason, "用户明确续行的依据")
        with exclusive_run(self.folder(lid)):
            entry = self.get(lid)
            state = entry["state"]
            if state["status"] != "noteworthy_result":
                raise ValueError("仅确认重要发现停止；普通失败不能通过此操作恢复")
            receipt_path = Path(state["notice_folder"]) / "receipt.json"
            if digest(receipt_path) != state.get("notice_receipt_hash"):
                raise ValueError("重要发现报告收据完整性发生变化")
            receipt = read(receipt_path)
            if set(receipt.get("artifacts", {})) != {"notice.json", "REPORT.md"}:
                raise ValueError("重要发现报告收据缺少必需文件")
            for name, sha in receipt["artifacts"].items():
                if digest(Path(state["notice_folder"]) / name) != sha:
                    raise ValueError("重要发现报告完整性发生变化")
            db = Campaigns(self.research)
            cid = state["campaign_id"]
            with exclusive_run(db.root / cid):
                c = db.get(cid)
                if c["state"]["status"] != "noteworthy_result":
                    raise ValueError("循环与总控停止状态不一致")
                ack = dict(at=utc_now(), reason=reason, campaign_id=cid,
                           evidence_refs=c["state"]["notice"]["evidence_refs"], notice_folder=state["notice_folder"],
                           notice=c["state"]["notice"], notice_signature=c["state"]["notice_signature"])
                db.update(cid, status="queued", stage="用户已确认接续；原预算与历史保留")
                self.update(lid, status="ready", error=None,
                            notice_acknowledgements=state.get("notice_acknowledgements", []) + [ack])
                self.event(lid, "noteworthy_acknowledged", **ack)
        return self.get(lid)

    def update(self, lid, **patch):
        folder = self.folder(lid)
        state = {**read(folder / "state.json"), **patch, "updated": utc_now()}
        write_json(folder / "state.json", state)

    def event(self, lid, event, **details):
        with (self.folder(lid) / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(dict(time=utc_now(), event=event, **details), ensure_ascii=False) + "\n")

    def stop(self, lid):
        entry = self.get(lid)
        write_json(self.folder(lid) / "stop.requested", dict(requested_at=utc_now()))
        cid = entry["state"]["campaign_id"]
        if cid:
            Campaigns(self.research).stop(cid)
        self.event(lid, "stop_requested")
        return self.get(lid)


class Supervisor:
    def __init__(self, research, probe=read_limits, controller_factory=Controller, notifier=None):
        self.research = research
        self.loops = Loops(research)
        self.campaigns = Campaigns(research)
        self.probe = probe
        self.controller_factory = controller_factory
        self.notifier = notifier
        self.quota_lock = threading.Lock()
        self.cached = None
        self.last_probe = 0

    def progress(self, lid):
        entry = self.loops.get(lid)
        state = entry["state"]
        title = entry["config"]["title"]
        labels = {"ready":"等待启动", "running":"自动推进中", "waiting_quota":"额度等待，恢复前会重新核实",
                  "needs_attention":"需要处理，失败已保留", "stopped":"已停止", "needs_new_evidence":"已有问题已收束，等待新依据",
                  "noteworthy_result":"发现值得报告的成果或问题，已停止，等待用户明确续行"}
        lines = [f"# {title}", "", f"更新：{utc_now()}", f"循环：{lid}",
                 f"状态：{labels.get(state['status'], state['status'])}", f"阶段：{state['phase']}", ""]
        cid = state.get("campaign_id")
        if cid:
            c = self.campaigns.get(cid)
            lines += [f"当前总控：{cid}", f"当前工作：{c['state']['stage']}",
                      f"Agent尝试：{len(c['jobs'])}/{c['config']['max_calls']}（失败也计入）", ""]
            for j in c["jobs"][-8:]:
                lines += [f"- {j['role']} / {j['task_id']}：{j['state']['status']}"]
        qpath = self.loops.folder(lid) / "quota_latest.json"
        if qpath.exists():
            q = read(qpath)
            lines += ["", f"额度观测时间：{q['observed_at']}"]
            for w in q.get("windows", []):
                from datetime import datetime, timezone
                stamp = w.get("reset_at")
                reset = datetime.fromtimestamp(stamp, timezone.utc).isoformat() if stamp else "未知"
                lines += [f"- {w['window_minutes']}分钟窗口：已用{w['used_percent']}%；重置{reset}（UTC）"]
        lines += ["", "已归档阶段："] + [f"- {c}" for c in state["archived"]]
        if state.get("error"):
            lines += ["", "停止原因：" + state["error"]]
        if entry["notice_policy"]:
            lines += ["", "新增停止条件已启用：出现重要学习成果、机制发现或重大完整性问题时，归档并停止。额度重置不会解除此停止。"]
        if state.get("notice_folder"):
            lines += ["", "重要发现报告：" + str(Path(state["notice_folder"]) / "REPORT.md")]
        lines += ["", "下一步：优先接裁决产生的子任务，再进入已登记的不同研究问题。结果、失败及下一计划均保存在checkpoints目录。",
                  "用tools/ml_loop.py show或stop加上述循环编号查看/停止。后台会话不保证出现在桌面侧栏。",
                  "电脑须保持开机。额度未知不猜五小时；网络、登录和实质复核阻断不会无限重试。所有历史已看过。"]
        folder = self.loops.folder(lid)
        text = "\n".join(lines)
        (folder / "STATUS.md").write_text(text, encoding="utf-8")
        body = "<!doctype html><meta charset='utf-8'><meta http-equiv='refresh' content='30'><title>ML自动推进</title><style>body{max-width:1000px;margin:40px auto;font:17px/1.7 system-ui;color:#172536;background:#f5f7fa}pre{white-space:pre-wrap;background:white;padding:28px;border-radius:12px}a{color:#1968bc}</style>"
        body += "<pre>" + html.escape(text) + "</pre><p><a href='checkpoints/'>阶段归档目录</a></p>"
        temp = folder / "STATUS.html.tmp"
        temp.write_text(body, encoding="utf-8")
        os.replace(temp, folder / "STATUS.html")

    def quota(self, lid, force=False):
        with self.quota_lock:
            if force or self.cached is None or time.monotonic() - self.last_probe > 10:
                try:
                    status = quota_status(self.probe())
                except Exception as exc:
                    status = dict(known=False, blocked=False, reset_at=None, windows=[], error=str(exc))
                self.cached, self.last_probe = status, time.monotonic()
                write_json(self.loops.folder(lid) / "quota_latest.json", dict(observed_at=utc_now(), **status))
                self.loops.event(lid, "quota_observed", quota=status)
            return self.cached

    def guard(self, lid):
        entry = self.loops.get(lid)
        if entry["state"]["status"] == "noteworthy_result":
            raise RuntimeError("重要发现已报告，等待用户明确续行，不受额度重置影响")
        if entry["stop_requested"]:
            raise RuntimeError("用户已请求停止阶段循环")
        status = self.quota(lid)
        if not status["known"]:
            raise RuntimeError("无法可靠读取额度/登录状态，停止派发：" + status.get("error", "unknown"))
        if status["blocked"]:
            raise DispatchDeferred(status)

    def wait_quota(self, lid, status):
        cfg = self.loops.get(lid)["config"]
        self.loops.update(lid, status="waiting_quota", quota=status, error=None)
        self.loops.event(lid, "waiting_quota", reset_at=status.get("reset_at"))
        if not cfg["resume_after_reset"]:
            return False
        while True:
            if self.loops.get(lid)["stop_requested"]:
                self.loops.update(lid, status="stopped")
                return False
            reset = status.get("reset_at")
            # No guessed five-hour reset. Unknown reset means bounded read-only rechecks.
            delay = min(30, max(1, reset - time.time() + 2)) if reset else 30
            time.sleep(delay)
            if reset and time.time() < reset:
                continue
            status = self.quota(lid, force=True)
            if not status["known"]:
                self.loops.update(lid, status="needs_attention", error="等待后无法验证额度；没有派发新会话")
                return False
            if not status["blocked"]:
                self.loops.event(lid, "quota_available_again")
                self.loops.update(lid, status="running", quota=status)
                return True
            self.loops.update(lid, quota=status)

    def checkpoint(self, lid, campaign):
        """Archive before choosing anything else; archived receipts are never overwritten."""
        folder = self.loops.folder(lid) / "checkpoints" / campaign["id"]
        task_ids = set(campaign["config"]["task_ids"]) | set(campaign["state"]["pending"]) | {
            job["task_id"] for job in campaign["jobs"] if job["role"] != "synthesis"}
        contexts = {tid: self.research.context(tid) for tid in sorted(task_ids)}
        for tid, context in contexts.items():
            if context["task"].get("submission"):
                self.research.verify_evidence(tid)
        if (folder / "receipt.json").exists():
            receipt = read(folder / "receipt.json")
            for name, sha in receipt["artifacts"].items():
                if digest(folder / name) != sha:
                    raise ValueError("阶段归档文件发生变化")
            if read(folder / "campaign.json") != campaign or read(folder / "tasks.json") != contexts:
                folder = folder / "versions" / content_id(dict(campaign=campaign, tasks=contexts))[:16]
        if not (folder / "receipt.json").exists():
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "campaign.json", campaign)
            write_json(folder / "tasks.json", contexts)
            history = campaign["state"]["history"]
            text = [f"# 阶段 {campaign['id']}", "", f"状态：{campaign['state']['status']}", ""]
            for stage in history:
                text += [stage["summary"], ""]
                for decision in stage["decisions"]:
                    text += [f"- {decision['task_id']}：{decision['decision']}；{decision['rationale']}"]
            text += ["", "下一阶段依据：优先接续裁决建立的子任务；没有子任务时，才进入既定的不同研究问题。",
                     "所有历史均已看过。保存分析包或通过双复核，并不生成未见样本，也不代表机构认证。"]
            (folder / "REPORT.md").write_text("\n".join(text), encoding="utf-8")
            write_json(folder / "receipt.json", dict(created_at=utc_now(), artifacts={
                name: digest(folder / name) for name in ["campaign.json", "tasks.json", "REPORT.md"]}))
        else:
            receipt = read(folder / "receipt.json")
            for name, sha in receipt["artifacts"].items():
                if digest(folder / name) != sha:
                    raise ValueError("阶段归档文件发生变化")
        self.loops.event(lid, "stage_archived", campaign_id=campaign["id"], receipt=str(folder / "receipt.json"))
        return folder

    def noteworthy(self, lid, campaign):
        checkpoint = self.checkpoint(lid, campaign)
        notice = campaign["state"]["notice"]
        folder = self.loops.folder(lid) / "notices" / (
            campaign["id"] + "-round-" + str(campaign["state"]["notice_round"]))
        if not (folder / "receipt.json").exists():
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "notice.json", dict(notice=notice, checkpoint=str(checkpoint),
                       campaign_id=campaign["id"], created_at=utc_now()))
            lines = ["# " + notice["title"], "", notice["explanation"], "",
                     "证据任务：" + ", ".join(notice["task_ids"]),
                     "关联证据：" + ", ".join(notice["evidence_refs"]), ""]
            if notice["kind"] == "integrity_issue":
                lines += ["这是需要告知的完整性问题；是否已解决，以归档的复核及裁决为准，不表示策略研究通过。", ""]
            lines += ["局限："] + ["- " + line for line in notice["limitations"]]
            lines += ["", "循环已停止在下一研究开始前。额度重置不会自动续行；用户明确要求继续才确认本报告并接续。",
                      "这是本地持久化报告；不保证自动向桌面聊天发送消息。"]
            (folder / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
            write_json(folder / "receipt.json", dict(created_at=utc_now(), artifacts={
                name: digest(folder / name) for name in ["notice.json", "REPORT.md"]}))
        else:
            receipt = read(folder / "receipt.json")
            if set(receipt.get("artifacts", {})) != {"notice.json", "REPORT.md"}:
                raise ValueError("重要发现报告收据缺少必需文件")
            if read(folder / "notice.json")["notice"] != notice:
                raise ValueError("重要发现报告与裁决不一致")
            for name, sha in receipt["artifacts"].items():
                if digest(folder / name) != sha:
                    raise ValueError("重要发现报告发生变化")
        self.loops.update(lid, status="noteworthy_result", pending=campaign["state"]["pending"],
                          notice_folder=str(folder), notice_receipt_hash=digest(folder / "receipt.json"),
                          error=notice["title"])
        self.loops.event(lid, "noteworthy_result", campaign_id=campaign["id"], report=str(folder / "REPORT.md"))
        if self.notifier:
            try:
                result = self.notifier(folder, notice)
                self.loops.event(lid, "noteworthy_notification_attempted", result=result)
            except Exception as exc:
                self.loops.event(lid, "noteworthy_notification_failed", error=str(exc))

    def park_failed_quota(self, lid, campaign, status):
        # Archive the failed attempt separately from the later stage conclusion.
        folder = self.loops.folder(lid) / "quota_failures" / (campaign["id"] + "-round-" + str(campaign["state"]["round"]))
        if not (folder / "campaign.json").exists():
            write_json(folder / "campaign.json", campaign)
            write_json(folder / "quota.json", status)
        self.campaigns.update(campaign["id"], status="waiting_quota", quota=status,
            quota_wait_started=time.time(), quota_failed_attempt=True, stage="执行中额度耗尽；失败尝试已保存")
        self.loops.event(lid, "failed_call_quota_confirmed", campaign_id=campaign["id"], archive=str(folder))

    def resume_failed_quota(self, cid):
        c = self.campaigns.get(cid)
        if not c["state"].get("quota_failed_attempt"):
            return
        waited = c["state"].get("quota_paused_seconds", 0)
        waited += max(0, time.time() - c["state"]["quota_wait_started"])
        self.campaigns.update(cid, status="needs_attention", quota_paused_seconds=waited,
            quota_wait_started=None, quota_failed_attempt=False)
        self.campaigns.retry(cid, "CLI结构化日志证实额度耗尽；服务器额度已重新核实可用。保留原失败、调用及执行时间预算，不重复已完成实验。")

    def advance(self, lid, campaign):
        entry = self.loops.get(lid)
        state, cfg = entry["state"], entry["config"]
        checkpoint = self.checkpoint(lid, campaign)
        if campaign["id"] in state["archived"]:
            raise ValueError("已接续过此阶段；拒绝重复推进")
        pending = campaign["state"]["pending"]
        decisions = [d for h in campaign["state"]["history"] for d in h["decisions"]]
        # Only an unstarted new child question can receive the next stage's budget.
        # Children researched/reviewed inside this campaign have already spent it.
        attempted = set(campaign["config"]["task_ids"]) | {
            j["task_id"] for j in campaign["jobs"] if j["role"] != "synthesis"
        }
        if any(d["decision"] == "defer" for d in decisions):
            raise RuntimeError("阶段裁决要求新证据；不能借连续循环绕过暂缓")
        cursor = state["agenda_cursor"]
        if not pending and cursor < len(cfg["agenda"]):
            parent = campaign["config"]["task_ids"][-1]
            if self.research.store.get(parent)["status"] != "completed":
                raise RuntimeError("上阶段尚未通过限定范围验收")
            spec = dict(cfg["agenda"][cursor], parent_id=parent)
            # A deterministic task id makes a crash between create/update recoverable.
            tid = "task-" + content_id(dict(loop=lid, agenda=cursor))[:12]
            try:
                self.research.store.get(tid)
            except ValueError:
                self.research.create(spec, actor="authorized-stage-loop", task_id=tid)
            pending = [tid]
            cursor += 1
        # Check the resolved candidates, including deterministic agenda recovery.
        # Work in another campaign/session also counts: an ID absent from these
        # jobs is not proof that its original budget is still untouched.
        for tid in pending:
            task = self.research.store.get(tid)
            if tid in attempted or any(e["kind"] != "created" for e in task["events"]):
                raise RuntimeError("接续任务已有执行历史；原任务保留，不另建总控清零预算")
        next_plan = dict(campaign_id=campaign["id"], pending=pending, agenda_cursor=cursor,
                         basis="synthesis_child_or_predeclared_distinct_question", created_at=utc_now())
        plan = checkpoint / "next_plan.json"
        if plan.exists() and read(plan)["pending"] != pending:
            raise ValueError("归档后的后续计划不一致")
        if not plan.exists():
            write_json(plan, next_plan)
        self.loops.update(lid, pending=pending, agenda_cursor=cursor, campaign_id=None,
            archived=state["archived"] + [campaign["id"]], phase=state["phase"] + 1,
            status="running" if pending else "needs_new_evidence")
        self.loops.event(lid, "next_stage_planned", **next_plan)

    def run(self, lid):
        folder = self.loops.folder(lid)
        with exclusive_run(folder):
            self.loops.update(lid, pid=os.getpid())
            stop_monitor = threading.Event()
            def monitor():
                while not stop_monitor.is_set():
                    try:
                        self.progress(lid)
                    except Exception as exc:
                        self.loops.event(lid, "progress_refresh_failed", error=str(exc))
                    stop_monitor.wait(15)
            watcher = threading.Thread(target=monitor, daemon=True)
            watcher.start()
            try:
                while True:
                    entry = self.loops.get(lid)
                    cfg, state = entry["config"], entry["state"]
                    if entry["stop_requested"]:
                        self.loops.update(lid, status="stopped")
                        break
                    if state["status"] not in {"ready", "running", "waiting_quota"}:
                        break
                    status = self.quota(lid, force=True)
                    if not status["known"]:
                        raise RuntimeError("无法读取真实额度；请从已登录的Windows用户环境启动，未派发付费会话")
                    if status["blocked"] and not self.wait_quota(lid, status):
                        break
                    cid = state["campaign_id"]
                    if not cid:
                        title = f"{cfg['title']} / {lid} / 阶段{state['phase']}"
                        old = [c for c in self.campaigns.list() if c["config"]["title"] == title]
                        if len(old) > 1:
                            raise RuntimeError("发现同阶段多份总控，停止并核查")
                        c = old[0] if old else self.campaigns.create(dict(title=title, task_ids=state["pending"], **cfg["campaign_budget"]))
                        cid = c["id"]
                        self.loops.update(lid, campaign_id=cid, status="running")
                        self.loops.event(lid, "campaign_selected", campaign_id=cid)
                    c = self.campaigns.get(cid)
                    if c["state"]["status"] == "waiting_quota" and c["state"].get("quota_failed_attempt"):
                        self.resume_failed_quota(cid)
                        c = self.campaigns.get(cid)
                    if c["state"]["status"] in {"queued", "starting", "waiting_quota"}:
                        kwargs = dict(before_job=lambda: self.guard(lid), guidance=cfg["guidance"])
                        if entry["notice_policy"]:
                            kwargs["notice_policy"] = entry["notice_policy"]
                        c = self.controller_factory(self.research, **kwargs).run(cid)
                    elif c["state"]["status"] == "running":
                        raise RuntimeError("上次总控未正常收取，先核查PID与日志；不会盲目恢复")
                    if c["state"]["status"] == "waiting_quota":
                        if not self.wait_quota(lid, c["state"]["quota"]):
                            break
                        continue
                    if c["state"]["status"] == "noteworthy_result":
                        self.noteworthy(lid, c)
                        break
                    if c["state"]["status"] == "needs_attention" and quota_failure(c):
                        status = self.quota(lid, force=True)
                        if status["known"] and status["blocked"]:
                            self.park_failed_quota(lid, c, status)
                            if not self.wait_quota(lid, status):
                                break
                            continue
                    if c["state"]["status"] not in {"completed", "budget_exhausted"}:
                        self.checkpoint(lid, c)
                        raise RuntimeError("总控需要处理，失败已归档：" + str(c["state"].get("error")))
                    self.advance(lid, c)
            except Exception as exc:
                self.loops.update(lid, status="needs_attention", error=str(exc))
                self.loops.event(lid, "halted", error=str(exc))
            finally:
                self.loops.update(lid, pid=None)
                stop_monitor.set()
                watcher.join(timeout=5)
                self.progress(lid)
        return self.loops.get(lid)
