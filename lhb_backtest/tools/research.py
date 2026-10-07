"""Stable JSON CLI for human-led and external-agent research workers."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.researchops.service import Research
from src.technical.artifacts import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description="持久化研究任务与Worker接口（标准输出仅JSON）")
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--root", type=Path, help="共享研究库绝对路径；新worktree应显式指定")
    cmds = parser.add_subparsers(dest="command", required=True)
    for name in ["catalog", "list", "init"]:
        cmds.add_parser(name)
    for name in ["context", "show", "export"]:
        p = cmds.add_parser(name)
        p.add_argument("task")
        if name == "export":
            p.add_argument("--output", type=Path, required=True)
    p = cmds.add_parser("create")
    p.add_argument("--file", required=True)
    p.add_argument("--actor", default="user")
    p = cmds.add_parser("claim")
    p.add_argument("task")
    p.add_argument("--worker", required=True)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--ttl", type=int, default=1800)
    for name in [
        "heartbeat",
        "checkpoint",
        "handoff",
        "register",
        "run",
        "submit",
        "cancel",
        "attach",
    ]:
        p = cmds.add_parser(name)
        p.add_argument("--session", type=Path, required=True)
        if name in {"checkpoint", "handoff"}:
            p.add_argument("--note", required=True)
        if name in {"register", "submit", "attach"}:
            p.add_argument("--file", required=True)
        if name in {"run", "cancel"}:
            p.add_argument("experiment")
        if name == "cancel":
            p.add_argument("--reason", required=True)
    p = cmds.add_parser("reopen")
    p.add_argument("task")
    p.add_argument("--actor", required=True)
    p.add_argument("--reason", required=True)
    p = cmds.add_parser("recover")
    p.add_argument("experiment")
    p = cmds.add_parser("review")
    p.add_argument("task")
    p.add_argument("--reviewer", required=True)
    p.add_argument("--decision", choices=["continue", "defer", "stop", "revise"], required=True)
    p.add_argument("--rationale", required=True)
    p.add_argument("--next-task", type=Path)
    args = parser.parse_args()
    r = Research(
        args.project,
        args.root,
        readonly=args.command in {"catalog", "list", "context", "show", "export"},
    )
    cmd = args.command
    if cmd == "init":
        value = r.seed()
    elif cmd == "catalog":
        value = r.catalog()
    elif cmd == "list":
        value = r.store.list()
    elif cmd in {"context", "show", "export"}:
        value = r.context(args.task) if cmd != "show" else r.store.get(args.task)
        if cmd == "export":
            write_json(args.output, value)
    elif cmd == "create":
        value = r.create(read(args.file), args.actor)
    elif cmd == "claim":
        if args.session.exists():
            raise ValueError("session文件已存在；请使用此Worker的新文件名，避免覆盖旧领取凭据")
        args.session.parent.mkdir(parents=True, exist_ok=True)
        session = r.store.claim(args.task, args.worker, args.ttl)
        write_json(args.session, session)
        value = {
            "task_id": args.task,
            "worker": args.worker,
            "session": str(args.session.resolve()),
            "ttl_seconds": args.ttl,
        }
    elif cmd == "heartbeat":
        value = r.store.heartbeat(read(args.session))
    elif cmd in {"checkpoint", "handoff"}:
        value = r.store.checkpoint(read(args.session), args.note, cmd == "handoff")
    elif cmd == "register":
        value = r.register(read(args.session), read(args.file))
    elif cmd == "attach":
        value = r.attach(read(args.session), read(args.file))
    elif cmd == "run":
        value = r.execute(read(args.session), args.experiment)
    elif cmd == "recover":
        value = r.recover(args.experiment)
    elif cmd == "cancel":
        value = r.cancel(read(args.session), args.experiment, args.reason)
    elif cmd == "reopen":
        value = r.store.reopen(args.task, args.actor, args.reason)
    elif cmd == "submit":
        value = r.submit(read(args.session), read(args.file))
    else:
        value = r.review(
            args.task,
            args.reviewer,
            args.decision,
            args.rationale,
            read(args.next_task) if args.next_task else None,
        )
    failed = cmd in {"run", "recover"} and value.get("status") == "failed"
    print(json.dumps({"ok": not failed, "data": value}, ensure_ascii=False, allow_nan=False))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        sys.exit(1)
