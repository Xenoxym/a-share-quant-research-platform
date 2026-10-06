"""User-authorized persistent ML stage loop; launch in the logged-in local account."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.researchops.loop import Loops, Supervisor
from src.researchops.notification import desktop_notice
from src.researchops.quota import read_limits, quota_status
from src.researchops.service import Research


def main():
    p = argparse.ArgumentParser(description="ML阶段循环：归档、双复核、接续、真实额度等待")
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--root", type=Path)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("quota")
    sub.add_parser("list")
    create = sub.add_parser("create")
    create.add_argument("--file", type=Path, required=True)
    for name in ["run", "show", "stop"]:
        sub.add_parser(name).add_argument("loop")
    policy = sub.add_parser("notice-policy")
    policy.add_argument("loop")
    policy.add_argument("--file", type=Path, required=True)
    policy.add_argument("--reason", required=True)
    acknowledge = sub.add_parser("acknowledge")
    acknowledge.add_argument("loop")
    acknowledge.add_argument("--reason", required=True)
    args = p.parse_args()
    r = Research(args.project, args.root, readonly=args.command in {"show", "list", "quota"})
    loops = Loops(r)
    if args.command == "quota":
        result = quota_status(read_limits())
    elif args.command == "create":
        result = loops.create(json.loads(args.file.read_text(encoding="utf-8-sig")))
    elif args.command == "run":
        result = Supervisor(r, notifier=desktop_notice).run(args.loop)
    elif args.command == "stop":
        result = loops.stop(args.loop)
    elif args.command == "notice-policy":
        result = loops.enable_noteworthy(args.loop, json.loads(args.file.read_text(encoding="utf-8-sig")), args.reason)
    elif args.command == "acknowledge":
        result = loops.acknowledge(args.loop, args.reason)
    elif args.command == "show":
        result = loops.get(args.loop)
    else:
        result = [loops.get(p.parent.name) for p in loops.root.glob("*/state.json")]
    print(json.dumps(dict(ok=True, data=result), ensure_ascii=False))
    return int(isinstance(result, dict) and result.get("state", {}).get("status") == "needs_attention")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps(dict(ok=False, error=str(exc)), ensure_ascii=False))
        sys.exit(1)
