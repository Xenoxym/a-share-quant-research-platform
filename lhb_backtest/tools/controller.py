"""Local bounded Codex campaigns. Run under the user's authenticated CLI account."""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.researchops.controller import Campaigns, Controller
from src.researchops.service import Research


def main():
    p = argparse.ArgumentParser(description="有预算的研究总控：派发、双复核、裁决、下一轮")
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--root", type=Path)
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    create = commands.add_parser("create")
    create.add_argument("--file", type=Path, required=True)
    for name in ["show", "run", "stop"]:
        commands.add_parser(name).add_argument("campaign")
    retry = commands.add_parser("retry")
    retry.add_argument("campaign")
    retry.add_argument("--reason", required=True)
    args = p.parse_args()
    research = Research(args.project, args.root, readonly=args.command in {"list", "show"})
    db = Campaigns(research)
    if args.command == "create":
        result = db.create(json.loads(args.file.read_text(encoding="utf-8-sig")))
    elif args.command == "run":
        result = Controller(research).run(args.campaign)
    elif args.command == "show":
        result = db.get(args.campaign)
    elif args.command == "stop":
        result = db.stop(args.campaign)
    elif args.command == "retry":
        result = db.retry(args.campaign, args.reason)
    else:
        result = db.list()
    print(json.dumps(dict(ok=True, data=result), ensure_ascii=False))
    return (
        1
        if isinstance(result, dict) and result.get("state", {}).get("status") == "needs_attention"
        else 0
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps(dict(ok=False, error=str(exc)), ensure_ascii=False))
        sys.exit(1)
