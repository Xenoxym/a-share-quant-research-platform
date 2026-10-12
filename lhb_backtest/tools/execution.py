"""Compact execution-plan CLI. Numerical logs stay in existing worker directories."""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.researchops.service import Research  # noqa: E402
from src.researchops.workflow import ExecutionPlans  # noqa: E402


def main():
    p = argparse.ArgumentParser(
        description="Run/resume a bounded plan with existing Research workers"
    )
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--root", type=Path)
    commands = p.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--file", type=Path, required=True)
    run.add_argument("--session", type=Path, required=True)
    run.add_argument("--steps", type=int, default=40)
    status = commands.add_parser("status")
    status.add_argument("plan_id")
    a = p.parse_args()
    research = Research(a.project, a.root, readonly=a.command == "status")
    plans = ExecutionPlans(research)
    if a.command == "status":
        value = plans.status(a.plan_id)
    else:
        value = plans.run(
            json.loads(a.session.read_text(encoding="utf-8-sig")),
            json.loads(a.file.read_text(encoding="utf-8-sig")),
            max_steps=a.steps,
        )
    print(json.dumps({"ok": True, "data": value}, ensure_ascii=False, allow_nan=False))
    return 1 if any(s["status"] == "blocked" for s in value["steps"]) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        sys.exit(1)
