"""Compact execution-plan CLI. Numerical logs stay in existing worker directories."""

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.researchops.service import Research  # noqa: E402
from src.researchops.workflow import ExecutionPlans  # noqa: E402


def _strict_json(path):
    def object_fields(pairs):
        value = dict(pairs)
        if len(value) != len(pairs):
            raise ValueError("Duplicate JSON keys")
        return value
    def reject_constant(value):
        raise ValueError("Non-finite JSON values")
    value = json.loads(path.read_text(encoding="utf-8-sig"),
                       object_pairs_hook=object_fields, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


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
    register = commands.add_parser("register", help="Register one proposal without starting work")
    register.add_argument("--file", type=Path, required=True)
    register.add_argument("--session", type=Path, required=True)
    derive=commands.add_parser("account-proposal",help="Build a saved-score account proposal; no fit or registration")
    derive.add_argument("--learning",required=True)
    derive.add_argument("--template",type=Path,required=True)
    derive.add_argument("--output",type=Path,required=True)
    matrix=commands.add_parser("learning-proposal",help="Bind a saved matrix to a fixed learning template; no fit or registration")
    matrix.add_argument("--matrix",required=True)
    matrix.add_argument("--template",type=Path,required=True)
    matrix.add_argument("--output",type=Path,required=True)
    backup=commands.add_parser("backup-learning",help="Copy one completed learning evidence bundle; no model loading")
    backup.add_argument("--learning",required=True)
    backup.add_argument("--output",type=Path,required=True)
    backup.add_argument("--max-bytes",type=int,required=True)
    offline=commands.add_parser("verify-backup",help="Verify copied bytes offline; never activate workers")
    offline.add_argument("--folder",type=Path,required=True)
    status = commands.add_parser("status")
    status.add_argument("plan_id")
    brief = commands.add_parser("brief", help="Read bounded task facts without running work")
    brief.add_argument("task_id")
    a = p.parse_args()
    if a.command=="verify-backup":
        from src.researchops.backup import verify_learning_backup
        print(json.dumps({"ok":True,"data":verify_learning_backup(a.folder)},ensure_ascii=False))
        return 0
    research = Research(a.project, a.root, readonly=a.command not in {"run", "register"})
    if a.command in {"account-proposal","learning-proposal"}:
        from src.researchops.account_proposal import account_from_learning
        from src.researchops.learning_proposal import learning_from_matrix
        from src.technical.artifacts import write_json,content_id
        output=a.output.resolve()
        if not output.is_relative_to(research.project) or any(p.is_symlink() or (hasattr(p,"is_junction") and p.is_junction()) for p in (a.output,*a.output.parents)):
            raise ValueError("Proposal output must be an ordinary project path")
        value=(account_from_learning(research,a.learning,_strict_json(a.template)) if a.command=="account-proposal"
               else learning_from_matrix(research,a.matrix,_strict_json(a.template)))
        if output.exists():
            if _strict_json(output)!=value:raise ValueError("Existing proposal differs; refusing overwrite")
            action="reused"
        else:
            output.parent.mkdir(parents=True,exist_ok=True)
            write_json(output,value);action="created"
        value=dict(path=str(output),action=action,proposal_id=content_id(value),model_fits=0,predict_calls=0,registered_experiments=0)
    elif a.command=="backup-learning":
        from src.researchops.backup import backup_learning_experiment
        value=backup_learning_experiment(research,a.learning,a.output,max_bytes=a.max_bytes)
    elif a.command == "register":
        ex = research.register(_strict_json(a.session), _strict_json(a.file))
        value = dict(experiment_id=ex["id"], task_id=ex["task_id"], status=ex["status"],
                     kind=ex["proposal"].get("kind", "technical"),
                     budget=research.brief(ex["task_id"])["budget"])
    elif a.command == "brief":
        value = research.brief(a.task_id)
    elif a.command == "status":
        value = ExecutionPlans(research).status(a.plan_id)
    else:
        value = ExecutionPlans(research).run(
            json.loads(a.session.read_text(encoding="utf-8-sig")),
            json.loads(a.file.read_text(encoding="utf-8-sig")),
            max_steps=a.steps,
        )
    print(json.dumps({"ok": True, "data": value}, ensure_ascii=False, allow_nan=False))
    return 1 if any(s["status"] == "blocked" for s in value.get("steps", [])) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        sys.exit(1)
