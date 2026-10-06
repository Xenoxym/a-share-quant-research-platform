"""Inspect registered ML results; execution stays on research.py register/run."""
import argparse
import json
from pathlib import Path
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.mlresearch.contracts import benchmark_proposal, completion_proposal
from src.technical.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description="机器学习基准设计与结果")
    parser.add_argument("--root", type=Path, default=PROJECT / "data/technical")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--snapshot", default="31eef7349722ec89cd1a3743")
    plan.add_argument("--output", type=Path)
    plan.add_argument("--complete", action="store_true", help="固定完整研究矩阵，复用原基准模型并生成账户")
    plan.add_argument("--source-run", default="20261003T165732-991ae261")
    commands.add_parser("list")
    for name in ["show", "verify"]:
        commands.add_parser(name).add_argument("run_id")
    args = parser.parse_args()
    if args.command == "plan":
        value = completion_proposal(args.source_run, args.snapshot) if args.complete else benchmark_proposal(args.snapshot)
        if args.output:
            if args.output.exists():
                raise ValueError("设计文件已存在，避免覆盖旧研究协议")
            write_json(args.output, value)
    elif args.command == "list":
        value = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((args.root / "ml_runs").glob("*/status.json"), reverse=True)]
    else:
        if not re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{8}", args.run_id):
            raise ValueError("ML运行编号无效")
        folder = args.root / "ml_runs" / args.run_id
        if args.command == "show":
            value = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        else:
            from src.mlresearch.audit import audit
            value = audit(folder, args.root)
    print(json.dumps({"ok": True, "data": value}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        sys.exit(1)
