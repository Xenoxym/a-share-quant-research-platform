"""CLI entry point; the same strategy card is used by the browser UI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.workbench.contracts import StrategyCard
from src.workbench.runner import run_research, verify_run


def main():
    parser = argparse.ArgumentParser(description="机构披露净买额研究工作台")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="打开本地策略卡片和研究界面")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-browser", action="store_true")
    serve.add_argument("--root", type=Path)
    run = commands.add_parser("run", help="运行完整模板")
    run.add_argument("--card", type=Path, help="JSON 策略卡片")
    run.add_argument("--start-date")
    run.add_argument("--end-date")
    run.add_argument("--root", type=Path)
    run.add_argument("--config", type=Path, help="现有项目 config.yaml")
    replay = commands.add_parser("replay", help="验证并重放一个已冻结的实验")
    replay.add_argument("run_dir", type=Path)
    check = commands.add_parser("verify", help="校验实验与冻结数据完整性")
    check.add_argument("run_dir", type=Path)
    commands.add_parser("template", help="输出默认策略卡片 JSON")
    args = parser.parse_args()
    if args.command == "serve":
        from src.workbench.server import serve
        return serve(PROJECT, args.root, args.port, not args.no_browser)
    if args.command == "template":
        print(json.dumps(StrategyCard().to_dict(), ensure_ascii=False, indent=2))
        return
    if args.command == "verify":
        manifest, _, _ = verify_run(args.run_dir)
        print(f"完整性检查通过: {manifest['run_id']}")
        return
    if args.command == "replay":
        root = args.run_dir.resolve().parent.parent
        folder = run_research(PROJECT, root=root, replay=args.run_dir, progress=lambda m: print(m, flush=True))
    else:
        values = json.loads(args.card.read_text(encoding="utf-8-sig")) if args.card else {}
        for field in ("start_date", "end_date"):
            if getattr(args, field):
                values[field] = getattr(args, field)
        folder = run_research(PROJECT, StrategyCard.from_dict(values), root=args.root,
                              config_path=args.config, progress=lambda m: print(m, flush=True))
    print(f"研究完成: {folder}\n交互报告: {folder / 'report.html'}", flush=True)


if __name__ == "__main__":
    main()
