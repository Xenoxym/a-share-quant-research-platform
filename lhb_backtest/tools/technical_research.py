"""Price/volume research CLI. Archived event experiments use tools/workbench.py."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.technical.contracts import ResearchSpec
from src.technical.runner import run, verify


def main():
    p = argparse.ArgumentParser(description="A 股技术策略研究")
    commands = p.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--no-browser", action="store_true")
    execute = commands.add_parser("run")
    execute.add_argument("--spec", type=Path)
    batch = commands.add_parser("batch")
    batch.add_argument("--spec", type=Path, required=True)
    batch.add_argument("--split", default="2024-01-01")
    foundation = commands.add_parser("foundations")
    foundation.add_argument("--start", default="2020-01-01")
    foundation.add_argument("--end", default="2026-09-24")
    commands.add_parser("template")
    commands.add_parser("regimes")
    for name in ("verify", "replay"):
        sub = commands.add_parser(name)
        sub.add_argument("run", type=Path)
    args = p.parse_args()
    if args.command == "template":
        print(json.dumps(ResearchSpec().to_dict(), ensure_ascii=False, indent=2))
    elif args.command == "serve":
        from src.technical.server import serve
        serve(PROJECT, args.port, not args.no_browser)
    elif args.command == "verify":
        print(verify(args.run)[0]["run_id"], "verified")
    elif args.command == "batch":
        from src.technical.laboratory import run_batch
        spec = ResearchSpec.from_dict(json.loads(args.spec.read_text(encoding="utf-8-sig")))
        print(run_batch(PROJECT, spec, args.split, progress=lambda m: print(m, flush=True)))
    elif args.command == "foundations":
        from src.technical.foundation_lab import run_suite
        print(run_suite(PROJECT, start=args.start, end=args.end, progress=lambda m: print(m, flush=True)))
    elif args.command == "regimes":
        from src.technical.regime_lab import run_suite
        print(run_suite(PROJECT, progress=lambda m: print(m, flush=True)))
    else:
        spec = ResearchSpec.from_dict(json.loads(args.spec.read_text(encoding="utf-8-sig"))) if args.command == "run" and args.spec else ResearchSpec()
        folder = run(PROJECT, spec, replay=args.run if args.command == "replay" else None,
                     progress=lambda m: print(m, flush=True))
        print(folder, flush=True)


if __name__ == "__main__":
    main()
