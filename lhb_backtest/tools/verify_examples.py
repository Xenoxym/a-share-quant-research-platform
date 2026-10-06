"""Run local analysis examples and research scripts with bounded runtimes.

Acquisition and full rebuild examples are excluded: validate those separately.
Scripts may write their normal charts/reports under data/output.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scripts = [p for p in sorted((ROOT / "samples").glob("sample_*.py"))
               if int(p.name.split("_")[1]) >= 3]
    scripts += sorted((ROOT / "research").glob("exp*.py"))
    results = []
    for script in scripts:
        started = time.perf_counter()
        with (args.output_dir / (script.stem + ".log")).open("w", encoding="utf-8") as log:
            try:
                completed = subprocess.run([sys.executable, str(script)], cwd=ROOT,
                    env={**os.environ, "PYTHONUTF8": "1", "MPLBACKEND": "Agg"},
                    stdout=log, stderr=subprocess.STDOUT, timeout=180, check=False)
                status = "pass" if completed.returncode == 0 else "fail"
            except subprocess.TimeoutExpired:
                status = "timeout"
        row = {"script": str(script.relative_to(ROOT)), "status": status,
               "seconds": time.perf_counter() - started}
        results.append(row)
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
    raise SystemExit(any(row["status"] != "pass" for row in results))


if __name__ == "__main__":
    main()
