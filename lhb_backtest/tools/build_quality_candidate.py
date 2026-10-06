"""Build a reversible candidate by restoring lost baseline records/cells."""
import argparse
from pathlib import Path
import json
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_sources.snapshot_merge import preserve_snapshot_history

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--current', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Candidate output already exists; refusing overwrite')
    shutil.copytree(a.current, a.output)
    result = preserve_snapshot_history(a.baseline, a.output)
    (a.output / 'history_restoration.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
