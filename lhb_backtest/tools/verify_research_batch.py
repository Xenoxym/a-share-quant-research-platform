"""Audit the frozen trial set, train-only selection and candidate replay consistency."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.technical.artifacts import digest, verify_artifacts, write_json
from src.technical.runner import verify
from src.technical.laboratory import select, score_trial
from verify_technical import audit


def check(folder):
    folder = Path(folder)
    root = folder.parent.parent
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    verify_artifacts(folder, manifest)
    protocol = result["protocol"]
    assert len(result["rows"]) == protocol["trial_count"] == 24
    snapshot_ids = set()
    for row in result["rows"]:
        p = root / "runs" / row["run_id"]
        assert digest(p / "manifest.json") == manifest["trial_manifests"][p.name]
        trial, _, _ = verify(p)
        snapshot_ids.add(trial["snapshot_id"])
        declared = protocol["trials"][row["trial"]-1]["spec"]
        assert trial["spec"] == declared
        fresh = score_trial(pd.read_parquet(p / "configured/equity.parquet"),
                            pd.read_parquet(p / "configured/fills.parquet"), declared["execution"]["initial_cash"], protocol["split_date"])
        assert fresh["screen_score"] == row["screen_score"]
        assert fresh["validation"] == row["validation"]
    assert snapshot_ids == {result["snapshot_id"]}
    chosen = select(result["rows"])
    assert result["selected_trial"] == chosen["trial"]
    confirmation = root / "runs" / result["confirmation_run"]
    assert digest(confirmation / "manifest.json") == manifest["confirmation_manifest"]
    for name in ["equity", "orders", "fills", "ledger", "positions", "warnings", "targets", "attribution"]:
        pd.testing.assert_frame_equal(pd.read_parquet(confirmation / "configured" / f"{name}.parquet"),
            pd.read_parquet(root / "runs" / chosen["run_id"] / "configured" / f"{name}.parquet"))
    return {"batch_id": result["batch_id"], "status": "pass", "trials": len(result["rows"]),
            "checks": ["frozen protocol and 24 manifests intact", "all trials use one snapshot", "all declared specs match runs",
                       "all split metrics independently recomputed", "selection uses only screening segment", "confirmation matches screening candidate across 8 tables"],
            "confirmation_accounting": audit(confirmation)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/technical/validation/v3/batch_audit.json"))
    args = parser.parse_args()
    value = check(args.batch)
    write_json(args.output, value)
    print(json.dumps(value, indent=2))
