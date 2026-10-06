"""Forensic counterfactuals for an archived template, not a new strategy.

Load the archived implementation and frozen inputs. Change only its fee
function, rerun the account, and keep all outputs outside the original run.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.workbench.artifacts import digest, write_json
from src.workbench.runner import verify_run


def audit(folder, output):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    manifest, snapshot, data_manifest = verify_run(folder)
    if output.exists():
        raise ValueError("审计输出已存在；请使用新的目录，保留已有证据")
    output.mkdir(parents=True)
    (output / "audit_source.py").write_bytes(Path(__file__).read_bytes())
    spec = importlib.util.spec_from_file_location(
        "archived_cost_audit", folder / "code/__init__.py",
        submodule_search_locations=[str(folder / "code")])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    portfolio = importlib.import_module(spec.name + ".portfolio")
    contracts = importlib.import_module(spec.name + ".contracts")
    card = contracts.StrategyCard.from_dict(manifest["card"])
    calendar = json.loads((snapshot / "calendar.json").read_text(encoding="utf-8"))
    sessions = [d for d in calendar if card.start_date <= d <= data_manifest["end"]]
    market = portfolio.Market(pd.read_parquet(snapshot / "bars.parquet"), calendar,
        pd.read_parquet(snapshot / "actions.parquet"), pd.read_parquet(snapshot / "status.parquet"),
        pd.read_parquet(snapshot / "metadata.parquet"), data_manifest["missing_action_files"])
    events = pd.read_parquet(folder / "events.parquet")
    original = portfolio.fees
    rows = []
    for scenario in ("original", "zero_slippage", "zero_transaction_cost"):
        def fee_model(value, side, day, config):
            cost = original(value, side, day, config)
            if scenario == "zero_transaction_cost":
                return {k: 0. for k in cost}
            if scenario == "zero_slippage":
                cost["slippage"] = 0.
                cost["total"] = sum(cost[k] for k in ("commission", "other_fee", "stamp_tax", "slippage"))
            return cost
        for name, mask, multiplier in [
            ("institutional", events.signal, 1.),
            ("baseline", events.eligibility_reason.eq("eligible"), 1.),
            ("institutional_matched", events.matched_treatment, 1.),
            ("matched_control", events.matched_control, 1.),
            ("institutional_stress", events.signal, 2.)]:
            print(f"{scenario} / {name}", flush=True)
            with patch.object(portfolio, "fees", fee_model):
                result = portfolio.simulate(events.loc[mask], market, sessions, card, name, multiplier)
            if scenario == "original":
                for table in ("equity", "fills", "orders", "ledger", "trades", "positions", "warnings"):
                    pd.testing.assert_frame_equal(result[table], pd.read_parquet(folder / name / f"{table}.parquet"), check_exact=True)
            target = output / scenario / name
            target.mkdir(parents=True)
            for key, frame in result.items():
                if key != "metrics":
                    frame.to_parquet(target / f"{key}.parquet", index=False)
            metric = result["metrics"]
            metric.update(scenario=scenario, cost_components={
                c: float(result["fills"][c].sum()) for c in ("commission", "other_fee", "stamp_tax", "slippage")})
            rows.append(metric)
            write_json(output / "progress.json", {"completed": len(rows), "total": 15, "rows": rows})
    result = {"source_run": folder.name, "snapshot_id": manifest["snapshot_id"],
        "source_code_hash": manifest["code_hash"], "audit_script_sha256": digest(Path(__file__)),
        "original_reproduced_exactly": True, "rows": rows,
        "limits": ["All scenarios preserve price-limit, halt, lot-size, volume and settlement assumptions.",
                   "Zero transaction cost removes commission, other fees, stamp tax and slippage, not dividend tax reserve.",
                   "Cash, quantities and subsequent trades are recomputed; this is not fee addback.",
                   "These are diagnostics of a retired template, not the technical strategy baseline."]}
    write_json(output / "result.json", result)
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows]).to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")
    write_json(output / "manifest.json", {"artifacts": {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}})
    print(json.dumps([{k: r[k] for k in ("name", "scenario", "total_return", "max_drawdown", "cost")} for r in rows]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.output)
