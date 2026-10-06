"""Offline end-to-end rebuild, isolated from all production outputs.

Exit 0 means pipeline invariants passed, not that the data is research-ready.
Read validation.json: snapshot readiness is reported separately and explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from main import step_clean_data, step_build_features, step_generate_labels
from src.backtest.filter_engine import run_lhb_backtest
from src.backtest.grid_search import grid_search
from src.data_sources.simtradedata_metadata import resolve_export_dir
from src.data_sources.snapshot_quality import inspect_snapshot
from src.reports.make_report import generate_html_report


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    import faulthandler
    faulthandler.dump_traceback_later(180, repeat=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--export-dir", type=Path, help="Isolated candidate vendor data; rebuild raw K line too")
    args = parser.parse_args()
    dest = (args.output_dir or ROOT / "data/validation" / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    dest.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier run.
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    raw = (ROOT / config["paths"]["raw_dir"]).resolve()
    source_hashes = {p.name: digest(p) for p in raw.glob("*.parquet")}
    original_raw = raw
    if args.export_dir:
        candidate = args.export_dir.resolve()
        os.environ['LHB_SIMTRADEDATA_EXPORT_DIR'] = str(candidate)
        from src.utils.calendar import invalidate_cache
        invalidate_cache()
        config['simtradedata']['export_dir'] = str(candidate)
        raw = dest / 'raw'
        raw.mkdir()
        for p in original_raw.glob('*.parquet'):
            if p.name != 'daily_kline.parquet':
                shutil.copy2(p,raw/p.name)
        from src.ingestion.load_kline_simtradedata import load_kline_simtradedata
        end = str(pd.read_parquet(original_raw/'lhb_summary.parquet',columns=['trade_date']).trade_date.max())[:10]
        load_kline_simtradedata(config['defaults']['start_date'],end,export_dir=candidate,
                               save_path=str(raw/'daily_kline.parquet'),incremental=False)
    for part in ("clean", "factor", "output"):
        config["paths"][part + "_dir"] = str(dest / part)
    config["paths"]["raw_dir"] = str(raw)
    (dest / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    step_clean_data(config)
    step_build_features(config)
    step_generate_labels(config)
    event_path = dest / "factor/lhb_event_labeled.parquet"
    events = pd.read_parquet(event_path)
    assert events.label_schema_version.eq(2).all()
    assert not events.duplicated(["stock_code", "trade_date"]).any()
    result = run_lhb_backtest(event_path=str(event_path),
                              filters=config["analysis"]["default_filters"], round_trip_cost=.002)
    assert len(result.detail) > 0, "No eligible historical events remain"
    assert result.detail.is_st.eq(0).all()
    assert result.detail.next_day_untradable.eq(0).all()
    assert "net_entry_return_1d" not in result.summary
    result.to_csv(str(dest / "output/backtest.csv"))
    grid = grid_search({"min_net_buy_ratio": [0., .03]}, event_path=str(event_path),
                       base_filters=config["analysis"]["default_filters"],
                       oos_split_date=config["analysis"]["oos_split_date"],
                       round_trip_cost=.002, generate_plots=False,
                       save_path=str(dest / "output/grid_training_rank.csv"))
    assert "oos_net_oo_return_1d" in grid
    assert len(grid) == 2
    import matplotlib
    matplotlib.use("Agg")
    generate_html_report(result, output_dir=str(dest / "output"), title="事件研究口径验证（不是组合净值）")
    readiness = inspect_snapshot(resolve_export_dir(), str(events.trade_date.max()), str(events.trade_date.min()))
    assert source_hashes == {p.name: digest(p) for p in original_raw.glob("*.parquet")}, "Original raw inputs changed during validation"
    clean = pd.read_parquet(dest / "clean/daily_kline.parquet", columns=["limit_price_status"])
    report = {
        "pipeline_invariants": "pass", "snapshot_readiness": readiness,
        "source_sha256": source_hashes, "events": len(events), "eligible_events": len(result.detail),
        "vendor_export_dir": str(resolve_export_dir()),
        "rebuilt_raw": str(raw) if args.export_dir else None,
        "limit_price_status_counts": clean.limit_price_status.value_counts().to_dict(),
        "eligible_date_end": str(result.detail.trade_date.max()),
        "unknown_st_events": int(events.is_st.isna().sum()),
        "entry_alpha_5d_count": int(events.entry_alpha_5d.notna().sum()),
        "limitations": ["Price labels are not executable portfolio PnL or total returns.",
                        "OOS has been viewed historically; this run creates no fresh final test."],
    }
    (dest / "validation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(dest), "pipeline": "pass", "data_ready": readiness["status"],
                      "events": len(events), "eligible": len(result.detail)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
