"""Freeze data, definitions and code; evaluate one strategy under three cost models."""
from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import time
import uuid

import pandas as pd

from .artifacts import content_id, digest, exclusive_run, records, verify_artifacts, write_json
from .contracts import ResearchSpec, SCENARIOS, describe
from .data import prepare_snapshot
from .market import Market
from .portfolio import simulate
from .signals import decisions, features

PACKAGE = Path(__file__).resolve().parent


def code_inventory():
    return {p.relative_to(PACKAGE).as_posix(): digest(p) for p in sorted(PACKAGE.rglob("*"))
            if p.suffix in {".py", ".html", ".js", ".css"}}


def verify(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    verify_artifacts(folder, manifest)
    snapshot = folder.parent.parent / "snapshots" / manifest["snapshot_id"]
    if digest(snapshot / "manifest.json") != manifest["snapshot_manifest_sha256"]:
        raise ValueError("快照清单已变化")
    sm = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    verify_artifacts(snapshot, sm)
    return manifest, snapshot, sm


def run(project, spec=None, *, root=None, replay=None, progress=lambda m: None,
        prepared=None, scenarios=None, research_context=None, publish=True,
        frozen_snapshot=None, lock_root=None, registered_run_id=None):
    project = Path(project).resolve()
    root = Path(root or project / "data/technical").resolve()
    spec = spec or ResearchSpec()
    compact = bool(research_context and research_context.get("role") == "screen")
    # A batch owns the outer exclusive lock and supplies immutable shared inputs.
    if lock_root is not None and frozen_snapshot is None:
        raise ValueError("独立执行锁仅用于显式冻结快照")
    with (nullcontext() if prepared is not None else exclusive_run(Path(lock_root or root))):
        start_clock = time.perf_counter()
        if registered_run_id is not None:
            if (prepared is None or frozen_snapshot is None or lock_root is None or publish
                    or not isinstance(registered_run_id, str) or not re.fullmatch(r"\d{8}T\d{6}-[a-f0-9]{8}", registered_run_id)):
                raise ValueError("Registered run identity requires explicit private prepared account")
        run_id = registered_run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
        folder = root / "runs" / run_id
        folder.mkdir(parents=True)
        write_json(folder / "status.json", {"status": "running", "run_id": run_id})
        try:
            code = code_inventory()
            if prepared is not None:
                snapshot, sm = prepared["snapshot"], prepared["manifest"]
            elif frozen_snapshot is not None:
                snapshot = Path(frozen_snapshot).resolve()
                if snapshot.parent != (root / "snapshots").resolve():
                    raise ValueError("快照必须来自本研究库")
                sm = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
                verify_artifacts(snapshot, sm)
                if spec.experiment.start_date < sm["start"] or (spec.experiment.end_date != "latest" and spec.experiment.end_date > sm["end"]):
                    raise ValueError("实验区间超出冻结数据覆盖")
            elif replay:
                old, snapshot, sm = verify(replay)
                if old["code_hash"] != content_id(code):
                    raise ValueError("源码与原实验不同；使用原实验 code/ 及环境重放")
                spec = ResearchSpec.from_dict(old["spec"])
            else:
                snapshot, sm = prepare_snapshot(project, root, spec, progress)
            # latest becomes a concrete end date in the archived definition.
            end = sm["end"] if spec.experiment.end_date == "latest" else min(spec.experiment.end_date, sm["end"])
            spec = replace(spec, experiment=replace(spec.experiment, end_date=end))
            write_json(folder / "spec.json", spec.to_dict())
            definition = describe(spec)
            if prepared is not None and prepared.get("signal_definition"):
                # The shared account consumes registered external scores, not momentum.
                definition.update(prepared["signal_definition"])
                definition["signal_interface"] = "frozen_external_decisions"
            write_json(folder / "definition.json", definition)
            for name in code:
                target = folder / "code" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(PACKAGE / name, target)
            packages=["numpy", "pandas", "duckdb", "pyarrow", "pyyaml"]
            if spec.strategy.family == "allocation":
                packages += ["hmmlearn", "scikit-learn", "scipy"]
            write_json(folder / "environment.json", {"python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in packages}})
            calendar = json.loads((snapshot / "calendar.json").read_text(encoding="utf-8"))
            sessions = [d for d in calendar if spec.experiment.start_date <= d <= end]
            if not sessions:
                raise ValueError("实验区间没有交易日")
            progress("计算连续参考价收益、动量和可选均线，生成调仓日全部候选及排除理由")
            extras = prepared.get("extras", {}) if prepared is not None else {}
            if prepared is None and spec.strategy.family == "allocation":
                from .regimes import prepare_components, prepare_states, allocation_weights, combine_decisions
                small, defensive, schedule, market, mf = prepare_components(snapshot, spec)
                states, models = prepare_states(mf, spec.experiment.start_date)
                weights=allocation_weights(states, spec.strategy.allocation_policy, spec.strategy.state_breadth_threshold)
                decision_table=combine_decisions(small, defensive, weights, spec.strategy.allocation)
                extras={"state_probabilities.parquet":weights, "state_models.json":models,
                        "market_features.parquet":mf, "component_small.parquet":small, "component_defensive.parquet":defensive}
            elif prepared is None:
                bars = features(pd.read_parquet(snapshot / "bars.parquet"), calendar, spec.strategy)
                metadata = pd.read_parquet(snapshot / "metadata.parquet")
                financial = pd.read_parquet(snapshot / "fundamentals.parquet") if (snapshot / "fundamentals.parquet").exists() else None
                decision_table, schedule = decisions(bars, calendar, metadata, spec, financial,
                                                     pd.read_parquet(snapshot / "benchmark.parquet"),
                                                     pd.read_parquet(snapshot / "actions.parquet"))
                market = Market(bars, calendar, pd.read_parquet(snapshot / "actions.parquet"),
                    pd.read_parquet(snapshot / "status.parquet"), metadata, sm["missing_action_files"])
                del bars
            else:
                decision_table, schedule, market = prepared["decisions"], prepared["schedule"], prepared["market"]
            for name, value in extras.items():
                if Path(name).name != name:
                    raise ValueError("附加证据文件必须在实验目录")
                if name.endswith(".parquet"):
                    value.to_parquet(folder / name, index=False)
                else:
                    write_json(folder / name, value)
            decision_table.to_parquet(folder / "decisions.parquet", index=False)
            if not compact:
                decision_table.to_csv(folder / "decisions.csv", index=False, encoding="utf-8-sig")
            write_json(folder / "schedule.json", schedule)
            portfolios = []
            for scenario, label in SCENARIOS.items():
                if scenarios is not None and scenario not in scenarios:
                    continue
                progress(f"重算账户：{label}")
                result = simulate(decision_table, schedule, market, sessions, spec, scenario)
                target = folder / scenario
                target.mkdir()
                for key, frame in result.items():
                    if key != "metrics":
                        frame.to_parquet(target / f"{key}.parquet", index=False)
                        if not compact:
                            frame.to_csv(target / f"{key}.csv", index=False, encoding="utf-8-sig")
                portfolios.append({"metrics": result["metrics"], "equity": records(result["equity"])})
            benchmark = pd.read_parquet(snapshot / "benchmark.parquet")
            benchmark = benchmark.loc[benchmark.trade_date.isin(sessions)].sort_values("trade_date")
            bench_rows = []
            if len(benchmark) and benchmark.close.iloc[0] > 0:
                bench_rows = [{"date": r.trade_date, "nav": r.close / benchmark.close.iloc[0]} for r in benchmark.itertuples()]
            payload = {"run_id": run_id, "name": spec.strategy.name, "spec": spec.to_dict(), "definition": definition,
                "data_start": sessions[0], "data_end": sessions[-1], "snapshot_id": sm["snapshot_id"],
                "code_hash": content_id(code), "portfolios": portfolios,
                "benchmark": {"label": "沪深300价格参考（依据更新脚本识别）", "rows": bench_rows,
                    "identity_status": "inferred_from_refresh_script",
                    "definition": "tools/refresh_official_benchmark.py 指定 000300.SS；原始 benchmark 表缺指数代码列，身份依据采集代码，尚无表内独立凭据。首日收盘归一，不含分红，也不是可交易账户。"},
                "data_quality": {"quote_codes": sm["quote_codes"], "bar_rows": sm["counts"]["bars"],
                    "missing_action_files": len(sm["missing_action_files"]),
                    "master_codes_without_quotes_in_window": len(sm["master_codes_without_quotes_in_window"]),
                    "input_tables": ["daily_kline", "stock_metadata", "stock_status", "trade_days", "exrights", "benchmark"]},
                "decision_dates": sorted(schedule), "funnel": decision_table.reason.value_counts().to_dict(),
                "selected_rows": int(decision_table.selected.sum()), "elapsed_seconds": time.perf_counter()-start_clock,
                "status": "completed", "research_status": "retrospective_unoptimized_hypothesis",
                "research_context": research_context}
            if "fundamentals" in sm["counts"]:
                payload["data_quality"]["input_tables"].append("fundamentals")
                payload["data_quality"]["fundamental_policy"] = sm["fundamental_policy"]
            from .diagnostics import contributions, summarize, execution_delays
            configured = folder / "configured"
            if configured.exists():
                eq = pd.read_parquet(configured / "equity.parquet")
                attribution = contributions(eq, pd.read_parquet(configured / "positions.parquet"),
                    pd.read_parquet(configured / "ledger.parquet"), spec.execution.initial_cash)
                attribution.to_parquet(configured / "attribution.parquet", index=False)
                if not compact:
                    attribution.to_csv(configured / "attribution.csv", index=False, encoding="utf-8-sig")
                payload["analysis"] = summarize(eq, attribution, pd.read_parquet(configured / "warnings.parquet"),
                    spec.execution.initial_cash, payload["benchmark"])
                payload["analysis"]["execution"] = execution_delays(pd.read_parquet(configured / "orders.parquet"),
                    pd.read_parquet(configured / "targets.parquet"), eq.date.tolist())
            write_json(folder / "result.json", payload)
            from .web import render_page
            (folder / "report.html").write_text(render_page({"offline": True, "result": payload}), encoding="utf-8")
            if code != code_inventory():
                raise ValueError("运行期间源码变化，本次结果未发布")
            manifest = {"status": "completed", "run_id": run_id, "spec": spec.to_dict(), "code_hash": content_id(code),
                "snapshot_id": sm["snapshot_id"], "snapshot_manifest_sha256": digest(snapshot / "manifest.json"),
                "artifacts": {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob("*")
                    if p.is_file() and p.name != "status.json"}}
            write_json(folder / "manifest.json", manifest)
            write_json(folder / "status.json", {"run_id": run_id, "status": "completed"})
            if publish:
                write_json(root / "latest.json", {"run_id": run_id})
            return folder
        except Exception as exc:
            write_json(folder / "status.json", {"run_id": run_id, "status": "failed", "error": str(exc)})
            raise
