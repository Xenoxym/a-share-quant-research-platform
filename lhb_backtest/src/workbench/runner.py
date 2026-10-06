"""One reproducible workflow shared by CLI, launcher and local web UI."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time
import uuid

import pandas as pd

from .artifacts import (content_id, digest, exclusive_run, records, utc_now,
                        verify_artifacts, write_json)
from .contracts import ASSUMPTIONS, FEATURE_REGISTRY, StrategyCard
from .data import prepare_snapshot
from .portfolio import Market, simulate
from .research import describe_research, price_labels, select_cohorts

PACKAGE = Path(__file__).resolve().parent


def code_inventory():
    return {str(path.relative_to(PACKAGE)).replace("\\", "/"): digest(path)
            for path in sorted(PACKAGE.rglob("*")) if path.suffix in {".py", ".html", ".css", ".js"}}


def verify_run(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("实验尚未完成")
    verify_artifacts(folder, manifest)
    snapshot = folder.parent.parent / "snapshots" / manifest["snapshot_id"]
    snapshot_manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if digest(snapshot / "manifest.json") != manifest["snapshot_manifest_sha256"]:
        raise ValueError("快照清单被修改")
    verify_artifacts(snapshot, snapshot_manifest)
    return manifest, snapshot, snapshot_manifest


def run_research(project, card=None, *, root=None, progress=None, config_path=None, replay=None):
    project = Path(project).resolve()
    root = Path(root or project / "data/workbench").resolve()
    card = card or StrategyCard()
    with exclusive_run(root):
        started = time.perf_counter()
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        folder = root / "runs" / run_id
        folder.mkdir(parents=True)
        write_json(folder / "card.json", card.to_dict())
        state = {"run_id": run_id, "status": "running", "started_at": utc_now(), "messages": []}

        def report(message):
            state["messages"].append({"at": utc_now(), "message": message})
            write_json(folder / "status.json", state)
            if progress:
                progress(message)

        try:
            source_code = code_inventory()
            source_hash = content_id(source_code)
            if replay:
                old, snapshot, data_manifest = verify_run(replay)
                if old["code_hash"] != source_hash:
                    raise ValueError("当前工作台源码与原实验不同；使用保存的 code/ 源码和 environment.json 重建环境后重放")
                card = StrategyCard.from_dict(old["card"])
                report("已验证历史输出和数据快照，使用冻结数据重放")
            else:
                snapshot, data_manifest = prepare_snapshot(project, root, card, report, config_path)
            write_json(folder / "card.json", card.to_dict())
            write_json(folder / "feature_registry.json", FEATURE_REGISTRY)
            write_json(folder / "assumptions.json", ASSUMPTIONS)
            from .rules import RULE_SOURCES
            write_json(folder / "rule_sources.json", RULE_SOURCES)
            write_json(folder / "environment.json", {"python": platform.python_version(),
                       "packages": {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "pyarrow", "duckdb", "pyyaml"]}})
            for name in source_code:
                target = folder / "code" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(PACKAGE / name, target)
            calendar = json.loads((snapshot / "calendar.json").read_text(encoding="utf-8"))
            sessions = [d for d in calendar if card.start_date <= d <= data_manifest["end"]]
            events = pd.read_parquet(snapshot / "events.parquet")
            report("按当时可用特征生成信号，并建立同日市值、动量、流动性配对")
            events, pairs = select_cohorts(events, card)
            bars = pd.read_parquet(snapshot / "bars.parquet")
            market = Market(bars, calendar, pd.read_parquet(snapshot / "actions.parquet"),
                            pd.read_parquet(snapshot / "status.parquet"),
                            pd.read_parquet(snapshot / "metadata.parquet"), data_manifest["missing_action_files"])
            del bars
            report("生成固定期限价格标签、年度描述及时间块不确定性区间")
            labels = price_labels(events, market, card, data_manifest["end"])
            events, pairs, research = describe_research(events, pairs, labels, sessions, card)
            events.to_parquet(folder / "events.parquet", index=False)
            pairs.to_parquet(folder / "pairs.parquet", index=False)
            labels.to_parquet(folder / "labels.parquet", index=False)
            portfolio_specs = [
                ("institutional", events.signal, 1., "完整机构信号组合"),
                ("baseline", events.eligibility_reason.eq("eligible"), 1., "全部基础事件组合"),
                ("institutional_matched", events.matched_treatment, 1., "配对机构组合"),
                ("matched_control", events.matched_control, 1., "配对对照组合"),
                ("institutional_stress", events.signal, 2., "机构组合：费用与滑点压力")]
            portfolios = []
            for name, mask, multiplier, label in portfolio_specs:
                report(f"模拟 {label}：{int(mask.sum()):,} 个候选事件")
                result = simulate(events.loc[mask], market, sessions, card, name, multiplier)
                destination = folder / name
                destination.mkdir()
                for table, frame in result.items():
                    if table != "metrics":
                        frame.to_parquet(destination / f"{table}.parquet", index=False)
                metrics = result["metrics"]
                metrics["label"] = label
                write_json(destination / "metrics.json", metrics)
                portfolios.append({"metrics": metrics, "equity": records(result["equity"])})
            if source_code != code_inventory():
                raise RuntimeError("计算期间工作台源码发生变化，结果未发布；请重试")
            report("账户对账完成，生成可追溯的交互报告")
            payload = {"run_id": run_id, "name": card.name, "card": card.to_dict(),
                       "snapshot_id": data_manifest["snapshot_id"], "code_hash": source_hash,
                       "data_start": sessions[0], "data_end": data_manifest["end"],
                       "research": research, "portfolios": portfolios, "assumptions": ASSUMPTIONS,
                       "features": FEATURE_REGISTRY, "snapshot_counts": data_manifest["counts"],
                       "status": "completed", "elapsed_seconds": time.perf_counter() - started}
            write_json(folder / "result.json", payload)
            from .report import export_report
            export_report(folder, payload, events, snapshot)
            artifacts = {str(p.relative_to(folder)).replace("\\", "/"): digest(p)
                         for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "status.json"}
            manifest = {"run_id": run_id, "status": "completed", "created_at": utc_now(),
                        "snapshot_id": data_manifest["snapshot_id"], "card": card.to_dict(),
                        "snapshot_manifest_sha256": digest(snapshot / "manifest.json"),
                        "code_hash": source_hash, "code": source_code, "artifacts": artifacts,
                        "replayed_from": str(replay) if replay else None}
            write_json(folder / "manifest.json", manifest)
            state["status"] = "completed"
            state["elapsed_seconds"] = time.perf_counter() - started
            report("研究完成；报告、账户流水、原始记录引用与完整性清单已保存")
            write_json(root / "latest.json", {"run_id": run_id})
            return folder
        except BaseException as exc:
            state["status"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
            state["error"] = str(exc)
            write_json(folder / "status.json", state)
            raise
