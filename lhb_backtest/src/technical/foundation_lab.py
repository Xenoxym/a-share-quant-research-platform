"""Predeclared, economically distinct baseline replication and evidence workflow."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import pandas as pd

from .artifacts import content_id, digest, exclusive_run, write_json
from .contracts import ResearchSpec, Strategy, Execution, Experiment, describe
from .data import prepare_snapshot
from .market import Market
from .runner import run, code_inventory
from .signals import decisions, features
from .foundation_review import build_evidence


def plan(start="2020-01-01", end="2026-09-24"):
    base = Strategy(top_k=100, min_history=252, lookback=252, skip=21, require_fundamentals=True)
    definitions = [
        ("cash", "零利息现金", "cash", "control", {}),
        *[(f"random_{seed}", f"同池随机底仓 · 种子 {seed}", "random", "random_control", {"seed": seed}) for seed in (11, 29, 47)],
        ("small", "小市值 · 最小100只", "size", "candidate", {}),
        ("small_ex_micro", "小市值 · 剔除最小30%后取100只", "size", "candidate", {"size_floor_quantile": .3}),
        ("small_positive_profit", "小市值 · 已公布年度盈利为正", "size", "candidate", {"positive_profit": True}),
        ("large", "大市值100只 · 规模对照", "large_size", "control", {}),
        ("value_earnings", "价值 · 年度盈利收益率", "earnings_yield", "candidate", {}),
        ("value_book", "价值 · 账面权益/市值", "book_to_price", "candidate", {}),
        ("profitability", "盈利能力 · 年度利润/权益", "profitability", "candidate", {}),
        ("low_volatility", "防御 · 63日低波动", "low_volatility", "candidate", {"lookback": 63, "skip": 0}),
        ("momentum", "动量 · 252/21 月度", "price", "candidate", {}),
        ("reversal", "反转 · 5日 周度", "price", "candidate", {"lookback": 5, "skip": 0, "direction": "reversal", "rebalance": "weekly"}),
        ("market_trend", "市场趋势 · MA126切换同种子底仓与现金", "market_trend", "candidate", {"trend_window": 126}),
    ]
    trials = []
    for key, name, family, role, params in definitions:
        spec = ResearchSpec(replace(base, name=name, family=family, **params), Execution(), Experiment(start, end))
        trials.append({"key": key, "name": name, "role": role, "spec": spec.to_dict(), "definition": describe(spec)})
    return {"version": 1, "kind": "foundation_replication", "trials": trials,
            "start_date": start, "end_date": end, "split_date": "2024-01-01",
            "notice": "15 个账户定义，包括 5 个对照；候选覆盖规模、价值、盈利能力、防御、动量、反转、趋势择时。含同家族对照，不称15个独立发现。所有历史已查看。",
            "cost_policy": "每个账户都独立重算零交易成本和配置成本；不把累计成本加回当反事实。",
            "data_policy": "共同252日连续历史与财务资格；已公布股本估算总市值，季度240天／年度550天陈旧上限。主板适配，不冒充论文全A股或多空原样复现。",
            "selection_policy": "不按总收益自动选冠军。联合时间块区间、逐年及区间证据只分配后续研究优先级，均需数据与真实执行确认。",
            "followup_policy": "小市值是事前指定主假设，首轮后必做其主要回撤和股本案例复核；进一步假设另行登记，不能改写本轮。"}


def run_suite(project, *, root=None, start="2020-01-01", end="2026-09-24", progress=lambda m: None):
    project = Path(project)
    root = Path(root or project / "data/technical")
    protocol = plan(start, end)
    with exclusive_run(root):
        begin = time.perf_counter()
        sid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8]
        folder = root / "foundations" / sid
        folder.mkdir(parents=True)
        protocol.update(suite_id=sid, code_hash=content_id(code_inventory()))
        write_json(folder / "protocol.json", protocol)
        write_json(folder / "status.json", {"suite_id": sid, "status": "running"})
        try:
            warmup = ResearchSpec.from_dict(protocol["trials"][4]["spec"])
            snapshot, sm = prepare_snapshot(project, root, warmup, progress)
            calendar = json.loads((snapshot / "calendar.json").read_text(encoding="utf-8"))
            raw = pd.read_parquet(snapshot / "bars.parquet")
            meta = pd.read_parquet(snapshot / "metadata.parquet")
            filings = pd.read_parquet(snapshot / "fundamentals.parquet")
            benchmark = pd.read_parquet(snapshot / "benchmark.parquet")
            market, cached, rows, last_feature_key = None, None, [], None
            from .signals import decision_dates
            dates = set(decision_dates(calendar, start, end, "monthly")) | set(decision_dates(calendar, start, end, "weekly"))
            with ThreadPoolExecutor(max_workers=2) as pool:
                for offset in range(0, len(protocol["trials"]), 2):
                    jobs = {}
                    for trial in protocol["trials"][offset:offset+2]:
                        spec = ResearchSpec.from_dict(trial["spec"])
                        s = spec.strategy
                        feature_key = (s.lookback, s.skip, s.trend_window, s.direction, s.family == "low_volatility")
                        if feature_key != last_feature_key:
                            cached = None
                            progress("计算基础特征："+trial["name"])
                            bars = features(raw, calendar, s)
                            if market is None:
                                market = Market(bars, calendar, pd.read_parquet(snapshot / "actions.parquet"),
                                    pd.read_parquet(snapshot / "status.parquet"), meta, sm["missing_action_files"])
                            cached = bars.loc[bars.trade_date.isin(dates)].copy()
                            last_feature_key = feature_key
                            del bars
                        table, schedule = decisions(cached, calendar, meta, spec, filings, benchmark,
                            pd.read_parquet(snapshot / "actions.parquet") if s.share_basis == "known_bonus" else None)
                        prepared = dict(snapshot=snapshot, manifest=sm, decisions=table, schedule=schedule, market=market)
                        future = pool.submit(run, project, spec, root=root, prepared=prepared,
                            scenarios=["zero_transaction_cost", "configured"], publish=False,
                            research_context={"role": "screen", "suite_id": sid, "key": trial["key"], "source": trial["definition"].get("source"), "hypothesis": trial["definition"].get("idea")})
                        jobs[future] = trial
                    for future in as_completed(jobs):
                        trial = jobs[future]
                        p = future.result()
                        result = json.loads((p / "result.json").read_text(encoding="utf-8"))
                        metrics = {q["metrics"]["scenario"]: q["metrics"] for q in result["portfolios"]}
                        rows.append({**trial, "run_id": p.name, "metrics": metrics["configured"],
                                     "zero_cost": metrics["zero_transaction_cost"], "execution": result["analysis"]["execution"]})
                        progress(f"完成 {len(rows)}/{len(protocol['trials'])}：{trial['name']}，含费 {metrics['configured']['total_return']:.2%}，零费 {metrics['zero_transaction_cost']['total_return']:.2%}")
                        write_json(folder / "progress.json", {"suite_id": sid, "rows": rows})
            ordering = {t["key"]: i for i, t in enumerate(protocol["trials"])}
            rows.sort(key=lambda r: ordering[r["key"]])
            progress("生成联合时间块证据、逐年表现、最差相对区间和个股贡献；不自动推荐冠军")
            evidence = build_evidence(root, rows, warmup.execution.initial_cash, protocol["split_date"])
            payload = {"suite_id": sid, "protocol": protocol, "rows": rows, "evidence": evidence,
                       "snapshot_id": sm["snapshot_id"], "status": "completed", "elapsed_seconds": time.perf_counter()-begin}
            if content_id(code_inventory()) != protocol["code_hash"]:
                raise ValueError("研究运行期间源码发生变化")
            write_json(folder / "result.json", payload)
            write_json(folder / "manifest.json", {"suite_id": sid, "artifacts": {p.name: digest(p) for p in folder.glob("*.json") if p.name != "status.json"},
                "run_manifests": {r["run_id"]: digest(root / "runs" / r["run_id"] / "manifest.json") for r in rows}})
            write_json(folder / "status.json", {"suite_id": sid, "status": "completed"})
            write_json(root / "latest_foundation.json", {"suite_id": sid})
            small = next(r for r in rows if r["key"] == "small")
            write_json(root / "latest.json", {"run_id": small["run_id"]})
            return folder
        except Exception as exc:
            write_json(folder / "status.json", {"suite_id": sid, "status": "failed", "error": str(exc)})
            raise
