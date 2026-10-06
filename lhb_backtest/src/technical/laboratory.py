"""Bounded, declared 24-trial screen. Historical validation is explicitly retrospective."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import pandas as pd

from .artifacts import content_id, digest, exclusive_run, write_json
from .contracts import ResearchSpec
from .data import prepare_snapshot
from .diagnostics import period_stats
from .market import Market
from .runner import code_inventory, run
from .signals import decisions, features

GRID = [("momentum", 63, 0), ("momentum", 126, 21), ("momentum", 252, 21),
        ("reversal", 5, 0), ("reversal", 10, 0), ("reversal", 20, 0)]


def plan(base, split_date):
    if base.experiment.end_date == "latest":
        raise ValueError("批量研究结束日期必须是具体日期，不能为 latest")
    if not base.experiment.start_date < split_date < base.experiment.end_date:
        raise ValueError("筛选分界日必须位于研究区间内；结束日期请使用具体日期")
    pd.Timestamp(split_date)  # fail before any experiment is created
    trials = []
    for direction, lookback, skip in GRID:
        for trend in (0, 126):
            for freq in ("monthly", "weekly"):
                name = f"{'动量' if direction == 'momentum' else '短期反转'} {lookback}/{skip} · {'MA126' if trend else '无趋势过滤'} · {'月调仓' if freq == 'monthly' else '周调仓'}"
                spec = replace(base, strategy=replace(base.strategy, name=name, direction=direction,
                    lookback=lookback, skip=skip, trend_window=trend, rebalance=freq, min_score=None,
                    family="price", require_fundamentals=False, positive_profit=False, size_floor_quantile=0.))
                trials.append({"trial": len(trials)+1, "spec": spec.to_dict()})
    return {"version": 1, "split_date": split_date, "trial_count": len(trials), "trials": trials,
            "selection": "仅按分界日前含费日收益均值/样本标准差×√252 排名（无风险收益设为0），平局按试验编号；无成交或无波动不参加。",
            "validation": "分界日及以后只报告，不用于选出候选；承接原账户持仓。历史已被研究者查看，不能称为未见样本外。",
            "followup": "自动为筛选段第一名重算三种成本。验证段收益为正且区间回撤不低于 -30% 才进入稳健性复核，否则暂缓；该门槛是研究排队规则，不是上线条件。",
            "hypotheses": ["中期相对强势可能延续", "短期下跌可能反弹", "趋势过滤是否减少逆势暴露", "更频繁调仓是否值得增加交易成本"],
            "sources": ["https://www.aqr.com/Insights/Datasets/Momentum-Indices-Monthly", "https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf"]}


def score_trial(eq, fills, initial, split):
    train = eq.loc[eq.date.lt(split)]
    valid = eq.loc[eq.date.ge(split)]
    if len(train) < 126 or len(valid) < 63:
        raise ValueError("批量研究至少需要 126 个筛选交易日和 63 个后续验证交易日")
    ret = train.equity.div(train.equity.shift().fillna(initial)).sub(1)
    score = float(ret.mean()/ret.std(ddof=1)*252**.5) if ret.std(ddof=1) > 1e-12 and (fills.date < split).any() else None
    return {"screen": period_stats(eq, initial, train.date.iloc[0], train.date.iloc[-1]),
            "validation": period_stats(eq, initial, valid.date.iloc[0], valid.date.iloc[-1]), "screen_score": score}


def select(rows):
    eligible = [r for r in rows if r["screen_score"] is not None]
    return min(eligible, key=lambda r: (-r["screen_score"], r["trial"])) if eligible else None


def run_batch(project, base, split_date, *, root=None, progress=lambda m: None, parent_run_id=None):
    project = Path(project)
    root = Path(root or project / "data/technical")
    protocol = plan(base, split_date)
    with exclusive_run(root):
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        folder = root / "batches" / batch_id
        folder.mkdir(parents=True)
        protocol.update(batch_id=batch_id, parent_run_id=parent_run_id, code_hash=content_id(code_inventory()))
        write_json(folder / "protocol.json", protocol)  # fixed BEFORE outcomes exist
        write_json(folder / "status.json", {"status": "running", "batch_id": batch_id})
        try:
            warmup_spec = replace(base, strategy=replace(base.strategy, lookback=252, skip=21, trend_window=126))
            snapshot, sm = prepare_snapshot(project, root, warmup_spec, progress)
            calendar = json.loads((snapshot / "calendar.json").read_text(encoding="utf-8"))
            sessions = [d for d in calendar if base.experiment.start_date <= d <= sm["end"]]
            if sum(d < split_date for d in sessions) < 126 or sum(d >= split_date for d in sessions) < 63:
                raise ValueError("批量研究至少需要 126 个筛选交易日和 63 个后续验证交易日")
            raw = pd.read_parquet(snapshot / "bars.parquet")
            metadata = pd.read_parquet(snapshot / "metadata.parquet")
            market, rows = None, []
            # Two accounts run concurrently; features and immutable quote index are shared.
            with ThreadPoolExecutor(max_workers=2) as pool:
                for pair in range(0, len(protocol["trials"]), 2):
                    declarations = protocol["trials"][pair:pair+2]
                    first_spec = ResearchSpec.from_dict(declarations[0]["spec"])
                    progress(f"特征组 {pair//2+1}/12：{first_spec.strategy.name}")
                    bars = features(raw, calendar, first_spec.strategy)
                    if market is None:
                        market = Market(bars, calendar, pd.read_parquet(snapshot / "actions.parquet"),
                            pd.read_parquet(snapshot / "status.parquet"), metadata, sm["missing_action_files"])
                    jobs = {}
                    for declaration in declarations:
                        spec = ResearchSpec.from_dict(declaration["spec"])
                        table, schedule = decisions(bars, calendar, metadata, spec)
                        prepared = dict(snapshot=snapshot, manifest=sm, decisions=table, schedule=schedule, market=market)
                        job = pool.submit(run, project, spec, root=root, prepared=prepared, scenarios=["configured"], publish=False,
                            research_context={"batch_id": batch_id, "trial": declaration["trial"], "role": "screen", "split_date": split_date})
                        jobs[job] = declaration
                    del bars
                    for job in as_completed(jobs):
                        p = job.result()
                        declaration = jobs[job]
                        r = json.loads((p / "result.json").read_text(encoding="utf-8"))
                        scored = score_trial(pd.read_parquet(p / "configured/equity.parquet"), pd.read_parquet(p / "configured/fills.parquet"), base.execution.initial_cash, split_date)
                        rows.append({"trial": declaration["trial"], "run_id": p.name, "name": r["name"], "metrics": r["portfolios"][0]["metrics"], **scored})
                        progress(f"已完成 {len(rows)}/24：{r['name']}")
                        write_json(folder / "progress.json", {"rows": sorted(rows, key=lambda r: r["trial"])})
            chosen = select(rows)
            confirmation = None
            if chosen:
                spec = ResearchSpec.from_dict(protocol["trials"][chosen["trial"]-1]["spec"])
                bars = features(raw, calendar, spec.strategy)
                table, schedule = decisions(bars, calendar, metadata, spec)
                del bars
                progress("对筛选段选出的候选重算三个费用情景；不按验证段更换候选")
                confirmation = run(project, spec, root=root,
                    prepared=dict(snapshot=snapshot, manifest=sm, decisions=table, schedule=schedule, market=market),
                    research_context={"batch_id": batch_id, "trial": chosen["trial"], "role": "confirmation", "split_date": split_date}, publish=False)
            outcome = "无可筛选候选" if not chosen else "进入稳健性复核" if chosen["validation"]["return"] > 0 and chosen["validation"]["max_drawdown"] >= -.30 else "暂缓：验证未过研究门槛"
            payload = {"batch_id": batch_id, "protocol": protocol, "rows": sorted(rows, key=lambda r: (r["screen_score"] is None, -(r["screen_score"] or 0), r["trial"])),
                       "selected_trial": chosen["trial"] if chosen else None, "confirmation_run": confirmation.name if confirmation else None,
                       "outcome": outcome, "snapshot_id": sm["snapshot_id"], "status": "completed",
                       "next_steps": ["检查候选的最大回撤、成本拖累和损益集中股票", "围绕同一假设预登记局部参数扰动及分时段稳定性复核", "新数据到来后积累前向模拟记录；历史排名不等于可交易能力"]}
            if content_id(code_inventory()) != protocol["code_hash"]:
                raise ValueError("批量运行期间源码发生变化，结果未发布")
            write_json(folder / "result.json", payload)
            write_json(folder / "manifest.json", {"artifacts": {p.name: digest(p) for p in folder.glob("*.json") if p.name != "status.json"},
                "trial_manifests": {r["run_id"]: digest(root / "runs" / r["run_id"] / "manifest.json") for r in rows},
                "confirmation_manifest": digest(confirmation / "manifest.json") if confirmation else None})
            write_json(folder / "status.json", {"status": "completed", "batch_id": batch_id})
            write_json(root / "latest_batch.json", {"batch_id": batch_id})
            return folder
        except Exception as exc:
            write_json(folder / "status.json", {"status": "failed", "batch_id": batch_id, "error": str(exc)})
            raise
