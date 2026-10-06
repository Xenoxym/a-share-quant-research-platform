"""Machine-readable comparative evidence for researchers and the workbench.

All labels are retrospective research triage, never a deployment decision.
"""
import numpy as np
import pandas as pd

from .artifacts import records
from .diagnostics import period_stats


def joint_block_intervals(differences, blocks=(21, 63), samples=1500, seed=20260927):
    """Joint max-standardized bootstrap band for mean daily active returns * 252.

    Same circular time blocks resample all columns jointly, retaining paired dates
    and cross-strategy dependence. This is an exploratory finite-history band, not
    a claim to remove adaptive research bias or future structural breaks.
    """
    x = np.asarray(differences, dtype=float)
    if x.ndim != 2 or len(x) < max(blocks)*2 or not np.isfinite(x).all():
        raise ValueError("联合重采样需要完整、对齐且足够长的逐日增量矩阵")
    rng = np.random.default_rng(seed)
    center = x.mean(axis=0)*252
    out = []
    for block in blocks:
        n = len(x)
        count = (n+block-1)//block
        means = []
        for _ in range(samples):
            starts = rng.integers(0, n, count)
            idx = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
            means.append(x[idx].mean(axis=0)*252)
        means = np.asarray(means)
        scale = means.std(axis=0, ddof=1)
        z = np.abs(means-center)/np.where(scale > 1e-14, scale, 1.)
        critical = float(np.quantile(z.max(axis=1), .95))
        out.append({"block_sessions": block, "estimate": center.tolist(),
                    "lower": (center-critical*scale).tolist(), "upper": (center+critical*scale).tolist(),
                    "joint_critical": critical, "samples": samples})
    return out


def build_evidence(root, rows, initial_cash, split_date):
    frames, returns = {}, {}
    for row in rows:
        rid = row["run_id"]
        f = pd.read_parquet(root / "runs" / rid / "configured/equity.parquet").set_index("date").sort_index()
        frames[row["key"]] = f
        returns[row["key"]] = f.equity.div(f.equity.shift().fillna(initial_cash)).sub(1)
    matrix = pd.DataFrame(returns)
    if matrix.isna().any().any():
        raise ValueError("基础策略账户日期未完全对齐")
    controls = [r["key"] for r in rows if r["role"] == "random_control"]
    # Analytical daily mixture; not an independently executed pooled account.
    reference = matrix[controls].mean(axis=1)
    candidates = [r["key"] for r in rows if r["role"] == "candidate"]
    active = matrix[candidates].sub(reference, axis=0)
    bands = joint_block_intervals(active.to_numpy())
    ref_nav = (1+reference).cumprod()
    for row in rows:
        key, rid = row["key"], row["run_id"]
        eq, r = frames[key], matrix[key]
        f = eq.reset_index()
        annual = []
        for year, g in f.groupby(f.date.str[:4]):
            stats = period_stats(f, initial_cash, g.date.iloc[0], g.date.iloc[-1])
            rr = reference.loc[g.date]
            stats.update(year=year, reference_return=float((1+rr).prod()-1),
                         relative_return=float((1+stats["return"])/(1+rr).prod()-1))
            annual.append(stats)
        relative_log = np.log1p(r)-np.log1p(reference)
        window = relative_log.rolling(63).sum().dropna()
        worst = []
        for end in window.nsmallest(len(window)).index:
            stop = matrix.index.get_loc(end)
            start = matrix.index[stop-62]
            if any(not (end < w["start"] or start > w["end"]) for w in worst):
                continue
            worst.append({"start": start, "end": end,
                          "relative_return": float(np.expm1(window.loc[end])),
                          "strategy_return": float((1+r.loc[start:end]).prod()-1),
                          "reference_return": float((1+reference.loc[start:end]).prod()-1)})
            if len(worst) == 2:
                break
        d = pd.read_parquet(root / "runs" / rid / "decisions.parquet")
        selected = d.loc[d.selected]
        attr = pd.read_parquet(root / "runs" / rid / "configured/attribution.parquet")
        stocks = attr.groupby("stock_code").net_pnl.sum().sort_values()
        positive, negative = stocks.loc[stocks.gt(0)], stocks.loc[stocks.lt(0)]
        summary = {"annual": annual, "worst_relative_windows": worst,
            "screen": period_stats(f, initial_cash, end=str(pd.Timestamp(split_date)-pd.Timedelta(days=1))[:10]),
            "later": period_stats(f, initial_cash, start=split_date),
            "mean_annual_active_return": float((r-reference).mean()*252),
            "relative_wealth_return": float((1+r).prod()/ref_nav.iloc[-1]-1),
            "reference_return": float(ref_nav.iloc[-1]-1),
            "positive_relative_years": sum(v["relative_return"] > 0 for v in annual),
            "year_count": len(annual),
            "selected_median_cap": float(selected.estimated_market_cap.median()) if len(selected) else None,
            "selected_median_share_report_age": float(selected.q_age_days.median()) if len(selected) else None,
            "worst_stocks": [{"stock_code": c, "net_pnl": float(v)} for c, v in stocks.head(5).items()],
            "best_stocks": [{"stock_code": c, "net_pnl": float(v)} for c, v in stocks.tail(5).sort_values(ascending=False).items()],
            "top5_profit_fraction": float(positive.nlargest(5).sum()/positive.sum()) if len(positive) else None,
            "top5_loss_fraction": float(negative.nsmallest(5).sum()/negative.sum()) if len(negative) else None,
            "selected_rows": len(selected), "eligible_rows": int(d.reason.eq("eligible").sum())}
        if key in candidates:
            idx = candidates.index(key)
            summary["joint_active_bands"] = [{"block_sessions": b["block_sessions"],
                "lower": b["lower"][idx], "upper": b["upper"][idx]} for b in bands]
            lo = min(b["lower"][idx] for b in bands)
            hi = max(b["upper"][idx] for b in bands)
            m = row["metrics"]
            reasons = []
            if m["unresolved_market_value"] > .001:
                verdict = "估值待核查"
                reasons.append("期末有未解决估值，暂不裁决策略有效性")
            elif lo > 0:
                verdict = "优先继续机制复核"
                reasons.append("相对随机参考的平均日增量，在两种块长的探索性联合带下界均为正")
            elif hi < 0:
                verdict = "停止当前实现的盲目调参"
                reasons.append("相对随机参考的平均日增量，在两种块长的探索性联合带上界均为负")
            elif summary["mean_annual_active_return"] > 0:
                verdict = "有限继续：证据未定"
                reasons.append("增量点估计为正，但不确定带跨零；不认定有效")
            else:
                verdict = "暂存：先定位失败机制"
                reasons.append("当前含费增量不占优，尚不足以普遍否定该家族")
            reasons += [f"后续已查看区间收益 {summary['later']['return']:.2%}；{summary['positive_relative_years']}/{len(annual)} 个年份相对收益为正",
                        "股本为已披露季报估计，财务无原始历史修订档案；未通过真实历史市值验收",
                        "全部区间均为回顾性研究；统计带未消除研究者既往看过数据的偏差"]
            summary["verdict"] = verdict
            summary["reasons"] = reasons
            summary["next_question"] = ("先核查最差相对区间与核心持仓的股本／公司行动，再检验低换手和成本压力；不按后段重选冠军"
                                          if summary["mean_annual_active_return"] > 0 else
                                          "先用零费用路径、入场延迟及典型亏损股区分无信息和执行失败；没有可区分的假设则暂存")
        else:
            summary.update(verdict="研究对照", reasons=["用于校准机会集或风险，不参与策略晋级排名"], next_question="保留原定义，随候选一起重放")
        row["evidence"] = summary
    return {"reference": "三组预登记稳定哈希选股账户的逐日等权收益参考；分析用组合，不是另一个实际撮合账户或官方指数。",
            "inference": "平均(策略含费日收益 − 同日参考收益)×252；不是 CAGR 差。21/63 日循环块、1500 次、跨候选联合 max-standardized 95% 探索带。不是未来保证，不构成实盘晋级。",
            "bands": bands, "candidate_order": candidates,
            "correlations": records(matrix.corr().rename_axis("key").reset_index()),
            "reference_daily": [{"date": d, "nav": float(v)} for d, v in ref_nav.items()]}
