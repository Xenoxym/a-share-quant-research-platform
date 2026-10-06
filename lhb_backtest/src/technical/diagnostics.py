"""Account-preserving period diagnostics; every stock contribution reconciles to equity.

These are retrospective explanations, never signal inputs or evidence of predictability.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import records


def contributions(equity, positions, ledger, initial_cash):
    """P&L(i,t) = MV(i,t)-MV(i,t-1)+cash_flow(i,t)+receivable_flow(i,t)."""
    days = equity.date.tolist()
    keys = ["date", "stock_code"]
    mv = positions[keys + ["market_value"]].copy()
    before = mv.rename(columns={"market_value": "opening_value"})
    before["date"] = before.date.map(dict(zip(days[:-1], days[1:])))
    before = before.dropna(subset=["date"])
    flow = ledger.copy()
    if "fee" not in flow:
        flow["fee"] = 0.
    flow = flow.groupby(keys)[["cash_flow", "receivable_flow", "fee"]].sum().reset_index()
    out = mv.merge(before, on=keys, how="outer", validate="one_to_one").merge(flow, on=keys, how="outer", validate="one_to_one").fillna(0.)
    out["net_pnl"] = out.market_value - out.opening_value + out.cash_flow + out.receivable_flow
    out["before_fees_pnl"] = out.net_pnl + out.fee
    prior = equity.set_index("date").equity.shift().fillna(initial_cash)
    out["contribution"] = out.net_pnl / out.date.map(prior)
    expected = equity.set_index("date").equity - prior
    actual = out.groupby("date").net_pnl.sum().reindex(days, fill_value=0.)
    if not np.allclose(actual, expected, atol=.001, rtol=1e-10):
        raise ValueError("逐股损益不能对账至账户权益变化")
    return out.sort_values(keys).reset_index(drop=True)


def period_stats(equity, initial_cash, start=None, end=None):
    eq = equity.sort_values("date").reset_index(drop=True)
    mask = eq.date.between(start or eq.date.iloc[0], end or eq.date.iloc[-1])
    sub = eq.loc[mask]
    if sub.empty:
        raise ValueError("所选区间内没有交易日")
    first = sub.index[0]
    base = float(eq.equity.iloc[first-1]) if first else float(initial_cash)
    path = np.r_[base, sub.equity.to_numpy()]
    return {"start": sub.date.iloc[0], "end": sub.date.iloc[-1], "sessions": len(sub),
            "opening_equity": base, "closing_equity": float(path[-1]), "pnl": float(path[-1]-base),
            "return": float(path[-1]/base-1), "max_drawdown": float((path/np.maximum.accumulate(path)-1).min()),
            "exposure": float(sub.exposure.mean()) if "exposure" in sub else None,
            "baseline_date": eq.date.iloc[first-1] if first else "初始资金"}


def market_states(benchmark, days):
    """Fixed observable rules, lagged one session. Never fit on full-sample returns."""
    b = pd.DataFrame(benchmark.get("rows", []), columns=["date", "nav"]).set_index("date").reindex(days)
    b["nav"] = pd.to_numeric(b.nav, errors="raise").astype(float)
    r = b.nav.pct_change(fill_method=None)
    trend = b.nav.div(b.nav.shift(126)).sub(1).shift(1)
    vol = r.rolling(20).std(ddof=1).mul(np.sqrt(252)).shift(1)
    state = pd.Series("历史不足", index=b.index)
    valid = trend.notna() & vol.notna()
    state.loc[valid] = np.where(trend.loc[valid] > 0, "上行", "下行") + np.where(vol.loc[valid] > .25, " · 高波动", " · 常态波动")
    return pd.DataFrame({"date": days, "state": state.to_numpy(), "trend_126": trend.to_numpy(), "volatility_20": vol.to_numpy()})


def drawdown_episodes(equity, initial_cash):
    peak, peak_day, active, episodes = initial_cash, equity.date.iloc[0], None, []
    for row in equity.itertuples():
        if row.equity >= peak:
            if active:
                active.update(recovery=row.date, end=row.date)
                episodes.append(active)
                active = None
            peak, peak_day = row.equity, row.date
        else:
            dd = row.equity / peak - 1
            if active is None:
                active = dict(start=row.date, peak_date=peak_day, trough=row.date, depth=dd, recovery=None)
            if dd < active["depth"]:
                active.update(trough=row.date, depth=dd)
            active["end"] = row.date
    if active:
        episodes.append(active)
    return sorted(episodes, key=lambda r: r["depth"])[:10]


def summarize(equity, attribution, warnings, initial_cash, benchmark):
    eq = equity.sort_values("date").copy()
    eq["daily_return"] = eq.equity.div(eq.equity.shift().fillna(initial_cash)).sub(1)
    eq["fee"] = eq.date.map(attribution.groupby("date").fee.sum()).fillna(0.)
    states = market_states(benchmark, eq.date.tolist())
    eq = eq.merge(states, on="date", validate="one_to_one")
    months = [period_stats(eq, initial_cash, sub.date.min(), sub.date.max()) | {"month": month}
              for month, sub in eq.groupby(eq.date.str[:7])]
    windows = []
    # Complete 63-session windows, independent of calendar-month length.
    for i in range(62, len(eq)):
        base = initial_cash if i == 62 else float(eq.equity.iloc[i-63])
        windows.append({"start": eq.date.iloc[i-62], "end": eq.date.iloc[i], "return": float(eq.equity.iloc[i]/base-1)})
    extremes = []
    occupied = set()
    for label, ordered in [("较差区间", sorted(windows, key=lambda r: r["return"])), ("较好区间", sorted(windows, key=lambda r: -r["return"]))]:
        count = 0
        for row in ordered:
            covered = set(eq.loc[eq.date.between(row["start"], row["end"]), "date"])
            if covered & occupied:
                continue
            extremes.append(row | {"label": label})
            occupied |= covered
            count += 1
            if count == 2:
                break
    worst = eq.nsmallest(8, "daily_return")
    best = eq.nlargest(3, "daily_return")
    cases = []
    for row in pd.concat([worst, best]).drop_duplicates("date").loc[lambda f: f.daily_return.ne(0)].itertuples():
        a = attribution.loc[attribution.date.eq(row.date)].sort_values("net_pnl", ascending=row.daily_return < 0)
        top = a.iloc[0] if len(a) else None
        cases.append({"date": row.date, "daily_return": row.daily_return, "stock_code": top.stock_code if top is not None else "",
                      "net_pnl": float(top.net_pnl) if top is not None else 0., "label": "亏损日" if row.daily_return < 0 else "上涨日"})
    flags = []
    if not warnings.empty:
        for (kind, code), sub in warnings.groupby(["kind", "stock_code"]):
            flags.append({"kind": kind, "stock_code": code, "start": sub.date.min(), "end": sub.date.max(), "count": len(sub)})
    state_rows = []
    for state, sub in eq.groupby("state"):
        state_rows.append({"state": state, "sessions": len(sub), "mean_daily_return": float(sub.daily_return.mean()),
                           "positive_fraction": float(sub.daily_return.gt(0).mean()), "fee": float(sub.fee.sum())})
    return {"months": months, "windows": extremes, "episodes": drawdown_episodes(eq, initial_cash),
            "cases": cases, "flags": sorted(flags, key=lambda r: (-r["count"], r["stock_code"])),
            "states": state_rows, "daily": records(eq[["date", "daily_return", "fee", "state", "trend_126", "volatility_20"]]),
            "attribution_error": float(abs(attribution.net_pnl.sum() - (eq.equity.iloc[-1]-initial_cash))),
            "state_definition": "以前一交易日收盘已知的市场参考指数 126 日涨跌正负 × 20 日日收益标准差√252 是否大于 25% 划分。阈值预设，不参与选股；标签属于描述性诊断。"}


@lru_cache(maxsize=4)
def load_account(folder, scenario="configured"):
    p = Path(folder)
    result = json.loads((p / "result.json").read_text(encoding="utf-8"))
    frames = {k: pd.read_parquet(p / scenario / f"{k}.parquet") for k in ["equity", "positions", "ledger", "orders", "warnings"]}
    frames["attribution"] = contributions(frames["equity"], frames["positions"], frames["ledger"], result["spec"]["execution"]["initial_cash"])
    return result, frames


def analysis_for(folder):
    result, f = load_account(str(folder))
    value = summarize(f["equity"], f["attribution"], f["warnings"], result["spec"]["execution"]["initial_cash"], result["benchmark"])
    value["execution"] = execution_for(folder)
    return value


def execution_delays(orders, targets, sessions):
    """Buying later than the scheduled first execution date is a material assumption."""
    buys = orders.loc[orders.side.eq("buy") & orders.filled_shares.gt(0)].copy()
    if buys.empty or targets.empty:
        return {"buy_fills": 0, "same_session_fraction": None, "median_delay": None, "notional_delayed_fraction": None, "examples": []}
    buys["day_index"] = buys.date.map({d: i for i, d in enumerate(sessions)})
    t = targets[["stock_code", "execution_date", "signal_date"]].copy()
    t["scheduled_index"] = t.execution_date.map({d: i for i, d in enumerate(sessions)})
    # A last-day signal may target a session beyond the experiment; no fill can match it.
    t = t.dropna(subset=["scheduled_index"])
    t["scheduled_index"] = t.scheduled_index.astype(int)
    matched = pd.merge_asof(buys.sort_values("day_index"), t.sort_values("scheduled_index"),
                            left_on="day_index", right_on="scheduled_index", by="stock_code", direction="backward")
    if matched.scheduled_index.isna().any():
        raise ValueError("买入成交缺少此前冻结的目标")
    matched["delay"] = matched.day_index - matched.scheduled_index
    matched["notional"] = matched.filled_shares * matched.price
    return {"buy_fills": len(matched), "same_session_fraction": float(matched.delay.eq(0).mean()),
            "median_delay": float(matched.delay.median()),
            "notional_delayed_fraction": float(matched.loc[matched.delay.gt(0), "notional"].sum()/matched.notional.sum()),
            "examples": records(matched.sort_values(["delay", "notional"], ascending=False).head(8)[["stock_code", "date", "signal_date", "execution_date", "delay", "notional"]])}


def execution_for(folder):
    _, f = load_account(str(folder))
    targets = pd.read_parquet(Path(folder) / "configured/targets.parquet")
    return execution_delays(f["orders"], targets, f["equity"].date.tolist())


def inspect_period(folder, start, end, scenario="configured"):
    result, f = load_account(str(folder), scenario)
    stats = period_stats(f["equity"], result["spec"]["execution"]["initial_cash"], start, end)
    start, end = stats["start"], stats["end"]
    a = f["attribution"].loc[f["attribution"].date.between(start, end)]
    stocks = a.groupby("stock_code")[["net_pnl", "fee", "before_fees_pnl", "receivable_flow"]].sum().reset_index().sort_values("net_pnl")
    stocks["contribution"] = stocks.net_pnl / stats["opening_equity"]
    worst_dates = a.loc[a.groupby("stock_code").net_pnl.idxmin()].set_index("stock_code").date if len(a) else pd.Series(dtype=str)
    stocks["worst_date"] = stocks.stock_code.map(worst_dates)
    # Keep full ranking for filtering and drill-down, not just the worst names.
    stats["fee"] = float(a.fee.sum())
    stats["reconciliation_error"] = float(abs(a.net_pnl.sum()-stats["pnl"]))
    orders = f["orders"].loc[f["orders"].date.between(start, end)]
    stats["fills"] = int(orders.filled_shares.gt(0).sum())
    stats["unfilled"] = int(orders.filled_shares.eq(0).sum())
    daily = f["equity"].loc[f["equity"].date.between(start, end), ["date", "equity", "exposure"]].copy()
    daily["pnl"] = daily.date.map(a.groupby("date").net_pnl.sum()).fillna(0.)
    return {"stats": stats, "stocks": records(stocks), "days": records(daily),
            "flags": records(f["warnings"].loc[f["warnings"].date.between(start, end)])}


def inspect_case(folder, day, code, scenario="configured"):
    p = Path(folder)
    result, f = load_account(str(p), scenario)
    snapshot = p.parent.parent / "snapshots" / result["snapshot_id"]
    calendar = json.loads((snapshot / "calendar.json").read_text(encoding="utf-8"))
    if day not in f["equity"].date.values:
        raise ValueError("日期不在该实验的交易日中")
    idx = calendar.index(day)
    start, end = calendar[max(0, idx-10)], calendar[min(len(calendar)-1, idx+5)]
    def subset(name):
        frame = f[name]
        return records(frame.loc[frame.date.between(start, end) & frame.stock_code.eq(code)])
    quotes = pd.read_parquet(snapshot / "bars.parquet", filters=[("stock_code", "=", code), ("trade_date", ">=", start), ("trade_date", "<=", end)])
    actions = pd.read_parquet(snapshot / "actions.parquet", filters=[("stock_code", "=", code), ("trade_date", ">=", start), ("trade_date", "<=", end)])
    decisions = pd.read_parquet(p / "decisions.parquet", filters=[("stock_code", "=", code), ("trade_date", "<", day)]).sort_values("trade_date").tail(1)
    manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    return {"date": day, "stock_code": code, "scenario": scenario, "quotes": records(quotes), "actions": records(actions),
            "decision": records(decisions), **{k: subset(k) for k in ["attribution", "orders", "positions", "ledger", "warnings"]},
            "evidence": {"snapshot_id": result["snapshot_id"], "code_hash": result["code_hash"],
                         "snapshot_manifest_sha256": manifest["snapshot_manifest_sha256"]}}
