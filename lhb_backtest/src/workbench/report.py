"""Shared, escaped offline report and detailed event evidence."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .artifacts import jsonable, records

WEB = Path(__file__).parent / "web"
PORTFOLIOS = {"institutional", "baseline", "institutional_matched", "matched_control", "institutional_stress"}
EVENT_COLUMNS = ["event_id", "trade_date", "stock_code", "stock_name", "inst_ratio", "inst_disclosed_net",
                 "float_value", "momentum_20", "avg_amount_20", "label_return", "label_status",
                 "eligibility_reason", "signal", "matched_treatment", "matched_control"]


def event_evidence(folder, snapshot, event_id):
    events = pd.read_parquet(folder / "events.parquet", filters=[("event_id", "=", event_id)])
    if len(events) != 1:
        raise ValueError("事件不存在")
    event = events.iloc[0]
    date, code = event.trade_date, event.stock_code
    calendar = json.loads((snapshot / "calendar.json").read_text(encoding="utf-8"))
    idx = calendar.index(date)
    start = calendar[max(0, idx - 20)]
    end = calendar[min(len(calendar) - 1, idx + 15)]
    bars = pd.read_parquet(snapshot / "bars.parquet", filters=[("stock_code", "=", code), ("trade_date", ">=", start), ("trade_date", "<=", end)])
    brokers = pd.read_parquet(snapshot / "brokers.parquet", filters=[("stock_code", "=", code), ("trade_date", "=", date)])
    actions = pd.read_parquet(snapshot / "actions.parquet", filters=[("stock_code", "=", code), ("trade_date", ">=", start), ("trade_date", "<=", end)])
    orders = []
    for portfolio in sorted(PORTFOLIOS):
        path = folder / portfolio / "orders.parquet"
        if path.exists():
            frame = pd.read_parquet(path, filters=[("event_id", "=", event_id)])
            frame["portfolio"] = portfolio
            orders.extend(records(frame))
    pairs = pd.read_parquet(folder / "pairs.parquet")
    pairs = pairs.loc[pairs.treated_event_id.eq(event_id) | pairs.control_event_id.eq(event_id)]
    return {"event": jsonable(event.to_dict()), "bars": records(bars), "brokers": records(brokers),
            "actions": records(actions), "orders": orders, "pairs": records(pairs),
            "source_reference": {"snapshot": snapshot.name, "event_key": {"stock_code": code, "trade_date": date},
                                 "summary": "clean/lhb_summary.parquet", "detail": "clean/lhb_broker_detail.parquet",
                                 "quotes": "clean/daily_kline.parquet", "valuation": f"valuation/{code.replace('.SH', '.SS')}.parquet"}}


def render_page(bootstrap):
    template = (WEB / "index.html").read_text(encoding="utf-8")
    payload = json.dumps(jsonable(bootstrap), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return template.replace("/*APP_STYLE*/", (WEB / "style.css").read_text(encoding="utf-8")).replace(
        "/*BOOTSTRAP*/", payload).replace("/*APP_SCRIPT*/", (WEB / "app.js").read_text(encoding="utf-8"))


def export_report(folder, payload, events, snapshot):
    # Full CSVs remain available; the self-contained HTML carries a bounded, unbiased recent preview.
    events.to_csv(folder / "events.csv", index=False, encoding="utf-8-sig")
    pd.read_parquet(folder / "pairs.parquet").to_csv(folder / "pairs.csv", index=False, encoding="utf-8-sig")
    for portfolio in sorted(PORTFOLIOS):
        for name in ["orders", "trades", "equity", "ledger", "positions", "warnings"]:
            frame = pd.read_parquet(folder / portfolio / f"{name}.parquet")
            frame.to_csv(folder / portfolio / f"{name}.csv", index=False, encoding="utf-8-sig")
    preview = events.loc[events.signal].sort_values(["trade_date", "stock_code"], ascending=[False, True]).head(100)
    # Offline preview supports every listed event without a server; full browsing uses the workbench.
    evidence = {}
    recent_trades = pd.read_parquet(folder / "institutional/trades.parquet")
    traded_ids = recent_trades.sort_values("exit_date", ascending=False).event_id.head(6).tolist()
    evidence_ids = list(dict.fromkeys(preview.event_id.head(12).tolist() + traded_ids))
    for eid in evidence_ids:
        evidence[eid] = event_evidence(folder, snapshot, eid)
    payload = dict(payload)
    payload["offline_events"] = records(preview[EVENT_COLUMNS])
    html = render_page({"offline": True, "result": payload, "evidence": evidence})
    (folder / "report.html").write_text(html, encoding="utf-8")
