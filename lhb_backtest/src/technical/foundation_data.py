"""Freeze reported fundamentals separately; never backfill today's capitalization."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import shutil
import uuid

import pandas as pd
import yaml

from .artifacts import content_id, digest, fingerprint, verify_artifacts, verify_inventory, write_json

FIELDS = ["total_shares", "np_parent_company_owners", "total_shareholder_equity", "total_assets"]
POLICY = {
    "availability": "Publication date must be strictly BEFORE decision date; no same-day release-time assumption.",
    "size": "Raw close times latest already-published reported total shares; CNY, NOT contemporaneous share-register market cap.",
    "annual": "Latest already-published December fiscal-year report; parent net profit / total shareholder equity is a simplified annual profitability proxy.",
    "staleness": "Quarterly share report at most 240 calendar days old; annual report at most 550 calendar days old, measured from fiscal period end.",
    "limits": ["Published-date filtering does not restore historical vendor vintages or remove restatement bias.",
               "Reported shares can lag issuance, cancellations and corporate actions; investigate selected names before promotion.",
               "Profit divided by total equity is not exact parent-equity ROE, QMJ, or Fama-French operating profitability."]}


def normalize_filings(frame):
    f = frame.copy()
    f["report_date"] = pd.to_datetime(f["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    raw = f["publ_date"].astype(str).str.replace("-", "", regex=False).str[:8]
    f["publication_date"] = pd.to_datetime(raw, format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    for c in FIELDS:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    valid = f.report_date.notna() & f.publication_date.notna() & f.publication_date.ge(f.report_date)
    rejected = f.loc[~valid, ["stock_code", "report_date", "publication_date"]].copy()
    f = f.loc[valid, ["stock_code", "report_date", "publication_date"] + FIELDS]
    if f.duplicated(["stock_code", "report_date"]).any():
        raise ValueError("财报期间重复；需定义修订版本，不能任意保留最后一条")
    return f.sort_values(["stock_code", "report_date"]), rejected


def prepare_foundation_snapshot(project, root, spec, progress):
    from .data import prepare_snapshot
    project, root = Path(project), Path(root)
    base_spec = replace(spec, strategy=replace(spec.strategy, family="price", require_fundamentals=False))
    if spec.strategy.family == "allocation":
        base_spec = replace(base_spec, strategy=replace(base_spec.strategy, min_history=756))
    base, bm = prepare_snapshot(project, root, base_spec, progress)
    cfg = yaml.safe_load((project / "config/config.yaml").read_text(encoding="utf-8"))
    export = (project / cfg["simtradedata"]["export_dir"]).resolve()
    codes = pd.read_parquet(base / "metadata.parquet").stock_code.tolist()
    paths = [export / "fundamentals" / (c.replace(".SH", ".SS") + ".parquet") for c in codes]
    missing = [p.stem.replace(".SS", ".SH") for p in paths if not p.exists()]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise ValueError("缺少带公告日期的历史财报，不能构建市值或财务信号")
    progress("冻结历史财报与公告日；总市值由当时已披露股本估算，未使用今日股本")
    inventory = fingerprint(paths)
    sid = content_id({"schema": "foundations-1", "base": bm["snapshot_id"], "sources": inventory, "policy": POLICY})[:24]
    folder = root / "snapshots" / sid
    if folder.exists():
        m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        verify_artifacts(folder, m)
        return folder, m
    temp = folder.with_name(sid + ".building-" + uuid.uuid4().hex[:8])
    temp.mkdir(parents=True)
    for name in bm["artifacts"]:
        shutil.copy2(base / name, temp / name)
    def read(p):
        f = pd.read_parquet(p, columns=["date", "publ_date"] + FIELDS)
        f["stock_code"] = p.stem.replace(".SS", ".SH")
        return f
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(read, paths))
    filings, rejected = normalize_filings(pd.concat(frames, ignore_index=True))
    # Retain pre-start reports for valid as-of joins; never infer missing quarters.
    filings = filings.loc[filings.publication_date.le(bm["end"])]
    filings.to_parquet(temp / "fundamentals.parquet", index=False)
    rejected.to_parquet(temp / "rejected_filings.parquet", index=False)
    benchmark = pd.read_parquet(export / "metadata/benchmark.parquet")
    benchmark["trade_date"] = pd.to_datetime(benchmark.pop("date").astype(str)).dt.strftime("%Y-%m-%d")
    benchmark = benchmark.loc[benchmark.trade_date.between(bm["start"], bm["end"])]
    benchmark.to_parquet(temp / "benchmark.parquet", index=False)
    m = {**bm, "snapshot_id": sid, "base_snapshot_id": bm["snapshot_id"],
         "sources": bm["sources"] + inventory, "fundamental_policy": POLICY,
         "missing_fundamentals": missing, "counts": {**bm["counts"], "fundamentals": len(filings),
             "rejected_filings": len(rejected), "benchmark": len(benchmark)},
         "artifacts": {p.name: digest(p) for p in temp.iterdir()}}
    verify_inventory(m["sources"])
    write_json(temp / "manifest.json", m)
    temp.rename(folder)
    return folder, m


def attach_reports(rows, filings, actions=None, adjust_bonus=False):
    """Latest fiscal period publicly available strictly before each decision date."""
    if filings is None:
        raise ValueError("此策略需要已冻结的历史财报")
    out = rows.copy()
    out["_row"] = range(len(out))
    out["_day"] = pd.to_datetime(out.trade_date)
    def attach(f, prefix):
        nonlocal out
        f = f.sort_values(["stock_code", "publication_date", "report_date"]).copy()
        # Late publication of an older fiscal period must not replace a newer one.
        f["_period"] = pd.to_datetime(f.report_date).astype("int64")
        f = f.loc[f._period.eq(f.groupby("stock_code")._period.cummax())]
        f = f.drop_duplicates(["stock_code", "publication_date"], keep="last")
        cols = ["stock_code", "report_date", "publication_date"] + FIELDS
        f = f[cols].rename(columns={c: prefix + c for c in cols if c != "stock_code"})
        f[prefix + "_pub"] = pd.to_datetime(f[prefix + "publication_date"])
        out = pd.merge_asof(out.sort_values("_day"), f.sort_values(prefix + "_pub"),
                            left_on="_day", right_on=prefix + "_pub", by="stock_code",
                            direction="backward", allow_exact_matches=False)
        out[prefix + "age_days"] = (out._day - pd.to_datetime(out[prefix + "report_date"])).dt.days
        out.drop(columns=prefix + "_pub", inplace=True)
    attach(filings, "q_")
    attach(filings.loc[filings.report_date.str.endswith("12-31")], "a_")
    if adjust_bonus:
        if actions is None:
            raise ValueError("送转股本桥接需要冻结公司行动")
        # bonus_ps is CASH in this vendor; allotted_ps changes share count.
        changes = actions.loc[actions.allotted_ps.ne(0) | actions.rationed_ps.ne(0),
                              ["stock_code", "trade_date", "allotted_ps", "rationed_ps"]].rename(columns={"trade_date":"action_date"})
        joined = out[["_row", "stock_code", "trade_date", "q_report_date"]].merge(changes, on="stock_code")
        joined = joined.loc[joined.action_date.gt(joined.q_report_date) & joined.action_date.le(joined.trade_date)]
        joined["factor"] = 1 + joined.allotted_ps
        multiplier = joined.groupby("_row").factor.prod()
        rights = joined.groupby("_row").rationed_ps.max()
        out["share_action_multiplier"] = out._row.map(multiplier).fillna(1.)
        out["unresolved_rights_since_report"] = out._row.map(rights).fillna(0.).gt(0)
        out["estimated_shares"] = out.q_total_shares * out.share_action_multiplier
        # Do not assume every offered rights share was subscribed.
        out.loc[out.unresolved_rights_since_report, "estimated_shares"] = float("nan")
        out["estimated_market_cap"] = out.close * out.estimated_shares
    else:
        out["estimated_market_cap"] = out.close * out.q_total_shares
    out["earnings_yield"] = out.a_np_parent_company_owners / out.estimated_market_cap
    out["book_to_price"] = out.q_total_shareholder_equity / out.estimated_market_cap
    out["profitability"] = out.a_np_parent_company_owners / out.a_total_shareholder_equity
    return out.sort_values("_row").drop(columns=["_row", "_day"]).reset_index(drop=True)
