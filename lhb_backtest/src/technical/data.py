"""Historical price/volume snapshots. No LHB table is opened or fingerprinted."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import uuid

import duckdb
import numpy as np
import pandas as pd
import yaml

from .artifacts import content_id, digest, fingerprint, verify_artifacts, verify_inventory, write_json
from .signals import BAR_COLS, MAIN_BOARD, check_keys


def dates(series):
    return pd.to_datetime(series.astype(str), errors="raise").dt.strftime("%Y-%m-%d")


def prepare_snapshot(project, root, spec, progress):
    if spec.strategy.family != "price" or spec.strategy.require_fundamentals:
        from .foundation_data import prepare_foundation_snapshot
        return prepare_foundation_snapshot(project, root, spec, progress)
    project, root = Path(project), Path(root)
    cfg = yaml.safe_load((project / "config/config.yaml").read_text(encoding="utf-8"))
    clean = (project / cfg["paths"]["clean_dir"]).resolve()
    export = (project / cfg["simtradedata"]["export_dir"]).resolve()
    meta_dir = export / "metadata"
    source_paths = [clean / "daily_kline.parquet", export / ".source_state.json",
        *[meta_dir / f"{name}.parquet" for name in ["stock_metadata", "stock_status", "trade_days", "benchmark"]]]
    planning = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in source_paths}
    state = json.loads((export / ".source_state.json").read_text(encoding="utf-8"))
    if state.get("status_policy") != "dated-flags-v1":
        raise ValueError("需先核实逐日历史状态")
    calendar = dates(pd.read_parquet(meta_dir / "trade_days.parquet").date).tolist()
    if calendar != sorted(set(calendar)):
        raise ValueError("交易日历重复或未排序")
    metadata = pd.read_parquet(meta_dir / "stock_metadata.parquet")
    metadata["stock_code"] = metadata.symbol.str.replace(".SS", ".SH", regex=False)
    metadata = metadata.loc[metadata.stock_code.str.match(MAIN_BOARD) & metadata.security_type.astype(str).eq("1"),
        ["stock_code", "stock_name", "listed_date", "de_listed_date"]].copy()
    if metadata.stock_code.duplicated().any():
        raise ValueError("证券主表代码重复")
    for key in ("listed_date", "de_listed_date"):
        metadata[key] = metadata[key].astype(str).str[:10]
        # 2900-01-01 is a vendor sentinel, outside pandas' Timestamp range.
        if not metadata[key].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise ValueError(f"证券主表 {key} 缺失或无效")
    with duckdb.connect() as con:
        con.read_parquet(str(clean / "daily_kline.parquet")).create_view("quotes")
        first, last = con.sql("SELECT min(trade_date), max(trade_date) FROM quotes").fetchone()
    available_end = min(str(last), state["status_history_end"])
    end = available_end if spec.experiment.end_date == "latest" else spec.experiment.end_date
    start = spec.experiment.start_date
    if end > available_end or start > end or start < state["status_history_start"]:
        raise ValueError(f"请求区间超出已覆盖行情/状态，截至 {available_end}")
    sessions = [d for d in calendar if start <= d <= end]
    if len(sessions) < 2:
        raise ValueError("至少需要两个交易日")
    end = sessions[-1]
    warmup = max(20, spec.strategy.lookback, spec.strategy.trend_window, spec.strategy.min_history)
    lookback = max(str(first), calendar[max(0, calendar.index(sessions[0])-warmup)])
    action_paths, missing = [], []
    for code in metadata.stock_code:
        path = export / "exrights" / (code.replace(".SH", ".SS") + ".parquet")
        if path.exists():
            action_paths.append(path)
        else:
            missing.append(code)
    progress("固定独立主板行情、证券主表、历史状态和公司行动；不读取龙虎榜")
    inventory = fingerprint(source_paths + action_paths)
    for p, stat in planning.items():
        if (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != stat:
            raise ValueError("规划输入已变化，请重试")
    key = content_id({"schema": 1, "sources": inventory, "start": lookback, "end": end, "missing_actions": missing})[:24]
    folder = root / "snapshots" / key
    if folder.exists():
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        verify_artifacts(folder, manifest)
        return folder, manifest
    temp = folder.with_name(key + ".building-" + uuid.uuid4().hex[:8])
    temp.mkdir(parents=True)
    with duckdb.connect() as con:
        con.read_parquet(str(clean / "daily_kline.parquet")).create_view("quotes")
        con.register("securities", metadata[["stock_code"]])
        bars = con.execute(f"SELECT {', '.join('q.'+c for c in BAR_COLS)} FROM quotes q JOIN securities USING(stock_code) WHERE trade_date BETWEEN ? AND ? ORDER BY stock_code, trade_date", [lookback, end]).df()
    bars.trade_date = dates(bars.trade_date)
    check_keys(bars, "行情")
    action_cols = ["allotted_ps", "rationed_ps", "rationed_px", "bonus_ps", "dividend"]
    def read_action(path):
        a = pd.read_parquet(path, columns=["date"] + action_cols)
        a["trade_date"] = dates(a.pop("date"))
        a["stock_code"] = path.stem.replace(".SS", ".SH")
        return a.loc[a.trade_date.between(lookback, end)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(read_action, action_paths))
    actions = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["stock_code", "trade_date"]+action_cols)
    actions[action_cols] = actions[action_cols].astype(float)
    check_keys(actions, "公司行动")
    if not np.isfinite(actions[action_cols].to_numpy(dtype=float)).all() or actions[action_cols].lt(0).any().any():
        raise ValueError("公司行动金额未知或非法")
    if not np.isclose(actions.bonus_ps, actions.dividend, atol=1e-6).all():
        raise ValueError("分红字段口径冲突")
    status = pd.read_parquet(meta_dir / "stock_status.parquet")
    status["trade_date"] = dates(status.pop("date"))
    status = status.loc[status.trade_date.between(lookback, end)]
    for kind in ("ST", "HALT"):
        if not set(sessions).issubset(set(status.loc[status.status_type.eq(kind), "trade_date"])):
            raise ValueError(f"缺少 {kind} 历史状态")
    benchmark = pd.read_parquet(meta_dir / "benchmark.parquet")
    benchmark["trade_date"] = dates(benchmark.pop("date"))
    benchmark = benchmark.loc[benchmark.trade_date.between(start, end)]
    tables = {"bars": bars, "actions": actions, "metadata": metadata, "status": status, "benchmark": benchmark}
    for name, frame in tables.items():
        frame.to_parquet(temp / f"{name}.parquet", index=False)
    write_json(temp / "calendar.json", calendar)
    manifest = {"snapshot_id": key, "start": lookback, "end": end, "sources": inventory,
        "universe_policy": "dated-mainboard-master-v1;no-LHB", "missing_action_files": missing,
        "counts": {name: len(f) for name, f in tables.items()}, "quote_codes": int(bars.stock_code.nunique()),
        "master_codes_without_quotes_in_window": sorted(set(metadata.stock_code)-set(bars.stock_code)),
        "limits": ["Missing historical securities or vendor revisions cannot be disproved by a snapshot hash.", "Missing action files do not remove stocks using future knowledge; held positions are flagged."],
        "artifacts": {p.name: digest(p) for p in temp.iterdir() if p.is_file()}}
    verify_inventory(inventory)
    write_json(temp / "manifest.json", manifest)
    temp.rename(folder)
    return folder, manifest
