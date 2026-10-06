"""Build immutable, scoped input snapshots directly from accepted clean data."""
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

MAIN_BOARD = r"^(?:60\d{4}\.SH|00\d{4}\.SZ)$"
KEYS = ["stock_code", "trade_date"]
BAR_COLS = KEYS + ["open", "high", "low", "close", "pre_close", "volume", "amount",
                   "high_limit", "low_limit", "limit_price_valid", "is_st"]


def dates(values):
    return pd.to_datetime(values.astype(str), errors="raise").dt.strftime("%Y-%m-%d")


def check_keys(frame, name):
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"{name} 存在空值或重复股票日期主键")


def configured_paths(project: Path, config_path=None):
    config = yaml.safe_load(Path(config_path or project / "config/config.yaml").read_text(encoding="utf-8"))
    return ((project / config["paths"]["clean_dir"]).resolve(),
            (project / config["simtradedata"]["export_dir"]).resolve())


def institutional_features(summary: pd.DataFrame, brokers: pd.DataFrame):
    """Side-specific disclosed flows; never deduplicate anonymous seats across sides."""
    out = summary.copy()
    check_keys(out, "龙虎榜汇总")
    if brokers.duplicated(KEYS + ["direction", "rank"]).any():
        raise ValueError("同一披露方向席位排名重复")
    grouped = brokers.groupby(KEYS, observed=True)
    for field in ["disclosure_kind", "window_days", "report_reason"]:
        if grouped[field].nunique(dropna=False).gt(1).any():
            raise ValueError(f"席位混合披露报告: {field}")
    meta = grouped.agg(disclosure_kind=("disclosure_kind", "first"),
                       seat_window_days=("window_days", "first"),
                       seat_report_reason=("report_reason", "first"))
    out = out.merge(meta, on=KEYS, how="left", validate="one_to_one")
    for side in ("buy", "sell"):
        part = brokers.loc[brokers.direction.eq(side)].copy()
        amount = pd.to_numeric(part[f"{side}_amount"], errors="coerce")
        good = np.isfinite(amount) & amount.ge(0) & part.broker_name.notna()
        institution = part.broker_name.str.contains("机构专用", regex=False, na=False)
        part["inst_amount"] = amount.where(institution, 0.0)
        part["valid"] = good
        values = part.groupby(KEYS, observed=True).agg(
            **{f"inst_{side}_amount": ("inst_amount", "sum"),
               f"{side}_known": ("valid", "all")})
        out = out.merge(values, on=KEYS, how="left", validate="one_to_one")
    out["seats_known"] = out.buy_known.eq(True) & out.sell_known.eq(True)
    out["inst_disclosed_net"] = (out.inst_buy_amount - out.inst_sell_amount).where(out.seats_known)
    out["event_id"] = out.trade_date + "_" + out.stock_code
    return out.drop(columns=["buy_known", "sell_known"])


def attach_features(events, bars, valuation, calendar):
    """No future labels: rolling windows end at the disclosure's trading day."""
    bars = bars.sort_values(KEYS).copy()
    check_keys(bars, "行情")
    index = {d: i for i, d in enumerate(calendar)}
    bars["session"] = bars.trade_date.map(index)
    if bars.session.isna().any():
        raise ValueError("行情日期不在市场日历内")
    groups = bars.groupby("stock_code", sort=False, observed=True)
    positive = bars.close.gt(0) & bars.pre_close.gt(0)
    bars["log_change"] = np.log((bars.close / bars.pre_close).where(positive))
    rolling = bars.groupby("stock_code", observed=True, sort=False).rolling(20, min_periods=20)
    momentum = np.expm1(rolling.log_change.sum().reset_index(level=0, drop=True))
    amount = rolling.amount.mean().reset_index(level=0, drop=True)
    volume = rolling.volume.mean().reset_index(level=0, drop=True)
    contiguous = bars.session.sub(groups.session.shift(19)).eq(19)
    bars["momentum_20"] = momentum.where(contiguous)
    bars["avg_amount_20"] = amount.where(contiguous)
    bars["avg_volume_20"] = volume.where(contiguous)
    features = bars[KEYS + ["close", "amount", "is_st", "limit_price_valid", "high_limit",
                            "momentum_20", "avg_amount_20", "avg_volume_20"]]
    out = events.merge(features, on=KEYS, how="left", validate="one_to_one")
    check_keys(valuation, "估值")
    out = out.merge(valuation[KEYS + ["float_value"]], on=KEYS, how="left", validate="one_to_one")
    out["inst_ratio"] = out.inst_disclosed_net / out.amount.where(out.amount.gt(0))
    next_session = dict(zip(calendar[:-1], calendar[1:]))
    out["entry_date"] = out.trade_date.map(next_session)
    out["known_at_assumed"] = out.entry_date.fillna("unknown") + "T09:00:00+08:00"
    out["base_reason"] = np.select(
        [~out.stock_code.str.match(MAIN_BOARD),
         ~out.summary_window_days.eq(1) | ~out.seat_window_days.eq(1) | ~out.disclosure_kind.eq("ordinary"),
         ~out.seats_known, out.close.isna(), ~out.is_st.eq(0), ~out.limit_price_valid.eq(True),
         ~np.isfinite(out.float_value) | out.float_value.le(0),
         ~np.isfinite(out.momentum_20) | ~np.isfinite(out.avg_amount_20) | out.avg_volume_20.le(0),
         ~np.isfinite(out.inst_ratio)],
        ["unsupported_board", "not_ordinary_single_day", "missing_seats", "missing_quote",
         "st_or_unknown", "unknown_price_limits", "missing_valuation", "missing_20_sessions", "invalid_flow"],
        default="eligible")
    return out.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)


def _read_stock_tables(paths, start, end, columns):
    def read(path):
        frame = pd.read_parquet(path, columns=["date"] + columns)
        frame["trade_date"] = dates(frame.pop("date"))
        frame["stock_code"] = path.stem.replace(".SS", ".SH")
        return frame.loc[frame.trade_date.between(start, end)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(read, paths))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=KEYS + columns)


def prepare_snapshot(project: Path, root: Path, card, progress=lambda message: None, config_path=None):
    clean, export = configured_paths(project, config_path)
    planning_paths = [clean / "lhb_summary.parquet", clean / "daily_kline.parquet",
                      export / "metadata/trade_days.parquet", export / ".source_state.json"]
    planning_stats = {str(p.resolve()): (p.stat().st_size, p.stat().st_mtime_ns) for p in planning_paths}
    progress("检查已发布数据、历史状态与请求区间")
    calendar = dates(pd.read_parquet(export / "metadata/trade_days.parquet").date).tolist()
    if calendar != sorted(set(calendar)):
        raise ValueError("交易日历重复或未排序")
    state = json.loads((export / ".source_state.json").read_text(encoding="utf-8"))
    if state.get("status_policy") != "dated-flags-v1":
        raise ValueError("需要经过历史核对的 ST/HALT 状态数据")
    summary_cols = KEYS + ["stock_name", "summary_selected_reason", "summary_window_days", "data_source"]
    summary = pd.read_parquet(clean / "lhb_summary.parquet", columns=summary_cols)
    summary.trade_date = dates(summary.trade_date)
    with duckdb.connect() as con:
        con.from_parquet(str(clean / "daily_kline.parquet")).create_view("quotes")
        min_date, max_date = con.sql("SELECT min(trade_date), max(trade_date) FROM quotes").fetchone()
    available_end = min(str(max_date), str(summary.trade_date.max()), state["status_history_end"])
    end = available_end if card.end_date == "latest" else card.end_date
    if end > available_end or card.start_date > end:
        raise ValueError(f"可研究数据截至 {available_end}，请求区间无法覆盖")
    sessions = [d for d in calendar if card.start_date <= d <= end]
    if len(sessions) < 2 or sessions[0] < state["status_history_start"] or sessions[0] < str(min_date):
        raise ValueError("请求区间至少包含两个已覆盖的交易日")
    end = sessions[-1]
    lookback = calendar[max(0, calendar.index(sessions[0]) - 20)]
    summary = summary.loc[summary.trade_date.between(card.start_date, end)].copy()
    codes = sorted(summary.loc[summary.stock_code.str.match(MAIN_BOARD), "stock_code"].unique())
    if not codes:
        raise ValueError("请求区间没有沪深主板事件")
    valuation_paths, action_paths, missing_actions = [], [], []
    for code in codes:
        name = code.replace(".SH", ".SS") + ".parquet"
        v, a = export / "valuation" / name, export / "exrights" / name
        if v.exists():
            valuation_paths.append(v)
        if a.exists():
            action_paths.append(a)
        else:
            missing_actions.append(code)
    paths = [clean / "lhb_summary.parquet", clean / "lhb_broker_detail.parquet", clean / "daily_kline.parquet",
             export / "metadata/trade_days.parquet", export / "metadata/stock_status.parquet",
             export / "metadata/stock_metadata.parquet", export / "metadata/benchmark.parquet",
             export / ".source_state.json", export / "manifest.json"] + valuation_paths + action_paths
    progress(f"校验来源内容指纹（{len(paths):,} 个文件），固定本次数据版本")
    inventory = fingerprint(paths)
    for record in inventory:
        if record["path"] in planning_stats and planning_stats[record["path"]] != (record["size"], record["mtime_ns"]):
            raise RuntimeError("准备范围期间数据发生变化，请等待更新完成后重试")
    key = content_id({"sources": [(v["path"], v["sha256"]) for v in inventory],
                      "start": card.start_date, "end": end, "builder": digest(Path(__file__)),
                      "scope": "main-board-v1"})[:24]
    folder = root / "snapshots" / key
    if (folder / "manifest.json").exists():
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        verify_artifacts(folder, manifest)
        progress("数据未变化，复用已校验的研究快照")
        return folder, manifest
    temp = folder.with_name(f".building-{uuid.uuid4().hex}")
    temp.mkdir(parents=True)
    progress("计算当时可用的机构披露、市值、动量与流动性特征")
    brokers = pd.read_parquet(clean / "lhb_broker_detail.parquet")
    brokers.trade_date = dates(brokers.trade_date)
    brokers = brokers.loc[brokers.trade_date.between(card.start_date, end)].copy()
    with duckdb.connect() as con:
        con.from_parquet(str(clean / "daily_kline.parquet")).create_view("quotes")
        selected_codes = pd.DataFrame({"stock_code": codes})
        con.register("selected_codes", selected_codes)
        columns = ", ".join(f"q.{c}" for c in BAR_COLS)
        bars = con.execute(f"SELECT {columns} FROM quotes q JOIN selected_codes USING(stock_code) "
                           "WHERE trade_date BETWEEN ? AND ? ORDER BY stock_code, trade_date", [lookback, end]).df()
    bars.trade_date = dates(bars.trade_date)
    valuation = _read_stock_tables(valuation_paths, card.start_date, end, ["float_value"])
    actions = _read_stock_tables(action_paths, lookback, end,
                                ["allotted_ps", "rationed_ps", "rationed_px", "bonus_ps", "dividend"])
    check_keys(actions, "公司行动")
    action_cols = ["allotted_ps", "rationed_ps", "rationed_px", "bonus_ps", "dividend"]
    if not np.isfinite(actions[action_cols].to_numpy(dtype=float)).all() or actions[action_cols].lt(0).any().any():
        raise ValueError("公司行动字段存在未知、负数或非有限值")
    if not np.isclose(actions.bonus_ps, actions.dividend, atol=1e-6).all():
        raise ValueError("公司行动现金分红字段口径不一致")
    events = attach_features(institutional_features(summary, brokers), bars, valuation, calendar)
    status = pd.read_parquet(export / "metadata/stock_status.parquet")
    status["trade_date"] = dates(status.pop("date"))
    status = status.loc[status.trade_date.between(lookback, end)]
    for kind in ("ST", "HALT"):
        if not set(sessions).issubset(set(status.loc[status.status_type.eq(kind), "trade_date"])):
            raise ValueError(f"请求区间缺少 {kind} 历史状态")
    metadata = pd.read_parquet(export / "metadata/stock_metadata.parquet")
    metadata["stock_code"] = metadata.symbol.str.replace(".SS", ".SH", regex=False)
    metadata = metadata.loc[metadata.stock_code.isin(codes), ["stock_code", "listed_date", "de_listed_date"]]
    benchmark = pd.read_parquet(export / "metadata/benchmark.parquet")
    benchmark["trade_date"] = dates(benchmark.pop("date"))
    benchmark = benchmark.loc[benchmark.trade_date.between(card.start_date, end)]
    frames = {"bars.parquet": bars, "events.parquet": events, "brokers.parquet": brokers,
              "actions.parquet": actions, "valuation.parquet": valuation, "status.parquet": status,
              "metadata.parquet": metadata, "benchmark.parquet": benchmark}
    for name, frame in frames.items():
        frame.to_parquet(temp / name, index=False)
    write_json(temp / "calendar.json", calendar)
    manifest = {"snapshot_id": key, "start": card.start_date, "end": end,
                "sources": inventory, "missing_action_files": missing_actions,
                "counts": {name: len(frame) for name, frame in frames.items()},
                "feature_rejections": events.base_reason.value_counts().to_dict(),
                "artifacts": {name: digest(temp / name) for name in [*frames, "calendar.json"]}}
    verify_inventory(inventory)
    write_json(temp / "manifest.json", manifest)
    temp.rename(folder)
    return folder, manifest
