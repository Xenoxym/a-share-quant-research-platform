"""Causal features and separately named labels, addressed on the market calendar."""
import numpy as np
import pandas as pd

from ..technical.foundation_data import attach_reports
from ..technical.signals import check_keys, decision_dates
from .contracts import FEATURES


def price_features(frame, calendar, min_history=60):
    b = frame.sort_values("trade_date").copy().reset_index(drop=True)
    b["session"] = b.trade_date.map({d: i for i, d in enumerate(calendar)})
    if b.session.isna().any():
        raise ValueError("行情日期不在快照日历")
    valid = b.close.gt(0) & b.pre_close.gt(0) & np.isfinite(b.close) & np.isfinite(b.pre_close)
    r = np.log((b.close / b.pre_close).where(valid))
    # Cumulative past reference returns, never adjusted with future actions.
    index = np.exp(r.fillna(0).cumsum())
    for window in [1, 5, 20, 60]:
        b[f"return_{window}"] = r.rolling(window).sum()
    for window in [5, 20, 60]:
        b[f"volatility_{window}"] = r.rolling(window).std()
        b[f"bias_{window}"] = index / index.rolling(window).mean() - 1
    b["range_1"] = (b.high - b.low) / b.pre_close
    b["body_1"] = (b.close - b.open) / b.pre_close
    b["avg_amount_20"] = b.amount.rolling(20).mean()
    b["amount_ratio_5_20"] = b.amount.rolling(5).mean() / b.avg_amount_20 - 1
    b["volume_ratio_5_20"] = b.volume.rolling(5).mean() / b.volume.rolling(20).mean() - 1
    b["amount_volatility_20"] = b.amount.rolling(20).std() / b.avg_amount_20
    b["close_location"] = ((b.close - b.low) / (b.high - b.low)).where(b.high.ne(b.low), 0.5)
    b["history_valid"] = b.session.sub(b.session.shift(min_history - 1)).eq(min_history - 1) & r.rolling(min_history).count().eq(min_history)
    b[list(FEATURES)] = b[list(FEATURES)].replace([np.inf, -np.inf], np.nan)
    b["_r"] = r
    return b


def forward_labels(features, calendar, horizon):
    b = features.copy()
    b["label_end"] = b.trade_date.map({d: calendar[i + horizon] for i, d in enumerate(calendar) if i + horizon < len(calendar)})
    future_sum = b._r.rolling(horizon).sum().shift(-horizon)
    contiguous = b.session.shift(-horizon).sub(b.session).eq(horizon)
    b["label_return"] = np.expm1(future_sum).where(contiguous)
    return b


def assign_splits(frame, spec):
    out = frame.copy()
    out["split"] = "excluded"
    for name, start, end, next_start in [
        ("train", spec.train_start, spec.train_end, spec.valid_start),
        ("valid", spec.valid_start, spec.valid_end, spec.test_start),
        ("test", spec.test_start, spec.test_end, None),
    ]:
        mask = out.trade_date.between(start, end)
        # Boundary purge is by the planned market label end, not panel row count.
        mature = out.label_end.notna() & out.label_end.le(end)
        if next_start:
            mature &= out.label_end.lt(next_start)
        out.loc[mask & mature, "split"] = name
    return out


def build_panel(bars, calendar, metadata, filings, spec, progress=lambda m: None, *, inference=False, eligibility="qualified"):
    if eligibility not in {"qualified", "available"}:
        raise ValueError("未知候选资格")
    check_keys(bars, "ML行情")
    schedule = decision_dates(calendar, spec.train_start, spec.test_end, spec.frequency)
    frames = []
    groups = bars.groupby("stock_code", sort=False, observed=True)
    for n, (code, raw) in enumerate(groups):
        features = price_features(raw, calendar, spec.min_history)
        labeled = forward_labels(features, calendar, spec.horizon)
        cols = ["stock_code", "trade_date", "close", "is_st", "volume", "history_valid", "avg_amount_20", "label_end", "label_return"] + list(FEATURES)
        frames.append(labeled.loc[labeled.trade_date.isin(schedule), cols])
        if n % 500 == 0:
            progress(f"构建过去价量特征与独立标签：{n + 1}/{groups.ngroups}只")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.merge(metadata[["stock_code", "listed_date", "de_listed_date"]], on="stock_code", validate="many_to_one")
    panel = attach_reports(panel, filings)
    cap_valid = panel.q_age_days.between(0, 240) & panel.estimated_market_cap.gt(0) & np.isfinite(panel.estimated_market_cap)
    panel["eligible"] = (panel.listed_date.le(panel.trade_date) & panel.de_listed_date.gt(panel.trade_date)
        & panel.is_st.eq(0) & panel.volume.gt(0) & panel.history_valid
        & panel.avg_amount_20.ge(spec.min_avg_amount) & cap_valid)
    panel["cap_valid"] = cap_valid
    if eligibility == "available":
        # No future-outcome, ST, liquidity, history-length or financial filter.
        panel["eligible"] = (panel.listed_date.le(panel.trade_date) & panel.de_listed_date.gt(panel.trade_date)
            & panel.volume.gt(0) & panel.close.gt(0) & np.isfinite(panel.close))
    panel["log_market_cap"] = np.log(panel.estimated_market_cap.where(cap_valid))
    panel = assign_splits(panel, spec)
    if inference:
        # Scoring does not require knowing the future: preserve terminal decisions.
        for split in ["train", "valid", "test"]:
            panel.loc[panel.trade_date.between(getattr(spec, split + "_start"), getattr(spec, split + "_end")), "split"] = split
    # Eligibility uses no label; keep unknown outcomes and disclose their coverage.
    panel = panel.loc[panel.eligible & panel.split.ne("excluded")].copy()
    panel["known_at"] = panel.trade_date + "T15:30:00+08:00"
    panel["label_observed"] = np.isfinite(panel.label_return)
    panel["label_relative"] = panel.label_return - panel.groupby("trade_date").label_return.transform("mean")
    cols = ["stock_code", "trade_date", "known_at", "label_end", "split", "label_observed", "label_return", "label_relative", "log_market_cap", "q_publication_date"] + list(FEATURES)
    if inference:
        cols += ["close", "avg_amount_20"]
    if eligibility == "available":
        cols += ["is_st", "history_valid", "cap_valid"]
    panel = panel[cols].sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
    check_keys(panel, "ML面板")
    if (panel.q_publication_date >= panel.trade_date).any():
        raise ValueError("股本资格使用了当日或未来公告")
    return panel
