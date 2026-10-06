"""Pure functions over prices/volume and dated security metadata. No event data."""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["stock_code", "trade_date"]
BAR_COLS = KEYS + ["open", "high", "low", "close", "pre_close", "volume", "amount",
                  "high_limit", "low_limit", "limit_price_valid", "is_st"]
MAIN_BOARD = r"^(?:60\d{4}\.SH|00\d{4}\.SZ)$"


def check_keys(frame, name):
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError(f"{name} 存在重复或缺失主键")


def features(bars, calendar, strategy):
    b = bars.sort_values(KEYS).copy().reset_index(drop=True)
    check_keys(b, "行情")
    b["session"] = b.trade_date.map({d: i for i, d in enumerate(calendar)})
    if b.session.isna().any():
        raise ValueError("行情日期不在市场日历中")
    valid = np.isfinite(b.close) & np.isfinite(b.pre_close) & b.close.gt(0) & b.pre_close.gt(0)
    b["r"] = np.log((b.close / b.pre_close).where(valid))
    g = b.groupby("stock_code", sort=False, observed=True)
    b["avg_amount_20"] = g.amount.transform(lambda v: v.rolling(20).mean())
    b["avg_volume_20"] = g.volume.transform(lambda v: v.rolling(20).mean())
    contiguous20 = b.session.sub(g.session.shift(19)).eq(19)
    b.loc[~contiguous20, ["avg_amount_20", "avg_volume_20"]] = np.nan
    s = strategy
    b["momentum"] = g.r.transform(lambda v: np.expm1(v.rolling(s.lookback-s.skip).sum().shift(s.skip)))
    b["x"] = np.exp(g.r.transform(lambda v: v.fillna(0).cumsum()))
    if s.trend_window:
        b["trend"] = b.x / b.groupby("stock_code", sort=False).x.transform(lambda v: v.rolling(s.trend_window).mean()) - 1
    else:
        b["trend"] = 0.
    if s.family == "low_volatility":
        b["volatility"] = g.r.transform(lambda v: v.rolling(s.lookback).std(ddof=1)) * np.sqrt(252)
    history = max(20, s.lookback, s.trend_window, s.min_history)
    b["history_valid"] = b.session.sub(g.session.shift(history-1)).eq(history-1) & g.r.transform(lambda v: v.rolling(history).count()).eq(history)
    b["score"] = b.momentum if s.direction == "momentum" else -b.momentum
    return b.drop(columns=["r", "x"])


def decision_dates(calendar, start, end, frequency):
    def period(d):
        return d[:7] if frequency == "monthly" else tuple(pd.Timestamp(d).isocalendar()[:2])
    return {d: nxt for d, nxt in zip(calendar[:-1], calendar[1:])
            if start <= d <= end and period(d) != period(nxt)}


def decisions(bars, calendar, metadata, spec, fundamentals=None, benchmark=None, actions=None):
    if spec.strategy.family != "price" or spec.strategy.require_fundamentals:
        from .foundation_signals import decide
        return decide(bars, calendar, metadata, spec, fundamentals, benchmark, actions)
    schedule = decision_dates(calendar, spec.experiment.start_date, spec.experiment.end_date, spec.strategy.rebalance)
    rows = bars.loc[bars.trade_date.isin(schedule)].copy()
    rows = rows.merge(metadata[["stock_code", "listed_date", "de_listed_date"]], on="stock_code", how="left", validate="many_to_one")
    s = spec.strategy
    listed = rows.listed_date.le(rows.trade_date) & rows.de_listed_date.gt(rows.trade_date)
    rows["reason"] = np.select([
        ~listed, ~rows.is_st.eq(0), ~rows.limit_price_valid.eq(1), ~rows.history_valid,
        ~np.isfinite(rows.avg_amount_20) | rows.avg_amount_20.lt(s.min_avg_amount) | rows.avg_volume_20.le(0),
        ~np.isfinite(rows.score), rows.score.le(s.min_score) if s.min_score is not None else np.zeros(len(rows), dtype=bool),
        rows.trend.le(0) if s.trend_window else np.zeros(len(rows), dtype=bool)],
        ["not_listed_or_unknown", "st_or_unknown", "unknown_limits", "insufficient_history", "liquidity", "invalid_score", "score_threshold", "trend_filter"], default="eligible")
    rows = rows.sort_values(["trade_date", "score", "stock_code"], ascending=[True, False, True])
    eligible = rows.reason.eq("eligible")
    rows["rank"] = pd.Series(index=rows.index, dtype=float)
    rows.loc[eligible, "rank"] = rows.loc[eligible].groupby("trade_date").cumcount() + 1
    rows["selected"] = rows["rank"].le(s.top_k)
    rows["target_weight"] = np.where(rows.selected, s.allocation / s.top_k, 0.)
    rows["execution_date"] = rows.trade_date.map(schedule)
    rows["known_at_assumed"] = rows.trade_date + "T15:30:00+08:00"
    cols = KEYS + ["known_at_assumed", "execution_date", "close", "momentum", "score", "trend", "avg_amount_20", "avg_volume_20", "reason", "rank", "selected", "target_weight"]
    return rows[cols].reset_index(drop=True), schedule
