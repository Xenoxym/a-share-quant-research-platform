"""Weekly execution-aligned labels and a label-independent score interface."""
import hashlib
import numpy as np
import pandas as pd

from ..technical.signals import decision_dates


def open_labels(panel, bars, calendar, spec):
    schedule = decision_dates(calendar, spec.train_start, spec.test_end, spec.frequency)
    dates = sorted(schedule)
    exits = {day: schedule[dates[i + 1]] for i, day in enumerate(dates[:-1])}
    out = panel[["stock_code", "trade_date", "split"]].copy()
    out["label_entry"] = out.trade_date.map(schedule)
    out["label_end"] = out.trade_date.map(exits)
    out["label_return"] = np.nan
    pos = {day: i for i, day in enumerate(calendar)}
    # Use reference-price links across actions, rather than raw open/open ratios.
    grouped = out.groupby("stock_code", sort=False).groups
    for code, raw in bars.groupby("stock_code", sort=False):
        if code not in grouped:
            continue
        rows = out.loc[grouped[code]]
        raw = raw[["trade_date", "open", "close", "pre_close"]].set_index("trade_date").reindex(calendar)
        r = np.log((raw.close / raw.pre_close).where(raw.close.gt(0) & raw.pre_close.gt(0))).to_numpy()
        sums = np.r_[0., np.cumsum(np.where(np.isfinite(r), r, 0.))]
        bad = np.r_[0, np.cumsum(~np.isfinite(r))]
        has_end = rows.label_end.notna()
        selected = rows[has_end]
        a = selected.label_entry.map(pos).to_numpy(dtype=int)
        b = selected.label_end.map(pos).to_numpy(dtype=int)
        op, pc = raw.open.to_numpy(), raw.pre_close.to_numpy()
        valid = (b > a) & (bad[b] == bad[a]) & np.isfinite(op[a]) & np.isfinite(pc[a]) & np.isfinite(op[b]) & np.isfinite(pc[b]) & (op[a] > 0) & (pc[a] > 0) & (op[b] > 0) & (pc[b] > 0)
        total = sums[b] - sums[a] + np.log(pc[a] / op[a]) + np.log(op[b] / pc[b])
        out.loc[selected.index, "label_return"] = np.where(valid, np.expm1(total), np.nan)
    out["label_observed"] = np.isfinite(out.label_return)
    out["label_relative"] = out.label_return - out.groupby("trade_date").label_return.transform("mean")
    return out


def fixed_scores(panel, seed):
    scores = {code: int(hashlib.sha256(f"{seed}:{code}".encode()).hexdigest()[:13], 16) / 16 ** 13 for code in panel.stock_code.unique()}
    return pd.DataFrame({"small_cap": -panel.log_market_cap, "reversal_5": -panel.return_5,
        "fixed_random": panel.stock_code.map(scores)}, index=panel.index)


def size_buckets(panel):
    # Stable ties by code; use only decision-time market cap.
    ordered = panel.sort_values(["trade_date", "log_market_cap", "stock_code"])
    rank = ordered.groupby("trade_date").cumcount()
    count = ordered.groupby("trade_date").stock_code.transform("size")
    return (rank * 10 // count).astype(int).reindex(panel.index)


def balanced_score(panel, score):
    ordered = panel.assign(_score=np.asarray(score), _bucket=size_buckets(panel)).sort_values(
        ["trade_date", "_bucket", "_score", "stock_code"], ascending=[True, True, False, True])
    rank = ordered.groupby(["trade_date", "_bucket"]).cumcount()
    return (-rank * 10 - ordered._bucket).reindex(panel.index).astype(float)


def score_decisions(panel, score, schedule, top_k, allocation=.98):
    # Never drop a stock because its future label is missing.
    ordered = panel.assign(score=np.asarray(score)).sort_values(["trade_date", "score", "stock_code"], ascending=[True, False, True]).copy()
    ordered["rank"] = ordered.groupby("trade_date").cumcount() + 1
    selected = ordered[ordered["rank"].le(top_k)].copy()
    selected["selected"] = True
    selected["reason"] = "eligible_external_score"
    selected["target_weight"] = allocation / top_k
    selected["execution_date"] = selected.trade_date.map(schedule)
    selected["known_at_assumed"] = selected.trade_date + "T15:30:00+08:00"
    return selected[["stock_code", "trade_date", "close", "score", "rank", "selected", "reason", "target_weight", "execution_date", "known_at_assumed"]]
