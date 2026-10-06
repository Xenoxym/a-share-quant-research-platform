"""Outcome-blind matching and descriptive historical robustness checks."""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_cohorts(events, card):
    events = events.copy()
    events["eligibility_reason"] = events.base_reason
    low_liquidity = events.base_reason.eq("eligible") & events.avg_amount_20.lt(card.min_avg_amount)
    events.loc[low_liquidity, "eligibility_reason"] = "below_liquidity_threshold"
    eligible = events.loc[events.eligibility_reason.eq("eligible")].copy()
    eligible["signal"] = eligible.inst_ratio.gt(card.min_inst_ratio)
    pairs = []
    for day, cohort in eligible.groupby("trade_date", sort=True):
        # Cross-sectional ranks use only information available before both orders.
        covariates = cohort[["float_value", "momentum_20", "avg_amount_20"]].rank(pct=True, method="average")
        treated = cohort.loc[cohort.signal].sort_values("event_id")
        controls = cohort.loc[~cohort.signal].sort_values("event_id")
        remaining = list(controls.index)
        for idx, row in treated.iterrows():
            if not remaining:
                break
            distances = (covariates.loc[remaining] - covariates.loc[idx]).abs().sum(axis=1)
            chosen = distances.idxmin()
            distance = float(distances.loc[chosen])
            if distance <= card.match_caliper:
                pairs.append({"pair_id": f"pair-{len(pairs) + 1:07d}", "trade_date": day,
                              "treated_event_id": row.event_id, "control_event_id": cohort.loc[chosen, "event_id"],
                              "distance": distance})
                remaining.remove(chosen)
    pairs = pd.DataFrame(pairs, columns=["pair_id", "trade_date", "treated_event_id", "control_event_id", "distance"])
    selected_ids = set(eligible.loc[eligible.signal, "event_id"])
    events["signal"] = events.event_id.isin(selected_ids)
    events["matched_treatment"] = events.event_id.isin(pairs.treated_event_id)
    events["matched_control"] = events.event_id.isin(pairs.control_event_id)
    return events, pairs


def price_labels(events, market, card, observation_end):
    """Fixed-date opening-price entitlement labels, not trade profits."""
    rows = []
    for event in events.loc[events.eligibility_reason.eq("eligible")].itertuples(index=False):
        entry = event.entry_date
        result = {"event_id": event.event_id, "label_return": np.nan, "label_exit_date": None,
                  "label_status": "beyond_observation_window"}
        if pd.isna(entry) or entry > observation_end:
            rows.append(result)
            continue
        start_idx = market.day_index[entry]
        exit_idx = start_idx + card.holding_days
        if exit_idx >= len(market.calendar) or market.calendar[exit_idx] > observation_end:
            rows.append(result)
            continue
        exit_day = market.calendar[exit_idx]
        result["label_exit_date"] = exit_day
        first, last = market.quote(event.stock_code, entry), market.quote(event.stock_code, exit_day)
        if first is None or last is None or first["open"] <= 0 or last["open"] <= 0:
            result["label_status"] = "missing_endpoint_quote"
            rows.append(result)
            continue
        if event.stock_code in market.missing_actions:
            result["label_status"] = "missing_action_source"
            rows.append(result)
            continue
        # Unit economic exposure, fractional bonus shares; no claim of an executable lot.
        shares, dividend = 1., 0.
        for day in market.calendar[start_idx + 1:exit_idx + 1]:
            action = market.actions.get((event.stock_code, day))
            if action:
                dividend += shares * action["dividend"] * (1 - card.dividend_tax_reserve)
                shares *= 1 + action["allotted_ps"]
        result["label_return"] = (shares * last["open"] + dividend) / first["open"] - 1
        result["label_status"] = "observed_price_label"
        rows.append(result)
    return pd.DataFrame(rows, columns=["event_id", "label_return", "label_exit_date", "label_status"])


def block_interval(pairs, sessions, samples, seed, block_length):
    valid = pairs.dropna(subset=["difference"])
    if valid.empty:
        return {"mean": None, "low": None, "high": None, "pairs": 0, "event_days": 0}
    daily = valid.groupby("trade_date").difference.agg(["sum", "count"]).reindex(sessions, fill_value=0)
    mean = float(valid.difference.mean())
    day_count = int(valid.trade_date.nunique())
    output = {"mean": mean, "pairs": len(valid), "event_days": day_count, "low": None, "high": None,
              "method": "circular moving trading-day blocks; paired mean", "block_length": block_length}
    if day_count < 30:
        return output
    rng = np.random.default_rng(seed)
    totals, counts = daily["sum"].to_numpy(), daily["count"].to_numpy()
    n = len(daily)
    estimates = []
    for _ in range(samples):
        starts = rng.integers(0, n, size=int(np.ceil(n / block_length)))
        indices = ((starts[:, None] + np.arange(block_length)) % n).ravel()[:n]
        denominator = counts[indices].sum()
        if denominator:
            estimates.append(totals[indices].sum() / denominator)
    output.update(low=float(np.quantile(estimates, .025)), high=float(np.quantile(estimates, .975)))
    return output


def describe_research(events, pairs, labels, sessions, card):
    annotated = events.merge(labels, on="event_id", how="left", validate="one_to_one")
    values = labels.set_index("event_id")
    pairs = pairs.copy()
    pairs["treated_return"] = pairs.treated_event_id.map(values.label_return)
    pairs["control_return"] = pairs.control_event_id.map(values.label_return)
    pairs["label_exit_date"] = pairs.treated_event_id.map(values.label_exit_date)
    pairs["difference"] = pairs.treated_return - pairs.control_return
    yearly = []
    for year in sorted({d[:4] for d in sessions}):
        # Purge labels crossing the boundary; every period is descriptive, not a new OOS claim.
        period = pairs.loc[pairs.trade_date.str.startswith(year) & pairs.label_exit_date.fillna("").le(year + "-12-31")]
        valid = period.dropna(subset=["difference"])
        yearly.append({"year": year, "valid_pairs": len(valid), "treated_mean": valid.treated_return.mean(),
                       "control_mean": valid.control_return.mean(), "difference": valid.difference.mean()})
    distribution = []
    for label, mask in [("全部基础事件", annotated.eligibility_reason.eq("eligible")),
                        ("机构信号", annotated.signal), ("配对机构组", annotated.matched_treatment),
                        ("配对对照组", annotated.matched_control)]:
        part = annotated.loc[mask]
        observed = part.label_return.dropna()
        distribution.append({"cohort": label, "events": len(part), "observed": len(observed),
                             "mean": observed.mean(), "median": observed.median(),
                             "p05": observed.quantile(.05), "p95": observed.quantile(.95),
                             "positive_rate": float(observed.gt(0).mean()) if len(observed) else None})
    balance = []
    eligible = annotated.loc[annotated.eligibility_reason.eq("eligible")]
    for feature in ["float_value", "momentum_20", "avg_amount_20"]:
        transformed = np.log(eligible[feature]) if feature != "momentum_20" else eligible[feature]
        scale = transformed.std(ddof=1)
        for stage, treated, control in [
            ("before", eligible.signal, ~eligible.signal),
            ("matched", eligible.matched_treatment, eligible.matched_control)]:
            balance.append({"feature": feature, "stage": stage,
                            "standardized_difference": (transformed.loc[treated].mean() - transformed.loc[control].mean()) / scale if scale > 0 else None})
    interval = block_interval(pairs, sessions, card.bootstrap_samples, card.seed, max(5, 2 * card.holding_days))
    summary = {"funnel": {"all_events": len(events), "base_eligible": int(events.eligibility_reason.eq("eligible").sum()),
                           "institutional_signals": int(events.signal.sum()), "matched_pairs": len(pairs),
                           "unmatched_signals": int(events.signal.sum()) - len(pairs),
                           "observed_pairs": int(pairs.difference.notna().sum())},
               "rejections": events.eligibility_reason.value_counts().to_dict(),
               "label_status": annotated.loc[annotated.eligibility_reason.eq("eligible"), "label_status"].value_counts().to_dict(),
               "distribution": distribution, "yearly": yearly, "balance": balance, "paired_interval": interval,
               "interpretation": "同日按市值、20日动量、20日成交额分位排名做无放回近邻配对。配对不使用结果；未匹配事件仍进入完整机构组合。价格标签不含交易费用或成交约束；时间块区间不能消除全部个股相关性、多次尝试及未观测混杂。"}
    return annotated, pairs, summary
