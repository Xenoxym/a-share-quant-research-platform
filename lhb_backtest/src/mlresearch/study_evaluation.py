"""All declared cases, paired account comparisons, and exposure diagnostics."""
import numpy as np
import pandas as pd

from .evaluation import block_interval
from .study_data import size_buckets


def signal_metrics(panel, labels, predictions, cases, spec):
    rows = []
    for case in cases:
        target = panel if case["target"] == "close_5" else labels
        for (split, day), indices in panel.groupby(["split", "trade_date"]).groups.items():
            if case.get("test_only") and split != "test":
                continue
            actual = target.loc[indices]
            # Evaluation labels must fit inside the reported phase; inference stays.
            cutoff = getattr(spec, split + "_end")
            if not actual.label_end.notna().all() or not actual.label_end.le(cutoff).all():
                continue
            ordered = predictions.loc[indices, ["stock_code", case["id"]]].sort_values([case["id"], "stock_code"], ascending=[False, True])
            top = actual.loc[ordered.index[:spec.top_k]]
            observed = actual.label_observed
            score = predictions.loc[indices, case["id"]]
            ic = score[observed].rank().corr(actual.loc[observed, "label_return"].rank()) if score[observed].nunique() > 1 else np.nan
            mse = float(np.mean((score[observed] - actual.loc[observed, "label_relative"]) ** 2)) if case.get("regression") else np.nan
            rows.append(dict(model=case["id"], target=case["target"], split=split, trade_date=day, rank_ic=ic,
                top_excess=top.label_return.mean() - actual.label_return.mean(), mse=mse,
                pool_count=len(indices), label_count=int(observed.sum()), top_label_count=int(top.label_observed.sum())))
    daily = pd.DataFrame(rows)
    summaries = []
    for (model, split), part in daily.groupby(["model", "split"]):
        summaries.append(dict(model=model, scope=split, dates=len(part), rank_ic=part.rank_ic.mean(),
            rank_ic_interval=block_interval(part.rank_ic, spec.seed), top_excess=part.top_excess.mean(),
            top_excess_interval=block_interval(part.top_excess, spec.seed), mse=part.mse.mean(),
            label_coverage=part.label_count.sum()/part.pool_count.sum(), top_label_coverage=part.top_label_count.sum()/(len(part)*spec.top_k)))
    return daily, summaries


def select_validation(panel, labels, score, spec):
    valid = panel.split.eq("valid") & labels.label_end.notna() & labels.label_end.le(spec.valid_end)
    table = pd.DataFrame({"day": panel.loc[valid, "trade_date"], "score": np.asarray(score)[valid], "y": labels.loc[valid, "label_return"]})
    values = [p.score.rank().corr(p.y.rank()) for _, p in table.groupby("day")]
    return float(np.nanmean(values))


def exposure_diagnostics(panel, labels, predictions, cases, spec):
    exposures, conditional = [], []
    buckets = size_buckets(panel)
    for day, indices in panel[panel.split.eq("test")].groupby("trade_date").groups.items():
        X = panel.loc[indices]
        Y = labels.loc[indices]
        if Y.label_end.notna().all() and Y.label_end.le(spec.test_end).all():
            Z = X[["log_market_cap", "volatility_20", "return_20"]].copy()
            Z["log_amount"] = np.log(X.avg_amount_20)
            valid = Y.label_observed & np.isfinite(Z).all(axis=1)
            design = Z.loc[valid].to_numpy()
            design = (design - design.mean(axis=0)) / np.maximum(design.std(axis=0), 1e-12)
            design = np.c_[np.ones(len(design)), design]
            y = Y.loc[valid, "label_return"].to_numpy()
            residual = pd.Series(y - design @ np.linalg.lstsq(design, y, rcond=None)[0], index=Z.index[valid])
        else:
            residual = pd.Series(dtype=float)
        for case in cases:
            score = predictions.loc[indices, case["id"]]
            top_indices = pd.DataFrame({"score": score, "code": X.stock_code}).sort_values(["score", "code"], ascending=[False, True]).index[:spec.top_k]
            top = panel.loc[top_indices]
            exposures.append(dict(model=case["id"], trade_date=day,
                median_cap_cny=float(np.exp(top.log_market_cap.median())), mean_log_cap=top.log_market_cap.mean(),
                mean_amount_20=top.avg_amount_20.mean(), mean_volatility_20=top.volatility_20.mean(),
                mean_return_20=top.return_20.mean(), small_decile_fraction=buckets.loc[top_indices].eq(0).mean()))
            if len(residual):
                by_bucket = []
                for bucket in range(10):
                    sel = buckets.loc[indices].eq(bucket) & Y.label_observed
                    if sel.sum() > 2:
                        by_bucket.append(score[sel].rank().corr(Y.loc[sel, "label_return"].rank()))
                conditional.append(dict(model=case["id"], trade_date=day,
                    residual_rank_ic=score.loc[residual.index].rank().corr(residual.rank()),
                    within_size_rank_ic=np.nanmean(by_bucket)))
    return pd.DataFrame(exposures), pd.DataFrame(conditional)


def account_statistics(equity, initial_cash, seed, reference=None):
    nav = equity.set_index("date").equity.astype(float) / initial_cash
    returns = nav.pct_change().fillna(nav.iloc[0] - 1)
    excess = returns  # No risk-free series: label this a zero-rate diagnostic.
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(252)) if excess.std(ddof=1) > 0 else None
    annual = []
    previous = 1.
    for year, group in nav.groupby(nav.index.str[:4]):
        path = np.r_[previous, group.to_numpy()]
        annual.append(dict(year=year, return_value=float(path[-1]/previous-1),
            max_drawdown=float(np.min(path/np.maximum.accumulate(path)-1)), days=len(group)))
        previous = path[-1]
    result = dict(sharpe_zero_rate=sharpe, annual=annual)
    if reference is not None:
        # Paired calendar weeks, retaining the initial cash before first observation.
        ref = reference.set_index("date").equity.astype(float) / initial_cash
        dates = pd.to_datetime(nav.index)
        week = dates.to_period("W-FRI").astype(str)
        a = nav.groupby(week).last(); b = ref.groupby(week).last()
        delta = a.pct_change().fillna(a.iloc[0]-1) - b.pct_change().fillna(b.iloc[0]-1)
        result.update(mean_weekly_return_delta_vs_size=float(delta.mean()),
            weekly_delta_interval=block_interval(delta, seed), comparison_weeks=len(delta))
    return result
