"""Signal diagnostics, never an executable account or annualized return."""
import numpy as np
import pandas as pd

SCORES = ["ridge", "hist_gbdt", "small_cap", "reversal_5", "fixed_random"]


def block_interval(values, seed, block=4, repetitions=400):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 * block:
        return [None, None]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block)))
        indices = (starts[:, None] + np.arange(block)) % len(values)
        draws.append(float(values[indices.ravel()[:len(values)]].mean()))
    return np.quantile(draws, [0.025, 0.975]).tolist()


def evaluate(predictions, spec):
    daily, groups = [], []
    for (split, day), pool in predictions.groupby(["split", "trade_date"], sort=True):
        actual = pool.label_return
        pool_mean = actual.mean()
        for model in SCORES:
            ordered = pool.sort_values([model, "stock_code"], ascending=[False, True]).copy()
            ordered["quantile"] = np.minimum(np.arange(len(ordered)) * spec.bins // len(ordered) + 1, spec.bins)
            observed = pool.label_observed
            rank_ic = pool.loc[observed, model].rank().corr(pool.loc[observed, "label_return"].rank()) if pool.loc[observed, model].nunique() > 1 else np.nan
            ic = pool.loc[observed, model].corr(pool.loc[observed, "label_return"]) if pool.loc[observed, model].nunique() > 1 else np.nan
            top = ordered.head(spec.top_k)
            by_bin = ordered.groupby("quantile").label_return.mean()
            daily.append(dict(split=split, trade_date=day, model=model, rank_ic=rank_ic, ic=ic,
                pool_count=len(pool), label_count=int(observed.sum()), pool_mean=pool_mean,
                top_count=len(top), top_label_count=int(top.label_observed.sum()),
                top_mean=top.label_return.mean(), top_excess=top.label_return.mean() - pool_mean,
                spread=by_bin.get(1, np.nan) - by_bin.get(spec.bins, np.nan)))
            for quantile, group in ordered.groupby("quantile"):
                groups.append(dict(split=split, trade_date=day, model=model, quantile=int(quantile),
                    count=len(group), label_count=int(group.label_observed.sum()),
                    mean_return=group.label_return.mean(), excess=group.label_return.mean() - pool_mean))
    daily = pd.DataFrame(daily)
    groups = pd.DataFrame(groups)
    summaries = []
    scopes = [(split, split, daily[daily.split.eq(split)]) for split in ["train", "valid", "test"]]
    scopes += [(f"test_{year}", "test", part) for year, part in daily[daily.split.eq("test")].groupby(daily.trade_date.str[:4])]
    for scope, split, subset in scopes:
        for model, part in subset.groupby("model"):
            small = subset[subset.model.eq("small_cap")].set_index("trade_date").top_excess
            delta = part.set_index("trade_date").top_excess - small
            prediction_rows = predictions[predictions.split.eq(split)]
            if scope.startswith("test_"):
                prediction_rows = prediction_rows[prediction_rows.trade_date.str.startswith(scope[-4:])]
            observed = prediction_rows[prediction_rows.label_observed]
            mse = np.mean((observed[model] - observed.label_relative) ** 2) if model in {"ridge", "hist_gbdt"} else None
            zero_mse = np.mean(observed.label_relative ** 2)
            summaries.append(dict(scope=scope, model=model, dates=len(part), rank_ic=part.rank_ic.mean(),
                rank_ic_positive_fraction=part.rank_ic.dropna().gt(0).mean(),
                rank_ic_interval=block_interval(part.rank_ic, spec.seed),
                top_excess=part.top_excess.mean(), spread=part.spread.mean(),
                top_excess_interval=block_interval(part.top_excess, spec.seed),
                top_excess_vs_size=delta.mean(), size_delta_interval=block_interval(delta, spec.seed),
                mse=mse, zero_prediction_mse=zero_mse,
                r2_vs_zero=(1 - mse / zero_mse) if mse is not None and zero_mse > 0 else None,
                label_coverage=float(part.label_count.sum() / part.pool_count.sum()),
                top_label_coverage=float(part.top_label_count.sum() / part.top_count.sum())))
    quantiles = groups.groupby(["split", "model", "quantile"])[["mean_return", "excess"]].mean().reset_index()
    return daily, groups, summaries, quantiles
