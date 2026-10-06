"""Deterministic artifact/split/prediction checks, not independent investment review."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..technical.artifacts import digest, verify_artifacts
from .contracts import MLSpec, feature_names
from .evaluation import SCORES


def audit(folder, root):
    folder, root = Path(folder), Path(root)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["spec"].get("study") == "financial_context":
        from .financial_audit import audit_financial
        return audit_financial(folder, root)
    if manifest["spec"].get("study") == "fixed_blend":
        from .blend_audit import audit_blend
        return audit_blend(folder, root)
    if manifest["spec"].get("study") == "universe":
        from .universe_audit import audit_universe
        return audit_universe(folder, root)
    if manifest["spec"].get("study") == "complete":
        from .study_audit import audit_study
        return audit_study(folder, root)
    verify_artifacts(folder, manifest)
    snapshot = root / "snapshots" / manifest["snapshot_id"]
    if digest(snapshot / "manifest.json") != manifest["snapshot_manifest_sha256"]:
        raise ValueError("ML快照清单身份不一致")
    verify_artifacts(snapshot, json.loads((snapshot / "manifest.json").read_text(encoding="utf-8")))
    spec = MLSpec.from_dict(manifest["spec"])
    panel = pd.read_parquet(folder / "panel.parquet")
    predicted = pd.read_parquet(folder / "predictions.parquet")
    if not panel[["stock_code", "trade_date", "split"]].equals(predicted[["stock_code", "trade_date", "split"]]):
        raise ValueError("预测与面板身份不一致")
    if panel.duplicated(["stock_code", "trade_date"]).any() or not np.isfinite(predicted[SCORES]).all().all():
        raise ValueError("ML主键或预测无效")
    if not panel.label_observed.equals(np.isfinite(panel.label_return)):
        raise ValueError("标签可观测标记不一致")
    for key in ["label_end", "label_observed", "label_return", "label_relative"]:
        if not panel[key].equals(predicted[key]):
            raise ValueError("预测文件的标签与面板不同")
    for split in ["train", "valid", "test"]:
        p = panel[panel.split.eq(split)]
        if p.empty or not p.trade_date.between(getattr(spec, split + "_start"), getattr(spec, split + "_end")).all() or not p.label_end.le(getattr(spec, split + "_end")).all() or not p.label_end.gt(p.trade_date).all():
            raise ValueError("日期切分或标签成熟边界无效")
    if not panel.q_publication_date.lt(panel.trade_date).all():
        raise ValueError("资格股本公告边界无效")
    train = panel[panel.split.eq("train") & panel.label_observed]
    names = feature_names(spec)
    sample = panel.sample(n=min(500, len(panel)), random_state=spec.seed)
    label_sample = sample.head(100)
    raw = pd.read_parquet(snapshot / "bars.parquet", columns=["stock_code", "trade_date", "close", "pre_close"],
        filters=[("stock_code", "in", label_sample.stock_code.unique().tolist())]).set_index(["stock_code", "trade_date"])
    calendar = json.loads((snapshot / "calendar.json").read_text(encoding="utf-8"))
    positions = {day: i for i, day in enumerate(calendar)}
    label_error = 0.0
    for row in label_sample.itertuples():
        i = positions[row.trade_date]
        dates = calendar[i + 1:i + spec.horizon + 1]
        if len(dates) != spec.horizon or dates[-1] != row.label_end:
            raise ValueError("标签末日不是市场日历目标")
        quotes = raw.reindex(pd.MultiIndex.from_tuples([(row.stock_code, d) for d in dates]))
        ratio = quotes.close / quotes.pre_close
        expected = float(np.prod(ratio) - 1) if ratio.notna().all() and np.isfinite(ratio).all() and ratio.gt(0).all() else np.nan
        if not np.allclose(expected, row.label_return, atol=1e-10, equal_nan=True):
            raise ValueError("市场日历寻址的未来收益复算不同")
        if np.isfinite(expected):
            label_error = max(label_error, abs(expected - row.label_return))
    errors = {}
    for name in ["ridge", "hist_gbdt"]:
        # Only our own hash-verified local model is loaded, never user uploads.
        model = joblib.load(folder / f"{name}.joblib")
        if not np.allclose(model.steps[0][1].statistics_, train[names].median().to_numpy(), rtol=1e-12, atol=1e-12):
            raise ValueError("预处理填补统计并非训练段")
        with threadpool_limits(limits=4):
            scores = model.predict(sample[names])
        error = float(np.max(np.abs(scores - predicted.loc[sample.index, name].to_numpy())))
        errors[name] = error
        if error > 1e-10:
            raise ValueError("保存模型无法复现预测抽样")
    metrics = pd.read_parquet(folder / "date_metrics.parquet")
    metric_error = 0.0
    for (_, day), p in predicted.groupby(["split", "trade_date"]):
        for name in SCORES:
            observed = p[p.label_observed]
            a, b = observed[name].rank().to_numpy(), observed.label_return.rank().to_numpy()
            expected_ic = np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else np.nan
            row = metrics[metrics.trade_date.eq(day) & metrics.model.eq(name)]
            if len(row) != 1:
                raise ValueError("日期指标记录缺失或重复")
            actual_ic = row.iloc[0].rank_ic
            top = p.sort_values([name, "stock_code"], ascending=[False, True]).head(spec.top_k)
            excess = top.label_return.mean() - p.label_return.mean()
            if not np.allclose([expected_ic, excess], [actual_ic, row.iloc[0].top_excess], atol=1e-12, equal_nan=True):
                raise ValueError("排序或Top收益指标数值不一致")
            if np.isfinite(expected_ic):
                metric_error = max(metric_error, abs(float(expected_ic - actual_ic)))
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    for summary in result["summaries"]:
        part = metrics[metrics.model.eq(summary["model"])]
        part = part[part.split.eq(summary["scope"])] if summary["scope"] in {"train", "valid", "test"} else part[part.split.eq("test") & part.trade_date.str.startswith(summary["scope"][-4:])]
        if not np.allclose([part.rank_ic.mean(), part.top_excess.mean()], np.asarray([summary["rank_ic"], summary["top_excess"]], dtype=float), atol=1e-12, equal_nan=True):
            raise ValueError("汇总指标不一致")
    return {"kind": "ml", "run_id": folder.name, "verified": True, "rows": len(panel),
        "prediction_sample_max_error": errors, "rank_ic_max_error": metric_error,
        "label_sample_max_error": label_error,
        "verification_scope": "文件与版本身份、时间切分、训练填补统计、市场日历标签抽样、保存模型预测抽样和排序指标复算；不是账户核查或独立研究复核"}
