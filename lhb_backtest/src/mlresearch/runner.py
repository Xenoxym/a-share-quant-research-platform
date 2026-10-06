from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ..technical.artifacts import content_id, digest, records, verify_artifacts, write_json
from .contracts import MLSpec, FEATURES, feature_names
from .dataset import build_panel
from .evaluation import evaluate


def fit_models(panel, spec):
    names = feature_names(spec)
    train = panel[panel.split.eq("train") & panel.label_observed]
    if len(train) < max(100, spec.tree_min_samples * 2):
        raise ValueError("ML训练样本不足")
    if train[names].notna().sum().eq(0).any():
        raise ValueError("训练段存在全空特征，不能悄悄删除输入列")
    X, y = train[names], train.label_relative
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=spec.ridge_alpha))
    tree = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(
        max_iter=spec.tree_iterations, max_leaf_nodes=spec.tree_leaves,
        min_samples_leaf=spec.tree_min_samples, learning_rate=spec.tree_learning_rate,
        l2_regularization=spec.tree_l2, early_stopping=False, random_state=spec.seed))
    models = {"ridge": ridge, "hist_gbdt": tree}
    with threadpool_limits(limits=4):
        for model in models.values():
            model.fit(X, y)
    return models


def predict_panel(panel, models, spec):
    predicted = panel[["stock_code", "trade_date", "label_end", "split", "label_observed", "label_return", "label_relative"]].copy()
    with threadpool_limits(limits=4):
        for name, model in models.items():
            predicted[name] = model.predict(panel[feature_names(spec)])
    predicted["small_cap"] = -panel.log_market_cap
    predicted["reversal_5"] = -panel.return_5
    scores = {c: int(hashlib.sha256(f"{spec.seed}:{c}".encode()).hexdigest()[:13], 16) / 16 ** 13 for c in panel.stock_code.unique()}
    predicted["fixed_random"] = panel.stock_code.map(scores)
    return predicted


def run(cfg, progress=lambda m: None):
    if cfg["spec"].get("study") == "financial_context":
        from .financial_study import run_financial
        return run_financial(cfg, progress)
    if cfg["spec"].get("study") == "fixed_blend":
        from .fixed_blend import run_blend
        return run_blend(cfg, progress)
    if cfg["spec"].get("study") == "universe":
        from .universe import run_universe
        return run_universe(cfg, progress)
    if cfg["spec"].get("study") == "complete":
        from .study import run_study
        return run_study(cfg, progress)
    started = time.perf_counter()
    root = Path(cfg["root"])
    snapshot = root / "snapshots" / cfg["snapshot_id"]
    sm = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if digest(snapshot / "manifest.json") != cfg["snapshot_manifest_hash"]:
        raise ValueError("ML快照清单变化")
    verify_artifacts(snapshot, sm)
    spec = MLSpec.from_dict(cfg["spec"])
    environment = {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in cfg["environment"]["packages"]}}
    if environment != cfg["environment"]:
        raise ValueError("ML执行环境与登记版本不同")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    folder = root / "ml_runs" / run_id
    folder.mkdir(parents=True)
    write_json(folder / "status.json", {"status": "running", "run_id": run_id, "kind": "ml"})
    try:
        write_json(folder / "spec.json", spec.to_dict())
        write_json(folder / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "joblib", "threadpoolctl"]}, "thread_limit": 4})
        calendar = json.loads((snapshot / "calendar.json").read_text(encoding="utf-8"))
        columns = ["stock_code", "trade_date", "open", "high", "low", "close", "pre_close", "volume", "amount", "is_st"]
        bars = pd.read_parquet(snapshot / "bars.parquet", columns=columns, filters=[("trade_date", "<=", spec.test_end)])
        metadata = pd.read_parquet(snapshot / "metadata.parquet")
        filings = pd.read_parquet(snapshot / "fundamentals.parquet")
        panel = build_panel(bars, calendar, metadata, filings, spec, progress)
        del bars
        counts = []
        for split in ["train", "valid", "test"]:
            part = panel[panel.split.eq(split)]
            if part.empty or part.label_observed.sum() < 100 or part.trade_date.nunique() < 2:
                raise ValueError(f"{split}有效日期或标签不足")
            counts.append(dict(split=split, rows=len(part), dates=part.trade_date.nunique(),
                observed_labels=int(part.label_observed.sum()), first_date=part.trade_date.min(), last_date=part.trade_date.max(), max_label_end=part.label_end.max()))
        panel.to_parquet(folder / "panel.parquet", index=False)
        write_json(folder / "dataset.json", {"counts": counts, "features": feature_names(spec), "feature_definitions": FEATURES,
            "feature_set": spec.feature_set, "panel_sha256": digest(folder / "panel.parquet"),
            "target": "未来h个市场交易日参考价衔接收益，减同日有标签候选池均值",
            "eligibility": "当日主板上市、非ST、连续历史、成交额和已公告且不过期的股本；不使用未来标签筛候选",
            "missing_labels": "训练需可观测标签；评价先对全部候选排序，再统计已知结果及覆盖率，存在非随机缺失风险",
            "boundary": "label_end不超过各段末日；多日跨界标签排除",
            "training": "只拟合train，valid不拟合不选择模型；固定两种模型均报告，不自动挑赢家",
            "evaluation_scope": "retrospective_time_holdout; not genuinely unseen history"})
        progress(f"训练固定岭回归和受限提升树，{counts[0]['rows']}个候选训练行")
        models = fit_models(panel, spec)
        for name, model in models.items():
            joblib.dump(model, folder / f"{name}.joblib")
        predictions = predict_panel(panel, models, spec)
        predictions.to_parquet(folder / "predictions.parquet", index=False)
        progress("评价日期内排序、十分组收益与Top100相对池收益")
        daily, groups, summaries, quantiles = evaluate(predictions, spec)
        daily.to_parquet(folder / "date_metrics.parquet", index=False)
        groups.to_parquet(folder / "quantile_dates.parquet", index=False)
        quantiles.to_csv(folder / "quantiles.csv", index=False, encoding="utf-8-sig")
        result = {"kind": "ml", "run_id": run_id, "name": spec.name, "snapshot_id": cfg["snapshot_id"],
            "research_context": {"task_id": cfg["task_id"], "experiment_id": cfg["experiment_id"]},
            "counts": counts, "summaries": summaries, "quantiles": records(quantiles), "elapsed_seconds": time.perf_counter() - started,
            "limitations": ["全部历史已被查看；这里只对模型拟合按时间隔离", "信号收益不是可成交账户收益，不计算年化或费用反事实", "股本滞后与供应商历史修订/退市覆盖仍受原快照限制", "未知未来标签可能非随机，展示总体及入选覆盖率", "4个采样日期循环块区间为逐项诊断，不校正全部研究尝试或多重比较", "固定随机11是一个种子对照，不是通用市场基准"]}
        write_json(folder / "result.json", result)
        rows = ["机器学习基准：" + spec.name, "", "训练/验证/测试按日期隔离；固定模型，不选择测试冠军。", "", "|区间|模型|RankIC|Top相对池收益/期|相对小市值/期|", "|---|---|---:|---:|---:|"]
        for row in summaries:
            if row["scope"] != "train":
                rows.append(f"|{row['scope']}|{row['model']}|{row['rank_ic']:.4f}|{row['top_excess']:.4%}|{row['top_excess_vs_size']:.4%}|")
        rows += ["", *result["limitations"]]
        (folder / "REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
        write_json(folder / "status.json", {"status": "completed", "run_id": run_id, "kind": "ml", "name": spec.name})
        write_json(folder / "manifest.json", {"kind": "ml", "run_id": run_id, "spec": spec.to_dict(),
            "snapshot_id": cfg["snapshot_id"], "snapshot_manifest_sha256": cfg["snapshot_manifest_hash"],
            "code_hash": cfg["engine_hash"], "artifacts": {p.name: digest(p) for p in folder.iterdir() if p.is_file()}})
        return folder
    except BaseException as exc:
        write_json(folder / "status.json", {"status": "failed", "run_id": run_id, "kind": "ml", "error": str(exc)})
        raise
