"""Numerical and linked-account checks of the finite study, not external review."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..technical.artifacts import digest, verify_artifacts
from ..technical.signals import decision_dates
from .contracts import MLSpec, FEATURES
from .study import study_plan
from .study_data import score_decisions
from .study_evaluation import signal_metrics, select_validation


def read(folder, name):
    return json.loads((folder / name).read_text(encoding="utf-8"))


def verify_links(folder, root):
    folder, root = Path(folder), Path(root)
    links = read(folder, "links.json")
    source = root / "ml_runs" / links["source_run_id"]
    if digest(source / "manifest.json") != links["source_manifest_sha256"]:
        raise ValueError("原基准清单变化")
    verify_artifacts(source, read(source, "manifest.json"))
    for link in links["accounts"]:
        account = root / "runs" / link["run_id"]
        if digest(account / "manifest.json") != link["manifest_sha256"]:
            raise ValueError("子账户清单变化")
        verify_artifacts(account, read(account, "manifest.json"))
        result = read(account, "result.json")
        if result["research_context"]["parent_ml_run"] != folder.name or result["research_context"]["case"] != link["case"]:
            raise ValueError("子账户归属变化")
        if result["definition"]["signal_parent"]["predictions_sha256"] != digest(folder / "predictions.parquet"):
            raise ValueError("子账户引用预测版本不同")
    return links


def audit_study(folder, root):
    folder, root = Path(folder), Path(root)
    manifest = read(folder, "manifest.json"); verify_artifacts(folder, manifest)
    spec = MLSpec.from_dict(manifest["spec"]); plan = study_plan(spec)
    if plan != read(folder, "study_plan.json"):
        raise ValueError("完整矩阵与冻结设计不同")
    snapshot = root / "snapshots" / manifest["snapshot_id"]
    if digest(snapshot / "manifest.json") != manifest["snapshot_manifest_sha256"]:
        raise ValueError("快照版本不同")
    verify_artifacts(snapshot, read(snapshot, "manifest.json"))
    links = verify_links(folder, root)
    panel = pd.read_parquet(folder / "panel.parquet")
    labels = pd.read_parquet(folder / "open_labels.parquet")
    predictions = pd.read_parquet(folder / "predictions.parquet")
    keys = ["stock_code", "trade_date", "split"]
    if not panel[keys].equals(labels[keys]) or not panel[keys].equals(predictions[keys]) or panel.duplicated(keys[:2]).any():
        raise ValueError("完整研究主键不一致")
    if not panel.q_publication_date.lt(panel.trade_date).all():
        raise ValueError("当日或未来公告用于输入资格")
    if not labels.label_observed.equals(np.isfinite(labels.label_return)):
        raise ValueError("标签覆盖标记错误")
    cases = plan["cases"]
    if set(a["case"] for a in links["accounts"]) != {c["id"] for c in cases}:
        raise ValueError("账户矩阵不完整")
    for case in cases:
        values = predictions.loc[panel.split.eq("test") if case.get("test_only") else panel.index, case["id"]]
        if not np.isfinite(values).all():
            raise ValueError("实际推断评分缺失")
    entries = read(folder, "fits.json")
    if len(entries) > plan["budget"]["max_new_fits"] or any(e["status"] != "completed" for e in entries):
        raise ValueError("新拟合预算或完成状态无效")
    models, medians = {}, {}
    for entry in entries:
        names = entry["features"]
        if not set(names) <= set(FEATURES) | {"log_market_cap"}:
            raise ValueError("未知输入列进入模型")
        train = panel.trade_date.between(spec.train_start, entry["cutoff"]) & labels.label_observed & labels.label_end.le(entry["cutoff"])
        if int(train.sum()) != entry["rows"] or labels.loc[train, "label_end"].max() != entry["max_label_end"] or entry["max_label_end"] > entry["cutoff"]:
            raise ValueError("拟合成熟边界不一致")
        path = folder / entry["model_file"]
        if digest(path) != entry["model_sha256"]:
            raise ValueError("模型版本不同")
        model = joblib.load(path); models[entry["id"]] = model
        cache = (entry["cutoff"], tuple(names))
        if cache not in medians:
            medians[cache] = panel.loc[train, names].median().to_numpy()
        np.testing.assert_allclose(model.steps[0][1].statistics_, medians[cache], rtol=1e-12, atol=1e-12)
    assignments = read(folder, "assignments.json")
    entry_map = {e["id"]:e for e in entries}
    prediction_error = 0.
    for a in assignments:
        p = panel[panel.trade_date.between(a["first_decision"], a["last_decision"])]
        p = p.sample(min(120, len(p)), random_state=spec.seed)
        if "source_run_id" in a:
            path = root / "ml_runs" / a["source_run_id"] / a["model_file"]
            if digest(path) != a["model_sha256"]:
                raise ValueError("复用模型版本不同")
            model, names = joblib.load(path), a["features"]
        else:
            e = entry_map[a["fit_id"]]; model, names = models[a["fit_id"]], e["features"]
            if a["case"].endswith(("_rolling", "_tuned")) and e["cutoff"] >= a["first_decision"]:
                # Tuned assignments include in-sample train/valid predictions, unused in tests.
                if a["case"].endswith("_rolling"):
                    raise ValueError("年度重训使用未来标签")
                p = panel[panel.split.eq("test")].sample(120, random_state=spec.seed)
        with threadpool_limits(limits=4):
            expected = model.predict(p[names])
        error = float(np.max(np.abs(expected-predictions.loc[p.index,a["case"]].to_numpy())))
        prediction_error = max(prediction_error,error)
        if error > 1e-10:
            raise ValueError("保存模型无法复现该段预测")
    calendar = read(snapshot, "calendar.json")
    schedule = decision_dates(calendar, spec.train_start, spec.test_end, spec.frequency)
    days = sorted(schedule); exits = {d:schedule[days[i+1]] for i,d in enumerate(days[:-1])}
    if not labels.label_entry.equals(panel.trade_date.map(schedule)) or not labels.label_end.equals(panel.trade_date.map(exits)):
        raise ValueError("开盘目标日历错误")
    sample = labels[labels.label_observed].sample(150, random_state=spec.seed)
    raw = pd.read_parquet(snapshot / "bars.parquet", columns=["stock_code", "trade_date", "open", "close", "pre_close"],
        filters=[("stock_code", "in", sample.stock_code.unique().tolist())]).set_index(["stock_code", "trade_date"])
    positions = {d:i for i,d in enumerate(calendar)}
    label_error = 0.
    for row in sample.itertuples():
        a,b=positions[row.label_entry],positions[row.label_end]
        part=raw.reindex(pd.MultiIndex.from_tuples([(row.stock_code,d) for d in calendar[a:b+1]]))
        factors=part.close.to_numpy()[:-1]/part.pre_close.to_numpy()[:-1]
        expected=float(np.prod(factors)*part.pre_close.iloc[0]/part.open.iloc[0]*part.open.iloc[-1]/part.pre_close.iloc[-1]-1)
        label_error=max(label_error,abs(expected-row.label_return))
    if label_error > 1e-10:
        raise ValueError("开盘衔接标签独立复算不一致")
    tuning = read(folder, "tuning.json")
    if tuning["test_used_for_selection"]:
        raise ValueError("测试参与参数选择")
    for a in tuning["attempts"]:
        e=entry_map[a["fit_id"]]
        if e["cutoff"] != spec.train_end:
            raise ValueError("调参拟合使用验证或测试期")
        with threadpool_limits(limits=4):
            score=models[e["id"]].predict(panel[e["features"]])
        np.testing.assert_allclose(select_validation(panel,labels,score,spec),a["valid_mean_rank_ic"],rtol=0,atol=1e-12)
    for kind in ["ridge","tree"]:
        attempts=[a for a in tuning["attempts"] if a["kind"]==kind]
        winner=max(attempts,key=lambda a:(a["valid_mean_rank_ic"],-a["order"]))
        if [a["order"] for a in attempts if a.get("selected")] != [winner["order"]]:
            raise ValueError("验证期选择不符合固定规则")
    recomputed, summaries=signal_metrics(panel,labels,predictions,cases,spec)
    observed=pd.read_parquet(folder / "date_metrics.parquet")
    pd.testing.assert_frame_equal(recomputed,observed,check_exact=False,atol=1e-12,rtol=1e-12)
    test=panel[panel.split.eq("test")]
    account_schedule=decision_dates(calendar,spec.test_start,spec.test_end,spec.frequency)
    max_account_error=0.; fill_count=0
    for link in links["accounts"]:
        account=root / "runs" / link["run_id"]
        actual=pd.read_parquet(account / "decisions.parquet")
        expected=score_decisions(test,predictions.loc[test.index,link["case"]],account_schedule,spec.top_k)
        pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected.reset_index(drop=True))
        if read(account,"schedule.json") != account_schedule:
            raise ValueError("真实账户调仓日历不同")
        audited=read(folder,"account_audit_"+link["case"]+".json")
        if audited["run_id"] != link["run_id"] or set(audited["scenarios"]) != set(plan["budget"]["cost_scenarios"]):
            raise ValueError("账户审计矩阵不完整")
        for v in audited["scenarios"].values():
            max_account_error=max(max_account_error,max(v["daily_errors"].values())); fill_count+=v["fills"]
        result=read(account,"result.json")
        if any(p["metrics"]["max_accounting_error"] >= .001 for p in result["portfolios"]):
            raise ValueError("账户自身恒等式不成立")
    return dict(kind="ml",study="complete",run_id=folder.name,verified=True,rows=len(panel),new_fits=len(entries),
        accounts=len(links["accounts"]),account_scenarios=len(links["accounts"])*2,checked_fills=fill_count,
        prediction_sample_max_error=prediction_error,label_sample_max_error=label_error,max_daily_account_error=max_account_error,
        verification_scope="冻结代码/输入/原模型及全部子账户完整性；成熟标签、年度截止、训练填补、保存模型预测/标签抽样、验证选择/指标/选股复算；全部账户现金股数开盘价费用已有数值审计；不是独立研究复核")
