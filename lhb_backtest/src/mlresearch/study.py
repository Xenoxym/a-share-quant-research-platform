"""A finite, preregistered ML study sharing the existing account engine.

The matrix is frozen before execution. Validation selects parameters; test never
does. Every fit (including failures) and every account has an immutable receipt.
"""
from datetime import datetime, timezone
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
from ..technical.contracts import ResearchSpec, Strategy, Execution, Experiment, describe
from ..technical.market import Market
from ..technical.runner import run as run_account
from ..technical.signals import decision_dates
from .contracts import MLSpec, FEATURES, feature_names
from .dataset import build_panel
from .study_data import open_labels, fixed_scores, balanced_score, score_decisions
from .study_evaluation import signal_metrics, select_validation, exposure_diagnostics, account_statistics


GROUPS = {
    "returns_bias": [k for k in FEATURES if k.startswith(("return_", "bias_"))],
    "volatility": [k for k in FEATURES if k.startswith("volatility_")],
    "activity": ["amount_ratio_5_20", "volume_ratio_5_20", "amount_volatility_20"],
    "candle": ["range_1", "body_1", "close_location"],
}


def study_plan(spec):
    cases = [
        dict(id="ridge_legacy", name="原岭回归 · 五日目标", target="close_5", regression=True),
        dict(id="tree_legacy", name="原提升树 · 五日目标", target="close_5", regression=True),
        *[dict(id=k, name=n, target="weekly_open", regression=False) for k, n in [
            ("small_cap", "同池小市值"), ("reversal_5", "五日反转"), ("fixed_random", "固定随机11")]],
        dict(id="ridge_open", name="岭回归 · 周期开盘目标", target="weekly_open", regression=True),
        dict(id="tree_open", name="提升树 · 周期开盘目标", target="weekly_open", regression=True),
        dict(id="tree_size", name="提升树 · 价量加市值", target="weekly_open", regression=True),
        *[dict(id="tree_without_" + k, name="提升树 · 删除" + n, target="weekly_open", regression=True)
          for k, n in [("returns_bias", "收益及均线偏离"), ("volatility", "波动率"), ("activity", "量额"), ("candle", "当日K线形状")]],
        dict(id="tree_shuffled", name="提升树 · 日期内打乱Y", target="weekly_open", regression=True),
        dict(id="ridge_rolling", name="岭回归 · 逐年扩展重训", target="weekly_open", regression=True, test_only=True),
        dict(id="tree_rolling", name="提升树 · 逐年扩展重训", target="weekly_open", regression=True, test_only=True),
        dict(id="ridge_tuned", name="岭回归 · 验证期选参数", target="weekly_open", regression=True, test_only=True),
        dict(id="tree_tuned", name="提升树 · 验证期选参数", target="weekly_open", regression=True, test_only=True),
        dict(id="tree_size_balanced", name="提升树 · 每规模组各取10只", target="weekly_open", regression=False),
        dict(id="random_size_balanced", name="固定随机 · 每规模组各取10只", target="weekly_open", regression=False),
    ]
    return dict(version=1, cases=cases, feature_groups=GROUPS,
        tuning=dict(primary="2023验证日期平均RankIC；并列按固定候选顺序", ridge_alphas=[1., 10., 1000.],
            trees=[dict(leaves=3, iterations=80, l2=10.), dict(leaves=7, iterations=80, l2=10.), dict(leaves=7, iterations=120, l2=100.)]),
        rolling="每测试年1月前，用2020起到前一年末成熟标签扩展重训；不在该年内再次拟合",
        budget=dict(max_new_fits=30, max_accounts=20, actual_planned_accounts=len(cases), cost_scenarios=["zero_transaction_cost", "configured"], timeout_seconds=7200),
        execution=Execution().__dict__, allocation=.98, top_k=spec.top_k,
        data_scope="same snapshot; all history previously seen; no test-based case/parameter selection",
        missing_outcomes="预测与交易不要求未来标签；训练仅使用截止日已成熟且可观测标签",
        endpoint="末日按持仓估值，不强制清仓；最后未成熟周信号仍交易，但不进入标签评价")


class Fits:
    def __init__(self, folder, panel, labels, spec, progress):
        self.folder, self.panel, self.labels, self.spec, self.progress = folder, panel, labels, spec, progress
        self.entries, self.cache, self.models, self.new_count = [], {}, {}, 0
        (folder / "models").mkdir()

    def save(self):
        write_json(self.folder / "fits.json", self.entries)

    def fit(self, kind, names, cutoff, params, *, shuffled=False):
        definition = dict(kind=kind, features=names, cutoff=cutoff, parameters=params, shuffled=shuffled, target="weekly_open")
        key = content_id(definition)[:16]
        if key in self.models:
            return key, self.models[key]
        if self.new_count >= 30:
            raise ValueError("达到登记的新拟合次数上限")
        self.new_count += 1
        train = self.panel.trade_date.between(self.spec.train_start, cutoff) & self.labels.label_observed & self.labels.label_end.le(cutoff)
        X = self.panel.loc[train, names]
        y = self.labels.loc[train, "label_relative"].copy()
        if len(X) < max(100, self.spec.tree_min_samples * 2) or X.notna().sum().eq(0).any():
            raise ValueError("拟合样本不足或全空输入")
        entry = dict(id=key, status="running", **definition, rows=int(train.sum()),
            max_decision_date=self.panel.loc[train, "trade_date"].max(), max_label_end=self.labels.loc[train, "label_end"].max())
        self.entries.append(entry); self.save()
        self.progress(f"拟合 {self.new_count}/30：{kind}，{len(names)}个X，标签截至{cutoff}，{len(X)}行")
        try:
            if shuffled:
                rng = np.random.default_rng(self.spec.seed)
                for _, ids in self.panel.loc[train].groupby("trade_date").groups.items():
                    y.loc[ids] = rng.permutation(y.loc[ids].to_numpy())
            if kind == "ridge":
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=params["alpha"]))
            else:
                model = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(
                    max_leaf_nodes=params["leaves"], max_iter=params["iterations"], l2_regularization=params["l2"],
                    min_samples_leaf=self.spec.tree_min_samples, learning_rate=self.spec.tree_learning_rate,
                    early_stopping=False, random_state=self.spec.seed))
            with threadpool_limits(limits=4):
                model.fit(X, y)
            path = self.folder / "models" / (key + ".joblib")
            joblib.dump(model, path)
            entry.update(status="completed", model_file=path.relative_to(self.folder).as_posix(), model_sha256=digest(path))
            self.models[key] = model
            self.save()
            return key, model
        except BaseException as exc:
            entry.update(status="failed", error=str(exc)); self.save()
            raise

    def predict(self, model, names, subset=None):
        X = self.panel[names] if subset is None else self.panel.loc[subset, names]
        with threadpool_limits(limits=4):
            return model.predict(X)


def run_study(cfg, progress=lambda m: None):
    start = time.perf_counter(); root = Path(cfg["root"])
    spec = MLSpec.from_dict(cfg["spec"]); plan = study_plan(spec)
    if plan != cfg["study_plan"]:
        raise ValueError("冻结研究矩阵与登记不同")
    environment = dict(python=platform.python_version(), packages={n: importlib.metadata.version(n) for n in cfg["environment"]["packages"]})
    if environment != cfg["environment"]:
        raise ValueError("执行环境与登记不同")
    snapshot = root / "snapshots" / cfg["snapshot_id"]
    source = root / "ml_runs" / spec.source_run_id
    sm = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if digest(snapshot / "manifest.json") != cfg["snapshot_manifest_hash"] or digest(source / "manifest.json") != cfg["source_manifest_hash"]:
        raise ValueError("源快照或原基准版本变化")
    verify_artifacts(snapshot, sm)
    verify_artifacts(source, json.loads((source / "manifest.json").read_text(encoding="utf-8")))
    rid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    folder = root / "ml_runs" / rid; folder.mkdir(parents=True)
    write_json(folder / "status.json", dict(kind="ml", study="complete", run_id=rid, status="running"))
    try:
        write_json(folder / "spec.json", spec.to_dict()); write_json(folder / "study_plan.json", plan)
        write_json(folder / "environment.json", environment)
        calendar = json.loads((snapshot / "calendar.json").read_text(encoding="utf-8"))
        bars = pd.read_parquet(snapshot / "bars.parquet", filters=[("trade_date", "<=", spec.test_end)])
        meta = pd.read_parquet(snapshot / "metadata.parquet")
        panel = build_panel(bars, calendar, meta, pd.read_parquet(snapshot / "fundamentals.parquet"), spec, progress, inference=True)
        panel.to_parquet(folder / "panel.parquet", index=False)
        progress("构建下一开盘到下一次周调仓开盘的衔接标签")
        labels = open_labels(panel, bars, calendar, spec)
        labels.to_parquet(folder / "open_labels.parquet", index=False)
        # Check reused X against the immutable original panel without refitting it.
        original = pd.read_parquet(source / "panel.parquet")
        joined = original.merge(panel, on=["stock_code", "trade_date"], suffixes=("_old", "_new"), validate="one_to_one")
        if len(joined) != len(original):
            raise ValueError("原基准候选不再完全匹配")
        for name in feature_names(spec):
            np.testing.assert_allclose(joined[name + "_old"], joined[name + "_new"], rtol=0, atol=1e-12, equal_nan=True)
        del original, joined
        fits = Fits(folder, panel, labels, spec, progress)
        predictions = panel[["stock_code", "trade_date", "split"]].copy()
        assignments = []
        names = list(FEATURES)
        default_ridge = dict(alpha=spec.ridge_alpha)
        default_tree = dict(leaves=spec.tree_leaves, iterations=spec.tree_iterations, l2=spec.tree_l2)

        def assign(case, kind, columns, cutoff, params, shuffled=False, subset=None):
            fit_id, model = fits.fit(kind, columns, cutoff, params, shuffled=shuffled)
            if case not in predictions:
                predictions[case] = np.nan
            if subset is None:
                predictions[case] = fits.predict(model, columns)
                first, last = panel.trade_date.min(), panel.trade_date.max()
            else:
                predictions.loc[subset, case] = fits.predict(model, columns, subset)
                first, last = panel.loc[subset, "trade_date"].min(), panel.loc[subset, "trade_date"].max()
            assignments.append(dict(case=case, fit_id=fit_id, first_decision=first, last_decision=last))
            return fit_id

        old_predictions = pd.read_parquet(source / "predictions.parquet")
        for case, old in [("ridge_legacy", "ridge"), ("tree_legacy", "hist_gbdt")]:
            model = joblib.load(source / (old + ".joblib"))
            predictions[case] = fits.predict(model, names)
            check = old_predictions[["stock_code", "trade_date", old]].merge(predictions[["stock_code", "trade_date", case]], validate="one_to_one")
            np.testing.assert_allclose(check[old], check[case], rtol=0, atol=1e-10)
            assignments.append(dict(case=case, source_run_id=spec.source_run_id, model_file=old + ".joblib", model_sha256=digest(source / (old + ".joblib")),
                features=names, first_decision=panel.trade_date.min(), last_decision=panel.trade_date.max(), cutoff=spec.train_end))
        del old_predictions
        controls = fixed_scores(panel, spec.seed)
        for k in controls:
            predictions[k] = controls[k]
        assign("ridge_open", "ridge", names, spec.train_end, default_ridge)
        assign("tree_open", "tree", names, spec.train_end, default_tree)
        assign("tree_size", "tree", names + ["log_market_cap"], spec.train_end, default_tree)
        for group, excluded in GROUPS.items():
            assign("tree_without_" + group, "tree", [n for n in names if n not in excluded], spec.train_end, default_tree)
        assign("tree_shuffled", "tree", names, spec.train_end, default_tree, shuffled=True)

        # No test predictions participate in this selection criterion.
        tuning = []
        for kind, parameters in [("ridge", [dict(alpha=a) for a in plan["tuning"]["ridge_alphas"]]), ("tree", plan["tuning"]["trees"])]:
            candidates = []
            for order, params in enumerate(parameters):
                fit_id, model = fits.fit(kind, names, spec.train_end, params)
                score = fits.predict(model, names)
                criterion = select_validation(panel, labels, score, spec)
                entry = dict(kind=kind, order=order, parameters=params, fit_id=fit_id, valid_mean_rank_ic=criterion)
                candidates.append(entry); tuning.append(entry)
            winner = max(candidates, key=lambda c: (c["valid_mean_rank_ic"], -c["order"]))
            winner["selected"] = True
            assign(kind + "_tuned", kind, names, spec.valid_end, winner["parameters"])
        write_json(folder / "tuning.json", dict(primary=plan["tuning"]["primary"], attempts=tuning, test_used_for_selection=False))
        for year in sorted(panel.loc[panel.split.eq("test"), "trade_date"].str[:4].unique()):
            subset = panel.split.eq("test") & panel.trade_date.str.startswith(year)
            cutoff = f"{int(year)-1}-12-31"
            for kind, params in [("ridge", default_ridge), ("tree", default_tree)]:
                assign(kind + "_rolling", kind, names, cutoff, params, subset=subset)
        predictions["tree_size_balanced"] = balanced_score(panel, predictions.tree_open)
        predictions["random_size_balanced"] = balanced_score(panel, predictions.fixed_random)
        write_json(folder / "assignments.json", assignments)
        predictions.to_parquet(folder / "predictions.parquet", index=False)
        cases = plan["cases"]
        daily, summaries = signal_metrics(panel, labels, predictions, cases, spec)
        daily.to_parquet(folder / "date_metrics.parquet", index=False)
        exposures, conditional = exposure_diagnostics(panel, labels, predictions, cases, spec)
        exposures.to_parquet(folder / "exposures.parquet", index=False)
        conditional.to_parquet(folder / "conditional_metrics.parquet", index=False)
        counts = [dict(split=k, rows=len(p), dates=p.trade_date.nunique(), observed_labels=int(labels.loc[p.index, "label_observed"].sum())) for k, p in panel.groupby("split")]
        write_json(folder / "dataset.json", dict(features=names, feature_definitions=FEATURES, feature_groups=GROUPS, counts=counts,
            source_run_id=spec.source_run_id, inference="所有当周合格候选；末端未来未知仍预测；与原已保存X及预测逐行核对",
            targets=dict(close_5="原五日收盘参考衔接相对收益", weekly_open="下一开盘到下一次周调仓开盘参考衔接收益减同日候选均值；不是账户净收益"),
            learning="平方误差；岭回归L2/提升树叶子正则；RankIC用于验证期候选选择；没有直接优化账户收益"))

        progress("所有模型拟合完成，开始固定19策略×两种费用情景的账户")
        bars = bars.sort_values(["stock_code", "trade_date"])
        bars["avg_volume_20"] = bars.groupby("stock_code", sort=False).volume.transform(lambda s: s.rolling(20).mean())
        audit_quotes = bars[["stock_code", "trade_date", "open"]].set_index(["stock_code", "trade_date"])
        market = Market(bars, calendar, pd.read_parquet(snapshot / "actions.parquet"), pd.read_parquet(snapshot / "status.parquet"), meta, sm["missing_action_files"])
        del bars
        test = panel[panel.split.eq("test")]
        schedule = decision_dates(calendar, spec.test_start, spec.test_end, spec.frequency)
        if set(schedule) != set(test.trade_date.unique()):
            raise ValueError("测试推断日期缺失，不能用静默空仓掩盖")
        accounts, equity_frames = [], {}
        from tools.verify_technical import audit as audit_account
        for i, case in enumerate(cases):
            progress(f"实际账户 {i+1}/{len(cases)}：{case['name']}")
            account_spec = ResearchSpec(strategy=Strategy(name=case["name"], lookback=60, skip=0, top_k=spec.top_k, rebalance=spec.frequency),
                execution=Execution(), experiment=Experiment(spec.test_start, spec.test_end))
            decision = score_decisions(test, predictions.loc[test.index, case["id"]], schedule, spec.top_k)
            definition = dict(momentum="不使用账户Strategy.lookback计算信号；使用冻结ML父实验中的外部评分", ranking="每周全部共同合格候选按该策略评分降序，代码升序打破并列，选前100只；规模组对照每十分组各取10只",
                universe="与原ML基准相同的主板资格：非ST、连续60日、20日均额2000万、有效已公告股本；不是先筛小市值", model=case,
                signal_parent=dict(run_id=rid, experiment_id=cfg["experiment_id"], predictions_sha256=digest(folder / "predictions.parquet")),
                parameter_origin="本次事前冻结的研究矩阵；账户中的lookback字段为接口占位，不参与选股", source="frozen ML study, not momentum index replication")
            account = run_account(cfg["project"], account_spec, root=root, publish=False,
                prepared=dict(snapshot=snapshot, manifest=sm, decisions=decision, schedule=schedule, market=market, signal_definition=definition),
                scenarios=plan["budget"]["cost_scenarios"], research_context=dict(role="screen", source="registered_ml_study", task_id=cfg["task_id"], experiment_id=cfg["experiment_id"], parent_ml_run=rid, case=case["id"]),
                progress=progress)
            audited = audit_account(account, quotes=audit_quotes)
            write_json(folder / ("account_audit_" + case["id"] + ".json"), audited)
            result = json.loads((account / "result.json").read_text(encoding="utf-8"))
            entry = dict(case=case["id"], name=case["name"], run_id=account.name, manifest_sha256=digest(account / "manifest.json"), portfolios=[])
            for p in result["portfolios"]:
                scenario = p["metrics"]["scenario"]
                eq = pd.read_parquet(account / scenario / "equity.parquet")
                equity_frames[(case["id"], scenario)] = eq
                entry["portfolios"].append(dict(metrics=p["metrics"], **account_statistics(eq, account_spec.execution.initial_cash, spec.seed)))
            accounts.append(entry)
            write_json(folder / "account_index.json", accounts)
        for account in accounts:
            for p in account["portfolios"]:
                scenario = p["metrics"]["scenario"]
                p.update(account_statistics(equity_frames[(account["case"], scenario)], 1_000_000., spec.seed, equity_frames[("small_cap", scenario)]))
        write_json(folder / "account_index.json", accounts)
        write_json(folder / "links.json", dict(source_run_id=spec.source_run_id, source_manifest_sha256=cfg["source_manifest_hash"],
            accounts=[dict(case=a["case"], run_id=a["run_id"], manifest_sha256=a["manifest_sha256"]) for a in accounts]))
        result = dict(kind="ml", study="complete", run_id=rid, name=spec.name, snapshot_id=cfg["snapshot_id"],
            research_context=dict(task_id=cfg["task_id"], experiment_id=cfg["experiment_id"]), counts=counts, summaries=summaries,
            quantiles=[], study_plan=plan, accounts=accounts, source_run_id=spec.source_run_id,
            new_fits=fits.new_count, reused_models=2, tuning=tuning,
            exposures=records(exposures.groupby("model").mean(numeric_only=True).reset_index()),
            conditional=records(conditional.groupby("model").mean(numeric_only=True).reset_index()),
            elapsed_seconds=time.perf_counter()-start,
            limitations=["所有历史已看；时间隔离和验证期选择不恢复真正未见历史", "多案例区间未校正所有研究尝试；不是确认性显著性或未来收益保证",
                "股本为公告报告代理，行业未中性化，缺失退市/修订及不明公司行动仍限制结论", "每规模组检验及残差相关是条件关联，不是因果Alpha",
                "开盘衔接Y不含账户费用/分红应收；涨停买不进、现金及整手约束会造成实际持仓与理想Top100不同",
                "有限验证期候选搜索与年度批量重训已实现；没有分钟/L2、实时推断服务或实盘执行"])
        write_json(folder / "result.json", result)
        from .study_report import render_report
        (folder / "REPORT.md").write_text(render_report(result), encoding="utf-8")
        manifest = dict(run_id=rid, kind="ml", study="complete", snapshot_id=cfg["snapshot_id"], snapshot_manifest_sha256=cfg["snapshot_manifest_hash"],
            source_manifest_sha256=cfg["source_manifest_hash"], spec=spec.to_dict(), code_hash=cfg["engine_hash"],
            artifacts={p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob("*") if p.is_file() and p.name != "status.json"})
        write_json(folder / "manifest.json", manifest)
        write_json(folder / "status.json", dict(kind="ml", study="complete", status="completed", run_id=rid, name=spec.name))
        return folder
    except BaseException as exc:
        write_json(folder / "status.json", dict(kind="ml", study="complete", status="failed", run_id=rid, error=str(exc)))
        raise
