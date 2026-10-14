"""Explicit frozen inputs for the existing native account; never executes by itself."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .audit import _require, audit_score_portfolio
from .learning import LearningResult
from .fixed_rules import FixedRuleResult
from .portfolio import ScorePortfolio, build_score_portfolio
from .registry import implementation_hashes
from ..technical.artifacts import content_id, digest, verify_artifacts
from ..technical.contracts import ResearchSpec, Strategy, describe
from ..technical.market import Market
from ..technical.signals import features

SNAPSHOT_FILES = {"calendar.json", "bars.parquet", "actions.parquet", "metadata.parquet",
                  "status.parquet", "benchmark.parquet"}
EXTRAS = {"alpha_predictions.parquet", "alpha_references.parquet", "alpha_learning_receipt.json",
          "alpha_portfolio_receipt.json", "alpha_account_receipt.json", "alpha_scoring_code.json"}
APP = Path(__file__).resolve().parents[2]


def _snapshot(snapshot, calendar, cfg):
    snapshot = Path(snapshot).resolve()
    sm = json.loads((snapshot/"manifest.json").read_text(encoding="utf-8"))
    _require(snapshot.parent.name=="snapshots" and re.fullmatch(r"[a-f0-9]{24}", snapshot.name)
             and sm["snapshot_id"]==snapshot.name, "Native snapshot location/identity differs")
    _require(SNAPSHOT_FILES.issubset(sm["artifacts"]), "Snapshot missing native evidence files")
    for name, sha in sm["artifacts"].items():
        _require(Path(name).name==name and "/" not in name and "\\" not in name
                 and isinstance(sha, str) and re.fullmatch(r"[a-f0-9]{64}", sha),
                 "Unsafe snapshot artifact path/hash")
        _require(not (snapshot/name).is_symlink(), "Snapshot artifacts must be ordinary files")
    verify_artifacts(snapshot, sm)
    _require(json.loads((snapshot/"calendar.json").read_text(encoding="utf-8"))==calendar,
             "Snapshot calendar differs from scored calendar")
    _require(sm["start"]<=cfg["start"]<=cfg["end"]<=sm["end"], "Account exceeds snapshot coverage")
    tables = {key:pd.read_parquet(snapshot/(key+".parquet"))
              for key in ("bars", "actions", "metadata", "status", "benchmark")}
    for key, table in tables.items():
        _require(sm["counts"][key]==len(table), "Snapshot row count differs: "+key)
    return snapshot, sm, tables


def _account_spec(spec, cfg):
    _require(isinstance(spec, ResearchSpec), "Native ResearchSpec required")
    spec = ResearchSpec.from_dict(spec.to_dict())
    _require(spec.strategy == replace(Strategy(), name=cfg["name"], top_k=cfg["top_k"],
                                      allocation=cfg["allocation"], rebalance=cfg["rebalance"]),
             "Native strategy must be the declared score-account placeholder")
    _require(spec.experiment.start_date==cfg["start"] and spec.experiment.end_date==cfg["end"],
             "Native account interval differs from score portfolio")
    return spec


def score_account_definition(spec, receipt, *, holding_policy=None):
    value = describe(spec)
    value.update(name=receipt["spec"]["name"], signal_interface="frozen_external_decisions",
        momentum="使用alpha_predictions.parquet的模型分数；原生动量参数不参与评分。",
        filters="调用方事先提供的历史可知eligible名单；保留全部评分行和排除理由，详情见模型与组合收据。",
        universe="已声明评分名单中的主板合格子集；没有隐式小市值筛选。",
        ranking="模型分数降序、代码升序处理并列；独立重建top_k和目标权重。",
        availability="逐行决策/执行时间见学习输出；参考原始收盘价与资格须在该决策时点已知。",
        parameter_origin="模型/特征/目标/股票范围为显式协议；原生Strategy只承载周月频率、top_k和配置比例。",
        source="alpha_learning_receipt.json及alpha_portfolio_receipt.json",
        forecast_horizon_policy=receipt["spec"]["horizon_policy"],
        target_mismatch_eligible_rows=receipt["target_mismatch_eligible_rows"],
        model_reproduction_scope="保留模型与训练身份声明；本账户核查不重新训练或重放模型数值。",
        endpoint=receipt["terminal_policy"])
    if receipt.get('source_kind')=='fixed_rule':
        value.update(momentum='使用固定规则分数；没有拟合模型，原生动量参数不参与评分。',
            ranking='固定规则分数降序、代码升序处理并列；独立重建top_k和目标权重。',
            filters='事先声明的历史可知eligible名单；保留全部评分行和排除理由，详情见规则与组合收据。',
            availability='逐行决策和执行时刻见规则输入；数值和资格须在决策时点已知。',
            parameter_origin='固定规则、原输入和股票范围已冻结；没有训练或参数优化。',
            model_reproduction_scope='没有拟合模型；核对规则输入、实际规则计算和账户。')
    if holding_policy is not None:
        value.update(holding_policy=holding_policy.to_dict(),
            ranking='逐费用按信号日真实持仓保留Rank阈值内股票；新买从入场排名范围选取，代码升序处理并列。',
            frozen_topk_role='decisions.parquet保留静态评分诊断；实际目标由各费用holding_choices和targets提供。',
            actual_state_policy='仅使用执行日前一个已完成信号日收盘状态；退出意向不提前释放真实持仓槽。')
    return value


def _code_bundle(result, portfolio, *, holding_policy=None):
    required = dict(result.receipt["implementation_hashes"] if type(result) is FixedRuleResult else result.receipt["fit_identity"]["implementation_hashes"])
    for name, sha in portfolio.receipt["implementation_hashes"].items():
        _require(name not in required or required[name]==sha, "Learning/portfolio code versions differ")
        required[name] = sha
    current = dict(implementation_hashes(list(required)+["src/alpharesearch/account.py", "src/alpharesearch/audit.py"]+(["src/alpharesearch/holding_policy.py","src/alpharesearch/holding_policy_audit.py"] if holding_policy is not None else [])))
    _require(all(current[name]==sha for name, sha in required.items()), "Declared scoring code differs from installed implementation")
    files = {name:dict(sha256=sha, text=(APP/name).read_bytes().replace(b"\r\n",b"\n").decode("utf-8"))
             for name, sha in current.items()}
    return dict(schema="score-account-code-bundle-v1", files=files)


def _reference_prices(portfolio, bars):
    eligible = portfolio.decisions.reason.eq("eligible_external_score")
    refs = portfolio.decisions.loc[eligible, ["stock_code", "trade_date", "close"]]
    raw = bars[["stock_code", "trade_date", "close"]]
    _require(not raw.duplicated(["stock_code", "trade_date"]).any(), "Duplicate snapshot quotes")
    combined = refs.merge(raw, on=["stock_code", "trade_date"], how="left", validate="one_to_one", suffixes=("_reference","_raw"))
    a, b = combined.close_reference.to_numpy(dtype=float), combined.close_raw.to_numpy(dtype=float)
    _require(np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,rtol=0,atol=1e-10),
             "Raw scored reference close differs from snapshot")


def prepare_score_account(result, references, calendar, policy, snapshot, spec, *, reuse_identity=None, holding_policy=None):
    """Return native runner prepared inputs, with copied evidence and clear definition."""
    cfg = policy.to_dict()
    spec = _account_spec(spec, cfg)
    if holding_policy is not None:
        from .holding_policy import ActualHoldingPolicy
        _require(type(holding_policy) is ActualHoldingPolicy and (reuse_identity is not None or type(result) is FixedRuleResult)
                 and holding_policy.max_positions==cfg['top_k'] and holding_policy.allocation==cfg['allocation'],
                 'Dynamic v3 requires explicit policy, completed reuse identity and matching portfolio limits')
    snapshot, sm, tables = _snapshot(snapshot, calendar, cfg)
    portfolio = build_score_portfolio(result, references, calendar, policy)
    selection = audit_score_portfolio(result, references, calendar, portfolio)
    _reference_prices(portfolio, tables["bars"])
    # Reuse native 20-session volume construction for capacity only, never ML X.
    bars = features(tables["bars"], calendar, spec.strategy)
    market = Market(bars, calendar.copy(), tables["actions"], tables["status"],
                    tables["metadata"], sm["missing_action_files"])
    if reuse_identity is None:
        bundle = _code_bundle(result, portfolio,holding_policy=holding_policy)
    else:
        from .score_reuse import reuse_code_bundle
        bundle = reuse_code_bundle(result, portfolio, reuse_identity, holding_policy=holding_policy)
    definition = score_account_definition(spec, portfolio.receipt, holding_policy=holding_policy)
    receipt = dict(schema="score-account-input-receipt-v1",
        snapshot_id=sm["snapshot_id"], snapshot_manifest_sha256=digest(snapshot/"manifest.json"),
        portfolio_receipt_id=content_id(portfolio.receipt), learning_receipt_id=content_id(result.receipt),
        calendar_id=content_id(calendar), account_spec_id=content_id(spec.to_dict()),
        code_bundle_id=content_id(bundle), signal_definition_id=content_id(definition),
        independent_selection=selection, account_results=False, numeric_model_reproduction=False,
        scope="prepared inputs only; caller's registered executor must run and audit the account")
    if reuse_identity is not None:
        receipt.update(schema="score-account-input-receipt-v2",
                       verification_mode="reuse_completed_identity_v1",reuse_identity_id=reuse_identity["identity_id"])
    extras = {"alpha_predictions.parquet":result.predictions.copy(),
              "alpha_references.parquet":references.copy(),
              "alpha_learning_receipt.json":json.loads(json.dumps(result.receipt)),
              "alpha_portfolio_receipt.json":json.loads(json.dumps(portfolio.receipt)),
              "alpha_account_receipt.json":receipt, "alpha_scoring_code.json":bundle}
    if type(result) is FixedRuleResult:
        extras['alpha_rule_inputs.parquet']=result.rule_inputs.copy()
        receipt.update(score_source_kind='fixed_rule',source_rule_id=result.receipt['rule_id'],model_fits=0)
    if holding_policy is not None:
        extras['alpha_holding_policy.json']=holding_policy.to_dict()
        receipt.update(schema='score-account-input-receipt-v4' if type(result) is FixedRuleResult else 'score-account-input-receipt-v3',holding_policy_id=content_id(holding_policy.to_dict()),
                       holding_evidence_schema='signal-close-actual-state-v1/actual-holding-selection-v1')
    return dict(holding_policy=holding_policy,snapshot=snapshot, manifest=sm, decisions=portfolio.decisions.copy(),
                schedule=portfolio.schedule.copy(), market=market, signal_definition=definition, extras=extras)


def read_score_account_archive(folder, manifest, snapshot, sm):
    """Validate archived inputs against the native spec and verified snapshot."""
    _require(EXTRAS.issubset(manifest["artifacts"]), "Native archive missing scored input evidence")
    def read(name):
        return json.loads((folder/name).read_text(encoding="utf-8"))
    source=read('alpha_learning_receipt.json')
    if source.get('schema')=='fixed-rule-score-receipt-v1':
        _require('alpha_rule_inputs.parquet' in manifest['artifacts'],'Fixed-rule input evidence missing')
        learning=FixedRuleResult(pd.read_parquet(folder/'alpha_predictions.parquet'),source,pd.read_parquet(folder/'alpha_rule_inputs.parquet'))
    else:learning=LearningResult(pd.read_parquet(folder/'alpha_predictions.parquet'),source)
    portfolio = ScorePortfolio(pd.read_parquet(folder/"decisions.parquet"), read("schedule.json"),
                               read("alpha_portfolio_receipt.json"))
    references = pd.read_parquet(folder/"alpha_references.parquet")
    calendar = json.loads((snapshot/"calendar.json").read_text(encoding="utf-8"))
    cfg = portfolio.receipt["spec"];spec = _account_spec(ResearchSpec.from_dict(manifest["spec"]), cfg)
    _, _, tables = _snapshot(snapshot, calendar, cfg)
    _reference_prices(portfolio, tables["bars"])
    receipt, bundle = read("alpha_account_receipt.json"), read("alpha_scoring_code.json")
    holding_policy=None
    if receipt.get('schema') in {'score-account-input-receipt-v3','score-account-input-receipt-v4'}:
        from .holding_policy import ActualHoldingPolicy
        _require('alpha_holding_policy.json' in manifest['artifacts'], 'Missing dynamic holding policy artifact')
        holding_policy=ActualHoldingPolicy.from_dict(read('alpha_holding_policy.json'))
        _require(holding_policy.max_positions==cfg['top_k'] and holding_policy.allocation==cfg['allocation'],
                 'Archived holding policy differs from portfolio limits')
    else:
        _require('alpha_holding_policy.json' not in manifest['artifacts'] and not (folder/'alpha_holding_policy.json').exists()
                 and not any(k in receipt for k in ('holding_policy_id','holding_evidence_schema')),
                 'Legacy static account contains hidden dynamic policy')
    definition = score_account_definition(spec, portfolio.receipt, holding_policy=holding_policy)
    _require(read("definition.json")==definition, "Archived signal definition differs")
    result = read("result.json")
    native_code = {q.relative_to(folder/"code").as_posix():digest(q)
                   for q in (folder/"code").rglob("*") if q.is_file()}
    _require(manifest["status"]==result["status"]=="completed"
             and manifest["run_id"]==result["run_id"]==folder.name
             and manifest["code_hash"]==result["code_hash"]==content_id(native_code),
             "Native completed identity/code inventory differs")
    _require(result["definition"]==definition, "Result signal definition differs")
    _require(receipt["schema"] in {"score-account-input-receipt-v1","score-account-input-receipt-v2","score-account-input-receipt-v3","score-account-input-receipt-v4"}
             and receipt["snapshot_id"]==sm["snapshot_id"]
             and receipt["snapshot_manifest_sha256"]==digest(snapshot/"manifest.json")
             and receipt["portfolio_receipt_id"]==content_id(portfolio.receipt)
             and receipt["learning_receipt_id"]==content_id(learning.receipt)
             and receipt["account_spec_id"]==content_id(spec.to_dict())
             and receipt["calendar_id"]==content_id(calendar)
             and receipt["code_bundle_id"]==content_id(bundle)
             and receipt["signal_definition_id"]==content_id(definition), "Archived score-account input identity differs")
    if receipt['schema'] in {'score-account-input-receipt-v2','score-account-input-receipt-v3'}:
        dynamic=receipt['schema']=='score-account-input-receipt-v3'
        _require(bundle.get('schema')==('score-account-code-bundle-v3' if dynamic else 'score-account-code-bundle-v2'),
                 'Receipt/bundle dynamic version differs')
        if dynamic:
            _require(receipt.get('holding_policy_id')==content_id(holding_policy.to_dict())
                     and receipt.get('holding_evidence_schema')=='signal-close-actual-state-v1/actual-holding-selection-v1'
                     and bundle.get('holding_policy')==holding_policy.to_dict(), 'Archived dynamic policy identity differs')
        from .score_reuse import validate_reuse_bundle
        files=validate_reuse_bundle(bundle,learning,portfolio)
        _require(receipt.get("verification_mode")=="reuse_completed_identity_v1"
                 and receipt.get("reuse_identity_id")==bundle["producer_identity"]["identity_id"],
                 "Archived score reuse receipt differs")
        required={n:v["sha256"] for n,v in files.items()}
    else:
        _require(bundle["schema"]=="score-account-code-bundle-v1", "Scoring code bundle schema differs")
        for name, value in bundle["files"].items():
            _require(hashlib.sha256(value["text"].encode("utf-8")).hexdigest()==value["sha256"],
                     "Archived scoring source changed")
        required = dict(learning.receipt["implementation_hashes"] if type(learning) is FixedRuleResult else learning.receipt["fit_identity"]["implementation_hashes"])
        if holding_policy is not None:
            _require(type(learning) is FixedRuleResult and receipt['schema']=='score-account-input-receipt-v4','Unfitted dynamic source required')
            _require(receipt.get('holding_policy_id')==content_id(holding_policy.to_dict()) and receipt.get('holding_evidence_schema')=='signal-close-actual-state-v1/actual-holding-selection-v1','Fixed-rule dynamic policy identity differs')
            required.update(implementation_hashes(['src/alpharesearch/holding_policy.py','src/alpharesearch/holding_policy_audit.py']))
        for name, sha in portfolio.receipt["implementation_hashes"].items():
            _require(name not in required or required[name]==sha, "Archived learning/portfolio code versions differ")
            required[name] = sha
        _require(all(bundle["files"].get(name, {}).get("sha256")==sha for name, sha in required.items()),
                 "Archived source differs from declared learning/portfolio implementation")
    # Independently compare frozen native files to overlapping source identities.
    for name, sha in required.items():
        if name.startswith("src/technical/"):
            native = folder/"code"/name.removeprefix("src/technical/")
            _require(native.is_file() and hashlib.sha256(native.read_bytes().replace(b"\r\n",b"\n")).hexdigest()==sha,
                     "Native frozen engine differs from score source code")
    expected = audit_score_portfolio(learning, references, calendar, portfolio)
    _require(receipt["independent_selection"]==expected, "Archived selection audit differs")
    return dict(holding_policy=None if holding_policy is None else holding_policy.to_dict(),learning=learning, portfolio=portfolio, references=references, calendar=calendar, **tables)
