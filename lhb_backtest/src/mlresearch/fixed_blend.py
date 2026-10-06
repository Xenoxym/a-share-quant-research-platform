"""Registered fixed score combinations; no fit or new model inference."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import uuid
import numpy as np
import pandas as pd
from ..technical.artifacts import digest, verify_artifacts, write_json
from ..technical.contracts import ResearchSpec, Strategy, Execution, Experiment
from ..technical.market import Market
from ..technical.runner import run as run_account
from ..technical.signals import decision_dates
from .contracts import MLSpec
from .study_data import score_decisions
from .study_evaluation import account_statistics

CONTROLS = ['small_cap', 'fixed_random', 'tree_open', 'tree_tuned']

def read(folder, name):
    return json.loads((folder / name).read_text(encoding='utf8'))

def blend_plan(spec):
    cases = [
        dict(id='size75_open25',name='75%规模+25%原开盘树',weights={'size':.75,'tree_open':.25}),
        dict(id='size50_open50',name='50%规模+50%原开盘树',weights={'size':.5,'tree_open':.5}),
        dict(id='size25_open75',name='25%规模+75%原开盘树',weights={'size':.25,'tree_open':.75}),
        dict(id='size50_tuned50',name='50%规模+50%验证选择树',weights={'size':.5,'tree_tuned':.5}),
        dict(id='small500_open',name='最小500池内原开盘树',pool_size=500),
        dict(id='small1000_open',name='最小1000池内原开盘树',pool_size=1000),
        dict(id='size50_lowvol20',name='50%规模+50%低20日波动',weights={'size':.5,'lowvol20':.5}),
        dict(id='open75_lowvol60',name='75%原开盘树+25%低60日波动',weights={'tree_open':.75,'lowvol60':.25}),
    ]
    return dict(version=1,study='fixed_blend',cases=cases,controls=CONTROLS,source_run_id=spec.source_run_id,
        rank='当日全部共同候选平均百分位秩；规模=-log_market_cap，低波动=-volatility',
        order='评分降序、代码升序；分池先市值升序代码升序，池外NaN不入选',
        allocation=.98,top_k=100,execution=Execution().__dict__,
        budget=dict(max_new_fits=0,max_new_model_predictions=0,max_accounts=8,actual_planned_accounts=8,max_execution_attempts=1,
            cost_scenarios=['zero_transaction_cost','configured'],timeout_seconds=7200),
        gate=dict(cagr_gain=.02,max_mdd_worsening=.02,max_cagr_loss=.02,min_mdd_improvement=.05,year_return_floor=-.02,min_years=2),
        missing_outcomes='推断和交易不使用未来标签；保留最后不成熟周',data_scope='retrospective; same exact original candidate keys')

def source_identity(source, root):
    """Freeze existing registered source receipt and all four saved audit identities."""
    result=read(source,'result.json'); eid=result['research_context']['experiment_id']
    job=root/'worker_jobs'/eid
    receipt=read(job,'receipt.json'); audit=read(job,'ml_audit.json')
    if receipt['status']!='completed' or receipt['run_id']!=source.name or receipt['audit_hash']!=digest(job/'ml_audit.json') or not audit['verified']:
        raise ValueError('原完整研究收据或核查身份无效')
    controls=[]
    for a in read(source,'account_index.json'):
        if a['case'] not in CONTROLS: continue
        account=root/'runs'/a['run_id']
        if digest(account/'manifest.json')!=a['manifest_sha256']: raise ValueError('原控制清单变化')
        verify_artifacts(account,read(account,'manifest.json'))
        saved=read(source,'account_audit_'+a['case']+'.json')
        if saved['run_id']!=account.name or set(saved['scenarios'])!={'configured','zero_transaction_cost'}:
            raise ValueError('原控制审计矩阵不完整')
        controls.append(dict(case=a['case'],run_id=a['run_id'],manifest_sha256=a['manifest_sha256'],audit_sha256=digest(source/('account_audit_'+a['case']+'.json'))))
    if len(controls)!=4: raise ValueError('缺少控制账户')
    return dict(source_experiment=eid,receipt_sha256=digest(job/'receipt.json'),audit_sha256=digest(job/'ml_audit.json'),controls=controls)

def scores(panel, pred, plan):
    keys=['stock_code','trade_date','split']
    if not panel[keys].equals(pred[keys]) or panel.duplicated(keys[:2]).any(): raise ValueError('源主键不一致')
    utilities=dict(size=-panel.log_market_cap,tree_open=pred.tree_open,tree_tuned=pred.tree_tuned,
        lowvol20=-panel.volatility_20,lowvol60=-panel.volatility_60)
    if any(not np.isfinite(s).all() for s in utilities.values()): raise ValueError('评分输入缺失，不能静默改名单')
    ranks={k:v.groupby(panel.trade_date).rank(method='average',pct=True) for k,v in utilities.items()}
    out=panel[keys].copy()
    ordered=panel.sort_values(['trade_date','log_market_cap','stock_code'])
    size_position=(ordered.groupby('trade_date').cumcount()+1).reindex(panel.index)
    for case in plan['cases']:
        if 'pool_size' in case:
            out[case['id']]=pred.tree_open.where(size_position.le(case['pool_size']))
        else:
            out[case['id']]=sum(ranks[k]*w for k,w in case['weights'].items())
    return out

def decisions(panel, score, schedule):
    valid=score.notna()
    return score_decisions(panel.loc[valid],score.loc[valid],schedule,100,.98)

def run_blend(cfg, progress=lambda m:None):
    started=time.perf_counter();root=Path(cfg['root']);spec=MLSpec.from_dict(cfg['spec']);plan=blend_plan(spec)
    if plan!=cfg['study_plan']:raise ValueError('冻结八例矩阵不同')
    env=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in cfg['environment']['packages']})
    if env!=cfg['environment']:raise ValueError('环境变化')
    snapshot=root/'snapshots'/cfg['snapshot_id'];source=root/'ml_runs'/spec.source_run_id
    sm=read(snapshot,'manifest.json');srcm=read(source,'manifest.json')
    if digest(snapshot/'manifest.json')!=cfg['snapshot_manifest_hash'] or digest(source/'manifest.json')!=cfg['source_manifest_hash']:
        raise ValueError('源身份变化')
    verify_artifacts(snapshot,sm);verify_artifacts(source,srcm)
    if source_identity(source,root)!=cfg['source_identity']:raise ValueError('源登记审计身份变化')
    rid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    folder=root/'ml_runs'/rid;folder.mkdir(parents=True)
    write_json(folder/'status.json',dict(kind='ml',study='fixed_blend',run_id=rid,status='running'))
    try:
        write_json(folder/'spec.json',spec.to_dict());write_json(folder/'study_plan.json',plan);write_json(folder/'environment.json',env)
        raw_panel=pd.read_parquet(source/'panel.parquet');raw_pred=pd.read_parquet(source/'predictions.parquet')
        keys=['stock_code','trade_date','split']
        if not raw_panel[keys].equals(raw_pred[keys]):raise ValueError('原完整输入主键变化')
        mask=raw_panel.split.eq('test')
        panel=raw_panel.loc[mask].reset_index(drop=True);pred=raw_pred.loc[mask].reset_index(drop=True)
        calendar=read(snapshot,'calendar.json');schedule=decision_dates(calendar,spec.test_start,spec.test_end,spec.frequency)
        if len(schedule)!=140 or set(schedule)!=set(panel.trade_date):raise ValueError('140周共同名单不一致')
        if not panel.q_publication_date.lt(panel.trade_date).all():raise ValueError('资格使用未来公告')
        predictions=scores(panel,pred,plan)
        predictions.to_parquet(folder/'predictions.parquet',index=False)
        write_json(folder/'source_identity.json',cfg['source_identity'])
        write_json(folder/'candidate_identity.json',dict(rows=len(panel),decision_weeks=len(schedule),exact_source_test_keys=True,
            source_panel_sha256=digest(source/'panel.parquet'),source_predictions_sha256=digest(source/'predictions.parquet'),new_fits=0,new_model_predictions=0))
        reused=[]
        for link in cfg['source_identity']['controls']:
            account=root/'runs'/link['run_id'];actual=pd.read_parquet(account/'decisions.parquet')
            expected=score_decisions(panel,pred[link['case']],schedule,100)
            pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected.reset_index(drop=True))
            r=read(account,'result.json')
            portfolios=[]
            for p in r['portfolios']:
                eq=pd.read_parquet(account/p['metrics']['scenario']/'equity.parquet')
                portfolios.append(dict(metrics=p['metrics'],**account_statistics(eq,1e6,spec.seed)))
            reused.append(dict(**link,name=next(a['name'] for a in read(source,'account_index.json') if a['case']==link['case']),
                portfolios=portfolios,reused=True,numerical_audit='original saved account audit; not rerun'))
        write_json(folder/'reused_accounts.json',reused)
        bars=pd.read_parquet(snapshot/'bars.parquet',filters=[('trade_date','<=',spec.test_end)]).sort_values(['stock_code','trade_date'])
        bars['avg_volume_20']=bars.groupby('stock_code',sort=False).volume.transform(lambda s:s.rolling(20).mean())
        quotes=bars[['stock_code','trade_date','open']].set_index(['stock_code','trade_date'])
        market=Market(bars,calendar,pd.read_parquet(snapshot/'actions.parquet'),pd.read_parquet(snapshot/'status.parquet'),pd.read_parquet(snapshot/'metadata.parquet'),sm['missing_action_files'])
        del bars,raw_panel,raw_pred
        from tools.verify_technical import audit as audit_account
        accounts=[]
        for i,case in enumerate(plan['cases']):
            progress(f"固定组合账户 {i+1}/8: {case['name']}")
            account_spec=ResearchSpec(strategy=Strategy(name=case['name'],lookback=60,skip=0,top_k=100,rebalance='weekly'),
                execution=Execution(),experiment=Experiment(spec.test_start,spec.test_end))
            dec=decisions(panel,predictions[case['id']],schedule)
            definition=dict(momentum='使用已登记源预测与当时规模/风险效用；无新增拟合或模型推断',ranking=plan['order'],universe='原完整研究当时共同名单逐键复用；不要求未来标签',
                model=case,signal_parent=dict(run_id=rid,experiment_id=cfg['experiment_id'],predictions_sha256=digest(folder/'predictions.parquet')),
                parameter_origin='PROTOCOL固定八种组合；不从评价期选权重',source='registered fixed_blend study')
            account=run_account(cfg['project'],account_spec,root=root,publish=False,
                prepared=dict(snapshot=snapshot,manifest=sm,decisions=dec,schedule=schedule,market=market,signal_definition=definition),
                scenarios=plan['budget']['cost_scenarios'],research_context=dict(role='screen',source='registered_ml_fixed_blend',task_id=cfg['task_id'],experiment_id=cfg['experiment_id'],parent_ml_run=rid,case=case['id']),progress=progress)
            audit=audit_account(account,quotes=quotes);write_json(folder/('account_audit_'+case['id']+'.json'),audit)
            r=read(account,'result.json');portfolios=[]
            for p in r['portfolios']:
                eq=pd.read_parquet(account/p['metrics']['scenario']/'equity.parquet')
                portfolios.append(dict(metrics=p['metrics'],**account_statistics(eq,1e6,spec.seed)))
            accounts.append(dict(case=case['id'],name=case['name'],run_id=account.name,manifest_sha256=digest(account/'manifest.json'),portfolios=portfolios))
            write_json(folder/'account_index.json',accounts)
        write_json(folder/'links.json',dict(source_run_id=source.name,source_manifest_sha256=cfg['source_manifest_hash'],
            accounts=[{k:a[k] for k in ['case','run_id','manifest_sha256']} for a in accounts]))
        result=dict(kind='ml',study='fixed_blend',run_id=rid,name=spec.name,snapshot_id=cfg['snapshot_id'],source_run_id=source.name,
            research_context=dict(task_id=cfg['task_id'],experiment_id=cfg['experiment_id']),study_plan=plan,accounts=accounts,reused_accounts=reused,
            summaries=[],counts=dict(test=dict(candidates=len(panel),decision_weeks=len(schedule))),new_fits=0,new_model_predictions=0,
            elapsed_seconds=time.perf_counter()-started,limitations=['所有历史已看；新账户是固定回顾性实验','待附加配对区间和双独立复核','原供应商修订、退市覆盖、季度股本及分红应收局限保留'])
        write_json(folder/'result.json',result)
        (folder/'REPORT.md').write_text('固定八组合已执行；全部账户指标见account_index.json，复用控制见reused_accounts.json。附加工作目录报告说明GATE和不确定性。本账户清单不可事后改写。',encoding='utf8')
        write_json(folder/'manifest.json',dict(run_id=rid,kind='ml',study='fixed_blend',spec=spec.to_dict(),snapshot_id=cfg['snapshot_id'],snapshot_manifest_sha256=cfg['snapshot_manifest_hash'],
            source_manifest_sha256=cfg['source_manifest_hash'],code_hash=cfg['engine_hash'],artifacts={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*') if p.is_file() and p.name!='status.json'}))
        write_json(folder/'status.json',dict(kind='ml',study='fixed_blend',run_id=rid,status='completed'))
        return folder
    except BaseException as exc:
        write_json(folder/'status.json',dict(kind='ml',study='fixed_blend',run_id=rid,status='failed',error=str(exc)));raise
