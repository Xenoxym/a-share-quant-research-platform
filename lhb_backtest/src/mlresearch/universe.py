"""Finite stock-universe contrasts. Extra boards receive learning metrics only."""
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
from ..technical.signals import MAIN_BOARD, decision_dates
from .contracts import MLSpec, FEATURES
from .dataset import build_panel
from .study import Fits
from .study_data import open_labels, fixed_scores, score_decisions
from .study_evaluation import signal_metrics, account_statistics


POOLS = {
    'qualified': '原主板资格（仅新增每周重抽随机账户；原模型/账户复用）',
    'no_financial': '主板取消财报股本资格，保留非ST/60日/成交额门槛',
    'mainboard_available': '当时有价有量的非ST主板，不要求60日、成交额或股本',
    'local_A_available': '本地全部当时有价有量A股，包括ST与新股；没有北交所；仅学习检验',
}


def universe_plan(spec):
    cases=[dict(id='qualified__weekly_random',name='原池 · 每周重抽随机100',pool='qualified',target='weekly_open',regression=False,account=True)]
    for pool in ['no_financial','mainboard_available','local_A_available']:
        for kind in ['ridge','tree','fixed_random','weekly_random']:
            cases.append(dict(id=pool+'__'+kind,name=POOLS[pool]+' · '+kind,pool=pool,target='weekly_open',regression=kind in {'ridge','tree'},account=pool!='local_A_available'))
    return dict(version=1,cases=cases,pools=POOLS,features=list(FEATURES),target='next weekly execution open to following weekly execution open; relative to each pool mean',
        training='2020—2022截止日已成熟标签；所有池固定原参数；不搜索、不重训、不用验证结果选择',
        missing_X='以各池训练段中位数填补缺失；新股不足60日的长窗口为NaN，未增加缺失指示特征',
        controls='fixed_random=hash(seed:code)，weekly_random=hash(seed:date:code)，各周Top100；11是种子',
        budget=dict(max_new_fits=6,max_accounts=9,actual_planned_accounts=9,cost_scenarios=['zero_transaction_cost','configured'],timeout_seconds=7200),
        allocation=.98,top_k=spec.top_k,data_scope='all history previously seen; local A excludes BSE; no full-A account claim')


def pool_masks(panel,spec):
    main=panel.stock_code.str.match(MAIN_BOARD)
    nonst=panel.is_st.eq(0)
    liquid=panel.avg_amount_20.ge(spec.min_avg_amount)
    base=main & nonst & panel.history_valid & liquid
    return dict(qualified=base & panel.cap_valid,no_financial=base,mainboard_available=main & nonst,local_A_available=pd.Series(True,index=panel.index))


def weekly_random(panel,seed):
    import hashlib
    return pd.Series([int(hashlib.sha256(f'{seed}:{d}:{c}'.encode()).hexdigest()[:13],16)/16**13 for d,c in zip(panel.trade_date,panel.stock_code)],index=panel.index)


def run_universe(cfg,progress=lambda m:None):
    started=time.perf_counter();root=Path(cfg['root']);spec=MLSpec.from_dict(cfg['spec']);plan=universe_plan(spec)
    if plan!=cfg['study_plan']:raise ValueError('冻结股票范围设计不同')
    env=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in cfg['environment']['packages']})
    if env!=cfg['environment']:raise ValueError('环境与登记不同')
    snapshot=root/'snapshots'/cfg['snapshot_id'];source=root/'ml_runs'/spec.source_run_id
    sm=json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
    if digest(snapshot/'manifest.json')!=cfg['snapshot_manifest_hash'] or digest(source/'manifest.json')!=cfg['source_manifest_hash']:raise ValueError('源身份变化')
    verify_artifacts(snapshot,sm);verify_artifacts(source,json.loads((source/'manifest.json').read_text(encoding='utf-8')))
    rid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    folder=root/'ml_runs'/rid;folder.mkdir()
    write_json(folder/'status.json',dict(kind='ml',study='universe',run_id=rid,status='running'))
    try:
        write_json(folder/'spec.json',spec.to_dict());write_json(folder/'study_plan.json',plan);write_json(folder/'environment.json',env)
        calendar=json.loads((snapshot/'calendar.json').read_text(encoding='utf-8'))
        bars=pd.read_parquet(snapshot/'bars.parquet')
        available=build_panel(bars,calendar,pd.read_parquet(snapshot/'metadata.parquet'),pd.read_parquet(snapshot/'fundamentals.parquet'),spec,progress,inference=True,eligibility='available')
        available.to_parquet(folder/'available_panel.parquet',index=False)
        labels_all=open_labels(available,bars,calendar,spec)
        labels_all.to_parquet(folder/'available_labels.parquet',index=False)
        del bars
        masks=pool_masks(available,spec);old=pd.read_parquet(source/'panel.parquet')
        original=available.loc[masks['qualified']].reset_index(drop=True)
        keys=['stock_code','trade_date','split']
        if not original[keys].equals(old[keys]):raise ValueError('原主板候选身份改变')
        for name in FEATURES:np.testing.assert_allclose(original[name],old[name],rtol=0,atol=1e-12,equal_nan=True)
        del old,original
        accounts_to_run=[];summaries=[];metrics=[];fits_all=[];counts=[];pool_test={};cross_test={};all_predictions=[]
        for pool,mask in masks.items():
            progress('范围对照：'+POOLS[pool])
            sub=folder/'pools'/pool;sub.mkdir(parents=True)
            panel=available.loc[mask].reset_index(drop=True)
            labels=labels_all.loc[mask].reset_index(drop=True).copy()
            labels['label_relative']=labels.label_return-labels.groupby('trade_date').label_return.transform('mean')
            panel.to_parquet(sub/'panel.parquet',index=False);labels.to_parquet(sub/'open_labels.parquet',index=False)
            pred=panel[keys].copy()
            if pool!='qualified':
                fits=Fits(sub,panel,labels,spec,progress)
                for kind,params in [('ridge',dict(alpha=spec.ridge_alpha)),('tree',dict(leaves=spec.tree_leaves,iterations=spec.tree_iterations,l2=spec.tree_l2))]:
                    key,model=fits.fit(kind,list(FEATURES),spec.train_end,params)
                    pred[pool+'__'+kind]=fits.predict(model,list(FEATURES))
                fits_all += [dict(pool=pool,**e) for e in fits.entries]
                pred[pool+'__fixed_random']=fixed_scores(panel,spec.seed).fixed_random
            pred[pool+'__weekly_random']=weekly_random(panel,spec.seed)
            pred.to_parquet(sub/'predictions.parquet',index=False)
            cases=[c for c in plan['cases'] if c['pool']==pool]
            daily,summary=signal_metrics(panel,labels,pred,cases,spec)
            daily['pool']=pool;metrics.append(daily)
            summaries += [dict(pool=pool,**e) for e in summary]
            for split,p in panel.groupby('split'):
                counts.append(dict(pool=pool,split=split,rows=len(p),dates=p.trade_date.nunique(),mean_candidates=len(p)/p.trade_date.nunique(),
                    observed_labels=int(labels.loc[p.index,'label_observed'].sum()),st_rows=int(p.is_st.eq(1).sum()),short_history_rows=int((~p.history_valid).sum()),missing_cap_rows=int((~p.cap_valid).sum())))
            test=panel.split.eq('test')
            pool_test[pool]=panel.loc[test]
            cross_test[pool]=pred.loc[test]
            all_predictions.append(pred.assign(pool=pool))
            for case in cases:
                if case['account']:accounts_to_run.append((case,panel.loc[test],pred.loc[test,case['id']]))
        write_json(folder/'fits.json',fits_all);write_json(folder/'dataset.json',dict(counts=counts,policy=plan))
        pd.concat(metrics,ignore_index=True).to_parquet(folder/'date_metrics.parquet',index=False)
        pd.concat(all_predictions,ignore_index=True).to_parquet(folder/'predictions.parquet',index=False)
        # Common original test rows separate a changed ranking from a changed pool mean.
        oldpred=pd.read_parquet(source/'predictions.parquet');oldlabels=pd.read_parquet(source/'open_labels.parquet')
        oldpanel=pd.read_parquet(source/'panel.parquet');base=oldpanel.split.eq('test')
        cross=oldpred.loc[base,keys].copy()
        for pool in ['no_financial','mainboard_available','local_A_available']:
            matched=oldpanel.loc[base,keys].merge(cross_test[pool],on=keys,how='left',validate='one_to_one')
            for kind in ['ridge','tree']:cross[pool+'__'+kind]=matched[pool+'__'+kind].to_numpy()
        cross['original__ridge']=oldpred.loc[base,'ridge_open'].to_numpy();cross['original__tree']=oldpred.loc[base,'tree_open'].to_numpy()
        cross=cross.reset_index(drop=True);common=oldpanel.loc[base].reset_index(drop=True);commonlabels=oldlabels.loc[base].reset_index(drop=True)
        cc=[dict(id=c,target='weekly_open',regression=True) for c in cross if c not in keys]
        cd,cs=signal_metrics(common,commonlabels,cross,cc,spec)
        cd.to_parquet(folder/'common_date_metrics.parquet',index=False);cross.to_parquet(folder/'common_predictions.parquet',index=False)
        write_json(folder/'common_summaries.json',cs)
        del available,labels_all,all_predictions,pool_test,cross_test,oldpred,oldlabels,oldpanel
        acct_snapshot=root/'snapshots'/sm['account_snapshot_id'];am=json.loads((acct_snapshot/'manifest.json').read_text(encoding='utf-8'))
        if digest(acct_snapshot/'manifest.json')!=sm['account_snapshot_manifest_sha256']:raise ValueError('主板账户快照变化')
        verify_artifacts(acct_snapshot,am)
        bars=pd.read_parquet(acct_snapshot/'bars.parquet').sort_values(['stock_code','trade_date'])
        bars['avg_volume_20']=bars.groupby('stock_code',sort=False).volume.transform(lambda s:s.rolling(20).mean())
        audit_quotes=bars[['stock_code','trade_date','open']].set_index(['stock_code','trade_date'])
        market=Market(bars,calendar,pd.read_parquet(acct_snapshot/'actions.parquet'),pd.read_parquet(acct_snapshot/'status.parquet'),pd.read_parquet(acct_snapshot/'metadata.parquet'),am['missing_action_files'])
        del bars
        schedule=decision_dates(calendar,spec.test_start,spec.test_end,spec.frequency)
        from tools.verify_technical import audit as audit_account
        accounts=[]
        for i,(case,test,score) in enumerate(accounts_to_run):
            progress(f"主板实际账户 {i+1}/{len(accounts_to_run)}：{case['id']}")
            if set(test.trade_date)!=set(schedule):raise ValueError('缺少决策周')
            dec=score_decisions(test,score,schedule,spec.top_k)
            account_spec=ResearchSpec(strategy=Strategy(name=case['id'],lookback=60,skip=0,top_k=100,rebalance='weekly'),execution=Execution(),experiment=Experiment(spec.test_start,spec.test_end))
            definition=dict(momentum='外部冻结价量模型/随机分数；lookback是接口占位',ranking='周收盘排序Top100，0.98%目标权重；未知未来不排除',universe=POOLS[case['pool']],model=case,
                signal_parent=dict(run_id=rid,experiment_id=cfg['experiment_id'],predictions_sha256=digest(folder/'predictions.parquet')),
                parameter_origin='预先固定原基准参数；不做测试挑选',source='registered universe contrast')
            acct=run_account(cfg['project'],account_spec,root=root,publish=False,prepared=dict(snapshot=acct_snapshot,manifest=am,decisions=dec,schedule=schedule,market=market,signal_definition=definition),
                scenarios=plan['budget']['cost_scenarios'],research_context=dict(role='screen',source='registered_ml_universe',task_id=cfg['task_id'],experiment_id=cfg['experiment_id'],parent_ml_run=rid,case=case['id']),progress=progress)
            audited=audit_account(acct,quotes=audit_quotes);write_json(folder/('account_audit_'+case['id']+'.json'),audited)
            ar=json.loads((acct/'result.json').read_text(encoding='utf-8'))
            entry=dict(case=case['id'],name=case['name'],run_id=acct.name,manifest_sha256=digest(acct/'manifest.json'),portfolios=[])
            for p in ar['portfolios']:
                eq=pd.read_parquet(acct/p['metrics']['scenario']/'equity.parquet')
                entry['portfolios'].append(dict(metrics=p['metrics'],**account_statistics(eq,1_000_000.,spec.seed)))
            accounts.append(entry);write_json(folder/'account_index.json',accounts)
        write_json(folder/'links.json',dict(source_run_id=spec.source_run_id,source_manifest_sha256=cfg['source_manifest_hash'],accounts=[dict(case=a['case'],run_id=a['run_id'],manifest_sha256=a['manifest_sha256']) for a in accounts]))
        result=dict(kind='ml',study='universe',run_id=rid,name=spec.name,snapshot_id=cfg['snapshot_id'],research_context=dict(task_id=cfg['task_id'],experiment_id=cfg['experiment_id']),
            counts=counts,summaries=summaries,quantiles=[],study_plan=plan,accounts=accounts,common_summaries=cs,new_fits=len(fits_all),elapsed_seconds=time.perf_counter()-started,
            limitations=sm['limits']+['价量特征不变；新股缺失按训练中位数填补可能丢失缺失信息。','不同池自身均值不同，Top相对池收益不可直接当成同基准增量；另报共同原主板名单指标。','没有对固定随机种子做多种子不确定性检查，不是通用市场基准。','历史全部已查看；此次是按用户问题预设对照的回顾性研究，不是未来验证。'])
        write_json(folder/'result.json',result)
        write_json(folder/'manifest.json',dict(kind='ml',study='universe',run_id=rid,spec=spec.to_dict(),snapshot_id=cfg['snapshot_id'],snapshot_manifest_sha256=cfg['snapshot_manifest_hash'],code_hash=cfg['engine_hash'],
            artifacts={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*') if p.is_file() and p.name!='status.json'}))
        write_json(folder/'status.json',dict(kind='ml',study='universe',run_id=rid,status='completed',name=spec.name))
        return folder
    except BaseException as exc:
        write_json(folder/'status.json',dict(kind='ml',study='universe',run_id=rid,status='failed',error=str(exc)));raise
