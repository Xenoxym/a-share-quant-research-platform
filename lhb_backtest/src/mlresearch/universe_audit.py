"""Verify immutable universe membership, maturity, saved scores and linked accounts."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ..technical.artifacts import digest,verify_artifacts
from ..technical.signals import decision_dates
from .contracts import MLSpec,FEATURES
from .study_audit import verify_links
from .study_data import fixed_scores,score_decisions
from .universe import universe_plan,pool_masks,weekly_random


def read(folder,name):return json.loads((folder/name).read_text(encoding='utf-8'))


def check_account_audit(audited,run_id,scenarios):
    if audited.get('run_id')!=run_id or set(audited.get('scenarios',{}))!=set(scenarios):
        raise ValueError('账户审计身份或情景不完整')
    for value in audited['scenarios'].values():
        errors=value.get('daily_errors',{})
        if set(errors)!={'cash','receivable','equity'} or any(not np.isfinite(v) or v>=.001 or v<0 for v in errors.values()):
            raise ValueError('账户审计误差超限')
    return sum(v['fills'] for v in audited['scenarios'].values())


def audit_universe(folder,root):
    folder,root=Path(folder),Path(root);manifest=read(folder,'manifest.json');verify_artifacts(folder,manifest)
    spec=MLSpec.from_dict(manifest['spec']);plan=universe_plan(spec)
    if plan!=read(folder,'study_plan.json'):raise ValueError('股票范围矩阵改变')
    snapshot=root/'snapshots'/manifest['snapshot_id']
    if digest(snapshot/'manifest.json')!=manifest['snapshot_manifest_sha256']:raise ValueError('快照清单改变')
    verify_artifacts(snapshot,read(snapshot,'manifest.json'));links=verify_links(folder,root)
    cases=plan['cases'];expected_accounts={c['id'] for c in cases if c['account']}
    if {a['case'] for a in links['accounts']}!=expected_accounts:raise ValueError('主板账户矩阵不完整')
    if len(links['accounts'])>plan['budget']['max_accounts']:raise ValueError('账户预算超限')
    available=pd.read_parquet(folder/'available_panel.parquet');all_labels=pd.read_parquet(folder/'available_labels.parquet');keys=['stock_code','trade_date','split']
    if available.duplicated(keys[:2]).any() or not available[keys].equals(all_labels[keys]):raise ValueError('可用样本主键无效')
    masks=pool_masks(available,spec);entries=read(folder,'fits.json')
    if len(entries)!=6 or any(e['status']!='completed' for e in entries):raise ValueError('拟合矩阵不完整')
    score_error=0.;metric_error=0.;ledger_checks=0
    for pool,mask in masks.items():
        sub=folder/'pools'/pool;panel=pd.read_parquet(sub/'panel.parquet');labels=pd.read_parquet(sub/'open_labels.parquet');pred=pd.read_parquet(sub/'predictions.parquet')
        if not panel[keys].equals(available.loc[mask,keys].reset_index(drop=True)) or not panel[keys].equals(pred[keys]) or not panel[keys].equals(labels[keys]):raise ValueError('股票范围或预测身份不一致')
        expected=labels.label_return-labels.groupby('trade_date').label_return.transform('mean')
        np.testing.assert_allclose(expected,labels.label_relative,atol=1e-12,equal_nan=True)
        if not labels.label_observed.equals(np.isfinite(labels.label_return)):raise ValueError('标签可观测标志错误')
        np.testing.assert_allclose(pred[pool+'__weekly_random'],weekly_random(panel,spec.seed),atol=0)
        if pool!='qualified':np.testing.assert_allclose(pred[pool+'__fixed_random'],fixed_scores(panel,spec.seed).fixed_random,atol=0)
        for e in [e for e in entries if e['pool']==pool]:
            mature=panel.trade_date.between(spec.train_start,e['cutoff']) & labels.label_observed & labels.label_end.le(e['cutoff'])
            if mature.sum()!=e['rows'] or e['max_label_end']!=labels.loc[mature,'label_end'].max() or e['max_label_end']>spec.train_end:raise ValueError('训练时间/成熟标签边界错误')
            path=sub/e['model_file']
            if digest(path)!=e['model_sha256']:raise ValueError('模型哈希错误')
            model=joblib.load(path)
            np.testing.assert_allclose(model.steps[0][1].statistics_,panel.loc[mature,list(FEATURES)].median(),rtol=1e-12,atol=1e-12)
            sample=panel.sample(min(160,len(panel)),random_state=spec.seed)
            with threadpool_limits(limits=4):score=model.predict(sample[list(FEATURES)])
            actual=pred.loc[sample.index,pool+'__'+e['kind']]
            err=float(np.max(np.abs(score-actual)));score_error=max(score_error,err)
            if err>1e-10:raise ValueError('保存模型无法复算预测')
        # Recalculate random dates' RankIC and Top100 against raw labels, not summary fields.
        daily=pd.read_parquet(folder/'date_metrics.parquet');daily=daily[daily.pool.eq(pool)]
        for row in daily.sample(min(40,len(daily)),random_state=spec.seed).itertuples():
            ids=panel.index[panel.trade_date.eq(row.trade_date)]
            s=pred.loc[ids,row.model];y=labels.loc[ids,'label_return'];valid=y.notna()
            ic=s[valid].rank().corr(y[valid].rank())
            top=pd.DataFrame(dict(score=s,code=panel.loc[ids,'stock_code'])).sort_values(['score','code'],ascending=[False,True]).head(100).index
            excess=labels.loc[top,'label_return'].mean()-y.mean()
            np.testing.assert_allclose([ic,excess],[row.rank_ic,row.top_excess],atol=1e-12,equal_nan=True)
            if np.isfinite(ic):metric_error=max(metric_error,abs(float(ic-row.rank_ic)))
        for case in [c for c in cases if c['pool']==pool and c['account']]:
            test=panel[panel.split.eq('test')];schedule=decision_dates(read(snapshot,'calendar.json'),spec.test_start,spec.test_end,'weekly')
            decision=score_decisions(test,pred.loc[test.index,case['id']],schedule,100)
            link=next(a for a in links['accounts'] if a['case']==case['id']);acct=root/'runs'/link['run_id']
            saved=pd.read_parquet(acct/'decisions.parquet')
            for col in ['stock_code','trade_date','rank','execution_date']:pd.testing.assert_series_equal(decision[col].reset_index(drop=True),saved[col].reset_index(drop=True),check_names=False)
            np.testing.assert_allclose(decision.score,saved.score,atol=1e-12)
            check_account_audit(read(folder,'account_audit_'+case['id']+'.json'),link['run_id'],plan['budget']['cost_scenarios'])
            ledger_checks+=1
    # Independent reference-price multiplication on sampled endpoints, including unknown labels.
    sample=all_labels.sample(min(180,len(all_labels)),random_state=spec.seed)
    raw=pd.read_parquet(snapshot/'bars.parquet',columns=['stock_code','trade_date','open','close','pre_close'],filters=[('stock_code','in',sample.stock_code.unique().tolist())]).set_index(['stock_code','trade_date'])
    calendar=read(snapshot,'calendar.json');pos={d:i for i,d in enumerate(calendar)};label_error=0.
    for r in sample.itertuples():
        expected=np.nan
        if pd.notna(r.label_end):
            interval=calendar[pos[r.label_entry]:pos[r.label_end]]
            rows=raw.reindex(pd.MultiIndex.from_tuples([(r.stock_code,d) for d in interval]))
            endpoints=raw.reindex(pd.MultiIndex.from_tuples([(r.stock_code,r.label_entry),(r.stock_code,r.label_end)]))
            ratio=rows.close/rows.pre_close
            if ratio.notna().all() and ratio.gt(0).all() and np.isfinite(ratio).all() and endpoints[['open','pre_close']].gt(0).all().all():
                expected=ratio.prod()*endpoints.pre_close.iloc[0]/endpoints.open.iloc[0]*endpoints.open.iloc[1]/endpoints.pre_close.iloc[1]-1
        np.testing.assert_allclose(expected,r.label_return,atol=1e-10,equal_nan=True)
        if np.isfinite(expected):label_error=max(label_error,abs(float(expected-r.label_return)))
    result=read(folder,'result.json');daily=pd.read_parquet(folder/'date_metrics.parquet')
    for summary in result['summaries']:
        part=daily[daily.model.eq(summary['model']) & daily.split.eq(summary['scope'])]
        np.testing.assert_allclose([part.rank_ic.mean(),part.top_excess.mean()],[summary['rank_ic'],summary['top_excess']],atol=1e-12,equal_nan=True)
    return dict(kind='ml',study='universe',run_id=folder.name,verified=True,rows=len(available),new_fits=len(entries),linked_accounts=ledger_checks,
        prediction_sample_max_error=score_error,label_sample_max_error=label_error,metric_sample_max_error=metric_error,
        verification_scope='frozen artifacts, full pool identity, mature training/imputation, sampled saved predictions and independently multiplied labels/metrics, nine full-ledger audited mainboard accounts; not vendor certification or independent research review')
