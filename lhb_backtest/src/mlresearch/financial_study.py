"""Formal finite PIT financial/context study; original account engine unchanged."""
from datetime import datetime, timezone
import importlib.metadata, json, platform, time, uuid
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from ..technical.artifacts import digest, verify_artifacts, write_json, content_id
from ..technical.foundation_data import attach_reports
from ..technical.contracts import ResearchSpec, Strategy, Execution, Experiment
from ..technical.market import Market
from ..technical.runner import run as run_account
from ..technical.signals import decision_dates
from .contracts import MLSpec, FEATURES
from .fixed_blend import read
from .study_data import score_decisions
from .study_evaluation import account_statistics, signal_metrics

BASE=list(FEATURES)+['log_market_cap']
FIN=['annual_ep','annual_bp','annual_profitability_proxy','annual_period_age','annual_publication_age',
     'annual_ep_missing','annual_bp_missing','annual_profitability_missing']
MKT=['market_return_20','market_return_60','market_volatility_20','market_volatility_60','market_breadth_20','market_breadth_60']
CONTROLS=['small_cap','fixed_random','tree_open','tree_tuned','tree_size']
PROTOCOL_SHA='d73653d672b12bd46cfdd24ad7fa277eea59a12564d371e4f1b9fa45c772a415'

def financial_plan(spec):
    return dict(version=1,study='financial_context',protocol_sha256=PROTOCOL_SHA,
        cases=[dict(id='finance',name='价量规模+年度财务',features=BASE+FIN),dict(id='market',name='价量规模+市场状态',features=BASE+MKT),
               dict(id='combined',name='价量规模+财务及市场',features=BASE+FIN+MKT),dict(id='shuffled',name='合并组日期内打乱Y',features=BASE+FIN+MKT,shuffled=True)],
        controls=CONTROLS,source_run_id=spec.source_run_id,allocation=.98,top_k=100,execution=Execution().__dict__,
        annual_definition='strict publication_date < decision; latest December fiscal period; <=550 days; annual EP/BP/profit-total-equity proxy; missing flags; no eligibility deletion',
        market_definition='current listed non-ST positive volume/price mainboard equal mean log return; rolling 20/60 sum/std(ddof=1)/mean breadth, including t',
        model=dict(iterations=80,leaves=7,min_samples=200,learning_rate=.05,l2=10.,seed=11,early_stopping=False,median='mature train only'),
        budget=dict(max_new_fits=4,max_accounts=4,max_execution_attempts=1,cost_scenarios=['zero_transaction_cost','configured'],timeout_seconds=7200),
        gate=dict(cagr_gain=.02,max_mdd_worsening=.02,max_cagr_loss=.02,min_mdd_improvement=.05,year_return_floor=-.02,min_years=2),
        data_scope='all history previously seen; exact original candidate keys; no future-label inference selection')

def financial_source_identity(source,root):
    eid=read(source,'result.json')['research_context']['experiment_id'];job=root/'worker_jobs'/eid
    receipt=read(job,'receipt.json');audit=read(job,'ml_audit.json')
    if receipt['status']!='completed' or receipt['run_id']!=source.name or receipt['audit_hash']!=digest(job/'ml_audit.json') or not audit['verified']:
        raise ValueError('source receipt/audit mismatch')
    controls=[]
    for a in read(source,'account_index.json'):
        if a['case'] not in CONTROLS:continue
        account=root/'runs'/a['run_id'];saved=read(source,'account_audit_'+a['case']+'.json')
        if digest(account/'manifest.json')!=a['manifest_sha256'] or saved['run_id']!=account.name or set(saved['scenarios'])!={'configured','zero_transaction_cost'}:raise ValueError('control identity mismatch')
        verify_artifacts(account,read(account,'manifest.json'))
        controls.append(dict(case=a['case'],run_id=a['run_id'],manifest_sha256=a['manifest_sha256'],audit_sha256=digest(source/('account_audit_'+a['case']+'.json'))))
    if {a['case'] for a in controls}!=set(CONTROLS):raise ValueError('five controls missing')
    return dict(source_experiment=eid,receipt_sha256=digest(job/'receipt.json'),audit_sha256=digest(job/'ml_audit.json'),controls=controls)

def financial_features(panel,filings):
    joined=attach_reports(panel[['stock_code','trade_date','close']],filings)
    cap=np.exp(panel.log_market_cap.to_numpy())
    valid=joined.a_age_days.between(0,550)
    profit=joined.a_np_parent_company_owners;equity=joined.a_total_shareholder_equity
    out=panel.copy()
    out['annual_ep']=(profit/cap).where(valid)
    out['annual_bp']=(equity/cap).where(valid)
    out['annual_profitability_proxy']=(profit/equity.where(equity.gt(0))).where(valid)
    out['annual_period_age']=joined.a_age_days.where(valid)
    out['annual_publication_age']=(pd.to_datetime(joined.trade_date)-pd.to_datetime(joined.a_publication_date)).dt.days.where(valid)
    for x in FIN[:5]:out[x]=out[x].replace([np.inf,-np.inf],np.nan)
    for x,k in [('annual_ep','annual_ep_missing'),('annual_bp','annual_bp_missing'),('annual_profitability_proxy','annual_profitability_missing')]:out[k]=out[x].isna().astype(float)
    provenance=joined[['stock_code','trade_date','a_report_date','a_publication_date','a_age_days','a_np_parent_company_owners','a_total_shareholder_equity']].copy()
    provenance['annual_valid']=valid
    return out,provenance

def market_calendar_scope(calendar, snapshot_start, snapshot_end, study_end, decisions):
    """Scope only the market proxy; preserve the full label/account calendar."""
    if list(calendar) != sorted(set(calendar)):
        raise ValueError('market calendar must be ordered and unique')
    upper = min(snapshot_end, study_end)
    scoped = [day for day in calendar if snapshot_start <= day <= upper]
    if not scoped:
        raise ValueError('market calendar does not intersect snapshot coverage')
    positions = {day: index for index, day in enumerate(calendar)}
    available = set(scoped)
    for day in sorted(set(decisions)):
        index = positions.get(day)
        if day not in available or index is None:
            raise ValueError('market decision outside covered calendar')
        if index < 59 or not set(calendar[index - 59:index + 1]) <= available:
            raise ValueError('market decision requires 60 covered sessions')
    return scoped


def market_features(bars,metadata,calendar):
    p=bars[['stock_code','trade_date','close','pre_close','volume','is_st']].merge(metadata[['stock_code','listed_date','de_listed_date']],on='stock_code',validate='many_to_one')
    valid=p.listed_date.le(p.trade_date)&p.de_listed_date.gt(p.trade_date)&p.is_st.eq(0)&p.volume.gt(0)&p.close.gt(0)&p.pre_close.gt(0)&np.isfinite(p.close)&np.isfinite(p.pre_close)
    p=p.loc[valid].copy();p['log_return']=np.log(p.close/p.pre_close);p['up']=p.log_return.gt(0).astype(float)
    d=p.groupby('trade_date',sort=True).agg(mean_log_return=('log_return','mean'),breadth=('up','mean'),members=('stock_code','size')).reindex(calendar)
    if d.mean_log_return.isna().any():raise ValueError('market daily proxy missing; cannot insert zero')
    for w in [20,60]:
        d[f'market_return_{w}']=d.mean_log_return.rolling(w).sum()
        d[f'market_volatility_{w}']=d.mean_log_return.rolling(w).std()
        d[f'market_breadth_{w}']=d.breadth.rolling(w).mean()
    return d.rename_axis('trade_date').reset_index()

def run_financial(cfg,progress=lambda m:None):
    started=time.perf_counter();root=Path(cfg['root']);spec=MLSpec.from_dict(cfg['spec']);plan=financial_plan(spec)
    if plan!=cfg['study_plan']:raise ValueError('financial plan changed')
    env=dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in cfg['environment']['packages']})
    if env!=cfg['environment']:raise ValueError('environment changed')
    snapshot=root/'snapshots'/cfg['snapshot_id'];source=root/'ml_runs'/spec.source_run_id
    sm=read(snapshot,'manifest.json');srcm=read(source,'manifest.json')
    if digest(snapshot/'manifest.json')!=cfg['snapshot_manifest_hash'] or digest(source/'manifest.json')!=cfg['source_manifest_hash']:raise ValueError('input manifest changed')
    verify_artifacts(snapshot,sm);verify_artifacts(source,srcm)
    if financial_source_identity(source,root)!=cfg['source_identity']:raise ValueError('source receipt changed')
    rid=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8];folder=root/'ml_runs'/rid;folder.mkdir()
    write_json(folder/'status.json',dict(kind='ml',study='financial_context',run_id=rid,status='running'))
    try:
        for name,v in [('spec.json',spec.to_dict()),('study_plan.json',plan),('environment.json',env),('source_identity.json',cfg['source_identity'])]:write_json(folder/name,v)
        raw=pd.read_parquet(source/'panel.parquet');oldpred=pd.read_parquet(source/'predictions.parquet');labels=pd.read_parquet(source/'open_labels.parquet')
        keys=['stock_code','trade_date','split']
        if not raw[keys].equals(oldpred[keys]) or not raw[keys].equals(labels[keys]) or raw.duplicated(keys[:2]).any():raise ValueError('source keys differ')
        calendar=read(snapshot,'calendar.json');bars=pd.read_parquet(snapshot/'bars.parquet');meta=pd.read_parquet(snapshot/'metadata.parquet');filings=pd.read_parquet(snapshot/'fundamentals.parquet')
        progress('构造严格公告日前财务与当日主板市场代理；不修改共同名单')
        proxy_calendar = market_calendar_scope(calendar, sm['start'], sm['end'], spec.test_end, raw.trade_date)
        panel,provenance=financial_features(raw,filings)
        daily=market_features(bars,meta,proxy_calendar)
        panel=panel.merge(daily[['trade_date']+MKT],on='trade_date',how='left',validate='many_to_one')
        if not panel[keys].equals(raw[keys]) or not np.isfinite(panel[MKT]).all().all():raise ValueError('market history missing or candidate order changed')
        panel.to_parquet(folder/'panel.parquet',index=False);provenance.to_parquet(folder/'financial_provenance.parquet',index=False);daily.to_parquet(folder/'market_daily.parquet',index=False)
        predictions=panel[keys].copy();(folder/'models').mkdir();entries=[]
        train=panel.trade_date.between(spec.train_start,spec.train_end)&labels.label_observed&labels.label_end.le(spec.train_end)
        if int(train.sum())!=378671:raise ValueError('mature train differs')
        for i,c in enumerate(plan['cases']):
            e=dict(id=c['id'],status='running',features=c['features'],cutoff=spec.train_end,rows=int(train.sum()),
                max_decision_date=panel.loc[train,'trade_date'].max(),max_label_end=labels.loc[train,'label_end'].max(),shuffled=c.get('shuffled',False),parameters=plan['model'])
            entries.append(e);write_json(folder/'fits.json',entries)
            progress(f"固定拟合 {i+1}/4：{c['name']}，{len(c['features'])}列")
            try:
                x=panel.loc[train,c['features']];y=labels.loc[train,'label_relative'].copy()
                if x.notna().sum().eq(0).any():raise ValueError('all-empty mature train feature')
                if e['shuffled']:
                    rng=np.random.default_rng(spec.seed)
                    for _,ids in panel.loc[train].groupby('trade_date').groups.items():y.loc[ids]=rng.permutation(y.loc[ids].to_numpy())
                e['training_y_sha256']=content_id(y.tolist())
                model=make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,
                    min_samples_leaf=200,learning_rate=.05,l2_regularization=10.,random_state=11,early_stopping=False))
                with threadpool_limits(limits=4):model.fit(x,y);predictions[c['id']]=model.predict(panel[c['features']])
                path=folder/'models'/(c['id']+'.joblib');joblib.dump(model,path)
                e.update(status='completed',model_file=path.relative_to(folder).as_posix(),model_sha256=digest(path));write_json(folder/'fits.json',entries)
            except BaseException as exc:e.update(status='failed',error=str(exc));write_json(folder/'fits.json',entries);raise
        predictions.to_parquet(folder/'predictions.parquet',index=False)
        diagnostic_cases=[dict(id=c['id'],regression=True,target='weekly_open') for c in plan['cases']]+[dict(id='tree_size',regression=True,target='weekly_open')]
        diagpred=predictions.assign(tree_size=oldpred.tree_size);metrics,summaries=signal_metrics(panel,labels,diagpred,diagnostic_cases,spec)
        metrics.to_parquet(folder/'date_metrics.parquet',index=False)
        write_json(folder/'dataset.json',dict(rows=len(panel),train_rows=int(train.sum()),max_label_end=labels.loc[train,'label_end'].max(),
            counts=[dict(split=s,rows=len(p),dates=p.trade_date.nunique(),missing={k:int(p[k].isna().sum()) for k in FIN[:5]}) for s,p in panel.groupby('split')],
            features=dict(base=BASE,annual=FIN,market=MKT),source_panel_sha256=digest(source/'panel.parquet'),source_labels_sha256=digest(source/'open_labels.parquet'),exact_source_keys=True))
        test=panel[panel.split.eq('test')];schedule=decision_dates(calendar,spec.test_start,spec.test_end,spec.frequency)
        if len(schedule)!=140 or set(schedule)!=set(test.trade_date):raise ValueError('test dates differ')
        reused=[]
        for link in cfg['source_identity']['controls']:
            acc=root/'runs'/link['run_id'];expected=score_decisions(test,oldpred.loc[test.index,link['case']],schedule,100)
            pd.testing.assert_frame_equal(expected.reset_index(drop=True),pd.read_parquet(acc/'decisions.parquet').reset_index(drop=True))
            r=read(acc,'result.json');portfolios=[]
            for p in r['portfolios']:
                eq=pd.read_parquet(acc/p['metrics']['scenario']/'equity.parquet');portfolios.append(dict(metrics=p['metrics'],**account_statistics(eq,1e6,11)))
            reused.append(dict(**link,name=next(a['name'] for a in read(source,'account_index.json') if a['case']==link['case']),portfolios=portfolios,reused=True,numerical_audit='source saved audit; not rerun'))
        write_json(folder/'reused_accounts.json',reused)
        bars=bars.sort_values(['stock_code','trade_date']);bars['avg_volume_20']=bars.groupby('stock_code',sort=False).volume.transform(lambda s:s.rolling(20).mean())
        quotes=bars[['stock_code','trade_date','open']].set_index(['stock_code','trade_date'])
        market=Market(bars,calendar,pd.read_parquet(snapshot/'actions.parquet'),pd.read_parquet(snapshot/'status.parquet'),meta,sm['missing_action_files'])
        del bars,raw,oldpred,filings,provenance
        from tools.verify_technical import audit as audit_account
        accounts=[]
        for i,c in enumerate(plan['cases']):
            progress(f"固定新账户 {i+1}/4：{c['name']}")
            rs=ResearchSpec(strategy=Strategy(name=c['name'],lookback=60,skip=0,top_k=100,rebalance='weekly'),execution=Execution(),experiment=Experiment(spec.test_start,spec.test_end))
            dec=score_decisions(test,predictions.loc[test.index,c['id']],schedule,100)
            definition=dict(momentum='原价量规模加严格公告日前财务及已知市场环境的固定树评分',ranking='score desc, code asc',universe='exact original source candidates; no future label or financial coverage filter',
                model=c,signal_parent=dict(run_id=rid,experiment_id=cfg['experiment_id'],predictions_sha256=digest(folder/'predictions.parquet')),
                parameter_origin='PROTOCOL fixed four; train2020-22; valid diagnostic only',source='registered financial_context study')
            acc=run_account(cfg['project'],rs,root=root,publish=False,prepared=dict(snapshot=snapshot,manifest=sm,decisions=dec,schedule=schedule,market=market,signal_definition=definition),
                scenarios=plan['budget']['cost_scenarios'],research_context=dict(role='screen',source='registered_ml_financial_context',task_id=cfg['task_id'],experiment_id=cfg['experiment_id'],parent_ml_run=rid,case=c['id']),progress=progress)
            audited=audit_account(acc,quotes=quotes);write_json(folder/('account_audit_'+c['id']+'.json'),audited)
            r=read(acc,'result.json');portfolios=[]
            for p in r['portfolios']:
                eq=pd.read_parquet(acc/p['metrics']['scenario']/'equity.parquet');portfolios.append(dict(metrics=p['metrics'],**account_statistics(eq,1e6,11)))
            accounts.append(dict(case=c['id'],name=c['name'],run_id=acc.name,manifest_sha256=digest(acc/'manifest.json'),portfolios=portfolios));write_json(folder/'account_index.json',accounts)
        write_json(folder/'links.json',dict(source_run_id=source.name,source_manifest_sha256=cfg['source_manifest_hash'],accounts=[{k:a[k] for k in ['case','run_id','manifest_sha256']} for a in accounts]))
        result=dict(kind='ml',study='financial_context',run_id=rid,name=spec.name,snapshot_id=cfg['snapshot_id'],source_run_id=source.name,
            research_context=dict(task_id=cfg['task_id'],experiment_id=cfg['experiment_id']),study_plan=plan,accounts=accounts,reused_accounts=reused,summaries=summaries,
            counts=dict(test=dict(candidates=len(test),decision_weeks=len(schedule))),new_fits=len(entries),elapsed_seconds=time.perf_counter()-started,
            limitations=['all history seen','annual not TTM or exact ROE; vendor vintages not restored','mainboard market proxy, not identified index','original accounting/execution approximations retained','independent method/data review pending'])
        write_json(folder/'result.json',result)
        (folder/'REPORT.md').write_text('固定四财务/市场模型已执行。完整模型、PIT来源、市场代理、全部账户和保存审计见清单；GATE与不确定性另见关联说明包。',encoding='utf8')
        write_json(folder/'manifest.json',dict(run_id=rid,kind='ml',study='financial_context',spec=spec.to_dict(),snapshot_id=cfg['snapshot_id'],snapshot_manifest_sha256=cfg['snapshot_manifest_hash'],
            source_manifest_sha256=cfg['source_manifest_hash'],code_hash=cfg['engine_hash'],artifacts={p.relative_to(folder).as_posix():digest(p) for p in folder.rglob('*') if p.is_file() and p.name!='status.json'}))
        write_json(folder/'status.json',dict(kind='ml',study='financial_context',run_id=rid,status='completed'));return folder
    except BaseException as exc:write_json(folder/'status.json',dict(kind='ml',study='financial_context',run_id=rid,status='failed',error=str(exc)));raise
