"""Separate PIT/rolling arithmetic and model/account checks; not method review."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from ..technical.artifacts import digest,verify_artifacts,content_id,write_json
from ..technical.signals import decision_dates
from .contracts import MLSpec
from .fixed_blend import read
from .financial_study import BASE,FIN,MKT,financial_plan,financial_source_identity,financial_features,market_features
from .blend_audit import reference_decisions
from .study_audit import verify_links
from .study_evaluation import account_statistics

def annual_reference(panel,filings):
    """Per-stock fiscal running maximum; searchsorted strict-before instead of asof."""
    n=len(panel);values=np.full((n,5),np.nan);rd=np.full(n,'',dtype=object);pub=np.full(n,'',dtype=object)
    annual=filings[filings.report_date.str.endswith('12-31')]
    by_code={c:f for c,f in annual.groupby('stock_code')}
    for code,ids in panel.groupby('stock_code',sort=False).groups.items():
        f=by_code.get(code)
        if f is None:continue
        f=f.sort_values(['publication_date','report_date']);retained=[];latest=''
        for r in f.itertuples(index=False):
            if r.report_date>=latest:retained.append(r);latest=r.report_date
        pubs=np.array([r.publication_date for r in retained]);days=panel.loc[ids,'trade_date'].to_numpy()
        choices=np.searchsorted(pubs,days,side='left')-1;ok=choices>=0
        rows=np.asarray(ids)[ok];chosen=choices[ok]
        period=np.array([retained[j].report_date for j in chosen]);publication=pubs[chosen]
        age=(pd.to_datetime(panel.loc[rows,'trade_date']).to_numpy()-pd.to_datetime(period).to_numpy())/np.timedelta64(1,'D')
        pub_age=(pd.to_datetime(panel.loc[rows,'trade_date']).to_numpy()-pd.to_datetime(publication).to_numpy())/np.timedelta64(1,'D')
        rd[rows]=period;pub[rows]=publication
        valid=(age>=0)&(age<=550);cap=np.exp(panel.loc[rows,'log_market_cap'].to_numpy())
        profit=np.array([retained[j].np_parent_company_owners for j in chosen],float)
        equity=np.array([retained[j].total_shareholder_equity for j in chosen],float)
        with np.errstate(divide='ignore',invalid='ignore'):
            v=np.column_stack([profit/cap,equity/cap,profit/np.where(equity>0,equity,np.nan),age,pub_age])
        v[~valid,:]=np.nan;v[~np.isfinite(v)]=np.nan;values[rows]=v
    out=pd.DataFrame(values,columns=FIN[:5]);out[FIN[5]]=out[FIN[0]].isna().astype(float);out[FIN[6]]=out[FIN[1]].isna().astype(float);out[FIN[7]]=out[FIN[2]].isna().astype(float)
    return out,rd,pub

def audit_market_calendar_scope(calendar, snapshot_start, snapshot_end, study_end, decisions):
    """Independent coverage check; never change label or execution dates."""
    if any(left >= right for left, right in zip(calendar, calendar[1:])):
        raise ValueError('market calendar must be ordered and unique')
    scoped = [day for day in calendar if snapshot_start <= day <= snapshot_end and day <= study_end]
    if not scoped:
        raise ValueError('market calendar does not intersect snapshot coverage')
    for day in set(decisions):
        if day not in scoped:
            raise ValueError('market decision outside covered calendar')
        index = calendar.index(day)
        window = calendar[max(0, index - 59):index + 1]
        if len(window) != 60 or any(date not in scoped for date in window):
            raise ValueError('market decision requires 60 covered sessions')
    return scoped


def market_reference(bars,metadata,calendar):
    """Independent numpy sums/counts and explicit rolling windows."""
    meta=metadata.set_index('stock_code');codes=bars.stock_code
    listing=codes.map(meta.listed_date).to_numpy();delisting=codes.map(meta.de_listed_date).to_numpy();days=bars.trade_date.to_numpy()
    close=bars.close.to_numpy();pre=bars.pre_close.to_numpy();vol=bars.volume.to_numpy();st=bars.is_st.to_numpy()
    ok=(listing<=days)&(delisting>days)&(st==0)&(vol>0)&(close>0)&(pre>0)&np.isfinite(close)&np.isfinite(pre)
    positions=bars.trade_date.map({d:i for i,d in enumerate(calendar)})
    ok=ok&positions.notna().to_numpy()
    ids=positions.loc[ok].to_numpy(int)
    r=np.log(close[ok]/pre[ok]);counts=np.bincount(ids,minlength=len(calendar))
    if np.any(counts==0):raise ValueError('market daily proxy missing; cannot insert zero')
    means=np.bincount(ids,weights=r,minlength=len(calendar))/counts
    breadth=np.bincount(ids,weights=(r>0).astype(float),minlength=len(calendar))/counts
    out=pd.DataFrame(dict(trade_date=calendar,mean_log_return=means,breadth=breadth,members=counts))
    for w in [20,60]:
        for col in ['return','volatility','breadth']:out[f'market_{col}_{w}']=np.nan
        for end in range(w-1,len(calendar)):
            window=means[end-w+1:end+1]
            out.loc[end,f'market_return_{w}']=np.sum(window)
            out.loc[end,f'market_volatility_{w}']=np.std(window,ddof=1)
            out.loc[end,f'market_breadth_{w}']=np.mean(breadth[end-w+1:end+1])
    return out

def audit_financial(folder,root):
    folder,root=Path(folder),Path(root);m=read(folder,'manifest.json');verify_artifacts(folder,m)
    spec=MLSpec.from_dict(m['spec']);plan=financial_plan(spec)
    if plan!=read(folder,'study_plan.json'):raise ValueError('fixed plan differs')
    links=verify_links(folder,root)
    if len(links['accounts'])!=4 or {a['case'] for a in links['accounts']}!={c['id'] for c in plan['cases']}:raise ValueError('matrix incomplete')
    source=root/'ml_runs'/spec.source_run_id;snapshot=root/'snapshots'/m['snapshot_id']
    if digest(snapshot/'manifest.json')!=m['snapshot_manifest_sha256']:raise ValueError('snapshot changed')
    verify_artifacts(snapshot,read(snapshot,'manifest.json'))
    if financial_source_identity(source,root)!=read(folder,'source_identity.json'):raise ValueError('source identities changed')
    panel=pd.read_parquet(folder/'panel.parquet');raw=pd.read_parquet(source/'panel.parquet');labels=pd.read_parquet(source/'open_labels.parquet');pred=pd.read_parquet(folder/'predictions.parquet')
    keys=['stock_code','trade_date','split']
    if not panel[keys].equals(raw[keys]) or not panel[keys].equals(labels[keys]) or not panel[keys].equals(pred[keys]) or panel.duplicated(keys[:2]).any():raise ValueError('candidate keys changed')
    pd.testing.assert_frame_equal(panel[raw.columns],raw,check_exact=True)
    if not panel.q_publication_date.lt(panel.trade_date).all():raise ValueError('future shares eligibility')
    filings=pd.read_parquet(snapshot/'fundamentals.parquet');provenance=pd.read_parquet(folder/'financial_provenance.parquet')
    ref,rd,pub=annual_reference(raw,filings)
    np.testing.assert_allclose(panel[FIN],ref[FIN],rtol=1e-12,atol=1e-12,equal_nan=True)
    if not np.array_equal(provenance.a_report_date.fillna('').to_numpy(),rd) or not np.array_equal(provenance.a_publication_date.fillna('').to_numpy(),pub):raise ValueError('annual identity differs')
    known=pub!=''
    if not (pub[known]<panel.loc[known,'trade_date'].to_numpy()).all():raise ValueError('annual publication not strict-before')
    bars=pd.read_parquet(snapshot/'bars.parquet',columns=['stock_code','trade_date','close','pre_close','volume','is_st']);meta=pd.read_parquet(snapshot/'metadata.parquet');calendar=read(snapshot,'calendar.json')
    snapshot_manifest=read(snapshot,'manifest.json')
    proxy_calendar=audit_market_calendar_scope(calendar,snapshot_manifest['start'],snapshot_manifest['end'],spec.test_end,raw.trade_date)
    daily=pd.read_parquet(folder/'market_daily.parquet');oracle=market_reference(bars,meta,proxy_calendar)
    if daily.trade_date.tolist()!=proxy_calendar:raise ValueError('saved market calendar differs')
    np.testing.assert_allclose(daily.drop(columns='trade_date'),oracle.drop(columns='trade_date'),atol=1e-12,rtol=1e-12,equal_nan=True)
    expected_mkt=panel[['trade_date']].merge(oracle[['trade_date']+MKT],on='trade_date',how='left')
    np.testing.assert_allclose(panel[MKT],expected_mkt[MKT],atol=1e-12,rtol=1e-12)
    prefix=[]
    for cutoff in ['2022-12-31','2023-12-31','2025-12-31']:
        mask=raw.trade_date.le(cutoff);before=raw.loc[mask].reset_index(drop=True)
        reconstructed,_=financial_features(before,filings[filings.publication_date.le(cutoff)])
        np.testing.assert_allclose(panel.loc[mask,FIN],reconstructed[FIN],atol=1e-12,rtol=1e-12,equal_nan=True)
        cal=[d for d in proxy_calendar if d<=cutoff];small=market_features(bars[bars.trade_date.le(cutoff)],meta,cal)
        np.testing.assert_allclose(daily[daily.trade_date.le(cutoff)].drop(columns='trade_date'),small.drop(columns='trade_date'),atol=1e-12,rtol=1e-12,equal_nan=True)
        prefix.append(dict(cutoff=cutoff,rows=int(mask.sum()),financial_equal=True,market_equal=True))
    train=panel.trade_date.between(spec.train_start,spec.train_end)&labels.label_observed&labels.label_end.le(spec.train_end)
    fits=read(folder,'fits.json')
    if len(fits)!=4 or any(e['status']!='completed' for e in fits):raise ValueError('fit budget/status invalid')
    prediction_error=0.
    for e,c in zip(fits,plan['cases']):
        if e['id']!=c['id'] or e['features']!=c['features'] or e['cutoff']!=spec.train_end or e['parameters']!=plan['model'] or e['rows']!=int(train.sum()) or e['max_label_end']!=labels.loc[train,'label_end'].max():raise ValueError('training contract differs')
        path=folder/e['model_file']
        if digest(path)!=e['model_sha256']:raise ValueError('model changed')
        model=joblib.load(path);np.testing.assert_allclose(model.steps[0][1].statistics_,panel.loc[train,e['features']].median().to_numpy(),atol=1e-12,rtol=1e-12)
        params=model.steps[1][1].get_params()
        for k,v in dict(max_iter=80,max_leaf_nodes=7,min_samples_leaf=200,learning_rate=.05,l2_regularization=10.,random_state=11,early_stopping=False).items():
            if params[k]!=v:raise ValueError('model parameter changed')
        y=labels.loc[train,'label_relative'].copy()
        if c.get('shuffled'):
            rng=np.random.default_rng(spec.seed)
            for _,ids in panel.loc[train].groupby('trade_date').groups.items():y.loc[ids]=rng.permutation(y.loc[ids].to_numpy())
        if e['training_y_sha256']!=content_id(y.tolist()):raise ValueError('training Y identity differs')
        with threadpool_limits(limits=4):expected=model.predict(panel[e['features']])
        error=float(np.max(np.abs(expected-pred[e['id']].to_numpy())));prediction_error=max(prediction_error,error)
        if error>1e-10:raise ValueError('saved model prediction differs')
    test=panel[panel.split.eq('test')];schedule=decision_dates(calendar,spec.test_start,spec.test_end,'weekly')
    if len(schedule)!=140 or set(schedule)!=set(test.trade_date):raise ValueError('test calendar changed')
    base=root/'runs'/next(a['run_id'] for a in read(folder,'reused_accounts.json') if a['case']=='small_cap');base_spec=read(base,'spec.json')
    indexed={a['case']:a for a in read(folder,'account_index.json')};maxerr=0.;fills=0;targets=0
    for link in links['accounts']:
        account=root/'runs'/link['run_id'];sp=read(account,'spec.json')
        if sp['execution']!=base_spec['execution'] or sp['experiment']!=base_spec['experiment']:raise ValueError('execution/account definition changed')
        a=dict(sp['strategy']);b=dict(base_spec['strategy']);a.pop('name');b.pop('name')
        if a!=b:raise ValueError('account strategy interface changed')
        target=reference_decisions(test,pred.loc[test.index,link['case']],schedule);observed=pd.read_parquet(account/'decisions.parquet')
        pd.testing.assert_frame_equal(target.reset_index(drop=True),observed.reset_index(drop=True),check_exact=False,atol=1e-12,rtol=0);targets+=len(target)
        saved=read(folder,'account_audit_'+link['case']+'.json')
        if saved['run_id']!=account.name or set(saved['scenarios'])!=set(plan['budget']['cost_scenarios']):raise ValueError('account audit missing')
        r=read(account,'result.json')
        for p in r['portfolios']:
            scenario=p['metrics']['scenario'];audit=saved['scenarios'][scenario];fills+=audit['fills'];maxerr=max(maxerr,max(audit['daily_errors'].values()))
            eq=pd.read_parquet(account/scenario/'equity.parquet');idx=next(x for x in indexed[link['case']]['portfolios'] if x['metrics']['scenario']==scenario)
            if idx['metrics']!=p['metrics'] or account_statistics(eq,1e6,11)!={k:idx[k] for k in ['annual','sharpe_zero_rate']}:raise ValueError('account index metrics differ')
    if maxerr>=.001:raise ValueError('account arithmetic error')
    # Audit output lives in job/ml_audit.json, never adds files to the frozen result.
    return dict(kind='ml',study='financial_context',run_id=folder.name,verified=True,rows=len(panel),test_rows=len(test),decision_weeks=140,new_fits=4,
        accounts=4,account_scenarios=8,reused_control_accounts=5,reused_control_scenarios=10,checked_target_rows=targets,checked_fills=fills,
        prediction_full_max_error=prediction_error,max_daily_account_error=maxerr,mature_train_rows=int(train.sum()),max_label_end=labels.loc[train,'label_end'].max(),
        annual_pit_all_rows_checked=True,market_all_rows_checked=True,prefix_checks=prefix,financial_missing_does_not_delete_candidates=True,
        verification_scope='文件身份/全部PIT公式与市场代理独立算术/三固定前缀/成熟训练填补和Y身份/保存模型全部评分及目标；原核查器逐日现金股数开盘费用；旧5控制仅身份与保存核查复用；未重拟合，非双独立方法复核或供应商证明')
