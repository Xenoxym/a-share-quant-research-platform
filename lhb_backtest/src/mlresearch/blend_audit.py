"""Independent rank/selection arithmetic and original linked account audits."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from ..technical.artifacts import digest, verify_artifacts
from ..technical.contracts import ResearchSpec
from ..technical.signals import decision_dates
from .contracts import MLSpec
from .fixed_blend import read, blend_plan, source_identity
from .study_audit import verify_links
from .study_evaluation import account_statistics

def reference_scores(panel, pred, cases):
    """Separate date loops and scipy average ranks, independent of scores()."""
    out=panel[['stock_code','trade_date','split']].copy()
    for c in cases:out[c['id']]=np.nan
    for day,ids in panel.groupby('trade_date').groups.items():
        p=panel.loc[ids];r=pred.loc[ids]
        n=len(p)
        raw=dict(size=-p.log_market_cap.to_numpy(),tree_open=r.tree_open.to_numpy(),tree_tuned=r.tree_tuned.to_numpy(),
            lowvol20=-p.volatility_20.to_numpy(),lowvol60=-p.volatility_60.to_numpy())
        ranks={k:rankdata(v,method='average')/n for k,v in raw.items()}
        order=sorted(range(n),key=lambda j:(p.log_market_cap.iloc[j],p.stock_code.iloc[j]))
        for c in cases:
            if 'pool_size' in c:
                chosen=order[:c['pool_size']]
                values=np.full(n,np.nan);values[chosen]=raw['tree_open'][chosen]
            else:
                values=np.zeros(n)
                for k,w in c['weights'].items():values+=w*ranks[k]
            out.loc[ids,c['id']]=values
    return out

def reference_decisions(panel,score,schedule):
    rows=[]
    for day,ids in panel.groupby('trade_date').groups.items():
        available=[i for i in ids if np.isfinite(score.loc[i])]
        top=sorted(available,key=lambda i:(-score.loc[i],panel.stock_code.loc[i]))[:100]
        for rank,i in enumerate(top,1):
            rows.append(dict(stock_code=panel.stock_code.loc[i],trade_date=day,close=panel.close.loc[i],score=score.loc[i],rank=rank,
                selected=True,reason='eligible_external_score',target_weight=.0098,execution_date=schedule[day],known_at_assumed=day+'T15:30:00+08:00'))
    return pd.DataFrame(rows)

def audit_blend(folder, root):
    folder,root=Path(folder),Path(root)
    m=read(folder,'manifest.json');verify_artifacts(folder,m)
    spec=MLSpec.from_dict(m['spec']);plan=blend_plan(spec)
    if plan!=read(folder,'study_plan.json'):raise ValueError('固定矩阵与登记不同')
    links=verify_links(folder,root)
    if len(links['accounts'])!=8 or set(a['case'] for a in links['accounts'])!={c['id'] for c in plan['cases']}:
        raise ValueError('八账户矩阵不完整')
    source=root/'ml_runs'/spec.source_run_id
    if source_identity(source,root)!=read(folder,'source_identity.json'):raise ValueError('源登记/审计身份变化')
    snapshot=root/'snapshots'/m['snapshot_id']
    if digest(snapshot/'manifest.json')!=m['snapshot_manifest_sha256']:raise ValueError('快照身份变化')
    verify_artifacts(snapshot,read(snapshot,'manifest.json'))
    raw=pd.read_parquet(source/'panel.parquet');oldpred=pd.read_parquet(source/'predictions.parquet')
    keys=['stock_code','trade_date','split']
    if not raw[keys].equals(oldpred[keys]) or raw.duplicated(keys[:2]).any():raise ValueError('源主键变化')
    selected=raw.split.eq('test');panel=raw.loc[selected].reset_index(drop=True);pred=oldpred.loc[selected].reset_index(drop=True)
    actual=pd.read_parquet(folder/'predictions.parquet')
    if not actual[keys].equals(panel[keys]) or len(panel)!=416379:raise ValueError('候选身份不同')
    expected=reference_scores(panel,pred,plan['cases'])
    score_error=0.
    for c in plan['cases']:
        col=c['id']
        if not actual[col].isna().equals(expected[col].isna()):raise ValueError('分池范围不同')
        finite=expected[col].notna()
        error=float(np.max(np.abs(actual.loc[finite,col]-expected.loc[finite,col])))
        score_error=max(score_error,error)
        if error>1e-14:raise ValueError('独立效用秩组合复算不同')
    calendar=read(snapshot,'calendar.json');schedule=decision_dates(calendar,spec.test_start,spec.test_end,spec.frequency)
    if len(schedule)!=140 or set(schedule)!=set(panel.trade_date):raise ValueError('决策周变化')
    if any(execution<=decision for decision,execution in schedule.items()):raise ValueError('交易未在决策之后')
    max_account_error=0.;fills=0;target_rows=0;metrics_error=0.
    baseline=root/'runs'/next(a['run_id'] for a in read(folder,'reused_accounts.json') if a['case']=='small_cap')
    base_spec=read(baseline,'spec.json')
    indexed={a['case']:a for a in read(folder,'account_index.json')}
    for link in links['accounts']:
        account=root/'runs'/link['run_id'];account_spec=read(account,'spec.json')
        if account_spec['execution']!=base_spec['execution'] or account_spec['experiment']!=base_spec['experiment']:
            raise ValueError('账户费用、资金或日期定义不同')
        a=dict(account_spec['strategy']);b=dict(base_spec['strategy']);a.pop('name');b.pop('name')
        if a!=b:raise ValueError('账户容量/资格接口不同')
        rs=ResearchSpec.from_dict(account_spec)
        observed=pd.read_parquet(account/'decisions.parquet')
        target=reference_decisions(panel,expected[link['case']],schedule)
        # Original implementation preserves source row order by date/score/code.
        pd.testing.assert_frame_equal(observed.reset_index(drop=True),target.reset_index(drop=True),check_exact=False,atol=1e-14,rtol=0)
        target_rows+=len(target)
        if read(account,'schedule.json')!=schedule:raise ValueError('账户日历变化')
        saved=read(folder,'account_audit_'+link['case']+'.json')
        if saved['run_id']!=account.name or set(saved['scenarios'])!=set(plan['budget']['cost_scenarios']):raise ValueError('完整账户审计缺失')
        result=read(account,'result.json')
        if len(result['portfolios'])!=2:raise ValueError('费用情景缺失')
        for p in result['portfolios']:
            scenario=p['metrics']['scenario'];s=saved['scenarios'][scenario]
            max_account_error=max(max_account_error,max(s['daily_errors'].values()));fills+=s['fills']
            if p['metrics']['max_accounting_error']>=.001:raise ValueError('账户恒等式错误')
            eq=pd.read_parquet(account/scenario/'equity.parquet')
            idx=next(x for x in indexed[link['case']]['portfolios'] if x['metrics']['scenario']==scenario)
            if idx['metrics']!=p['metrics']:raise ValueError('索引账户指标不同')
            expected_stats=account_statistics(eq,rs.execution.initial_cash,spec.seed)
            if expected_stats!= {k:idx[k] for k in expected_stats}:raise ValueError('分年/路径指标不同')
    if max_account_error>=.001:raise ValueError('独立账户逐日核查误差超限')
    result=read(folder,'result.json')
    if result['new_fits']!=0 or result['new_model_predictions']!=0:raise ValueError('超出零fit预算')
    return dict(kind='ml',study='fixed_blend',run_id=folder.name,verified=True,rows=len(panel),decision_weeks=len(schedule),new_fits=0,new_model_predictions=0,
        accounts=8,account_scenarios=16,reused_control_accounts=4,reused_control_scenarios=8,checked_target_rows=target_rows,checked_fills=fills,
        rank_score_max_error=score_error,max_daily_account_error=max_account_error,
        verification_scope='全部新账户原始开盘价、费用、现金/股数逐日独立算术核查；源候选键及平均百分位组合、分池、选股全量独立复算；旧四控制仅核对身份及原保存账户核查，不重跑；非独立方法复核/非全研究复现/非供应商证明')
