"""Causal market-state inference and implementable component target portfolios.

Training uses past data only; trading uses a forward filter, never Viterbi or
full-sequence smoothed posterior probabilities. States describe observations;
the state-to-allocation map is an explicit hypothesis, not a discovered oracle.
"""
from dataclasses import replace
import warnings
import numpy as np
import pandas as pd

FEATURES = ['trend63', 'log_vol20', 'breadth20', 'downside5']
POLICIES = {
    'small': ('小市值原型', '100%小市值'),
    'small_half': ('小市值半仓＋现金', '50%小市值，50%现金；固定减仓对照'),
    'defensive': ('低波动原型', '100%低波动'),
    'fixed_mix': ('固定各半组合', '50%小市值，50%低波动；不择时'),
    'vol_budget': ('简单波动率预算', '小市值比例=min(1, 15%/市场代理20日年化波动)，其余现金'),
    'observable': ('可解释状态切换', '市场63日趋势<0且20日上涨广度<40%时持低波动，否则持小市值'),
    'hmm2_mix': ('HMM两状态软切换', '小市值比例=1−高波动状态概率，低波动比例=高波动状态概率'),
    'hmm2_cash': ('HMM两状态风险预算', '小市值比例=1−0.75×高波动状态概率，其余现金'),
    'hmm3_mix': ('HMM三状态软切换', '小市值比例=P(低波动状态)+0.5×P(中波动状态)，其余低波动'),
    'hmm2_lag5': ('HMM信号延迟五日对照', '与两状态软切换相同，但故意使用五个交易日前的概率'),
}


def definition(spec):
    s=spec.strategy; name, rule=POLICIES[s.allocation_policy]
    if s.allocation_policy=='observable':
        rule=f'市场63日趋势<0且20日上涨广度<{s.state_breadth_threshold:.0%}时持低波动，否则持小市值'
    return dict(family='allocation', idea=name+'：在已经定义的小市值和低波动分支之间配置资金。',
        raw_data='独立主板历史价量、证券状态；季度已披露股本与已发生送转用于选股',
        feature_description='63日市场趋势、20日市场波动、20日上涨广度、5日平均大跌比例',
        formula=rule, signal=rule, score='先独立构造两个分支，再按配置比例合并同一股票的目标权重',
        target='单股目标权重=小市值分支原权重×小市值配置比例+低波动分支原权重×低波动配置比例；同股合并，不足名额不重新归一。按决策日权益及原始收盘价取整至100股。',
        ranking=f'每个分支最多{s.top_k}只；重叠股票合并权重，账户最多{2*s.top_k}只；总目标配置上限{s.allocation:.0%}。',
        filters='两个分支采用共同252日历史、非ST、上市状态、流动性与已公布财报资格；小市值按桥接后估算市值，低波动按63日波动。',
        availability='每年用此前最多756个交易日重新拟合（至少252日），标准化也只用训练段；评估日逐日forward filter。决策在收盘后，下一交易日开始执行。',
        capacity='单个分支100只时合并最多200只；交易仍受现金、整手、停牌、涨跌停和参与率约束，所有对照采用同一容量上限。',
        parameter_origin=f'固定2/3状态、对角高斯分布、种子11/29按训练似然选择。15%波动目标及状态配置映射为固定研究设定；本次广度阈值{s.state_breadth_threshold:.0%}。首轮为40%，后续修改属于单独实验，不代表原始预登记。',
        source='https://hmmlearn.readthedocs.io/en/stable/tutorial.html',
        limitations='HMM状态按训练均值中的波动从低到高排序，不叫已识别牛熊；年度重估可能改变状态含义。高斯、对角协方差和重叠窗口均为近似。状态能描述风险不等于能预测哪个策略胜出。所有历史已被研究者查看。季度股本与供应商修订偏差仍在；分红扣准备金后计应收、不复投；送转股按除权日到账是假设，不参与配股。市场冲击和未成交机会成本未独立估计。')


def market_features(raw, calendar, metadata):
    b=raw.sort_values(['stock_code','trade_date']).merge(metadata[['stock_code','listed_date','de_listed_date']],on='stock_code',validate='many_to_one')
    b['session']=b.trade_date.map({d:i for i,d in enumerate(calendar)})
    valid=b.close.gt(0)&b.pre_close.gt(0)&np.isfinite(b.close)&np.isfinite(b.pre_close)
    b['r']=np.log((b.close/b.pre_close).where(valid))
    g=b.groupby('stock_code',sort=False)
    b['r20']=g.r.transform(lambda x:x.rolling(20,min_periods=20).sum())
    continuous=b.session.sub(g.session.shift(19)).eq(19)
    usable=continuous&b.r20.notna()&b.is_st.eq(0)&b.listed_date.le(b.trade_date)&b.de_listed_date.gt(b.trade_date)
    b=b.loc[usable].copy()
    b['up20']=b.r20.gt(0);b['down5pct']=b.r.lt(np.log(.95))
    daily=b.groupby('trade_date').agg(market_log_return=('r','mean'),breadth20=('up20','mean'),down_fraction=('down5pct','mean'),universe_count=('stock_code','size')).reindex(calendar)
    # This is a descriptive equal-weight daily market proxy, not a tradable index.
    daily['market_nav']=np.exp(daily.market_log_return.fillna(0).cumsum())
    daily['trend63']=daily.market_log_return.rolling(63,min_periods=63).sum()
    daily['vol20']=daily.market_log_return.rolling(20,min_periods=20).std(ddof=1)*np.sqrt(252)
    daily['log_vol20']=np.log(daily.vol20.clip(lower=.001))
    daily['downside5']=daily.down_fraction.rolling(5,min_periods=5).mean()
    daily.index.name='date'
    return daily.reset_index().loc[lambda x:np.isfinite(x[FEATURES]).all(axis=1)].reset_index(drop=True)


def gaussian_filter(x, start, transition, means, variances, previous=None):
    """P(z_t | x_1..t); previous is the final filtered posterior at t-1."""
    x=np.asarray(x,float); probs=[]; prior=np.asarray(start,float) if previous is None else np.asarray(previous,float)@transition
    for row in x:
        loglike=-.5*(np.log(2*np.pi*variances)+(row-means)**2/variances).sum(axis=1)
        logpost=np.log(np.maximum(prior,1e-300))+loglike
        post=np.exp(logpost-logpost.max());post/=post.sum()
        probs.append(post);prior=post@transition
    return np.asarray(probs).reshape(len(x),len(start))


def infer_states(frame, states=2, start='2022-01-01', min_train=252, max_train=756):
    try:
        from hmmlearn.hmm import GaussianHMM
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise ValueError('状态研究需要项目 research 依赖：hmmlearn 与 scikit-learn') from exc
    f=frame.sort_values('date').reset_index(drop=True); outputs=[]; models=[]
    years=sorted(f.loc[f.date.ge(start),'date'].str[:4].unique())
    for year in years:
        test=f.loc[f.date.str[:4].eq(year)]
        train=f.loc[f.date.lt(year+'-01-01')].tail(max_train)
        if len(train)<min_train:
            raise ValueError(f'{year}之前只有{len(train)}个有效训练日，需要{min_train}')
        center=train[FEATURES].mean().to_numpy();scale=train[FEATURES].std(ddof=0).clip(lower=1e-8).to_numpy()
        x=(train[FEATURES].to_numpy()-center)/scale; fits=[]
        with threadpool_limits(limits=1):
            for seed in (11,29):
                m=GaussianHMM(n_components=states,covariance_type='diag',n_iter=150,tol=.01,min_covar=.001,random_state=seed,implementation='scaling')
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');m.fit(x)
                ll=float(m.score(x))
                if np.isfinite(ll):fits.append((ll,seed,m,[str(w.message) for w in caught]))
        if not fits:raise ValueError('所有HMM初始化失败')
        ll,seed,m,warn=max(fits,key=lambda v:v[0])
        order=np.argsort(m.means_[:,FEATURES.index('log_vol20')],kind='stable')
        means=m.means_[order];var=np.maximum(np.diagonal(m.covars_,axis1=1,axis2=2)[order],1e-8)
        pi=m.startprob_[order];a=m.transmat_[np.ix_(order,order)]
        previous=gaussian_filter(x,pi,a,means,var)[-1]
        probs=gaussian_filter((test[FEATURES].to_numpy()-center)/scale,pi,a,means,var,previous)
        out=test[['date']].copy()
        for j in range(states):out[f'p{j}']=probs[:,j]
        out['training_end']=train.date.iloc[-1];out['training_start']=train.date.iloc[0];out['model_year']=year
        outputs.append(out.loc[out.date.ge(start)])
        history=list(m.monitor_.history)
        models.append(dict(year=year,states=states,training_start=train.date.iloc[0],training_end=train.date.iloc[-1],training_rows=len(train),
            seed=seed,training_log_likelihood=ll,iterations=m.monitor_.iter,likelihood_increment=float(history[-1]-history[-2]) if len(history)>1 else None,
            initializations=[dict(seed=z[1],log_likelihood=z[0]) for z in fits],center=center.tolist(),scale=scale.tolist(),
            means=means.tolist(),variances=var.tolist(),transition=a.tolist(),initial=pi.tolist(),means_raw=(means*scale+center).tolist(),warnings=warn))
    return pd.concat(outputs,ignore_index=True),models


def allocation_weights(frame, policy, breadth_threshold=.4):
    f=frame.copy();n=len(f)
    p=f.get('hmm2_p1',pd.Series(np.nan,index=f.index))
    if policy=='small':small=np.ones(n);defensive=np.zeros(n)
    elif policy=='small_half':small=np.full(n,.5);defensive=np.zeros(n)
    elif policy=='defensive':small=np.zeros(n);defensive=np.ones(n)
    elif policy=='fixed_mix':small=defensive=np.full(n,.5)
    elif policy=='vol_budget':small=np.minimum(1.,.15/f.vol20);defensive=np.zeros(n)
    elif policy=='observable':small=(~(f.trend63.lt(0)&f.breadth20.lt(breadth_threshold))).astype(float);defensive=1-small
    elif policy=='hmm2_cash':small=1-.75*p;defensive=np.zeros(n)
    elif policy in {'hmm2_mix','hmm2_lag5'}:
        used=p if policy=='hmm2_mix' else p.shift(5).fillna(.5)
        small=1-used;defensive=used
    elif policy=='hmm3_mix':small=f.hmm3_p0+.5*f.hmm3_p1;defensive=1-small
    else:raise ValueError('未知配置规则')
    f['small_fraction']=small;f['defensive_fraction']=defensive;f['cash_fraction']=1-f.small_fraction-f.defensive_fraction
    values=f[['small_fraction','defensive_fraction','cash_fraction']]
    if not np.isfinite(values).all().all() or (values< -1e-8).any().any() or (values>1+1e-8).any().any():
        raise ValueError('缺状态概率或配置权重非法')
    return f


def combine_decisions(small, defensive, weights, allocation=.98):
    """Merge overlapping securities once; retain both component selection flags."""
    keys=['stock_code','trade_date'];s=small.loc[small.selected].copy();d=defensive.loc[defensive.selected].copy()
    # Original target weights preserve missing-name cash; no re-normalization.
    s['small_weight']=s.target_weight;d['defensive_weight']=d.target_weight
    base=pd.concat([s,d],ignore_index=True).drop_duplicates(keys,keep='first').drop(columns=['target_weight','rank','selected','small_weight','defensive_weight'])
    base=base.merge(s[keys+['small_weight']],on=keys,how='left').merge(d[keys+['defensive_weight']],on=keys,how='left')
    base[['small_weight','defensive_weight']]=base[['small_weight','defensive_weight']].fillna(0.)
    w=weights.rename(columns={'date':'trade_date'})
    base=base.merge(w,on='trade_date',how='left',validate='many_to_one')
    if base[['small_fraction','defensive_fraction']].isna().any().any():
        raise ValueError('调仓日缺少当时可得的配置权重')
    base['target_weight']=base.small_weight*base.small_fraction+base.defensive_weight*base.defensive_fraction
    base['selected']=base.target_weight.gt(1e-10)
    base['reason']=np.where(base.selected,'eligible','allocation_zero')
    base=base.sort_values(['trade_date','target_weight','stock_code'],ascending=[True,False,True])
    base['rank']=base.groupby('trade_date').cumcount()+1
    base['score']=base.target_weight
    totals=base.groupby('trade_date').target_weight.sum()
    if totals.gt(allocation+1e-8).any():raise AssertionError('合并配置超限')
    return base.reset_index(drop=True)


def prepare_components(snapshot, spec):
    import json
    from .signals import features, decisions
    from .market import Market
    calendar=json.loads((snapshot/'calendar.json').read_text(encoding='utf-8'))
    raw=pd.read_parquet(snapshot/'bars.parquet');meta=pd.read_parquet(snapshot/'metadata.parquet')
    filings=pd.read_parquet(snapshot/'fundamentals.parquet');actions=pd.read_parquet(snapshot/'actions.parquet')
    shared=dict(min_history=252,require_fundamentals=True,share_basis='known_bonus',trend_window=0,min_score=None,size_floor_quantile=0.,positive_profit=False)
    ss=replace(spec,strategy=replace(spec.strategy,family='size',lookback=252,skip=21,**shared))
    ds=replace(spec,strategy=replace(spec.strategy,family='low_volatility',lookback=63,skip=0,**shared))
    bars=features(raw,calendar,ds.strategy)
    sm=json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'))
    market=Market(bars,calendar,actions,pd.read_parquet(snapshot/'status.parquet'),meta,sm['missing_action_files'])
    # Reuse price columns and the 63-day volatility; size doesn't use momentum.
    small,schedule=decisions(bars,calendar,meta,ss,filings,None,actions)
    defensive,_=decisions(bars,calendar,meta,ds,filings,None,actions)
    mf=market_features(raw,calendar,meta)
    return small,defensive,schedule,market,mf


def prepare_states(mf, start):
    output=mf.loc[mf.date.ge(start)].copy();models={}
    for k in (2,3):
        state,models[str(k)]=infer_states(mf,k,start)
        state=state.rename(columns={c:f'hmm{k}_{c}' for c in state.columns if c!='date'})
        output=output.merge(state,on='date',validate='one_to_one')
    return output,models
