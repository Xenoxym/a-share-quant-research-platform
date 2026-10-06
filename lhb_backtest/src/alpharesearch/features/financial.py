"""Decision-time annual/quarterly financial proxies with explicit publication lineage."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import FeatureBlock
from ..contracts import MissingReason as M, Unit, VintagePolicy, timestamp, required_id
from ...technical.artifacts import content_id, jsonable

FIELDS = ['total_shares', 'np_parent_company_owners', 'total_shareholder_equity', 'total_assets']
UNITS = {
    'reported_shares_proxy': Unit.SHARES.value, 'reported_market_cap_proxy': Unit.AMOUNT.value,
    'annual_parent_profit': Unit.AMOUNT.value, 'annual_total_equity': Unit.AMOUNT.value,
    'annual_total_assets': Unit.AMOUNT.value,
    'annual_ep': Unit.RATIO.value, 'annual_bp': Unit.RATIO.value,
    'annual_profitability_proxy': Unit.RATIO.value,
    'quarter_period_age_days': Unit.COUNT.value, 'quarter_publication_age_days': Unit.COUNT.value,
    'annual_period_age_days': Unit.COUNT.value, 'annual_publication_age_days': Unit.COUNT.value,
    'quarter_stale': Unit.BINARY.value, 'annual_stale': Unit.BINARY.value,
}
KEYS = ['sample_id', 'trade_date', 'stock_code']


@dataclass
class FinancialFeatures:
    block: FeatureBlock
    lineage: pd.DataFrame


def _dates(frame, name):
    if frame[name].isna().any() or not frame[name].astype(str).str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
        raise ValueError('Canonical dates required: ' + name)
    pd.to_datetime(frame[name], format='%Y-%m-%d', errors='raise')


def _stocks(frame):
    if frame.stock_code.isna().any() or not frame.stock_code.astype(str).str.fullmatch(r'\d{6}\.(SH|SZ|BJ)').all():
        raise ValueError('Exchange-qualified stock codes required')
    frame['stock_code'] = frame.stock_code.astype(str)


def financial_features(decisions, filings, *, source_id, version_id, quote_source_id,
                       quote_version_id, vintage, quote_vintage=VintagePolicy.MARKET,
                       allow_weak_vintage=False, max_share_age_days=240, max_annual_age_days=550):
    """Keep every decision row; select latest known period and its known revision.

    Required quote clocks are explicit. Optional publication_at is an exact aware
    timestamp; absent values use the next civil midnight after publication_date.
    Declared source vintages cannot be established by date filtering alone.
    """
    for value,name in [(source_id,'source_id'),(version_id,'version_id'),(quote_source_id,'quote_source_id'),(quote_version_id,'quote_version_id')]:
        required_id(value,name)
    vintage=VintagePolicy(vintage);quote_vintage=VintagePolicy(quote_vintage)
    strong={VintagePolicy.HISTORICAL, VintagePolicy.MARKET}
    combined=vintage if vintage not in strong else quote_vintage
    for age in (max_share_age_days,max_annual_age_days):
        if type(age) is not int or age < 1:raise ValueError('Staleness days must be positive integers')
    required=set(KEYS)|{'decision_at','close','quote_observed_end','quote_known_at'}
    filing_fields={'stock_code','report_date','publication_date',*FIELDS}
    if not required <= set(decisions) or not filing_fields <= set(filings):
        raise ValueError('Financial inputs lack keys, prices, clocks or filing fields')
    p=decisions[list(required)].copy().reset_index(drop=True)
    f=filings[list(filing_fields)+(['publication_at'] if 'publication_at' in filings else [])].copy()
    _stocks(p);_stocks(f)
    for frame,name in [(p,'trade_date'),(f,'report_date'),(f,'publication_date')]:_dates(frame,name)
    if p.sample_id.isna().any() or p.sample_id.duplicated().any() or p.duplicated(['trade_date','stock_code']).any():
        raise ValueError('Decision keys must be unique and non-null')
    if not f.publication_date.ge(f.report_date).all():raise ValueError('Publication predates fiscal period end')
    if not f.report_date.str[-5:].isin(['03-31','06-30','09-30','12-31']).all():
        raise ValueError('This package requires declared calendar-quarter fiscal periods')
    p['decision_at']=pd.to_datetime(p.decision_at.map(timestamp),utc=True).astype('datetime64[ns, UTC]')
    if not p.decision_at.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d').eq(p.trade_date).all():
        raise ValueError('Decision differs from its local trading date')
    for name in ('quote_observed_end','quote_known_at'):
        p[name]=pd.to_datetime(p[name].map(lambda x: None if pd.isna(x) else timestamp(x)),utc=True).astype('datetime64[ns, UTC]')
    price=pd.to_numeric(p.close,errors='coerce')
    price_valid=np.isfinite(price)&price.gt(0)&p.quote_known_at.notna()&p.quote_observed_end.notna()
    known=p.quote_known_at.notna();observed=p.quote_observed_end.notna()
    if not known.eq(observed).all() or (known & (p.quote_known_at.lt(p.quote_observed_end)|p.quote_known_at.gt(p.decision_at))).any():
        raise ValueError('Quote clocks are missing or conflict with decision time')
    if (observed & ~p.quote_observed_end.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d').eq(p.trade_date)).any():
        raise ValueError('Current-close quote must be observed on the decision date')
    f['observed_end']=(pd.to_datetime(f.report_date).dt.tz_localize('Asia/Shanghai')+pd.Timedelta(hours=23,minutes=59,seconds=59)).dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
    fallback=(pd.to_datetime(f.publication_date).dt.tz_localize('Asia/Shanghai')+pd.Timedelta(days=1)).dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
    if 'publication_at' in f:
        exact=pd.to_datetime(f.publication_at.map(lambda x: None if pd.isna(x) else timestamp(x)),utc=True).astype('datetime64[ns, UTC]')
        if (exact.notna() & ~exact.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d').eq(f.publication_date)).any():
            raise ValueError('Exact publication differs from declared publication date')
        f['known_at']=exact.fillna(fallback);f['publication_precision']=np.where(exact.notna(),'timestamp','date')
    else:f['known_at']=fallback;f['publication_precision']='date'
    if f.known_at.lt(f.observed_end).any():raise ValueError('Filing is known before the fiscal period ends')
    if f.duplicated(['stock_code','report_date','known_at']).any():
        raise ValueError('Ambiguous same-time filing versions; preserve source and resolve externally')
    for name in FIELDS:f[name]=pd.to_numeric(f[name],errors='coerce').astype(float)
    # Identity uses this record only, not a future rolling success statistic.
    f['filing_id']=[content_id(jsonable(list(row))) for row in f[['stock_code','report_date','publication_date','known_at']+FIELDS].itertuples(index=False,name=None)]
    p['_row']=range(len(p))

    def attach(frame,prefix):
        frame=frame.sort_values(['stock_code','known_at','report_date','filing_id']).copy()
        frame['_period']=pd.to_datetime(frame.report_date).astype('datetime64[ns]').astype('int64')
        # Future late releases of older periods cannot replace newer known periods.
        frame=frame.loc[frame._period.eq(frame.groupby('stock_code')._period.cummax())]
        frame=frame.drop_duplicates(['stock_code','known_at'],keep='last')
        names=['stock_code','report_date','publication_date','known_at','observed_end','filing_id','publication_precision']+FIELDS
        right=frame[names].rename(columns={x:prefix+x for x in names if x!='stock_code'})
        return pd.merge_asof(p.sort_values('decision_at'),right.sort_values(prefix+'known_at'),
            left_on='decision_at',right_on=prefix+'known_at',by='stock_code',direction='backward',allow_exact_matches=True).sort_values('_row').reset_index(drop=True)

    q=attach(f,'q_');a=attach(f.loc[f.report_date.str.endswith('12-31')],'a_')
    q_exists=q.q_filing_id.notna();a_exists=a.a_filing_id.notna()
    if not allow_weak_vintage and ((vintage not in strong and (q_exists.any() or a_exists.any())) or (quote_vintage not in strong and (q_exists & price_valid).any())):
        raise ValueError('Weak historical financial/quote vintage requires explicit opt-in')
    dates=pd.to_datetime(p.trade_date)
    q_age=(dates-pd.to_datetime(q.q_report_date)).dt.days.astype(float)
    a_age=(dates-pd.to_datetime(a.a_report_date)).dt.days.astype(float)
    q_valid=q_exists&q_age.between(0,max_share_age_days);a_valid=a_exists&a_age.between(0,max_annual_age_days)
    shares=q.q_total_shares;profit=a.a_np_parent_company_owners;equity=a.a_total_shareholder_equity;assets=a.a_total_assets
    shares_good=q_valid&np.isfinite(shares)&shares.gt(0)
    cap=price*shares;cap_good=shares_good&price_valid&np.isfinite(cap)&cap.gt(0)
    values=p[KEYS].copy();values['observed_end']=p.decision_at;values['known_at']=p.decision_at
    missing=p[KEYS].copy()

    def put(name,raw,good,exists,age_valid,undefined=None):
        raw=pd.Series(raw,index=p.index,dtype=float);good=pd.Series(good,index=p.index,dtype=bool)&np.isfinite(raw)
        why=np.full(len(p),M.INVALID.value,dtype=object)
        why[(exists & ~age_valid).to_numpy()]=M.STALE.value
        why[~exists.to_numpy()]=M.UNCOVERED.value
        if undefined is not None:why[undefined.to_numpy()]=M.UNDEFINED.value
        values[name]=raw.where(good);missing[name]=np.where(good,M.PRESENT.value,why)

    put('reported_shares_proxy',shares,shares_good,q_exists,q_valid)
    put('reported_market_cap_proxy',cap,cap_good,q_exists,q_valid)
    for name,raw,positive in [('annual_parent_profit',profit,False),('annual_total_equity',equity,False),('annual_total_assets',assets,True)]:
        good=a_valid&np.isfinite(raw)&(raw.ge(0) if positive else True);put(name,raw,good,a_exists,a_valid)
    both=q_exists&a_exists;fresh=q_valid&a_valid
    for name,numerator in [('annual_ep',profit),('annual_bp',equity)]:
        good=a_valid&np.isfinite(numerator)&cap_good
        put(name,numerator/cap.where(cap_good),good,both,fresh)
    profit_good=a_valid&np.isfinite(profit)&np.isfinite(equity)
    put('annual_profitability_proxy',profit/equity.where(equity.gt(0)),profit_good&equity.gt(0),a_exists,a_valid,profit_good&equity.le(0))
    for prefix,joined,exists,age,fresh_flag in [('quarter',q,q_exists,q_age,q_valid),('annual',a,a_exists,a_age,a_valid)]:
        rawprefix='q_' if prefix=='quarter' else 'a_'
        publication_age=(dates-pd.to_datetime(joined[rawprefix+'publication_date'])).dt.days.astype(float)
        put(prefix+'_period_age_days',age,exists,exists,exists)
        put(prefix+'_publication_age_days',publication_age,exists,exists,exists)
        put(prefix+'_stale',~fresh_flag,exists,exists,exists)
    lineage=pd.concat([p[KEYS+['quote_observed_end','quote_known_at']].copy(),q[[x for x in q if x.startswith('q_')]],a[[x for x in a if x.startswith('a_')]]],axis=1)
    block=FeatureBlock(values,missing,UNITS.copy(),{
        'schema':'financial-proxies-v1','key_columns':KEYS.copy(),'source_id':source_id,'version_id':version_id,
        'vintage':combined.value,'filing_vintage':vintage.value,'quote_vintage':quote_vintage.value,
        'quote_source_id':quote_source_id,'quote_version_id':quote_version_id,
        'weak_vintage_allowed':allow_weak_vintage,'max_share_age_days':max_share_age_days,'max_annual_age_days':max_annual_age_days,
        'amounts':'CNY; shares in shares as declared by upstream ingestion',
        'annual':'December fiscal period; parent profit / total equity is a proxy, not exact parent ROE or TTM',
        'size':'current unadjusted close times latest already-published reported shares; no action bridge',
        'clock':'decision-time transformation; selected publication/quote clocks retained in lineage',
        'missing':'no current-company fundamentals or blanket zero; missing filings never delete decisions',
    }).validate()
    return FinancialFeatures(block,lineage)
