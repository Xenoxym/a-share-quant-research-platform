"""Declared-member market context proxies; calendar gaps remain missing."""
import numpy as np
import pandas as pd

from .base import FeatureBlock
from ..contracts import MissingReason as M, Unit, VintagePolicy, required_id


def market_context_features(bars, calendar, *, source_id, version_id, universe_id,
                            vintage, windows=(20,60), membership=None,
                            available_delay_seconds=60):
    """Cross-sectional mean log returns and breadth, then complete-session rolling.

    Supplied membership describes a market proxy, not a certified full exchange
    universe. If omitted, explicitly use all supplied date/stock quote keys.
    Calendar must be the bounded market slice, including genuine missing sessions.
    """
    for value,name in [(source_id,'source_id'),(version_id,'version_id'),(universe_id,'universe_id')]:required_id(value,name)
    vintage=VintagePolicy(vintage).value
    calendar=list(calendar)
    if not calendar or calendar!=sorted(set(calendar)) or not all(isinstance(x,str) and len(x)==10 for x in calendar):
        raise ValueError('Bounded canonical ordered unique calendar required')
    parsed=pd.to_datetime(calendar,format='%Y-%m-%d',errors='raise')
    if parsed.strftime('%Y-%m-%d').tolist()!=calendar:raise ValueError('Canonical calendar dates required')
    windows=tuple(windows)
    if not windows or len(set(windows))!=len(windows) or any(type(w) is not int or w<2 for w in windows):
        raise ValueError('Rolling windows must be unique integers >=2')
    if type(available_delay_seconds) is not int or available_delay_seconds<0:
        raise ValueError('Nonnegative declared market feed delay required')
    keys=['trade_date','stock_code'];required=set(keys)|{'close','pre_close','volume'}
    if not required<=set(bars):raise ValueError('Market quotes lack declared keys and fields')
    p=bars[list(required)].copy()
    members=p[keys].copy() if membership is None else membership[keys].copy()
    for frame in (p,members):
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():raise ValueError('Market keys must be unique and non-null')
        if not frame.trade_date.isin(calendar).all():raise ValueError('Market input falls outside bounded calendar')
        if not frame.stock_code.astype(str).str.fullmatch(r'\d{6}\.(SH|SZ|BJ)').all():raise ValueError('Exchange-qualified market stocks required')
    joined=members.merge(p,on=keys,how='left',validate='one_to_one',indicator=True)
    close=pd.to_numeric(joined.close,errors='coerce');pre=pd.to_numeric(joined.pre_close,errors='coerce');volume=pd.to_numeric(joined.volume,errors='coerce')
    valid=np.isfinite(close)&np.isfinite(pre)&np.isfinite(volume)&close.gt(0)&pre.gt(0)&volume.gt(0)
    ratio=close/pre.where(pre.gt(0))
    valid &= np.isfinite(ratio)&ratio.gt(0)
    joined['log_return']=np.log(ratio.where(valid));joined['up']=joined.log_return.gt(0).where(valid).astype(float)
    grouped=joined.groupby('trade_date')
    count=grouped.size().reindex(calendar,fill_value=0).astype(float)
    valid_count=grouped.log_return.count().reindex(calendar,fill_value=0).astype(float)
    daily=grouped.agg(daily_mean_log_return=('log_return','mean'),daily_breadth=('up','mean'),daily_cross_section_dispersion=('log_return','std')).reindex(calendar)
    values=pd.DataFrame({'trade_date':calendar})
    local=pd.Series(parsed).dt.tz_localize('Asia/Shanghai')
    values['observed_end']=(local+pd.Timedelta(hours=15)).dt.tz_convert('UTC')
    values['known_at']=values.observed_end+pd.Timedelta(seconds=available_delay_seconds)
    missing=values[['trade_date']].copy();units={}

    def put(name,raw,unit,why):
        raw=np.asarray(raw,dtype=float);good=np.isfinite(raw)
        values[name]=np.where(good,raw,np.nan);missing[name]=np.where(good,M.PRESENT.value,why);units[name]=unit.value

    empty=np.where(count.to_numpy()==0,M.UNCOVERED.value,M.INVALID.value)
    put('market_declared_members',count,Unit.COUNT,M.UNCOVERED.value)
    put('market_valid_members',valid_count,Unit.COUNT,M.UNCOVERED.value)
    put('market_valid_fraction',valid_count/count.where(count.gt(0)),Unit.RATIO,M.UNDEFINED.value)
    put('market_daily_mean_log_return',daily.daily_mean_log_return,Unit.LOG_RETURN,empty)
    put('market_daily_breadth',daily.daily_breadth,Unit.RATIO,empty)
    dispersion_reason=empty.copy();dispersion_reason[valid_count.eq(1).to_numpy()]=M.UNDEFINED.value
    put('market_daily_cross_section_dispersion',daily.daily_cross_section_dispersion,Unit.LOG_RETURN,dispersion_reason)
    for window in windows:
        why=np.where(np.arange(len(calendar))<window-1,M.HISTORY.value,M.INCOMPLETE.value)
        put('market_return_'+str(window),daily.daily_mean_log_return.rolling(window,min_periods=window).sum(),Unit.LOG_RETURN,why)
        put('market_volatility_'+str(window),daily.daily_mean_log_return.rolling(window,min_periods=window).std(ddof=1),Unit.LOG_RETURN,why)
        put('market_breadth_'+str(window),daily.daily_breadth.rolling(window,min_periods=window).mean(),Unit.RATIO,why)
    return FeatureBlock(values,missing,units,{
        'schema':'market-context-v1','key_columns':['trade_date'],'source_id':source_id,'version_id':version_id,
        'vintage':vintage,'universe_id':universe_id,'windows':list(windows),
        'membership':'explicit supplied membership' if membership is not None else 'all supplied quote keys',
        'return_proxy':'cross-sectional mean log(close/pre_close); rolling sum is not arithmetic equal-weight account return',
        'volume_filter':'finite positive volume; positive finite close and reference close',
        'coverage':'counts describe declared inputs; not an all-market completeness certificate',
        'availability':'declared daily close+delay in Asia/Shanghai; not measured original feed latency',
        'available_delay_seconds':available_delay_seconds,'missing':'missing sessions are not filled with zero; full rolling calendar required',
    }).validate()
