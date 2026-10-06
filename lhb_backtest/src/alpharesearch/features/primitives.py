"""White-listed unadjusted A-share daily observations for formula research."""
import numpy as np
import pandas as pd

from .base import FeatureBlock
from ..contracts import MissingReason as M, Unit, VintagePolicy, required_id

UNITS={**{n:Unit.PRICE.value for n in ('open','high','low','close','pre_close','high_limit','low_limit')},
       'volume':Unit.SHARES.value,'amount':Unit.AMOUNT.value,
       'source_limit_valid':Unit.BINARY.value,'positive_volume_observed':Unit.BINARY.value}


def daily_primitives(bars, *, source_id, version_id, vintage=VintagePolicy.MARKET,
                     available_delay_seconds=60):
    """Reported values, not an asserted tick feed or executed-trade opportunity.

    Zero-volume prices may be carried by a vendor; the separate positive-volume
    flag exposes that fact. Quote clocks are a declared after-close research rule.
    """
    required_id(source_id,'source_id');required_id(version_id,'version_id');vintage=VintagePolicy(vintage).value
    if type(available_delay_seconds) is not int or not 0<=available_delay_seconds<=86400:raise ValueError('Bounded daily availability delay required')
    fields=['trade_date','stock_code','open','high','low','close','pre_close','volume','amount','high_limit','low_limit','limit_price_valid']
    if not set(fields)<=set(bars):raise ValueError('Raw daily primitive inputs are missing')
    b=bars[fields].copy().sort_values(['trade_date','stock_code']).reset_index(drop=True)
    keys=['trade_date','stock_code']
    if b.empty or b[keys].isna().any().any() or b.duplicated(keys).any():raise ValueError('Unique non-null daily keys required')
    if not b.trade_date.astype(str).str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():raise ValueError('Canonical daily dates required')
    pd.to_datetime(b.trade_date,format='%Y-%m-%d',errors='raise')
    if not b.stock_code.astype(str).str.fullmatch(r'(?:60\d{4}|68\d{4})\.SH|(?:00\d{4}|30\d{4})\.SZ|(?:43|83|87|92)\d{4}\.BJ').all():raise ValueError('Declared CNY A-share source required')
    for name in ('open','high','low','close','pre_close','volume','amount','high_limit','low_limit'):
        b[name]=pd.to_numeric(b[name],errors='coerce').astype(float)
    flags=b.limit_price_valid
    if not flags.map(lambda x: pd.isna(x) or isinstance(x,(bool,np.bool_))).all():raise ValueError('Actual Boolean limit validity required')
    flag_known=flags.notna();limit_valid=flags.fillna(False).astype(bool)
    ohlc=b[['open','high','low','close']]
    price_valid=np.isfinite(ohlc).all(axis=1)&ohlc.gt(0).all(axis=1)&b.high.ge(b[['open','close']].max(axis=1))&b.low.le(b[['open','close']].min(axis=1))&b.high.ge(b.low)
    band=limit_valid&np.isfinite(b.high_limit)&np.isfinite(b.low_limit)&b.low_limit.gt(0)&b.high_limit.gt(b.low_limit)&price_valid&b.high.le(b.high_limit+1e-8)&b.low.ge(b.low_limit-1e-8)
    values=b[keys].copy();missing=b[keys].copy()
    for name in UNITS:
        if name=='source_limit_valid':raw=limit_valid.astype(float);good=flag_known;why=M.STATUS.value
        elif name=='positive_volume_observed':raw=b.volume.gt(0).astype(float);good=np.isfinite(b.volume)&b.volume.ge(0);why=M.INVALID.value
        elif name in ('high_limit','low_limit'):
            raw=b[name];good=band;why=np.where(limit_valid,M.INVALID.value,M.STATUS.value)
        else:
            raw=b[name]
            good=price_valid if name in ('open','high','low','close') else (np.isfinite(raw)&(raw.gt(0) if name=='pre_close' else raw.ge(0)))
            why=M.INVALID.value
        values[name]=raw.where(good);missing[name]=np.where(good,M.PRESENT.value,why)
    local=pd.to_datetime(values.trade_date).dt.tz_localize('Asia/Shanghai')
    values['observed_end']=(local+pd.Timedelta(hours=15)).dt.tz_convert('UTC');values['known_at']=values.observed_end+pd.Timedelta(seconds=available_delay_seconds)
    return FeatureBlock(values,missing,UNITS.copy(),{
        'schema':'daily-observation-primitives-v1','key_columns':keys,'source_id':source_id,'version_id':version_id,'vintage':vintage,
        'frequency':'daily','adjustment':'unadjusted reported prices; close/lag(close) may contain corporate-action gaps',
        'volume_unit':'shares as declared by upstream ingestion; no new independent vendor-unit audit',
        'availability':'declared daily close+delay in Asia/Shanghai','available_delay_seconds':available_delay_seconds,
        'missing':'invalid values unknown; valid volume/amount zero retained; unknown limits never reconstructed from current names',
        'zero_volume':'reported OHLC may be carried quotes; positive-volume observation flag provided; not proof of tradability',
    }).validate()
