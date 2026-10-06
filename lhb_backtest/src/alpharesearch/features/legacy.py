"""Exact original16 representation adapter; no silent change of legacy semantics."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import FeatureBlock
from ..contracts import MissingReason as M, Unit, VintagePolicy, required_id
from ...mlresearch.contracts import FEATURES
from ...mlresearch.dataset import price_features

UNITS={name:(Unit.LOG_RETURN.value if name.startswith(('return_','volatility_')) else Unit.RATIO.value) for name in FEATURES}


@dataclass
class LegacyFeatures:
    block: FeatureBlock
    diagnostics: pd.DataFrame


def legacy_price_volume_features(bars, calendar, *, source_id, version_id,
                                vintage=VintagePolicy.MARKET, min_history=60,
                                available_delay_seconds=60):
    required_id(source_id,'source_id');required_id(version_id,'version_id');vintage=VintagePolicy(vintage).value
    if type(min_history) is not int or min_history<1:raise ValueError('Positive history length required')
    if type(available_delay_seconds) is not int or not 0<=available_delay_seconds<=86400:raise ValueError('Bounded daily availability delay required')
    calendar=list(calendar)
    if not calendar or calendar!=sorted(set(calendar)):raise ValueError('Canonical ordered unique market calendar required')
    dates=pd.to_datetime(calendar,format='%Y-%m-%d',errors='raise')
    if dates.strftime('%Y-%m-%d').tolist()!=calendar:raise ValueError('Canonical calendar dates required')
    required=['trade_date','stock_code','open','high','low','close','pre_close','volume','amount']
    if not set(required)<=set(bars):raise ValueError('Original16 daily inputs missing')
    b=bars[required].copy()
    if b.empty or b[['trade_date','stock_code']].isna().any().any() or b.duplicated(['trade_date','stock_code']).any():raise ValueError('Unique non-null daily keys required')
    if not b.trade_date.isin(calendar).all() or not b.stock_code.astype(str).str.fullmatch(r'\d{6}\.(SH|SZ|BJ)').all():raise ValueError('Quote keys disagree with declared calendar/stocks')
    for name in required[2:]:b[name]=pd.to_numeric(b[name],errors='coerce').astype(float)
    groups=[]
    for _,group in b.groupby('stock_code',sort=True):
        one=price_features(group,calendar,min_history=min_history)
        one['quote_row_count']=np.arange(1,len(one)+1)
        one['quote_gap_sessions']=one.session.diff().sub(1).clip(lower=0)
        groups.append(one)
    result=pd.concat(groups,ignore_index=True)
    keys=['trade_date','stock_code'];values=result[keys+list(FEATURES)].copy()
    local=pd.to_datetime(values.trade_date).dt.tz_localize('Asia/Shanghai')
    values['observed_end']=(local+pd.Timedelta(hours=15)).dt.tz_convert('UTC')
    values['known_at']=values.observed_end+pd.Timedelta(seconds=available_delay_seconds)
    missing=result[keys].copy()
    for name in FEATURES:
        good=np.isfinite(values[name])
        window=int(name.rsplit('_',1)[-1]) if name.startswith(('return_','volatility_','bias_')) else (20 if name in {'amount_ratio_5_20','volume_ratio_5_20','amount_volatility_20'} else 1)
        reason=np.where(result.quote_row_count.lt(window),M.HISTORY.value,M.INVALID.value)
        # Preserve the original finite outputs, including its stated compatibility exceptions.
        missing[name]=np.where(good,M.PRESENT.value,reason)
    diagnostics=result[keys+['history_valid','quote_row_count','quote_gap_sessions']].copy()
    block=FeatureBlock(values,missing,UNITS.copy(),{
        'schema':'legacy-price-volume16-v1','key_columns':keys,'source_id':source_id,'version_id':version_id,
        'vintage':vintage,'min_history':min_history,'available_delay_seconds':available_delay_seconds,
        'availability':'declared daily close+delay in Asia/Shanghai',
        'windows':'original stock quote-row rolling; missing market sessions are not inserted',
        'compatibility':'exact original16; flat range location=.5; cumulative index fills missing returns with0 internally',
        'history':'original contiguous min_history validity is separate diagnostic, not a hidden stock deletion',
        'missing':'preserve legacy finite values; unknowns remain missing, no output-level zero fill',
    }).validate()
    return LegacyFeatures(block,diagnostics)
