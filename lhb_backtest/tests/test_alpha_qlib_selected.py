"""Contract tests use an explicit stub, not an alleged formula oracle.

The isolated OSS09 integration separately calls the real fixed Qlib provider.
"""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from src.alpharesearch.features.base import FeatureBlock
from src.alpharesearch.features.qlib_selected import FORMULAS, QLIB_VERSION, QLIB_WHEEL_SHA256, qlib_selected_features
from src.alpharesearch.contracts import Unit, MissingReason as M
from src.alpharesearch.registry import definitions_for_block, FeatureRegistry, materialization_receipt, verify_materialization
from src.alpharesearch.assembly import assemble_features


def fixture():
    dates = tuple(pd.bdate_range('2024-01-02', periods=25).strftime('%Y-%m-%d'))
    values = pd.DataFrame([dict(trade_date=d, stock_code=c, close=10+i+k*.1,
        observed_end=d+'T15:00:00+08:00', known_at=d+'T15:01:00+08:00')
        for k,c in enumerate(('000001.SZ','000002.SZ')) for i,d in enumerate(dates)])
    missing = values[['trade_date','stock_code']].copy();missing['close'] = M.PRESENT.value
    block = FeatureBlock(values,missing,{'close':Unit.PRICE.value},dict(schema='daily-observation-primitives-v1',
        key_columns=['trade_date','stock_code'],source_id='declared_quote',version_id='original_v1',vintage='market_observation'))
    def stub(code, expression, calendar):
        assert expression in {x[1] for x in FORMULAS} and calendar == dates
        close = block.values.loc[block.values.stock_code.eq(code)].sort_values('trade_date').close.to_numpy(dtype=float)
        return close.astype('<f4').astype(float) if expression == '$close' else np.zeros(len(dates))
    return block,dates,stub


def adapt(block=None,calendar=None,evaluator=None,**kwargs):
    b,c,e = fixture()
    return qlib_selected_features(b if block is None else block,c if calendar is None else calendar,e if evaluator is None else evaluator,
        backend_version=kwargs.pop('backend_version',QLIB_VERSION),backend_wheel_sha256=kwargs.pop('backend_wheel_sha256',QLIB_WHEEL_SHA256),**kwargs)


def test_strict_warmup_units_and_clock_resolution():
    b,c,e = fixture();out=adapt(b,c,e)
    assert len(out.values)==50 and len(out.units)==6
    assert out.units['close']==Unit.PRICE.value and out.units['slope20_to_close']==Unit.RATIO.value
    for code in ('000001.SZ','000002.SZ'):
        v=out.values.loc[out.values.stock_code.eq(code)];m=out.missing.loc[out.missing.stock_code.eq(code)]
        assert m.rsquare20.iloc[:19].eq(M.HISTORY.value).all() and v.rsquare20.iloc[:19].isna().all()
        assert m.close_lag5_ratio.iloc[:5].eq(M.HISTORY.value).all()
    assert out.values.known_at.dt.year.eq(2024).all()
    assert tuple(out.values.observed_end.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d').unique())==c


@pytest.mark.parametrize('names',[['Ref($close,-1)'],['Mean($close,19)'],['close','close'],[],['LABEL']])
def test_arbitrary_or_future_expressions_never_reach_backend(names):
    def forbidden(*args):raise AssertionError('rejected names must not call backend')
    with pytest.raises(ValueError,match='allowed'):adapt(evaluator=forbidden,selected_names=names)


@pytest.mark.parametrize('field,value',[('backend_version','0.9.8'),('backend_wheel_sha256','a'*64)])
def test_unverified_backend_identity_rejects(field,value):
    with pytest.raises(ValueError,match='tested'):adapt(**{field:value})


@pytest.mark.parametrize('kind',['reverse','duplicate','noncanonical','missing_date'])
def test_calendar_errors_reject(kind):
    b,c,e=fixture();c=list(c)
    if kind=='reverse':c.reverse()
    if kind=='duplicate':c.insert(1,c[0])
    if kind=='noncanonical':c[0]='2024-1-02'
    if kind=='missing_date':c.pop(3)
    with pytest.raises(ValueError):adapt(b,c,e)


def test_stock_grid_gap_rejects_before_skipping_a_session():
    b,c,e=fixture();b.values=b.values.drop(index=3).reset_index(drop=True);b.missing=b.missing.drop(index=3).reset_index(drop=True)
    with pytest.raises(ValueError,match='complete declared grid'):adapt(b,c,e)


@pytest.mark.parametrize('kind',['shifted_close','bool','complex','inf','wrong_shape'])
def test_misaligned_or_invalid_backend_values_reject(kind):
    b,c,e=fixture()
    def bad(code,expression,dates):
        v=e(code,expression,dates)
        if kind=='shifted_close':return np.roll(v,1)
        if kind=='bool':return np.ones(len(v),dtype=bool)
        if kind=='complex':return v.astype(complex)
        if kind=='inf':v[3]=np.inf
        if kind=='wrong_shape':return v[:,None]
        return v
    with pytest.raises(ValueError):adapt(b,c,bad)


def test_source_units_cannot_be_relabelled_by_adapter():
    b,c,e=fixture();b.units['close']=Unit.RATIO.value
    with pytest.raises(ValueError,match='CNY'):adapt(b,c,e)


def test_nan_source_is_explicit_incomplete_window_and_never_zero_filled():
    b,c,e=fixture();b.values.loc[10,'close']=np.nan;b.missing.loc[10,'close']=M.INVALID.value
    out=adapt(b,c,e);v=out.values.loc[out.values.stock_code.eq('000001.SZ')];m=out.missing.loc[out.missing.stock_code.eq('000001.SZ')]
    assert np.isnan(v.close.iloc[10]) and m.close.iloc[10]==M.INCOMPLETE.value
    assert v.mean20_to_close.iloc[19:].isna().all() and m.mean20_to_close.iloc[19:].eq(M.INCOMPLETE.value).all()
    assert out.values.loc[out.values.stock_code.eq('000002.SZ'),'mean20_to_close'].iloc[19:].notna().all()


def test_dependency_availability_is_propagated_and_decision_cutoff_rejects():
    b,c,e=fixture();b.values.loc[5,'known_at']=c[-1]+'T16:00:00+08:00'
    out=adapt(b,c,e);one=out.values.loc[out.values.trade_date.eq(c[19]) & out.values.stock_code.eq('000001.SZ')].iloc[0]
    assert one.known_at==pd.Timestamp(c[-1]+'T16:00:00+08:00').tz_convert('UTC')
    member=pd.DataFrame([dict(sample_id='decision',trade_date=c[19],stock_code='000001.SZ',decision_at=c[19]+'T15:10:00+08:00')])
    with pytest.raises(ValueError,match='available'):assemble_features(member,[out],universe_id='declared',block_ids={'qlib':'a'*64})
    close=adapt(b,c,e,selected_names=['close'])
    result=assemble_features(member,[close],universe_id='declared',block_ids={'qlib':'b'*64})
    assert result.block.values['qlib.close'].notna().all()


def test_local_observation_date_and_timezone_are_checked():
    b,c,e=fixture();b.values.loc[0,'observed_end']=c[1]+'T15:00:00+08:00';b.values.loc[0,'known_at']=c[1]+'T15:01:00+08:00'
    with pytest.raises(ValueError,match='local dates'):adapt(b,c,e)
    b,c,e=fixture();b.values.loc[0,'observed_end']=c[0]+'T15:00:00'
    with pytest.raises(ValueError,match='timezone'):adapt(b,c,e)


def test_missing_reason_mismatch_rejects_upstream():
    b,c,e=fixture();b.missing.loc[0,'close']=M.INVALID.value
    with pytest.raises(ValueError,match='missingness'):adapt(b,c,e)


def test_registry_and_existing_assembly_use_six_versioned_definitions():
    b,c,e=fixture();out=adapt(b,c,e);defs=definitions_for_block(out)
    registry=FeatureRegistry.from_dict(FeatureRegistry(defs).to_dict());assert len(registry.definitions)==6
    assert registry.resolve('qlib.close').unit==Unit.PRICE.value
    member=pd.DataFrame([dict(sample_id=code,trade_date=c[-1],stock_code=code,decision_at=c[-1]+'T15:10:00+08:00') for code in ('000001.SZ','000002.SZ')])
    assembled=assemble_features(member,[out],universe_id='declared',block_ids={'qlib':'a'*64})
    assert len(assembled.block.values)==2 and set(assembled.block.units)=={x.key for x in defs}
    receipt=materialization_receipt(out,defs,artifact_hashes={'values.parquet':'a'*64})
    verify_materialization(receipt)
    assert not receipt['historical_execution_certified'] and not receipt['independent_reproduction_claimed']


def test_weak_vintage_is_preserved_and_requires_existing_opt_in():
    b,c,e=fixture();b.metadata['vintage']='latest_snapshot_only';out=adapt(b,c,e)
    assert out.metadata['vintage']=='latest_snapshot_only'
    member=pd.DataFrame([dict(sample_id='d',trade_date=c[-1],stock_code='000001.SZ',decision_at=c[-1]+'T15:10:00+08:00')])
    with pytest.raises(ValueError,match='Weak'):assemble_features(member,[out],universe_id='declared',block_ids={'qlib':'a'*64})
