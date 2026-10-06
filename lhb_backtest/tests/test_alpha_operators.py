"""Hand-calculated full-calendar, membership, clock and missing-value checks."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.alpharesearch.contracts import MissingReason as M
from src.alpharesearch.dsl import Expr as E, compile_expression
from src.alpharesearch.features.base import FeatureBlock
from src.alpharesearch.registry import FeatureDefinition,FeatureRegistry
from src.alpharesearch.operators import ExpressionEvaluator,LeafBinding

DAYS=['2024-01-02','2024-01-03','2024-01-04','2024-01-05','2024-01-08']
STOCKS=['000001.SZ','000002.SZ','000003.SZ']


def fixture(*,values=None,membership=None,calendar=None,vintage='market_observation',future=False,domain='stock_day'):
    days=DAYS if calendar is None else calendar
    data=values or {STOCKS[0]:[1.,2.,3.,4.,5.],STOCKS[1]:[3.,3.,3.,3.,3.],STOCKS[2]:[10.,9.,8.,7.,6.]}
    rows=[dict(trade_date=d,stock_code=c,x=data[c][i]) for i,d in enumerate(days) for c in data] if domain=='stock_day' else [dict(trade_date=d,x=float(i+1)) for i,d in enumerate(days)]
    v=pd.DataFrame(rows);keys=['trade_date','stock_code'] if domain=='stock_day' else ['trade_date']
    v['observed_end']=pd.to_datetime(v.trade_date+'T15:00:00+08:00',utc=True);v['known_at']=v.observed_end+pd.Timedelta(minutes=1)
    if future:v.loc[0,'known_at']=pd.Timestamp('2024-01-02T16:00:00+08:00')
    missing=v[keys].copy();missing['x']=np.where(v.x.notna(),M.PRESENT.value,M.INVALID.value)
    block=FeatureBlock(v,missing,{'x':'ratio'},{'key_columns':keys,'source_id':'fixture','version_id':'v1','vintage':vintage}).validate()
    definition=FeatureDefinition('raw.x','ratio',domain,'hand input',('fixture',),'{}',(('fixture','a'*64),),'as-of','explicit missing')
    registry=FeatureRegistry([definition]);binding=LeafBinding(block,'x',definition.definition_id,'b'*64)
    p=pd.DataFrame([dict(sample_id=d+c,trade_date=d,stock_code=c,decision_at=d+'T15:10:00+08:00') for d in days for c in STOCKS]) if membership is None else membership
    clocks=pd.DataFrame({'trade_date':days,'decision_at':[d+'T15:10:00+08:00' for d in days]})
    return registry,{'raw.x':binding},dict(calendar=days,membership=p,decision_clocks=clocks,universe_id='hand_universe')


def run(expr,**kw):
    registry,leaves,context=fixture(**kw);return ExpressionEvaluator(registry,leaves,**context).evaluate(compile_expression(expr,registry))


def for_stock(block,code=STOCKS[0],reason=False):
    frame=block.missing if reason else block.values
    return frame.loc[frame.stock_code.eq(code),'score'].to_numpy()


def test_calendar_lag_does_not_skip_missing_stock_day():
    vals={STOCKS[0]:[1.,np.nan,3.,4.,5.]};block=run(E.call('lag',E.ref('raw.x'),periods=1),values=vals)
    np.testing.assert_allclose(for_stock(block),[np.nan,1.,np.nan,3.,4.],equal_nan=True)
    assert for_stock(block,reason=True)[0]==M.HISTORY.value and for_stock(block,reason=True)[2]==M.INVALID.value
    # Missing source row has a different reason than a reported invalid value.
    r,l,c=fixture();b=l['raw.x'].block;mask=~((b.values.trade_date==DAYS[1])&(b.values.stock_code==STOCKS[0]))
    reduced=FeatureBlock(b.values.loc[mask].reset_index(drop=True),b.missing.loc[mask].reset_index(drop=True),b.units,b.metadata)
    l['raw.x']=replace(l['raw.x'],block=reduced);out=ExpressionEvaluator(r,l,**c).evaluate(compile_expression(E.call('lag',E.ref('raw.x'),periods=1),r))
    assert for_stock(out,reason=True)[2]==M.UNCOVERED.value


@pytest.mark.parametrize('operator,expected',[
    ('ts_mean',[np.nan,np.nan,2.,3.,4.]),('ts_sum',[np.nan,np.nan,6.,9.,12.]),
    ('ts_std',[np.nan,np.nan,1.,1.,1.]),('ts_min',[np.nan,np.nan,1.,2.,3.]),
    ('ts_max',[np.nan,np.nan,3.,4.,5.]),('ts_rank',[np.nan,np.nan,1.,1.,1.])])
def test_full_window_values_match_hand_calculation(operator,expected):
    b=run(E.call(operator,E.ref('raw.x'),window=3));np.testing.assert_allclose(for_stock(b),expected,equal_nan=True)
    assert list(for_stock(b,reason=True)[:2])==[M.HISTORY.value]*2


def test_window_with_middle_gap_never_uses_partial_mean():
    b=run(E.call('ts_mean',E.ref('raw.x'),window=3),values={STOCKS[0]:[1.,np.nan,3.,4.,5.]})
    np.testing.assert_allclose(for_stock(b),[np.nan,np.nan,np.nan,np.nan,4.],equal_nan=True)
    assert for_stock(b,reason=True)[2]==M.INCOMPLETE.value


def test_cross_section_uses_current_membership_not_source_rows():
    r,l,c=fixture();c['membership']=c['membership'].loc[~c['membership'].stock_code.eq(STOCKS[2])].copy()
    result=ExpressionEvaluator(r,l,**c).evaluate(compile_expression(E.call('cs_rank',E.ref('raw.x')),r))
    np.testing.assert_allclose(for_stock(result),[.5,.5,.75,1.,1.])
    assert len(result.values)==10
    # Tied ranks are average rank divided by the number of valid members.
    np.testing.assert_allclose(for_stock(result,STOCKS[1]),[1.,1.,.75,.5,.5])


def test_cross_section_excludes_missing_and_flat_zscore_is_undefined():
    b=run(E.call('cs_rank',E.ref('raw.x')),values={STOCKS[0]:[1.]*5,STOCKS[1]:[np.nan]*5})
    assert (for_stock(b)==1).all() and pd.isna(for_stock(b,STOCKS[1])).all()
    z=run(E.call('cs_zscore',E.ref('raw.x')),values={c:[1.]*5 for c in STOCKS})
    assert z.values.score.isna().all() and z.missing.score.eq(M.UNDEFINED.value).all()


def test_known_after_decision_is_not_used_and_cannot_contaminate_prior_window():
    b=run(E.call('ts_mean',E.ref('raw.x'),window=2),future=True)
    assert pd.isna(for_stock(b)[1]) and for_stock(b,reason=True)[1]==M.INCOMPLETE.value
    leaf=run(E.ref('raw.x'),future=True)
    assert for_stock(leaf,reason=True)[0]==M.NOT_KNOWN.value


def test_where_does_not_inherit_unused_branch_missingness_and_missing_condition_is_unknown():
    expr=E.call('where',E.const(0),E.call('log',E.const(-1)),E.ref('raw.x'))
    b=run(expr);np.testing.assert_array_equal(for_stock(b),[1.,2.,3.,4.,5.])
    b=run(E.call('where',E.ref('raw.x'),E.const(9),E.const(4)),values={STOCKS[0]:[0.,1.,np.nan,-1.,0.]})
    np.testing.assert_allclose(for_stock(b),[4.,9.,np.nan,9.,4.],equal_nan=True)
    assert for_stock(b,reason=True)[2]==M.INVALID.value


@pytest.mark.parametrize('expr',[
    E.call('div',E.ref('raw.x'),E.const(0)),
    E.call('mul',E.ref('raw.x'),E.call('exp',E.const(10000))),
    E.call('log',E.call('neg',E.ref('raw.x'))),
])
def test_undefined_math_stays_missing_and_never_becomes_infinity(expr):
    b=run(expr);assert b.values.score.isna().all() and b.missing.score.eq(M.UNDEFINED.value).all()


def test_comparison_does_not_turn_nan_into_false_zero():
    b=run(E.call('gt',E.ref('raw.x'),E.const(0)),values={STOCKS[0]:[0.,-1.,np.nan,1.,2.]})
    np.testing.assert_allclose(for_stock(b),[0.,0.,np.nan,1.,1.],equal_nan=True)


def test_corr_cov_flat_and_rank_tie_hand_values():
    corr=run(E.call('ts_corr',E.ref('raw.x'),E.ref('raw.x'),window=3))
    np.testing.assert_allclose(for_stock(corr)[2:],[1.,1.,1.]);assert pd.isna(for_stock(corr,STOCKS[1])).all()
    cov=run(E.call('ts_cov',E.ref('raw.x'),E.ref('raw.x'),window=3))
    np.testing.assert_allclose(for_stock(cov)[2:],[1.,1.,1.]);np.testing.assert_allclose(for_stock(cov,STOCKS[1])[2:],[0.,0.,0.])
    rank=run(E.call('ts_rank',E.ref('raw.x'),window=3));np.testing.assert_allclose(for_stock(rank,STOCKS[1])[2:],[2/3]*3)


def test_market_broadcast_is_daily_scalar_for_every_member():
    r,l,c=fixture(domain='market_day');expr=E.call('broadcast',E.call('ts_mean',E.ref('raw.x'),window=2))
    b=ExpressionEvaluator(r,l,**c).evaluate(compile_expression(expr,r))
    for code in STOCKS:np.testing.assert_allclose(for_stock(b,code),[np.nan,1.5,2.5,3.5,4.5],equal_nan=True)


def test_future_prefix_membership_and_values_stay_stable_when_new_stock_is_added_later():
    r,l,c=fixture();p=c['membership'];c['membership']=p.loc[~(p.stock_code.eq(STOCKS[2])&p.trade_date.isin(DAYS[:3]))].copy()
    expr=E.call('cs_rank',E.call('ts_mean',E.ref('raw.x'),window=2));full=ExpressionEvaluator(r,l,**c).evaluate(compile_expression(expr,r))
    prior=dict(c,calendar=DAYS[:3],membership=c['membership'].loc[c['membership'].trade_date.isin(DAYS[:3])],decision_clocks=c['decision_clocks'].iloc[:3])
    short=ExpressionEvaluator(r,l,**prior).evaluate(compile_expression(expr,r));mask=full.values.trade_date.isin(DAYS[:3])
    pd.testing.assert_frame_equal(short.values,full.values.loc[mask].reset_index(drop=True));pd.testing.assert_frame_equal(short.missing,full.missing.loc[mask].reset_index(drop=True))


def test_weak_vintage_default_rejects_and_opt_in_remains_visible():
    r,l,c=fixture(vintage='latest_snapshot_only');compiled=compile_expression(E.ref('raw.x'),r)
    with pytest.raises(ValueError,match='vintage'):ExpressionEvaluator(r,l,**c).evaluate(compiled)
    b=ExpressionEvaluator(r,l,**c,allow_weak_vintage=True).evaluate(compiled)
    assert b.metadata['vintage']=='latest_snapshot_only' and not b.metadata['historical_execution_certified']


@pytest.mark.parametrize('case',['grid_budget','buffer_budget','membership_clock','missing_clock','duplicate_membership','duplicate_calendar','wrong_unit','wrong_version'])
def test_contract_and_resource_failures_are_explicit(case):
    r,l,c=fixture();options={}
    if case=='grid_budget':options['max_grid_cells']=1
    if case=='buffer_budget':options['max_estimated_buffer_bytes']=1
    if case=='membership_clock':c['membership'].loc[0,'decision_at']=DAYS[0]+'T14:00:00+08:00'
    if case=='missing_clock':c['decision_clocks']=c['decision_clocks'].iloc[1:]
    if case=='duplicate_membership':c['membership']=pd.concat([c['membership'],c['membership'].iloc[:1]])
    if case=='duplicate_calendar':c['calendar']=DAYS+[DAYS[-1]]
    if case=='wrong_unit':l['raw.x'].block.units['x']='log_return'
    if case=='wrong_version':l['raw.x']=replace(l['raw.x'],definition_id='c'*64)
    with pytest.raises(ValueError):ExpressionEvaluator(r,l,**c,**options).evaluate(compile_expression(E.ref('raw.x'),r))


def test_lag_zero_nested_delta_and_nonzero_signed_power():
    b=run(E.call('signed_power',E.call('delta',E.ref('raw.x'),periods=0),power=.5))
    assert b.values.score.eq(0).all()
    b=run(E.call('lag',E.ref('raw.x'),periods=0));np.testing.assert_array_equal(for_stock(b),[1,2,3,4,5])


def test_binary_rolling_constants_are_rejected_instead_of_ambiguous_daily_broadcast():
    r,_,_=fixture()
    for args in [(E.const(1),E.ref('raw.x')),(E.ref('raw.x'),E.const(1))]:
        with pytest.raises(ValueError):compile_expression(E.call('ts_corr',*args,window=3),r)
    # Saved proposals remain plain JSON and require the exact registered versions.
    assert isinstance(json.dumps(compile_expression(E.ref('raw.x'),r).to_dict()),str)
