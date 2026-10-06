"""Hand-checkable primitive values and non-executable expression counterexamples."""
import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.alpharesearch.contracts import MissingReason as M, Unit
from src.alpharesearch.features.primitives import daily_primitives
from src.alpharesearch.registry import FeatureDefinition, FeatureRegistry, definitions_for_block
from src.alpharesearch.dsl import Expr as E, ExpressionLimits, compile_expression, verify_compiled


def bars():
    return pd.DataFrame([dict(trade_date=d,stock_code=c,open=10.,high=11.,low=9.,close=10.5,
        pre_close=10.,volume=1000.,amount=10500.,high_limit=11.,low_limit=9.,limit_price_valid=True)
        for d in ('2024-01-03','2024-01-04') for c in ('000001.SZ','600001.SH')])


def primitive(b=None):
    return daily_primitives(bars() if b is None else b,source_id='quotes',version_id='q1')


def registry():
    block=primitive();r=FeatureRegistry(definitions_for_block(block))
    for key,domain in [('market.context','market_day'),('report.concentration','report')]:
        r.add(FeatureDefinition(key,Unit.RATIO.value,domain,'test definition',('declared source',),'{}',
            (('fixture','a'*64),),'declared clock','missing remains missing'))
    return r


def compile_(expr,**kw):return compile_expression(expr,registry(),**kw)


def example():
    falling=E.call('cs_rank',E.call('neg',E.call('delta',E.ref('daily.close'),periods=5)))
    volume=E.call('cs_rank',E.call('div',E.ref('daily.volume'),E.call('ts_mean',E.ref('daily.volume'),window=20)))
    return E.call('mul',falling,volume)


def test_daily_primitive_units_and_raw_values_with_future_field_ignored():
    b=bars();b['future_5d_return']=999.;block=primitive(b)
    assert len(block.units)==11 and 'future_5d_return' not in block.values
    assert block.units['close']==Unit.PRICE.value and block.units['amount']==Unit.AMOUNT.value
    assert block.values.iloc[0].volume==1000. and block.values.iloc[0].amount==10500.
    assert str(block.values.iloc[0].known_at)=='2024-01-03 07:01:00+00:00'
    assert len(definitions_for_block(block))==11


def test_zero_volume_preserves_reported_price_and_exposes_nontraded_observation():
    b=bars();b.loc[0,['volume','amount']]=0.;block=primitive(b)
    assert block.values.iloc[0].close==10.5 and block.values.iloc[0].volume==0
    assert block.values.iloc[0].positive_volume_observed==0
    assert block.missing.iloc[0].volume==M.PRESENT.value


@pytest.mark.parametrize('field,value',[('low',12.),('open',-1.),('close',np.inf)])
def test_invalid_joint_ohlc_does_not_survive_in_formula(field,value):
    b=bars();b.loc[0,field]=value;block=primitive(b)
    assert block.values.iloc[0][['open','high','low','close']].isna().all()
    assert block.missing.iloc[0].close==M.INVALID.value


def test_band_source_flag_and_band_numbers_are_distinct_checks():
    b=bars();b.loc[0,'limit_price_valid']=False;b.loc[1,'high_limit']=10.
    block=primitive(b)
    assert block.values.iloc[0].source_limit_valid==0 and pd.isna(block.values.iloc[0].high_limit)
    assert block.missing.iloc[0].high_limit==M.STATUS.value
    assert block.values.iloc[1].source_limit_valid==1 and block.missing.iloc[1].high_limit==M.INVALID.value


@pytest.mark.parametrize('mutation',['numeric_flag','duplicate','bad_date','wrong_currency','negative_amount'])
def test_primitive_malformed_input_rejects_or_is_explicitly_missing(mutation):
    b=bars()
    if mutation=='negative_amount':
        b.loc[0,'amount']=-1;block=primitive(b)
        assert block.missing.iloc[0].amount==M.INVALID.value;return
    if mutation=='numeric_flag':b['limit_price_valid']=1
    if mutation=='duplicate':b=pd.concat([b,b.iloc[:1]])
    if mutation=='bad_date':b.loc[0,'trade_date']='20240103'
    if mutation=='wrong_currency':b.loc[0,'stock_code']='AAPL.US'
    with pytest.raises(ValueError):primitive(b)


def test_example_binds_versions_domains_dimensions_and_additional_history():
    r=registry();result=compile_expression(example(),r)
    assert result.dimension==(0,0,0) and result.domain=='stock_day'
    assert result.additional_history_sessions==19 and len(result.dependencies)==2
    assert {r.resolve('daily.close').definition_id,r.resolve('daily.volume').definition_id}==set(result.dependencies)
    value=json.loads(json.dumps(result.to_dict()));assert verify_compiled(value,r)==result
    assert compile_expression(E.from_dict(example().to_dict()),r).expression_id==result.expression_id


def test_price_times_shares_is_money_and_division_is_dimensionless():
    result=compile_(E.call('mul',E.ref('daily.close'),E.ref('daily.volume')),require_alpha=False)
    assert result.dimension==(1,0,0)
    assert compile_(E.call('div',E.ref('daily.close'),E.ref('daily.pre_close'))).dimension==(0,0,0)


@pytest.mark.parametrize('expr',[
    E.call('add',E.ref('daily.close'),E.ref('daily.amount')),
    E.call('gt',E.ref('daily.close'),E.const(5)),
    E.call('log',E.ref('daily.close')),
    E.call('exp',E.ref('daily.volume')),
    E.call('where',E.const(1),E.ref('daily.close'),E.ref('daily.amount')),
    E.ref('daily.close'),E.const(1),E.ref('market.context'),E.ref('report.concentration'),
    E.call('ts_mean',E.ref('report.concentration'),window=5),
    E.call('mul',E.ref('daily.positive_volume_observed'),E.ref('market.context')),
])
def test_invalid_dimension_domain_or_alpha_root_rejects(expr):
    with pytest.raises(ValueError):compile_(expr)


def test_market_context_requires_explicit_broadcast_and_where_retains_unit():
    assert compile_(E.call('broadcast',E.ref('market.context'))).domain=='stock_day'
    expr=E.call('where',E.call('gt',E.ref('daily.close'),E.const(10,Unit.PRICE)),
        E.ref('daily.close'),E.ref('daily.pre_close'))
    assert compile_(expr,require_alpha=False).dimension==(1,-1,0)
    with pytest.raises(ValueError):compile_(E.call('broadcast',E.ref('daily.close')))


@pytest.mark.parametrize('periods',[-1,True,1.5,'1'])
def test_future_or_noninteger_lag_rejects(periods):
    with pytest.raises(ValueError):compile_(E.call('lag',E.ref('daily.volume'),periods=periods),require_alpha=False)


def test_nested_windows_add_history_and_enforce_budget():
    expr=E.call('ts_rank',E.call('delta',E.ref('daily.close'),periods=5),window=20)
    assert compile_(expr).additional_history_sessions==24
    with pytest.raises(ValueError,match='historical'):compile_(expr,limits=ExpressionLimits(max_additional_history_sessions=23))


@pytest.mark.parametrize('limits',[ExpressionLimits(max_nodes=2),ExpressionLimits(max_depth=2),ExpressionLimits(max_constant_magnitude=1)])
def test_complexity_and_literal_budgets_reject(limits):
    expr=E.call('add',E.ref('daily.positive_volume_observed'),E.call('neg',E.const(2)))
    with pytest.raises(ValueError):compile_(expr,limits=limits)


def test_json_is_structured_not_executable_and_operator_arity_is_strict():
    with pytest.raises(ValueError):E.from_dict("__import__('os').system('bad')")
    with pytest.raises(ValueError):E.call('__import__',E.const(1))
    with pytest.raises(ValueError):compile_(E.call('mul',E.ref('daily.close')))
    with pytest.raises(ValueError):compile_(E.call('cs_rank',E.ref('daily.close'),window=2))
    with pytest.raises(ValueError):E.from_dict(example().to_dict(),max_nodes=2)
    with pytest.raises(ValueError):compile_(E.ref("__import__('os')"))


def test_ambiguous_reference_requires_definition_id_and_registry_changes_reject_saved_proposal():
    r=registry();old=r.resolve('daily.close');compiled=compile_expression(example(),r)
    new=replace(old,parameters_json='{"version":2}');r.add(new)
    with pytest.raises(ValueError,match='ambiguous'):compile_expression(example(),r)
    bound=E.call('cs_rank',E.ref('daily.close',old.definition_id))
    assert compile_expression(bound,r).dependencies==(old.definition_id,)
    with pytest.raises(ValueError,match='changed'):verify_compiled(compiled.to_dict(),r)


@pytest.mark.parametrize('field,value',[('dimension',[1,0,0]),('expression_id','b'*64),('additional_history_sessions',0),('nodes',1),('purpose','strategy_success')])
def test_saved_contract_corruption_rejects(field,value):
    r=registry();saved=copy.deepcopy(compile_expression(example(),r).to_dict());saved[field]=value
    with pytest.raises(ValueError):verify_compiled(saved,r)


def test_rank_normalizes_price_and_covariance_keeps_physical_product():
    assert compile_(E.call('cs_rank',E.ref('daily.close'))).dimension==(0,0,0)
    corr=compile_(E.call('ts_corr',E.ref('daily.close'),E.ref('daily.volume'),window=5))
    assert corr.additional_history_sessions==4
    cov=compile_(E.call('ts_cov',E.ref('daily.close'),E.ref('daily.volume'),window=5),require_alpha=False)
    assert cov.dimension==(1,0,0)
