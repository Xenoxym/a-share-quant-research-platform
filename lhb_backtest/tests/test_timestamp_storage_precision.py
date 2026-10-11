"""Same instants across timestamp units; any actual nanosecond difference fails."""
import pandas as pd
import pytest
from src.alpharesearch.contracts import same_timestamp_instants, timestamps
from src.alpharesearch.prepared_assembly import prepared_assembly_receipt, adapt_prepared_assembly
from src.alpharesearch.dsl import Expr, compile_expression
from src.alpharesearch.expression_assembly import assemble_declared_expression_blocks
from tests.test_prepared_assembly import case as source_case, reference, evaluator
from tests.test_expression_assembly import case as expression_case, assemble


def clock(unit='ns'):
    return pd.Series(pd.date_range('2024-01-05T07:30:00Z', periods=3, freq='7D')).dt.as_unit(unit)


@pytest.mark.parametrize('left,right',[('ns','us'),('us','ns'),('ms','ns')])
def test_same_instants_accept_different_storage_units_without_mutating(left,right):
    a,b=clock(left),clock(right);aa,bb=a.copy(),b.copy()
    assert str(a.dtype)!=str(b.dtype)
    assert not a.equals(b)
    assert same_timestamp_instants(a,b)
    pd.testing.assert_series_equal(a,aa);pd.testing.assert_series_equal(b,bb)


@pytest.mark.parametrize('offset',[-1,1])
def test_actual_single_nanosecond_difference_rejected(offset):
    a,b=clock('ns'),clock('us');a.iloc[1]+=pd.Timedelta(nanoseconds=offset)
    assert not same_timestamp_instants(a,b)


def test_timezone_normalizes_actual_instant_but_index_and_length_remain_strict():
    a=clock();b=a.dt.tz_convert('Asia/Shanghai')
    assert same_timestamp_instants(a,b)
    b.index=[3,4,5];assert not same_timestamp_instants(a,b)
    assert not same_timestamp_instants(a,a.iloc[:2])


@pytest.mark.parametrize('mutation',['naive','missing'])
def test_naive_and_missing_clocks_cannot_pass(mutation):
    a=clock();b=a.dt.tz_localize(None) if mutation=='naive' else a.copy()
    if mutation=='missing':b.iloc[0]=pd.NaT
    with pytest.raises(ValueError,match='timezone|missing'):
        same_timestamp_instants(a,b)


def mixed_prepared():
    block,registry,p,_,hashes,_=source_case()
    p['decision_at']=timestamps(p.decision_at).dt.as_unit('us')
    for field in ('known_at','observed_end'):
        block.values[field]=timestamps(block.values[field]).dt.as_unit('ns')
    source=reference(block,registry,hashes)
    receipt=prepared_assembly_receipt(block,registry,p,source,artifact_hashes=hashes)
    return block,registry,p,source,hashes,receipt


def test_prepared_leaf_to_expression_assembly_positive_real_code_mixed_resolution():
    values=mixed_prepared();block,registry,p,source,hashes,receipt=values
    adapted=adapt_prepared_assembly(block,registry,p,source,receipt,artifact_hashes=hashes)
    assert len(adapted.leaves)==3
    expression=compile_expression(Expr.ref('stock.p'),registry)
    result=evaluator(values).evaluate(expression)
    # Force output ns versus decisions us as a Parquet round trip may produce.
    for field in ('known_at','observed_end'):
        result.values[field]=timestamps(result.values[field]).dt.as_unit('ns')
    candidate=dict(candidate_id='synthetic_exact_time',family='synthetic',hypothesis='Synthetic precise clock compatibility',expression=expression.to_dict())
    assembled=assemble_declared_expression_blocks(p,[(candidate,result,{'values':'e'*64})],registry,
        universe_id='tiny_mainboard',provenance={'schema':'synthetic_only'},max_rows=4,max_matrix_cells=4)
    assert assembled.block.values['expr.synthetic_exact_time'].tolist()==[1.,1.,3.,4.]
    assert assembled.block.values[['sample_id','trade_date','stock_code']].equals(p[['sample_id','trade_date','stock_code']])


@pytest.mark.parametrize('offset',[-1,1])
def test_prepared_resigned_receipt_still_refuses_actual_ns_change(offset):
    values=mixed_prepared();block,registry,p,source,hashes,_=values
    for field in ('known_at','observed_end'):
        block.values.loc[0,field]+=pd.Timedelta(nanoseconds=offset)
    # Producing a fresh receipt cannot make an actually different instant legal.
    with pytest.raises(ValueError,match='conservative decision-time'):
        prepared_assembly_receipt(block,registry,p,source,artifact_hashes=hashes)


@pytest.mark.parametrize('offset',[0,-1,1])
def test_expression_same_ns_us_accepted_but_ns_difference_rejected(offset):
    p,registry,candidate,block=expression_case()
    p['decision_at']=timestamps(p.decision_at).dt.as_unit('us')
    for field in ('known_at','observed_end'):
        block.values[field]=timestamps(block.values[field]).dt.as_unit('ns')
        if offset:block.values.loc[0,field]+=pd.Timedelta(nanoseconds=offset)
    if offset:
        with pytest.raises(ValueError,match='conservative decision clock'):
            assemble(p,registry,[(candidate,block,{'values':'e'*64})])
    else:
        assert assemble(p,registry,[(candidate,block,{'values':'e'*64})]).block.missing['expr.synthetic_00'].tolist()==['present','source_uncovered','present']


# Object/ISO inputs use the generic parser and must retain all nine digits too.
@pytest.mark.parametrize('fraction', ['000000001', '123456789'])
def test_object_iso_offsets_preserve_exact_nanosecond_instants(fraction):
    from src.alpharesearch.contracts import timestamp
    local = '2024-01-05T15:10:00.' + fraction + '+08:00'
    utc = '2024-01-05T07:10:00.' + fraction + 'Z'
    left = pd.Series([local], dtype=object)
    right = pd.Series([utc], dtype=object)
    original = left.copy()
    assert timestamp(local).nanosecond == int(fraction[-3:])
    assert same_timestamp_instants(left, right)
    pd.testing.assert_series_equal(left, original)


@pytest.mark.parametrize('changed_fraction', ['123456788', '123456790'])
def test_object_iso_single_nanosecond_difference_is_not_truncated(changed_fraction):
    left = pd.Series(['2024-01-05T07:10:00.123456789Z'], dtype=object)
    right = pd.Series(['2024-01-05T07:10:00.' + changed_fraction + 'Z'], dtype=object)
    assert not same_timestamp_instants(left, right)


def test_mixed_aware_timestamp_datetime_and_iso_objects_keep_each_instant():
    from datetime import datetime, timezone
    mixed = pd.Series([pd.Timestamp('2024-01-05T07:10:00.123456789Z'),
                       datetime(2024, 1, 5, 7, 10, 0, 123456, tzinfo=timezone.utc),
                       '2024-01-05T15:10:00.987654321+08:00'], dtype=object)
    expected = pd.Series(pd.to_datetime(['2024-01-05T07:10:00.123456789Z',
                                        '2024-01-05T07:10:00.123456000Z',
                                        '2024-01-05T07:10:00.987654321Z'], utc=True))
    assert same_timestamp_instants(mixed, expected)


@pytest.mark.parametrize('invalid', [
    '2024-01-05T07:10:00.123456789', '2024-01-05', pd.NaT, None,
    '2024-01-05T07:10:00.1234567890Z',
    '2024-01-05T07:10:00.0000000001+08:00', 20240105, 1.5,
])
def test_object_timestamp_refuses_naive_missing_numeric_and_excess_precision(invalid):
    from src.alpharesearch.contracts import timestamp
    with pytest.raises(ValueError):
        timestamp(invalid)
    with pytest.raises(ValueError):
        timestamps(pd.Series([invalid], dtype=object))
