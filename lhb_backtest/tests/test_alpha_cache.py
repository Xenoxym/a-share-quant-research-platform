"""Shared-expression equality and cache invalidation/integrity/resource failures."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.alpharesearch.cache import CachedExpressionEvaluator
from src.alpharesearch.contracts import timestamps
from src.alpharesearch.dsl import Expr as E,compile_expression
from src.alpharesearch.operators import ExpressionEvaluator
from tests.test_alpha_operators import fixture,DAYS,STOCKS


def evaluator(folder,setup=None,**kwargs):
    r,l,c=fixture() if setup is None else setup
    return CachedExpressionEvaluator(r,l,**c,cache_root=folder,**kwargs),r


def example():return E.call('cs_rank',E.call('ts_mean',E.ref('raw.x'),window=3))


def test_shared_subtree_cold_warm_and_plain_execution_match(tmp_path):
    e,r=evaluator(tmp_path/'cache');c=compile_expression(example(),r)
    plain_r,plain_l,plain_context=fixture();plain=ExpressionEvaluator(plain_r,plain_l,**plain_context).evaluate(c)
    cold=e.evaluate(c);computed=e.stats['computed_nodes'];warm=e.evaluate(c)
    pd.testing.assert_frame_equal(cold.values,plain.values);pd.testing.assert_frame_equal(cold.missing,plain.missing)
    pd.testing.assert_frame_equal(cold.values,warm.values);assert e.stats['computed_nodes']==computed and e.stats['memory_hits']>=1
    second,r=evaluator(tmp_path/'cache');disk=second.evaluate(c)
    pd.testing.assert_frame_equal(disk.values,cold.values);assert second.stats['disk_hits']==1 and second.stats['computed_nodes']==0
    assert cold.metadata['version_id']==disk.metadata['version_id'] and not disk.metadata['historical_execution_certified']


def test_multiple_formulas_share_a_real_rolling_subexpression(tmp_path):
    e,r=evaluator(tmp_path/'cache');mean=E.call('ts_mean',E.ref('raw.x'),window=3)
    cs=compile_expression(E.call('cs_rank',mean),r);nonlinear=compile_expression(E.call('mul',mean,mean),r)
    results=list(e.evaluate_many([cs,nonlinear],max_expressions=2))
    assert len(results)==2 and e.stats['memory_hits']>=2
    assert e.stats['computed_nodes']==4  # ref, rolling mean, rank, multiply.
    assert e.stats['disk_writes']==4


@pytest.mark.parametrize('mutation',['value','clock','source_version','materialization','membership','decision','calendar','engine'])
def test_complete_cache_key_changes_when_an_input_contract_changes(tmp_path,mutation):
    folder=tmp_path/'cache';first,r=evaluator(folder);c=compile_expression(example(),r);original=first.evaluate(c)
    setup=fixture();rr,l,context=setup
    if mutation=='value':l['raw.x'].block.values.loc[0,'x']=999.
    if mutation=='clock':l['raw.x'].block.values.loc[0,'known_at']=pd.Timestamp(DAYS[0]+'T15:02:00+08:00')
    if mutation=='source_version':l['raw.x'].block.metadata['version_id']='changed'
    if mutation=='materialization':l['raw.x']=replace(l['raw.x'],materialization_id='c'*64)
    if mutation=='membership':context['membership']=context['membership'].loc[~context['membership'].stock_code.eq(STOCKS[-1])]
    if mutation=='decision':
        context['membership']['decision_at']=context['membership'].trade_date+'T15:20:00+08:00';context['decision_clocks']['decision_at']=context['decision_clocks'].trade_date+'T15:20:00+08:00'
    if mutation=='calendar':
        context['calendar']=DAYS[1:];context['membership']=context['membership'].loc[context['membership'].trade_date.isin(DAYS[1:])];context['decision_clocks']=context['decision_clocks'].iloc[1:]
    second,_=evaluator(folder,setup)
    if mutation=='engine':second.engine=dict(second.engine,validation_fixture_change=True)
    changed=second.evaluate(c);assert second.stats['computed_nodes']>=1
    assert changed.metadata['version_id']!=original.metadata['version_id']


@pytest.mark.parametrize('mutation',['artifact','manifest','incomplete','shape','dtype','missing_reason','extra_archive'])
def test_corrupt_cache_is_rejected_instead_of_silently_recomputed(tmp_path,mutation):
    folder=tmp_path/'cache';e,r=evaluator(folder);c=compile_expression(E.ref('raw.x'),r);e.evaluate(c)
    entry=next(p for p in folder.iterdir() if p.is_dir());manifest_path=entry/'manifest.json';file=entry/'arrays.npz';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if mutation=='artifact':file.write_bytes(file.read_bytes()+b'corrupt')
    elif mutation=='manifest':manifest['request']['shape']=[999,999];manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    elif mutation=='incomplete':manifest_path.unlink()
    else:
        with np.load(file,allow_pickle=False) as z:v=z['values'];m=z['reasons']
        if mutation=='shape':v=v[:1];m=m[:1]
        if mutation=='dtype':v=v.astype('float32')
        if mutation=='missing_reason':m=m.copy();m[0,0]=255
        if mutation=='extra_archive':np.savez(file,values=v,reasons=m,unexpected=np.ones(1))
        else:np.savez(file,values=v,reasons=m)
        from src.technical.artifacts import digest
        manifest['artifact_sha256']=digest(file);manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    second,_=evaluator(folder)
    with pytest.raises(ValueError):second.evaluate(c)


def test_lru_and_disk_budget_preserve_correct_results_without_unbounded_storage(tmp_path):
    e,r=evaluator(tmp_path/'cache',memory_cache_bytes=150,disk_cache_bytes=1)
    c=compile_expression(example(),r);out=e.evaluate(c)
    assert len(out.values)==15 and e._memory_bytes<=150 and e.stats['memory_evictions']>0
    assert e.stats['disk_writes']==0 and e.stats['write_budget_skips']>0
    assert not list((tmp_path/'cache').glob('*/arrays.npz'))


def test_prepared_leaf_budget_and_batch_count_reject_before_computation(tmp_path):
    e,r=evaluator(tmp_path/'small',prepared_leaf_bytes=1);c=compile_expression(E.ref('raw.x'),r)
    with pytest.raises(ValueError,match='Prepared-leaf'):e.evaluate(c)
    e,r=evaluator(tmp_path/'cache')
    with pytest.raises(ValueError,match='count'):list(e.evaluate_many([c,c],max_expressions=1))
    assert e.stats['computed_nodes']==0


def test_owned_lock_busy_skips_write_and_old_artifacts_remain(tmp_path):
    folder=tmp_path/'cache';e,r=evaluator(folder);lock=folder/'.cache-write.lock';lock.write_text('another writer',encoding='ascii')
    b=e.evaluate(compile_expression(example(),r));assert len(b.values)==15 and e.stats['write_busy_skips']>0
    assert lock.read_text(encoding='ascii')=='another writer' and e.stats['disk_writes']==0


def test_prepared_snapshot_arrays_are_read_only_and_fixed_for_instance(tmp_path):
    e,r=evaluator(tmp_path/'cache');c=compile_expression(E.ref('raw.x'),r);one=e.evaluate(c)
    e.leaves['raw.x'].block.values.loc[0,'x']=999.
    two=e.evaluate(c);pd.testing.assert_frame_equal(one.values,two.values)
    arr=next(iter(e._leaf_cache.values()))
    with pytest.raises(ValueError):arr.values[0,0]=0.
    # A fresh instance hashes newly supplied values; no reuse of a false old ID.


def test_aware_vector_fast_path_matches_object_checks_and_rejects_nat_or_naive():
    aware=pd.Series(pd.to_datetime(['2024-01-02T15:00:00+08:00','2024-01-03T15:00:00+08:00']))
    pd.testing.assert_series_equal(timestamps(aware),timestamps(aware.astype(object)))
    for bad in [pd.Series(pd.to_datetime(['2024-01-02'])),pd.Series([pd.NaT],dtype='datetime64[ns, UTC]')]:
        with pytest.raises(ValueError):timestamps(bad)


def test_explicit_compilation_limits_survive_cache_subtree_verification(tmp_path):
    from src.alpharesearch.dsl import ExpressionLimits
    e,r=evaluator(tmp_path/'cache')
    compiled=compile_expression(E.call('lag',E.ref('raw.x'),periods=280),r,limits=ExpressionLimits(max_additional_history_sessions=300))
    out=e.evaluate(compiled);assert out.values.score.isna().all()
