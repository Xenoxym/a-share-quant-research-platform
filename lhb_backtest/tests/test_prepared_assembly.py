"""Small semantic counterexamples for prepared leaves; never fit or account."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from src.alpharesearch.batch import AlphaBatchSpec
from src.alpharesearch.dsl import Expr, compile_expression
from src.alpharesearch.features.base import FeatureBlock
from src.alpharesearch.operators import ExpressionEvaluator
from src.alpharesearch.prepared_assembly import (
    adapt_prepared_assembly, prepared_assembly_receipt, PreparedAssemblyEvaluator,
    verify_pointwise_expression,
)
from src.alpharesearch.registry import FeatureDefinition, FeatureRegistry
from src.technical.artifacts import content_id, digest, write_json

KEYS = ['sample_id', 'trade_date', 'stock_code']


def definition(key, domain='stock_day', unit='ratio'):
    return FeatureDefinition(key, unit, domain, 'Explicit synthetic leaf', ('synthetic.source',),
        '{}', (('tests/test_prepared_assembly.py', 'a'*64),), 'conservative decision clock',
        'unknown is missing; no zero fill')


def reference(block, registry, hashes):
    payload = dict(schema='screen-assembly-reference-v1',
        artifacts=dict(assembly_values=hashes['values'], assembly_missing=hashes['missing'], assembly_definition=hashes['definition']),
        registry_version=registry.version_id, assembly_version=block.metadata['version_id'],
        source_bindings=block.metadata['source_bindings'], rows=len(block.values), upstream_execution_certified=False)
    return dict(payload, reference_id=content_id(payload))


def case():
    days = ['2024-01-05']*2 + ['2024-01-12']*2
    p = pd.DataFrame(dict(sample_id=['a','b','c','d'], trade_date=days,
        stock_code=['600001.SH','600002.SH']*2, decision_at=[d+'T15:30:00+08:00' for d in days]))
    values = p[KEYS].copy()
    values['stock.p'] = [1.,1.,3.,4.]
    values['stock.q'] = [2.,4.,6.,8.]
    values['market.m'] = [.2,.2,-.2,-.2]
    values['observed_end'] = p.decision_at
    values['known_at'] = p.decision_at
    missing = p[KEYS].copy()
    for key in ('stock.p','stock.q','market.m'):missing[key]='present'
    registry = FeatureRegistry((definition('stock.p'), definition('stock.q'), definition('market.m','market_day')))
    meta = dict(schema='assembled-feature-matrix-v1', key_columns=KEYS, source_id='synthetic_prepared',
        version_id='b'*64, vintage='market_observation', universe_id='tiny_mainboard', registry_version=registry.version_id,
        source_bindings=[dict(namespace='synthetic', definition_ids=[d.definition_id for d in registry.definitions])])
    block = FeatureBlock(values, missing, {k:'ratio' for k in ('stock.p','stock.q','market.m')}, meta).validate()
    hashes = {k:'c'*64 for k in ('values','missing','definition','registry','source_receipt','membership')}
    source = reference(block, registry, hashes)
    receipt = prepared_assembly_receipt(block,registry,p,source,artifact_hashes=hashes)
    return block,registry,p,source,hashes,receipt


def adapted(values):
    block,registry,p,source,hashes,receipt = values
    return adapt_prepared_assembly(block,registry,p,source,receipt,artifact_hashes=hashes)


def evaluator(values, cls=PreparedAssemblyEvaluator):
    block,registry,p,_,_,_ = values
    leaves = adapted(values).leaves
    days = pd.bdate_range('2024-01-02','2024-01-12').strftime('%Y-%m-%d').tolist()
    clocks = pd.DataFrame(dict(trade_date=days, decision_at=[d+'T15:30:00+08:00' for d in days]))
    return cls(registry,leaves,calendar=days,membership=p,decision_clocks=clocks,universe_id='tiny_mainboard')


def test_preserves_complete_definitions_clocks_and_shared_domains():
    values=case();block,registry,p,_,_,receipt=values;before=registry.to_dict()
    result=adapted(values)
    assert result.leaves['stock.p'].block is result.leaves['stock.q'].block is result.stock_block
    assert result.leaves['market.m'].block is result.market_block
    assert result.market_block.metadata['key_columns']==['trade_date']
    assert len(result.market_block.values)==2
    assert result.stock_block.values.known_at.tolist()==p.decision_at.tolist()
    assert registry.to_dict()==before and block.metadata['source_bindings']==result.stock_block.metadata['source_bindings']
    assert result.receipt['original_calculation_reproduced'] is False
    assert result.leaves['market.m'].definition_id==registry.resolve('market.m').definition_id
    with pytest.raises(ValueError,match='Alpha root'):
        compile_expression(Expr.ref('market.m'),registry)


def test_full_same_date_cross_section_ties_are_average():
    values=case();reg=values[1]
    rank=compile_expression(Expr.call('cs_rank',Expr.ref('stock.p')),reg)
    output=evaluator(values).evaluate(rank)
    assert output.values.score.tolist()==[.75,.75,.5,1.]
    expression=compile_expression(Expr.call('mul',Expr.ref('stock.q'),Expr.call('broadcast',Expr.ref('market.m'))),reg)
    assert evaluator(values).evaluate(expression).values.score.tolist()==pytest.approx([.4,.8,-1.2,-1.6])


@pytest.mark.parametrize('field',['known_at','observed_end'])
def test_future_or_rewritten_assembly_clock_refused(field):
    values=case();adapted(values)  # legal positive before mutation
    values[0].values.loc[0,field]='2024-01-05T15:31:00+08:00'
    # Keep the underlying FeatureBlock internally legal when moving observed_end.
    if field=='observed_end':values[0].values.loc[0,'known_at']='2024-01-05T15:31:00+08:00'
    with pytest.raises(ValueError,match='conservative decision-time'):
        adapted(values)


@pytest.mark.parametrize('mutation',['value','reason','source_hash','source_bindings'])
def test_receipt_rejects_coherent_source_content_changes(mutation):
    values=case();adapted(values);block,_,_,_,hashes,_=values
    if mutation=='value':block.values.loc[0,'stock.p']=99.
    elif mutation=='reason':
        block.values.loc[0,'stock.p']=np.nan;block.missing.loc[0,'stock.p']='source_uncovered'
    elif mutation=='source_hash':hashes['values']='d'*64
    else:block.metadata['source_bindings'][0]['source_note']='changed lineage'
    with pytest.raises(ValueError,match='receipt|source receipt'):
        adapted(values)


@pytest.mark.parametrize('field',['value','reason','availability'])
def test_market_day_conflicts_cannot_be_deduplicated(field):
    values=case();adapted(values);block,reg,p,_,hashes,_=values
    if field=='value':block.values.loc[1,'market.m']=.3
    elif field=='reason':
        block.values.loc[1,'market.m']=np.nan;block.missing.loc[1,'market.m']='source_uncovered'
    else:
        block.values.loc[1,['known_at','observed_end']]='2024-01-05T15:29:00+08:00'
    source=reference(block,reg,hashes)
    if field=='availability':
        with pytest.raises(ValueError,match='conservative decision-time'):
            prepared_assembly_receipt(block,reg,p,source,artifact_hashes=hashes)
    else:
        receipt=prepared_assembly_receipt(block,reg,p,source,artifact_hashes=hashes)
        with pytest.raises(ValueError,match='Market prepared values/reasons/availability disagree'):
            adapt_prepared_assembly(block,reg,p,source,receipt,artifact_hashes=hashes)


def test_unknown_lhb_like_value_remains_missing_not_numeric_zero():
    values=case();block,reg,p,_,hashes,_=values
    block.values.loc[0,'stock.p']=np.nan;block.missing.loc[0,'stock.p']='source_uncovered'
    values=block,reg,p,reference(block,reg,hashes),hashes,None
    receipt=prepared_assembly_receipt(*values[:4],artifact_hashes=hashes)
    values=(*values[:5],receipt)
    result=evaluator(values).evaluate(compile_expression(Expr.ref('stock.p'),reg))
    assert np.isnan(result.values.score.iloc[0]) and result.missing.score.iloc[0]=='source_uncovered'


@pytest.mark.parametrize('op,parameters',[('lag',dict(periods=0)),('delta',dict(periods=1)),('ts_mean',dict(window=2)),('ts_rank',dict(window=2))])
def test_weekly_sparse_prepared_source_forbids_all_temporal_nodes(op,parameters):
    values=case();reg=values[1];adapted(values)
    compiled=compile_expression(Expr.call(op,Expr.ref('stock.p'),**parameters),reg)
    # Legacy DSL can compile it; the new source gate, including lag0, refuses it.
    with pytest.raises(ValueError,match='forbid all lag/delta/TS'):
        verify_pointwise_expression(compiled,reg)
    with pytest.raises(ValueError,match='forbid all lag/delta/TS'):
        evaluator(values).evaluate(compiled)
    assert isinstance(evaluator(values,ExpressionEvaluator),ExpressionEvaluator)


def test_report_domain_cannot_be_relabelled_as_prepared_stock():
    values=case();adapted(values);block,reg,p,_,hashes,_=values
    new=FeatureRegistry(replace(d,domain='report') if d.key=='stock.p' else d for d in reg.definitions)
    block.metadata['registry_version']=new.version_id
    block.metadata['source_bindings'][0]['definition_ids']=[d.definition_id for d in new.definitions]
    with pytest.raises(ValueError,match='unit/domain differs'):
        prepared_assembly_receipt(block,new,p,reference(block,new,hashes),artifact_hashes=hashes)


def document(version, registry, materialization='e'*64):
    expression=compile_expression(Expr.ref('stock.p'),registry)
    doc=dict(schema=version,name='tiny_batch',dataset_id='tiny_dataset',universe_id='tiny_mainboard',
        ranges=dict(warmup_start='2024-01-02',development_start='2024-01-05',development_end='2024-01-12',
                    holdout_start='2024-01-15',holdout_end='2024-01-19'),
        candidates=[dict(candidate_id='tiny_candidate',hypothesis='Fixed synthetic signal',family='tiny_family',expression=expression.to_dict())],
        blocks=dict(prepared=dict(values='values',missing='missing',definition='definition',receipt='receipt')),
        bindings={'stock.p':dict(block='prepared',column='p',definition_id=registry.resolve('stock.p').definition_id,materialization_id=materialization)},
        budget=dict(max_candidates=64,max_input_bytes=1000000,max_grid_cells=1000,max_buffer_bytes=1000000,max_model_fits=0,max_accounts=0),
        allow_weak_vintage=False,evaluation_scope='retrospective_time_split',selection_rule='all_candidates_no_selection',
        primary_objectives=['account_return','max_drawdown'])
    if version!='alpha-feature-batch-v1':doc['storage_layout']='shared_index_v1'
    if version=='alpha-feature-batch-v3':
        doc.update(execution_partition='forecast_features',forecast_warmup_start='2024-01-15',
            training_selection=dict(receipt_role='train_receipt',summary_role='train_summary',library_role='train_library',source_receipt_role='train_source'))
    if version=='alpha-feature-batch-v4':
        doc.update(leaf_source='prepared_assembly_v1',temporal_policy='decision_date_pointwise_only')
        doc['blocks']['prepared']['source_receipt']='source_receipt';doc['bindings']['stock.p']['column']='stock.p'
    return doc


@pytest.mark.parametrize('version',['alpha-feature-batch-v1','alpha-feature-batch-v2','alpha-feature-batch-v3'])
def test_old_batch_contract_keeps_raw_meaning_and_refuses_hidden_adapter(version):
    registry=case()[1];doc=document(version,registry)
    assert AlphaBatchSpec.from_dict(doc,registry).to_dict()['bindings']['stock.p']['column']=='p'
    doc.update(leaf_source='prepared_assembly_v1',temporal_policy='decision_date_pointwise_only')
    with pytest.raises(ValueError,match='Strict alpha feature batch schema'):
        AlphaBatchSpec.from_dict(doc,registry)


def test_v4_inspection_routes_frozen_prepared_files_without_raw_definition_regeneration(tmp_path,monkeypatch):
    from src.researchops import alpha_experiments
    block,registry,p,_,_,_=case()
    folder=tmp_path/'research/tiny';folder.mkdir(parents=True)
    paths={role:folder/(role+'.json') for role in ('definition','registry','source_receipt','receipt','calendar')}
    for role in ('values','missing','membership','decision_clocks'):paths[role]=folder/(role+'.parquet')
    block.values.to_parquet(paths['values'],index=False);block.missing.to_parquet(paths['missing'],index=False)
    p.to_parquet(paths['membership'],index=False)
    write_json(paths['registry'],registry.to_dict());write_json(paths['definition'],dict(block.metadata,units=block.units))
    days=pd.bdate_range('2024-01-02','2024-01-19').strftime('%Y-%m-%d').tolist()
    write_json(paths['calendar'],dict(trade_dates=days))
    pd.DataFrame(dict(trade_date=days,decision_at=[d+'T15:30:00+08:00' for d in days])).to_parquet(paths['decision_clocks'],index=False)
    hashes={k:digest(paths[k]) for k in ('values','missing','definition','registry','membership')}
    source=reference(block,registry,hashes);write_json(paths['source_receipt'],source)
    hashes['source_receipt']=digest(paths['source_receipt'])
    receipt=prepared_assembly_receipt(block,registry,p,source,artifact_hashes=hashes);write_json(paths['receipt'],receipt)
    def forbidden(*args,**kwargs):raise AssertionError('Raw feature regeneration must not be called for prepared v4')
    monkeypatch.setattr(alpha_experiments,'definitions_for_block',forbidden)
    proposal=dict(spec=document('alpha-feature-batch-v4',registry,receipt['adapter_id']),
        inputs=[dict(role=role,path=path.relative_to(tmp_path).as_posix(),bytes=path.stat().st_size,sha256=digest(path)) for role,path in paths.items()])
    spec,_,info=alpha_experiments.inspect_inputs(tmp_path,proposal,include_evaluator=True)
    assert spec.to_dict()['schema']=='alpha-feature-batch-v4'
    assert info['temporal_policy']=='decision_date_pointwise_only' and info['original_raw_clocks_reconstructed'] is False
    assert isinstance(info['_evaluator'],PreparedAssemblyEvaluator)
    # Actual file mutation must be rejected before table loading or evaluation.
    paths['values'].write_bytes(paths['values'].read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Frozen input content changed'):
        alpha_experiments.inspect_inputs(tmp_path,proposal)
