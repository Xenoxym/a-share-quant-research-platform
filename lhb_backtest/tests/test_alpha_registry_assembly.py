"""Definition identity, honest provenance and whole-membership matrix counterexamples."""
from copy import deepcopy
from dataclasses import replace
import json

import pandas as pd
import pytest

from src.alpharesearch.assembly import assemble_features
from src.alpharesearch.contracts import MissingReason as M
from src.alpharesearch.features.legacy import legacy_price_volume_features
from src.alpharesearch.features.market import market_context_features
from src.alpharesearch.registry import (
    FeatureDefinition, FeatureRegistry, definitions_for_block, implementation_hashes,
    coverage_diagnostics, materialization_receipt, verify_materialization,
)
from src.mlresearch.dataset import price_features
from src.mlresearch.contracts import FEATURES

HASH='a'*64


def fixture():
    cal=pd.bdate_range('2024-01-02',periods=65).strftime('%Y-%m-%d').tolist()
    bars=pd.DataFrame([dict(trade_date=d,stock_code=c,open=10.,high=11.,low=9.,close=10.,pre_close=10.,volume=100.,amount=1000.)
        for d in cal for c in ('000001.SZ','000002.SZ')])
    return bars,cal


def legacy(b=None,cal=None):
    bars,days=fixture()
    return legacy_price_volume_features(bars if b is None else b,days if cal is None else cal,source_id='quote',version_id='q1')


def definition(**kw):
    args=dict(key='test.signal',unit='ratio',domain='stock_day',description='Hand-calculated test signal',dependencies=('raw.close',),
        parameters_json='{"window":20,"skip":1}',implementation_hashes=(('test.py',HASH),),timing_rule='After daily close',missing_rule='Explicit missing reasons')
    args.update(kw);return FeatureDefinition(**args)


def membership(cal):
    return pd.DataFrame([dict(sample_id=d+c,trade_date=d,stock_code=c,decision_at=d+'T15:10:00+08:00')
        for d in cal[-3:] for c in ('000001.SZ','000002.SZ','000003.SZ')])


def assembled(p=None,blocks=None,ids=None,**kw):
    _,cal=fixture();return assemble_features(membership(cal) if p is None else p,[legacy().block] if blocks is None else blocks,
        universe_id='all_declared',block_ids={'pv16':HASH} if ids is None else ids,**kw)


def test_definition_parameter_order_does_not_change_identity_and_versions_are_explicit():
    one=definition();same=definition(parameters_json='{"skip":1,"window":20}')
    assert one.definition_id==same.definition_id
    two=replace(one,parameters_json='{"skip":1,"window":60}')
    registry=FeatureRegistry([one,same,two]);assert len(registry.definitions)==2
    with pytest.raises(ValueError,match='ambiguous'):registry.resolve('test.signal')
    assert registry.resolve('test.signal',one.definition_id)==one
    loaded=FeatureRegistry.from_dict(json.loads(json.dumps(registry.to_dict())))
    assert loaded.version_id==registry.version_id


@pytest.mark.parametrize('field,value',[('key','return_5'),('unit','probability'),('domain','future_label'),('dependencies',()),('parameters_json','{"x":NaN}'),('implementation_hashes',()),('implementation_hashes',(('a.py','not_hash'),))])
def test_bad_feature_definition_contracts_reject(field,value):
    with pytest.raises(ValueError):definition(**{field:value})


def test_definition_and_registry_mutation_are_detected():
    registry=FeatureRegistry([definition()]);one=registry.to_dict();one['definitions'][0]['unit']='binary'
    with pytest.raises(ValueError):FeatureRegistry.from_dict(one)
    two=registry.to_dict();two['version_id']='b'*64
    with pytest.raises(ValueError):FeatureRegistry.from_dict(two)


def test_implementation_paths_are_bounded_and_source_hashes_stable():
    hashes=implementation_hashes(['src/alpharesearch/contracts.py'])
    assert len(hashes)==1 and len(hashes[0][1])==64
    with pytest.raises(ValueError):implementation_hashes(['../secret.py'])
    with pytest.raises(ValueError):implementation_hashes(['src/not_python.json'])


def test_original16_matches_old_outputs_without_labels_or_hidden_stock_filter():
    bars,cal=fixture();new=legacy(bars,cal)
    old=pd.concat([price_features(group,cal) for _,group in bars.groupby('stock_code',sort=True)],ignore_index=True)
    pd.testing.assert_frame_equal(new.block.values[['trade_date','stock_code']+list(FEATURES)],old[['trade_date','stock_code']+list(FEATURES)])
    assert len(new.block.values)==len(bars) and 'label_return' not in new.block.values
    assert new.diagnostics.history_valid.sum()==12


def test_legacy_flat_range_and_gap_compatibility_are_visible_not_redefined():
    bars,cal=fixture();bars=bars.loc[bars.trade_date.ne(cal[10])].copy();bars.loc[bars.trade_date.eq(cal[-1]),['open','high','low','close','pre_close']]=10.
    result=legacy(bars,cal);last=result.block.values.trade_date.eq(cal[-1])
    assert result.block.values.loc[last,'close_location'].eq(.5).all()
    assert result.diagnostics.quote_gap_sessions.max()==1
    assert 'missing market sessions are not inserted' in result.block.metadata['windows']
    defs=definitions_for_block(result.block);assert len(defs)==16
    assert all('Legacy quote-row' in d.limitations[0] for d in defs)


def test_legacy_and_registry_future_append_preserve_past_values_and_definition():
    bars,cal=fixture();one=legacy(bars.loc[bars.trade_date.le(cal[-2])],cal);two=legacy(bars,cal)
    pd.testing.assert_frame_equal(one.block.values,two.block.values.loc[two.block.values.trade_date.le(cal[-2])].reset_index(drop=True))
    assert [d.definition_id for d in definitions_for_block(one.block)]==[d.definition_id for d in definitions_for_block(two.block)]


def test_coverage_does_not_fit_normalize_or_select_and_retains_missing_reasons():
    block=legacy().block;report=coverage_diagnostics(block)
    assert report['fits']==0 and report['selection_applied'] is False
    assert report['columns']['return_60']['present']==12
    assert report['columns']['return_60']['missing_reasons'][M.HISTORY.value]==118
    assert report['columns']['return_60']['min']==0


def test_cached_artifact_registration_does_not_certify_current_execution():
    block=legacy().block;defs=definitions_for_block(block)
    receipt=materialization_receipt(block,defs,artifact_hashes={'values.parquet':HASH})
    assert not receipt['declared_implementation_matches'] and not receipt['historical_execution_certified']
    assert receipt['status']=='execution_not_verified_against_definition'
    assert verify_materialization(json.loads(json.dumps(receipt)))


def test_declared_matching_execution_is_only_consistency_and_requires_reference():
    block=legacy().block;defs=definitions_for_block(block);hashes=dict(defs[0].implementation_hashes)
    with pytest.raises(ValueError):materialization_receipt(block,defs,artifact_hashes={'values.parquet':HASH},execution_hashes=hashes)
    receipt=materialization_receipt(block,defs,artifact_hashes={'values.parquet':HASH},execution_hashes=hashes,execution_ref='fixture_verified_worker')
    assert receipt['declared_implementation_matches'] and not receipt['independent_reproduction_claimed']
    changed=deepcopy(receipt);changed['rows']+=1
    with pytest.raises(ValueError):verify_materialization(changed)


def test_materialization_definition_names_and_units_must_match_block():
    block=legacy().block;defs=list(definitions_for_block(block));defs[0]=replace(defs[0],unit='binary')
    with pytest.raises(ValueError):materialization_receipt(block,defs,artifact_hashes={'values.parquet':HASH})


def test_assembly_preserves_all_members_and_separates_namespaces():
    bars,cal=fixture();price=legacy().block;mkt=market_context_features(bars,cal,source_id='quote',version_id='q1',universe_id='declared_market',vintage='market_observation')
    result=assembled(blocks=[price,mkt],ids={'pv16':HASH,'market':'b'*64})
    assert len(result.block.values)==9 and len(result.block.units)==28
    assert len(result.registry.definitions)==28
    unseen=result.block.values.stock_code.eq('000003.SZ')
    assert result.block.values.loc[unseen,'pv16.return_1'].isna().all()
    assert result.block.missing.loc[unseen,'pv16.return_1'].eq(M.UNCOVERED.value).all()
    assert result.block.values.loc[unseen,'market.market_return_60'].notna().all()


def test_same_instance_metadata_cannot_replace_materialization_content_identity():
    one=assembled();two=assembled(ids={'pv16':'b'*64})
    assert one.block.metadata['version_id']!=two.block.metadata['version_id']
    assert one.registry.version_id==two.registry.version_id


def test_selected_columns_are_explicit_and_unknowns_reject():
    result=assembled(selected_keys=['pv16.return_1','pv16.return_60'])
    assert set(result.block.units)=={'pv16.return_1','pv16.return_60'}
    with pytest.raises(ValueError):assembled(selected_keys=['pv16.nonexistent'])


def test_weak_blocks_require_opt_in_and_do_not_upgrade_source_vintage():
    block=legacy().block;block.metadata['vintage']='latest_snapshot_only'
    with pytest.raises(ValueError,match='[Ww]eak'):assembled(blocks=[block])
    result=assembled(blocks=[block],allow_weak_vintage=True)
    assert result.block.metadata['vintage']=='latest_snapshot_only' and result.source_bindings[0]['weak_vintage_used']


def test_future_feature_clock_cannot_enter_earlier_decision():
    _,cal=fixture();p=membership(cal);p['decision_at']=p.trade_date+'T14:00:00+08:00'
    with pytest.raises(ValueError,match='available'):assembled(p=p)


def test_duplicate_version_and_invalid_membership_reject():
    block=legacy().block
    with pytest.raises(ValueError):assembled(blocks=[block,block])
    _,cal=fixture();p=membership(cal)
    with pytest.raises(ValueError):assembled(p=pd.concat([p,p.iloc[:1]]))
    with pytest.raises(ValueError):assembled(ids={'pv16':'not_hash'})


def test_extra_future_source_rows_do_not_change_previous_assembled_values():
    bars,cal=fixture();cut=cal[-2];block=legacy(bars.loc[bars.trade_date.le(cut)],cal).block
    p=membership(cal);p=p.loc[p.trade_date.le(cut)].reset_index(drop=True)
    first=assembled(p=p,blocks=[block]);later=assembled(p=p,blocks=[legacy().block])
    pd.testing.assert_frame_equal(first.block.values,later.block.values)
    pd.testing.assert_frame_equal(first.block.missing,later.block.missing)
