"""Projection keeps the old scoped learning result and unscoped key rejection."""
import json
from pathlib import Path
import pandas as pd
import pytest
from src.researchops.learn_experiments import protocol,inspect_inputs
from src.researchops.alpha_experiments import verify_inputs
from src.researchops.screen_experiments import assembly_reference
from src.alpharesearch.assembly import FeatureAssembly
from src.alpharesearch.features.base import FeatureBlock
from src.alpharesearch.registry import FeatureRegistry
from src.alpharesearch.learning import LearningSpec,_prepare,_prepared_parts
from src.technical.artifacts import digest,write_json
from tests.test_alpha_learn_registration import setup


def source_case(tmp_path):
    ops,session,proposal,folder=setup(tmp_path)
    cfg=proposal['spec']['learning'];cfg['features']=cfg['features'][:1]
    for role in ['assembly_values','assembly_missing','decisions','labels']:
        path=folder/(role+'.parquet');frame=pd.read_parquet(path)
        row=frame.iloc[[0]].copy();row['sample_id']='outside_sample';row['trade_date']='2023-12-29';row['stock_code']='600999.SH'
        pd.concat([frame,row],ignore_index=True).to_parquet(path,index=False)
    reseal(proposal,folder)
    return ops,session,proposal,folder


def reseal(proposal,folder):
    meta=json.loads((folder/'definition.json').read_text(encoding='utf-8'));units=meta.pop('units')
    registry=FeatureRegistry.from_dict(json.loads((folder/'registry.json').read_text(encoding='utf-8')))
    block=FeatureBlock(pd.read_parquet(folder/'assembly_values.parquet'),pd.read_parquet(folder/'assembly_missing.parquet'),units,meta)
    assembly=FeatureAssembly(block,registry,tuple(meta['source_bindings']))
    write_json(folder/'receipt.json',assembly_reference(assembly,{r:digest(folder/n) for r,n in [('assembly_values','assembly_values.parquet'),('assembly_missing','assembly_missing.parquet'),('assembly_definition','definition.json')]}))
    for record in proposal['inputs']:
        name={'assembly_definition':'definition.json','assembly_receipt':'receipt.json','registry':'registry.json'}.get(record['role'],record['role']+'.parquet')
        path=folder/name;record.update(sha256=digest(path),bytes=path.stat().st_size)


def full_prepare(ops,proposal):
    spec,model,_=protocol(proposal['spec']);paths=verify_inputs(ops.project,proposal['inputs'],proposal['spec']['budget']['max_input_bytes'])
    registry=FeatureRegistry.from_dict(json.loads(paths['registry'].read_text(encoding='utf-8')))
    meta=json.loads(paths['assembly_definition'].read_text(encoding='utf-8'));units=meta.pop('units')
    block=FeatureBlock(pd.read_parquet(paths['assembly_values']),pd.read_parquet(paths['assembly_missing']),units,meta)
    assembly=FeatureAssembly(block,registry,tuple(meta['source_bindings']))
    return _prepare(spec,model,pd.read_parquet(paths['decisions']),assembly,pd.read_parquet(paths['labels']))


def test_selected_column_and_dates_reproduce_every_prepared_value_and_receipt(tmp_path,monkeypatch):
    ops,_,proposal,_=source_case(tmp_path);expected=_prepared_parts(full_prepare(ops,proposal))
    reader=pd.read_parquet;calls=[]
    def selected(path,*args,**kwargs):
        calls.append((Path(path).name,kwargs.get('columns'),kwargs.get('filters')))
        return reader(path,*args,**kwargs)
    monkeypatch.setattr(pd,'read_parquet',selected)
    _,_,_,info=inspect_inputs(ops.project,proposal,include_data=True)
    actual=_prepared_parts(info['prepared'])
    for left,right in zip(expected,actual):
        if isinstance(left,pd.DataFrame):pd.testing.assert_frame_equal(left,right)
        elif isinstance(left,pd.Series):pd.testing.assert_series_equal(left,right)
        else:assert left==right
    assert len(calls)==8 and all(columns is not None for _,columns,_ in calls)
    values=[x for x in calls if x[0]=='assembly_values.parquet' and x[2] is not None][0]
    assert values[1]==['sample_id','trade_date','stock_code']+proposal['spec']['learning']['features']+['observed_end','known_at']
    decisions,assembly,labels=info['data']
    assert not assembly.block.values.trade_date.eq('2023-12-29').any()
    assert set(assembly.block.units)==set(proposal['spec']['learning']['features'])
    assert labels.trade_date.le(proposal['spec']['learning']['train_end']).all()


@pytest.mark.parametrize('role',['assembly_values','assembly_missing','decisions','labels'])
def test_bad_keys_outside_scope_are_not_hidden_by_projection(tmp_path,role):
    ops,_,proposal,folder=source_case(tmp_path);path=folder/(role+'.parquet');frame=pd.read_parquet(path)
    frame.loc[len(frame)-1,'sample_id']=frame.sample_id.iloc[0];frame.to_parquet(path,index=False);reseal(proposal,folder)
    with pytest.raises(ValueError,match='unique non-null'):full_prepare(ops,proposal)
    with pytest.raises(ValueError,match='unique non-null'):inspect_inputs(ops.project,proposal)


def test_full_input_row_cap_is_not_redefined_as_projected_rows(tmp_path):
    ops,_,proposal,folder=source_case(tmp_path)
    proposal['spec']['learning']['max_input_rows']=len(pd.read_parquet(folder/'assembly_values.parquet'))-1
    with pytest.raises(ValueError,match='input row cap'):full_prepare(ops,proposal)
    with pytest.raises(ValueError,match='input row cap'):inspect_inputs(ops.project,proposal)


def test_prediction_label_values_never_change_training_or_projection(tmp_path):
    ops,_,proposal,folder=source_case(tmp_path);before=_prepared_parts(full_prepare(ops,proposal))
    path=folder/'labels.parquet';frame=pd.read_parquet(path);outside=frame.trade_date.gt(proposal['spec']['learning']['train_end'])
    frame.loc[outside,'target_return']=999999.;frame.to_parquet(path,index=False);reseal(proposal,folder)
    after=_prepared_parts(inspect_inputs(ops.project,proposal,include_data=True)[3]['prepared'])
    for left,right in zip(before,after):
        if isinstance(left,pd.DataFrame):pd.testing.assert_frame_equal(left,right)
        elif isinstance(left,pd.Series):pd.testing.assert_series_equal(left,right)
        else:assert left==right


def test_pandas_index_metadata_preserves_logical_columns(tmp_path):
    ops,_,proposal,folder=source_case(tmp_path)
    for role in ['assembly_values','assembly_missing','decisions','labels']:
        path=folder/(role+'.parquet');frame=pd.read_parquet(path);frame.index.name='source_row_index';frame.to_parquet(path,index=True)
    reseal(proposal,folder);expected=_prepared_parts(full_prepare(ops,proposal));actual=_prepared_parts(inspect_inputs(ops.project,proposal,include_data=True)[3]['prepared'])
    for left,right in zip(expected,actual):
        if isinstance(left,pd.DataFrame):pd.testing.assert_frame_equal(left,right)
        elif isinstance(left,pd.Series):pd.testing.assert_series_equal(left,right)
        else:assert left==right


def test_label_identity_conflict_outside_training_range_is_still_rejected(tmp_path):
    ops,_,proposal,folder=source_case(tmp_path)
    path=folder/"labels.parquet";labels=pd.read_parquet(path)
    # Keep sample_id globally unique, but move a training label outside scope.
    labels.loc[0,"trade_date"]="2019-12-30"
    labels.to_parquet(path,index=False);reseal(proposal,folder)
    for prepare in [full_prepare,inspect_inputs]:
        with pytest.raises(ValueError,match="stock/date keys conflict with sample identity"):
            prepare(ops if prepare is full_prepare else ops.project,proposal)
