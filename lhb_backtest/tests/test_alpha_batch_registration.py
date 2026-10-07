"""Prospective candidate plans, frozen reference identity and fail-closed dispatch."""
import copy
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from src.alpharesearch.batch import AlphaBatchSpec
from src.alpharesearch.dsl import Expr as E,compile_expression
from src.alpharesearch.features.primitives import daily_primitives
from src.alpharesearch.registry import FeatureRegistry,definitions_for_block,materialization_receipt
from src.researchops.service import Research
from src.researchops.alpha_experiments import verify_registered,inspect_inputs
from src.technical.artifacts import digest
from tests.test_alpha_dsl import bars
from tests.test_researchops import TASK

PROJECT=Path(__file__).resolve().parents[1]


def setup(tmp_path,max_experiments=2):
    project=tmp_path/'project';project.mkdir()
    for source in (PROJECT/'src').rglob('*.py'):
        if '__pycache__' in source.parts:continue
        dest=project/source.relative_to(PROJECT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    folder=project/'data/alpharesearch/test_input';folder.mkdir(parents=True)
    block=daily_primitives(bars(),source_id='fixture_daily',version_id='q1');defs=definitions_for_block(block);registry=FeatureRegistry(defs)
    block.values.to_parquet(folder/'values.parquet',index=False);block.missing.to_parquet(folder/'missing.parquet',index=False)
    receipt=materialization_receipt(block,defs,artifact_hashes={n:digest(folder/n) for n in ('values.parquet','missing.parquet')})
    objects={'definition.json':dict(block.metadata,units=block.units),'receipt.json':receipt,'registry.json':registry.to_dict(),'calendar.json':{'trade_dates':['2024-01-02','2024-01-03','2024-01-04','2024-01-05','2024-01-08','2024-01-09','2024-01-10']}}
    for name,obj in objects.items():(folder/name).write_text(json.dumps(obj,ensure_ascii=False),encoding='utf-8')
    p=block.values[['trade_date','stock_code']].copy();p=pd.concat([p,p.iloc[:2].assign(trade_date='2024-01-08')],ignore_index=True)
    p['sample_id']=p.trade_date+p.stock_code;p['decision_at']=p.trade_date+'T15:10:00+08:00';p.to_parquet(folder/'membership.parquet',index=False)
    days=objects['calendar.json']['trade_dates'];pd.DataFrame({'trade_date':days,'decision_at':[d+'T15:10:00+08:00' for d in days]}).to_parquet(folder/'decision_clocks.parquet',index=False)
    expr=compile_expression(E.call('cs_rank',E.call('div',E.ref('daily.close'),E.ref('daily.pre_close'))),registry)
    bindings={key:{'block':'daily','column':key.split('.')[1],'definition_id':registry.resolve(key).definition_id,'materialization_id':receipt['materialization_id']} for key in ('daily.close','daily.pre_close')}
    spec={'schema':'alpha-feature-batch-v1','name':'hand feature batch','dataset_id':'fixture_snapshot','universe_id':'fixture_membership',
        'ranges':{'warmup_start':'2024-01-02','development_start':'2024-01-03','development_end':'2024-01-05','holdout_start':'2024-01-08','holdout_end':'2024-01-10'},
        'candidates':[{'candidate_id':'relative_close_rank','hypothesis':'hand formula identity check','family':'ratio','expression':expr.to_dict()}],
        'blocks':{'daily':{'values':'daily_values','missing':'daily_missing','definition':'daily_definition','receipt':'daily_receipt'}},'bindings':bindings,
        'budget':{'max_candidates':4,'max_input_bytes':10_000_000,'max_grid_cells':100,'max_buffer_bytes':1_000_000,'max_model_fits':0,'max_accounts':0},
        'allow_weak_vintage':False,'evaluation_scope':'retrospective_time_split','selection_rule':'all_candidates_no_selection','primary_objectives':['account_return','max_drawdown']}
    role_files={'daily_values':'values.parquet','daily_missing':'missing.parquet','daily_definition':'definition.json','daily_receipt':'receipt.json','registry':'registry.json','calendar':'calendar.json','membership':'membership.parquet','decision_clocks':'decision_clocks.parquet'}
    inputs=[{'role':role,'path':(folder/name).relative_to(project).as_posix(),'sha256':digest(folder/name),'bytes':(folder/name).stat().st_size} for role,name in role_files.items()]
    proposal={'kind':'alpha_batch','hypothesis':'prospective engineering freeze','expected_observation':'consistent candidate identity','falsification':'changed input must fail','spec':spec,'inputs':inputs,'timeout_seconds':60}
    ops=Research(project);task=ops.create(dict(TASK,max_experiments=max_experiments));session=ops.store.claim(task['id'],'test-batch-worker')
    return ops,session,proposal,registry,folder


def test_first_class_registration_is_frozen_planned_and_holdout_not_executed(tmp_path):
    ops,session,p,registry,_=setup(tmp_path);ex=ops.register(session,p);checked=verify_registered(ops,ex)
    assert ex['proposal']['kind']=='alpha_batch' and ex['status']=='planned'
    assert checked['candidate_count']==1 and checked['executed_candidates']==0 and checked['accounts']==0
    assert ex['proposal']['inspection']['execution_rows']==4 and ex['proposal']['inspection']['holdout_rows_executed']==0
    assert ex['proposal']['inspection']['execution_days']==['2024-01-02','2024-01-03','2024-01-04','2024-01-05']
    assert AlphaBatchSpec.from_dict(p['spec'],registry).planned_attempts[0]['status']=='planned'


def test_duplicate_description_or_input_order_returns_original_registration(tmp_path):
    ops,session,p,_,_=setup(tmp_path);first=ops.register(session,p);second=copy.deepcopy(p)
    second['hypothesis']='new description';second['spec']['name']='display renamed';second['inputs'].reverse()
    again=ops.register(session,second);assert first['id']==again['id'] and again['proposal']['hypothesis']==p['hypothesis']
    assert len(ops.store.get(session['task_id'])['experiments'])==1


def test_counted_identical_formulas_and_cancelled_batch_do_not_refund_budget(tmp_path):
    ops,session,p,r,_=setup(tmp_path,max_experiments=1);second=copy.deepcopy(p['spec']['candidates'][0]);second['candidate_id']='same_formula_counted';p['spec']['candidates'].append(second)
    ex=ops.register(session,p);assert ex['proposal']['candidate_count']==2
    ops.store.cancel(session,ex['id'],'engineering-only no execution yet');p['spec']['candidates'][1]['candidate_id']='changed_candidate'
    with pytest.raises(ValueError,match='预算'):ops.register(session,p)


def test_execute_uses_new_registered_worker_without_rewriting_old_protocol(tmp_path):
    ops,session,p,_,_=setup(tmp_path);ex=ops.register(session,p)
    final=ops.execute(session,ex['id'])
    assert final['status']=='completed' and final['result']['candidate_count']==1
    assert final['result']['fits']==final['result']['accounts']==0


@pytest.mark.parametrize('mutation',['overlap','future_lag','duplicate_name','unused_binding','budget_count','fit_budget','claim_success','missing_input','wrong_column','wrong_receipt','extra_input','outside_path','wrong_hash','byte_budget'])
def test_invalid_protocol_or_input_rejects_before_registration(tmp_path,mutation):
    ops,session,p,r,_=setup(tmp_path)
    if mutation=='overlap':p['spec']['ranges']['holdout_start']='2024-01-05'
    if mutation=='future_lag':p['spec']['candidates'][0]['expression']['ast']=E.call('lag',E.ref('daily.close'),periods=-1).to_dict()
    if mutation=='duplicate_name':p['spec']['candidates']*=2
    if mutation=='unused_binding':p['spec']['bindings']['daily.volume']=copy.deepcopy(p['spec']['bindings']['daily.close'])
    if mutation=='budget_count':p['spec']['budget']['max_candidates']=0
    if mutation=='fit_budget':p['spec']['budget']['max_model_fits']=1
    if mutation=='claim_success':p['spec']['selection_rule']='best_strategy_is_profitable'
    if mutation=='missing_input':p['inputs']=p['inputs'][1:]
    if mutation=='wrong_column':p['spec']['bindings']['daily.close']['column']='open'
    if mutation=='wrong_receipt':p['spec']['bindings']['daily.close']['materialization_id']='c'*64
    if mutation=='extra_input':p['inputs'].append(dict(p['inputs'][0],role='extra'))
    if mutation=='outside_path':p['inputs'][0]['path']='../external.parquet'
    if mutation=='wrong_hash':p['inputs'][0]['sha256']='f'*64
    if mutation=='byte_budget':p['spec']['budget']['max_input_bytes']=1
    with pytest.raises((ValueError,KeyError)):ops.register(session,p)
    assert not ops.store.get(session['task_id'])['experiments']


@pytest.mark.parametrize('mutation',['source_input','code','input_manifest','candidate_manifest','ready','db_identity'])
def test_changed_frozen_registration_cannot_be_verified(tmp_path,mutation):
    ops,session,p,_,folder=setup(tmp_path);ex=ops.register(session,p);job=ops.root/'worker_jobs'/ex['id']
    if mutation=='source_input':(folder/'calendar.json').write_text('{}',encoding='utf-8')
    if mutation=='code':(job/'code/src/alpharesearch/dsl.py').write_text('changed',encoding='utf-8')
    if mutation=='input_manifest':(job/'input.json').write_text('{}',encoding='utf-8')
    if mutation=='candidate_manifest':(job/'candidate_manifest.json').write_text('{}',encoding='utf-8')
    if mutation=='ready':(job/'ready.json').write_text('{}',encoding='utf-8')
    if mutation=='db_identity':ex=copy.deepcopy(ex);ex['proposal']['candidate_count']=999
    with pytest.raises(ValueError):verify_registered(ops,ex)


def test_new_source_code_does_not_rewrite_old_frozen_registration(tmp_path):
    ops,session,p,_,_=setup(tmp_path);ex=ops.register(session,p)
    (ops.project/'src/alpharesearch/cache.py').write_text('a future version',encoding='utf-8')
    assert verify_registered(ops,ex)['verified']
    assert (ops.root/'worker_jobs'/ex['id']/'code/src/alpharesearch/cache.py').read_text(encoding='utf-8')!='a future version'


def test_development_inspection_never_uses_holdout_membership_rows(tmp_path):
    ops,_,p,_,_=setup(tmp_path);_,_,info=inspect_inputs(ops.project,p)
    assert info['execution_rows']==4 and info['holdout_rows_executed']==0



@pytest.mark.parametrize('mutation',['missing_development_day','duplicate','unsorted','invalid_date'])
def test_calendar_cannot_silently_delete_development_rows(tmp_path,mutation):
    ops,session,p,_,folder=setup(tmp_path)
    path=folder/'calendar.json';doc=json.loads(path.read_text(encoding='utf-8'))
    if mutation=='missing_development_day':doc['trade_dates'].remove('2024-01-03')
    if mutation=='duplicate':doc['trade_dates'].append('2024-01-03')
    if mutation=='unsorted':doc['trade_dates']=doc['trade_dates'][::-1]
    if mutation=='invalid_date':doc['trade_dates'][0]='2024-01-99'
    path.write_text(json.dumps(doc),encoding='utf-8')
    record=next(x for x in p['inputs'] if x['role']=='calendar')
    record.update(sha256=digest(path),bytes=path.stat().st_size)
    with pytest.raises(ValueError):ops.register(session,p)
    assert not ops.store.get(session['task_id'])['experiments']


@pytest.mark.parametrize('field,value',[
    ('schema','invalid'),('fit_budget',99),('account_budget',99),
    ('counting_rule','failed trials refunded'),('extra','undeclared'),
])
def test_candidate_contract_tampering_is_rejected_even_with_updated_ready_hash(tmp_path,field,value):
    ops,session,p,_,_=setup(tmp_path);ex=ops.register(session,p);job=ops.root/'worker_jobs'/ex['id']
    path=job/'candidate_manifest.json';plan=json.loads(path.read_text(encoding='utf-8'))
    plan[field]=value;path.write_text(json.dumps(plan),encoding='utf-8')
    ready_path=job/'ready.json';ready=json.loads(ready_path.read_text(encoding='utf-8'))
    ready['candidate_manifest_hash']=digest(path);ready_path.write_text(json.dumps(ready),encoding='utf-8')
    with pytest.raises(ValueError,match='candidate/registry'):verify_registered(ops,ex)


def test_fake_completed_alpha_dispatch_rejected(tmp_path):
    ops,session,p,_,_=setup(tmp_path);ex=ops.register(session,p)
    with ops.store.connection(True) as db:
        db.execute("UPDATE experiments SET status='completed' WHERE id=?",(ex['id'],))
    report={'summary':'fake completion','findings':['must reject'],
            'limitations':['fixture only'],'next_steps':['real audit required']}
    with pytest.raises(ValueError,match='execution evidence'):ops.submit(session,report)


@pytest.mark.parametrize('addition',['shadow_python','bytecode'])
def test_unregistered_import_files_in_frozen_code_are_rejected(tmp_path,addition):
    ops,session,p,_,_=setup(tmp_path);ex=ops.register(session,p);code=ops.root/'worker_jobs'/ex['id']/'code'
    extra=code/('src/alpharesearch/dsl/__init__.py' if addition=='shadow_python'
                else 'src/alpharesearch/undeclared.pyc')
    extra.parent.mkdir(parents=True,exist_ok=True);extra.write_bytes(b'not executed')
    with pytest.raises(ValueError,match='file set'):verify_registered(ops,ex)


def test_inspection_leaf_buffers_have_a_finite_preparation_budget(tmp_path):
    ops,session,p,_,_=setup(tmp_path);p['spec']['budget']['max_buffer_bytes']=1
    with pytest.raises(ValueError,match='buffer budget'):ops.register(session,p)


def test_freeze_directory_failure_is_recorded_and_consumes_budget(tmp_path,monkeypatch):
    ops,session,p,_,_=setup(tmp_path,max_experiments=1)
    original=Path.mkdir
    def fail_code_dir(path,*args,**kwargs):
        if path.name=='code' and path.parent.parent.name=='worker_jobs':
            raise OSError('fixture code directory failure')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'mkdir',fail_code_dir)
    with pytest.raises(OSError,match='fixture'):ops.register(session,p)
    task=ops.store.get(session['task_id'])
    assert len(task['experiments'])==1 and task['experiments'][0]['status']=='failed'
    assert any(event['kind']=='freeze_failed' for event in task['events'])


def test_explicit_binding_accepts_registry_with_multiple_versions(tmp_path):
    from dataclasses import replace
    ops,session,p,registry,folder=setup(tmp_path)
    definition=registry.resolve('daily.close')
    registry.add(replace(definition,description=definition.description+' alternate declared version'))
    for candidate in p['spec']['candidates']:
        candidate['expression']=compile_expression(E.from_dict(candidate['expression']['ast']),registry).to_dict()
    path=folder/'registry.json';path.write_text(json.dumps(registry.to_dict()),encoding='utf-8')
    record=next(x for x in p['inputs'] if x['role']=='registry')
    record.update(sha256=digest(path),bytes=path.stat().st_size)
    ex=ops.register(session,p)
    assert verify_registered(ops,ex)['verified']
    assert ex['proposal']['inspection']['execution_rows']==4
