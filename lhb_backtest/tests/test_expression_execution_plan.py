"""One declared formula -> forecast -> matrix -> actual learner -> account plan."""
from copy import deepcopy
import json
from pathlib import Path
import pandas as pd
import pytest
from tests.test_alpha_account_registration import setup,learning_proposal_for_account
from tests.test_prepared_assembly import case,document
from src.alpharesearch.registry import FeatureRegistry
from src.alpharesearch.features.base import FeatureBlock
from src.alpharesearch.prepared_assembly import prepared_assembly_receipt
from src.alpharesearch.dsl import Expr,compile_expression
from src.researchops.workflow import ExecutionPlans,validate_plan
from src.technical.artifacts import write_json,digest


def chain(tmp_path):
    ops,old_session,account,folder=setup(tmp_path)
    learning=learning_proposal_for_account(ops,folder)
    learning['spec']['schema']='registered-alpha-learn-v2'
    learning['spec']['budget']['max_checkpoint_bytes']=1000000
    paths={r['role']:ops.project/r['path'] for r in learning['inputs']}
    definition=json.loads(paths['assembly_definition'].read_text(encoding='utf-8'));units=definition.pop('units')
    block=FeatureBlock(pd.read_parquet(paths['assembly_values']),pd.read_parquet(paths['assembly_missing']),units,definition)
    registry=FeatureRegistry.from_dict(json.loads(paths['registry'].read_text(encoding='utf-8')))
    membership=pd.read_parquet(paths['decisions'])[['sample_id','trade_date','stock_code','decision_at']]
    source=json.loads(paths['assembly_receipt'].read_text(encoding='utf-8'))
    raw=folder/'expressions';raw.mkdir()
    roles={k:paths[v] for k,v in [('values','assembly_values'),('missing','assembly_missing'),('definition','assembly_definition'),('registry','registry'),('source_receipt','assembly_receipt')]}
    roles['membership']=raw/'membership.parquet';membership.to_parquet(roles['membership'],index=False)
    hashes={k:digest(p) for k,p in roles.items()}
    prepared=prepared_assembly_receipt(block,registry,membership,source,artifact_hashes=hashes)
    roles['receipt']=raw/'receipt.json';write_json(roles['receipt'],prepared)
    days=pd.bdate_range('2024-01-01','2024-01-26').strftime('%Y-%m-%d').tolist()
    roles['calendar']=raw/'calendar.json';write_json(roles['calendar'],dict(trade_dates=days))
    roles['decision_clocks']=raw/'clocks.parquet'
    clocks=pd.DataFrame(dict(trade_date=days,decision_at=[d+'T15:30:00+08:00' for d in days]))
    actual=membership.groupby('trade_date').decision_at.first()
    mask=clocks.trade_date.isin(actual.index);clocks.loc[mask,'decision_at']=clocks.loc[mask,'trade_date'].map(actual)
    clocks.to_parquet(roles['decision_clocks'],index=False)
    _,example_registry,*_=case();doc=document('alpha-feature-batch-v4',example_registry,prepared['adapter_id'])
    doc['universe_id']=block.metadata['universe_id']
    doc['ranges'].update(development_start='2024-01-02',development_end='2024-01-04',holdout_start='2024-01-05',holdout_end='2024-01-12')
    key=learning['spec']['learning']['features'][0]
    binding=doc['bindings'].pop('stock.p');binding.update(column=key,definition_id=registry.resolve(key).definition_id)
    doc['bindings']={key:binding}
    doc['candidates'][0]['expression']=compile_expression(Expr.call('cs_rank',Expr.ref(key)),registry).to_dict()
    proposal=dict(kind='alpha_batch',spec=doc,inputs=[dict(role=r,path=p.relative_to(ops.project).as_posix(),sha256=digest(p),bytes=p.stat().st_size) for r,p in roles.items()],timeout_seconds=60,hypothesis='One fixed software formula',expected_observation='No outcome selection',falsification='Any lost keys or clocks')
    learning['spec']['learning']['features'].append('expr.'+doc['candidates'][0]['candidate_id'])
    account['spec']['schema']='registered-alpha-account-v2'
    account['spec']['score_source']={'kind':'registered_learning','experiment_id':'exp-'+'a'*12,'verification_mode':'reuse_completed_identity_v1'}
    task=ops.create(dict(title='finite formula matrix learning account',question='fixed tiny pipeline',rationale='software integration only',success_criteria=['No duplicate jobs'],stop_criteria=['Four worker starts only'],max_experiments=4))
    session=ops.store.claim(task['id'],'tiny-v4')
    plan=dict(schema='execution-plan-v4',id='formula-chain',task_id=task['id'],max_starts=4,timeout_seconds=1800,
        steps=[dict(id='dev',needs=[],proposal=proposal),dict(id='forecast',needs=['dev'],forecast_from_batch=dict(development_step='dev',name='tiny_forecast',timeout_seconds=60)),
            dict(id='learn',needs=['dev','forecast'],learning_from_expressions=dict(development_step='dev',forecast_step='forecast',matrix_folder='research/chain-matrix',template=learning,max_rows=100,max_matrix_cells=1000,max_output_bytes=1000000)),
            dict(id='account',needs=['learn'],account_from_learning=dict(learning_step='learn',template=account))])
    return ops,session,plan


def test_real_v4_pipeline_reuses_matrix_after_pre_registration_interruption(tmp_path,monkeypatch):
    ops,session,plan=chain(tmp_path);plans=ExecutionPlans(ops)
    register=ops.register
    def interrupt(session,proposal):
        if proposal['kind']=='alpha_learn':raise SystemExit('matrix complete before learner registration')
        return register(session,proposal)
    monkeypatch.setattr(ops,'register',interrupt)
    with pytest.raises(SystemExit,match='matrix complete'):
        initial=plans.run(session,plan)
        pytest.fail(str([(r['name'],r['status'],r['error']) for r in initial['steps']]))
    assert (ops.project/'research/chain-matrix/receipt.json').is_file()
    assert len(ops.store.get(session['task_id'])['experiments'])==2
    # Self-consistent re-sealing cannot turn modified values into real producer outputs.
    from src.researchops.learning_proposal import learning_from_expression_batches
    from src.technical.artifacts import content_id
    matrix=ops.project/'research/chain-matrix';values=matrix/'assembly_values.parquet';receipt=matrix/'receipt.json'
    saved_values=values.read_bytes();saved_receipt=receipt.read_bytes()
    frame=pd.read_parquet(values);feature=next(k for k in frame if k.startswith('expr.'))
    frame.loc[0,feature]+=1e-10;frame.to_parquet(values,index=False)
    sealed=json.loads(receipt.read_text(encoding='utf-8'));sealed['artifacts']['assembly_values']=digest(values)
    sealed.pop('reference_id');sealed['reference_id']=content_id(sealed);write_json(receipt,sealed)
    records=ops.store.get(session['task_id'])['experiments']
    with pytest.raises(ValueError,match='Matrix content differs'):
        learning_from_expression_batches(ops,records[0]['id'],records[1]['id'],'research/chain-matrix',
            plan['steps'][2]['learning_from_expressions']['template'],max_rows=100,max_matrix_cells=1000,max_output_bytes=1000000)
    values.write_bytes(saved_values);receipt.write_bytes(saved_receipt)
    monkeypatch.setattr(ops,'register',register)
    import src.researchops.matrix_materialization as materializer
    monkeypatch.setattr(materializer,'matrix_from_registered_expressions',lambda *a,**k:pytest.fail('Complete matrix must be reused'))
    result=plans.run(session,plan)
    assert result['completed'],[(r['name'],r['status'],r['error']) for r in result['steps']]
    experiments=ops.store.get(session['task_id'])['experiments']
    assert len(experiments)==4 and result['starts']==4
    assert [x['proposal']['kind'] for x in experiments].count('alpha_learn')==1
    assert plans.run(session,plan)['starts']==4
    from src.researchops.plan_backup import backup_plan_evidence,verify_plan_backup
    backup=ops.project/'research/chain-backup'
    with pytest.raises(ValueError,match='before copying'):
        backup_plan_evidence(ops,plan['id'],'research/too-small',max_bytes=1024)
    assert not (ops.project/'research/too-small').exists()
    copied=backup_plan_evidence(ops,plan['id'],backup,max_bytes=100000000)
    assert copied['experiments']==4 and copied['offline_verified'] and not copied['activation_allowed']
    import shutil
    offline=tmp_path/'offline-copy';shutil.copytree(backup,offline)
    moved=ops.project.with_name('original-unavailable');ops.project.rename(moved)
    try:
        assert verify_plan_backup(offline)==copied
        # No source files or old project path can participate in this check.
        extra=offline/'unexpected.txt';extra.write_text('not listed',encoding='utf-8')
        with pytest.raises(ValueError,match='inventory'):verify_plan_backup(offline)
        extra.unlink()
        member=next((offline/'project').rglob('predictions.parquet'));member.write_bytes(b'changed')
        with pytest.raises(ValueError,match='artifact differs'):verify_plan_backup(offline)
    finally:moved.rename(ops.project)



@pytest.mark.parametrize('fault',['missing_dependency','bool_cap','same_producer','empty_folder','bad_timeout'])
def test_v4_declarations_are_strict(tmp_path,fault):
    _,_,plan=chain(tmp_path)
    if fault=='missing_dependency':plan['steps'][2]['needs']=['dev']
    if fault=='bool_cap':plan['steps'][2]['learning_from_expressions']['max_rows']=True
    if fault=='same_producer':plan['steps'][2]['learning_from_expressions']['forecast_step']='dev'
    if fault=='empty_folder':plan['steps'][2]['learning_from_expressions']['matrix_folder']=''
    if fault=='bad_timeout':plan['steps'][1]['forecast_from_batch']['timeout_seconds']=0
    with pytest.raises(ValueError):validate_plan(plan)
