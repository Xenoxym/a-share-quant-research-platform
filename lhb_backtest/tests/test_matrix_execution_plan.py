"""Real tiny matrix/fit/account chain and durable interrupted registration."""
import copy
import shutil
import pytest
from src.researchops.workflow import ExecutionPlans,validate_plan
from src.researchops.learning_proposal import MATRIX_FILES
from tests.test_alpha_account_registration import setup,learning_proposal_for_account


def chain(tmp_path):
    ops,session,account,folder=setup(tmp_path)
    learning=learning_proposal_for_account(ops,folder)
    learning['spec']['schema']='registered-alpha-learn-v2'
    learning['spec']['budget']['max_checkpoint_bytes']=1000000
    matrix=ops.project/'data/matrix';matrix.mkdir()
    for record in learning['inputs']:
        if record['role'] in MATRIX_FILES:
            shutil.copy2(ops.project/record['path'],matrix/MATRIX_FILES[record['role']])
    account['spec']['schema']='registered-alpha-account-v2'
    account['spec']['score_source']={'kind':'registered_learning','experiment_id':'exp-'+'a'*12,
        'verification_mode':'reuse_completed_identity_v1'}
    plan=dict(schema='execution-plan-v3',id='matrix-chain',task_id=session['task_id'],max_starts=2,
        timeout_seconds=1800,steps=[dict(id='learn',needs=[],learning_from_matrix=dict(
            matrix_folder='data/matrix',template=learning)),dict(id='account',needs=['learn'],
            account_from_learning=dict(learning_step='learn',template=account))])
    return ops,session,plan


def test_real_matrix_fit_account_recovery_reuses_original_registration(tmp_path,monkeypatch):
    ops,session,plan=chain(tmp_path);plans=ExecutionPlans(ops);save=plans._save
    def interrupted(*args,**kwargs):
        if args[2]=='learn' and kwargs.get('status')=='registered':
            raise SystemExit('registered before checkpoint')
        return save(*args,**kwargs)
    monkeypatch.setattr(plans,'_save',interrupted)
    with pytest.raises(SystemExit):plans.run(session,plan)
    original=ops.store.get(session['task_id'])['experiments']
    assert len(original)==1 and original[0]['status']=='planned'
    monkeypatch.setattr(plans,'_save',save)
    import src.researchops.learning_proposal as adapters
    monkeypatch.setattr(adapters,'learning_from_matrix',lambda *a:pytest.fail('must recover original registration'))
    first=plans.run(session,plan)
    assert first['completed'] and first['starts']==2
    exs=ops.store.get(session['task_id'])['experiments']
    assert len(exs)==2 and first['steps'][0]['experiment_id']==original[0]['id']
    learn,account=exs
    assert learn['result']['verified'] and account['result']['verified']
    assert account['result']['additional_audit_fits']==0
    assert account['proposal']['spec']['score_source']['experiment_id']==learn['id']
    with ops.store.connection(True) as db:
        db.execute('UPDATE execution_plans SET deadline=0 WHERE id=?',(plan['id'],))
    monkeypatch.setattr(ops,'register',lambda *a:pytest.fail('no second registration'))
    monkeypatch.setattr(ops,'execute',lambda *a:pytest.fail('no second worker'))
    again=plans.run(session,plan)
    assert again['completed'] and again['starts']==2
    assert [s['experiment_id'] for s in again['steps']]==[s['experiment_id'] for s in first['steps']]


@pytest.mark.parametrize('fault',['v1','v2','extra','folder','template','both'])
def test_strict_matrix_step_contract(tmp_path,fault):
    _,_,plan=chain(tmp_path)
    if fault in {'v1','v2'}:plan['schema']='execution-plan-'+fault
    elif fault=='extra':plan['steps'][0]['learning_from_matrix']['automatic_selection']=True
    elif fault=='folder':plan['steps'][0]['learning_from_matrix']['matrix_folder']=''
    elif fault=='template':plan['steps'][0]['learning_from_matrix']['template']['inspection']={}
    elif fault=='both':plan['steps'][0]['proposal']={}
    with pytest.raises(ValueError):validate_plan(plan)


def test_exhausted_budget_never_builds_or_registers(tmp_path,monkeypatch):
    ops,session,plan=chain(tmp_path);plan['max_starts']=0
    import src.researchops.learning_proposal as adapters
    monkeypatch.setattr(adapters,'learning_from_matrix',lambda *a:pytest.fail('budget before builder'))
    result=ExecutionPlans(ops).run(session,plan)
    assert result['starts']==0 and result['steps'][0]['status']=='blocked'
    assert not ops.store.get(session['task_id'])['experiments']
    changed=copy.deepcopy(plan);changed['steps'][0]['learning_from_matrix']['matrix_folder']='data/other'
    with pytest.raises(ValueError,match='immutable'):ExecutionPlans(ops).run(session,changed)
