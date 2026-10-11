"""Identity evidence fixtures are synthetic records, not fitted strategies."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import platform
import pytest
from src.researchops.completed_learning_identity import verify_completed_learning_identity
from src.researchops.resources import WorkerResources
from src.technical.artifacts import content_id,digest,write_json


def fixture_record(tmp_path):
    project=tmp_path/'p';project.mkdir();root=project/'data/technical'
    eid='exp-'+'a'*12;job=root/'worker_jobs'/eid;out=job/'learn_result';out.mkdir(parents=True)
    code=job/'code'
    sources={n:'# synthetic prior source '+n+'\n' for n in ['src/alpharesearch/models.py','src/alpharesearch/contracts.py','src/technical/artifacts.py','src/researchops/learn_output_audit.py']}
    for n,t in sources.items():q=code/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(t,encoding='utf8')
    inv={n:digest(code/n) for n in sources};inputfile=project/'research/input.txt';inputfile.parent.mkdir();inputfile.write_text('unchanged input')
    proposal=dict(kind='alpha_learn',spec=dict(schema='registered-alpha-learn-v3',budget=dict(max_input_bytes=10000)),
        inputs=[dict(role='decisions',path='research/input.txt',sha256=digest(inputfile),bytes=inputfile.stat().st_size)],
        engine_inventory=inv,engine_hash=content_id(inv),environment=dict(python=platform.python_version(),packages={}),
        resources=WorkerResources().to_dict(),planned_model_fits=1,planned_accounts=0)
    cfg=dict(proposal,project=str(project),root=str(root),task_id='task-fixture',experiment_id=eid)
    write_json(job/'input.json',cfg);write_json(job/'code_manifest.json',dict(artifacts=inv));write_json(job/'candidate_manifest.json',dict(fixture='synthetic ledger'))
    write_json(job/'ready.json',dict(kind='alpha_learn',input_hash=digest(job/'input.json'),code_manifest_hash=digest(job/'code_manifest.json'),candidate_manifest_hash=digest(job/'candidate_manifest.json')))
    learning=dict(fit_id='f'*64,prediction_id='e'*64,actual_fit_attempts_this_call=1,fit_intents_used=1,
        prediction_outputs=dict(rows=2,sha256='d'*64),latest_training_label_known_at='2024-01-01T01:00:00+00:00',
        fit_identity=dict(settings=dict(fit_cutoff='2024-01-02T01:00:00+00:00'),implementation_hashes={n:hashlib.sha256(t.encode()).hexdigest() for n,t in sources.items() if 'researchops' not in n}))
    write_json(out/'learning_receipt.json',learning)
    (out/'predictions.parquet').write_bytes(b'synthetic evidence bytes, not market predictions')
    manifest=dict(schema='registered-alpha-learn-manifest-v3',experiment_id=eid,task_id='task-fixture',input_hash=digest(job/'input.json'),engine_hash=proposal['engine_hash'],candidate_manifest_hash=digest(job/'candidate_manifest.json'),artifacts={q.name:digest(q) for q in out.iterdir()})
    write_json(out/'manifest.json',manifest)
    launch=dict(schema='alpha-worker-launch-v1',experiment_id=eid,pid=99999999,created=1.,interpreter_chain=[],input_hash=digest(job/'input.json'),resources=proposal['resources'])
    write_json(job/'launch.json',launch)
    write_json(job/'execution_stats.json',dict(schema='alpha-worker-observation-v1',experiment_id=eid,launch=launch,child_exit_code=0,parent_observed_to_exit=True))
    write_json(job/'receipt.json',dict(status='completed',kind='alpha_learn',experiment_id=eid,result_manifest_hash=digest(out/'manifest.json')))
    audit=dict(schema='registered-alpha-learn-audit-v3',verified=True,experiment_id=eid,model_fits=1,additional_audit_fits=0,model_scores_independently_reproduced=True,audit_code_hash=inv['src/researchops/learn_output_audit.py'],fit_id=learning['fit_id'],prediction_id=learning['prediction_id'])
    audit['audit_id']=content_id(audit);write_json(job/'alpha_audit.json',audit)
    result=dict(audit,input_hash=digest(job/'input.json'),engine_hash=proposal['engine_hash'],output_manifest_hash=digest(out/'manifest.json'),**{k:digest(job/n) for k,n in [('audit_hash','alpha_audit.json'),('receipt_hash','receipt.json'),('launch_hash','launch.json'),('observation_hash','execution_stats.json')]})
    ex=dict(id=eid,task_id='task-fixture',proposal=proposal,result=result,status='completed',pid=launch['pid'])
    research=SimpleNamespace(project=project,root=root,store=SimpleNamespace(experiment=lambda _:copy.deepcopy(ex)))
    return research,ex,job


def test_positive_completed_identity_never_deserializes_or_predicts(tmp_path,monkeypatch):
    import joblib
    monkeypatch.setattr(joblib,'load',lambda *a,**k:pytest.fail('checkpoint load forbidden'))
    r,ex,job=fixture_record(tmp_path)
    identity=verify_completed_learning_identity(r,ex['id'])
    assert identity['numeric_model_reproduction'] is False
    assert identity['additional_predict_calls']==identity['additional_model_fits']==0
    assert identity['identity_id']==content_id({k:v for k,v in identity.items() if k!='identity_id'})
    assert identity['producer_files']['src/alpharesearch/models.py']['text'].startswith('# synthetic')


@pytest.mark.parametrize('change',['input','prediction','receipt','audit','code','ready','db_status','db_verified','cancel','supervision','observation','unsupported_schema','launch','extra_output'])
def test_changed_completed_evidence_is_rejected(change,tmp_path):
    r,ex,job=fixture_record(tmp_path)
    baseline=verify_completed_learning_identity(r,ex['id'])
    assert baseline['experiment_id']==ex['id'] and baseline['additional_predict_calls']==0
    if change=='input':(r.project/'research/input.txt').write_text('changed input')
    elif change=='prediction':(job/'learn_result/predictions.parquet').write_bytes(b'changed scores')
    elif change=='receipt':(job/'learn_result/learning_receipt.json').write_text('{}')
    elif change=='audit':(job/'alpha_audit.json').write_text('{}')
    elif change=='code':(job/'code/src/alpharesearch/models.py').write_text('# changed')
    elif change=='ready':(job/'ready.json').write_text('{}')
    elif change=='db_status':ex['status']='failed'
    elif change=='db_verified':ex['result']['verified']=False
    elif change=='cancel':write_json(job/'cancel_request.json',dict(reason='late cancellation'))
    elif change=='supervision':write_json(job/'supervision_failure.json',dict(error='late failure'))
    elif change=='observation':(job/'execution_stats.json').write_text('{}')
    elif change=='unsupported_schema':ex['proposal']['spec']['schema']='unsupported'
    elif change=='launch':(job/'launch.json').write_text('{}')
    else:(job/'learn_result/unlisted.txt').write_text('extra')
    reasons={'input':'Frozen input content changed','prediction':'快照完整性检查失败','receipt':'快照完整性检查失败','audit':'Completed producer metadata changed','code':'快照完整性检查失败','ready':'Frozen batch job identity differs','db_status':'Completed audited registered learning required','db_verified':'Completed audited registered learning required','cancel':'Cancellation or supervision failure','supervision':'Cancellation or supervision failure','observation':'Completed producer metadata changed','unsupported_schema':'Unsupported completed learning producer schema','launch':'Completed producer metadata changed','extra_output':'Completed learning output file set differs'}
    with pytest.raises(ValueError,match=reasons[change]):verify_completed_learning_identity(r,ex['id'])


def test_database_changes_during_read_are_rejected(tmp_path):
    r,ex,job=fixture_record(tmp_path);calls=[]
    def records(_):
        calls.append(1);value=copy.deepcopy(ex)
        if len(calls)>1:value['result']['audit_id']='0'*64
        return value
    r.store.experiment=records
    with pytest.raises(ValueError,match='database record changed'):verify_completed_learning_identity(r,ex['id'])

@pytest.fixture(autouse=True)
def forbid_model_calls(monkeypatch):
    import joblib
    from sklearn.linear_model import Ridge,ElasticNet,HuberRegressor
    from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    def forbidden(*args,**kwargs):pytest.fail('Completed identity must not fit, predict or load checkpoint')
    for cls in [Ridge,ElasticNet,HuberRegressor,RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor,MLPRegressor]:
        monkeypatch.setattr(cls,'fit',forbidden);monkeypatch.setattr(cls,'predict',forbidden)
    monkeypatch.setattr(joblib,'load',forbidden)
