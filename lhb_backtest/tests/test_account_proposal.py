"""Saved-score proposal wiring; fixtures are synthetic identity evidence, no strategy fits."""
import copy
from pathlib import Path
import pytest
from src.researchops.account_proposal import account_from_learning,validate_account_template
from src.researchops.workflow import validate_plan
from src.technical.artifacts import digest
from tests.test_completed_learning_identity import fixture_record
from tests.test_score_reuse_versions import account_protocol
from tests.test_alpha_portfolio import scored_example


def example_template(research):
    spec=account_protocol(scored_example())
    records=[]
    (research.project/"research").mkdir(exist_ok=True)
    for role in ['predictions','learning_receipt','references','calendar']:
        path=research.project/'research'/('template_'+role+'.txt');path.write_text('fixed '+role,encoding='utf-8')
        records.append(dict(role=role,path=path.relative_to(research.project).as_posix(),sha256=digest(path),bytes=path.stat().st_size))
    return dict(kind='alpha_account',hypothesis='fixed model contrast',expected_observation='same dates and costs',
        falsification='no return/drawdown improvement',spec=spec,inputs=records,timeout_seconds=60)


def test_saved_identity_rebinds_only_producer_and_name_without_template_mutation(tmp_path):
    research,ex,job=fixture_record(tmp_path)
    template=example_template(research);before=copy.deepcopy(template)
    proposal=account_from_learning(research,ex['id'],template)
    assert template==before
    assert proposal['spec']['score_source']['experiment_id']==ex['id']
    for role in ['references','calendar']:
        assert next(x for x in proposal['inputs'] if x['role']==role)==next(x for x in before['inputs'] if x['role']==role)
    for role,name in [('predictions','predictions.parquet'),('learning_receipt','learning_receipt.json')]:
        record=next(x for x in proposal['inputs'] if x['role']==role)
        assert record['sha256']==digest(job/'learn_result'/name)
        assert (research.project/record['path']).resolve()==(job/'learn_result'/name).resolve()
    for section in ['budget','scenarios','snapshot_id','snapshot_manifest_hash']:
        assert proposal['spec'][section]==before['spec'][section]
    assert proposal['spec']['account']['execution']==before['spec']['account']['execution']


@pytest.mark.parametrize('fault',['producer_bytes','failed_producer','reference_bytes','duplicate_role','private_metadata','v1'])
def test_bad_producer_or_changed_template_cannot_create_proposal(tmp_path,fault):
    research,ex,job=fixture_record(tmp_path);template=example_template(research)
    if fault=='producer_bytes':(job/'learn_result/predictions.parquet').write_bytes(b'changed')
    if fault=='failed_producer':ex['status']='failed'
    if fault=='reference_bytes':(research.project/'research/template_references.txt').write_text('changed')
    if fault=='duplicate_role':template['inputs'][-1]=copy.deepcopy(template['inputs'][0])
    if fault=='private_metadata':template['engine_inventory']={}
    if fault=='v1':template['spec']['schema']='registered-alpha-account-v1'
    with pytest.raises(ValueError):account_from_learning(research,ex['id'],template)


@pytest.mark.parametrize('fault',['v1_plan','missing_dependency','future_dependency','extra_field','wrong_template'])
def test_derivation_schema_rejects_ambiguous_or_unbound_steps(tmp_path,fault):
    research,_,_=fixture_record(tmp_path);template=example_template(research)
    value=dict(schema='execution-plan-v2',id='fixed',task_id='task-fixed',max_starts=1,timeout_seconds=60,
        steps=[dict(id='learn',needs=[],experiment_id='exp-aaaaaaaaaaaa'),
            dict(id='account',needs=['learn'],account_from_learning=dict(learning_step='learn',template=template))])
    if fault=='v1_plan':value['schema']='execution-plan-v1'
    if fault=='missing_dependency':value['steps'][1]['needs']=[]
    if fault=='future_dependency':value['steps'][1]['account_from_learning']['learning_step']='future'
    if fault=='extra_field':value['steps'][1]['account_from_learning']['shell']='echo bad'
    if fault=='wrong_template':template['kind']='alpha_learn'
    with pytest.raises(ValueError):validate_plan(value)


def test_plan_v2_retains_same_template_and_explicit_dependency(tmp_path):
    research,_,_=fixture_record(tmp_path);template=example_template(research)
    value=dict(schema='execution-plan-v2',id='fixed',task_id='task-fixed',max_starts=1,timeout_seconds=60,
        steps=[dict(id='learn',needs=[],experiment_id='exp-aaaaaaaaaaaa'),
            dict(id='account',needs=['learn'],account_from_learning=dict(learning_step='learn',template=template))])
    assert validate_plan(value)==value


def test_cli_output_reuses_identical_and_refuses_different_or_escaped(tmp_path,monkeypatch,capsys):
    import json,sys
    from tools import execution
    research,ex,_=fixture_record(tmp_path);template=example_template(research)
    file=research.project/'research/template.json';file.write_text(json.dumps(template),encoding='utf-8')
    output=research.project/'research/derived.json'
    monkeypatch.setattr(execution,'Research',lambda *a,**k:research)
    args=['execution.py','account-proposal','--learning',ex['id'],'--template',str(file),'--output',str(output)]
    monkeypatch.setattr(sys,'argv',args)
    assert execution.main()==0
    first=json.loads(capsys.readouterr().out)['data']
    assert first['action']=='created' and first['model_fits']==0 and first['registered_experiments']==0
    original=output.read_bytes()
    assert execution.main()==0
    assert json.loads(capsys.readouterr().out)['data']['action']=='reused'
    assert output.read_bytes()==original
    template['spec']['account']['execution']['slippage_bps']+=1
    file.write_text(json.dumps(template),encoding='utf-8')
    with pytest.raises(ValueError,match='refusing overwrite'):execution.main()
    assert output.read_bytes()==original
    monkeypatch.setattr(sys,'argv',args[:-1]+[str(tmp_path/'escaped.json')])
    with pytest.raises(ValueError,match='ordinary project path'):execution.main()
    assert not (tmp_path/'escaped.json').exists()
