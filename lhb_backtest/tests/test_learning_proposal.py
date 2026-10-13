"""Matrix binding is a cheap proposal step, not a substitute for registration."""
import copy,json,shutil,subprocess,sys
from pathlib import Path
import pandas as pd
import pytest
from src.researchops.learning_proposal import learning_from_matrix,MATRIX_FILES
from src.researchops.learn_experiments import inspect_inputs
from src.technical.artifacts import write_json
from tests.test_alpha_learn_registration import setup


def matrix_case(tmp_path):
    ops,session,proposal,folder=setup(tmp_path)
    new=ops.project/'data/new_matrix';new.mkdir()
    for name in MATRIX_FILES.values():shutil.copy2(folder/name,new/name)
    return ops,session,proposal,new


def test_exact_template_choices_and_preparation_preserved_without_reading_values(tmp_path,monkeypatch):
    ops,session,proposal,new=matrix_case(tmp_path)
    expected=copy.deepcopy(proposal)
    for record in expected['inputs']:
        if record['role'] in MATRIX_FILES:record['path']='data/new_matrix/'+MATRIX_FILES[record['role']]
    def forbidden(*args,**kwargs):raise AssertionError('Proposal adapter decoded full table')
    with monkeypatch.context() as patched:
        patched.setattr(pd,'read_parquet',forbidden)
        actual=learning_from_matrix(ops,'data/new_matrix',proposal)
    assert actual==expected and proposal!=actual
    before=inspect_inputs(ops.project,proposal)[-1]
    after=inspect_inputs(ops.project,actual)[-1]
    assert before==after
    ex=ops.register(session,actual)
    assert ex['status']=='planned' and ex['proposal']['inspection']==before
    with ops.store.connection() as db:
        assert db.execute("SELECT count(*) FROM experiments WHERE task_id=?",(session["task_id"],)).fetchone()[0]==1


@pytest.mark.parametrize('mutation',['receipt','feature','budget','path','nonpublic'])
def test_bad_matrix_or_template_rejected(tmp_path,mutation):
    ops,_,proposal,new=matrix_case(tmp_path);folder='data/new_matrix'
    if mutation=='receipt':
        document=json.loads((new/'receipt.json').read_text());document['rows']+=1;write_json(new/'receipt.json',document)
    if mutation=='feature':proposal['spec']['learning']['features']=['absent.feature']
    if mutation=='budget':proposal['spec']['budget']['max_input_bytes']=1
    if mutation=='path':folder='../outside'
    if mutation=='nonpublic':proposal['inspection']={}
    with pytest.raises(ValueError):learning_from_matrix(ops,folder,proposal)


def test_cli_writes_reuses_and_refuses_different_existing_proposal(tmp_path):
    ops,_,proposal,_=matrix_case(tmp_path)
    template=ops.project/'research/template.json';template.parent.mkdir();write_json(template,proposal)
    output=ops.project/'research/learning.json'
    args=[sys.executable,str(Path(__file__).resolve().parents[1]/'tools/execution.py'),'--project',str(ops.project),
        'learning-proposal','--matrix','data/new_matrix','--template',str(template),'--output',str(output)]
    def call():return subprocess.run(args,capture_output=True,text=True,encoding='utf-8',timeout=60)
    first=call();assert first.returncode==0,first.stderr+first.stdout
    result=json.loads(first.stdout)['data']
    assert result['action']=='created' and result['model_fits']==result['registered_experiments']==0
    assert json.loads(call().stdout)['data']['action']=='reused'
    before=output.read_bytes();template_document=json.loads(template.read_text());template_document['hypothesis']='new question';write_json(template,template_document)
    failed=call();assert failed.returncode==1 and 'refusing overwrite' in failed.stdout
    assert output.read_bytes()==before
