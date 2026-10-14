"""Public fixed-rule preparation is finite and does not train/register."""
import copy,json
import pandas as pd
import pytest
from tests.test_fixed_rule_accounts import fixed_case
from src.researchops.fixed_rule_proposal import save_fixed_rule_scores,account_from_fixed_rules
from src.researchops.account_experiments import inspect_inputs


def test_generator_proposal_preserves_account_conditions_and_registers(tmp_path):
    ops,session,template,_,result,files=fixed_case(tmp_path)
    original=copy.deepcopy(template)
    info=save_fixed_rule_scores(ops,files['rule_inputs'].relative_to(ops.project).as_posix(),result.receipt['rule_identity']['spec'],
        'research/fixed-scores',max_rows=100,max_input_bytes=1000000,max_output_bytes=1000000)
    assert info['model_fits']==info['predict_calls']==info['registered_experiments']==0
    assert not ops.store.get(session['task_id'])['experiments']
    proposal=account_from_fixed_rules(ops,'research/fixed-scores',template)
    assert template==original
    for key in ['snapshot_id','snapshot_manifest_hash','portfolio','account','scenarios','budget']:
        assert proposal['spec'][key]==template['spec'][key]
    assert proposal['spec']['score_source']==dict(kind='fixed_rule',rule_id=result.receipt['rule_id'])
    inspect_inputs(ops.project,proposal,ops.root)
    ex=ops.register(session,proposal)
    assert ex['proposal']['planned_model_fits']==0
    with pytest.raises(ValueError,match='Existing matrix'):
        save_fixed_rule_scores(ops,files['rule_inputs'].relative_to(ops.project).as_posix(),result.receipt['rule_identity']['spec'],
            'research/fixed-scores',max_rows=100,max_input_bytes=1000000,max_output_bytes=1000000)


def test_generator_preflight_budget_and_partial_output_never_publish_receipt(tmp_path):
    ops,_,_,_,result,files=fixed_case(tmp_path)
    source=files['rule_inputs'].relative_to(ops.project).as_posix()
    with pytest.raises(ValueError,match='before loading'):
        save_fixed_rule_scores(ops,source,result.receipt['rule_identity']['spec'],'research/too-many',max_rows=1,max_input_bytes=1000000,max_output_bytes=1000000)
    assert not (ops.project/'research/too-many').exists()
    with pytest.raises(ValueError,match='incomplete files retained'):
        save_fixed_rule_scores(ops,source,result.receipt['rule_identity']['spec'],'research/partial',max_rows=100,max_input_bytes=1000000,max_output_bytes=1)
    assert not (ops.project/'research/partial/fixed_rule_receipt.json').exists()
    assert (ops.project/'research/partial/rule_inputs.parquet').is_file()


def test_proposal_checks_byte_cap_before_parsing_receipt(tmp_path,monkeypatch):
    ops,_,template,_,result,files=fixed_case(tmp_path)
    save_fixed_rule_scores(ops,files['rule_inputs'].relative_to(ops.project).as_posix(),result.receipt['rule_identity']['spec'],
        'research/fixed-scores',max_rows=100,max_input_bytes=1000000,max_output_bytes=1000000)
    template['spec']['budget']['max_input_bytes']=1
    (ops.project/'research/fixed-scores/fixed_rule_receipt.json').write_text('invalid JSON',encoding='utf-8')
    with pytest.raises(ValueError,match='before receipt loading'):
        account_from_fixed_rules(ops,'research/fixed-scores',template)


def test_resealed_rule_score_is_rejected_by_registration(tmp_path):
    ops,session,template,_,result,files=fixed_case(tmp_path)
    save_fixed_rule_scores(ops,files['rule_inputs'].relative_to(ops.project).as_posix(),result.receipt['rule_identity']['spec'],
        'research/fixed-scores',max_rows=100,max_input_bytes=1000000,max_output_bytes=1000000)
    path=ops.project/'research/fixed-scores/predictions.parquet'
    data=pd.read_parquet(path);data.loc[0,'score']+=1.;data.to_parquet(path,index=False)
    proposal=account_from_fixed_rules(ops,'research/fixed-scores',template)
    with pytest.raises(ValueError):ops.register(session,proposal)
    assert not ops.store.get(session['task_id'])['experiments']
