"""Unfitted real-provenance path tested with synthetic market inputs only."""
import copy,json,hashlib
from pathlib import Path
import pandas as pd
import pytest
from tests.test_alpha_account_registration import setup
from src.alpharesearch.fixed_rules import build_fixed_rule_result,validate_fixed_rule_result
from src.alpharesearch.portfolio import build_score_portfolio,ScorePortfolioSpec
from src.alpharesearch.audit import audit_score_portfolio
from src.researchops.account_experiments import inspect_inputs
from src.researchops.account_intents import count
from src.technical.artifacts import write_json,digest


def fixed_case(tmp_path,kind='negative_reported_market_cap_proxy'):
    ops,session,p,folder=setup(tmp_path)
    files={x['role']:ops.project/x['path'] for x in p['inputs']}
    old=json.loads(files['learning_receipt'].read_text(encoding='utf-8'))
    rows=pd.read_parquet(files['predictions'])
    inputs=rows[['sample_id','trade_date','stock_code','decision_at','execution_at','target_end_at']].copy()
    cfg=dict(schema='fixed-rule-v1',kind=kind,seed=11 if kind=='stable_stock_hash' else None,
        universe_id=old['fit_identity']['settings']['universe_id'],value_definition='financial.reported_market_cap_proxy' if kind!='stable_stock_hash' else None)
    if kind=='negative_reported_market_cap_proxy':
        mapping={stock:(i+1)*100. for i,stock in enumerate(sorted(inputs.stock_code.unique()))}
        inputs['value']=inputs.stock_code.map(mapping);inputs['value_known_at']=inputs.decision_at
    result=build_fixed_rule_result(inputs,cfg,max_rows=100)
    result.predictions.to_parquet(files['predictions'],index=False);write_json(files['learning_receipt'],result.receipt)
    files['rule_inputs']=folder/'rule_inputs.parquet';inputs.to_parquet(files['rule_inputs'],index=False)
    p['spec'].update(schema='registered-alpha-account-v4',score_source=dict(kind='fixed_rule',rule_id=result.receipt['rule_id']))
    p['inputs']=[dict(role=role,path=path.relative_to(ops.project).as_posix(),sha256=digest(path),bytes=path.stat().st_size) for role,path in files.items()]
    p['hypothesis']='Synthetic fixed rule, genuinely no fitted model'
    return ops,session,p,folder,result,files


def test_hand_calculated_cap_ranks_and_independent_selection(tmp_path):
    ops,_,p,_,result,files=fixed_case(tmp_path)
    assert 'fit_id' not in result.predictions and 'fit_identity' not in result.receipt and 'fitted_model' not in result.receipt
    assert result.receipt['model_fits']==0 and count(dict(p,planned_accounts=2))==(2,0)
    assert result.predictions.score.tolist()==(-result.rule_inputs.value).tolist()
    refs=pd.read_parquet(files['references']);calendar=json.loads(files['calendar'].read_text(encoding='utf-8'))
    portfolio=build_score_portfolio(result,refs,calendar,ScorePortfolioSpec.from_dict(p['spec']['portfolio']))
    for _,rows in portfolio.decisions.groupby('trade_date'):
        assert rows.loc[rows.selected,'stock_code'].tolist()==sorted(rows.stock_code.unique())[:2]
    audit_score_portfolio(result,refs,calendar,portfolio)
    assert portfolio.receipt['source_fit_id'] is None and portfolio.receipt['source_kind']=='fixed_rule'
    inspect_inputs(ops.project,p,ops.root)


def test_stable_seed_means_stock_preference_not_eleven_stocks_or_weekly_lottery(tmp_path):
    _,_,_,_,result,_=fixed_case(tmp_path,'stable_stock_hash')
    assert result.predictions.groupby('stock_code').score.nunique().eq(1).all()
    for row in result.predictions.itertuples():
        expected=int(hashlib.sha256(('11:'+row.stock_code).encode()).hexdigest()[:13],16)/16**13
        assert row.score==expected


@pytest.mark.parametrize('fault',['future_value','missing_value','nonpositive','repeated_keys','resealed_score','pretend_learning','wrong_source_kind'])
def test_bad_sources_never_register(tmp_path,fault):
    ops,session,p,_,result,files=fixed_case(tmp_path)
    if fault in {'future_value','missing_value','nonpositive','repeated_keys'}:
        data=result.rule_inputs.copy()
        if fault=='future_value':data.loc[0,'value_known_at']='2099-01-01T00:00:00Z'
        elif fault=='missing_value':data.loc[0,'value']=float('nan')
        elif fault=='nonpositive':data.loc[0,'value']=0.
        else:data=pd.concat([data,data.iloc[:1]],ignore_index=True)
        with pytest.raises(ValueError):build_fixed_rule_result(data,result.receipt['rule_identity']['spec'],max_rows=100)
    else:
        if fault=='resealed_score':
            result.predictions.loc[0,'score']+=1e-10
            from src.alpharesearch.learning import _fingerprint
            result.receipt['prediction_outputs']=_fingerprint(result.predictions)
            result.predictions.to_parquet(files['predictions'],index=False);write_json(files['learning_receipt'],result.receipt)
            for item in p['inputs']:
                path=files[item['role']];item.update(sha256=digest(path),bytes=path.stat().st_size)
        elif fault=='pretend_learning':p['spec']['score_source']=dict(kind='registered_learning',experiment_id='exp-'+'a'*12,verification_mode='reuse_completed_identity_v1')
        else:p['spec']['schema']='registered-alpha-account-v1'
        with pytest.raises(ValueError):ops.register(session,p)
    assert not ops.store.get(session['task_id'])['experiments']


@pytest.mark.parametrize('dynamic',[False,True])
def test_actual_synthetic_rule_account_archives_inputs_without_any_model_job(tmp_path,dynamic):
    ops,session,p,_,result,_=fixed_case(tmp_path)
    if dynamic:
        from src.alpharesearch.holding_policy import ActualHoldingPolicy
        p['spec']['holding_policy']=ActualHoldingPolicy(max_positions=p['spec']['portfolio']['top_k'],new_entry_rank=p['spec']['portfolio']['top_k'],retain_rank=3,allocation=p['spec']['portfolio']['allocation']).to_dict()
    ex=ops.register(session,p)
    assert ex['proposal']['planned_model_fits']==0
    done=ops.execute(session,ex['id'])
    assert done['status']=='completed',done.get('error')
    assert len(ops.store.get(session['task_id'])['experiments'])==1
    native=ops.root/'runs'/done['run_id']
    assert (native/'alpha_rule_inputs.parquet').is_file()
    assert (native/'alpha_holding_policy.json').exists()==dynamic
    definition=json.loads((native/'definition.json').read_text(encoding='utf-8'))
    assert '没有拟合模型' in definition['momentum']
    receipt=json.loads((native/'alpha_learning_receipt.json').read_text(encoding='utf-8'))
    assert receipt['schema']=='fixed-rule-score-receipt-v1' and receipt['model_fits']==0
    assert 'fit_identity' not in receipt
    archived=pd.read_parquet(native/'alpha_rule_inputs.parquet')
    pd.testing.assert_frame_equal(archived,result.rule_inputs)
