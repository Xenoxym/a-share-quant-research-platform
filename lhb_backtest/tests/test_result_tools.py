"""Adapter unit tests; mocked completion audits are not evidence of real models."""
from copy import deepcopy
from types import SimpleNamespace
import json
import pandas as pd
import pytest
from src.researchops import result_tools as tools
from src.technical.artifacts import write_json,digest
from tests.test_split_diagnostics import example


def test_registered_diagnostics_wiring_reports_missing_splits(tmp_path,monkeypatch):
    project=tmp_path;root=project/'data/technical';eid='exp-'+'a'*12
    output=root/'worker_jobs'/eid/'learn_result';output.mkdir(parents=True)
    s,l,_=example();s=s.loc[s.trade_date!='2024-01-02'];l=l.loc[l.trade_date!='2024-01-02']
    s.to_parquet(output/'predictions.parquet',index=False)
    path=project/'data/labels.parquet';l.to_parquet(path,index=False)
    ex=dict(proposal=dict(spec=dict(learning=dict(fit_cutoff='2024-01-08T00:00:00Z',predict_start='2024-01-09',predict_end='2024-01-16'),budget=dict(max_input_bytes=1000000)),
        inputs=[dict(role='labels',path='data/labels.parquet',sha256=digest(path),bytes=path.stat().st_size)]))
    research=SimpleNamespace(project=project,root=root,store=SimpleNamespace(experiment=lambda _:deepcopy(ex)))
    write_json(output/'learning_receipt.json',dict(schema='causal-learning-receipt-v2'))
    identity=dict(identity_id='test-only',predictions_sha256=digest(output/'predictions.parquet'),learning_receipt_sha256=digest(output/'learning_receipt.json'))
    monkeypatch.setattr(tools,'verify_completed_learning_identity',lambda *a:identity)
    result=tools.diagnose_registered_learning(research,eid,max_rows=100)
    assert result['diagnostics']['missing_splits']==['train','validation']
    assert result['diagnostics']['segments'][0]['mature_rows']==6
    assert result['diagnostics']['segments'][0]['unseen_claim'] is False
    for kind in ['date_rank','date_centered']:
        unit='dimensionless' if kind=='date_rank' else 'decimal_return'
        write_json(output/'learning_receipt.json',dict(schema='causal-learning-receipt-v3',score_unit=unit,predicted_score_is_raw_return=False,
            training_target_receipt=dict(spec=dict(kind=kind),unit=unit,predicted_score_is_raw_return=False)))
        identity['learning_receipt_sha256']=digest(output/'learning_receipt.json')
        transformed=tools.diagnose_registered_learning(research,eid,max_rows=100)
        assert transformed['diagnostics']['segments'][0]['metrics']['mse'] is None
    # Many extra labels must be refused before their pandas materialization.
    many=pd.concat([l]*20,ignore_index=True)
    many.to_parquet(path,index=False)
    ex['proposal']['inputs'][0].update(sha256=digest(path),bytes=path.stat().st_size)
    original=pd.read_parquet
    with monkeypatch.context() as m:
        def bounded_read(target,*args,**kwargs):
            if target==path:pytest.fail('Labels were loaded before their row-cap check')
            return original(target,*args,**kwargs)
        m.setattr(pd,'read_parquet',bounded_read)
        with pytest.raises(ValueError,match='before loading'):
            tools.diagnose_registered_learning(research,eid,max_rows=100)
    l.to_parquet(path,index=False);ex['proposal']['inputs'][0].update(sha256=digest(path),bytes=path.stat().st_size)
    identity['predictions_sha256']='0'*64
    with pytest.raises(ValueError,match='changed'):tools.diagnose_registered_learning(research,eid,max_rows=100)


def test_account_index_groups_conditions_not_names_or_scores(tmp_path,monkeypatch):
    root=tmp_path/'data/technical';records={};folders={}
    for i in range(3):
        eid='exp-'+str(i)*12;folder=root/'runs'/eid;folder.mkdir(parents=True);folders[eid]=folder
        spec=dict(portfolio=dict(name='display'+str(i),top_k=100),account=dict(strategy=dict(name='display'+str(i)),fee=1 if i==2 else 0),
            scenarios=['zero_transaction_cost','configured'],snapshot_id='same',snapshot_manifest_hash='same',score_source=dict(kind='registered_learning',experiment_id='different'+str(i)))
        records[eid]=dict(status='completed',proposal=dict(kind='alpha_account',spec=spec))
        job=root/'worker_jobs'/eid;job.mkdir(parents=True);write_json(job/'input.json',dict(experiment_id=eid))
        write_json(folder/'alpha_portfolio_receipt.json',dict(account_eligible_membership='same',reference_inputs='same',calendar_id='same',planned_horizons='same',implementation_hashes={'source':'same'}))
        write_json(folder/'alpha_account_receipt.json',dict(code_bundle_id='same'))
        write_json(folder/'result.json',dict(code_hash='same',portfolios=[dict(metrics=dict(total_return=i/10))]))
        write_json(folder/'alpha_scoring_code.json',dict(verification_mode='reuse_completed_identity_v1',producer_identity='different'+str(i),consumer_files={'core':dict(sha256='same')}))
        rows=pd.DataFrame(dict(trade_date=['2024-01-05']*2,stock_code=['600001.SH','600002.SH'],reason=['eligible_external_score']*2))
        if i==1:rows=rows.iloc[::-1]
        rows.to_parquet(folder/'decisions.parquet',index=False)
    research=SimpleNamespace(root=root,store=SimpleNamespace(experiment=lambda eid:deepcopy(records[eid])))
    monkeypatch.setattr(tools,'verify_completed',lambda *a:None)
    monkeypatch.setattr(tools,'native_folder',lambda cfg:folders[cfg['experiment_id']])
    report=tools.account_comparison_index(research,list(records))
    assert sorted(len(x['accounts']) for x in report['groups'])==[1,2]
    assert report['account_backtests']==0 and 'winner' not in report
    with pytest.raises(ValueError):tools.account_comparison_index(research,[next(iter(records))]*2)


def test_reports_are_bounded_project_research_files_without_overwrite(tmp_path):
    research=SimpleNamespace(project=tmp_path)
    with pytest.raises(ValueError,match='cap'):tools.write_result_report(research,'research/tiny.json',{'long':'x'*100},max_bytes=2)
    assert not (tmp_path/'research/tiny.json').exists()
    result=tools.write_result_report(research,'research/good.json',{'ok':True},max_bytes=100)
    assert result['model_fits']==0
    with pytest.raises(ValueError,match='Existing'):tools.write_result_report(research,'research/good.json',{},max_bytes=100)
    with pytest.raises(ValueError,match='escaped'):tools.write_result_report(research,'outside.json',{},max_bytes=100)
