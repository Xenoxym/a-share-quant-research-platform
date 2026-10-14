"""Synthetic clock tests; these fixtures are not real registered model results."""
import json
from datetime import datetime
import pandas as pd
import pytest
from tests.test_alpha_account_audit import inputs
from src.researchops.prediction_seal import _copy_seal,verify_prediction_seal,seal_registered_predictions
from src.technical.artifacts import write_json,digest


def case(tmp_path):
    data,_,account=inputs(tmp_path/'technical');result,_,_,policy=data
    scores=result.predictions.loc[result.predictions.trade_date.eq(result.predictions.trade_date.min())]
    source=tmp_path/'source';source.mkdir();pred=source/'p.parquet';receipt=source/'r.json'
    scores.to_parquet(pred,index=False);write_json(receipt,result.receipt)
    identity=dict(experiment_id='exp-'+'a'*12,identity_id='software-test',predictions_sha256=digest(pred),learning_receipt_sha256=digest(receipt))
    protocol=dict(schema='prospective-account-rules-v1',portfolio=policy.to_dict(),account=account.to_dict(),
        scenarios=['zero_transaction_cost','configured'],evaluation_end='2024-01-31T00:00:00Z')
    return pred,receipt,identity,protocol


def run(args,folder,clocks=None,**overrides):
    clocks=iter(clocks or ['2024-01-06T00:00:00+00:00']*2)
    cfg=dict(max_bytes=1000000,max_rows=100,clock=lambda:datetime.fromisoformat(next(clocks)));cfg.update(overrides)
    return _copy_seal(*args,folder,**cfg)


def test_successful_copy_is_offline_verifiable_and_does_not_read_outcomes(tmp_path):
    args=case(tmp_path);folder=tmp_path/'seal';record=run(args,folder)
    assert record['additional_fits']==record['additional_predict_calls']==0 and record['outcomes_read'] is False
    assert verify_prediction_seal(folder,max_bytes=1000000,max_rows=100)==record
    assert pd.read_parquet(folder/'predictions.parquet').equals(pd.read_parquet(args[0]))
    with pytest.raises(ValueError,match='Existing'):run(args,folder)
    (folder/'protocol.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='artifact differs'):verify_prediction_seal(folder,max_bytes=1000000,max_rows=100)


@pytest.mark.parametrize('fault',['backdate','already_executed','cross_entry_during_copy','row_budget','byte_budget','changed_source','changed_rule'])
def test_invalid_seal_never_publishes_record(tmp_path,fault):
    args=list(case(tmp_path));folder=tmp_path/'seal';extra={};clocks=None
    if fault=='backdate':clocks=['2024-01-04T00:00:00Z']*2
    elif fault=='already_executed':clocks=['2024-01-08T02:00:00Z']*2
    elif fault=='cross_entry_during_copy':clocks=['2024-01-08T01:29:59Z','2024-01-08T01:30:00Z']
    elif fault=='row_budget':extra['max_rows']=1
    elif fault=='byte_budget':extra['max_bytes']=1
    elif fault=='changed_source':args[2]['predictions_sha256']='0'*64
    elif fault=='changed_rule':args[3]['scenarios']=['configured']
    with pytest.raises(ValueError):run(args,folder,clocks,**extra)
    assert not (folder/'seal.json').exists()


def test_public_api_has_no_backdate_parameter():
    import inspect
    assert 'clock' not in inspect.signature(seal_registered_predictions).parameters
    assert 'sealed_at' not in inspect.signature(seal_registered_predictions).parameters


def test_source_swap_during_copy_cannot_be_resealed_as_original_identity(tmp_path,monkeypatch):
    import src.researchops.prediction_seal as module
    args=case(tmp_path);folder=tmp_path/'seal';original=module.shutil.copyfile
    def replace_then_copy(source,destination):
        if source==args[0]:
            changed=pd.read_parquet(source);changed['score']+=1.;changed.to_parquet(source,index=False)
        return original(source,destination)
    monkeypatch.setattr(module.shutil,'copyfile',replace_then_copy)
    with pytest.raises(ValueError,match='completed identity'):run(args,folder)
    assert not (folder/'seal.json').exists()
