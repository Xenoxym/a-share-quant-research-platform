import copy
import pandas as pd
import pytest
from src.alpharesearch.learning import KEYS,CLOCKS,LABELS
from src.alpharesearch.split_diagnostics import diagnose_splits


def example():
    scores=[];labels=[]
    for day in ['2024-01-02','2024-01-09','2024-01-16']:
        for i in range(3):
            entry=(pd.Timestamp(day)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')+'T09:30:00+08:00'
            end=(pd.Timestamp(day)+pd.Timedelta(days=3)).strftime('%Y-%m-%d')+'T09:30:00+08:00'
            key=dict(sample_id=day+str(i),trade_date=day,stock_code=f'{600000+i}.SH')
            scores.append(dict(**key,decision_at=day+'T15:30:00+08:00',execution_at=entry,target_end_at=end,score=i/10))
            labels.append(dict(**key,target_return=i/10,label_start=entry,label_end=end,label_known_at=end,label_observed=True))
    parts=[dict(name=n,start=d,end=d,role='in_sample_replay' if n=='train' else 'post_fit_retrospective',used_for_selection=n=='validation')
        for n,d in zip(['train','validation','test'],['2024-01-02','2024-01-09','2024-01-16'])]
    return pd.DataFrame(scores),pd.DataFrame(labels),parts


def run(s,l,p):return diagnose_splits(s,l,p,fit_cutoff='2024-01-08T00:00:00Z',as_of='2024-01-18T00:00:00Z',max_rows=100,score_target=dict(kind='raw_return',unit='decimal_return',predicted_score_is_raw_return=True))


def test_hand_calculation_unknown_labels_and_missing_validation():
    s,l,p=example();l.loc[8,['target_return','label_start','label_end','label_known_at']]=[None]*4;l.loc[8,'label_observed']=False
    report=run(s,l,p)
    assert report['segments'][0]['metrics']==dict(mse=0.,mae=0.,rank_ic_mean=1.,rank_ic_dates=1)
    # Final labels have not matured by the as-of date; they are not zero returns.
    assert report['segments'][2]['pending_rows']==2 and report['segments'][2]['unobserved_rows']==1
    assert report['segments'][2]['metrics'] is None
    assert all(not x['tradable_history'] and not x['unseen_claim'] for x in report['segments'])
    partial=run(s.loc[s.trade_date!='2024-01-09'],l.loc[l.trade_date!='2024-01-09'],[p[0],p[2]])
    assert partial['missing_splits']==['validation']
    assert partial['model_fits']==partial['account_backtests']==0


@pytest.mark.parametrize('fault',['overlap','pretend_training_is_postfit','pre_fit_test','bad_horizon','duplicate','missing_status','late_training_label','uncovered','nan_score'])
def test_counterexamples(fault):
    s,l,p=example()
    if fault=='overlap':p[1]['start']=p[0]['start']
    elif fault=='pretend_training_is_postfit':p[0]['role']='post_fit_retrospective'
    elif fault=='pre_fit_test':p[0]['name']='test';p=p[:1];s=s.iloc[:3];l=l.iloc[:3];p[0]['role']='post_fit_retrospective'
    elif fault=='bad_horizon':l.loc[0,'label_end']='2024-01-06T09:30:00+08:00'
    elif fault=='duplicate':s=pd.concat([s,s.iloc[:1]])
    elif fault=='missing_status':l=l.iloc[1:]
    elif fault=='late_training_label':l.loc[0,'label_known_at']='2024-02-01T00:00:00Z'
    elif fault=='uncovered':p=p[:2]
    elif fault=='nan_score':s.loc[0,'score']=float('nan')
    with pytest.raises(ValueError):run(s,l,p)


@pytest.mark.parametrize('kind',['date_rank','date_centered','date_zscore'])
def test_nonraw_targets_never_report_raw_return_squared_error(kind):
    s,l,p=example();s['score']*=5
    target=dict(kind=kind,unit='decimal_return' if kind=='date_centered' else 'dimensionless',predicted_score_is_raw_return=False)
    result=diagnose_splits(s,l,p,fit_cutoff='2024-01-08T00:00:00Z',as_of='2024-02-01T00:00:00Z',max_rows=100,score_target=target)
    for part in result['segments']:
        assert part['metrics']['mse'] is None and part['metrics']['mae'] is None
        assert part['metrics']['rank_ic_mean']==1.
        assert 'differs from raw return' in part['error_metric_scope']
