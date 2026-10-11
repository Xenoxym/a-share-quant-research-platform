"""Real-holding counterexamples; scores are explicit engineering values."""
import json
from dataclasses import FrozenInstanceError
import pandas as pd
import pytest
from src.alpharesearch.holding_policy import (ActualHolding, SignalCloseState, ActualHoldingPolicy,
                                            select_actual_holdings, identity)
from src.alpharesearch.holding_policy_audit import audit_actual_selection

SD, ED='2024-01-12','2024-01-15'
def rows(n=151):
    return pd.DataFrame([dict(stock_code=f'{600000+i:06}.SH',trade_date=SD,execution_date=ED,
        score=float(n-i),close=10.,reason='eligible_external_score',rank=9999,selected=False)
        for i in range(n)])
def state(codes=(), scenario='configured'):
    held=tuple(ActualHolding(code,100,10.,SD,'2024-01-08') for code in sorted(codes))
    return SignalCloseState(SD,scenario,1000.,0.,1000.+1000*len(held),held)
def select(r,s,p=None):
    p=p or ActualHoldingPolicy()
    return select_actual_holdings(r,s,p,signal_date=SD,execution_date=ED)

def test_never_bought_previous_desire_does_not_become_retention():
    r=rows();r.loc[r.index==129,'selected']=True
    result=select(r,state())
    assert '600129.SH' not in {x.stock_code for x in result.targets}
    assert len(result.targets)==100 and all(x.reason=='new_entry' for x in result.targets)

def test_actual_rank150_retained_and_rank151_exits():
    r=rows();s=state(['600149.SH','600150.SH']);choice=select(r,s)
    assert '600149.SH' in {x.stock_code for x in choice.targets}
    assert choice.exits[0].stock_code=='600150.SH' and choice.exits[0].reason=='outside_retain_rank'
    assert len(choice.targets)==100
    assert audit_actual_selection(r,s.to_dict(),ActualHoldingPolicy().to_dict(),choice.to_dict())['production_selector_called'] is False

def test_ties_use_code_and_ignore_saved_rank_or_selected_flag():
    r=rows(4).iloc[::-1].copy();r['score']=1.;r['rank']=0;r['selected']=False
    p=ActualHoldingPolicy(2,2,3,.8)
    x=select(r,state(),p)
    assert [t.stock_code for t in x.targets]==['600000.SH','600001.SH']
    assert [t.target_weight for t in x.targets]==[.4,.4]

@pytest.mark.parametrize('missing',[True,False])
def test_missing_or_ineligible_actual_holding_exits(missing):
    r=rows(4);s=state(['600003.SH'])
    if missing:r=r.loc[r.stock_code.ne('600003.SH')]
    else:r.loc[r.stock_code.eq('600003.SH'),'reason']='ineligible_declared'
    x=select(r,s)
    assert x.exits[0].actual_shares==100 and x.exits[0].reason=='no_eligible_score'

@pytest.mark.parametrize('change',['future_quote','future_purchase','duplicate','zero_shares','negative_cash','equity'])
def test_invalid_actual_state_is_rejected(change):
    h=ActualHolding('600000.SH',100,10.,SD,'2024-01-08')
    with pytest.raises(ValueError):
        if change=='future_quote':SignalCloseState(SD,'configured',1000.,0.,2000.,(ActualHolding('600000.SH',100,10.,ED,SD),))
        elif change=='future_purchase':SignalCloseState(SD,'configured',1000.,0.,2000.,(ActualHolding('600000.SH',100,10.,SD,ED),))
        elif change=='duplicate':SignalCloseState(SD,'configured',1000.,0.,3000.,(h,h))
        elif change=='zero_shares':ActualHolding('600000.SH',0,10.,SD,SD)
        elif change=='negative_cash':SignalCloseState(SD,'configured',-1.,0.,999.,(h,))
        else:SignalCloseState(SD,'configured',1000.,0.,9999.,(h,))

@pytest.mark.parametrize('change',['date','entry','duplicate','nan','zero_close'])
def test_invalid_score_date_membership_or_number_rejected(change):
    r=rows(4)
    if change=='date':r.loc[0,'trade_date']=ED
    elif change=='entry':r.loc[0,'execution_date']=SD
    elif change=='duplicate':r.loc[1,'stock_code']=r.loc[0,'stock_code']
    elif change=='nan':r.loc[0,'score']=float('nan')
    else:r.loc[0,'close']=0.
    with pytest.raises(ValueError):select(r,state())

def test_actual_state_is_immutable_and_scenario_changes_identity():
    s=state(['600000.SH'])
    with pytest.raises(FrozenInstanceError):s.cash=0.
    assert s.state_id!=state(['600000.SH'],'zero_transaction_cost').state_id

@pytest.mark.parametrize('change',['retained_reason','rank','exit_reason','state_id'])
def test_independent_check_rejects_resigned_selection_semantics(change):
    r=rows();s=state(['600150.SH']);p=ActualHoldingPolicy();x=json.loads(json.dumps(select(r,s,p).to_dict()))
    if change=='retained_reason':x['targets'][0]['reason']='retain_actual_holding'
    elif change=='rank':x['targets'][0]['rank']=2
    elif change=='exit_reason':x['exits'][0]['reason']='no_eligible_score'
    else:x['state_id']='0'*64
    with pytest.raises(ValueError):audit_actual_selection(r,s.to_dict(),p.to_dict(),x)
