"""Causality, target merging and execution boundaries for state allocation."""
from dataclasses import replace
import json
import threading
from urllib.request import urlopen

import numpy as np
import pandas as pd
import pytest

from src.technical.contracts import ResearchSpec, Strategy, Execution, Experiment, describe
from src.technical.regimes import FEATURES, gaussian_filter, infer_states, market_features, allocation_weights, combine_decisions
from src.technical.regime_lab import plan
from src.technical.market import Market
from src.technical.portfolio import simulate
from src.technical.server import make_server
from tests.test_technical import bars, master


def test_forward_filter_matches_bayes_recursion_and_never_revises_past():
    pi=np.array([.6,.4]);a=np.array([[.9,.1],[.2,.8]]);mu=np.array([[0.],[2.]]);var=np.ones((2,1))
    x=np.array([[.1],[1.9],[20.]])
    got=gaussian_filter(x,pi,a,mu,var)
    likelihood=np.exp(-.5*(x[0]-mu[:,0])**2);p=pi*likelihood;p/=p.sum()
    assert got[0]==pytest.approx(p)
    prior=p@a;expect=prior*np.exp(-.5*(x[1]-mu[:,0])**2);expect/=expect.sum()
    assert got[1]==pytest.approx(expect)
    assert gaussian_filter(x[:2],pi,a,mu,var)==pytest.approx(got[:2])
    assert gaussian_filter(x[1:],pi,a,mu,var,got[0])==pytest.approx(got[1:])


def test_annual_fit_never_uses_test_year_and_future_mutation_preserves_earlier_year():
    pytest.importorskip('hmmlearn')
    rng=np.random.default_rng(17)
    dates=pd.bdate_range('2020-01-01','2023-01-20').strftime('%Y-%m-%d')
    x=rng.normal(size=(len(dates),4));x[150:260]+=2;x[410:490]-=2
    f=pd.DataFrame(x,columns=FEATURES);f['date']=dates
    p,m=infer_states(f,2,'2022-01-01')
    assert p.training_end.lt(p.date).all()
    assert all(v['training_end']<v['year']+'-01-01' for v in m)
    assert all(v['means_raw'][0][1]<=v['means_raw'][1][1] for v in m)
    changed=f.copy();changed.loc[changed.date.ge('2023-01-01'),FEATURES]*=10
    q,n=infer_states(changed,2,'2022-01-01')
    pd.testing.assert_frame_equal(p.loc[p.date.lt('2023-01-01')],q.loc[q.date.lt('2023-01-01')])
    assert m==n  # changed only the prediction observations, not either training set
    cut,_=infer_states(f,2,'2022-06-01')
    pd.testing.assert_frame_equal(p.loc[p.date.ge('2022-06-01')].reset_index(drop=True),cut)


def test_market_features_are_past_only_and_missing_sessions_exclude_stock():
    cal=pd.bdate_range('2020-01-01',periods=150).strftime('%Y-%m-%d').tolist()
    raw=bars(cal,('600001.SH','600002.SH'))
    raw.loc[raw.stock_code.eq('600001.SH'),'close']=10.01
    meta=master(['600001.SH','600002.SH'])
    a=market_features(raw,cal,meta)
    mutated=raw.copy();mutated.loc[mutated.trade_date.gt(cal[110]),'close']=20
    b=market_features(mutated,cal,meta)
    pd.testing.assert_frame_equal(a.loc[a.date.le(cal[110])],b.loc[b.date.le(cal[110])])
    gap=raw.loc[~(raw.trade_date.eq(cal[100])&raw.stock_code.eq('600001.SH'))]
    c=market_features(gap,cal,meta).set_index('date')
    assert c.loc[cal[100]:cal[119],'universe_count'].eq(1).all()
    assert c.loc[cal[120],'universe_count']==2
    assert c.loc[cal[105],'breadth20']==0


def state_frame():
    return pd.DataFrame(dict(date=pd.bdate_range('2022-01-01',periods=10).strftime('%Y-%m-%d'),
        hmm2_p1=np.linspace(0,1,10),hmm3_p0=.2,hmm3_p1=.3,hmm3_p2=.5,vol20=.3,trend63=-.1,breadth20=.2))


@pytest.mark.parametrize('policy',['small','small_half','defensive','fixed_mix','vol_budget','observable','hmm2_mix','hmm2_cash','hmm3_mix','hmm2_lag5'])
def test_allocation_policies_have_explicit_bounded_weights(policy):
    f=state_frame();w=allocation_weights(f,policy)
    values=w[['small_fraction','defensive_fraction','cash_fraction']]
    np.testing.assert_allclose(values.sum(axis=1),np.ones(10))
    assert (values>=-1e-12).all().all() and (values<=1+1e-12).all().all()
    if policy=='hmm2_lag5':
        assert w.defensive_fraction.iloc[:5].eq(.5).all()
        assert w.defensive_fraction.iloc[5:].tolist()==f.hmm2_p1.iloc[:5].tolist()


def test_overlap_merges_once_without_rescaling_missing_names_or_hiding_missing_states():
    def component(codes):
        return pd.DataFrame([dict(stock_code=c,trade_date='2022-01-31',rank=i+1,selected=True,target_weight=.245,close=10.) for i,c in enumerate(codes)])
    s=component(['600001.SH','600002.SH']);d=component(['600002.SH','600003.SH'])
    w=pd.DataFrame([dict(date='2022-01-31',small_fraction=.25,defensive_fraction=.75)])
    merged=combine_decisions(s,d,w).set_index('stock_code')
    assert len(merged)==3
    assert merged.target_weight.to_dict()==pytest.approx({'600001.SH':.06125,'600002.SH':.245,'600003.SH':.18375})
    assert merged.target_weight.sum()==pytest.approx(.49)  # missing K slots stay cash
    w.loc[0,['small_fraction','defensive_fraction']]=[0,0]
    assert not combine_decisions(s,d,w).selected.any()
    with pytest.raises(ValueError,match='缺少'):
        combine_decisions(s,d,w.assign(date='2022-02-01'))


def test_breadth_threshold_changes_only_explicit_rule_and_is_in_definition():
    f=state_frame().assign(breadth20=.35)
    assert allocation_weights(f,'observable',.3).small_fraction.eq(1).all()
    assert allocation_weights(f,'observable',.5).small_fraction.eq(0).all()
    spec=ResearchSpec.from_dict(plan()['trials'][0]['spec'])
    spec=replace(spec,strategy=replace(spec.strategy,allocation_policy='observable',state_breadth_threshold=.3))
    assert '30%' in describe(spec)['formula']
    with pytest.raises(ValueError,match='非法'):
        allocation_weights(f.assign(hmm2_p1=1.5),'hmm2_mix')


def test_allocation_account_can_hold_two_components_and_exits_zero_weights():
    cal=pd.bdate_range('2022-01-28','2022-02-04').strftime('%Y-%m-%d').tolist();codes=['600001.SH','600002.SH']
    m=Market(bars(cal,codes),cal,metadata=master(codes))
    conf=ResearchSpec(Strategy(family='allocation',min_history=252,require_fundamentals=True,share_basis='known_bonus',top_k=1),
        Execution(initial_cash=10000,max_participation=.05),Experiment('2022-01-28','2022-02-04'))
    d=pd.DataFrame([dict(trade_date=cal[0],stock_code=c,selected=True,rank=i+1,close=10.,target_weight=.4) for i,c in enumerate(codes)]+
                   [dict(trade_date=cal[2],stock_code=c,selected=False,rank=i+1,close=10.,target_weight=0.) for i,c in enumerate(codes)])
    out=simulate(d,{cal[0]:cal[1],cal[2]:cal[3]},m,cal,conf,'configured')
    assert set(out['positions'].loc[lambda x:x.date.eq(cal[1]),'stock_code'])==set(codes)
    assert out['positions'].loc[lambda x:x.date.eq(cal[-1])].empty
    assert out['metrics']['max_accounting_error']<1e-8


def test_protocol_and_api_make_fixed_mapping_and_training_limits_reviewable(tmp_path):
    p=plan();assert len(p['trials'])==12
    assert len({t['key'] for t in p['trials']})==12
    for t in p['trials']:
        conf=ResearchSpec.from_dict(t['spec']);assert describe(conf)['family']=='allocation'
        assert conf.experiment.start_date=='2022-01-01'
    with pytest.raises(ValueError,match='2022'):
        replace(conf,experiment=Experiment('2020-01-01','2021-01-01'))
    with pytest.raises(ValueError,match='隐藏'):
        replace(conf,strategy=replace(conf.strategy,positive_profit=True))
    server=make_server(tmp_path,port=0);t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
    try:
        with urlopen(f'http://127.0.0.1:{server.server_port}/api/regimes/plan') as r:
            assert len(json.load(r)['trials'])==12
        with urlopen(f'http://127.0.0.1:{server.server_port}/api/regimes') as r:assert json.load(r)==[]
    finally:server.shutdown();server.server_close();t.join()
