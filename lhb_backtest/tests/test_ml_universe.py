"""Membership/unknown outcomes and random-control semantics are research requirements."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from src.mlresearch.contracts import MLSpec,FEATURES
from src.mlresearch.dataset import build_panel
from src.mlresearch.study_data import fixed_scores,score_decisions
from src.mlresearch.universe import pool_masks,weekly_random,universe_plan
from src.mlresearch.universe_audit import check_account_audit

def test_pool_flags_do_not_consult_future_and_distinguish_boards():
    p=pd.DataFrame(dict(stock_code=['000001.SZ','600001.SH','300001.SZ','688001.SH','600002.SH'],is_st=[0,0,0,1,0],history_valid=[True,False,True,True,True],avg_amount_20=[3e7,1e6,3e7,3e7,3e7],cap_valid=[False,False,True,True,True],label_return=[np.nan]*5))
    masks=pool_masks(p,MLSpec())
    assert p[masks['qualified']].stock_code.tolist()==['600002.SH']
    assert p[masks['no_financial']].stock_code.tolist()==['000001.SZ','600002.SH']
    assert p[masks['mainboard_available']].stock_code.tolist()==['000001.SZ','600001.SH','600002.SH']
    assert masks['local_A_available'].all()
    p.label_return=999
    for name,value in pool_masks(p,MLSpec()).items():assert value.equals(masks[name])

def test_seed_is_not_stock_count_and_fresh_random_changes_weekly():
    codes=[f'{i:06d}.SZ' for i in range(1000)]
    p=pd.DataFrame(dict(stock_code=codes*2,trade_date=['2024-01-05']*1000+['2024-01-12']*1000,close=10.,log_market_cap=0.,return_5=0.))
    fixed=fixed_scores(p,11).fixed_random.to_numpy().reshape(2,1000)
    assert np.array_equal(fixed[0],fixed[1])
    fresh=weekly_random(p,11).to_numpy().reshape(2,1000)
    assert not np.array_equal(fresh[0],fresh[1])
    d=score_decisions(p,weekly_random(p,11),{'2024-01-05':'2024-01-08','2024-01-12':'2024-01-15'},100)
    assert d.groupby('trade_date').size().eq(100).all()
    first=set(d[d.trade_date.eq('2024-01-05')].stock_code);second=set(d[d.trade_date.eq('2024-01-12')].stock_code)
    assert len(first & second)<30

def test_new_stock_without_filings_is_kept_even_without_future():
    cal=pd.bdate_range('2024-01-01','2024-01-19').strftime('%Y-%m-%d').tolist()
    bars=pd.DataFrame(dict(stock_code='300001.SZ',trade_date=cal[:10],open=10.,high=11.,low=9.,close=10.,pre_close=10.,volume=100.,amount=1000.,is_st=1))
    meta=pd.DataFrame(dict(stock_code=['300001.SZ'],listed_date=['2024-01-01'],de_listed_date=['2900-01-01']))
    filings=pd.DataFrame(dict(stock_code=['600001.SH']*2,report_date=['2022-12-31','2023-09-30'],publication_date=['2023-04-20','2023-10-20'],total_shares=[1e8]*2,np_parent_company_owners=[1e6]*2,total_shareholder_equity=[1e8]*2,total_assets=[2e8]*2))
    spec=MLSpec(test_end='2024-01-12')
    p=build_panel(bars,cal,meta,filings,spec,inference=True,eligibility='available')
    assert len(p)>0 and p.is_st.eq(1).all() and not p.history_valid.any() and not p.cap_valid.any()
    assert not p.label_observed.iloc[-1]
    assert p.return_60.isna().all()

def test_universe_plan_excludes_extra_boards_from_mainboard_account():
    spec=MLSpec(study='universe',source_run_id='20261003T195515-13624598')
    plan=universe_plan(spec)
    assert sum(c['account'] for c in plan['cases'])==9
    assert all(not c['account'] for c in plan['cases'] if c['pool']=='local_A_available')
    assert plan['budget']['max_new_fits']==6
    with pytest.raises(ValueError):replace(spec,source_run_id=None)

def test_account_audit_contract_rejects_incomplete_or_bad_errors():
    good=dict(run_id='account-1',scenarios={s:dict(daily_errors=dict(cash=1e-9,receivable=0.,equity=1e-9),fills=20) for s in ['configured','zero_transaction_cost']})
    assert check_account_audit(good,'account-1',['configured','zero_transaction_cost'])==40
    with pytest.raises(ValueError):check_account_audit(good,'another-account',['configured','zero_transaction_cost'])
    with pytest.raises(ValueError):check_account_audit(good,'account-1',['configured'])
    good['scenarios']['configured']['daily_errors']['cash']=np.nan
    with pytest.raises(ValueError):check_account_audit(good,'account-1',['configured','zero_transaction_cost'])
