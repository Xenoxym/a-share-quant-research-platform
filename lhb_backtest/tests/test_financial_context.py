import sys
from pathlib import Path
import numpy as np
import pandas as pd
from src.mlresearch.financial_study import financial_features,market_features,financial_plan,FIN,MKT
from src.mlresearch.financial_audit import annual_reference,market_reference
from src.mlresearch.contracts import MLSpec

def filings():
    return pd.DataFrame([
        ['X','2021-12-31','2022-03-01',100.,10.,200.,300.],
        ['X','2022-12-31','2023-03-01',100.,-20.,250.,350.],
        ['X','2021-12-31','2023-05-01',100.,99.,999.,999.],
    ],columns=['stock_code','report_date','publication_date','total_shares','np_parent_company_owners','total_shareholder_equity','total_assets'])

def panel(days,codes=None):
    return pd.DataFrame(dict(stock_code=codes or ['X']*len(days),trade_date=days,close=10.,log_market_cap=np.log(1000.)))

def test_same_day_not_available_negative_profit_kept_and_late_older_ignored():
    p=panel(['2023-03-01','2023-03-02','2023-05-02']);out,_=financial_features(p,filings());ref,_,_=annual_reference(p,filings())
    np.testing.assert_allclose(out[FIN],ref[FIN],equal_nan=True)
    np.testing.assert_allclose(out.annual_ep,[.01,-.02,-.02]);np.testing.assert_allclose(out.annual_bp,[.2,.25,.25])

def test_stale_and_missing_keep_all_candidates():
    p=panel(['2025-01-01','2023-05-02'],['X','UNKNOWN']);out,_=financial_features(p,filings())
    assert len(out)==2 and out.annual_ep.isna().all() and out.annual_ep_missing.eq(1).all()

def test_financial_prefix_stability():
    p=panel(['2023-02-01','2023-03-02']);a,_=financial_features(p,filings());b,_=financial_features(p,filings()[filings().publication_date.le('2023-03-02')])
    np.testing.assert_allclose(a[FIN],b[FIN],equal_nan=True)

def test_market_formulas_prefix_and_st_unknown_exclusion():
    days=pd.date_range('2021-01-01',periods=80).strftime('%Y-%m-%d').tolist()
    frames=[]
    for code,st,sign in [('X',0,1),('Y',0,-1),('Z',1,100)]:
        frames.append(pd.DataFrame(dict(stock_code=code,trade_date=days,pre_close=10.,close=10*np.exp(sign*np.arange(80)*.0001),volume=1.,is_st=st)))
    b=pd.concat(frames,ignore_index=True);m=pd.DataFrame(dict(stock_code=['X','Y','Z'],listed_date='2000-01-01',de_listed_date='2900-01-01'))
    out=market_features(b,m,days);oracle=market_reference(b,m,days)
    np.testing.assert_allclose(out[MKT],oracle[MKT],atol=1e-12,equal_nan=True);assert out.members.eq(2).all()
    before=market_features(b[b.trade_date.le(days[65])],m,days[:66]);np.testing.assert_allclose(out.loc[:65,MKT],before[MKT],atol=1e-12,equal_nan=True)

def test_formal_type_plan_and_prohibited_parameter_budget():
    s=MLSpec(study='financial_context',source_run_id='20261003T195515-13624598');p=financial_plan(s)
    assert MLSpec.from_dict(s.to_dict()).study=='financial_context'
    assert len(p['cases'])==4 and p['budget']['max_new_fits']==4 and len(p['controls'])==5
    assert [len(c['features']) for c in p['cases']]==[25,23,31,31]
