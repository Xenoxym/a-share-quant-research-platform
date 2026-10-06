"""Leakage, calendar endpoints, unknown outcomes and real-account adapter tests."""
from dataclasses import replace
import numpy as np
import pandas as pd

from src.mlresearch.contracts import MLSpec, FEATURES
from src.mlresearch.study import Fits, study_plan
from src.mlresearch.study_data import open_labels, score_decisions, balanced_score, size_buckets
from src.mlresearch.study_evaluation import select_validation, account_statistics
from src.technical.contracts import ResearchSpec, Strategy, Experiment
from src.technical.market import Market
from src.technical.portfolio import simulate
from src.technical.signals import decision_dates


def test_open_label_calendar_gap_action_and_endpoint():
    cal = pd.bdate_range("2024-01-01", "2024-01-26").strftime("%Y-%m-%d").tolist()
    spec = MLSpec(test_end="2024-01-26")
    bars = pd.DataFrame(dict(stock_code="A", trade_date=cal, open=10., close=10., pre_close=10.))
    bars.loc[bars.trade_date.eq("2024-01-08"), "open"] = 11.
    bars.loc[bars.trade_date.ge("2024-01-10"), ["open","close","pre_close"]] = 5.
    bars.loc[bars.trade_date.eq("2024-01-15"), "open"] = 6.
    panel = pd.DataFrame(dict(stock_code=["A", "A"], trade_date=["2024-01-05", "2024-01-19"], split="test"))
    labels = open_labels(panel,bars,cal,spec)
    assert labels.label_entry.iloc[0] == "2024-01-08"
    assert labels.label_end.iloc[0] == "2024-01-15"
    np.testing.assert_allclose(labels.label_return.iloc[0],12/11-1,atol=1e-12)
    assert not labels.label_observed.iloc[1]  # Terminal inference is retained.
    gap = bars[bars.trade_date.ne("2024-01-11")]
    assert not open_labels(panel,gap,cal,spec).label_observed.iloc[0]


def test_unknown_future_is_selected_and_adapter_enters_real_cash_account():
    panel = pd.DataFrame(dict(stock_code=["A","B"], trade_date="2024-01-05", close=10., label_return=[np.nan,.2]))
    decisions = score_decisions(panel,[2.,1.],{"2024-01-05":"2024-01-08"},1)
    assert decisions.stock_code.tolist()==["A"]
    cal=["2024-01-05","2024-01-08","2024-01-09"]
    bars = pd.DataFrame([dict(stock_code=c,trade_date=d,open=10.,high=10.,low=10.,close=10.,pre_close=10.,volume=1e8,amount=1e9,
        high_limit=11.,low_limit=9.,limit_price_valid=True,is_st=0,avg_volume_20=1e8) for c in ["A","B"] for d in cal])
    meta=pd.DataFrame(dict(stock_code=["A","B"],listed_date="2010-01-01",de_listed_date="9999-12-31"))
    market=Market(bars,cal,pd.DataFrame(),pd.DataFrame(columns=["status_type","stock_code","begin_date","end_date"]),meta,[])
    spec=ResearchSpec(strategy=Strategy(top_k=1,rebalance="weekly"),experiment=Experiment(cal[0],cal[-1]))
    result=simulate(decisions,{cal[0]:cal[1]},market,cal,spec,"zero_transaction_cost")
    assert not result["fills"].empty
    assert result["fills"].date.min()==cal[1]
    assert result["positions"].stock_code.unique().tolist()==["A"]
    assert result["metrics"]["max_accounting_error"] < .001


def test_fit_ignores_unmatured_and_future_labels(tmp_path):
    rng=np.random.default_rng(11); n=260
    panel=pd.DataFrame({name:rng.normal(size=n) for name in FEATURES})
    panel["trade_date"]=["2022-01-07"]*200+["2022-12-30"]*30+["2024-01-05"]*30
    labels=pd.DataFrame(dict(label_relative=rng.normal(size=n),label_observed=True,label_end=["2022-01-14"]*200+["2023-01-06"]*30+["2024-01-12"]*30))
    spec=MLSpec(tree_min_samples=2)
    a,b=tmp_path/"a",tmp_path/"b"; a.mkdir(); b.mkdir()
    first=Fits(a,panel,labels,spec,lambda m:None)
    key,model=first.fit("ridge",list(FEATURES),spec.train_end,{"alpha":10.})
    labels2=labels.copy(); labels2.loc[200:,"label_relative"]=1e12
    changed=panel.copy(); changed.loc[200:,list(FEATURES)]=1e15
    second=Fits(b,changed,labels2,spec,lambda m:None)
    _,other=second.fit("ridge",list(FEATURES),spec.train_end,{"alpha":10.})
    np.testing.assert_allclose(model.predict(panel[list(FEATURES)]),other.predict(panel[list(FEATURES)]),atol=1e-12)
    assert first.entries[0]["rows"]==200
    assert first.fit("ridge",list(FEATURES),spec.train_end,{"alpha":10.})[0]==key
    assert first.new_count==1  # Shared default tuning/refit configurations are not repeated.


def test_validation_selection_cannot_see_test():
    panel=pd.DataFrame(dict(trade_date=["2023-01-06"]*4+["2024-01-05"]*4,split=["valid"]*4+["test"]*4))
    labels=pd.DataFrame(dict(label_end=["2023-01-13"]*4+["2024-01-12"]*4,label_return=np.arange(8.)))
    score=np.arange(8.); spec=MLSpec()
    first=select_validation(panel,labels,score,spec)
    labels.loc[4:,"label_return"]=[999.,-500.,-600.,1.]
    score[4:]=[-1.,3.,2.,1.]
    assert select_validation(panel,labels,score,spec)==first==1.


def test_size_balanced_selection_and_finite_plan():
    n=1000
    panel=pd.DataFrame(dict(stock_code=[f"{i:06d}" for i in range(n)],trade_date="2024-01-05",close=10.,log_market_cap=np.arange(n,dtype=float)))
    scores=balanced_score(panel,np.arange(n,dtype=float))
    selected=score_decisions(panel,scores,{"2024-01-05":"2024-01-08"},100)
    assert size_buckets(panel).loc[selected.index].value_counts().eq(10).all()
    plan=study_plan(MLSpec())
    assert len(plan["cases"])==19
    assert len({c["id"] for c in plan["cases"]})==19
    assert plan["budget"]["max_new_fits"]==30
    assert plan["budget"]["cost_scenarios"]==["zero_transaction_cost","configured"]


def test_year_returns_include_initial_and_prior_year_equity():
    eq=pd.DataFrame(dict(date=["2024-12-30","2024-12-31","2025-01-02"],equity=[110.,120.,108.]))
    s=account_statistics(eq,100.,11,eq)
    np.testing.assert_allclose([a["return_value"] for a in s["annual"]],[.2,-.1])
    assert s["mean_weekly_return_delta_vs_size"]==0.
