from dataclasses import replace
import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest
import yaml

from src.technical.contracts import Execution, Experiment, ResearchSpec, Strategy, describe
from src.technical.data import prepare_snapshot
from src.technical.market import Market
from src.technical.portfolio import costs, simulate
from src.technical.signals import decision_dates, decisions, features
from src.technical.server import make_server


def bars(calendar, codes=("600001.SH",), price=10.):
    return pd.DataFrame([dict(stock_code=c, trade_date=d, open=price, high=price,
        low=price, close=price, pre_close=price, volume=1_000_000., amount=10_000_000.,
        high_limit=price*1.1, low_limit=price*.9, limit_price_valid=True, is_st=0,
        avg_volume_20=1_000_000.) for c in codes for d in calendar])


def master(codes):
    return pd.DataFrame([dict(stock_code=c, stock_name=c, listed_date="2000-01-01", de_listed_date="2900-01-01") for c in codes])


def spec(**strategy):
    return ResearchSpec(strategy=Strategy(lookback=5, skip=1, top_k=1, min_avg_amount=0, allocation=1., **strategy),
        execution=Execution(initial_cash=10000, max_participation=.05),
        experiment=Experiment(start_date="2020-01-01", end_date="2020-04-30"))


def test_momentum_endpoints_and_future_independence():
    cal = pd.bdate_range("2020-01-01", periods=40).strftime("%Y-%m-%d").tolist()
    b = bars(cal)
    r = np.arange(40)/10000
    b["close"] = 10*np.exp(r)
    s = replace(Strategy(), lookback=5, skip=2)
    f = features(b, cal, s)
    assert f.iloc[25].momentum == pytest.approx(np.expm1(r[21:24].sum()))
    mutated = b.copy()
    mutated.loc[mutated.trade_date.gt(cal[25]), "close"] *= 10
    pd.testing.assert_frame_equal(f.loc[f.trade_date.le(cal[25])], features(mutated, cal, s).loc[lambda d:d.trade_date.le(cal[25])])


def test_missing_market_session_breaks_window():
    cal = pd.bdate_range("2020-01-01", periods=50).strftime("%Y-%m-%d").tolist()
    b = bars(cal).drop(index=30)
    f = features(b, cal, replace(Strategy(), lookback=20, skip=0))
    assert not f.loc[f.trade_date.eq(cal[40]), "history_valid"].iloc[0]


def test_schedule_does_not_use_partial_last_month_as_month_end():
    cal = pd.bdate_range("2020-01-01", "2020-03-10").strftime("%Y-%m-%d").tolist()
    assert decision_dates(cal, cal[0], "2020-02-14", "monthly") == {"2020-01-31": "2020-02-03"}
    assert decision_dates(cal, "2020-02-03", "2020-02-06", "weekly") == {}


def test_selection_does_not_need_event_data_and_ties_are_defined():
    cal = pd.bdate_range("2020-01-01", "2020-02-04").strftime("%Y-%m-%d").tolist()
    b = features(bars(cal, ("600002.SH", "600001.SH")), cal, spec().strategy)
    d, _ = decisions(b, cal, master(["600002.SH", "600001.SH"]), spec())
    assert d.loc[d.selected, "stock_code"].tolist() == ["600001.SH"]
    filtered, _ = decisions(b, cal, master(["600002.SH", "600001.SH"]), spec(min_score=0.))
    assert not filtered.selected.any()  # strict > 0, not >= 0


def test_cost_components_zero_and_stamp_boundary():
    e = Execution()
    c = costs(20000, "sell", "2023-08-27", e, "configured")
    assert c == pytest.approx(dict(commission=6, other_fee=.4, stamp_tax=20, slippage=20, total=46.4))
    assert costs(20000, "sell", "2023-08-28", e, "configured")["stamp_tax"] == 10
    assert costs(20000, "buy", "2023-08-28", e, "zero_slippage")["total"] == pytest.approx(6.4)
    assert all(v == 0 for v in costs(20000, "sell", "2023-08-28", e, "zero_transaction_cost").values())


def execute_fixture(*, trapped=False, scenario="configured", actions=None, initial_cash=10000):
    cal = pd.bdate_range("2020-01-30", "2020-03-06").strftime("%Y-%m-%d").tolist()
    b = bars(cal, ("600001.SH", "600002.SH"))
    if trapped:
        mask = b.stock_code.eq("600001.SH") & b.trade_date.eq("2020-03-02")
        b.loc[mask, ["open", "low", "close"]] = 9.
    market = Market(b, cal, actions=actions, metadata=master(["600001.SH", "600002.SH"]))
    ds = pd.DataFrame([dict(trade_date="2020-01-31", stock_code="600001.SH", selected=True, rank=1, close=10., target_weight=1.),
                       dict(trade_date="2020-02-28", stock_code="600002.SH", selected=True, rank=1, close=10., target_weight=1.)])
    conf = replace(spec(), execution=replace(spec().execution, initial_cash=initial_cash))
    return simulate(ds, {"2020-01-31":"2020-02-03", "2020-02-28":"2020-03-02"}, market, cal, conf, scenario)


def test_target_rebalance_holds_beyond_five_days_and_prior_cash_only():
    out = execute_fixture()
    sells = out["fills"].loc[out["fills"].side.eq("sell")]
    assert sells.date.tolist() == ["2020-03-02"]
    buys_b = out["fills"].loc[out["fills"].stock_code.eq("600002.SH")]
    # Proceeds from March 2 cannot finance a March 2 purchase.
    assert buys_b.date.min() == "2020-03-03"
    assert out["equity"].cash.min() >= 0
    assert out["metrics"]["max_accounting_error"] < 1e-8


def test_unfilled_exit_is_retried_and_does_not_disappear():
    out = execute_fixture(trapped=True)
    blocked = out["orders"].loc[out["orders"].date.eq("2020-03-02") & out["orders"].side.eq("sell")].iloc[0]
    assert blocked.reason == "open_at_lower_limit" and blocked.filled_shares == 0
    assert out["positions"].loc[lambda d:d.date.eq("2020-03-02")].stock_code.tolist() == ["600001.SH"]
    assert out["fills"].loc[lambda d:d.side.eq("sell")].date.tolist() == ["2020-03-03"]


def test_zero_cost_reruns_quantities_and_is_not_addback():
    full, zero = execute_fixture(initial_cash=11000), execute_fixture(scenario="zero_transaction_cost", initial_cash=11000)
    assert zero["metrics"]["total_cost"] == 0
    assert zero["metrics"]["final_equity"] == pytest.approx(11000)
    assert not full["fills"].filled_shares.equals(zero["fills"].filled_shares)


def test_dividend_receivable_and_split_no_phantom_profit():
    cal = ["2020-01-31", "2020-02-03", "2020-02-04"]
    b = bars(cal)
    b.loc[b.trade_date.eq(cal[-1]), ["open", "high", "low", "close", "pre_close"]] = 4.5
    b.loc[b.trade_date.eq(cal[-1]), ["high_limit", "low_limit"]] = [4.95, 4.05]
    actions = pd.DataFrame([dict(stock_code="600001.SH", trade_date=cal[-1], allotted_ps=1., rationed_ps=0., rationed_px=0., dividend=1.)])
    ds = pd.DataFrame([dict(trade_date=cal[0], stock_code="600001.SH", selected=True, rank=1, close=10., target_weight=1.)])
    conf = replace(spec(), execution=replace(spec().execution, dividend_tax_reserve=0.))
    out = simulate(ds, {cal[0]:cal[1]}, Market(b, cal, actions=actions), cal, conf, "zero_transaction_cost")
    assert out["metrics"]["final_equity"] == pytest.approx(10000)
    assert out["equity"].iloc[-1].receivable == 900
    assert out["ledger"].loc[lambda d:d.kind.eq("corporate_action")].cash_flow.sum() == 0


def test_side_specific_limits():
    cal=["2020-01-31"]
    b=bars(cal)
    b.loc[0, ["open", "high", "close"]]=11.
    m=Market(b,cal)
    assert m.fill_block("600001.SH",cal[0],"buy")[0] == "open_at_upper_limit"
    assert m.fill_block("600001.SH",cal[0],"sell")[0] == ""


def test_halted_corporate_action_does_not_revalue_bonus_shares_at_old_price():
    cal=["2020-01-31","2020-02-03","2020-02-04","2020-02-05"]
    b=bars(cal)  # Halt day still carries vendor's old price of 10.
    b.loc[b.trade_date.eq(cal[-1]),['open','high','low','close','pre_close']]=5.
    b.loc[b.trade_date.eq(cal[-1]),['high_limit','low_limit']]=[5.5,4.5]
    action=pd.DataFrame([dict(stock_code='600001.SH',trade_date=cal[2],allotted_ps=1.,rationed_ps=0.,rationed_px=0.,dividend=0.)])
    status=pd.DataFrame([dict(trade_date=cal[2],status_type='HALT',symbols=['600001.SS'])])
    ds=pd.DataFrame([dict(trade_date=cal[0],stock_code='600001.SH',selected=True,rank=1,close=10.,target_weight=1.)])
    out=simulate(ds,{cal[0]:cal[1]},Market(b,cal,actions=action,status=status),cal,spec(),'zero_transaction_cost')
    np.testing.assert_allclose(out['equity'].equity,10000,atol=1e-8)
    assert out['positions'].loc[lambda d:d.date.eq(cal[2]),'mark'].iloc[0]==5.
    assert 'unexplained_price_reference_change' not in out['metrics']['warnings']


def test_delisted_quote_cannot_create_normal_fill():
    cal=['2020-01-31']
    metadata=master(['600001.SH']);metadata['de_listed_date']=cal[0]
    market=Market(bars(cal),cal,metadata=metadata)
    assert market.fill_block('600001.SH',cal[0],'buy')[0]=='delisted_unsettled'
    assert market.fill_block('600001.SH',cal[0],'sell')[0]=='delisted_unsettled'


def test_snapshot_from_directory_with_no_lhb_files(tmp_path):
    clean, export=tmp_path/'clean',tmp_path/'export'
    clean.mkdir();(export/'metadata').mkdir(parents=True);(export/'exrights').mkdir()
    (tmp_path/'config').mkdir()
    (tmp_path/'config/config.yaml').write_text(yaml.safe_dump({'paths':{'clean_dir':'clean'},'simtradedata':{'export_dir':'export'}}),encoding='utf-8')
    cal=pd.bdate_range('2020-01-01','2020-03-06').strftime('%Y-%m-%d').tolist()
    bars(cal).drop(columns=['avg_volume_20']).to_parquet(clean/'daily_kline.parquet')
    pd.DataFrame({'date':cal}).to_parquet(export/'metadata/trade_days.parquet')
    pd.DataFrame([dict(symbol='600001.SS',stock_name='test',listed_date='2000-01-01',de_listed_date='2900-01-01',security_type='1')]).to_parquet(export/'metadata/stock_metadata.parquet')
    pd.DataFrame([dict(date=d,status_type=k,symbols=[]) for d in cal for k in ['ST','HALT']]).to_parquet(export/'metadata/stock_status.parquet')
    pd.DataFrame({'date':cal,'close':10.}).to_parquet(export/'metadata/benchmark.parquet')
    (export/'.source_state.json').write_text(json.dumps({'status_policy':'dated-flags-v1','status_history_start':cal[0],'status_history_end':cal[-1]}))
    conf=replace(spec(),experiment=Experiment('2020-01-01','latest'))
    folder,manifest=prepare_snapshot(tmp_path,tmp_path/'output',conf,lambda m:None)
    assert manifest['quote_codes']==1
    b=features(pd.read_parquet(folder/'bars.parquet'),cal,conf.strategy)
    d,_=decisions(b,cal,pd.read_parquet(folder/'metadata.parquet'),replace(conf,experiment=Experiment('2020-01-01',cal[-1])))
    assert d.selected.sum()==2


def test_local_server_definition_validation_and_download_boundary(tmp_path):
    server=make_server(tmp_path,port=0)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    url=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(url) as response:
            page=response.read().decode()
        import re
        boot=json.loads(re.search(r'window.TECHNICAL=(.*?);</script>',page).group(1))
        data=json.dumps(ResearchSpec().to_dict()).encode()
        req=Request(url+'/api/definition',data=data,headers={'X-Research-Token':boot['token'],'Content-Type':'application/json'})
        with urlopen(req) as response:
            definition=json.load(response)
        assert '252' in definition['definition']['momentum']
        batch_spec = replace(ResearchSpec(), experiment=Experiment(end_date="2026-09-24"))
        batch_body = json.dumps({"spec": batch_spec.to_dict(), "split_date": "2024-01-01"}).encode()
        with urlopen(Request(url+'/api/batch-plan', data=batch_body, headers={'X-Research-Token':boot['token']})) as response:
            assert json.load(response)['trial_count'] == 24
        with pytest.raises(HTTPError) as malformed:
            urlopen(Request(url+'/api/batch-plan', data=b'[]', headers={'X-Research-Token':boot['token']}))
        assert malformed.value.code == 400
        with pytest.raises(HTTPError) as e:
            urlopen(Request(url+'/api/run',data=data))
        assert e.value.code==403
        with pytest.raises(HTTPError):
            urlopen(url+'/api/runs/../../config/config.yaml')
    finally:
        server.shutdown();server.server_close();worker.join(timeout=2)


def test_contract_rejects_unbounded_or_ambiguous_values():
    with pytest.raises(ValueError):
        ResearchSpec.from_dict({'strategy':{'lookback':5,'skip':5}})
    with pytest.raises(ValueError):
        ResearchSpec.from_dict({'execution':{'slippage_bps':float('nan')}})
    with pytest.raises(ValueError):
        ResearchSpec.from_dict({'strategy':{'hold_five_days':True}})
    assert '元' in describe(ResearchSpec())['inputs']
