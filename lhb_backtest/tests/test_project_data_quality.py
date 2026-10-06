import json
from pathlib import Path
import pandas as pd
import pytest
from src.data_sources.project_quality import inspect_project_data


@pytest.mark.parametrize('raw_code',['600000.SH','600000.SS'])
def test_gate_rejects_future_predictors_and_stale_downstream_files(tmp_path,monkeypatch,raw_code):
    import src.data_sources.simtradedata_metadata as metadata
    monkeypatch.setattr(metadata,'resolve_export_dir',lambda:None)
    summary=pd.DataFrame({'stock_code':['600000.SH','920001.BJ'],'trade_date':['2026-09-24']*2})
    detail=pd.concat([summary.assign(flag=flag,rank=1,broker_name='席位',buy_amount=1.,sell_amount=1.) for flag in ['买入','卖出']],ignore_index=True)
    events=summary.assign(summary_selected_reason='日涨幅',seat_report_reason='日涨幅',market_supported=[True,False],is_st=[0.,float('nan')])
    frames={'raw/daily_kline':summary.iloc[:1].assign(stock_code=raw_code),'raw/lhb_summary':summary,'raw/lhb_broker_detail':detail,
            'clean/daily_kline':summary.iloc[:1],'clean/lhb_summary':summary,
            'clean/lhb_broker_detail':detail.assign(direction=detail.flag.map({'买入':'buy','卖出':'sell'})),
            'factor/lhb_event_daily':events,'factor/lhb_event_factors':events,
            'factor/lhb_event_labeled':events.assign(**{f'future_return_{n}d':float('nan') for n in [1,2,3,5,10]})}
    for name,frame in frames.items():
        p=tmp_path/'data'/(name+'.parquet');p.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(p,index=False)
    result=inspect_project_data(tmp_path)
    assert result['status']=='pass',result
    assert result['checks']['unsupported_market_events']==1
    path=tmp_path/'data/factor/lhb_event_factors.parquet'
    events.assign(return_1d=.2).to_parquet(path,index=False)
    result=inspect_project_data(tmp_path)
    assert result['status']=='fail' and result['checks']['factors_forbidden_predictors']==['return_1d']


def test_status_outside_verified_history_is_unknown(tmp_path):
    from src.data_sources.simtradedata_metadata import load_status_lookup
    (tmp_path/'metadata').mkdir()
    pd.DataFrame({'date':['20181228','20190102'],'status_type':['ST','ST'],'symbols':[[],[]]}).to_parquet(tmp_path/'metadata/stock_status.parquet')
    (tmp_path/'.source_state.json').write_text(json.dumps({'status_policy':'dated-flags-v1','status_history_start':'2019-01-01','status_history_end':'2026-09-24'}))
    assert load_status_lookup('ST',tmp_path)=={'2019-01-02':set()}
