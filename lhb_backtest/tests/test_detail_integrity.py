import pandas as pd
import pytest
from src.ingestion.fetch_lhb_detail import _complete_pair_set, _fetch_both_sides, _merge_and_save
from src.ingestion.lhb_detail_archive import canonical_seats, fetch_range
from src.cleaning.clean_lhb import clean_lhb_broker_detail


@pytest.mark.parametrize('code,expected',[('920088.SZ','920088.BJ'),('900939.SZ','900939.SH'),('900939','900939.SH'),('113575.SZ','113575.SH'),('123099.SZ','123099.SZ')])
def test_legacy_exchange_suffix_is_corrected(code,expected):
    from src.cleaning.normalize_codes import normalize_stock_code
    assert normalize_stock_code(code)==expected


def seats():
    return pd.DataFrame([dict(trade_date='2026-09-24',stock_code='600000.SH',flag=flag,rank=rank,
                             broker_name='机构专用',broker_code=str(rank),buy_amount=10.,sell_amount=20.,net_amount=-10.,
                             report_type=kind,report_reason='日涨幅' if days==1 else '连续三个交易日',window_days=days)
                         for kind,days in [('daily',1),('three',3)] for flag in ['买入','卖出'] for rank in [1,2]])


def test_complete_pair_requires_both_sides():
    assert not _complete_pair_set(seats())  # mixed reports are not a valid event table
    frame=canonical_seats(seats())
    assert _complete_pair_set(frame)=={('2026-09-24','600000.SH')}
    assert not _complete_pair_set(frame[frame.flag.eq('买入')])
    class Client:
        def fetch_lhb_stock_detail_both(self,*args):return frame.iloc[:0]
    with pytest.raises(RuntimeError,match='incomplete'):
        _fetch_both_sides(Client(),'600000.SH','2026-09-24')


def test_canonical_never_mixes_daily_and_three_day_reports():
    result=canonical_seats(seats())
    assert len(result)==4 and set(result.report_type)=={'daily'}
    assert set(result.flag)=={'买入','卖出'}


def test_refresh_replaces_stale_rank_tail(tmp_path):
    old=seats().query("report_type=='daily'")
    new=old[old['rank'].eq(1)]
    result=_merge_and_save(old,[new],tmp_path/'detail.parquet')
    assert len(result)==2


def test_explicit_sell_survives_missing_buy(tmp_path):
    frame=seats().query("report_type=='daily'").copy();frame['buy_amount']=float('nan')
    result=clean_lhb_broker_detail(raw_df=frame,save_path=str(tmp_path/'clean.parquet'))
    assert result.sell_amount.eq(20).all()


def test_pagination_count_mismatch_fails(monkeypatch):
    import src.ingestion.lhb_detail_archive as module
    monkeypatch.setattr(module.time,'sleep',lambda n:None)
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'success':True,'result':{'pages':1,'count':2,'data':[{'x':1}]}}
    class Session:
        def get(self,*args,**kwargs):return Response()
    with pytest.raises(RuntimeError,match='count mismatch'):
        fetch_range('2026-09-01','2026-09-24','BUY',Session())


def test_event_predictors_do_not_include_source_future_returns(tmp_path,monkeypatch):
    from src.features import build_lhb_features as builder
    monkeypatch.setattr(builder,'_compute_historical_st',lambda df:pd.Series(0.,index=df.index))
    s=tmp_path/'summary.parquet';k=tmp_path/'kline.parquet'
    pd.DataFrame({'stock_code':['600000.SH'],'trade_date':['2026-09-24'],'return_1d':[.5]}).to_parquet(s)
    pd.DataFrame({'stock_code':['600000.SH'],'trade_date':['2026-09-24'],'close':[10.]}).to_parquet(k)
    result=builder.build_lhb_event_table(str(s),str(tmp_path/'absent.parquet'),str(k),str(tmp_path/'events.parquet'))
    assert 'return_1d' not in result


def test_summary_client_rejects_partial_pagination(monkeypatch):
    from src.data_sources import akshare_client as module
    monkeypatch.setattr(module.time,'sleep',lambda n:None)
    monkeypatch.setattr(module.ak,'stock_lhb_detail_em',lambda **kw:pd.DataFrame({'one_row':[1]}))
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'success':True,'result':{'count':2}}
    monkeypatch.setattr(module.requests,'get',lambda *a,**kw:Response())
    with pytest.raises(RuntimeError,match='pagination'):
        module.AKShareClient().fetch_lhb_summary('2026-09-24','2026-09-24')


def test_calendar_fallback_matches_2026_exchange_holidays(monkeypatch):
    from src.utils import calendar as cal
    monkeypatch.setattr(cal,'_trading_dates_cache',['2025-12-31'])
    days=cal.get_trading_dates('2026-01-01','2026-02-24')
    assert '2026-01-02' not in days and '2026-02-16' not in days
    assert '2026-02-24' in days


@pytest.mark.parametrize('reason,days',[
    ('连续三个交易日内涨幅偏离',3),('有价格涨跌幅限制的连续3个交易日涨幅',3),
    ('连续30个交易日严重异常',30),('最近3个有成交的交易日涨幅',3),
    ('连续十个交易日异常',10),('交易异常期间3次出现异常波动',0),
    ('日换手率达到30%',1),('',0)])
def test_reporting_window_is_not_assumed_single_day(reason,days):
    from src.data_sources.reporting_windows import reporting_window_days
    assert reporting_window_days(reason)==days


def test_summary_selects_whole_report_without_cross_window_cell_filling():
    from src.ingestion.fetch_lhb_summary import _aggregate_multi_reason_rows
    frame=pd.DataFrame([
        dict(stock_code='600000.SH',trade_date='2026-09-24',lhb_reason='连续三个交易日涨幅',lhb_buy_total=100.,lhb_net_buy=20.),
        dict(stock_code='600000.SH',trade_date='2026-09-24',lhb_reason='日涨幅',lhb_buy_total=10.,lhb_net_buy=float('nan'))])
    row=_aggregate_multi_reason_rows(frame).iloc[0]
    assert row.lhb_buy_total==10 and pd.isna(row.lhb_net_buy)
    assert row.summary_window_days==1 and row.summary_selected_reason=='日涨幅'
    assert '连续三个交易日' in row.lhb_reason


def test_summary_aggregation_is_idempotent_and_unique_rows_have_provenance():
    from src.ingestion.fetch_lhb_summary import _aggregate_multi_reason_rows
    frame=pd.DataFrame([
        dict(stock_code='600000.SH',trade_date='2026-09-24',lhb_reason='连续三个交易日涨幅',lhb_buy_total=100.),
        dict(stock_code='600000.SH',trade_date='2026-09-24',lhb_reason='日涨幅',lhb_buy_total=10.)])
    once=_aggregate_multi_reason_rows(frame)
    pd.testing.assert_frame_equal(once,_aggregate_multi_reason_rows(once))
    assert _aggregate_multi_reason_rows(frame.iloc[:1]).summary_window_days.iloc[0]==3


def test_margin_disclosure_preserves_one_sided_source_without_inventing_sells():
    frame=seats().query("report_type=='daily'").copy()
    frame=frame[frame.flag.eq('买入')]
    frame['report_reason']='单只标的证券当日融资买入数量达到50%'
    frame['sell_amount']=float('nan')
    result=canonical_seats(frame)
    assert len(result)==2 and result.disclosure_kind.eq('margin_buy').all()
    assert result.sell_amount.isna().all()
    frame['report_reason']='日涨幅'
    assert canonical_seats(frame).empty


def test_empty_source_response_allowed_only_in_count_checked_subqueries(monkeypatch):
    import src.ingestion.lhb_detail_archive as module
    monkeypatch.setattr(module.time,'sleep',lambda n:None)
    class Response:
        def raise_for_status(self):pass
        def json(self):return {'success':False,'code':9201,'message':'返回数据为空','result':None}
    class Session:
        def get(self,*args,**kwargs):return Response()
    with pytest.raises(RuntimeError,match='Unconfirmed'):
        fetch_range('2026-09-25','2026-09-25','BUY',Session())
    assert fetch_range('2026-09-25','2026-09-25','BUY',Session(),_allow_empty=True).empty


def test_refreshed_summary_replaces_old_values(tmp_path):
    from src.ingestion.fetch_lhb_summary import _merge_and_save
    old=pd.DataFrame([dict(stock_code='600000.SH',trade_date='2026-09-24',lhb_reason='日涨幅',lhb_buy_total=10.)])
    new=old.assign(lhb_buy_total=20.)
    result=_merge_and_save(old,[new],tmp_path/'summary.parquet')
    assert len(result)==1 and result.lhb_buy_total.iloc[0]==20.


def test_menu_update_requires_detail_success(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import tools.launcher as menu
    import src.ingestion.load_kline_simtradedata as kline
    import src.ingestion.fetch_lhb_summary as summary
    import src.ingestion.fetch_lhb_detail as detail
    def result(status):
        frame=pd.DataFrame();frame.attrs['update_report']={'status':status,'unresolved_dates':[]};return frame
    monkeypatch.setattr(kline,'load_kline_simtradedata',lambda *a,**k:result('pass'))
    monkeypatch.setattr(summary,'fetch_lhb_summary',lambda *a,**k:result('pass'))
    monkeypatch.setattr(detail,'fetch_lhb_detail',lambda *a,**k:result('fail'))
    updated=SimpleNamespace(export_dir=tmp_path,reused=True)
    assert not menu._finish_project_update('2026-09-24','2026-09-24',updated,tmp_path/'report.json')
    monkeypatch.setattr(detail,'fetch_lhb_detail',lambda *a,**k:result('pass'))
    assert menu._finish_project_update('2026-09-24','2026-09-24',updated,tmp_path/'report.json')


def test_cli_default_end_tracks_published_manifest(tmp_path):
    import json
    from main import resolve_end_date
    (tmp_path/'manifest.json').write_text(json.dumps({'version':'2026-09-24'}))
    config={'defaults':{'end_date':'latest'},'simtradedata':{'export_dir':str(tmp_path)}}
    assert resolve_end_date(config)=='2026-09-24'
    assert resolve_end_date(config,'2026-09-22')=='2026-09-22'


def test_archive_publication_reuses_unchanged_inputs_and_refuses_new_gaps(tmp_path,monkeypatch):
    import json
    from src.ingestion import lhb_detail_archive as archive
    summary=tmp_path/'summary.parquet';output=tmp_path/'detail.parquet'
    frame=seats().query("report_type=='daily'").copy()
    pd.DataFrame([{'stock_code':'600000.SH','trade_date':'2026-09-24','summary_selected_reason':'日涨幅'}]).to_parquet(summary)
    directory=tmp_path/'lhb_detail_source';directory.mkdir()
    for side,flag in [('BUY','买入'),('SELL','卖出')]:
        part=frame[frame.flag.eq(flag)];path=directory/f'2026-09-24_2026-09-24_{side}.parquet';part.to_parquet(path,index=False)
        path.with_suffix('.json').write_text(json.dumps({'schema':2,'rows':len(part),'bytes':path.stat().st_size,'start':'2026-09-24','end':'2026-09-24','side':side}))
    monkeypatch.setattr(archive,'acquire_archive',lambda *a:None)
    first=archive.refresh_detail_archive(summary,output);stamp=output.stat().st_mtime_ns
    assert first.attrs['update_report']['status']=='pass'
    monkeypatch.chdir(tmp_path)
    second=archive.refresh_detail_archive('summary.parquet','detail.parquet')
    assert second.attrs['update_report']['reused'] and output.stat().st_mtime_ns==stamp
    pd.DataFrame([{'stock_code':'000001.SZ','trade_date':'2026-09-24','summary_selected_reason':'日涨幅'}]).to_parquet(summary)
    with pytest.raises(RuntimeError,match='lack complete'):
        archive.refresh_detail_archive(summary,output)
    assert output.stat().st_mtime_ns==stamp


def test_identical_anonymous_seats_within_one_page_do_not_trigger_redownload(monkeypatch):
    import src.ingestion.lhb_detail_archive as module
    monkeypatch.setattr(module.time,'sleep',lambda n:None)
    row={'SECUCODE':'600000.SH','TRADE_DATE':'2026-09-24','OPERATEDEPT_NAME':'机构专用','OPERATEDEPT_CODE':'anon','BUY':10.,'SELL':0.,'NET':10.,'TOTAL_BUYRIO':.1,'TOTAL_SELLRIO':0.,'CHANGE_TYPE':'1','EXPLANATION':'日涨幅','TRADE_ID':'event','SECURITY_CODE':'600000'}
    class Response:
        def __init__(self,rows):self.rows=rows
        def raise_for_status(self):pass
        def json(self):return {'success':True,'result':{'pages':2,'count':3,'data':self.rows}}
    class Session:
        calls=0
        def get(self,*args,**kw):
            self.calls+=1
            return Response([row,row] if self.calls==1 else [dict(row,BUY=20.,NET=20.)])
    session=Session();result=fetch_range('2026-09-01','2026-09-24','BUY',session)
    assert len(result)==3 and session.calls==2

def test_investor_categories_are_preserved_but_not_seat_features(tmp_path):
    from src.features.build_lhb_features import _pivot_broker_detail
    frame=seats().query("report_type=='daily'").copy()
    frame['broker_name']=['自然人','中小投资者','自然人','中小投资者']
    canonical=canonical_seats(frame)
    assert len(canonical)==4
    assert canonical.disclosure_kind.eq('investor_category').all()
    path=tmp_path/'clean.parquet'
    clean_lhb_broker_detail(raw_df=canonical,save_path=str(path))
    wide=_pivot_broker_detail(str(path))
    assert len(wide)==1 and wide.disclosure_kind.iloc[0]=='investor_category'
    assert 'buy1_amount' not in wide and 'buy1_concentration' not in wide
