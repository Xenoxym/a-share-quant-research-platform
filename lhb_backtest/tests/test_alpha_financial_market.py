"""Financial vintage, fiscal period and market gap counterexamples with hand values."""
import numpy as np
import pandas as pd
import pytest

from src.alpharesearch.contracts import MissingReason as M
from src.alpharesearch.features.financial import financial_features
from src.alpharesearch.features.market import market_context_features
from src.technical.foundation_data import attach_reports


def decisions(dates=('2024-04-26','2024-04-29')):
    return pd.DataFrame([dict(sample_id=str(i)+c,trade_date=d,stock_code=c,decision_at=d+'T15:10:00+08:00',
        close=10.,quote_observed_end=d+'T15:00:00+08:00',quote_known_at=d+'T15:01:00+08:00')
        for i,d in enumerate(dates) for c in ('000001.SZ','000002.SZ')])


def filings():
    return pd.DataFrame([dict(stock_code='000001.SZ',report_date=period,publication_date=pub,
        total_shares=100.,np_parent_company_owners=profit,total_shareholder_equity=equity,total_assets=1000.)
        for period,pub,profit,equity in [('2023-12-31','2024-04-26',100.,400.),('2024-03-31','2024-04-28',25.,450.)]])


def financial(p=None,f=None,**kw):
    options=dict(source_id='filings',version_id='v1',quote_source_id='quote',quote_version_id='q1',vintage='historical_versions')
    options.update(kw)
    return financial_features(decisions() if p is None else p,filings() if f is None else f,**options)


def test_date_only_publication_waits_until_complete_local_date_and_keeps_all_stocks():
    result=financial();v=result.block.values
    assert len(v)==4 and len(result.block.units)==14
    same=v.trade_date.eq('2024-04-26');assert v.loc[same,'annual_ep'].isna().all()
    row=v.trade_date.eq('2024-04-29') & v.stock_code.eq('000001.SZ')
    assert v.loc[row,'reported_market_cap_proxy'].iloc[0]==1000
    assert v.loc[row,'annual_ep'].iloc[0]==.1 and v.loc[row,'annual_bp'].iloc[0]==.4
    assert v.loc[row,'annual_profitability_proxy'].iloc[0]==.25
    assert result.lineage.loc[row,'q_report_date'].iloc[0]=='2024-03-31'
    assert result.lineage.loc[row,'a_report_date'].iloc[0]=='2023-12-31'
    assert v.loc[v.stock_code.eq('000002.SZ'),'annual_ep'].isna().all()


def test_explicit_publication_clock_can_use_same_day_only_after_release():
    f=filings();f['publication_at']=['2024-04-26T14:30:00+08:00','2024-04-28T14:30:00+08:00']
    result=financial(f=f);row=result.block.values.trade_date.eq('2024-04-26') & result.block.values.stock_code.eq('000001.SZ')
    assert result.block.values.loc[row,'annual_ep'].iloc[0]==.1
    f.loc[0,'publication_at']='2024-04-26T16:00:00+08:00'
    assert financial(f=f).block.values.loc[row,'annual_ep'].isna().all()


def test_old_period_published_late_cannot_replace_newer_known_period():
    f=filings();f=pd.concat([f,pd.DataFrame([dict(stock_code='000001.SZ',report_date='2023-09-30',publication_date='2024-04-28',total_shares=9999.,np_parent_company_owners=9999.,total_shareholder_equity=9999.,total_assets=9999.)])])
    result=financial(f=f);row=result.lineage.trade_date.eq('2024-04-29') & result.lineage.stock_code.eq('000001.SZ')
    assert result.lineage.loc[row,'q_report_date'].iloc[0]=='2024-03-31'
    assert result.block.values.loc[row,'reported_shares_proxy'].iloc[0]==100.


def test_new_revision_does_not_change_before_announcement_and_is_used_after():
    f=filings();revision=f.iloc[:1].copy();revision['publication_date']='2024-05-10';revision['np_parent_company_owners']=200.
    p=decisions(('2024-04-29','2024-05-13'));one=financial(p,f);two=financial(p,pd.concat([f,revision]))
    first=one.block.values.trade_date.eq('2024-04-29')
    pd.testing.assert_frame_equal(one.block.values.loc[first],two.block.values.loc[first])
    row=two.block.values.trade_date.eq('2024-05-13') & two.block.values.stock_code.eq('000001.SZ')
    assert two.block.values.loc[row,'annual_ep'].iloc[0]==.2
    # A later annual restatement does not roll back the newer quarter share period.
    assert two.lineage.loc[row,'q_report_date'].iloc[0]=='2024-03-31'


def test_duplicate_same_time_revision_is_not_arbitrarily_selected():
    f=filings()
    with pytest.raises(ValueError,match='Ambiguous'):financial(f=pd.concat([f,f.iloc[:1]]))


def test_stale_values_remain_missing_while_age_is_visible():
    result=financial(p=decisions(('2026-01-05',)));v=result.block.values;m=result.block.missing;row=v.stock_code.eq('000001.SZ')
    assert v.loc[row,'annual_parent_profit'].isna().all() and m.loc[row,'annual_parent_profit'].eq(M.STALE.value).all()
    assert v.loc[row,'annual_period_age_days'].iloc[0]>550 and v.loc[row,'annual_stale'].iloc[0]==1
    assert m.loc[~row,'annual_parent_profit'].eq(M.UNCOVERED.value).all()


@pytest.mark.parametrize('equity',[0.,-400.])
def test_negative_profit_or_equity_not_blanket_dropped_but_profitability_denominator_scoped(equity):
    f=filings();f.loc[0,'np_parent_company_owners']=-100.;f.loc[0,'total_shareholder_equity']=equity
    result=financial(f=f);row=result.block.values.trade_date.eq('2024-04-29') & result.block.values.stock_code.eq('000001.SZ')
    assert result.block.values.loc[row,'annual_ep'].iloc[0]==-.1
    assert result.block.values.loc[row,'annual_bp'].iloc[0]==equity/1000.
    assert result.block.missing.loc[row,'annual_profitability_proxy'].iloc[0]==M.UNDEFINED.value


@pytest.mark.parametrize('field,value',[('quote_known_at','2024-04-26T16:00:00+08:00'),('quote_observed_end','2024-04-25T15:00:00+08:00'),('decision_at','2024-04-26T15:10:00'),('stock_code','000001'),('trade_date','20240426')])
def test_invalid_decision_or_price_clock_rejects(field,value):
    p=decisions();p.loc[0,field]=value
    with pytest.raises(ValueError):financial(p=p)


def test_missing_quote_does_not_remove_filing_but_prevents_cap_and_ep():
    p=decisions();p['close']=np.nan;p['quote_known_at']=None;p['quote_observed_end']=None
    result=financial(p=p);row=result.block.values.trade_date.eq('2024-04-29') & result.block.values.stock_code.eq('000001.SZ')
    assert result.block.values.loc[row,'annual_parent_profit'].iloc[0]==100.
    assert result.block.values.loc[row,'annual_ep'].isna().all()


def test_weak_vintage_default_rejected_and_retained_when_allowed():
    with pytest.raises(ValueError,match='[Ww]eak'):financial(vintage='latest_snapshot_only')
    assert financial(vintage='latest_snapshot_only',allow_weak_vintage=True).block.metadata['vintage']=='latest_snapshot_only'


def test_original_financial_proxy_agrees_when_timing_and_staleness_agree():
    p=decisions();f=filings();new=financial(p,f);old=attach_reports(p[['trade_date','stock_code','close']],f)
    np.testing.assert_allclose(new.block.values.reported_market_cap_proxy,old.estimated_market_cap,equal_nan=True)
    np.testing.assert_allclose(new.block.values.annual_ep,old.earnings_yield,equal_nan=True)
    np.testing.assert_allclose(new.block.values.annual_profitability_proxy,old.profitability,equal_nan=True)


def test_empty_source_and_future_only_source_keep_rows_missing():
    p=decisions();f=filings();empty=financial(p,f.iloc[:0]);future=financial(p,f.assign(publication_date='2025-01-01'))
    assert len(empty.block.values)==len(p) and empty.block.values.annual_ep.isna().all()
    pd.testing.assert_frame_equal(empty.block.values,future.block.values)


def market_bars():
    return pd.DataFrame([dict(trade_date=d,stock_code=c,close=close,pre_close=10.,volume=100.)
        for d in ('2024-01-03','2024-01-04','2024-01-05') for c,close in [('000001.SZ',11.),('000002.SZ',9.)]])


def market(b=None,cal=None,**kw):
    options=dict(source_id='quote',version_id='q1',universe_id='declared_quote_proxy',vintage='market_observation',windows=(2,3))
    options.update(kw)
    return market_context_features(market_bars() if b is None else b,['2024-01-03','2024-01-04','2024-01-05'] if cal is None else cal,**options)


def test_market_geometric_log_proxy_and_breadth_hand_values():
    f=market();v=f.values
    mean=(np.log(1.1)+np.log(.9))/2
    np.testing.assert_allclose(v.market_daily_mean_log_return,mean)
    assert v.market_daily_breadth.eq(.5).all() and v.market_declared_members.eq(2).all()
    assert v.market_return_3.iloc[-1]==pytest.approx(mean*3)
    assert v.market_volatility_3.iloc[-1]==pytest.approx(0.)
    assert f.missing.market_return_3.iloc[0]==M.HISTORY.value
    assert v.known_at.iloc[0]==pd.Timestamp('2024-01-03T07:01:00Z')


def test_missing_market_session_is_not_filled_with_zero_or_compressed():
    bars=market_bars();bars=bars.loc[bars.trade_date.ne('2024-01-04')];f=market(bars)
    assert f.values.market_daily_mean_log_return.iloc[1]!=f.values.market_daily_mean_log_return.iloc[1]
    assert f.values.market_return_3.isna().all() and f.missing.market_return_3.iloc[-1]==M.INCOMPLETE.value


def test_membership_denominator_exposes_missing_quotes_without_claiming_full_market():
    bars=market_bars();members=bars[['trade_date','stock_code']];bars=bars.loc[bars.stock_code.eq('000001.SZ')]
    f=market(bars,membership=members);assert f.values.market_valid_fraction.eq(.5).all()
    assert f.values.market_daily_breadth.eq(1).all()
    assert f.values.market_daily_cross_section_dispersion.isna().all()


def test_zero_volume_and_bad_prices_excluded_without_inf_or_zero_return():
    bars=market_bars();bars.loc[bars.stock_code.eq('000001.SZ'),'volume']=0
    bars.loc[bars.trade_date.eq('2024-01-04'),'pre_close']=0
    f=market(bars);assert f.values.market_daily_breadth.iloc[0]==0
    assert f.values.market_daily_mean_log_return.isna().iloc[1]
    assert not np.isinf(f.values[list(f.units)].to_numpy()).any()


def test_market_future_append_preserves_prior_values_and_missingness():
    bars=market_bars();f=market(bars);extra=bars.iloc[:2].copy();extra['trade_date']='2024-01-08'
    later=market(pd.concat([bars,extra]),['2024-01-03','2024-01-04','2024-01-05','2024-01-08'])
    pd.testing.assert_frame_equal(f.values,later.values.iloc[:3]);pd.testing.assert_frame_equal(f.missing,later.missing.iloc[:3])


@pytest.mark.parametrize('case',['duplicate_quote','calendar_missing','calendar_duplicate','bad_window','bad_code'])
def test_bad_market_contracts_reject(case):
    b=market_bars();cal=['2024-01-03','2024-01-04','2024-01-05'];kw={}
    if case=='duplicate_quote':b=pd.concat([b,b.iloc[:1]])
    if case=='calendar_missing':cal=cal[:2]
    if case=='calendar_duplicate':cal.append(cal[-1])
    if case=='bad_window':kw['windows']=(1,)
    if case=='bad_code':b.loc[0,'stock_code']='000001'
    with pytest.raises(ValueError):market(b,cal,**kw)
