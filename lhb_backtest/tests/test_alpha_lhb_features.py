"""Concentration, disclosure windows, missingness and decision-time counterexamples."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.alpharesearch.contracts import MissingReason as M
from src.alpharesearch.features.lhb import report_features, latest_report_features
from src.alpharesearch.panel import attach_events
from src.alpharesearch.snapshots import build_event_tables

DAY = '日涨幅偏离值达到7%的证券'
THREE = '连续三个交易日涨幅偏离值累计达到20%的证券'
CAL = ['2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08', '2024-01-09']


def tables(reason=DAY, date='2024-01-05', rid='R1', names=('机构专用', '沪股通专用', '营业部'), amounts=(60., 30., 10.)):
    summary = pd.DataFrame([dict(trade_date=date, stock_code='000001.SZ', lhb_reason=reason,
        lhb_buy_total=100., lhb_sell_total=40., lhb_net_buy=9999., lhb_turnover=140., daily_amount=1000., return_5d=.99)])
    seats = pd.DataFrame([dict(trade_date=date, stock_code='000001.SZ', report_id=rid,
        report_type='ordinary', report_reason=reason, flag='买入', broker_name=name,
        buy_amount=amount, sell_amount=0., source_version=2)
        for name, amount in zip(names, amounts)] + [dict(trade_date=date, stock_code='000001.SZ',
        report_id=rid, report_type='ordinary', report_reason=reason, flag='卖出', broker_name='营业部',
        buy_amount=0., sell_amount=40., source_version=2)])
    return summary, seats


def build(**kw):
    s, a = tables(**kw);return build_event_tables(s, a)[:2]


def features(r=None, a=None):
    if r is None:r, a = build()
    return report_features(r, a, source_id='test_source', version_id='test_snapshot')


def panel(r, last='2024-01-08', weak=True):
    p = pd.DataFrame([dict(trade_date=d, stock_code=c, decision_at=d+'T15:10:00+08:00')
        for d in CAL if d <= last for c in ('000001.SZ', '000002.SZ')])
    return attach_events(p, r, CAL, universe_id='all_supplied_keys', window_sessions=2, allow_weak_vintage=weak)


def project(p=None, block=None, **kw):
    r, a = build();p = panel(r) if p is None else p
    block = features(r, a) if block is None else block
    return latest_report_features(p, block, allow_weak_vintage=True, **kw)


def test_concentration_formulas_and_separate_side_lists():
    f = features();v = f.values.iloc[0]
    assert len(f.units) == 34
    assert v.buy_disclosed_amount == 100 and v.sell_disclosed_amount == 40
    assert v.buy_top1_share == pytest.approx(.6) and v.buy_top2_share == pytest.approx(.9)
    assert v.buy_hhi == pytest.approx(.46)
    assert v.buy_entropy_log5 == pytest.approx(-sum(x*np.log(x) for x in (.6,.3,.1))/np.log(5))
    assert v.sell_top1_share == 1 and v.sell_top2_share == 1 and v.sell_entropy_log5 == 0
    assert v.buy_institution_labeled_rows == 1 and v.buy_institution_amount_share == .6
    assert v.buy_northbound_labeled_rows == 1 and v.buy_northbound_amount_share == .3
    assert v.net_from_reported_totals == 60 and v.reported_net_imbalance == pytest.approx(60/140)
    assert v.reported_buy_sell_ratio == 2.5 and v.single_day_net_to_reported_daily_amount == .06
    assert not any('return_' in c or 'net_buy' in c for c in f.values)


def test_anonymous_institution_occurrences_are_not_distinct_owner_claims_or_deduplicated():
    r, a = build(names=('机构专用', '机构专用'), amounts=(50.,50.));v = features(r,a).values.iloc[0]
    assert v.buy_disclosed_rows == 2 and v.buy_institution_labeled_rows == 2
    assert v.buy_institution_amount_share == 1 and v.buy_hhi == .5


def test_bank_name_is_not_inferred_as_institution_or_northbound():
    r, a = build(names=('摩根大通某分公司',), amounts=(100.,));v = features(r,a).values.iloc[0]
    assert v.buy_institution_labeled_rows == 0 and v.buy_institution_amount_share == 0
    assert v.buy_northbound_labeled_rows == 0


@pytest.mark.parametrize('reason', [THREE, '异常期间价格涨幅达到100%的证券'])
def test_multi_or_unknown_window_never_divides_by_daily_amount(reason):
    r,a = build(reason=reason);f = features(r,a)
    assert f.values.buy_hhi.iloc[0] == pytest.approx(.46)
    assert f.values.single_day_net_to_reported_daily_amount.isna().all()
    assert f.missing.single_day_net_to_reported_daily_amount.eq(M.UNDEFINED.value).all()


def test_overlapping_investor_categories_are_not_broker_concentration():
    r,a = build(names=('自然人','中小投资者'),amounts=(100.,100.));f=features(r,a)
    assert f.values.is_investor_category.iloc[0] == 1
    for n in f.units:
        if n.startswith(('buy_','sell_','reported_','single_day_','net_from_')):
            assert f.values[n].isna().all() and f.missing[n].eq(M.UNDEFINED.value).all()


@pytest.mark.parametrize('reason,side', [('融资买入量达到当日该证券总交易量50%以上的证券','buy'),('融券卖出量达到当日该证券总交易量50%以上的证券','sell')])
def test_legitimate_one_sided_disclosure_keeps_own_side_only(reason,side):
    s,a=tables(reason=reason);a=a.loc[a.flag.eq('买入' if side=='buy' else '卖出')]
    r,a,_=build_event_tables(s,a);f=features(r,a);opposite='sell' if side=='buy' else 'buy'
    assert f.values[side+'_hhi'].notna().all()
    assert f.values[opposite+'_disclosed_amount'].isna().all()
    assert f.missing[opposite+'_disclosed_amount'].eq(M.UNDEFINED.value).all()
    assert f.values.reported_net_imbalance.isna().all()


def test_zero_amount_side_keeps_zero_amount_but_undefined_shares():
    r,a=build(amounts=(0.,0.,0.));f=features(r,a)
    assert f.values.buy_disclosed_amount.iloc[0] == 0 and f.values.buy_disclosed_rows.iloc[0] == 3
    assert f.values.buy_hhi.isna().all() and f.missing.buy_hhi.eq(M.UNDEFINED.value).all()
    assert f.values.buy_institution_labeled_rows.iloc[0] == 1


@pytest.mark.parametrize('bad',[np.nan,-1.,np.inf])
def test_bad_own_side_amount_never_becomes_partial_sum(bad):
    r,a=build();a.loc[a.direction.eq('buy') & a['rank'].eq(1),'buy_amount']=bad;f=features(r,a)
    assert f.values.buy_disclosed_amount.isna().all()
    assert f.missing.buy_disclosed_amount.eq(M.INVALID.value).all()
    assert f.values.sell_hhi.isna().all()  # ordinary report completeness is joint


def test_ordinary_missing_sell_side_is_incomplete_not_legitimate_zero():
    s,a=tables();a=a.loc[a.flag.eq('买入')];r,a,_=build_event_tables(s,a);f=features(r,a)
    assert f.values.buy_hhi.isna().all() and f.values.sell_disclosed_amount.isna().all()
    assert f.missing.buy_hhi.eq(M.INCOMPLETE.value).all()


def test_source_only_summary_stays_uncovered():
    s,a=tables();r,a,_=build_event_tables(s.iloc[:0],a);f=features(r,a)
    assert f.values.buy_hhi.notna().all()
    assert f.values.reported_buy_total.isna().all() and f.missing.reported_buy_total.eq(M.UNCOVERED.value).all()


@pytest.mark.parametrize('column',['lhb_buy_total','lhb_sell_total','lhb_turnover','daily_amount'])
def test_negative_summary_amount_invalidates_its_relationships(column):
    r,a=build();r[column]=-10.;f=features(r,a)
    name={'lhb_buy_total':'reported_buy_total','lhb_sell_total':'reported_sell_total','lhb_turnover':'reported_turnover','daily_amount':'single_day_net_to_reported_daily_amount'}[column]
    assert f.values[name].isna().all() and f.missing[name].eq(M.INVALID.value).all()


def test_zero_summary_denominator_is_undefined_without_epsilon():
    r,a=build();r['lhb_sell_total']=0.;r['daily_amount']=0.;f=features(r,a)
    assert f.values.reported_sell_total.iloc[0] == 0
    assert f.missing.reported_buy_sell_ratio.eq(M.UNDEFINED.value).all()
    assert f.missing.single_day_net_to_reported_daily_amount.eq(M.UNDEFINED.value).all()


@pytest.mark.parametrize('case',['duplicate_seat','orphan_seat','rank_gap','count_mismatch','unit','naive_clock','unknown_kind','vintage_mix'])
def test_malformed_contracts_reject(case):
    r,a=build()
    if case=='duplicate_seat':a=pd.concat([a,a.iloc[:1]])
    if case=='orphan_seat':a.loc[0,'event_id']='orphan'
    if case=='rank_gap':a.loc[a.direction.eq('buy') & a['rank'].eq(1),'rank']=4
    if case=='count_mismatch':r['buy_rows']=4
    if case=='unit':r['amount_unit']='wan_cny'
    if case=='naive_clock':r['known_at']=pd.Timestamp('2024-01-06')
    if case=='unknown_kind':r['disclosure_kind']='unknown'
    if case=='vintage_mix':r=pd.concat([r,r.assign(event_id='new',vintage='historical_versions')])
    with pytest.raises(ValueError):features(r,a)


def test_source_order_and_future_fields_do_not_change_values():
    r,a=build();f=features(r,a);r['return_5d']=-999.;a['future_outcome']=99.
    other=features(r.sample(frac=1,random_state=2),a.sample(frac=1,random_state=2))
    pd.testing.assert_frame_equal(f.values,other.values);pd.testing.assert_frame_equal(f.missing,other.missing)


def test_projection_keeps_all_stocks_and_only_known_reports():
    result=project(windows=(1,3));b=result.block
    assert len(b.values)==8 and b.values.stock_code.nunique()==2 and len(b.units)==60
    before=b.values.trade_date.lt('2024-01-08') | b.values.stock_code.eq('000002.SZ')
    assert b.values.loc[before,'lhb_w1_report_present'].isna().all()
    assert b.values.loc[~before,'lhb_w1_buy_hhi'].iloc[0]==pytest.approx(.46)
    assert len(result.lineage)==1 and result.lineage.age_sessions.iloc[0]==1


def test_empty_links_and_uncovered_absence_do_not_become_zero():
    r,a=build();result=project(panel(r,last='2024-01-05'),features(r,a))
    assert len(result.lineage)==0 and result.block.values.lhb_w1_report_present.isna().all()


def test_explicit_complete_no_event_may_set_presence_zero_not_amounts():
    r,a=build();p=panel(r,last='2024-01-05');p.panel['lhb_report_count_complete']=True
    p.panel['lhb_event_state']=M.NO_EVENT.value
    b=project(p,features(r,a),windows=(1,)).block
    assert b.values.lhb_w1_report_present.eq(0).all()
    assert b.values.lhb_w1_buy_disclosed_amount.isna().all()
    assert b.missing.lhb_w1_buy_disclosed_amount.eq(M.NO_EVENT.value).all()


def test_weak_vintage_is_not_silently_upgraded_at_projection():
    r,a=build();p=panel(r)
    with pytest.raises(ValueError,match='[Ww]eak'):latest_report_features(p,features(r,a))


@pytest.mark.parametrize('field,value',[('stock_code','000002.SZ'),('known_at','2024-01-05T00:00:00Z'),('window_days',3),('event_id','orphan'),('age_sessions',-1)])
def test_conflicting_link_contract_rejects(field,value):
    r,a=build();p=panel(r);p.links=p.links.copy();p.links.loc[0,field]=value
    with pytest.raises(ValueError):project(p,features(r,a))


def test_latest_incomplete_report_does_not_fall_back_to_prettier_old_data():
    s,a=tables(date='2024-01-04',rid='old');s2,a2=tables(date='2024-01-05',rid='new')
    a2=a2.loc[a2.flag.eq('买入')]
    r,a,_=build_event_tables(pd.concat([s,s2]),pd.concat([a,a2]));p=panel(r)
    p=attach_events(p.panel[['trade_date','stock_code','decision_at']],r,CAL,universe_id='all_supplied_keys',window_sessions=3,allow_weak_vintage=True)
    assert len(p.links.loc[p.links.sample_id.eq(p.panel.loc[p.panel.trade_date.eq('2024-01-08') & p.panel.stock_code.eq('000001.SZ'),'sample_id'].iloc[0])])==2
    result=project(p,features(r,a),windows=(1,));row=result.block.values.trade_date.eq('2024-01-08') & result.block.values.stock_code.eq('000001.SZ')
    assert result.block.values.loc[row,'lhb_w1_buy_hhi'].isna().all()
    assert result.block.missing.loc[row,'lhb_w1_buy_hhi'].eq(M.INCOMPLETE.value).all()


def test_future_append_and_row_shuffle_preserve_prior_decisions():
    r,a=build();b=features(r,a);p=panel(r);first=project(p,b)
    s2,a2=tables(date='2024-01-08',rid='future');r2,a2,_=build_event_tables(s2,a2)
    later=project(panel(pd.concat([r,r2]),last='2024-01-09'),features(pd.concat([r,r2]),pd.concat([a,a2])))
    mask=later.block.values.trade_date.le('2024-01-08')
    pd.testing.assert_frame_equal(first.block.values,later.block.values.loc[mask].reset_index(drop=True))
    pd.testing.assert_frame_equal(first.block.missing,later.block.missing.loc[mask].reset_index(drop=True))
    shuffled=deepcopy(p);shuffled.links=shuffled.links.sample(frac=1,random_state=5)
    pd.testing.assert_frame_equal(first.block.values,project(shuffled,b).block.values)


def test_report_windows_project_separately_without_summing():
    s,a=tables();s3,a3=tables(reason=THREE,rid='R3',amounts=(600.,300.,100.))
    r,a,_=build_event_tables(pd.concat([s,s3]),pd.concat([a,a3]));b=project(panel(r),features(r,a),windows=(1,3)).block
    row=b.values.stock_code.eq('000001.SZ') & b.values.trade_date.eq('2024-01-08')
    assert b.values.loc[row,'lhb_w1_buy_disclosed_amount'].iloc[0]==100
    assert b.values.loc[row,'lhb_w3_buy_disclosed_amount'].iloc[0]==1000
    assert b.values.loc[row,'lhb_w3_single_day_net_to_reported_daily_amount'].isna().all()


def test_wrong_rank_order_cannot_define_top1_concentration():
    r,a=build();buy=a.direction.eq('buy');a.loc[buy,'rank']=[3,2,1]
    with pytest.raises(ValueError,match='descending'):features(r,a)
