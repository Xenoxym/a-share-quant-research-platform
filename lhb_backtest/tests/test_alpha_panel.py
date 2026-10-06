"""Causal missingness, whole membership and budget counterexamples for E03."""
from dataclasses import replace

import pandas as pd
import pytest

from src.alpharesearch.contracts import MissingReason, VintagePolicy
from src.alpharesearch.panel import CoverageProof, attach_events
from src.technical.artifacts import content_id

CAL = ['2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08', '2024-01-09', '2024-01-10']


def membership(dates=('2024-01-08',), codes=('000001.SZ', '000002.SZ'), time='15:10'):
    return pd.DataFrame([dict(trade_date=d, stock_code=c, decision_at=d+'T'+time+':00+08:00',
                               future_label=9999) for d in dates for c in codes])


def reports(date='2024-01-05', code='000001.SZ', known='2024-01-06T00:00:00+08:00',
            vintage='market_observation', rid='r1', window=1):
    return pd.DataFrame([dict(event_id=content_id([rid]), trade_date=date, stock_code=code,
                               known_at=known, observed_end=date+'T15:00:00+08:00',
                               window_days=window, disclosure_kind='ordinary',
                               seat_status='complete_disclosure', vintage=vintage)])


def attach(p=None, r=None, **kwargs):
    return attach_events(membership() if p is None else p, reports() if r is None else r,
                         CAL, universe_id='declared_whole_membership', window_sessions=3, **kwargs)


def proof(dates=CAL, markets=('SZ',), **kwargs):
    return CoverageProof(frozenset(dates), frozenset(markets), content_id(['absence-evidence']),
                         VintagePolicy.HISTORICAL, **kwargs)


def test_unlisted_stocks_stay_and_uncertified_absence_is_missing():
    result = attach();p = result.panel.set_index('stock_code')
    assert len(p) == 2 and 'future_label' not in p
    assert p.loc['000001.SZ', 'lhb_event_state'] == 'present'
    assert p.loc['000001.SZ', 'lhb_observed_report_count'] == 1
    assert p.loc['000002.SZ', 'lhb_event_state'] == 'source_uncovered'
    assert pd.isna(p.loc['000002.SZ', 'lhb_observed_report_count'])
    assert not p.lhb_report_count_complete.any()


def test_friday_report_cannot_enter_friday_close_decision_but_monday_can():
    result = attach(membership(dates=['2024-01-05', '2024-01-08']))
    assert not result.links.decision_at.dt.strftime('%Y-%m-%d').eq('2024-01-05').any()
    assert result.links.age_sessions.tolist() == [1]
    assert pd.isna(result.panel.loc[result.panel.trade_date.eq('2024-01-05'), 'lhb_observed_report_count']).all()


def test_earlier_same_date_clock_does_not_use_later_publication():
    r = reports(date='2024-01-08', known='2024-01-08T18:00:00+08:00')
    assert attach(membership(), r).links.empty
    assert len(attach(membership(time='18:01'), r).links) == 1


def test_future_report_append_does_not_change_old_values_or_missing_flags():
    p = membership(dates=['2024-01-05', '2024-01-08']);old = attach(p)
    future = reports(date='2024-01-10', code='000002.SZ', known='2024-01-11T00:00:00+08:00', rid='future')
    new = attach(p, pd.concat([reports(), future]))
    pd.testing.assert_frame_equal(old.panel, new.panel);pd.testing.assert_frame_equal(old.links, new.links)


def test_unavailable_same_date_future_event_not_used_to_set_missingness():
    p = membership();empty = reports().iloc[:0];old = attach(p, empty)
    later = reports(date='2024-01-08', known='2024-01-08T18:00:00+08:00')
    new = attach(p, later)
    pd.testing.assert_frame_equal(old.panel, new.panel);pd.testing.assert_frame_equal(old.links, new.links)


def test_explicit_complete_publication_coverage_makes_no_event_zero():
    result = attach(coverage=proof());p = result.panel.set_index('stock_code')
    assert p.loc['000002.SZ', 'lhb_event_state'] == 'verified_no_event'
    assert p.loc['000002.SZ', 'lhb_observed_report_count'] == 0
    assert pd.isna(p.loc['000002.SZ', 'lhb_latest_age_sessions'])
    assert p.lhb_report_count_complete.all() and p.lhb_coverage_evidence_id.notna().all()


def test_partial_dates_and_unsupported_market_cannot_certify_absence():
    p = membership(codes=['000002.SZ', '600001.SH']);r = reports().iloc[:0]
    x = attach(p, r, coverage=proof(dates=['2024-01-05']))
    assert x.panel.lhb_event_state.eq('source_uncovered').all()
    y = attach(p, r, coverage=proof())
    assert y.panel.loc[y.panel.stock_code.eq('600001.SH'), 'lhb_event_state'].iloc[0] == 'source_uncovered'


def test_calendar_history_shortage_is_explicit_even_if_coverage_claims_dates():
    p = membership(dates=['2024-01-03']);r = reports().iloc[:0]
    x = attach(p, r, coverage=proof())
    assert x.panel.lhb_event_state.eq('insufficient_history').all()
    assert not x.panel.lhb_report_count_complete.any()


def test_source_only_and_all_report_windows_can_link_without_summing_money():
    r = pd.concat([reports(rid='one', window=1), reports(rid='three', window=3)])
    x = attach(r=r)
    assert len(x.links) == 2 and set(x.links.window_days) == {1, 3}
    assert x.panel.loc[x.panel.stock_code.eq('000001.SZ'), 'lhb_observed_report_count'].iloc[0] == 2
    assert not any('amount' in c for c in x.panel)


def test_weak_vintage_is_strict_by_default_and_retained_on_explicit_opt_in():
    r = reports(vintage='latest_snapshot_only')
    with pytest.raises(ValueError, match='vintage'):attach(r=r)
    x = attach(r=r, allow_weak_vintage=True)
    assert x.links.vintage.eq('latest_snapshot_only').all()
    assert x.panel.loc[x.panel.stock_code.eq('000001.SZ'), 'lhb_vintage_warning'].all()


def test_future_weak_vintage_does_not_make_an_earlier_decision_fail():
    r = reports(date='2024-01-10', known='2024-01-11T00:00:00+08:00', vintage='latest_snapshot_only')
    assert attach(r=r).links.empty


def test_coverage_weak_vintage_cannot_silently_make_absence_certain():
    c = replace(proof(), vintage=VintagePolicy.LATEST_ONLY)
    with pytest.raises(ValueError, match='coverage.*vintage'):attach(coverage=c)
    assert attach(coverage=c, allow_weak_vintage=True).panel.lhb_vintage_warning.all()


@pytest.mark.parametrize('known', [None, '2024-01-09T18:00:00+08:00'])
def test_coverage_proof_contradicting_publication_cannot_create_zero(known):
    with pytest.raises(ValueError, match='contradicts'):attach(r=reports(known=known), coverage=proof())


def test_partial_current_date_publication_does_not_complete_that_date():
    r = reports(date='2024-01-08', known='2024-01-08T15:05:00+08:00')
    x = attach(r=r, coverage=proof())
    row = x.panel.loc[x.panel.stock_code.eq('000001.SZ')].iloc[0]
    assert row.lhb_event_state == 'present' and not row.lhb_report_count_complete


def test_link_budget_fails_before_large_repeat_allocation(monkeypatch):
    import src.alpharesearch.panel as module
    def forbidden(*args, **kwargs):raise AssertionError('Budget checked too late')
    monkeypatch.setattr(module.np, 'repeat', forbidden)
    p = membership(dates=['2024-01-08', '2024-01-09']);r = pd.concat([reports(rid='a'), reports(rid='b')])
    with pytest.raises(ValueError, match='budget'):attach(p, r, max_links=1)


def test_missing_report_clock_is_not_guessed_or_exposed_as_event_presence():
    r = reports(known=None);a = attach(r=r);b = attach(r=r.iloc[:0])
    pd.testing.assert_frame_equal(a.panel, b.panel)


def test_input_order_and_host_timezone_do_not_change_keyed_values():
    p = membership(dates=['2024-01-08', '2024-01-09']);x = attach(p)
    y = attach(p.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(x.panel.sort_values('sample_id').reset_index(drop=True),
                                  y.panel.sort_values('sample_id').reset_index(drop=True))
    pd.testing.assert_frame_equal(x.links, y.links)


def test_declared_universe_changes_sample_identity_not_membership():
    p = membership();r = reports();x = attach(p, r)
    y = attach_events(p, r, CAL, universe_id='other_declared_membership', window_sessions=3)
    assert x.panel.sample_id.isin(y.panel.sample_id).sum() == 0 and len(x.panel) == len(y.panel)


def test_duplicate_membership_calendar_naive_clock_and_non_session_dates_rejected():
    p = membership();r = reports()
    with pytest.raises(ValueError, match='Duplicate'):attach(pd.concat([p, p]), r)
    with pytest.raises(ValueError, match='ordered'):attach_events(p, r, CAL[::-1], universe_id='u')
    p.decision_at = '2024-01-08T15:10:00'
    with pytest.raises(ValueError, match='timezone'):attach(p, r)
    p = membership(dates=['2024-01-06'])
    with pytest.raises(ValueError, match='calendar'):attach(p, r)


def test_selected_event_receipt_is_not_an_absence_proof():
    with pytest.raises(ValueError, match='Listed-event'):proof(scope='listed_summary_events')
    with pytest.raises(ValueError, match='immutable'):CoverageProof(frozenset(CAL), frozenset(['SZ']), 'file_exists', VintagePolicy.HISTORICAL)
    with pytest.raises(ValueError, match='markets'):proof(markets=['US'])


def test_wrong_observation_date_or_before_end_publication_rejected():
    r = reports();r.observed_end = '2024-01-06T15:00:00+08:00'
    with pytest.raises(ValueError, match='declared local'):attach(r=r)
    r = reports(known='2024-01-05T14:00:00+08:00')
    with pytest.raises(ValueError, match='before'):attach(r=r)


def test_mixed_nanosecond_and_microsecond_timestamps_compare_on_one_clock():
    p = membership();r = reports()
    p['decision_at'] = pd.to_datetime(p.decision_at, utc=True).astype('datetime64[us, UTC]')
    r['known_at'] = pd.to_datetime(r.known_at, utc=True).astype('datetime64[ns, UTC]')
    r['observed_end'] = pd.to_datetime(r.observed_end, utc=True).astype('datetime64[ms, UTC]')
    x = attach(p, r, coverage=proof())
    assert len(x.links) == 1
    assert x.panel.lhb_report_count_complete.all()
