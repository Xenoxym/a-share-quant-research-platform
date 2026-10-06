"""Original report preservation, leakage and immutable snapshot counterexamples."""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.alpharesearch import snapshots as module
from src.alpharesearch.snapshots import build_event_tables, create_event_snapshot, verify_snapshot


def report_inputs(reason='日涨幅偏离值达到7%的证券', date='2024-01-05', code='000001.SZ', rid='R1'):
    summary = pd.DataFrame([dict(trade_date=date, stock_code=code, lhb_reason=reason,
                                 lhb_buy_total=200., lhb_sell_total=100., lhb_net_buy=100.,
                                 lhb_turnover=300., daily_amount=1000., return_5d=.99,
                                 lhb_interpret='future success rate', stock_name='current name')])
    seats = pd.DataFrame([dict(trade_date=date, stock_code=code, report_id=rid, report_type='T1',
                               report_reason=reason, flag=side, broker_name='机构专用',
                               broker_code='anonymous', buy_amount=buy, sell_amount=sell,
                               net_amount=buy-sell, source_version=2, RISE_PROBABILITY_3DAY=.99)
                          for side, buy, sell in [('买入', 100., 10.), ('买入', 100., 10.), ('卖出', 20., 80.)]])
    return summary, seats


def write_sources(app, summary=None, seats=None):
    s, a = report_inputs()
    if summary is not None:s = summary
    if seats is not None:a = seats
    raw = app / 'data/raw';raw.mkdir(parents=True)
    s.to_parquet(raw / 'lhb_summary_reports.parquet', index=False)
    archive = raw / 'lhb_detail_source';archive.mkdir()
    for side, flag in [('BUY', '买入'), ('SELL', '卖出')]:
        frame = a.loc[a.flag.eq(flag)]
        file = archive / ('2024-01-01_2024-01-31_' + side + '.parquet')
        frame.to_parquet(file, index=False)
        file.with_suffix('.json').write_text(json.dumps(dict(schema=2, rows=len(frame),
                                                            bytes=file.stat().st_size,
                                                            start='2024-01-01', end='2024-01-31', side=side)))
    return app


def test_future_provider_fields_are_never_predictors_or_identities():
    s, a = report_inputs();r1, b1, _ = build_event_tables(s, a)
    s.return_5d = -100;s.lhb_interpret = 'changed using future';a.RISE_PROBABILITY_3DAY = 0
    r2, b2, _ = build_event_tables(s, a)
    pd.testing.assert_frame_equal(r1, r2);pd.testing.assert_frame_equal(b1, b2)
    assert not any('return_' in c or 'PROBABILITY' in c or 'interpret' in c for c in [*r1, *b1])


def test_anonymous_identical_seats_are_two_occurrences():
    s, a = report_inputs();r, b, quality = build_event_tables(s, a)
    assert len(b.loc[b.direction.eq('buy')]) == 2
    assert b.loc[b.direction.eq('buy'), 'rank'].tolist() == [1, 2]
    assert b.seat_ordinal_id.is_unique
    assert quality['anonymous_or_other_identical_occurrences_retained'] == 2
    assert r.seat_status.iloc[0] == 'complete_disclosure'


def test_report_identity_and_seat_order_ignore_source_row_order():
    s, a = report_inputs();x, y, _ = build_event_tables(s, a)
    u, v, _ = build_event_tables(s.sample(frac=1, random_state=2), a.sample(frac=1, random_state=3))
    pd.testing.assert_frame_equal(x, u);pd.testing.assert_frame_equal(y, v)


def test_multi_day_reports_not_collapsed_into_one_stock_date():
    s, a = report_inputs();s3, a3 = report_inputs('连续三个交易日涨幅偏离值累计达到20%的证券', rid='R3')
    a3.report_type = 'T3'
    r, b, _ = build_event_tables(pd.concat([s, s3]), pd.concat([a, a3]))
    assert len(r) == 2 and set(r.window_days) == {1, 3}
    assert r.event_id.is_unique and len(b) == 6


def test_unknown_window_remains_unknown():
    s, a = report_inputs('异常期间价格涨幅达到100%的证券');r, _, _ = build_event_tables(s, a)
    assert r.window_days.iloc[0] == 0 and not r.window_known.iloc[0]


def test_missing_ordinary_side_is_retained_as_incomplete():
    s, a = report_inputs();r, _, _ = build_event_tables(s, a.loc[a.flag.eq('买入')])
    assert r.seat_status.iloc[0] == 'incomplete_or_invalid' and r.sell_rows.iloc[0] == 0


@pytest.mark.parametrize('reason,side,kind', [('融资买入金额达到要求', '买入', 'margin_buy'),
                                          ('融券卖出数量达到要求', '卖出', 'short_sell')])
def test_legitimate_one_sided_monitoring_is_not_fabricated(reason, side, kind):
    s, a = report_inputs(reason);r, b, _ = build_event_tables(s, a.loc[a.flag.eq(side)])
    assert r.seat_status.iloc[0] == 'complete_disclosure' and r.disclosure_kind.iloc[0] == kind
    assert b.direction.nunique() == 1


def test_overlapping_investor_categories_are_not_brokerage_seats():
    s, a = report_inputs();a.loc[a.index[0], 'broker_name'] = '自然人'
    r, b, _ = build_event_tables(s, a)
    assert r.disclosure_kind.iloc[0] == 'investor_category'
    assert r.seat_status.iloc[0] == 'non_seat_disclosure' and len(b) == len(a)


@pytest.mark.parametrize('amount', [None, -1, float('inf'), 'invalid'])
def test_bad_ranked_amount_is_not_silently_zero(amount):
    s, a = report_inputs();a.buy_amount = a.buy_amount.astype(object);a.loc[0, 'buy_amount'] = amount
    r, b, _ = build_event_tables(s, a)
    assert r.seat_status.iloc[0] == 'incomplete_or_invalid' and not b.seat_amount_valid.all()


def test_more_than_five_source_rows_are_not_truncated():
    s, a = report_inputs();a = pd.concat([a, a.loc[a.flag.eq('买入')], a.loc[a.flag.eq('买入')]])
    r, b, _ = build_event_tables(s, a)
    assert r.buy_rows.iloc[0] == 6 and len(b) == 7
    assert r.seat_status.iloc[0] == 'incomplete_or_invalid'


def test_exact_summary_repetitions_recorded_but_conflicts_rejected():
    s, a = report_inputs();r, _, q = build_event_tables(pd.concat([s, s]), a)
    assert r.summary_occurrences.iloc[0] == 2 and q['summary_exact_repetitions_collapsed'] == 1
    other = s.copy();other.lhb_buy_total = 999
    with pytest.raises(ValueError, match='Conflicting'):build_event_tables(pd.concat([s, other]), a)


def test_ambiguous_provider_identity_does_not_pick_first():
    s, a = report_inputs();other = a.copy();other.report_id = 'different'
    with pytest.raises(ValueError, match='Ambiguous'):build_event_tables(s, pd.concat([a, other]))


def test_unmatched_summary_rejected_and_source_only_preserved():
    s, a = report_inputs();other, seats = report_inputs(code='000002.SZ', rid='R2')
    with pytest.raises(ValueError, match='no archived'):build_event_tables(pd.concat([s, other]), a)
    r, _, q = build_event_tables(s, pd.concat([a, seats]))
    assert q['source_only_reports'] == 1
    assert r.loc[r.stock_code.eq('000002.SZ'), 'lhb_buy_total'].isna().all()


def test_no_false_timestamp_or_historical_vintage_certification():
    s, a = report_inputs();r, _, q = build_event_tables(s, a)
    assert r.known_at.iloc[0].isoformat() == '2024-01-05T16:00:00+00:00'
    assert r.observed_end.iloc[0].isoformat() == '2024-01-05T07:00:00+00:00'
    assert r.vintage.iloc[0] == 'latest_snapshot_only' and not r.original_timestamp_available.any()
    assert not q['no_event_coverage_certified']


@pytest.mark.parametrize('column,value', [('trade_date', '20240105'), ('trade_date', '2024-02-30'),
                                          ('stock_code', '000001'), ('flag', 'unknown'),
                                          ('report_id', None), ('report_reason', '')])
def test_bad_source_keys_fail_closed(column, value):
    s, a = report_inputs();a.loc[0, column] = value
    with pytest.raises((ValueError, TypeError)):build_event_tables(s, a)


def test_immutable_snapshot_reuse_and_predictor_columns(tmp_path):
    app = write_sources(tmp_path / 'app');folder = create_event_snapshot(app)
    m = verify_snapshot(folder);again = create_event_snapshot(app)
    assert folder == again and m == verify_snapshot(again)
    assert m['quality']['source_reports'] == 1 and m['quality']['source_rows'] == 3
    assert all(not Path(r['path']).is_absolute() for r in m['identity']['inputs']['sources'])
    assert not {'return_5d', 'lhb_interpret'} & set(pd.read_parquet(folder / 'reports.parquet'))


def test_snapshot_identity_binds_artifacts_even_if_manifest_hash_is_edited(tmp_path):
    app = write_sources(tmp_path / 'app');folder = create_event_snapshot(app)
    file = folder / 'seats.parquet';data = pd.read_parquet(file);data.buy_amount += 10;data.to_parquet(file)
    m = json.loads((folder / 'manifest.json').read_text());m['artifacts']['seats.parquet'] = module.digest(file)
    (folder / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='binds'):verify_snapshot(folder)


def test_corruption_and_unsafe_manifest_paths_rejected(tmp_path):
    app = write_sources(tmp_path / 'app');folder = create_event_snapshot(app)
    file = folder / 'reports.parquet';file.write_bytes(file.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='integrity'):verify_snapshot(folder)
    m = json.loads((folder / 'manifest.json').read_text());m['artifacts']['../outside'] = '0'*64
    (folder / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):verify_snapshot(folder)


def test_archive_marker_mismatch_prevents_any_publication(tmp_path):
    app = write_sources(tmp_path / 'app');marker = next((app / 'data/raw/lhb_detail_source').glob('*.json'))
    data = json.loads(marker.read_text());data['rows'] += 1;marker.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='marker'):create_event_snapshot(app)
    assert not (app / 'data/alpharesearch/event_snapshots').exists()


def test_source_change_during_read_prevents_publication(tmp_path, monkeypatch):
    app = write_sources(tmp_path / 'app');load = module._load_sources
    def changed(*args):
        result = load(*args);f = app / 'data/raw/lhb_summary_reports.parquet'
        f.write_bytes(f.read_bytes() + b'changed');return result
    monkeypatch.setattr(module, '_load_sources', changed)
    with pytest.raises(ValueError, match='Source changed'):create_event_snapshot(app)
    assert not (app / 'data/alpharesearch/event_snapshots').exists()


def test_future_report_append_preserves_all_old_report_and_seat_values():
    s, a = report_inputs();old_r, old_b, _ = build_event_tables(s, a)
    future_s, future_a = report_inputs(date='2024-01-12', rid='FUTURE')
    new_r, new_b, _ = build_event_tables(pd.concat([s, future_s]), pd.concat([a, future_a]))
    pd.testing.assert_frame_equal(old_r, new_r.loc[new_r.trade_date.eq('2024-01-05')].reset_index(drop=True))
    pd.testing.assert_frame_equal(old_b, new_b.loc[new_b.event_id.isin(old_r.event_id)].reset_index(drop=True))


def test_mixed_ingestion_versions_are_not_historical_revision_evidence():
    s, a = report_inputs();a.loc[0, 'source_version'] = 3
    with pytest.raises(ValueError, match='Mixed ingestion'):build_event_tables(s, a)
