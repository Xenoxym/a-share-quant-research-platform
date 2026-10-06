"""Original-report LHB features and window-separated decision projections (E05).

Disclosed rows are observations, not beneficial-owner identities. Side lists,
report windows, and overlapping category disclosures are never union-summed.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import FeatureBlock
from ..contracts import MissingReason as M, Unit, VintagePolicy, timestamp, required_id
from ..panel import EventPanel, STRONG_VINTAGES

KEYS = ['event_id', 'trade_date', 'stock_code']
SIDE_UNITS = {
    'disclosed_amount': Unit.AMOUNT.value, 'disclosed_rows': Unit.COUNT.value,
    'top1_share': Unit.RATIO.value, 'top2_share': Unit.RATIO.value,
    'hhi': Unit.RATIO.value, 'entropy_log5': Unit.RATIO.value,
    'institution_labeled_rows': Unit.COUNT.value, 'institution_amount_share': Unit.RATIO.value,
    'northbound_labeled_rows': Unit.COUNT.value, 'northbound_amount_share': Unit.RATIO.value,
}
SUMMARY_UNITS = {
    'reported_buy_total': Unit.AMOUNT.value, 'reported_sell_total': Unit.AMOUNT.value,
    'reported_turnover': Unit.AMOUNT.value, 'net_from_reported_totals': Unit.AMOUNT.value,
    'reported_net_imbalance': Unit.RATIO.value, 'reported_buy_sell_ratio': Unit.RATIO.value,
    'single_day_net_to_reported_daily_amount': Unit.RATIO.value,
    'single_day_turnover_to_reported_daily_amount': Unit.RATIO.value,
}
KINDS = ('ordinary', 'margin_buy', 'short_sell', 'investor_category')
STRUCTURE_UNITS = {'window_days': Unit.COUNT.value, 'window_known': Unit.BINARY.value,
                   **{'is_' + k: Unit.BINARY.value for k in KINDS}}
UNITS = {**{side + '_' + name: unit for side in ('buy', 'sell')
            for name, unit in SIDE_UNITS.items()}, **SUMMARY_UNITS, **STRUCTURE_UNITS}


def report_features(reports, seats, *, source_id, version_id):
    """Calculate within-report features; preserve all report keys and timing limits.

    Callers verify the immutable snapshot before reading it. This function checks
    table structure and computes values; it does not authenticate vendor records.
    """
    required_id(source_id, 'source_id');required_id(version_id, 'version_id')
    required = set(KEYS) | {'observed_end', 'known_at', 'vintage', 'amount_unit',
        'window_days', 'window_known', 'disclosure_kind', 'buy_rows', 'sell_rows',
        'seat_values_valid', 'seat_status', 'summary_status', 'lhb_buy_total',
        'lhb_sell_total', 'lhb_turnover', 'daily_amount'}
    fields = {'event_id', 'seat_ordinal_id', 'direction', 'rank', 'broker_name',
              'buy_amount', 'sell_amount', 'seat_amount_valid', 'seat_name_valid'}
    if not required <= set(reports) or not fields <= set(seats):
        raise ValueError('Original-report snapshot columns are missing')
    r = reports[list(required)].copy().sort_values(KEYS).reset_index(drop=True)
    a = seats[list(fields)].copy()
    if r[KEYS].isna().any().any() or r.event_id.duplicated().any():
        raise ValueError('Report keys must be non-null and unique')
    if not r.event_id.astype(str).str.strip().ne('').all():
        raise ValueError('Empty report identity')
    if not r.trade_date.astype(str).str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
        raise ValueError('Canonical report dates required')
    pd.to_datetime(r.trade_date, format='%Y-%m-%d', errors='raise')
    if not r.stock_code.astype(str).str.fullmatch(r'\d{6}\.(SH|SZ|BJ)').all():
        raise ValueError('Exchange-qualified report stocks required')
    if not r.amount_unit.eq(Unit.AMOUNT.value).all():
        raise ValueError('CNY amounts required; no implicit ten-thousand-yuan conversion')
    vintages = r.vintage.unique()
    if len(vintages) != 1:
        raise ValueError('Partition report blocks by one explicit vintage policy')
    vintage = VintagePolicy(vintages[0]).value
    for name in ('observed_end', 'known_at'):
        r[name] = pd.to_datetime(r[name].map(timestamp), utc=True).astype('datetime64[ns, UTC]')
    if not r.observed_end.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d').eq(r.trade_date).all():
        raise ValueError('Report observation date differs from its local date')
    if not r.disclosure_kind.isin(KINDS).all() or not r.seat_status.isin(
            ['complete_disclosure', 'incomplete_or_invalid', 'non_seat_disclosure']).all():
        raise ValueError('Unknown report disclosure status')
    if not r.summary_status.isin(['matched', 'source_only']).all():
        raise ValueError('Unknown summary status')
    for name in ('window_known', 'seat_values_valid'):
        if not r[name].map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise ValueError('Report status must be boolean: ' + name)
    if not r.window_days.isin([0, 1, 3, 10, 30]).all() or not r.window_known.eq(r.window_days.gt(0)).all():
        raise ValueError('Unknown windows must be zero with an explicit unknown flag')
    if a[['event_id', 'seat_ordinal_id']].isna().any().any() or a.seat_ordinal_id.duplicated().any():
        raise ValueError('Seat occurrence identities must be non-null and unique')
    if not a.event_id.isin(r.event_id).all() or not a.direction.isin(['buy', 'sell']).all():
        raise ValueError('Seat is orphaned or has an unknown side')
    rank = pd.to_numeric(a['rank'], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(rank).all() or (rank < 1).any() or (rank != np.floor(rank)).any():
        raise ValueError('Seat ranks must be positive integers')
    if a.duplicated(['event_id', 'direction', 'rank']).any():
        raise ValueError('Duplicate within-report seat rank')
    a['rank'] = rank.astype(int)
    counts = a.groupby(['event_id', 'direction']).size().unstack(fill_value=0)
    rank_max = a.groupby(['event_id', 'direction'])['rank'].max().unstack(fill_value=0)
    for side in ('buy', 'sell'):
        n = r.event_id.map(counts.get(side, pd.Series(dtype=float))).fillna(0)
        declared = pd.to_numeric(r[side + '_rows'], errors='raise')
        if not n.eq(declared).all():
            raise ValueError('Header and actual seat row counts disagree')
        maximum = r.event_id.map(rank_max.get(side, pd.Series(dtype=float))).fillna(0)
        if not maximum.eq(n).all():
            raise ValueError('Seat ranks must be a complete contiguous ordering')
    for name in ('seat_amount_valid', 'seat_name_valid'):
        if not a[name].map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise ValueError('Seat validity must be boolean: ' + name)
    a['amount'] = np.where(a.direction.eq('buy'), pd.to_numeric(a.buy_amount, errors='coerce'),
                           pd.to_numeric(a.sell_amount, errors='coerce'))
    a['label'] = a.broker_name.fillna('').astype(str).str.strip()
    a['valid'] = np.isfinite(a.amount) & a.amount.ge(0) & a.label.ne('') & a.seat_amount_valid & a.seat_name_valid
    a['institution'] = a.label.eq('机构专用')
    a['northbound'] = a.label.isin(['沪股通专用', '深股通专用'])
    content_valid = r.event_id.map(a.groupby('event_id').valid.all()).fillna(False)
    valid = content_valid & r.seat_values_valid
    ordinary = r.disclosure_kind.eq('ordinary')
    category = r.disclosure_kind.eq('investor_category')
    complete = r.seat_status.eq('complete_disclosure') & valid
    ordinary_shape = r.buy_rows.between(1, 5) & r.sell_rows.between(1, 5)
    margin_shape = r.disclosure_kind.eq('margin_buy') & r.buy_rows.between(1, 5) & r.sell_rows.eq(0)
    short_shape = r.disclosure_kind.eq('short_sell') & r.sell_rows.between(1, 5) & r.buy_rows.eq(0)
    complete &= (ordinary & ordinary_shape) | margin_shape | short_shape
    values = r[KEYS + ['observed_end', 'known_at']].copy()
    reasons = r[KEYS].copy()

    def put(name, raw, good, why=M.STATUS):
        raw = pd.Series(raw, index=r.index, dtype=float)
        good = pd.Series(good, index=r.index, dtype=bool) & np.isfinite(raw)
        values[name] = raw.where(good)
        reasons[name] = np.where(good, M.PRESENT.value, why.value if isinstance(why, M) else why)

    for side in ('buy', 'sell'):
        own = a.loc[a.direction.eq(side)].sort_values(['event_id', 'rank']).copy()
        if (own.loc[own.valid].groupby('event_id').amount.diff() > 0).any():
            raise ValueError('Seat rank does not agree with descending side amount')
        g = own.groupby('event_id', sort=False)
        totals = g.amount.sum(min_count=1)
        own['p'] = own.amount / own.event_id.map(totals).where(lambda x: x.gt(0) & np.isfinite(x))
        own['hhi'] = own.p ** 2
        with np.errstate(divide='ignore', invalid='ignore'):
            own['entropy'] = np.where(own.p.gt(0), -own.p * np.log(own.p) / np.log(5.), 0.)
        mapped = lambda series: r.event_id.map(series).astype(float)
        raw = {
            'disclosed_amount': mapped(totals), 'disclosed_rows': mapped(g.size()),
            'top1_share': mapped(own.loc[own['rank'].eq(1)].set_index('event_id').p),
            'top2_share': mapped(own.loc[own['rank'].le(2)].groupby('event_id').p.sum(min_count=1)),
            'hhi': mapped(own.groupby('event_id').hhi.sum(min_count=1)),
            'entropy_log5': mapped(own.groupby('event_id').entropy.sum(min_count=1)),
            'institution_labeled_rows': mapped(g.institution.sum()),
            'institution_amount_share': mapped(own.loc[own.institution].groupby('event_id').amount.sum()).fillna(0) / mapped(totals),
            'northbound_labeled_rows': mapped(g.northbound.sum()),
            'northbound_amount_share': mapped(own.loc[own.northbound].groupby('event_id').amount.sum()).fillna(0) / mapped(totals),
        }
        applicable = ordinary | r.disclosure_kind.eq('margin_buy' if side == 'buy' else 'short_sell')
        good = complete & applicable & np.isfinite(mapped(totals))
        why = np.full(len(r), M.INCOMPLETE.value, dtype=object)
        why[~valid.to_numpy()] = M.INVALID.value
        why[(~applicable | category).to_numpy()] = M.UNDEFINED.value
        for name, data in raw.items():
            ratio = SIDE_UNITS[name] == Unit.RATIO.value
            allowed = good & (mapped(totals).gt(0) if ratio else True)
            missing = why.copy()
            missing[(good & ~allowed).to_numpy()] = M.UNDEFINED.value
            missing[(complete & applicable & ~np.isfinite(mapped(totals))).to_numpy()] = M.INVALID.value
            put(side + '_' + name, data, allowed, missing)
    buy = pd.to_numeric(r.lhb_buy_total, errors='coerce')
    sell = pd.to_numeric(r.lhb_sell_total, errors='coerce')
    turnover = pd.to_numeric(r.lhb_turnover, errors='coerce')
    daily = pd.to_numeric(r.daily_amount, errors='coerce')
    eligible = ordinary & r.summary_status.eq('matched')
    b = eligible & np.isfinite(buy) & buy.ge(0)
    s = eligible & np.isfinite(sell) & sell.ge(0)
    t = eligible & np.isfinite(turnover) & turnover.ge(0)
    pair = b & s & np.isfinite(buy + sell)
    net = buy - sell
    definitions = {
        'reported_buy_total': (buy, b, b), 'reported_sell_total': (sell, s, s),
        'reported_turnover': (turnover, t, t), 'net_from_reported_totals': (net, pair, pair),
        'reported_net_imbalance': (net / (buy + sell).where((buy + sell).gt(0)), pair & (buy + sell).gt(0), pair),
        'reported_buy_sell_ratio': (buy / sell.where(sell.gt(0)), pair & sell.gt(0), pair),
        'single_day_net_to_reported_daily_amount': (net / daily.where(daily.gt(0)), pair & r.window_days.eq(1) & np.isfinite(daily) & daily.gt(0), pair & r.window_days.eq(1) & np.isfinite(daily) & daily.ge(0)),
        'single_day_turnover_to_reported_daily_amount': (turnover / daily.where(daily.gt(0)), t & r.window_days.eq(1) & np.isfinite(daily) & daily.gt(0), t & r.window_days.eq(1) & np.isfinite(daily) & daily.ge(0)),
    }
    for name, (raw, good, base_valid) in definitions.items():
        why = np.where(r.summary_status.eq('source_only'), M.UNCOVERED.value, M.INVALID.value)
        why[(base_valid & ~good).to_numpy()] = M.UNDEFINED.value
        why[~ordinary.to_numpy()] = M.UNDEFINED.value
        if name.startswith('single_day_'):
            why[(ordinary & ~r.window_days.eq(1)).to_numpy()] = M.UNDEFINED.value
        put(name, raw, good, why)
    put('window_days', r.window_days, r.window_known)
    put('window_known', r.window_known, True)
    for kind in KINDS:put('is_' + kind, r.disclosure_kind.eq(kind), True)
    return FeatureBlock(values, reasons, UNITS.copy(), {
        'schema': 'lhb-report-features-v1', 'key_columns': KEYS.copy(),
        'source_id': source_id, 'version_id': version_id, 'vintage': vintage,
        'labels': 'exact disclosed name labels, not beneficial owner IDs',
        'entropy_denominator': 'log(5), never log(observed row count)',
        'amounts': 'reported CNY; list sums do not reconstruct a union of traders',
        'timing': 'copied from verified snapshot; no new historical publication certification',
        'missing': 'explicit, never blanket zero', 'inputs_exclude_future_labels': True,
    }).validate()


@dataclass
class LHBDecisionFeatures:
    block: FeatureBlock
    lineage: pd.DataFrame


def latest_report_features(panel: EventPanel, report_block: FeatureBlock, *,
                           windows=(1, 3, 10, 30, 0), allow_weak_vintage=False):
    """Project latest known ordinary report per window without shrinking membership.

    Same-date ties use knowledge timestamp, then event ID. They do not use amounts,
    future labels, prediction quality, or completeness to choose a report.
    """
    report_block.validate()
    if tuple(report_block.metadata['key_columns']) != tuple(KEYS) or report_block.units != UNITS:
        raise ValueError('Expected an original-report feature block')
    windows = tuple(windows)
    if not windows or len(set(windows)) != len(windows) or any(type(w) is not int or w not in (0, 1, 3, 10, 30) for w in windows):
        raise ValueError('Explicit unique supported report windows required')
    p = panel.panel.copy().reset_index(drop=True)
    required = ['sample_id', 'trade_date', 'stock_code', 'decision_at', 'lhb_event_state', 'lhb_report_count_complete', 'lhb_vintage_warning']
    if not set(required) <= set(p) or p.sample_id.isna().any() or p.sample_id.duplicated().any():
        raise ValueError('Unique decision panel samples required')
    if p.duplicated(['trade_date', 'stock_code']).any() or p[['trade_date', 'stock_code']].isna().any().any():
        raise ValueError('Unique non-null stock/date membership required')
    if not p.lhb_event_state.isin([x.value for x in M]).all() or any(not p[n].map(lambda v: isinstance(v, (bool, np.bool_))).all() for n in ('lhb_report_count_complete', 'lhb_vintage_warning')):
        raise ValueError('Invalid event coverage declaration')
    if not allow_weak_vintage and p.lhb_vintage_warning.any():
        raise ValueError('Weak historical coverage/report vintage requires explicit opt-in')
    p['decision_at'] = pd.to_datetime(p.decision_at.map(timestamp), utc=True).astype('datetime64[ns, UTC]')
    if not p.decision_at.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d').eq(p.trade_date).all():
        raise ValueError('Decision timestamp differs from local membership date')
    r = report_block.values.set_index('event_id', drop=False)
    reasons = report_block.missing.set_index('event_id', drop=False)
    links = panel.links.copy().reset_index(drop=True)
    required_links = {'sample_id', 'event_id', 'stock_code', 'event_date', 'known_at', 'age_sessions', 'window_days', 'disclosure_kind', 'vintage'}
    if not required_links <= set(links) or links[['sample_id', 'event_id']].isna().any().any() or links.duplicated(['sample_id', 'event_id']).any():
        raise ValueError('Invalid or duplicate decision report links')
    if not links.sample_id.isin(p.sample_id).all() or not links.event_id.isin(r.index).all():
        raise ValueError('Report link points outside supplied panel/block')
    links['known_at'] = pd.to_datetime(links.known_at.map(timestamp), utc=True).astype('datetime64[ns, UTC]')
    sample = p.set_index('sample_id')
    if len(links):
        linked = r.loc[links.event_id].reset_index(drop=True)
        cutoff = links.sample_id.map(sample.decision_at)
        expected_window = linked.window_days.fillna(0).to_numpy()
        expected_kind = np.array(KINDS, dtype=object)[linked[['is_' + k for k in KINDS]].to_numpy().argmax(axis=1)]
        age = pd.to_numeric(links.age_sessions, errors='raise').to_numpy(dtype=float)
        matches = (links.stock_code.eq(links.sample_id.map(sample.stock_code)) &
                   links.stock_code.eq(linked.stock_code) & links.event_date.eq(linked.trade_date) &
                   links.known_at.eq(linked.known_at) & links.window_days.eq(expected_window) &
                   links.disclosure_kind.eq(expected_kind) & links.vintage.eq(report_block.metadata['vintage']))
        if not matches.all() or not links.known_at.le(cutoff).all() or not linked.observed_end.le(cutoff).all():
            raise ValueError('Report link conflicts with identity/timing/window declaration')
        if not np.isfinite(age).all() or (age < 0).any() or (age != np.floor(age)).any():
            raise ValueError('Event age must be a nonnegative observed session count')
        if not allow_weak_vintage and not links.vintage.isin(STRONG_VINTAGES).all():
            raise ValueError('Weak historical vintage requires explicit opt-in')
    keys = ['sample_id', 'trade_date', 'stock_code']
    # These clocks describe a transformation at the decision, not a new publication.
    value_columns = {k: p[k] for k in keys}
    value_columns.update(observed_end=p.decision_at, known_at=p.decision_at)
    reason_columns = {k: p[k] for k in keys}
    units = {};lineage = []
    names = [n for n in UNITS if n not in STRUCTURE_UNITS]
    ordinary = links.loc[links.disclosure_kind.eq('ordinary')]
    for window in windows:
        prefix = 'lhb_w' + str(window) + '_'
        latest = ordinary.loc[ordinary.window_days.eq(window)].sort_values(
            ['sample_id', 'event_date', 'known_at', 'event_id'], kind='stable').drop_duplicates('sample_id', keep='last')
        ids = p.sample_id.map(latest.set_index('sample_id').event_id)
        selected = ids.notna()
        absence = p.lhb_report_count_complete & ~selected
        fallback = p.lhb_event_state.replace({M.PRESENT.value: M.STATUS.value}).to_numpy(dtype=object)
        fallback[absence.to_numpy()] = M.NO_EVENT.value
        for name in names:
            col = prefix + name
            value_columns[col] = ids.map(r[name]).astype(float)
            reason_columns[col] = pd.Series(np.where(selected, ids.map(reasons[name]), fallback), index=p.index)
            units[col] = UNITS[name]
        presence = prefix + 'report_present';age_col = prefix + 'age_sessions'
        value_columns[presence] = np.where(selected, 1., np.where(absence, 0., np.nan))
        reason_columns[presence] = np.where(selected | absence, M.PRESENT.value, fallback)
        value_columns[age_col] = p.sample_id.map(latest.set_index('sample_id').age_sessions).astype(float)
        reason_columns[age_col] = np.where(selected, M.PRESENT.value, fallback)
        units[presence] = Unit.BINARY.value;units[age_col] = Unit.COUNT.value
        lineage.append(latest[['sample_id', 'event_id', 'event_date', 'known_at', 'age_sessions', 'window_days']])
    result = FeatureBlock(pd.DataFrame(value_columns), pd.DataFrame(reason_columns), units, {
        'schema': 'lhb-latest-decision-features-v1', 'key_columns': keys,
        'source_id': report_block.metadata['source_id'], 'version_id': report_block.metadata['version_id'],
        'vintage': report_block.metadata['vintage'], 'weak_vintage_allowed': allow_weak_vintage,
        'universe_id': panel.metadata['universe_id'], 'observation_window_sessions': panel.metadata['window_sessions'],
        'report_windows': list(windows), 'report_selection': 'ordinary; latest event_date, known_at, event_id',
        'aggregate_amounts_across_reports': False, 'membership_rows': len(p),
        'absence': 'zero presence only under caller-declared all-report coverage',
        'clock': 'decision-time transformation; source publication retained in lineage',
    }).validate()
    return LHBDecisionFeatures(result, pd.concat(lineage, ignore_index=True))
