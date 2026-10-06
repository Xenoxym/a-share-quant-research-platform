"""Decision-time original-report links for any declared stock/date membership.

This is a data adapter, not a stock-selection rule or a trained strategy. It never
uses unavailable reports to choose missingness flags. Absence requires explicit
all-original-report coverage evidence; selected-seat coverage is insufficient.
"""
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from .contracts import MissingReason, VintagePolicy, timestamp
from ..technical.artifacts import content_id

REPORT_FIELDS = ['event_id', 'trade_date', 'stock_code', 'known_at', 'observed_end',
                 'window_days', 'disclosure_kind', 'seat_status', 'vintage']
LINK_FIELDS = ['sample_id', 'event_id', 'stock_code', 'decision_at', 'event_date',
               'known_at', 'age_sessions', 'window_days', 'disclosure_kind',
               'seat_status', 'vintage']
STRONG_VINTAGES = {VintagePolicy.HISTORICAL.value, VintagePolicy.MARKET.value}


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Dates must be canonical YYYY-MM-DD')
    pd.Timestamp(value)  # rejects impossible civil dates
    return value


@dataclass(frozen=True)
class CoverageProof:
    """Operator-supplied, scoped absence certificate, not inferred from filenames.

    The evidence ID must identify externally verified all-report coverage. This
    class validates a declaration, not the truth or existence of that evidence.
    A receipt covering only listed events must not be promoted into this contract.
    """
    report_dates: frozenset
    markets: frozenset
    evidence_id: str
    vintage: VintagePolicy
    scope: str = 'all_original_reports'
    availability_rule: str = 'all_date_reports_known_by_next_midnight'

    def __post_init__(self):
        object.__setattr__(self, 'report_dates', frozenset(_date(d) for d in self.report_dates))
        object.__setattr__(self, 'markets', frozenset(self.markets))
        object.__setattr__(self, 'vintage', VintagePolicy(self.vintage))
        if not self.markets or not self.markets <= {'SH', 'SZ', 'BJ'}:
            raise ValueError('Coverage markets must be declared explicitly')
        if not re.fullmatch(r'[0-9a-f]{64}', self.evidence_id):
            raise ValueError('Coverage needs an immutable evidence identifier')
        if self.availability_rule != 'all_date_reports_known_by_next_midnight':
            raise ValueError('Coverage must declare complete date publication availability')
        if self.scope != 'all_original_reports':
            raise ValueError('Listed-event coverage cannot certify no-event absence')


@dataclass
class EventPanel:
    panel: pd.DataFrame
    links: pd.DataFrame
    metadata: dict


def attach_events(membership, reports, calendar, *, universe_id, window_sessions=20,
                  coverage=None, allow_weak_vintage=False, max_links=1_000_000):
    """Keep membership unchanged; link all reports known by each decision.

    Observation window: current trading date plus preceding window_sessions-1
    calendar sessions, additionally constrained by actual/declared known_at.
    No current-day report is assumed published merely because its date matches.
    Known report count is a lower bound when absence coverage is not certified.
    """
    if not isinstance(universe_id, str) or not universe_id.strip():
        raise ValueError('Declare the supplied universe membership identity')
    if type(window_sessions) is not int or window_sessions < 1:
        raise ValueError('Observation window must be a positive session count')
    if type(max_links) is not int or max_links < 1:
        raise ValueError('Link budget must be positive')
    if type(allow_weak_vintage) is not bool:
        raise ValueError('Weak vintage permission must be explicit Boolean')
    if coverage is not None and not isinstance(coverage, CoverageProof):
        raise ValueError('Absence coverage must use CoverageProof')
    calendar = [_date(d) for d in calendar]
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError('Trading calendar must be ordered and unique')
    dates = {d: i for i, d in enumerate(calendar)}
    required = ['trade_date', 'stock_code', 'decision_at']
    if not set(required) <= set(membership):
        raise ValueError('Membership requires date, stock and decision timestamp')
    # Extra caller columns, including labels, do not automatically become X.
    p = membership[required].reset_index(drop=True).copy()
    if p.empty:
        raise ValueError('Empty membership is not a panel study')
    p['trade_date'] = p.trade_date.map(_date)
    if p.stock_code.isna().any() or not p.stock_code.astype(str).str.fullmatch(r'\d{6}\.(SH|SZ|BJ)').all():
        raise ValueError('Invalid exchange-qualified stock code')
    p['stock_code'] = p.stock_code.astype(str)
    if p.duplicated(['trade_date', 'stock_code']).any():
        raise ValueError('Duplicate date/stock membership key')
    p['decision_at'] = pd.to_datetime(p.decision_at.map(timestamp), utc=True).astype('datetime64[ns, UTC]')
    if not p.decision_at.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d').eq(p.trade_date).all():
        raise ValueError('Decision must fall on the declared local trading date')
    p['_session'] = p.trade_date.map(dates)
    if p._session.isna().any():
        raise ValueError('Membership date absent from canonical trading calendar')
    p['_session'] = p._session.astype(int)
    p['sample_id'] = [content_id([universe_id, d, c, t.isoformat()])
                      for d, c, t in p[required].itertuples(index=False, name=None)]
    if not set(REPORT_FIELDS) <= set(reports):
        raise ValueError('Report snapshot lacks timing/identity fields')
    r = reports[REPORT_FIELDS].copy()
    if r.event_id.isna().any() or r.event_id.duplicated().any():
        raise ValueError('Report identities must be non-null and unique')
    r['trade_date'] = r.trade_date.map(_date)
    # Future and pre-calendar data are not needed for this historical slice.
    earliest = calendar[max(0, int(p._session.min()) - window_sessions + 1)]
    r = r.loc[r.trade_date.between(earliest, p.trade_date.max()) &
              r.stock_code.isin(p.stock_code)].reset_index(drop=True)
    r['_session'] = r.trade_date.map(dates)
    if r._session.isna().any():
        raise ValueError('Observed report date absent from canonical trading calendar')
    r['_session'] = r._session.astype(int)
    r['observed_end'] = pd.to_datetime(r.observed_end.map(timestamp), utc=True).astype('datetime64[ns, UTC]')
    r['known_at'] = pd.to_datetime(r.known_at.map(lambda v: None if pd.isna(v) else timestamp(v)), utc=True).astype('datetime64[ns, UTC]')
    if not r.observed_end.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d').eq(r.trade_date).all():
        raise ValueError('Report observation must end on its declared local report date')
    known = r.known_at.notna()
    if (r.loc[known, 'known_at'] < r.loc[known, 'observed_end']).any():
        raise ValueError('Report cannot be known before its observation ends')
    if not r.vintage.isin([v.value for v in VintagePolicy]).all():
        raise ValueError('Unknown report vintage declaration')
    if coverage is not None:
        scoped = r.trade_date.isin(coverage.report_dates) & r.stock_code.str[-2:].isin(coverage.markets)
        boundary = (pd.to_datetime(r.trade_date).dt.tz_localize('Asia/Shanghai') + pd.Timedelta(days=1)).dt.tz_convert('UTC')
        if (scoped & (~known | r.known_at.gt(boundary))).any():
            raise ValueError('Coverage publication assumption contradicts report availability')
    r = r.loc[known & r.known_at.le(p.decision_at.max())].reset_index(drop=True)
    groups = []
    total = 0
    report_groups = r.groupby('stock_code', sort=False).groups
    for code, indices in p.groupby('stock_code', sort=True).groups.items():
        pi = p.loc[indices].sort_values(['decision_at', '_session']).index.to_numpy()
        ri = np.asarray(report_groups.get(code, []), dtype=int)
        if not len(ri):continue
        decisions = p.loc[pi, 'decision_at'].astype('int64').to_numpy()
        sessions = p.loc[pi, '_session'].to_numpy()
        observed = r.loc[ri, '_session'].to_numpy()
        start = np.maximum(np.searchsorted(decisions, r.loc[ri, 'known_at'].astype('int64').to_numpy()),
                           np.searchsorted(sessions, observed))
        stop = np.searchsorted(sessions, observed + window_sessions)
        count = np.maximum(stop - start, 0)
        total += int(count.sum())
        if total > max_links:
            raise ValueError('Link materialization budget exceeded; partition the input dates')
        groups.append((pi, ri, start, stop, count))
    rows, events = [], []
    for pi, ri, start, stop, count in groups:
        active = count > 0
        if not active.any():continue
        events.append(np.repeat(ri[active], count[active]))
        rows.append(np.concatenate([pi[lo:hi] for lo, hi in zip(start[active], stop[active])]))
    if rows:
        pi = np.concatenate(rows);ri = np.concatenate(events)
        if not allow_weak_vintage and not r.loc[ri, 'vintage'].isin(STRONG_VINTAGES).all():
            raise ValueError('Eligible report has latest-only or unknown historical vintage')
        left = p.loc[pi, ['sample_id', 'stock_code', 'decision_at']].reset_index(drop=True)
        right = r.loc[ri, ['event_id', 'trade_date', 'known_at', 'window_days',
                            'disclosure_kind', 'seat_status', 'vintage']].reset_index(drop=True)
        right = right.rename(columns={'trade_date': 'event_date'})
        links = pd.concat([left, right], axis=1)
        links['age_sessions'] = p.loc[pi, '_session'].to_numpy() - r.loc[ri, '_session'].to_numpy()
        links = links[LINK_FIELDS].sort_values(['sample_id', 'event_id']).reset_index(drop=True)
    else:
        links = pd.DataFrame(columns=LINK_FIELDS)
    counts = links.groupby('sample_id').size()
    p['lhb_observed_report_count'] = p.sample_id.map(counts).astype(float)
    p['lhb_latest_age_sessions'] = p.sample_id.map(links.groupby('sample_id').age_sessions.min()).astype(float)
    p['lhb_vintage_warning'] = p.sample_id.isin(links.loc[~links.vintage.isin(STRONG_VINTAGES), 'sample_id'])
    start = p._session.to_numpy() - window_sessions + 1
    history = start >= 0
    local_calendar = pd.to_datetime(pd.Series(calendar)).dt.tz_localize('Asia/Shanghai')
    release = (local_calendar + pd.Timedelta(days=1)).dt.tz_convert('UTC').astype('datetime64[ns, UTC]').astype('int64').to_numpy()
    ready = np.minimum(p._session.to_numpy(), np.searchsorted(release, p.decision_at.astype('int64').to_numpy(), side='right') - 1)
    lo = np.maximum(start, 0);hi = ready
    complete = np.zeros(len(p), dtype=bool)
    if coverage is not None:
        covered = np.array([d in coverage.report_dates for d in calendar])
        missing = np.r_[0, np.cumsum(~covered)]
        markets = p.stock_code.str[-2:].isin(coverage.markets).to_numpy()
        complete = history & (hi >= lo) & markets & ((missing[np.maximum(hi+1, 0)] - missing[lo]) == 0)
        # Exact current-day publications can exist before the full date is closed.
        # Such observed partial dates must not be certified by prior-date coverage.
        if len(links):
            latest = links.groupby('sample_id').event_date.max()
            latest_session = p.sample_id.map(latest).map(dates).to_numpy()
            complete &= ~(latest_session > ready)
        if complete.any() and coverage.vintage.value not in STRONG_VINTAGES:
            if not allow_weak_vintage:
                raise ValueError('Absence coverage has latest-only or unknown vintage')
            p.loc[complete, 'lhb_vintage_warning'] = True
    present = p.lhb_observed_report_count.notna().to_numpy()
    state = np.full(len(p), MissingReason.UNCOVERED.value, dtype=object)
    state[~history] = MissingReason.HISTORY.value
    state[history & (hi < lo)] = MissingReason.NOT_KNOWN.value
    state[complete] = MissingReason.NO_EVENT.value
    state[present] = MissingReason.PRESENT.value
    p['lhb_event_state'] = state
    p.loc[(~present) & complete, 'lhb_observed_report_count'] = 0.
    p['lhb_history_complete'] = history
    p['lhb_report_count_complete'] = complete
    p['lhb_coverage_evidence_id'] = None
    if coverage is not None:p.loc[complete, 'lhb_coverage_evidence_id'] = coverage.evidence_id
    p = p.drop(columns='_session')
    return EventPanel(p, links, {'schema': 'lhb-decision-panel-v1', 'universe_id': universe_id,
                               'window_sessions': window_sessions, 'links': len(links),
                               'membership_rows': len(p), 'weak_vintage_allowed': allow_weak_vintage,
                               'coverage_declared': coverage is not None,
                               'counts_are_observed_lower_bounds_without_coverage': True,
                               'window': 'current_date_and_previous_n_minus_one_sessions'})
