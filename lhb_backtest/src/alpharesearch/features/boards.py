"""Causal daily candle and price-limit structure; no inferred intraday order.

Limits must already be validated by the source adapter. Current names, fixed
percentage thresholds and future terminal streak lengths are never inputs.
"""
import numpy as np
import pandas as pd

from .base import FeatureBlock
from ..contracts import MissingReason as M, Unit, VintagePolicy, required_id


def _streak(flags, sessions):
    """Exact observed-calendar streak only after a known non-limit boundary."""
    exact = np.full(len(flags), np.nan);lower = exact.copy();age = exact.copy()
    previous = None;previous_session = None;run = 0;complete = False;last_hit = None
    for i, (flag, session) in enumerate(zip(flags, sessions)):
        if np.isfinite(flag):
            if flag == 0:
                run = 0;complete = True
            else:
                adjacent = previous_session is not None and session == previous_session + 1
                if adjacent and previous is not None:
                    run = run + 1 if previous == 1 else 1
                    # A prior known non-hit establishes an exact start boundary.
                    complete = complete if previous == 1 else True
                else:
                    run = 1;complete = False
                last_hit = session
            lower[i] = run
            if complete:exact[i] = run
            previous = flag
        else:
            previous = None;run = 0;complete = False
        if last_hit is not None:age[i] = session - last_hit
        previous_session = session
    return exact, lower, age


def price_board_features(bars, calendar, *, source_id, version_id,
                         vintage=VintagePolicy.MARKET, available_delay_seconds=60,
                         tick_size=0.01):
    """29 structural features, reasons and declared end-of-day availability.

    Caller supplies historical calendar and source/version identity. The default
    15:01 local availability is a declared research rule, not measured live feed
    latency. Whole-day OHLC cannot be consumed by a pre-close decision.
    A-share source prices are declared CNY/share with an explicit scalar tick.
    """
    required_id(source_id, 'source_id');required_id(version_id, 'version_id')
    vintage = VintagePolicy(vintage)
    if type(available_delay_seconds) is not int or not 0 <= available_delay_seconds <= 86400:
        raise ValueError('Declare a non-negative bounded daily availability delay')
    if isinstance(tick_size, bool) or not np.isfinite(tick_size) or tick_size <= 0:
        raise ValueError('Price tick must be finite and positive')
    required = {'trade_date', 'stock_code', 'open', 'high', 'low', 'close',
                'pre_close', 'volume', 'high_limit', 'low_limit', 'limit_price_valid'}
    if not required <= set(bars):
        raise ValueError('Missing daily source fields, including explicit limit validity')
    calendar = list(calendar)
    if not calendar or calendar != sorted(set(calendar)):
        raise ValueError('Trading calendar must be ordered and unique')
    c = pd.to_datetime(pd.Series(calendar), format='%Y-%m-%d', errors='raise')
    if c.dt.strftime('%Y-%m-%d').tolist() != calendar:
        raise ValueError('Calendar dates must be canonical')
    b = bars[list(required)].copy()
    if b.empty or b[['stock_code', 'trade_date']].isna().any().any() or b.duplicated(['stock_code', 'trade_date']).any():
        raise ValueError('Daily stock/date keys must be non-null and unique')
    supported = b.stock_code.astype(str).str.fullmatch(r'(?:60\d{4}|68\d{4})\.SH|(?:00\d{4}|30\d{4})\.SZ|(?:43|83|87|92)\d{4}\.BJ')
    if not supported.all():
        raise ValueError('This block requires declared CNY A-share source prices')
    b['_session'] = b.trade_date.map({d: i for i, d in enumerate(calendar)})
    if b._session.isna().any():raise ValueError('Daily source date absent from calendar')
    b = b.sort_values(['stock_code', '_session']).reset_index(drop=True)
    for name in ['open', 'high', 'low', 'close', 'pre_close', 'volume', 'high_limit', 'low_limit']:
        b[name] = pd.to_numeric(b[name], errors='coerce').astype(float)
        b.loc[~np.isfinite(b[name]), name] = np.nan
    flags = b.limit_price_valid
    if not flags.map(lambda v: pd.isna(v) or isinstance(v, (bool, np.bool_))).all():
        raise ValueError('Limit validity requires actual Boolean values')
    eps = tick_size / 2
    o, h, l, close, previous, volume, upper, lower = [b[n].to_numpy() for n in
                   ['open', 'high', 'low', 'close', 'pre_close', 'volume', 'high_limit', 'low_limit']]
    def on_tick(x):return np.isfinite(x) & (np.abs(x / tick_size - np.round(x / tick_size)) < 1e-5)
    have = np.isfinite(np.column_stack([o, h, l, close])).all(axis=1)
    tick_valid = np.logical_and.reduce([on_tick(x) for x in [o, h, l, close]])
    o, h, l, close = [np.round(x / tick_size) * tick_size for x in [o, h, l, close]]
    # Align all on-grid prices, not only OHLC: identical ticks must have zero
    # distance/gap instead of an artificial sign from mixed float representations.
    previous, upper, lower = [np.where(on_tick(x), np.round(x / tick_size) * tick_size, x)
                              for x in [previous, upper, lower]]
    shape = have & (np.minimum.reduce([o, h, l, close]) > 0) & (h >= l) & (h + eps >= np.maximum(o, close)) & (l - eps <= np.minimum(o, close))
    shape &= tick_valid
    traded = np.isfinite(volume) & (volume > 0)
    valid = shape & traded
    bar_reason = np.full(len(b), M.PRESENT.value, dtype=object)
    bar_reason[~have] = M.INCOMPLETE.value
    bar_reason[have & ~shape] = M.INVALID.value
    bar_reason[shape & ~traded] = M.STATUS.value
    declared = flags.eq(True).fillna(False).to_numpy(dtype=bool)
    have_limits = np.isfinite(upper) & np.isfinite(lower)
    bands = have_limits & (lower > 0) & (upper > lower) & on_tick(upper) & on_tick(lower)
    bands &= (h <= upper + eps) & (l >= lower - eps)
    limit_valid = valid & declared & bands
    limit_reason = bar_reason.copy()
    limit_reason[valid & ~declared] = M.STATUS.value
    limit_reason[valid & declared & ~have_limits] = M.INCOMPLETE.value
    limit_reason[valid & declared & have_limits & ~bands] = M.INVALID.value
    keys = ['trade_date', 'stock_code']
    values = b[keys].copy();missing = values.copy();units = {}
    local = pd.to_datetime(b.trade_date, format='%Y-%m-%d').dt.tz_localize('Asia/Shanghai')
    values['observed_end'] = (local + pd.Timedelta(hours=15)).dt.tz_convert('UTC').astype('datetime64[ns, UTC]')
    values['known_at'] = values.observed_end + pd.Timedelta(seconds=available_delay_seconds)
    def put(name, data, unit, reason):
        data = np.asarray(data, dtype=float).copy();reason = np.asarray(reason, dtype=object).copy()
        data[reason != M.PRESENT.value] = np.nan
        values[name] = data;missing[name] = reason;units[name] = unit.value
    width = h - l
    fraction_reason = bar_reason.copy()
    fraction_reason[valid & (width == 0)] = M.UNDEFINED.value
    def ratio(numerator, denominator):
        result = np.full(len(b), np.nan)
        np.divide(numerator, denominator, out=result, where=np.isfinite(denominator) & (denominator > 0))
        return result
    put('upper_wick_fraction', ratio(np.maximum(h - np.maximum(o, close), 0), width), Unit.RATIO, fraction_reason)
    put('lower_wick_fraction', ratio(np.maximum(np.minimum(o, close) - l, 0), width), Unit.RATIO, fraction_reason)
    put('absolute_body_fraction', ratio(np.abs(close-o), width), Unit.RATIO, fraction_reason)
    put('body_direction', np.sign(close-o), Unit.RATIO, bar_reason)
    put('zero_range_bar', width == 0, Unit.BINARY, bar_reason)
    gap_reason = bar_reason.copy();gap_reason[valid & (~np.isfinite(previous) | (previous <= 0))] = M.INCOMPLETE.value
    put('overnight_gap_ratio', ratio(o, previous)-1, Unit.RATIO, gap_reason)
    gaps = b.groupby('stock_code', sort=False)._session.diff().to_numpy() - 1
    reason = np.where(np.isfinite(gaps), M.PRESENT.value, M.HISTORY.value)
    put('quote_gap_sessions', gaps, Unit.COUNT, reason)
    equal = lambda x, y: np.abs(x-y) < eps
    cu, cd, ou, od = equal(close, upper), equal(close, lower), equal(o, upper), equal(o, lower)
    hu, ld = equal(h, upper), equal(l, lower)
    board_flags = {'close_upper_limit': cu, 'close_lower_limit': cd,
                   'open_upper_limit': ou, 'open_lower_limit': od,
                   'touched_upper_limit': hu, 'touched_lower_limit': ld,
                   'one_price_upper_limit': cu & equal(o, upper) & equal(h, upper) & equal(l, upper),
                   'one_price_lower_limit': cd & equal(o, lower) & equal(h, lower) & equal(l, lower),
                   'hit_both_limits': hu & ld,
                   'upper_touch_close_inside': hu & ~cu,
                   'lower_touch_close_inside': ld & ~cd,
                   'upper_open_lower_close': ou & cd,
                   'lower_open_upper_close': od & cu}
    for name, data in board_flags.items():put(name, data, Unit.BINARY, limit_reason)
    distance_reason = limit_reason.copy();distance_reason[limit_valid & (~np.isfinite(previous) | (previous <= 0))] = M.INCOMPLETE.value
    put('upper_limit_distance_close', ratio(upper-close, previous), Unit.RATIO, distance_reason)
    put('lower_limit_distance_close', ratio(close-lower, previous), Unit.RATIO, distance_reason)
    put('limit_span_ratio', ratio(upper-lower, previous), Unit.RATIO, distance_reason)
    for side in ('upper', 'lower'):
        exact = np.full(len(b), np.nan);bound = exact.copy();age = exact.copy()
        flag = values['close_' + side + '_limit'].to_numpy()
        for indices in b.groupby('stock_code', sort=False).groups.values():
            pos = np.asarray(indices, dtype=int)
            exact[pos], bound[pos], age[pos] = _streak(flag[pos], b.loc[pos, '_session'].to_numpy())
        reason = limit_reason.copy();reason[limit_valid & ~np.isfinite(exact)] = M.HISTORY.value
        put(side + '_close_streak', exact, Unit.COUNT, reason)
        put(side + '_close_streak_lower_bound', bound, Unit.COUNT, limit_reason)
        reason = np.where(np.isfinite(age), M.PRESENT.value, M.HISTORY.value)
        put('age_since_observed_' + side + '_close', age, Unit.COUNT, reason)
    metadata = {'schema': 'daily-structure-boards-v1', 'key_columns': keys,
                'source_id': source_id, 'version_id': version_id, 'vintage': vintage.value,
                'availability_basis': 'declared_market_availability_rule',
                'available_delay_seconds': available_delay_seconds, 'tick_size_cny': tick_size,
                'limit_basis': 'explicitly_valid_source_limits; may be upstream rule-generated',
                'streak_definition': 'consecutive_market_calendar_sessions_with_valid_traded_bars',
                'intraday_order_supported': False, 'orderbook_supported': False,
                'feature_count': len(units), 'rows': len(values)}
    return FeatureBlock(values, missing, units, metadata).validate()
