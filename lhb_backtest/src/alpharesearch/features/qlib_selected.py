"""Six fixed Qlib expressions translated into existing causal feature contracts.

The injected evaluator supplies the mature backend, not a new formula engine.
Its store and installed wheel must be independently verified by the executor.
This module cannot authenticate a callback or historical vendor vintages.
"""
import numpy as np
import pandas as pd

from .base import FeatureBlock
from ..contracts import MissingReason as M, Unit, timestamps

QLIB_VERSION = '0.9.7'
QLIB_WHEEL_SHA256 = 'dfbee9f0f3005fe805798e2a21c73b198272f1341d5f0d7771e127166faac08e'
FORMULAS = (
    ('close', '$close', 1, Unit.PRICE.value, 'Float32 mirror of the reported unadjusted CNY closing price'),
    ('close_lag5_ratio', 'Ref($close,5)/$close-1', 6, Unit.RATIO.value, 'Close five declared sessions ago / current close - 1; positive for a decline'),
    ('mean20_to_close', 'Mean($close,20)/$close-1', 20, Unit.RATIO.value, 'Mean close over twenty declared sessions / current close - 1'),
    ('slope20_to_close', 'Slope($close,20)/$close', 20, Unit.RATIO.value, 'Twenty-session regression slope / current close; one session is the time unit'),
    ('rsquare20', 'Rsquare($close,20)', 20, Unit.RATIO.value, 'Twenty-session close-on-session regression R squared; a flat window is undefined'),
    ('resi20_to_close', 'Resi($close,20)/$close', 20, Unit.RATIO.value, 'Last-point residual of twenty-session regression / current close'),
)


def qlib_selected_features(quote_block, calendar, evaluate, *, backend_version,
                           backend_wheel_sha256, selected_names=None):
    """Adapt a complete declared stock/day grid; preserve explicit missing values.

    evaluate(stock_code, fixed_expression, calendar) returns one numeric vector.
    The executor binds that callback to the verified Qlib provider and same store.
    Close vectors must exactly equal the source's float32 mirror, including NaNs.
    Rolling features require a complete valid window, unlike Qlib's short-prefix
    values. Missing observations inside any chosen window remain incomplete;
    no imputation or crossing of an absent calendar row is allowed. Row clocks
    conservatively cover the largest selected dependency window. No trade or
    target is constructed here; caller-owned decision assembly checks availability.
    """
    if backend_version != QLIB_VERSION or backend_wheel_sha256 != QLIB_WHEEL_SHA256:
        raise ValueError('Only the explicitly tested Qlib wheel is supported')
    if not callable(evaluate):
        raise ValueError('An explicitly bound backend evaluator is required')
    names = tuple(x[0] for x in FORMULAS) if selected_names is None else tuple(selected_names)
    allowed = {x[0]: x for x in FORMULAS}
    if not names or len(set(names)) != len(names) or any(n not in allowed for n in names):
        raise ValueError('Select unique allowed feature names; arbitrary expressions are forbidden')
    quote_block.validate()
    if quote_block.metadata['schema'] != 'daily-observation-primitives-v1' or quote_block.units.get('close') != Unit.PRICE.value:
        raise ValueError('Declared unadjusted daily CNY price block required')
    if quote_block.metadata.get('key_columns') != ['trade_date', 'stock_code']:
        raise ValueError('Daily stock/date source keys required')
    dates = tuple(calendar)
    if not dates or dates != tuple(sorted(set(dates))) or any(not isinstance(d, str) for d in dates):
        raise ValueError('Unique ordered canonical calendar required')
    parsed = pd.to_datetime(list(dates), format='%Y-%m-%d', errors='raise')
    if tuple(parsed.strftime('%Y-%m-%d')) != dates:
        raise ValueError('Canonical calendar dates required')
    values = quote_block.values
    if values.empty or not values.stock_code.astype(str).str.fullmatch(r'\d{6}\.(SH|SZ|BJ)').all():
        raise ValueError('Nonempty exchange-qualified source grid required')
    if set(values.trade_date) != set(dates):
        raise ValueError('Source dates must match the complete declared calendar')
    units = {n: allowed[n][3] for n in names}
    max_window = max(allowed[n][2] for n in names)
    output_values, output_missing = [], []
    for code in sorted(values.stock_code.unique()):
        source = values.loc[values.stock_code.eq(code)].sort_values('trade_date').reset_index(drop=True)
        if tuple(source.trade_date) != dates:
            raise ValueError('Each stock needs a complete declared grid; gaps cannot be silently skipped')
        observed = timestamps(source.observed_end)
        known = timestamps(source.known_at)
        if tuple(observed.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d')) != dates:
            raise ValueError('Source observation clocks must agree with their local dates')
        close = source.close.to_numpy(dtype=float)
        with np.errstate(over='ignore', invalid='ignore'):
            mirror = close.astype('<f4').astype(float)
        if np.isinf(mirror).any():
            raise ValueError('Source price cannot be represented by the tested float32 store')
        def vector(expression):
            result = np.asarray(evaluate(code, expression, dates))
            if result.shape != (len(dates),) or result.dtype.kind not in 'fiu':
                raise ValueError('Backend must return one real numeric value per calendar row')
            result = result.astype(float, copy=True)
            if np.isinf(result).any():
                raise ValueError('Infinite backend values are rejected')
            return result
        backend_close = vector('$close')
        if not np.array_equal(backend_close, mirror, equal_nan=True):
            raise ValueError('Backend close does not match the declared source mirror')
        out = source[['trade_date', 'stock_code']].copy()
        why = out.copy()
        valid = np.isfinite(mirror)
        obs_ns = observed.astype('datetime64[ns, UTC]').astype('int64').to_numpy()
        known_ns = known.astype('datetime64[ns, UTC]').astype('int64').to_numpy()
        # Only availability propagation is done locally; formula values stay external.
        out['observed_end'] = pd.to_datetime([max(obs_ns[max(0, i-max_window+1):i+1]) for i in range(len(dates))], utc=True)
        out['known_at'] = pd.to_datetime([max(known_ns[max(0, i-max_window+1):i+1]) for i in range(len(dates))], utc=True)
        for name in names:
            _, expression, window, _, _ = allowed[name]
            raw = backend_close.copy() if name == 'close' else vector(expression)
            reasons = np.full(len(dates), M.PRESENT.value, dtype=object)
            for i in range(len(dates)):
                if i+1 < window:
                    reasons[i] = M.HISTORY.value
                elif not valid[i-window+1:i+1].all():
                    reasons[i] = M.INCOMPLETE.value
                elif not np.isfinite(raw[i]):
                    reasons[i] = M.UNDEFINED.value
            raw[reasons != M.PRESENT.value] = np.nan
            out[name] = raw
            why[name] = reasons
        output_values.append(out)
        output_missing.append(why)
    result_values = pd.concat(output_values, ignore_index=True).sort_values(['trade_date', 'stock_code']).reset_index(drop=True)
    result_missing = pd.concat(output_missing, ignore_index=True).sort_values(['trade_date', 'stock_code']).reset_index(drop=True)
    return FeatureBlock(result_values, result_missing, units, {
        'schema': 'qlib-selected-expressions-v1', 'key_columns': ['trade_date', 'stock_code'],
        'source_id': quote_block.metadata['source_id'], 'version_id': quote_block.metadata['version_id'],
        'vintage': quote_block.metadata['vintage'], 'frequency': 'daily',
        'backend_version': backend_version, 'backend_wheel_sha256': backend_wheel_sha256,
        'formulas': {n: allowed[n][1] for n in names}, 'storage': 'float32 source mirror',
        'adjustment': 'unadjusted; corporate-action gaps may distort these features',
        'timing': 'max source observed_end/known_at over largest selected backward dependency window; decision layer verifies cutoff',
        'missing': 'strict full-window warmup; invalid source window incomplete; remaining backend NaN undefined; never zero filled',
        'calendar': 'complete caller-declared grid; missing calendar rows reject, not inferred exchange completeness',
        'backend_binding': 'caller verifies provider/store/wheel; matching close and fixed expressions do not certify execution',
        'historical_execution_certified': False, 'independent_reproduction_claimed': False,
    }).validate()
