"""Decades of pre-snapshot calendar must not poison a finite market proxy."""
import numpy as np
import pandas as pd
import pytest

from src.mlresearch.financial_study import market_calendar_scope, market_features, MKT
from src.mlresearch.financial_audit import audit_market_calendar_scope, market_reference


@pytest.fixture
def inputs():
    days = pd.bdate_range('2019-01-02', periods=85).strftime('%Y-%m-%d').tolist()
    older = pd.bdate_range('1990-01-01', '2018-12-31').strftime('%Y-%m-%d').tolist()
    calendar = older + days
    bars = pd.DataFrame(dict(stock_code='X', trade_date=days, close=10 * np.exp(np.arange(85) / 10000), pre_close=10., volume=1., is_st=0))
    meta = pd.DataFrame(dict(stock_code=['X'], listed_date=['1980-01-01'], de_listed_date=['9999-12-31']))
    return calendar, days, bars, meta


def test_long_calendar_scope_matches_independent_reference_and_keeps_original(inputs):
    calendar, days, bars, meta = inputs
    original = calendar.copy()
    scope = market_calendar_scope(calendar, days[0], days[-1], days[79], days[59:80])
    oracle_scope = audit_market_calendar_scope(calendar, days[0], days[-1], days[79], days[59:80])
    assert scope == oracle_scope == days[:80]
    actual = market_features(bars, meta, scope)
    oracle = market_reference(bars, meta, oracle_scope)
    np.testing.assert_allclose(actual[MKT], oracle[MKT], atol=1e-12, equal_nan=True)
    assert np.isfinite(actual.loc[59:, MKT]).all().all()
    assert actual.members.eq(1).all()
    assert calendar == original
    prefix = market_features(bars[bars.trade_date.le(days[68])], meta, scope[:69])
    np.testing.assert_allclose(actual.loc[:68, MKT], prefix[MKT], atol=1e-12, equal_nan=True)


@pytest.mark.parametrize('scope', [market_calendar_scope, audit_market_calendar_scope])
def test_short_lookback_is_rejected_even_when_old_calendar_has_many_days(inputs, scope):
    calendar, days, _, _ = inputs
    with pytest.raises(ValueError, match='60 covered sessions'):
        scope(calendar, days[0], days[-1], days[-1], [days[58]])


@pytest.mark.parametrize('proxy', [market_features, market_reference])
def test_missing_day_inside_coverage_is_rejected_instead_of_filled_with_zero(inputs, proxy):
    _, days, bars, meta = inputs
    with pytest.raises(ValueError, match='missing'):
        proxy(bars[bars.trade_date.ne(days[37])], meta, days)


@pytest.mark.parametrize('scope', [market_calendar_scope, audit_market_calendar_scope])
@pytest.mark.parametrize('invalid', ['unknown_decision', 'after_study', 'duplicate_calendar'])
def test_unknown_outside_and_duplicate_calendar_rejected(inputs, scope, invalid):
    calendar, days, _, _ = inputs
    decisions = [days[79]]
    if invalid == 'unknown_decision': decisions = ['2019-01-01']
    if invalid == 'after_study': decisions = [days[84]]
    if invalid == 'duplicate_calendar': calendar = calendar + [days[-1]]
    with pytest.raises(ValueError):
        scope(calendar, days[0], days[-1], days[79], decisions)
