"""Causal counterexamples for availability, units and target maturity (no market fit)."""
from datetime import date, datetime
import json

import pytest

from src.alpharesearch.contracts import (
    AvailabilityBasis, FeatureValue, LabelPurpose, LabelWindow, LearningTiming,
    MissingReason, SourceTiming, Unit, VintagePolicy, known_at_from_publication,
    mature_mask, timestamp,
)


def source(known='2024-01-05T15:05:00+08:00', vintage=VintagePolicy.MARKET):
    return SourceTiming('daily_bars', 'source-v1', '2024-01-05T15:00:00+08:00',
                        known, AvailabilityBasis.MARKET_RULE, vintage,
                        retrieved_at='2026-10-06T18:00:00+08:00')


def week():
    return LearningTiming('2024-01-05T15:10:00+08:00', '2024-01-08T09:30:00+08:00',
        LabelWindow('2024-01-08T09:30:00+08:00', '2024-01-12T15:00:00+08:00',
                    '2024-01-12T15:01:00+08:00'))


def test_clock_normalization_does_not_depend_on_host_timezone():
    assert timestamp('2024-01-05T15:10:00+08:00') == timestamp('2024-01-05T07:10:00Z')


@pytest.mark.parametrize('value', ['2024-01-05', '2024-01-05T15:00:00', datetime(2024, 1, 5), None])
def test_timezone_cannot_be_silently_inferred(value):
    with pytest.raises(ValueError):
        timestamp(value)


def test_same_date_announcement_is_not_assumed_known_before_close():
    known = known_at_from_publication('2024-01-05', 'date')
    assert known == timestamp('2024-01-06T00:00:00+08:00')
    annual = SourceTiming('finance', 'original-v1', '2023-12-31T23:59:59+08:00',
                         known, AvailabilityBasis.DATE_ONLY, VintagePolicy.HISTORICAL)
    with pytest.raises(ValueError, match='not known'):
        annual.require_available('2024-01-05T15:10:00+08:00')
    annual.require_available('2024-01-08T09:00:00+08:00')


def test_conservative_date_rule_is_not_a_next_trading_day_rule():
    assert known_at_from_publication(date(2024, 1, 5), 'date').weekday() == 4  # UTC Friday
    assert known_at_from_publication('2024-03-10', 'date', 'America/New_York') == timestamp('2024-03-11T04:00:00Z')


def test_precise_evening_announcement_keeps_actual_clock():
    assert known_at_from_publication('2024-01-05T18:37:00+08:00', 'timestamp') == timestamp('2024-01-05T10:37:00Z')


def test_unknown_availability_stays_unknown():
    unknown = SourceTiming('event', 'v1', '2024-01-05T15:00:00+08:00',
                           known_at_from_publication(None, 'unknown'),
                           AvailabilityBasis.UNKNOWN, VintagePolicy.UNKNOWN)
    with pytest.raises(ValueError, match='not known'):
        unknown.require_available('2026-10-06T12:00:00+08:00', allow_weak_vintage=True)


def test_download_date_is_provenance_not_historical_publication():
    row = source()
    week().require_inputs([row])  # Retrieved years later does not itself change public bar time.
    assert row.retrieved_at > week().decision_at


def test_future_bar_cannot_be_a_feature_for_last_friday():
    future = SourceTiming('daily_bars', 'v1', '2024-01-08T15:00:00+08:00',
                          '2024-01-08T15:05:00+08:00', AvailabilityBasis.MARKET_RULE,
                          VintagePolicy.MARKET)
    with pytest.raises(ValueError, match='not known'):
        week().require_inputs([future])


def test_original_announcement_date_does_not_restore_latest_financial_vintage():
    latest = source(vintage=VintagePolicy.LATEST_ONLY)
    with pytest.raises(ValueError, match='vintage'):
        week().require_inputs([latest])
    week().require_inputs([latest], allow_weak_vintage=True)
    assert latest.to_dict()['historical_vintage_supported'] is False


def test_known_at_before_observation_end_is_rejected():
    with pytest.raises(ValueError, match='observation'):
        source(known='2024-01-05T14:59:00+08:00')


def test_past_friday_to_monday_open_label_cannot_be_account_target():
    label = LabelWindow('2024-01-05T15:00:00+08:00', '2024-01-08T09:30:00+08:00',
                        '2024-01-08T09:31:00+08:00')
    with pytest.raises(ValueError, match='pre-entry'):
        LearningTiming(week().decision_at, week().execution_at, label)


def test_forecast_label_must_be_explicit_and_stays_distinct_from_account_window():
    label = LabelWindow('2024-01-05T15:00:00+08:00', '2024-01-08T09:30:00+08:00',
                        '2024-01-08T09:31:00+08:00', LabelPurpose.FORECAST)
    row = LearningTiming(week().decision_at, week().execution_at, label)
    assert row.to_dict()['label_purpose'] == 'forecast_only'


def test_training_cutoff_uses_outcome_known_time_not_only_feature_date():
    row = week()
    assert mature_mask([row], '2024-01-05T16:00:00+08:00') == [False]
    assert mature_mask([row], '2024-01-12T15:00:00+08:00') == [False]
    assert mature_mask([row], '2024-01-12T15:01:00+08:00') == [True]


def test_maturity_iterator_does_not_get_consumed_by_validation():
    assert mature_mask((row for row in [week()]), '2024-01-12T15:01:00+08:00') == [True]


@pytest.mark.parametrize('reason', [MissingReason.NO_EVENT, MissingReason.UNCOVERED,
    MissingReason.INVALID, MissingReason.INCOMPLETE, MissingReason.NOT_KNOWN])
def test_missing_reasons_survive_serialization_and_never_turn_into_zero(reason):
    missing = FeatureValue(None, Unit.AMOUNT, reason)
    observed_zero = FeatureValue(0, Unit.AMOUNT)
    assert missing.to_dict() != observed_zero.to_dict()
    assert json.loads(json.dumps(missing.to_dict()))['missing_reason'] == reason.value
    with pytest.raises(ValueError, match='Missingness'):
        FeatureValue(0, Unit.AMOUNT, reason)


@pytest.mark.parametrize('value,unit', [(float('nan'), Unit.RATIO), (float('inf'), Unit.AMOUNT),
    (True, Unit.AMOUNT), (0, Unit.PRICE), (-1, Unit.COUNT), (1.5, Unit.COUNT), (2, Unit.BINARY)])
def test_invalid_numeric_observations_do_not_silently_enter_model(value, unit):
    with pytest.raises(ValueError):
        FeatureValue(value, unit)


def test_unit_and_time_provenance_roundtrip_is_explicit():
    assert FeatureValue(-10, Unit.AMOUNT).to_dict()['unit'] == 'cny'
    assert FeatureValue(True, Unit.BINARY).value == 1
    assert source().to_dict()['basis'] == 'declared_market_availability_rule'
    assert week().to_dict()['label_start'] == '2024-01-08T01:30:00+00:00'

def test_past_window_without_observed_outcome_never_matures_by_clock_alone():
    label = LabelWindow('2024-01-08T09:30:00+08:00', '2024-01-12T15:00:00+08:00',
                        None, observed=False)
    row = LearningTiming(week().decision_at, week().execution_at, label)
    assert row.mature_by('2026-10-06T15:00:00+08:00') is False
    assert row.to_dict()['label_known_at'] is None


def test_anticipated_end_is_not_a_known_timestamp_for_missing_outcome():
    with pytest.raises(ValueError, match='Unobserved'):
        LabelWindow('2024-01-08T09:30:00+08:00', '2024-01-12T15:00:00+08:00',
                    '2024-01-12T15:01:00+08:00', observed=False)
