"""Shared causal contracts for extensible feature/model research (E01).

These checks validate declared timing and provenance. They cannot establish that a
vendor supplied the true original publication or unrevised historical version.
"""
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
import math
from zoneinfo import ZoneInfo


class Unit(str, Enum):
    PRICE = 'cny_per_share'
    AMOUNT = 'cny'
    SHARES = 'shares'
    RATIO = 'ratio'
    LOG_RETURN = 'log_return'
    COUNT = 'count'
    SECONDS = 'seconds'
    BINARY = 'binary'


class MissingReason(str, Enum):
    PRESENT = 'present'
    NO_EVENT = 'verified_no_event'
    UNCOVERED = 'source_uncovered'
    INVALID = 'invalid_record'
    UNDEFINED = 'undefined_value'
    INCOMPLETE = 'incomplete_window'
    NOT_KNOWN = 'not_yet_known'
    HISTORY = 'insufficient_history'
    STALE = 'stale_observation'
    STATUS = 'unknown_status'


class AvailabilityBasis(str, Enum):
    EXACT = 'exact_publication_timestamp'
    DATE_ONLY = 'conservative_date_only'
    MARKET_RULE = 'declared_market_availability_rule'
    UNKNOWN = 'unknown'


class VintagePolicy(str, Enum):
    HISTORICAL = 'historical_versions'
    MARKET = 'market_observation'
    LATEST_ONLY = 'latest_snapshot_only'
    UNKNOWN = 'unknown'


class LabelPurpose(str, Enum):
    EXECUTION = 'execution_window'
    FORECAST = 'forecast_only'


def timestamp(value):
    """Require an explicit timezone; normalize to UTC without using the host clock."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Timestamp must include an explicit timezone')
    return value.astimezone(timezone.utc)


def timestamps(values):
    """Vector fast path for already-aware pandas clocks; objects keep strict checks.

    No timezone is inferred for naive values. NaT is rejected on both paths.
    """
    import pandas as pd
    if isinstance(values.dtype,pd.DatetimeTZDtype):
        if values.isna().any():raise ValueError('Timestamp cannot be missing')
        return values.dt.tz_convert('UTC')
    return pd.to_datetime(values.map(timestamp),utc=True)


def known_at_from_publication(value, precision, market_timezone='Asia/Shanghai'):
    """Date-only publication is usable after that complete local civil date.

    This is a conservative research rule, not an asserted original announcement
    timestamp or a guarantee of live ingestion. Actual sessions are handled by the
    calendar/panel layer; the next civil date is not assumed to be a trading day.
    """
    if precision == 'timestamp':
        return timestamp(value)
    if precision == 'unknown' and value is None:
        return None
    if precision != 'date':
        raise ValueError('Unknown publication precision or inconsistent value')
    if isinstance(value, datetime):
        raise ValueError('Date-only input must not silently truncate a timestamp')
    if isinstance(value, str):
        original = value
        value = date.fromisoformat(value)
        if value.isoformat() != original:
            raise ValueError('Publication date must be canonical YYYY-MM-DD')
    if not isinstance(value, date):
        raise ValueError('Publication date is missing or invalid')
    local_midnight = datetime.combine(value + timedelta(days=1), time.min,
                                      tzinfo=ZoneInfo(market_timezone))
    return local_midnight.astimezone(timezone.utc)


def required_id(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 500:
        raise ValueError(name + ' must be a nonempty provenance identifier')
    return value


@dataclass(frozen=True)
class SourceTiming:
    source_id: str
    version_id: str
    observed_end: datetime
    known_at: datetime | None
    basis: AvailabilityBasis
    vintage: VintagePolicy
    retrieved_at: datetime | None = None

    def __post_init__(self):
        required_id(self.source_id, 'source_id')
        required_id(self.version_id, 'version_id')
        object.__setattr__(self, 'basis', AvailabilityBasis(self.basis))
        object.__setattr__(self, 'vintage', VintagePolicy(self.vintage))
        object.__setattr__(self, 'observed_end', timestamp(self.observed_end))
        if self.known_at is not None:
            object.__setattr__(self, 'known_at', timestamp(self.known_at))
        if self.retrieved_at is not None:
            object.__setattr__(self, 'retrieved_at', timestamp(self.retrieved_at))
        if (self.basis == AvailabilityBasis.UNKNOWN) != (self.known_at is None):
            raise ValueError('Unknown availability must remain explicitly unknown')
        if self.known_at is not None and self.known_at < self.observed_end:
            raise ValueError('Information cannot be known before its observation ends')

    @property
    def historical_vintage_supported(self):
        return self.vintage in {VintagePolicy.HISTORICAL, VintagePolicy.MARKET}

    def require_available(self, decision_at, *, allow_weak_vintage=False):
        decision_at = timestamp(decision_at)
        if self.known_at is None or self.known_at > decision_at:
            raise ValueError('Input was not known by the decision cutoff')
        if not self.historical_vintage_supported and not allow_weak_vintage:
            raise ValueError('Latest-only/unknown vintage cannot claim strict historical availability')
        return self

    def to_dict(self):
        return dict(source_id=self.source_id, version_id=self.version_id,
                    observed_end=self.observed_end.isoformat(),
                    known_at=self.known_at.isoformat() if self.known_at else None,
                    basis=self.basis.value, vintage=self.vintage.value,
                    retrieved_at=self.retrieved_at.isoformat() if self.retrieved_at else None,
                    historical_vintage_supported=self.historical_vintage_supported)


@dataclass(frozen=True)
class LabelWindow:
    start: datetime
    end: datetime
    known_at: datetime | None
    purpose: LabelPurpose = LabelPurpose.EXECUTION
    observed: bool = True

    def __post_init__(self):
        for name in ['start', 'end']:
            object.__setattr__(self, name, timestamp(getattr(self, name)))
        if self.known_at is not None:
            object.__setattr__(self, 'known_at', timestamp(self.known_at))
        object.__setattr__(self, 'purpose', LabelPurpose(self.purpose))
        if type(self.observed) is not bool or self.observed != (self.known_at is not None):
            raise ValueError('Unobserved outcome must not claim a known timestamp')
        if self.end < self.start or (self.known_at is not None and self.known_at < self.end):
            raise ValueError('Label window or outcome availability is inconsistent')


@dataclass(frozen=True)
class LearningTiming:
    decision_at: datetime
    execution_at: datetime
    label: LabelWindow

    def __post_init__(self):
        object.__setattr__(self, 'decision_at', timestamp(self.decision_at))
        object.__setattr__(self, 'execution_at', timestamp(self.execution_at))
        if not isinstance(self.label, LabelWindow):
            raise ValueError('Learning timing needs an explicit LabelWindow')
        if self.execution_at <= self.decision_at:
            raise ValueError('Execution must follow the information decision cutoff')
        if self.label.end <= self.decision_at:
            raise ValueError('A prediction outcome must end after the decision')
        if self.label.purpose == LabelPurpose.EXECUTION and self.label.start < self.execution_at:
            raise ValueError('An execution-window target cannot include a pre-entry return')

    def require_inputs(self, inputs, *, allow_weak_vintage=False):
        for source in inputs:
            if not isinstance(source, SourceTiming):
                raise ValueError('Each input needs a SourceTiming contract')
            source.require_available(self.decision_at, allow_weak_vintage=allow_weak_vintage)
        return self

    def mature_by(self, training_cutoff):
        cutoff = timestamp(training_cutoff)
        return self.label.observed and self.label.known_at <= cutoff

    def to_dict(self):
        return dict(decision_at=self.decision_at.isoformat(), execution_at=self.execution_at.isoformat(),
                    label_start=self.label.start.isoformat(), label_end=self.label.end.isoformat(),
                    label_known_at=self.label.known_at.isoformat() if self.label.known_at else None,
                    label_observed=self.label.observed, label_purpose=self.label.purpose.value)


def mature_mask(timings, training_cutoff):
    timings = list(timings)
    cutoff = timestamp(training_cutoff)
    if any(not isinstance(row, LearningTiming) for row in timings):
        raise ValueError('Maturity selection requires LearningTiming rows')
    return [row.mature_by(cutoff) for row in timings]


@dataclass(frozen=True)
class FeatureValue:
    value: float | None
    unit: Unit
    missing_reason: MissingReason = MissingReason.PRESENT

    def __post_init__(self):
        object.__setattr__(self, 'unit', Unit(self.unit))
        object.__setattr__(self, 'missing_reason', MissingReason(self.missing_reason))
        if self.value is None:
            if self.missing_reason == MissingReason.PRESENT:
                raise ValueError('Missing numeric value needs an explicit reason')
            return
        if self.missing_reason != MissingReason.PRESENT:
            raise ValueError('Missingness cannot silently become an observed numeric value')
        if isinstance(self.value, bool) and self.unit != Unit.BINARY:
            raise ValueError('Boolean cannot silently become a financial amount or return')
        if not isinstance(self.value, (int, float)) or not math.isfinite(self.value):
            raise ValueError('Feature must be finite or explicitly missing')
        if self.unit == Unit.PRICE and self.value <= 0:
            raise ValueError('Observed raw price must be positive')
        if self.unit == Unit.COUNT and (self.value < 0 or int(self.value) != self.value):
            raise ValueError('Count must be a nonnegative integer')
        if self.unit == Unit.BINARY and self.value not in {0, 1}:
            raise ValueError('Binary feature must be zero or one')
        object.__setattr__(self, 'value', float(self.value))

    def to_dict(self):
        return dict(value=self.value, unit=self.unit.value, missing_reason=self.missing_reason.value)
