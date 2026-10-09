"""Independent hand examples and causal counterfactuals; no real data/fit/account."""
import numpy as np
import pandas as pd
import pytest

from src.alpharesearch.contracts import MissingReason as M, Unit, VintagePolicy
from src.alpharesearch.features.base import FeatureBlock

from src.alpharesearch import sequence as seq
CALENDAR = ["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
A, B = "600001.SH", "000001.SZ"


def fixture(*, extra=(), vintage=VintagePolicy.MARKET.value):
    raw = [
        (A, "2023-01-02", 10., .1, "2023-01-02T15:01:00+08:00"),
        (A, "2023-01-04", 12., .3, "2023-01-05T08:00:00+08:00"),
        (A, "2023-01-05", 13., np.nan, "2023-01-05T15:01:00+08:00"),
        (B, "2023-01-02", 20., .2, "2023-01-02T15:01:00+08:00"),
        (B, "2023-01-03", 21., .3, "2023-01-03T15:01:00+08:00"),
        (A, "2023-01-06", 999., .9, "2023-01-06T15:01:00+08:00"),
    ] + list(extra)
    values = pd.DataFrame(raw, columns=["stock_code", "trade_date", "price", "pulse", "known_at"])
    values["observed_end"] = values.trade_date+"T15:00:00+08:00"
    missing = values[["stock_code", "trade_date"]].copy()
    missing["price"] = M.PRESENT.value
    missing["pulse"] = np.where(values.pulse.isna(), M.UNDEFINED.value, M.PRESENT.value)
    return FeatureBlock(values, missing, {"price": Unit.PRICE.value, "pulse": Unit.RATIO.value},
        dict(key_columns=["stock_code", "trade_date"], source_id="synthetic-feed", version_id="synthetic-v1", vintage=vintage, frequency="daily"))


def samples():
    return pd.DataFrame([(day, stock, day+"T15:30:00+08:00") for day, stock in [
        ("2023-01-04", A), ("2023-01-05", A), ("2023-01-04", B), ("2023-01-02", A)]], columns=seq.SAMPLE_KEYS)


def dataset(block=None, membership=None, calendar=None, features=("price", "pulse"), **kwargs):
    return seq.CausalWindowDataset(fixture() if block is None else block,
        samples() if membership is None else membership, CALENDAR if calendar is None else calendar,
        features, spec=seq.WindowSpec(sessions=3, **kwargs))


def test_hand_spelled_windows_preserve_holes_clocks_stock_and_feature_order():
    batch = dataset().batch([0, 1, 2, 3])
    expected = np.array([
        [[10., .1], [np.nan, np.nan], [np.nan, np.nan]],
        [[np.nan, np.nan], [12., .3], [13., np.nan]],
        [[20., .2], [21., .3], [np.nan, np.nan]],
        [[np.nan, np.nan], [np.nan, np.nan], [10., .1]],
    ])
    np.testing.assert_equal(batch.values, expected)
    np.testing.assert_equal(batch.observed, np.isfinite(expected))
    assert batch.dates.tolist() == [CALENDAR[:3], CALENDAR[1:4], CALENDAR[:3], [None, None, CALENDAR[0]]]
    codes = seq.REASON_CODES
    np.testing.assert_equal(batch.reason_codes[0], [[codes[M.PRESENT.value]]*2, [codes[M.UNCOVERED.value]]*2, [codes[M.UNCOVERED.value]]*2])
    assert batch.reason_codes[1, 2, 1] == codes[M.UNDEFINED.value]
    assert (batch.reason_codes[3, :2] == codes[M.HISTORY.value]).all()
    assert batch.features == ("price", "pulse")


def test_future_append_including_old_dated_late_publication_preserves_every_old_input():
    before = dataset().batch([0, 1, 2, 3])
    block = fixture(extra=[(B, "2023-01-04", 888., .99, "2023-01-09T15:01:00+08:00"),
                           ("600002.SH", "2023-01-03", 77., .8, "2023-01-06T15:01:00+08:00")])
    after = dataset(block, calendar=CALENDAR+["2023-01-09"]).batch([0, 1, 2, 3])
    for key in ("values", "observed", "reason_codes", "dates"):
        np.testing.assert_equal(getattr(before, key), getattr(after, key))
    pd.testing.assert_frame_equal(before.samples, after.samples)


def test_poisoning_unavailable_values_cannot_change_past_window():
    before = dataset().batch([0])
    block = fixture();block.values.loc[block.values.stock_code.eq(A) & block.values.trade_date.eq("2023-01-04"), "price"] = 999999.
    after = dataset(block).batch([0])
    np.testing.assert_equal(before.values, after.values)
    np.testing.assert_equal(before.reason_codes, after.reason_codes)


def test_owns_source_and_membership_against_caller_mutation():
    block, membership = fixture(), samples();store = dataset(block, membership);before = store.batch([0])
    block.values.loc[0, "price"] = 10000.;membership.loc[0, "stock_code"] = B
    block.values.loc[0, "stock_code"] = "600099.SH"
    block.values.loc[0, "trade_date"] = "2023-01-06"
    block.values.loc[0, "known_at"] = "2026-01-01T15:30:00+08:00"
    block.missing.loc[0, "price"] = M.UNDEFINED.value
    membership.loc[0, "decision_at"] = "2026-01-01T15:30:00+08:00"
    after = store.batch([0]);np.testing.assert_equal(before.values, after.values)
    pd.testing.assert_frame_equal(before.samples, after.samples)


def test_requested_sample_and_feature_axes_are_kept():
    batch = dataset(features=("pulse", "price")).batch([2, 0])
    assert batch.samples.stock_code.tolist() == [B, A]
    np.testing.assert_equal(batch.values[0, 0], [.2, 20.])
    np.testing.assert_equal(batch.values[1, 0], [.1, 10.])


@pytest.mark.parametrize("corruption", ["calendar_duplicate", "source_duplicate", "target_in_samples", "target_in_source", "naive_clock", "wrong_decision_date", "unknown_feature", "weak_vintage"])
def test_invalid_contracts_are_rejected(corruption):
    block, membership, calendar, features = fixture(), samples(), list(CALENDAR), ["price", "pulse"]
    if corruption == "calendar_duplicate":calendar.append(calendar[-1])
    elif corruption == "source_duplicate":
        block.values = pd.concat([block.values, block.values.iloc[[0]]], ignore_index=True)
        block.missing = pd.concat([block.missing, block.missing.iloc[[0]]], ignore_index=True)
    elif corruption == "target_in_samples":membership["future_return"] = 99.
    elif corruption == "target_in_source":block.values["future_return"] = 99.
    elif corruption == "naive_clock":membership.loc[0, "decision_at"] = "2023-01-04T15:30:00"
    elif corruption == "wrong_decision_date":membership.loc[0, "decision_at"] = "2023-01-05T15:30:00+08:00"
    elif corruption == "unknown_feature":features.append("future_return")
    elif corruption == "weak_vintage":block.metadata["vintage"] = VintagePolicy.LATEST_ONLY.value
    with pytest.raises(ValueError):dataset(block, membership, calendar, features)


def test_weak_vintage_allowance_is_explicit_and_preserved():
    block = fixture(vintage=VintagePolicy.LATEST_ONLY.value)
    store = seq.CausalWindowDataset(block, samples(), CALENDAR, ["price"], spec=seq.WindowSpec(sessions=3), allow_weak_vintage=True)
    assert store.batch([0]).vintage == VintagePolicy.LATEST_ONLY.value


def test_batch_and_source_budgets_reject():
    with pytest.raises(ValueError, match="Source partition"):
        dataset(max_source_rows=5)
    store = dataset(max_batch_cells=5)
    with pytest.raises(ValueError, match="Window batch"):
        store.batch([0])
    store = dataset(max_batch_samples=1)
    with pytest.raises(ValueError, match="Window batch"):
        store.batch([0, 1])


@pytest.mark.parametrize("indices", [[], [True], [-1], [4], [1.0]])
def test_invalid_indices_do_not_silently_choose_a_sample(indices):
    with pytest.raises(ValueError):dataset().batch(indices)


@pytest.mark.parametrize("known_unit,end_unit,decision_unit", [("s", "us", "ns"), ("us", "ns", "s"), ("ns", "s", "ms"), ("ms", "ns", "us")])
def test_mixed_timestamp_units_have_same_past_values_and_future_append_invariance(known_unit, end_unit, decision_unit):
    baseline = dataset().batch([0, 1, 2, 3])
    membership = samples()
    membership["decision_at"] = pd.to_datetime(membership.decision_at, utc=True).astype("datetime64["+decision_unit+", UTC]")
    def typed(block):
        block.values["known_at"] = pd.to_datetime(block.values.known_at, utc=True).astype("datetime64["+known_unit+", UTC]")
        block.values["observed_end"] = pd.to_datetime(block.values.observed_end, utc=True).astype("datetime64["+end_unit+", UTC]")
        return block
    before = dataset(typed(fixture()), membership).batch([0, 1, 2, 3])
    after = dataset(typed(fixture(extra=[(B, "2023-01-04", 888., .99, "2023-01-09T15:01:00+08:00")])), membership).batch([0, 1, 2, 3])
    for key in ("values", "observed", "reason_codes", "dates"):
        np.testing.assert_equal(getattr(baseline, key), getattr(before, key))
        np.testing.assert_equal(getattr(before, key), getattr(after, key))


def test_before_close_does_not_use_end_of_day_observation():
    membership = samples().iloc[[3]].copy()
    membership["decision_at"] = "2023-01-02T14:59:00+08:00"
    batch = dataset(membership=membership).batch([0])
    assert np.isnan(batch.values).all() and not batch.observed.any()
    assert (batch.reason_codes[0, -1] == seq.REASON_CODES[M.UNCOVERED.value]).all()


@pytest.mark.parametrize("sample_limit,cell_limit,maximum_consumed", [(2, 1000000, 3), (512, 11, 2)])
def test_index_iterator_consumption_stops_before_budget_allocation(sample_limit, cell_limit, maximum_consumed):
    count = 0
    def unlimited():
        nonlocal count
        while True:
            count += 1
            if count > maximum_consumed:
                raise AssertionError("Iterator consumed beyond admission allowance")
            yield 0
    store = dataset(max_batch_samples=sample_limit, max_batch_cells=cell_limit)
    with pytest.raises(ValueError, match="Window batch"):
        store.batch(unlimited())
    assert count == maximum_consumed


@pytest.mark.parametrize("kind", ["object_na", "nullable_float", "nullable_integer"])
def test_contract_valid_missing_numeric_dtypes_keep_hand_values_and_reasons(kind):
    block = fixture()
    expected = np.array([[[np.nan, np.nan], [12., .3], [13., np.nan]]])
    if kind == "object_na":
        block.values["pulse"] = block.values.pulse.astype(object)
        block.values.loc[2, "pulse"] = pd.NA
    elif kind == "nullable_float":
        block.values["pulse"] = block.values.pulse.astype("Float64")
    else:
        block.values["price"] = block.values.price.astype("Int64")
        block.values.loc[2, "price"] = pd.NA
        block.missing.loc[2, "price"] = M.UNDEFINED.value
        expected[0, 2, 0] = np.nan
    batch = dataset(block).batch([1])
    np.testing.assert_equal(batch.values, expected)
    np.testing.assert_equal(batch.observed, np.isfinite(expected))
    assert batch.reason_codes[0, 2, 1] == seq.REASON_CODES[M.UNDEFINED.value]
    if kind == "nullable_integer":
        assert batch.reason_codes[0, 2, 0] == seq.REASON_CODES[M.UNDEFINED.value]


@pytest.mark.parametrize("delay,cutoff,observed_last", [
    (60, "2023-01-05T15:30:00+08:00", True),
    (1800, "2023-01-05T15:01:00+08:00", False),
])
def test_formal_daily_primitives_pipeline_respects_calendar_and_availability(delay, cutoff, observed_last):
    from src.alpharesearch.features.primitives import daily_primitives
    # Jan 3 has no quote; it must stay an empty calendar slot rather than
    # pulling Jan 2 in its place. Jan 5 close is only available after delay.
    bars = pd.DataFrame([
        ["2023-01-02", A, 10., 11., 9., 10.5, 9.5, 100., 1000., 12., 8., True],
        ["2023-01-04", A, 12., 13., 11., 12.5, 10.5, 200., 2000., 14., 10., True],
        ["2023-01-05", A, 13., 14., 12., 13.5, 12.5, 300., 3000., 15., 11., True],
    ], columns=["trade_date", "stock_code", "open", "high", "low", "close",
                "pre_close", "volume", "amount", "high_limit", "low_limit", "limit_price_valid"])
    block = daily_primitives(bars, source_id="hand-primitive-feed", version_id="v1",
                             available_delay_seconds=delay)
    membership = pd.DataFrame([["2023-01-05", A, cutoff]], columns=seq.SAMPLE_KEYS)
    batch = seq.CausalWindowDataset(block, membership, CALENDAR, ("close", "volume"),
                                   spec=seq.WindowSpec(sessions=3)).batch([0])
    assert batch.dates.tolist() == [["2023-01-03", "2023-01-04", "2023-01-05"]]
    expected = [[np.nan, np.nan], [12.5, 200.],
                [13.5, 300.] if observed_last else [np.nan, np.nan]]
    np.testing.assert_equal(batch.values[0], expected)
    np.testing.assert_equal(batch.observed[0], np.isfinite(expected))
    assert (batch.reason_codes[0, 0] == seq.REASON_CODES[M.UNCOVERED.value]).all()
    assert batch.source_id == "hand-primitive-feed" and batch.version_id == "v1"
