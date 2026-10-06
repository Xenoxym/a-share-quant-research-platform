"""Hand calculations and causal board/streak counterexamples for E04."""

import numpy as np
import pandas as pd
import pytest

from src.alpharesearch.contracts import FeatureValue, MissingReason, Unit
from src.alpharesearch.features.boards import price_board_features

CAL = ['2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-08']


def bars(rows=None):
    if rows is None:rows = [dict(open=10., high=12., low=9., close=11.)]
    return pd.DataFrame([dict(trade_date=CAL[i], stock_code='000001.SZ', pre_close=10., volume=1000,
                               high_limit=12., low_limit=8., limit_price_valid=True,
                               final_streak_using_future=99, future_label=.99, **row)
                         for i, row in enumerate(rows)])


def block(b=None, **kwargs):
    return price_board_features(bars() if b is None else b, CAL,
                                 source_id='declared_daily_source', version_id='frozen_source_hash', **kwargs)


def test_hand_calculated_wick_body_gap_and_band_features():
    x = block();v=x.values.iloc[0]
    assert v.upper_wick_fraction == pytest.approx(1/3)
    assert v.lower_wick_fraction == pytest.approx(1/3)
    assert v.absolute_body_fraction == pytest.approx(1/3)
    assert v.body_direction == 1 and v.overnight_gap_ratio == 0
    assert v.upper_limit_distance_close == pytest.approx(.1)
    assert v.lower_limit_distance_close == pytest.approx(.3)
    assert v.limit_span_ratio == pytest.approx(.4)
    assert len(x.units) == 29
    assert not {'final_streak_using_future','future_label','high_limit','is_st'} & set(x.values)
    assert np.isclose(v.upper_wick_fraction+v.lower_wick_fraction+v.absolute_body_fraction, 1)


def test_touch_and_close_limit_are_distinct_and_no_intraday_order_inferred():
    x=block();v=x.values.iloc[0]
    assert v.touched_upper_limit == 1 and v.close_upper_limit == 0 and v.upper_touch_close_inside == 1
    b=bars([dict(open=12.,high=12.,low=8.,close=8.)]);v=block(b).values.iloc[0]
    assert v.hit_both_limits == 1 and v.upper_open_lower_close == 1
    assert not block(b).metadata['intraday_order_supported']
    assert not any('first_touch' in c or 'seal_duration' in c for c in block(b).values)


def test_true_one_price_board_has_undefined_candle_fractions_not_fake_zero():
    b=bars([dict(open=12.,high=12.,low=12.,close=12.)]);x=block(b);v=x.values.iloc[0]
    assert v.one_price_upper_limit == 1 and v.zero_range_bar == 1
    assert pd.isna(v.upper_wick_fraction) and pd.isna(v.absolute_body_fraction)
    assert x.missing.upper_wick_fraction.iloc[0] == 'undefined_value'
    assert FeatureValue(None,Unit.RATIO,MissingReason.UNDEFINED).value is None


def test_floating_price_noise_does_not_create_a_false_nonzero_range():
    b=bars([dict(open=12.,high=12.+1e-12,low=12.,close=12.)]);x=block(b)
    assert x.values.zero_range_bar.iloc[0] == 1 and pd.isna(x.values.upper_wick_fraction.iloc[0])


def test_invalid_or_unknown_limit_cannot_be_reconstructed_from_ten_percent_name():
    b=bars();b['stock_name']='ST current name';b.limit_price_valid=False
    x=block(b);assert x.values.close_upper_limit.isna().all()
    assert x.missing.close_upper_limit.iloc[0]=='unknown_status'
    assert np.isfinite(x.values.upper_wick_fraction).all()
    b.limit_price_valid=True;b.high_limit=11
    x=block(b);assert x.values.close_upper_limit.isna().all()
    assert x.missing.close_upper_limit.iloc[0]=='invalid_record'


def test_current_up_streak_needs_known_start_and_never_terminal_future_length():
    b=bars([dict(open=10.,high=11.,low=9.,close=10.),dict(open=11.,high=12.,low=10.,close=12.),
            dict(open=12.,high=12.,low=11.,close=12.),dict(open=12.,high=12.,low=12.,close=12.)])
    x=block(b)
    assert x.values.upper_close_streak.tolist()==[0,1,2,3]
    assert x.values.upper_close_streak_lower_bound.tolist()==[0,1,2,3]
    early=block(b.iloc[:2]);pd.testing.assert_frame_equal(early.values,x.values.iloc[:2].reset_index(drop=True))
    pd.testing.assert_frame_equal(early.missing,x.missing.iloc[:2].reset_index(drop=True))
    assert x.values.age_since_observed_upper_close.iloc[1:4].tolist()==[0,0,0]


def test_cold_start_up_streak_is_a_lower_bound_until_non_up_boundary():
    b=bars([dict(open=12.,high=12.,low=12.,close=12.),dict(open=12.,high=12.,low=12.,close=12.),
            dict(open=10.,high=11.,low=9.,close=10.),dict(open=12.,high=12.,low=12.,close=12.)])
    x=block(b)
    assert x.values.upper_close_streak.iloc[:2].isna().all()
    assert x.values.upper_close_streak_lower_bound.tolist()==[1,2,0,1]
    assert x.values.upper_close_streak.iloc[2:].tolist()==[0,1]
    assert x.missing.upper_close_streak.iloc[0]=='insufficient_history'


def test_quote_gap_or_unknown_limit_breaks_certainty_not_calendar_weekends():
    b=bars([dict(open=10.,high=11.,low=9.,close=10.),dict(open=12.,high=12.,low=12.,close=12.),
            dict(open=12.,high=12.,low=12.,close=12.),dict(open=12.,high=12.,low=12.,close=12.),
            dict(open=12.,high=12.,low=12.,close=12.)])
    b=b.drop(index=2);x=block(b)
    assert pd.isna(x.values.upper_close_streak.iloc[2]) and x.values.upper_close_streak_lower_bound.iloc[2]==1
    assert x.values.quote_gap_sessions.iloc[2]==1
    # Friday->Monday is adjacent on the market calendar, not a three-day gap.
    assert x.values.quote_gap_sessions.iloc[3]==0 and x.values.upper_close_streak_lower_bound.iloc[3]==2
    b=bars([dict(open=10.,high=11.,low=9.,close=10.),dict(open=12.,high=12.,low=12.,close=12.),
            dict(open=12.,high=12.,low=12.,close=12.)]);b.loc[1,'limit_price_valid']=False
    x=block(b);assert pd.isna(x.values.upper_close_streak.iloc[2]) and x.values.upper_close_streak_lower_bound.iloc[2]==1


def test_no_volume_is_not_a_traded_touch_or_one_price_board():
    b=bars([dict(open=12.,high=12.,low=12.,close=12.)]);b.volume=0;x=block(b)
    assert x.values.one_price_upper_limit.isna().all() and x.values.zero_range_bar.isna().all()
    assert x.missing.one_price_upper_limit.iloc[0]=='unknown_status'


@pytest.mark.parametrize('column,value', [('high',9.),('open',0.),('low',float('inf')),('close',10.005)])
def test_inconsistent_missing_or_off_tick_ohlc_not_a_valid_shape(column,value):
    b=bars();b.loc[0,column]=value;x=block(b)
    assert x.values.upper_wick_fraction.isna().all() and x.values.close_upper_limit.isna().all()


def test_clock_declared_after_close_and_timezone_independent():
    x=block();assert x.values.observed_end.iloc[0].isoformat()=='2024-01-02T07:00:00+00:00'
    assert x.values.known_at.iloc[0].isoformat()=='2024-01-02T07:01:00+00:00'
    assert block(available_delay_seconds=600).values.known_at.iloc[0].hour==7
    assert block(available_delay_seconds=600).values.known_at.iloc[0].minute==10


def test_stock_isolation_and_input_order_do_not_change_feature_values():
    first=bars([dict(open=10.,high=11.,low=9.,close=10.),dict(open=12.,high=12.,low=12.,close=12.)])
    second=first.copy();second.stock_code='000002.SZ';second[['open','high','low','close']]=12.
    both=pd.concat([first,second]);x=block(both.sample(frac=1,random_state=3));y=block(first)
    pd.testing.assert_frame_equal(y.values,x.values.loc[x.values.stock_code.eq('000001.SZ')].reset_index(drop=True))
    assert x.values.loc[x.values.stock_code.eq('000002.SZ'),'upper_close_streak'].isna().all()


def test_future_provider_fields_cannot_change_any_current_feature():
    b=bars();x=block(b);b.future_label=-99;b.final_streak_using_future=9999;y=block(b)
    pd.testing.assert_frame_equal(x.values,y.values);pd.testing.assert_frame_equal(x.missing,y.missing)


def test_feature_block_rejects_value_reason_unit_label_and_clock_errors():
    x=block();x.values.loc[0,'close_upper_limit']=2
    with pytest.raises(ValueError,match='Binary'):x.validate()
    x=block();x.missing.loc[0,'upper_wick_fraction']='source_uncovered'
    with pytest.raises(ValueError,match='disagree'):x.validate()
    x=block();x.values['future_label']=.99
    with pytest.raises(ValueError,match='Undeclared'):x.validate()
    x=block();x.values.known_at='2024-01-02T06:00:00+00:00'
    with pytest.raises(ValueError,match='before'):x.validate()
    x=block();x.values.known_at='2024-01-02T07:01:00'
    with pytest.raises(ValueError,match='timezone'):x.validate()


def test_duplicate_non_calendar_bad_flag_and_invalid_tick_rejected():
    b=bars()
    with pytest.raises(ValueError,match='keys'):block(pd.concat([b,b]))
    b=bars();b.trade_date='2024-01-06'
    with pytest.raises(ValueError,match='calendar'):block(b)
    b=bars();b.limit_price_valid='True'
    with pytest.raises(ValueError,match='Boolean'):block(b)
    with pytest.raises(ValueError,match='tick'):block(tick_size=0)
    b=bars();b.stock_code='900001.SH'
    with pytest.raises(ValueError,match='CNY A-share'):block(b)


@pytest.mark.parametrize('price',[.35,4.35,9.95,19.99,12.59])
def test_identical_tick_prices_have_exact_zero_distance_and_gap(price):
    b=bars([dict(open=price,high=price,low=price,close=price)])
    b.pre_close=price;b.high_limit=price;b.low_limit=round(price*.8,2)
    x=block(b);v=x.values.iloc[0]
    assert v.close_upper_limit == 1
    assert v.upper_limit_distance_close == 0.
    assert v.overnight_gap_ratio == 0.
