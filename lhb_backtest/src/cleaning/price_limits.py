"""Validate supplied prices; never infer an executable limit from pct_chg.

Only seasoned non-ST SH/SZ A shares are supported here. Special regimes
remain unknown; original source prices are retained for investigation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_price_limits(df: pd.DataFrame, metadata: pd.DataFrame | None,
                          st_lookup: dict | None) -> pd.DataFrame:
    out = df.copy()
    codes = out.stock_code.astype(str)
    dates = out.trade_date.astype(str)
    pre = pd.to_numeric(out.pre_close, errors="coerce")
    known_st = dates.isin(st_lookup or {})
    st_keys = {f"{d}|{c}" for d, codes_ in (st_lookup or {}).items() for c in codes_}
    is_st = (dates + "|" + codes).isin(st_keys)
    out["is_st"] = np.where(known_st, is_st.astype(float), np.nan)
    listed = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    if metadata is not None and {"symbol", "listed_date"}.issubset(metadata):
        meta = metadata.drop_duplicates("symbol", keep="last").copy()
        meta["symbol"] = meta.symbol.str.replace(".SS", ".SH", regex=False)
        mapping = pd.to_datetime(meta.set_index("symbol").listed_date, errors="coerce")
        listed = codes.map(mapping)
    seasoned = (pd.to_datetime(dates) - listed).dt.days >= 30
    supported = codes.str.match(r"^(?:60\d{4}\.SH|00\d{4}\.SZ|30\d{4}\.SZ|688\d{3}\.SH)$")
    eligible = supported & seasoned & known_st & ~is_st & pre.gt(0)
    rate = pd.Series(.10, index=out.index)
    rate.loc[codes.str.startswith("688")] = .20
    rate.loc[codes.str.startswith("30") & dates.ge("2020-08-24")] = .20
    # Integer cents, round-half-up, not NumPy bankers rounding.
    cents = np.floor(pre * 100 + .5)
    expected_hi = np.floor(cents * (1 + rate) + .5 + 1e-8) / 100
    expected_lo = np.floor(cents * (1 - rate) + .5 + 1e-8) / 100
    hi = pd.to_numeric(out.get("high_limit", pd.Series(np.nan, index=out.index)), errors="coerce")
    lo = pd.to_numeric(out.get("low_limit", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["source_high_limit"] = hi
    out["source_low_limit"] = lo
    matches = (hi - expected_hi).abs().lt(1e-6) & (lo - expected_lo).abs().lt(1e-6)
    consistent = out.high.le(hi + 1e-4) & out.low.ge(lo - 1e-4)
    valid = eligible & matches & consistent
    out["limit_price_valid"] = valid
    out["limit_price_status"] = np.select(
        [~eligible, ~matches, ~consistent],
        ["unknown_regime_or_status", "source_rule_mismatch", "price_outside_limits"],
        default="validated_standard_regime",
    )
    out["high_limit"] = hi.where(valid)
    out["low_limit"] = lo.where(valid)
    return out
