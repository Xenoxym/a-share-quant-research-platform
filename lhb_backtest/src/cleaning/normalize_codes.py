"""Stock code and date normalization utilities."""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd


_EXCHANGE_SUFFIX_MAP = {
    "0": ".SZ",
    "3": ".SZ",
    "6": ".SH",
    "4": ".BJ",
    "8": ".BJ",
}


def normalize_stock_code(code: str) -> str:
    """Normalize a stock code to the ``"000001.SZ"`` format.

    Handles:
    - ``"000001"`` -> ``"000001.SZ"`` (infer from first digit)
    - ``"000001.SZ"`` -> ``"000001.SZ"`` (already correct)
    - ``"sz.000001"`` / ``"sh.600000"`` -> baostock format
    - ``"SZ000001"`` / ``"sz000001"``
    - ``"1.000001"`` (numeric exchange prefix)
    """
    if not code or not isinstance(code, str):
        return str(code)

    code = code.strip()

    # Already in dotted format: 000001.SZ / 600000.SH
    # Also handle SimTradeData's .SS suffix (Yahoo-style) → internal .SH
    if re.match(r"^\d{6}\.[A-Z]{2}$", code):
        if code.startswith("92"):
            return code[:6] + ".BJ"
        if code.startswith(("900", "110", "111", "113", "118")):
            return code[:6] + ".SH"
        if code.endswith(".SS"):
            return code[:-3] + ".SH"
        return code

    # Baostock: sh.600000 / sz.000001
    m = re.match(r"^(sh|sz|bj)\.(\d{6})$", code, re.IGNORECASE)
    if m:
        exch = m.group(1).upper()
        pure = m.group(2)
        suffix = {"SH": ".SH", "SZ": ".SZ", "BJ": ".BJ"}.get(exch, ".SZ")
        return f"{pure}{suffix}"

    # Prefixed: SZ000001, sz000001, SH600000
    m = re.match(r"^(SH|SZ|BJ|sh|sz|bj)(\d{6})$", code)
    if m:
        exch = m.group(1).upper()
        pure = m.group(2)
        suffix = {"SH": ".SH", "SZ": ".SZ", "BJ": ".BJ"}.get(exch, ".SZ")
        return f"{pure}{suffix}"

    # Numeric exchange prefix: 1.000001 (1=SZ, 0=SH)
    m = re.match(r"^(\d)\.(\d{6})$", code)
    if m:
        pure = m.group(2)
        return f"{pure}{_infer_suffix(pure)}"

    # Pure 6-digit code
    m = re.match(r"^(\d{6})$", code)
    if m:
        pure = m.group(1)
        return f"{pure}{_infer_suffix(pure)}"

    return code


def _infer_suffix(pure_code: str) -> str:
    """Infer exchange suffix from the first digit of a 6-digit code."""
    if pure_code.startswith("92"):
        return ".BJ"
    if pure_code.startswith(("900", "110", "111", "113", "118")):
        return ".SH"
    first = pure_code[0] if pure_code else ""
    return _EXCHANGE_SUFFIX_MAP.get(first, ".SZ")


def normalize_date(date_val) -> Optional[str]:
    """Normalize a date value to ``"YYYY-MM-DD"`` string format.

    Handles strings (``"20220101"``, ``"2022-01-01"``), datetime, Timestamp, and NaT.
    """
    if date_val is None or (isinstance(date_val, float) and pd.isna(date_val)):
        return None
    if isinstance(date_val, pd.Timestamp):
        if pd.isna(date_val):
            return None
        return date_val.strftime("%Y-%m-%d")
    s = str(date_val).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    except Exception:
        return s[:10] if len(s) >= 10 else s


def normalize_codes_in_df(
    df: pd.DataFrame,
    code_col: str = "stock_code",
    date_col: str = "trade_date",
) -> pd.DataFrame:
    """Apply :func:`normalize_stock_code` and :func:`normalize_date` to a DataFrame.

    Normalisation runs once per *unique* value and is broadcast with ``map``:
    on multi-million-row kline frames this is orders of magnitude faster than
    a per-row ``apply`` (codes/dates have only a few thousand unique values).
    """
    out = df.copy()
    if code_col in out.columns:
        codes = out[code_col].astype(str)
        code_map = {c: normalize_stock_code(c) for c in codes.unique()}
        out[code_col] = codes.map(code_map)
    if date_col in out.columns:
        date_map = {d: normalize_date(d) for d in out[date_col].unique()}
        out[date_col] = out[date_col].map(date_map)
    return out
