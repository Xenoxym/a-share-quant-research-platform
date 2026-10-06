"""Limit-up detection and board type classification for A-share stocks."""

from __future__ import annotations

import pandas as pd

_LIMIT_THRESHOLDS = {
    "main_sh": 9.8,
    "main_sz": 9.8,
    "chinext": 19.5,
    "star": 19.5,
    "bse": 29.5,
    "st": 4.8,
}


def get_board_type(stock_code: str) -> str:
    """Determine the board type from a stock code.

    Parameters
    ----------
    stock_code : str
        Stock code in ``"000001.SZ"`` format.

    Returns
    -------
    str
        One of: ``"main_sh"``, ``"main_sz"``, ``"chinext"``, ``"star"``,
        ``"bse"``, or ``"unknown"``.
    """
    pure = stock_code.split(".")[0].strip() if isinstance(stock_code, str) else ""
    if not pure or len(pure) < 3:
        return "unknown"

    prefix2 = pure[:2]
    prefix3 = pure[:3]

    if prefix3 in ("600", "601", "603", "605"):
        return "main_sh"
    if prefix3 in ("000", "001", "002", "003"):
        return "main_sz"
    if prefix3 in ("300", "301"):
        return "chinext"
    if prefix3 == "688":
        return "star"
    if prefix2 in ("43", "83", "87", "92"):
        return "bse"
    if pure[0] in ("4", "8"):
        return "bse"

    return "unknown"


def get_limit_up_threshold(stock_code: str, stock_name: str = "") -> float:
    """Legacy display heuristic; never use for historical execution or labels.

    Parameters
    ----------
    stock_code : str
        Stock code in ``"000001.SZ"`` format.
    stock_name : str
        Stock name (used to detect ST stocks).

    Returns
    -------
    float
        Threshold percentage: 4.8 (ST), 9.8 (main board), 19.5 (ChiNext/STAR),
        or 29.5 (BSE).
    """
    if stock_name and "ST" in stock_name.upper():
        return _LIMIT_THRESHOLDS["st"]

    board = get_board_type(stock_code)
    return _LIMIT_THRESHOLDS.get(board, 9.8)


def detect_limit_up(pct_chg: float, stock_code: str, stock_name: str = "") -> bool:
    """Legacy approximate display helper, not historical limit-up validation.

    Parameters
    ----------
    pct_chg : float
        Daily percentage change (e.g. 10.0 means 10%).
    stock_code : str
        Stock code in ``"000001.SZ"`` format.
    stock_name : str
        Stock name (used to detect ST stocks).

    Returns
    -------
    bool
        True if the stock is considered to have hit limit-up.
    """
    if pd.isna(pct_chg):
        return False
    threshold = get_limit_up_threshold(stock_code, stock_name)
    return float(pct_chg) >= threshold


def detect_limit_up_series(
    df: pd.DataFrame,
    pct_chg_col: str = "pct_chg",
    code_col: str = "stock_code",
    name_col: str = "stock_name",
) -> pd.Series:
    """Apply limit-up detection to each row of a DataFrame.

    Returns a boolean Series aligned with the input DataFrame index.
    """
    if df.empty:
        return pd.Series(dtype=bool)

    names = df[name_col] if name_col in df.columns else pd.Series("", index=df.index)

    return pd.Series(
        [
            detect_limit_up(pct, code, name)
            for pct, code, name in zip(df[pct_chg_col], df[code_col], names)
        ],
        index=df.index,
    )
