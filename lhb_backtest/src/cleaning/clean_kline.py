"""Clean raw daily K-line data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.cleaning.normalize_codes import normalize_codes_in_df
from src.utils.io import load_parquet, save_parquet
from src.utils.logging import get_logger

logger = get_logger("clean_kline")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RAW = _PROJECT_ROOT / "data" / "raw" / "daily_kline.parquet"
_DEFAULT_CLEAN = _PROJECT_ROOT / "data" / "clean" / "daily_kline.parquet"

_NUMERIC_COLS = [
    "open", "high", "low", "close", "pre_close",
    "pct_chg", "volume", "amount", "turnover_rate",
]


def clean_daily_kline(
    raw_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    save_path: str | None = None,
    data_source: str | None = None,
    validate_limits: bool = True,
) -> pd.DataFrame:
    """Clean raw daily K-line data.

    Steps:
    1. Load raw data.
    2. Normalize stock codes and dates.
    3. Convert numeric columns to float.
    4. Compute ``pre_close`` from ``close`` and ``pct_chg`` if missing.
    5. Drop exact duplicates on ``(stock_code, trade_date)``.
    6. Sort by ``stock_code, trade_date``.
    7. Add ``data_source`` column.
    8. Save to parquet.

    Returns the cleaned DataFrame.
    """
    if raw_df is not None:
        df = raw_df.copy()
        logger.info(f"Cleaning kline from provided DataFrame ({len(df)} rows)")
    else:
        p = raw_path or str(_DEFAULT_RAW)
        logger.info(f"Loading raw kline from {p}")
        df = load_parquet(p)

    if df.empty:
        logger.warning("Empty raw kline data")
        return df

    # Normalize codes and dates
    df = normalize_codes_in_df(df, code_col="stock_code", date_col="trade_date")

    # Convert numeric columns
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute pre_close if missing
    if "pre_close" not in df.columns or df["pre_close"].isna().all():
        if "close" in df.columns and "pct_chg" in df.columns:
            df["pre_close"] = df["close"] / (1 + df["pct_chg"] / 100.0)
            logger.info("Computed pre_close from close and pct_chg")
        else:
            df["pre_close"] = None

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["stock_code", "trade_date"], keep="first")
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"Dropped {dropped} duplicate kline rows")

    # Sort
    df = df.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)

    # Add data source
    if data_source is not None:
        df["data_source"] = data_source
    elif "data_source" not in df:
        df["data_source"] = "unknown"
    if validate_limits:
        from src.cleaning.price_limits import validate_price_limits
        from src.data_sources.simtradedata_metadata import resolve_export_dir, load_status_lookup
        root = resolve_export_dir()
        meta_path = root / "metadata/stock_metadata.parquet" if root else None
        metadata = pd.read_parquet(meta_path) if meta_path and meta_path.exists() else None
        df = validate_price_limits(df, metadata, load_status_lookup("ST"))

    # Select output columns.
    # high_limit / low_limit are optional bonus columns from SimTradeData;
    # they are included when present and silently omitted otherwise so that
    # AKShare / BaoStock sources remain fully compatible.
    base_cols = [
        "trade_date", "stock_code", "open", "high", "low", "close",
        "pre_close", "pct_chg", "volume", "amount", "turnover_rate", "data_source",
    ]
    optional_cols = ["high_limit", "low_limit", "source_high_limit", "source_low_limit",
                     "limit_price_valid", "limit_price_status", "is_st"]
    for col in base_cols:
        if col not in df.columns:
            df[col] = None
    out_cols = base_cols + [c for c in optional_cols if c in df.columns]
    df = df[out_cols]

    # Save
    out_path = save_path or str(_DEFAULT_CLEAN)
    save_parquet(df, out_path)
    logger.info(f"Clean kline saved: {out_path} ({len(df)} rows)")

    return df
