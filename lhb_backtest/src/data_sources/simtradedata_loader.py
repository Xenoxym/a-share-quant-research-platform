"""Adapter: reads SimTradeData export/ Parquet files → internal K-line schema.

SimTradeData export layout (export/cn/):
    stocks/      {XXXXXX.SZ|SS}.parquet  cols: date,open,high,low,close,preclose,volume,money,high_limit,low_limit
    valuation/   {XXXXXX.SZ|SS}.parquet  cols: date,pe_ttm,pb,ps_ttm,pcf,turnover_rate,total_shares,a_floats,...

Internal K-line schema produced by this module:
    trade_date, stock_code, open, high, low, close, pre_close,
    pct_chg, volume, amount, turnover_rate, high_limit, low_limit, data_source
"""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from src.utils.logging import get_logger

logger = get_logger("simtradedata_loader")

# Rename: SimTradeData stocks/ → internal schema
_STOCKS_RENAME: dict[str, str] = {
    "date": "trade_date",
    "preclose": "pre_close",
    "money": "amount",
}

# Valuation columns to pull from valuation/ parquet (besides date + stock_code)
_VALUATION_PULL = ["turnover_rate"]

_A_SHARE_FILE = re.compile(
    r"^(?:6\d{5}\.SS|(?:00|30)\d{4}\.SZ|(?:43|83|87|92)\d{4}\.BJ)$"
)


def load_stocks_parquet(
    export_dir: str | Path,
    start_date: str,
    end_date: str,
    include_valuation: bool = True,
) -> pd.DataFrame:
    """Load SimTradeData stocks/ Parquet files into internal K-line schema.

    Parameters
    ----------
    export_dir : str or Path
        Path to SimTradeData export/cn/ directory (must contain a stocks/ sub-dir).
    start_date : str
        Start date inclusive, ``"YYYY-MM-DD"``.
    end_date : str
        End date inclusive, ``"YYYY-MM-DD"``.
    include_valuation : bool
        If True, left-join ``turnover_rate`` from valuation/ parquet files.

    Returns
    -------
    pd.DataFrame
        DataFrame with internal schema ready for ``clean_daily_kline``.
        Columns: trade_date, stock_code, open, high, low, close, pre_close,
                 pct_chg, volume, amount, turnover_rate, high_limit, low_limit, data_source.
    """
    export_dir = Path(export_dir)
    stocks_dir = export_dir / "stocks"

    if not stocks_dir.exists():
        raise FileNotFoundError(
            f"SimTradeData stocks/ directory not found: {stocks_dir}\n"
            "Check that simtradedata.export_dir in config.yaml points to the export/cn/ directory."
        )

    # SimTradeData's TDX importer deliberately exports both stocks and ETFs.
    # This backtester is stock-only, so do not let ETF/LOF files inflate daily
    # coverage or enter the event universe.
    parquet_files = sorted(
        path for path in stocks_dir.glob("*.parquet")
        if _is_a_share_stock_code(path.stem)
    )
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in: {stocks_dir}")

    logger.info(f"Found {len(parquet_files)} stock parquet files – loading {start_date} ~ {end_date}")

    tables: list[pa.Table] = []

    for f in tqdm(parquet_files, desc="Loading SimTradeData stocks", unit="stock", leave=False):
        stock_code_raw = f.stem  # e.g. "000001.SZ", "600519.SS"
        try:
            date_filter = _build_date_filter(f, start_date, end_date)
            table = _normalize_date_column(pq.read_table(f, filters=date_filter))
        except Exception as exc:
            raise RuntimeError(f"Cannot safely import stock file {f}") from exc

        if len(table) == 0:
            continue

        table = table.append_column(
            "stock_code_raw",
            pa.array([stock_code_raw] * len(table), type=pa.string()),
        )
        tables.append(table)

    if not tables:
        logger.warning(f"No rows found for date range {start_date} ~ {end_date}")
        return pd.DataFrame()

    # Bound Arrow conversion concurrency and release the input buffers before
    # the valuation join; full-market imports otherwise retain both datasets.
    df = pa.concat_tables(tables, promote_options="default").to_pandas(use_threads=False)
    del tables
    logger.info(f"Loaded {len(df):,} rows for {df['stock_code_raw'].nunique():,} stocks")

    # Rename fields to internal schema
    df = df.rename(columns={**_STOCKS_RENAME, "stock_code_raw": "stock_code"})

    # Normalise trade_date to "YYYY-MM-DD" string (internal schema contract)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")

    # Compute pct_chg from close / pre_close
    df["pct_chg"] = _compute_pct_chg(df)

    # Ensure bonus columns exist (SimTradeData computes them at export time)
    for col in ("high_limit", "low_limit"):
        if col not in df.columns:
            df[col] = None  # becomes NaN in float context

    df["data_source"] = "simtradedata"

    # Optionally join turnover_rate
    if include_valuation:
        df = _join_valuation(df, export_dir, start_date, end_date)
    else:
        if "turnover_rate" not in df.columns:
            df["turnover_rate"] = None

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_date_column(table: pa.Table) -> pa.Table:
    """Release and incremental files may use date32, us, ns or text dates."""
    position = table.schema.get_field_index('date')
    if position < 0:
        raise ValueError('Missing required date column')
    values = pd.to_datetime(table['date'].to_pandas(),format='mixed')
    return table.set_column(position,'date',pa.array(values,type=pa.timestamp('ns')))

def _build_date_filter(parquet_path: Path, start_date: str, end_date: str) -> list:
    """Build a PyArrow read filter matched to the file's ``date`` column type.

    SimTradeData production exports store ``date`` as TIMESTAMP, so filters
    must use ``pd.Timestamp`` — comparing string to timestamp raises
    ArrowNotImplementedError. Some exports (and test fixtures) store ``date``
    as plain strings, where string bounds work directly.
    """
    schema = pq.read_schema(parquet_path)
    if "date" in schema.names and pa.types.is_temporal(schema.field("date").type):
        return [("date", ">=", pd.Timestamp(start_date)), ("date", "<=", pd.Timestamp(end_date))]
    return [("date", ">=", start_date), ("date", "<=", end_date)]


def _compute_pct_chg(df: pd.DataFrame) -> pd.Series:
    """(close / pre_close - 1) × 100 ; NaN where pre_close ≤ 0."""
    pre = pd.to_numeric(df["pre_close"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    valid = pre > 0
    result = pd.Series(float("nan"), index=df.index, dtype="float64")
    result.loc[valid] = (close.loc[valid] / pre.loc[valid] - 1.0) * 100.0
    return result


def _join_valuation(
    df: pd.DataFrame,
    export_dir: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Left-join valuation columns (turnover_rate, …) from valuation/ parquet files."""
    val_dir = export_dir / "valuation"
    if not val_dir.exists():
        logger.warning(f"valuation/ dir not found ({val_dir}) — turnover_rate will be null")
        df["turnover_rate"] = None
        return df

    val_files = sorted(
        path for path in val_dir.glob("*.parquet")
        if _is_a_share_stock_code(path.stem)
    )
    if not val_files:
        logger.warning("No valuation parquet files — turnover_rate will be null")
        df["turnover_rate"] = None
        return df

    pull_cols = ["date"] + _VALUATION_PULL
    tables: list[pa.Table] = []

    for f in tqdm(val_files, desc="Loading valuation", unit="stock", leave=False):
        stock_code_raw = f.stem
        try:
            schema = pq.read_schema(f)
            available = [c for c in pull_cols if c in schema.names]
            if "date" not in available:
                raise ValueError('Missing required valuation date column')
            table = _normalize_date_column(pq.read_table(f, columns=available, filters=_build_date_filter(f, start_date, end_date)))
            if len(table) == 0:
                continue
            table = table.append_column(
                "stock_code",
                pa.array([stock_code_raw] * len(table), type=pa.string()),
            )
            tables.append(table)
        except Exception as exc:
            raise RuntimeError(f"Cannot safely import valuation file {f}") from exc

    if not tables:
        logger.warning("Valuation join produced no data — turnover_rate will be null")
        df["turnover_rate"] = None
        return df

    val_df = pa.concat_tables(tables, promote_options="default").to_pandas(use_threads=False)
    del tables
    val_df = val_df.rename(columns={"date": "trade_date"})
    val_df["trade_date"] = pd.to_datetime(val_df["trade_date"]).dt.strftime("%Y-%m-%d")

    # Deduplicate valuation join keys (keep last)
    val_df = val_df.drop_duplicates(subset=["trade_date", "stock_code"], keep="last")

    before_cols = set(df.columns)
    df = df.merge(
        val_df[["trade_date", "stock_code"] + [c for c in _VALUATION_PULL if c in val_df.columns]],
        on=["trade_date", "stock_code"],
        how="left",
    )
    new_cols = set(df.columns) - before_cols
    logger.info(f"Valuation join added columns: {sorted(new_cols)} ({val_df['stock_code'].nunique():,} stocks)")

    return df


def _is_a_share_stock_code(code: str) -> bool:
    """Return True for mainland A-share stock symbols, excluding funds/ETFs."""
    return _A_SHARE_FILE.fullmatch(code) is not None
