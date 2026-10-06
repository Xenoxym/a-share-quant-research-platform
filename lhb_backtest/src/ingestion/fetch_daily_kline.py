"""Fetch daily K-line data from data sources."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import numpy as np
import duckdb
from tqdm import tqdm

from src.utils.io import save_parquet, _resolve_path
from src.cleaning.normalize_codes import normalize_stock_code
from src.utils.logging import get_logger

logger = get_logger("fetch_kline")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SAVE = _PROJECT_ROOT / "data" / "raw" / "daily_kline.parquet"


def fetch_daily_kline(
    start_date: str,
    end_date: str,
    stock_codes: list[str] | None = None,
    source: str = "akshare",
    save_path: str | None = None,
) -> pd.DataFrame:
    """Fetch daily K-line data for stocks.

    Parameters
    ----------
    start_date : str
        Start date in ``"YYYY-MM-DD"`` format.
    end_date : str
        End date in ``"YYYY-MM-DD"`` format.
    stock_codes : list[str] or None
        Specific stock codes to fetch. If None, fetch for all A-share stocks.
    source : str
        Data source to use: ``"akshare"`` (default) or ``"baostock"``.
    save_path : str or None
        Path to save raw data. Default: ``data/raw/daily_kline.parquet``.

    Returns
    -------
    pd.DataFrame
        Concatenated K-line data for all requested stocks.
    """
    start_date = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    if start_date > end_date:
        raise ValueError("start_date must not exceed end_date")
    out_path = _resolve_path(save_path or _DEFAULT_SAVE)
    # Old live-source files may contain adjusted prices or lots. Never mix
    # them (or a validated SimTradeData snapshot) into this acquisition.
    if out_path.exists():
        with duckdb.connect() as conn:
            old = conn.read_parquet(str(out_path))
            if not {"data_source", "quote_contract"}.issubset(old.columns):
                raise ValueError("Existing K-lines have no verified unit contract; use a separate save_path")
            tags = old.project("data_source, quote_contract").distinct().fetchall()
            if tags != [(source, "raw-shares-cny-v1")]:
                raise ValueError("Cannot mix K-line sources or units; use a separate save_path")
    client = _get_client(source)
    all_data, failed = [], []
    try:
        if stock_codes is None:
            stock_codes = _get_stock_list(source, client)
        stock_codes = list(dict.fromkeys(normalize_stock_code(str(c)) for c in stock_codes))
        if not stock_codes:
            raise RuntimeError("No stock universe returned; existing K-lines were preserved")
        logger.info(f"Fetching kline for {len(stock_codes)} stocks ({start_date} ~ {end_date})")
        for code in tqdm(stock_codes, desc="Fetching K-line"):
            try:
                symbol = code.split(".")[0] if source == "akshare" else code
                df = client.fetch_daily_kline(symbol, start_date, end_date)
                if df is None or df.empty:
                    raise ValueError("Empty response; completeness cannot be established")
                df = df.copy()
                df["stock_code"] = df.stock_code.map(normalize_stock_code)
                df["trade_date"] = pd.to_datetime(df.trade_date).dt.strftime("%Y-%m-%d")
                if (not df.stock_code.eq(code).all()
                        or not df.trade_date.between(start_date, end_date).all()
                        or df.duplicated(["stock_code", "trade_date"]).any()):
                    raise ValueError("Unexpected stock/date keys from source")
                numeric = ["open", "high", "low", "close", "volume", "amount"]
                values = df[numeric].apply(pd.to_numeric, errors="coerce")
                if (not np.isfinite(values.to_numpy()).all()
                        or values[["open", "high", "low", "close"]].le(0).any().any()
                        or values[["volume", "amount"]].lt(0).any().any()
                        or values.high.lt(values[["open", "close", "low"]].max(axis=1)).any()
                        or values.low.gt(values[["open", "close", "high"]].min(axis=1)).any()):
                    raise ValueError("Invalid OHLC, volume or amount")
                df[numeric] = values
                all_data.append(df)
            except Exception as exc:
                failed.append(code)
                logger.warning(f"Failed to fetch kline for {code}: {exc}")
            time.sleep(0.3)
    finally:
        if hasattr(client, "logout"):
            client.logout()
    from src.ingestion.update_contract import write_report
    report = {"status": "fail" if failed else "pass", "source": source,
              "start": start_date, "end": end_date, "failed_stocks": failed,
              "requested_stocks": len(stock_codes), "received_stocks": len(all_data)}
    if failed:
        write_report(out_path.with_suffix(".update.json"), report)
        raise RuntimeError(f"Incomplete K-line acquisition ({len(failed)} stocks); existing data preserved")
    result = pd.concat(all_data, ignore_index=True).sort_values(["stock_code", "trade_date"])
    result["data_source"] = source
    result["quote_contract"] = "raw-shares-cny-v1"
    if out_path.exists():
        from src.data_sources.parquet_merge import merge_kline_file
        merge_kline_file(out_path, result)
    else:
        save_parquet(result, out_path)
    write_report(out_path.with_suffix(".update.json"), report)
    result.attrs["update_report"] = report
    logger.info(f"Raw kline saved to {out_path}")

    return result


def _get_client(source: str):
    """Create the appropriate data source client."""
    if source == "akshare":
        from src.data_sources.akshare_client import AKShareClient
        return AKShareClient()
    elif source == "baostock":
        from src.data_sources.baostock_client import BaostockClient
        client = BaostockClient()
        client.login()
        return client
    elif source == "tushare":
        from src.data_sources.tushare_client import TushareClient
        return TushareClient()
    else:
        raise ValueError(f"Unknown data source: {source}")


def _get_stock_list(source: str, client) -> list[str]:
    """Get the list of all A-share stock codes."""
    logger.info("Fetching stock list...")
    try:
        if hasattr(client, "fetch_stock_list"):
            stock_df = client.fetch_stock_list()
            if stock_df is not None and not stock_df.empty:
                codes = stock_df["stock_code"].tolist()
                logger.info(f"Got {len(codes)} stocks from {source}")
                return codes
    except Exception as e:
        logger.warning(f"Failed to get stock list from {source}: {e}")

    from src.data_sources.akshare_client import AKShareClient
    fallback = AKShareClient()
    stock_df = fallback.fetch_stock_list()
    codes = stock_df["stock_code"].tolist() if not stock_df.empty else []
    logger.info(f"Got {len(codes)} stocks from akshare fallback")
    return codes
