"""Baostock client for fetching A-share daily K-line data (fallback source)."""

from __future__ import annotations

import time
from typing import Optional

import baostock as bs
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("baostock_client")

_API_CALL_INTERVAL = 0.3

_KLINE_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"
)

_COLUMN_MAP = {
    "date": "trade_date",
    "code": "stock_code",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "preclose": "pre_close",
    "volume": "volume",
    "amount": "amount",
    "turn": "turnover_rate",
    "pctChg": "pct_chg",
}

_NUMERIC_COLS = [
    "open", "high", "low", "close", "pre_close",
    "volume", "amount", "turnover_rate", "pct_chg",
]


def _to_baostock_code(stock_code: str) -> str:
    """Convert our standard code format to baostock format.

    ``"600000.SH"`` -> ``"sh.600000"``, ``"000001.SZ"`` -> ``"sz.000001"``.
    Pure 6-digit codes are also accepted and auto-detected.
    """
    if "." in stock_code:
        parts = stock_code.split(".")
        pure_code = parts[0].strip().zfill(6)
        suffix = parts[1].strip().upper()
        exchange = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(suffix, "sz")
    else:
        pure_code = stock_code.strip().zfill(6)
        first = pure_code[0]
        exchange = "sh" if first == "6" else "sz"

    return f"{exchange}.{pure_code}"


def _to_standard_code(bs_code: str) -> str:
    """Convert baostock code back to our standard format.

    ``"sh.600000"`` -> ``"600000.SH"``, ``"sz.000001"`` -> ``"000001.SZ"``.
    """
    parts = bs_code.split(".")
    if len(parts) == 2:
        exchange = parts[0].strip().upper()
        pure_code = parts[1].strip().zfill(6)
        suffix_map = {"SH": "SH", "SZ": "SZ", "BJ": "BJ"}
        suffix = suffix_map.get(exchange, "SZ")
        return f"{pure_code}.{suffix}"
    return bs_code


class BaostockClient:
    """Client wrapping Baostock APIs with standardized output format.

    Baostock requires explicit ``login()`` / ``logout()`` calls.  The client
    tracks its own session state and automatically logs in on the first data
    request if the caller has not done so already.
    """

    def __init__(self) -> None:
        # Direct CLI/API calls need the same EOF/timeout/paging protection as
        # the official downloader launched with compat on PYTHONPATH.
        import compat.sitecustomize  # noqa: F401
        self._logged_in = False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def login(self) -> None:
        """Login to the Baostock service."""
        if self._logged_in:
            return
        try:
            result = bs.login()
            if result.error_code == "0":
                self._logged_in = True
                logger.info("Baostock login successful")
            else:
                logger.warning(
                    "Baostock login returned error: {} - {}",
                    result.error_code, result.error_msg,
                )
        except Exception as exc:
            logger.warning("Baostock login failed: {}", exc)

    def logout(self) -> None:
        """Logout from the Baostock service."""
        if not self._logged_in:
            return
        try:
            bs.logout()
            self._logged_in = False
            logger.info("Baostock logout successful")
        except Exception as exc:
            logger.warning("Baostock logout failed: {}", exc)

    def _ensure_login(self) -> None:
        if not self._logged_in:
            self.login()
        if not self._logged_in:
            raise ConnectionError("Baostock login did not establish a session")

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_daily_kline(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch daily K-line data for one stock via Baostock.

        Parameters
        ----------
        stock_code : str
            Stock code in ``"000001.SZ"`` format or pure 6-digit code.
        start_date : str
            Start date in ``"YYYY-MM-DD"`` or ``"YYYYMMDD"`` format.
        end_date : str
            End date in ``"YYYY-MM-DD"`` or ``"YYYYMMDD"`` format.

        Returns
        -------
        pd.DataFrame
            Columns: trade_date, stock_code, open, high, low, close,
            volume, amount, pct_chg, turnover_rate.
        """
        self._ensure_login()

        bs_code = _to_baostock_code(stock_code)
        fmt_start = _format_date(start_date)
        fmt_end = _format_date(end_date)

        logger.info(
            "Fetching daily kline via Baostock for {} ({} ~ {})",
            bs_code, fmt_start, fmt_end,
        )
        time.sleep(_API_CALL_INTERVAL)

        try:
            rs = bs.query_history_k_data_plus(
                code=bs_code,
                fields=_KLINE_FIELDS,
                start_date=fmt_start,
                end_date=fmt_end,
                frequency="d",
                adjustflag="3",  # 不复权: match the raw-price project contract
            )
        except Exception as exc:
            logger.warning("Baostock query failed for {}: {}", bs_code, exc)
            return _empty_kline_df()

        if rs is None or rs.error_code != "0":
            msg = getattr(rs, "error_msg", "unknown")
            logger.warning("Baostock query error for {}: {}", bs_code, msg)
            return _empty_kline_df()

        df = rs.get_data()
        if rs.error_code != "0":
            raise RuntimeError(f"Baostock pagination failed: {rs.error_code}: {rs.error_msg}")
        if df.empty:
            logger.warning("Empty kline response from Baostock for {}", bs_code)
            return _empty_kline_df()

        df = df.rename(columns=_COLUMN_MAP)

        for col in _NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "stock_code" in df.columns:
            df["stock_code"] = df["stock_code"].apply(_to_standard_code)

        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")

        target_cols = [
            "trade_date", "stock_code", "open", "high", "low", "close",
            "pre_close", "volume", "amount", "pct_chg", "turnover_rate",
        ]
        for col in target_cols:
            if col not in df.columns:
                df[col] = None

        return df[target_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> BaostockClient:
        self.login()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.logout()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_date(date_str: str) -> str:
    """Normalize a date string to ``YYYY-MM-DD`` for Baostock."""
    clean = date_str.replace("-", "")
    return f"{clean[:4]}-{clean[4:6]}-{clean[6:8]}"


def _empty_kline_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "trade_date", "stock_code", "open", "high", "low", "close",
        "volume", "amount", "pct_chg", "turnover_rate",
    ])
