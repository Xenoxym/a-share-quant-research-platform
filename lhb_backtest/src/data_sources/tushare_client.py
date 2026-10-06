"""Tushare client for fetching A-share market and 龙虎榜 data (optional fallback)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("tushare_client")

_API_CALL_INTERVAL = 0.6  # Tushare has strict rate limits on free accounts

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

_KLINE_COLUMN_MAP = {
    "ts_code": "stock_code",
    "trade_date": "trade_date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "pre_close": "pre_close",
    "vol": "volume",
    "amount": "amount",
    "pct_chg": "pct_chg",
    "turnover_rate": "turnover_rate",
}

_DAILY_BASIC_COLUMN_MAP = {
    "ts_code": "stock_code",
    "trade_date": "trade_date",
    "turnover_rate": "turnover_rate",
    "float_share": "float_share",
    "total_mv": "total_market_cap",
    "circ_mv": "float_market_cap",
    "volume_ratio": "volume_ratio",
    "pe": "pe",
    "pb": "pb",
}

_LHB_COLUMN_MAP = {
    "ts_code": "stock_code",
    "trade_date": "trade_date",
    "name": "stock_name",
    "close": "close",
    "pct_change": "pct_chg",
    "turnover_rate": "turnover_rate",
    "amount": "daily_amount",
    "l_buy": "lhb_buy_total",
    "l_sell": "lhb_sell_total",
    "l_amount": "lhb_turnover",
    "net_amount": "lhb_net_buy",
    "net_rate": "net_buy_ratio",
    "amount_rate": "lhb_turnover_ratio",
    "float_values": "float_market_cap",
    "reason": "lhb_reason",
}


def _load_token_from_config() -> Optional[str]:
    """Try to read the tushare token from config/config.yaml."""
    try:
        import yaml

        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            return cfg.get("tushare", {}).get("token")
    except Exception:
        pass
    return None


def _format_ts_date(date_str: str) -> str:
    """Normalize a date string to ``YYYYMMDD`` for Tushare API calls."""
    return date_str.replace("-", "")


def _format_output_date(date_str: str) -> str:
    """Normalize a date string to ``YYYY-MM-DD`` for output."""
    clean = str(date_str).replace("-", "")[:8]
    return f"{clean[:4]}-{clean[4:6]}-{clean[6:8]}"


class TushareClient:
    """Client wrapping Tushare Pro APIs with standardized output format.

    Tushare requires a personal API token.  The client resolves the token
    from (in order of precedence):

    1. Explicit ``token`` constructor argument.
    2. Environment variable ``TUSHARE_TOKEN``.
    3. ``config/config.yaml`` under ``tushare.token``.

    If no token is found the client initialises in a *degraded* state and
    all data methods return empty DataFrames with a warning.
    """

    def __init__(self, token: Optional[str] = None) -> None:
        resolved_token = (
            token
            or os.environ.get("TUSHARE_TOKEN")
            or _load_token_from_config()
        )

        self._api = None
        if resolved_token:
            try:
                import tushare as ts

                ts.set_token(resolved_token)
                self._api = ts.pro_api()
                logger.info("Tushare Pro API initialized successfully")
            except Exception as exc:
                logger.warning("Failed to initialize Tushare Pro API: {}", exc)
        else:
            logger.warning(
                "No Tushare token found. Set TUSHARE_TOKEN env var, "
                "pass token= to constructor, or add tushare.token to config.yaml."
            )

    @property
    def is_available(self) -> bool:
        """Whether the Tushare API is initialised and usable."""
        return self._api is not None

    # ------------------------------------------------------------------
    # Daily K-line
    # ------------------------------------------------------------------

    def fetch_daily_kline(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Fetch daily K-line data for one stock via Tushare.

        Parameters
        ----------
        stock_code : str
            Stock code in ``"000001.SZ"`` format (matches Tushare natively).
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
        if not self.is_available:
            logger.warning("Tushare API not available; returning empty kline DataFrame")
            return _empty_kline_df()

        ts_start = _format_ts_date(start_date)
        ts_end = _format_ts_date(end_date)

        logger.info(
            "Fetching daily kline via Tushare for {} ({} ~ {})",
            stock_code, ts_start, ts_end,
        )
        time.sleep(_API_CALL_INTERVAL)

        try:
            df = self._api.daily(
                ts_code=stock_code,
                start_date=ts_start,
                end_date=ts_end,
            )
        except Exception as exc:
            logger.warning("Tushare daily() failed for {}: {}", stock_code, exc)
            return _empty_kline_df()

        if df is None or df.empty:
            logger.warning("Empty kline response from Tushare for {}", stock_code)
            return _empty_kline_df()

        df = df.rename(columns=_KLINE_COLUMN_MAP)

        if "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].apply(_format_output_date)

        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1000
        if "volume" in df.columns:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100

        target_cols = [
            "trade_date", "stock_code", "open", "high", "low", "close",
            "pre_close", "volume", "amount", "pct_chg", "turnover_rate",
        ]
        for col in target_cols:
            if col not in df.columns:
                df[col] = None

        df = df[target_cols].sort_values("trade_date").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Daily basic indicators
    # ------------------------------------------------------------------

    def fetch_daily_basic(self, trade_date: str) -> pd.DataFrame:
        """Fetch daily basic data for all stocks on a given date.

        Includes float_market_cap, total_market_cap, turnover_rate, etc.

        Parameters
        ----------
        trade_date : str
            Trade date in ``"YYYY-MM-DD"`` or ``"YYYYMMDD"`` format.

        Returns
        -------
        pd.DataFrame
            Columns: stock_code, trade_date, turnover_rate, float_market_cap,
            total_market_cap, volume_ratio, pe, pb.
        """
        if not self.is_available:
            logger.warning("Tushare API not available; returning empty daily_basic DataFrame")
            return _empty_daily_basic_df()

        ts_date = _format_ts_date(trade_date)

        logger.info("Fetching daily_basic via Tushare for {}", ts_date)
        time.sleep(_API_CALL_INTERVAL)

        try:
            df = self._api.daily_basic(trade_date=ts_date)
        except Exception as exc:
            logger.warning("Tushare daily_basic() failed for {}: {}", ts_date, exc)
            return _empty_daily_basic_df()

        if df is None or df.empty:
            logger.warning("Empty daily_basic response from Tushare for {}", ts_date)
            return _empty_daily_basic_df()

        df = df.rename(columns=_DAILY_BASIC_COLUMN_MAP)

        if "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].apply(_format_output_date)

        if "float_market_cap" in df.columns:
            df["float_market_cap"] = (
                pd.to_numeric(df["float_market_cap"], errors="coerce") * 10000
            )
        if "total_market_cap" in df.columns:
            df["total_market_cap"] = (
                pd.to_numeric(df["total_market_cap"], errors="coerce") * 10000
            )

        target_cols = [
            "stock_code", "trade_date", "turnover_rate", "float_market_cap",
            "total_market_cap", "volume_ratio", "pe", "pb",
        ]
        for col in target_cols:
            if col not in df.columns:
                df[col] = None

        return df[target_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # LHB (Dragon-Tiger List) detail
    # ------------------------------------------------------------------

    def fetch_lhb_detail(self, trade_date: str) -> pd.DataFrame:
        """Fetch 龙虎榜 detail for a date via Tushare.

        Parameters
        ----------
        trade_date : str
            Trade date in ``"YYYY-MM-DD"`` or ``"YYYYMMDD"`` format.

        Returns
        -------
        pd.DataFrame
            Columns: trade_date, stock_code, stock_name, close, pct_chg,
            lhb_net_buy, lhb_buy_total, lhb_sell_total, lhb_turnover,
            daily_amount, net_buy_ratio, lhb_turnover_ratio,
            turnover_rate, float_market_cap, lhb_reason.
        """
        if not self.is_available:
            logger.warning("Tushare API not available; returning empty LHB DataFrame")
            return _empty_lhb_df()

        ts_date = _format_ts_date(trade_date)

        logger.info("Fetching LHB detail via Tushare for {}", ts_date)
        time.sleep(_API_CALL_INTERVAL)

        try:
            df = self._api.top_list(trade_date=ts_date)
        except Exception as exc:
            logger.warning("Tushare top_list() failed for {}: {}", ts_date, exc)
            return _empty_lhb_df()

        if df is None or df.empty:
            logger.warning("Empty LHB response from Tushare for {}", ts_date)
            return _empty_lhb_df()

        df = df.rename(columns=_LHB_COLUMN_MAP)

        if "trade_date" in df.columns:
            df["trade_date"] = df["trade_date"].apply(_format_output_date)

        target_cols = [
            "trade_date", "stock_code", "stock_name", "close", "pct_chg",
            "lhb_net_buy", "lhb_buy_total", "lhb_sell_total", "lhb_turnover",
            "daily_amount", "net_buy_ratio", "lhb_turnover_ratio",
            "turnover_rate", "float_market_cap", "lhb_reason",
        ]
        for col in target_cols:
            if col not in df.columns:
                df[col] = None

        return df[target_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helper factories for empty DataFrames with correct schemas
# ---------------------------------------------------------------------------

def _empty_kline_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "trade_date", "stock_code", "open", "high", "low", "close",
        "volume", "amount", "pct_chg", "turnover_rate",
    ])


def _empty_daily_basic_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "stock_code", "trade_date", "turnover_rate", "float_market_cap",
        "total_market_cap", "volume_ratio", "pe", "pb",
    ])


def _empty_lhb_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "trade_date", "stock_code", "stock_name", "close", "pct_chg",
        "lhb_net_buy", "lhb_buy_total", "lhb_sell_total", "lhb_turnover",
        "daily_amount", "net_buy_ratio", "lhb_turnover_ratio",
        "turnover_rate", "float_market_cap", "lhb_reason",
    ])
