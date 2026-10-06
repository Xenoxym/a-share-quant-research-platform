"""AKShare client for fetching A-share market data and 龙虎榜 (Dragon-Tiger List) data."""

from __future__ import annotations

import time

import akshare as ak
import pandas as pd
import requests

from src.utils.logging import get_logger

logger = get_logger("akshare_client")

# AKShare Chinese -> English column mappings
_KLINE_COLUMN_MAP = {
    "日期": "trade_date",
    "股票代码": "stock_code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
    "换手率": "turnover_rate",
}

# stock_lhb_detail_em — range call, returns all events within start_date/end_date
# Actual column name for date is "上榜日" (NOT "上榜日期")
_LHB_SUMMARY_COLUMN_MAP = {
    "序号": "_seq",
    "代码": "stock_code",
    "名称": "stock_name",
    "上榜日": "trade_date",          # actual API field name — was wrongly mapped before
    "解读": "lhb_interpret",         # short categorical label (e.g. "涨幅偏离")
    "收盘价": "close",
    "涨跌幅": "pct_chg",
    "龙虎榜净买额": "lhb_net_buy",
    "龙虎榜买入额": "lhb_buy_total",
    "龙虎榜卖出额": "lhb_sell_total",
    "龙虎榜成交额": "lhb_turnover",
    "市场总成交额": "daily_amount",
    "净买额占总成交比": "net_buy_ratio",
    "成交额占总成交比": "lhb_turnover_ratio",
    "换手率": "turnover_rate",
    "流通市值": "float_market_cap",
    "上榜原因": "lhb_reason",
    "上榜后1日": "return_1d",        # EastMoney pre-computed forward returns (%)
    "上榜后2日": "return_2d",
    "上榜后5日": "return_5d",
    "上榜后10日": "return_10d",
}

# stock_lhb_stock_detail_em — per-stock per-date broker seat detail
# "买入金额" always means the broker's PURCHASE amount for that session.
# "净额" = buy - sell, so implied sell = buy - net (negative net → net seller).
_LHB_BROKER_DETAIL_COLUMN_MAP = {
    "序号": "rank",
    "交易营业部名称": "broker_name",
    "买入金额": "buy_amount",
    "卖出金额": "sell_amount",
    "买入金额-占总成交比例": "buy_amount_ratio",
    "卖出金额-占总成交比例": "sell_amount_ratio",
    "净额": "net_amount",
    "类型": "_lhb_type",             # same content as lhb_reason — dropped on output
}

_STOCK_LIST_COLUMN_MAP = {
    "code": "stock_code",
    "name": "stock_name",
}

_API_CALL_INTERVAL = 0.5  # seconds between API calls


def _add_exchange_suffix(code: str) -> str:
    """Convert a pure 6-digit stock code to exchange-suffixed format.

    '000001' -> '000001.SZ', '600000' -> '600000.SH', '430047' -> '430047.BJ'
    """
    code = str(code).strip().zfill(6)
    if code.startswith("92"):
        return f"{code}.BJ"
    first = code[0]
    if first in ("0", "3"):
        return f"{code}.SZ"
    elif first == "6" or code.startswith(("900", "110", "111", "113", "118")):
        return f"{code}.SH"
    elif first in ("4", "8"):
        return f"{code}.BJ"
    return f"{code}.SZ"


class AKShareClient:
    """Client wrapping AKShare APIs with standardized output format."""

    def fetch_daily_kline(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "",
    ) -> pd.DataFrame:
        """Fetch daily K-line for one stock.

        Parameters
        ----------
        symbol : str
            Pure 6-digit stock code, e.g. ``"000001"``.
        start_date : str
            Start date in ``"YYYY-MM-DD"`` or ``"YYYYMMDD"`` format.
        end_date : str
            End date in ``"YYYY-MM-DD"`` or ``"YYYYMMDD"`` format.
        adjust : str
            Price adjustment: ``""`` (raw, default), ``"hfq"`` (back-adjusted),
            or ``"qfq"`` (forward-adjusted). The pipeline uses raw prices.

        Returns
        -------
        pd.DataFrame
            Columns: trade_date, stock_code, open, high, low, close,
            volume, amount, pct_chg, turnover_rate.
        """
        symbol = str(symbol).strip().zfill(6)
        ak_start = start_date.replace("-", "")
        ak_end = end_date.replace("-", "")

        logger.info("Fetching daily kline for {} ({} ~ {})", symbol, ak_start, ak_end)
        time.sleep(_API_CALL_INTERVAL)

        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=ak_start,
                end_date=ak_end,
                adjust=adjust,
                timeout=30,
            )
        except Exception as exc:
            logger.warning("Failed to fetch kline for {}: {}", symbol, exc)
            return _empty_kline_df()

        if df is None or df.empty:
            logger.warning("Empty kline response for {}", symbol)
            return _empty_kline_df()

        df = df.rename(columns=_KLINE_COLUMN_MAP)

        # EastMoney reports volume in lots; the project contract is shares.
        if "volume" in df:
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce") * 100

        stock_code_with_suffix = _add_exchange_suffix(symbol)
        df["stock_code"] = stock_code_with_suffix

        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")

        target_cols = [
            "trade_date", "stock_code", "open", "high", "low", "close",
            "volume", "amount", "pct_chg", "turnover_rate",
        ]
        for col in target_cols:
            if col not in df.columns:
                df[col] = None

        return df[target_cols].reset_index(drop=True)

    def fetch_lhb_summary(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch 龙虎榜 summary for a date range in ONE API call.

        Parameters
        ----------
        start_date : str
            Start date ``"YYYY-MM-DD"`` or ``"YYYYMMDD"``.
        end_date : str
            End date ``"YYYY-MM-DD"`` or ``"YYYYMMDD"`` (inclusive).

        Returns
        -------
        pd.DataFrame
            Columns: trade_date, stock_code, stock_name, lhb_interpret, close,
            pct_chg, lhb_net_buy, lhb_buy_total, lhb_sell_total, lhb_turnover,
            daily_amount, net_buy_ratio, lhb_turnover_ratio, turnover_rate,
            float_market_cap, lhb_reason, return_1d, return_2d, return_5d, return_10d.
        """
        ak_start = start_date.replace("-", "")
        ak_end = end_date.replace("-", "")

        logger.info("Fetching LHB summary {} ~ {}", ak_start, ak_end)
        time.sleep(_API_CALL_INTERVAL)

        try:
            df = ak.stock_lhb_detail_em(start_date=ak_start, end_date=ak_end)
        except Exception as exc:
            logger.warning("Failed to fetch LHB summary {} ~ {}: {}", ak_start, ak_end, exc)
            raise

        if df is None or df.empty:
            raise RuntimeError(f'Unconfirmed empty LHB summary: {ak_start}~{ak_end}')

        # AKShare returns the concatenated frame but discards pagination totals.
        # Independently confirm the source count before accepting the range.
        lo=pd.Timestamp(ak_start).strftime('%Y-%m-%d')
        hi=pd.Timestamp(ak_end).strftime('%Y-%m-%d')
        response=requests.get('https://datacenter-web.eastmoney.com/api/data/v1/get',
            params={'reportName':'RPT_DAILYBILLBOARD_DETAILSNEW','columns':'SECURITY_CODE,TRADE_DATE',
                    'filter':f"(TRADE_DATE>='{lo}')(TRADE_DATE<='{hi}')",
                    'pageNumber':'1','pageSize':'1','source':'WEB','client':'WEB'},timeout=(10,45))
        response.raise_for_status()
        body=response.json()
        if not body.get('success') or not body.get('result'):
            raise RuntimeError('Could not verify LHB summary source count')
        if len(df)!=int(body['result']['count']):
            raise RuntimeError('LHB summary pagination is incomplete or source changed during acquisition')

        df = df.rename(columns=_LHB_SUMMARY_COLUMN_MAP)

        if "stock_code" in df.columns:
            df["stock_code"] = df["stock_code"].apply(
                lambda x: _add_exchange_suffix(str(x).strip().zfill(6))
            )

        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y-%m-%d")

        target_cols = [
            "trade_date", "stock_code", "stock_name", "lhb_interpret", "close", "pct_chg",
            "lhb_net_buy", "lhb_buy_total", "lhb_sell_total", "lhb_turnover",
            "daily_amount", "net_buy_ratio", "lhb_turnover_ratio",
            "turnover_rate", "float_market_cap", "lhb_reason",
            "return_1d", "return_2d", "return_5d", "return_10d",
        ]
        for col in target_cols:
            if col not in df.columns:
                df[col] = None

        return df[target_cols].reset_index(drop=True)

    def fetch_lhb_stock_detail_both(
        self, stock_code: str, date: str
    ) -> pd.DataFrame:
        """Fetch top-5 buy AND sell broker seat detail for one stock on one date.

        Makes two API calls (flag="买入" then flag="卖出") via
        ``stock_lhb_stock_detail_em`` and returns the combined DataFrame.

        Parameters
        ----------
        stock_code : str
            Stock code ``"000001.SZ"`` or pure 6-digit.
        date : str
            Trade date ``"YYYY-MM-DD"`` or ``"YYYYMMDD"``.

        Returns
        -------
        pd.DataFrame
            Columns: trade_date, stock_code, flag, rank, broker_name,
            buy_amount, buy_amount_ratio, sell_amount_ratio, net_amount.
            ``flag`` is "买入" (top buyers) or "卖出" (top sellers).
        """
        buy_df = self._fetch_broker_side(stock_code, date, "买入")
        sell_df = self._fetch_broker_side(stock_code, date, "卖出")
        return pd.concat([buy_df, sell_df], ignore_index=True)

    def _fetch_broker_side(
        self, stock_code: str, date: str, flag: str
    ) -> pd.DataFrame:
        """Internal: fetch one side (买入 or 卖出) of broker detail."""
        pure_code = stock_code.split(".")[0].strip().zfill(6)
        ak_date = date.replace("-", "")
        code_with_suffix = _add_exchange_suffix(pure_code)
        formatted_date = pd.to_datetime(ak_date).strftime("%Y-%m-%d")

        logger.info(
            "Fetching LHB broker {} {} {}", code_with_suffix, ak_date, flag
        )
        time.sleep(_API_CALL_INTERVAL / 2)  # two calls per pair — halve sleep

        try:
            df = ak.stock_lhb_stock_detail_em(
                symbol=pure_code, date=ak_date, flag=flag
            )
        except Exception as exc:
            # Re-raise so that the outer with_retry() in fetch_lhb_detail can
            # apply back-off retries and record the failure in failures_lhb_detail.csv.
            # Do NOT silently return empty here — that would mark the pair as
            # "resolved" even though we have no buy or sell data.
            logger.warning(
                "Failed LHB broker {} {} {}: {}", code_with_suffix, ak_date, flag, exc
            )
            raise

        if df is None or df.empty:
            return _empty_broker_detail_df()

        df = df.rename(columns=_LHB_BROKER_DETAIL_COLUMN_MAP)
        df["trade_date"] = formatted_date
        df["stock_code"] = code_with_suffix
        df["flag"] = flag

        # When a stock qualifies for multiple LHB criteria on the same day the
        # API may return one set of 5 seats per criterion (same seats repeated).
        # Truncate to the first 5 rows — the API already ranks by amount so
        # head(5) gives the true top-5 for this side.
        # NOTE: do NOT dedup by broker_name — some institutions are anonymous
        #       (same placeholder name ≠ same institution).
        if "_lhb_type" in df.columns:
            # Preserve a whole report; never combine rows from different windows.
            from src.data_sources.reporting_windows import reporting_window_days
            types = sorted(df["_lhb_type"].dropna().astype(str).unique(),
                           key=lambda value: (reporting_window_days(value) or 999, value))
            if types:
                df = df[df["_lhb_type"].astype(str).eq(types[0])].copy()
                df["report_reason"] = types[0]
        amount_column = "buy_amount" if flag == "买入" else "sell_amount"
        if amount_column in df:
            df = df.sort_values(amount_column, ascending=False, kind="stable")
        df = df.head(5)

        df = df.drop(columns=["_lhb_type"], errors="ignore")
        df = df.reset_index(drop=True)
        df["rank"] = df.index + 1

        target_cols = [
            "trade_date", "stock_code", "flag", "rank",
            "broker_name", "buy_amount", "sell_amount", "buy_amount_ratio",
            "sell_amount_ratio", "net_amount", "report_reason",
        ]
        for col in target_cols:
            if col not in df.columns:
                df[col] = None

        return df[target_cols].reset_index(drop=True)

    def fetch_stock_list(self) -> pd.DataFrame:
        """Fetch current A-share stock list with basic info.

        Returns
        -------
        pd.DataFrame
            Columns: stock_code, stock_name.
        """
        logger.info("Fetching A-share stock list")
        time.sleep(_API_CALL_INTERVAL)

        try:
            df = ak.stock_info_a_code_name()
        except Exception as exc:
            logger.warning("Failed to fetch stock list: {}", exc)
            return pd.DataFrame(columns=["stock_code", "stock_name"])

        if df is None or df.empty:
            logger.warning("Empty stock list response")
            return pd.DataFrame(columns=["stock_code", "stock_name"])

        df = df.rename(columns=_STOCK_LIST_COLUMN_MAP)

        if "stock_code" in df.columns:
            df["stock_code"] = df["stock_code"].apply(
                lambda x: _add_exchange_suffix(str(x).strip().zfill(6))
            )

        return df[["stock_code", "stock_name"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helper factories for empty DataFrames with correct schemas
# ---------------------------------------------------------------------------

def _empty_kline_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "trade_date", "stock_code", "open", "high", "low", "close",
        "volume", "amount", "pct_chg", "turnover_rate",
    ])


def _empty_lhb_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "trade_date", "stock_code", "stock_name", "lhb_interpret", "close", "pct_chg",
        "lhb_net_buy", "lhb_buy_total", "lhb_sell_total", "lhb_turnover",
        "daily_amount", "net_buy_ratio", "lhb_turnover_ratio",
        "turnover_rate", "float_market_cap", "lhb_reason",
        "return_1d", "return_2d", "return_5d", "return_10d",
    ])


def _empty_broker_detail_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "trade_date", "stock_code", "flag", "rank",
        "broker_name", "buy_amount", "buy_amount_ratio",
        "sell_amount_ratio", "net_amount",
    ])
