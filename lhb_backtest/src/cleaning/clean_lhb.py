"""Clean raw LHB summary and broker detail data, and filter LHB events."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cleaning.normalize_codes import normalize_codes_in_df
from src.utils.io import load_parquet, save_parquet
from src.utils.logging import get_logger

logger = get_logger("clean_lhb")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_LHB_SUMMARY_NUMERIC_COLS = [
    "close", "pct_chg", "lhb_net_buy", "lhb_buy_total", "lhb_sell_total",
    "lhb_turnover", "daily_amount", "net_buy_ratio", "lhb_turnover_ratio",
    "turnover_rate", "float_market_cap",
    "return_1d", "return_2d", "return_5d", "return_10d",
]

# Unit contract (see README "数据量纲契约"):
#   - pct_chg / turnover_rate stay in PERCENT (matches limit-up thresholds).
#   - every other ratio / return column is a DECIMAL FRACTION.
# The EastMoney API returns these columns in percent, so the clean layer
# converts them exactly once (raw keeps original API units).
_PERCENT_TO_DECIMAL_COLS = [
    "net_buy_ratio", "lhb_turnover_ratio",
    "return_1d", "return_2d", "return_5d", "return_10d",
]

_BROKER_DETAIL_NUMERIC_COLS = [
    "buy_amount", "sell_amount", "buy_amount_ratio", "sell_amount_ratio", "net_amount",
]

# Default keywords for the 上榜原因 filter.
# Keep stocks whose lhb_reason contains at least one of these strings.
_DEFAULT_REASON_KEYWORDS = ["涨幅"]


# ---------------------------------------------------------------------------
# Event filter (used by fetch_lhb_detail before deciding which pairs to fetch)
# ---------------------------------------------------------------------------

def _aggregate_multi_reason_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Use the same coherent reporting-window selection as ingestion."""
    from src.ingestion.fetch_lhb_summary import _aggregate_multi_reason_rows as aggregate
    return aggregate(df)


def filter_lhb_events(
    df: pd.DataFrame,
    reason_keywords: list[str] | None = None,
) -> pd.DataFrame:
    """Apply three sequential filters to a raw LHB summary DataFrame.

    Filters (applied in this order):
    1. **Non-ST**: remove stocks whose name starts with ``ST`` or ``*ST``.
    2. **上榜原因**: keep only rows where ``lhb_reason`` contains at least
       one keyword from ``reason_keywords`` (default: ``["涨幅"]``).
    3. **First-appearance**: for each stock, remove dates where the stock
       *also appeared on the LHB on the immediately preceding trading day*
       (consecutive-day appearances).  Only the first day of each streak
       is kept.

    Parameters
    ----------
    df : pd.DataFrame
        Must have at minimum columns ``stock_code``, ``trade_date``,
        ``stock_name``, ``lhb_reason``.
    reason_keywords : list[str] or None
        Override the default keyword list.

    Returns
    -------
    pd.DataFrame
        Filtered subset of *df*.
    """
    if df.empty:
        return df

    before = len(df)
    df = _filter_remove_st(df)
    after_st = len(df)

    df = _filter_lhb_reason(df, reason_keywords or _DEFAULT_REASON_KEYWORDS)
    after_reason = len(df)

    df = _filter_first_appearance(df)
    after_first = len(df)

    logger.info(
        f"filter_lhb_events: {before} → {after_st} (ST) "
        f"→ {after_reason} (reason) → {after_first} (first-appearance)"
    )
    return df


def _filter_remove_st(df: pd.DataFrame) -> pd.DataFrame:
    if "stock_name" not in df.columns:
        return df
    st_mask = df["stock_name"].str.contains(r"^\*?ST", case=False, na=False, regex=True)
    return df[~st_mask].reset_index(drop=True)


def _filter_lhb_reason(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if "lhb_reason" not in df.columns or not keywords:
        return df
    pattern = "|".join(keywords)
    mask = df["lhb_reason"].str.contains(pattern, na=False)
    return df[mask].reset_index(drop=True)


def _filter_first_appearance(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first day of each consecutive LHB streak per stock.

    A streak is consecutive when a stock appeared on the LHB on *both*
    trading-day T and trading-day T-1.  The T-1 appearance is looked up
    in the same DataFrame, so the full history should be passed in (not
    just the newly-fetched subset) for accurate cross-batch detection.
    """
    if df.empty:
        return df
    if "trade_date" not in df.columns or "stock_code" not in df.columns:
        return df

    from src.utils.calendar import get_trading_dates

    dates_str = df["trade_date"].astype(str)
    min_date = dates_str.min()
    max_date = dates_str.max()

    # Extend lookback by ~2 weeks so we can detect streaks that start just
    # before the earliest date in this DataFrame.
    lookback_start = (
        pd.Timestamp(min_date) - pd.Timedelta(days=14)
    ).strftime("%Y-%m-%d")

    all_trading = get_trading_dates(lookback_start, max_date)
    # Build prev_trading_day lookup: date → previous trading day (or "")
    prev_map = {
        d: (all_trading[i - 1] if i > 0 else "") for i, d in enumerate(all_trading)
    }

    df2 = df.copy()
    df2["_prev_td"] = dates_str.map(prev_map).fillna("")

    # Vectorised: check whether (stock_code, prev_trading_day) exists in df
    existing_keys: set[str] = set(
        df2["stock_code"].astype(str) + "|" + dates_str
    )
    check_keys = df2["stock_code"].astype(str) + "|" + df2["_prev_td"].astype(str)
    df2["_is_consecutive"] = check_keys.isin(existing_keys)

    # NOTE: original index is preserved intentionally — filter_engine
    # intersects it with the caller's DataFrame index. (A reset_index here
    # previously turned the intersection into "keep the first N positional
    # rows", silently dropping all later-dated events.)
    result = df2[~df2["_is_consecutive"]].drop(columns=["_prev_td", "_is_consecutive"])
    return result


# ---------------------------------------------------------------------------
# Clean LHB summary
# ---------------------------------------------------------------------------

def clean_lhb_summary(
    raw_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    save_path: str | None = None,
    data_source: str = "akshare",
) -> pd.DataFrame:
    """Clean raw LHB summary data.

    Steps: normalise codes/dates → numeric cast → dedup → sort → save.
    The cleaned output retains ALL events (no filtering); use
    ``filter_lhb_events()`` separately when building the feature table.
    """
    if raw_df is not None:
        df = raw_df.copy()
        logger.info(f"Cleaning LHB summary from provided DataFrame ({len(df)} rows)")
    else:
        default = _PROJECT_ROOT / "data" / "raw" / "lhb_summary.parquet"
        p = raw_path or str(default)
        logger.info(f"Loading raw LHB summary from {p}")
        df = load_parquet(p)

    if df.empty:
        logger.warning("Empty raw LHB summary data")
        return df

    df = normalize_codes_in_df(df, code_col="stock_code", date_col="trade_date")

    for col in _LHB_SUMMARY_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Enforce unit contract: API percent columns → decimal fractions
    for col in _PERCENT_TO_DECIMAL_COLS:
        if col in df.columns:
            df[col] = df[col] / 100.0

    # Aggregate multi-reason duplicates: join lhb_reason / lhb_interpret with "；"
    before = len(df)
    df = _aggregate_multi_reason_rows(df)
    if (merged := before - len(df)):
        logger.info(f"Merged {merged} multi-reason duplicate rows (reasons concatenated with '；')")

    df = df.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
    df["data_source"] = data_source

    out_cols = [
        "trade_date", "stock_code", "stock_name", "lhb_interpret",
        "close", "pct_chg",
        "lhb_net_buy", "lhb_buy_total", "lhb_sell_total", "lhb_turnover",
        "daily_amount", "net_buy_ratio", "lhb_turnover_ratio",
        "turnover_rate", "float_market_cap", "lhb_reason",
        "return_1d", "return_2d", "return_5d", "return_10d",
        "data_source",
    ]
    for col in out_cols:
        if col not in df.columns:
            df[col] = None
    out_cols += [c for c in ("summary_selected_reason", "summary_window_days") if c in df]
    df = df[out_cols]

    out_path = save_path or str(_PROJECT_ROOT / "data" / "clean" / "lhb_summary.parquet")
    save_parquet(df, out_path)
    logger.info(f"Clean LHB summary saved: {out_path} ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Clean LHB broker detail
# ---------------------------------------------------------------------------

def clean_lhb_broker_detail(
    raw_path: str | None = None,
    raw_df: pd.DataFrame | None = None,
    save_path: str | None = None,
    data_source: str = "akshare",
) -> pd.DataFrame:
    """Clean raw LHB broker detail data.

    The raw data from ``stock_lhb_stock_detail_em`` has:
    - ``flag``        "买入" | "卖出"   (which side was ranked)
    - ``buy_amount``  broker's purchase amount (always)
    - ``net_amount``  buy − sell (negative → net seller)

    This step derives:
    - ``direction``   "buy" | "sell"  (from flag)
    - ``sell_amount`` buy_amount − net_amount  (implied sell amount)

    After this step ``build_lhb_features._pivot_broker_detail`` can use
    ``direction == "buy" / "sell"`` and ``buy_amount / sell_amount`` as
    before.
    """
    if raw_df is not None:
        df = raw_df.copy()
        logger.info(f"Cleaning LHB broker detail from provided DataFrame ({len(df)} rows)")
    else:
        default = _PROJECT_ROOT / "data" / "raw" / "lhb_broker_detail.parquet"
        p = raw_path or str(default)
        logger.info(f"Loading raw LHB broker detail from {p}")
        df = load_parquet(p)

    if df.empty:
        logger.warning("Empty raw LHB broker detail data")
        return df

    df = normalize_codes_in_df(df, code_col="stock_code", date_col="trade_date")

    for col in _BROKER_DETAIL_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "broker_name" in df.columns:
        df["broker_name"] = df["broker_name"].astype(str).str.strip()

    # Derive direction from flag
    if "flag" in df.columns:
        df["direction"] = df["flag"].map({"买入": "buy", "卖出": "sell"}).fillna("unknown")
    elif "direction" not in df.columns:
        df["direction"] = "unknown"

    # Derive sell_amount: implied = buy_amount − net_amount
    # For buy-side rows this gives the broker's net sell (usually small).
    # For sell-side rows this gives the broker's actual sale amount (large).
    if "buy_amount" in df.columns and "net_amount" in df.columns:
        inferred = pd.to_numeric(df["buy_amount"], errors="coerce") - \
                   pd.to_numeric(df["net_amount"], errors="coerce")
        if "sell_amount" in df:
            df["sell_amount"] = pd.to_numeric(df["sell_amount"], errors="coerce").combine_first(inferred)
        else:
            df["sell_amount"] = inferred

    if "rank" not in df.columns:
        df["rank"] = df.groupby(
            ["trade_date", "stock_code", "direction"]
        ).cumcount() + 1

    before = len(df)
    df = df.drop_duplicates(
        subset=["trade_date", "stock_code", "direction", "rank"],
        keep="first",
    )
    if (dropped := before - len(df)):
        logger.info(f"Dropped {dropped} duplicate broker detail rows")

    df = df.sort_values(
        ["trade_date", "stock_code", "direction", "rank"]
    ).reset_index(drop=True)
    df["data_source"] = data_source

    out_cols = [
        "trade_date", "stock_code", "direction", "rank",
        "broker_name", "buy_amount", "sell_amount", "net_amount",
        "buy_amount_ratio", "sell_amount_ratio",
        "data_source",
    ]
    out_cols += [c for c in ('broker_code','report_id','report_type','report_reason','window_days','source_version','disclosure_kind') if c in df]
    for col in out_cols:
        if col not in df.columns:
            df[col] = None
    df = df[out_cols]

    out_path = save_path or str(_PROJECT_ROOT / "data" / "clean" / "lhb_broker_detail.parquet")
    save_parquet(df, out_path)
    logger.info(f"Clean LHB broker detail saved: {out_path} ({len(df)} rows)")
    return df
