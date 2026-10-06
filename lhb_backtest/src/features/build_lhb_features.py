"""Build the LHB event feature table by joining clean data sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import load_parquet, save_parquet
from src.utils.logging import get_logger

logger = get_logger("build_features")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_lhb_event_table(
    clean_lhb_summary_path: str | None = None,
    clean_lhb_broker_path: str | None = None,
    clean_kline_path: str | None = None,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Build the ``factor_lhb_event_daily`` table.

    Joins:
    1. ``clean_lhb_summary`` (base: one row per stock-date event).
    2. ``clean_lhb_broker_detail`` (pivoted to wide format: buy1..buy5, sell1..sell5).
    3. ``clean_daily_kline`` (OHLCV fields for the event date).

    Parameters
    ----------
    clean_lhb_summary_path : str or None
        Path to clean LHB summary parquet.
    clean_lhb_broker_path : str or None
        Path to clean LHB broker detail parquet.
    clean_kline_path : str or None
        Path to clean daily kline parquet.
    save_path : str or None
        Output path. Default: ``data/factor/lhb_event_daily.parquet``.

    Returns
    -------
    pd.DataFrame
        The wide-format event feature table.
    """
    clean_dir = _PROJECT_ROOT / "data" / "clean"

    # Load clean tables
    lhb_summary = load_parquet(clean_lhb_summary_path or str(clean_dir / "lhb_summary.parquet"))
    logger.info(f"Loaded LHB summary: {len(lhb_summary)} rows")

    kline = load_parquet(clean_kline_path or str(clean_dir / "daily_kline.parquet"))
    logger.info(f"Loaded daily kline: {len(kline)} rows")

    # Load and pivot broker detail if available
    broker_path = clean_lhb_broker_path or str(clean_dir / "lhb_broker_detail.parquet")
    broker_wide = _pivot_broker_detail(broker_path)

    # Start from LHB summary as the base
    # Provider-supplied post-event returns are labels, never information
    # available when constructing the event's predictors.
    events = lhb_summary.drop(columns=['return_1d','return_2d','return_5d','return_10d','lhb_interpret'],errors='ignore').copy()
    if events.duplicated(['stock_code','trade_date']).any():
        raise ValueError('Duplicate stock/date keys in clean LHB summary')
    events['market_supported']=events.stock_code.str.match(r'^(?:60\d{4}\.SH|688\d{3}\.SH|00\d{4}\.SZ|30\d{4}\.SZ)$')

    # Join with pivoted broker detail
    if broker_wide is not None and not broker_wide.empty:
        events = events.merge(
            broker_wide,
            on=["stock_code", "trade_date"],
            how="left",
            suffixes=("", "_broker"),
            validate="one_to_one",
        )
        logger.info(f"After broker join: {len(events)} rows")

    # Join with daily kline (high_limit / low_limit enable exact limit-up
    # detection downstream instead of the pct_chg threshold approximation)
    kline_cols = [
        "trade_date", "stock_code", "open", "high", "low", "close",
        "pre_close", "pct_chg", "volume", "amount", "turnover_rate",
        "high_limit", "low_limit",
    ]
    kline_subset = kline[[c for c in kline_cols if c in kline.columns]].copy()

    # Avoid column conflicts with LHB summary
    merge_cols = ["trade_date", "stock_code"]
    kline_only_cols = [c for c in kline_subset.columns if c not in events.columns or c in merge_cols]
    kline_to_merge = kline_subset[kline_only_cols].copy()
    kline_to_merge['quote_available']=True

    events = events.merge(kline_to_merge, on=merge_cols, how="left", suffixes=("", "_kline"), validate="one_to_one")
    events['quote_available']=events.quote_available.eq(True)
    logger.info(f"After kline join: {len(events)} rows")

    # Source totals cover the union of disclosed buy and sell seats. Summing
    # just buy1..5 (or sell1..5) is not the same quantity. Keep absent totals
    # unknown rather than fabricate a smaller total from the wide table.
    for column in ("lhb_buy_total", "lhb_sell_total"):
        if column not in events:
            events[column] = float("nan")

    if "lhb_net_buy" not in events.columns or events["lhb_net_buy"].isna().all():
        if "lhb_buy_total" in events.columns and "lhb_sell_total" in events.columns:
            events["lhb_net_buy"] = events["lhb_buy_total"] - events["lhb_sell_total"]

    # Historical ST flag from SimTradeData stock_status (exact per-date
    # membership). Missing historical coverage remains unknown.
    events["is_st"] = _compute_historical_st(events)

    # Save
    out_path = save_path or str(_PROJECT_ROOT / "data" / "factor" / "lhb_event_daily.parquet")
    save_parquet(events, out_path)
    logger.info(f"Event feature table saved: {out_path} ({len(events)} rows)")

    return events


def _compute_historical_st(events: pd.DataFrame) -> pd.Series:
    """Per-event historical ST flag (1.0 / 0.0 / NaN).

    Uses the SimTradeData ``stock_status.parquet`` ST membership when
    available; current names cannot establish historical status.
    """
    from src.data_sources.simtradedata_metadata import load_status_lookup

    st_lookup = load_status_lookup("ST")
    if st_lookup is not None:
        flags = [
            float(code in st_lookup[date]) if date in st_lookup and supported else float("nan")
            for code, date, supported in zip(
                events["stock_code"].astype(str), events["trade_date"].astype(str),
                events.stock_code.str.match(r'^(?:60\d{4}\.SH|688\d{3}\.SH|00\d{4}\.SZ|30\d{4}\.SZ)$')
            )
        ]
        return pd.Series(flags, index=events.index)

    logger.warning("stock_status unavailable — historical is_st remains unknown")
    return pd.Series(float("nan"), index=events.index)


def _pivot_broker_detail(broker_path: str) -> pd.DataFrame | None:
    """Load broker detail and pivot to wide format.

    Creates columns: buy1_broker, buy1_amount, ..., sell5_broker, sell5_amount.
    """
    try:
        broker = load_parquet(broker_path)
    except FileNotFoundError:
        logger.warning("Could not load broker detail, skipping")
        return None

    if broker.empty:
        return None

    logger.info(f"Pivoting broker detail: {len(broker)} rows")

    keys = ["stock_code", "trade_date"]
    if broker[keys].isna().any().any():
        raise ValueError("Broker detail contains null event keys")
    groups = broker.groupby(keys, sort=True, observed=True)
    wide = groups.size().to_frame("_rows").drop(columns="_rows")
    metadata = {"report_reason": "seat_report_reason", "window_days": "seat_window_days",
                "disclosure_kind": "disclosure_kind"}
    for source, target in metadata.items():
        if source in broker:
            if groups[source].nunique().gt(1).any():
                raise ValueError(f"Mixed seat reports: {source}")
            wide[target] = groups[source].first()

    ranked = broker
    if "disclosure_kind" in wide:
        # Classifications overlap; their rows must never become brokerage seats.
        category_keys = wide.index[wide.disclosure_kind.eq("investor_category")]
        ranked = broker.loc[~pd.MultiIndex.from_frame(broker[keys]).isin(category_keys)]
    for side in ("buy", "sell"):
        part = ranked.loc[ranked.direction.eq(side)].copy()
        if part.empty:
            continue
        part["rank"] = pd.to_numeric(part["rank"], errors="coerce")
        if part["rank"].isna().any() or part.duplicated(keys + ["rank"]).any():
            raise ValueError(f"Invalid or duplicate {side} seat ranks")
        # Stable ordering preserves separate anonymous institutions with equal amounts.
        part = part.sort_values(keys + ["rank"], kind="stable")
        part["_position"] = part.groupby(keys, observed=True).cumcount() + 1
        part = part.loc[part._position.le(5)]
        part["amount"] = pd.to_numeric(part.get(f"{side}_amount"), errors="coerce")
        for column, default in (("broker_name", ""), ("broker_code", None)):
            if column not in part:
                part[column] = default
        pivot = part.pivot(index=keys, columns="_position",
                          values=["broker_name", "broker_code", "amount"])
        names = {"broker_name": "broker", "broker_code": "broker_code", "amount": "amount"}
        pivot.columns = [f"{side}{position}_{names[column]}" for column, position in pivot.columns]
        wide = wide.join(pivot, validate="one_to_one")
        amount_cols = [f"{side}{i}_amount" for i in range(1, 6) if f"{side}{i}_amount" in wide]
        wide[amount_cols] = wide[amount_cols].apply(pd.to_numeric, errors="coerce")
        if side == "buy":
            total = wide[amount_cols].sum(axis=1, min_count=1)
            wide["buy1_concentration"] = wide["buy1_amount"] / total.where(total.gt(0))

    wide = wide.reset_index()
    logger.info(f"Pivoted broker detail: {len(wide)} stock-date rows")
    return wide
