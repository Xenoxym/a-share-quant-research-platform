"""Compute derived LHB factor columns on the event table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import load_parquet, save_parquet
from src.utils.logging import get_logger

logger = get_logger("compute_factors")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that returns NaN when the denominator is zero."""
    return numerator / denominator.replace(0, np.nan)


def _detect_window_days(reason: pd.Series) -> pd.Series:
    """LHB reporting window in trading days, derived from 上榜原因 text.

    "连续三个交易日…" reasons report seat amounts cumulated over 3 days while
    the summary numeric fields may be single-day values — mixing the two in
    one ratio produces nonsense (observed buy1_amount 30x lhb_buy_total).
    Use the selected report's reason, not the concatenation of all reports.
    Unknown exceptional periods are represented by zero.
    """
    from src.data_sources.reporting_windows import reporting_window_days
    return reason.map(reporting_window_days).to_numpy()


def compute_lhb_factors(
    event_df: pd.DataFrame | None = None,
    event_path: str | None = None,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Compute derived factor columns on the LHB event table.

    Parameters
    ----------
    event_df : pd.DataFrame or None
        Pre-loaded event DataFrame. Takes precedence over *event_path*.
    event_path : str or None
        Path to the event parquet file.
        Default: ``data/factor/lhb_event_daily.parquet``.
    save_path : str or None
        If provided, save the result to this parquet path.
        Default: ``data/factor/lhb_event_factors.parquet``.

    Returns
    -------
    pd.DataFrame
        Event DataFrame with new factor columns appended.
    """
    if event_df is not None:
        df = event_df.copy()
    else:
        p = event_path or str(_PROJECT_ROOT / "data" / "factor" / "lhb_event_daily.parquet")
        logger.info(f"Loading event table from {p}")
        df = load_parquet(p)

    if df.empty:
        logger.warning("Empty event table, no factors to compute")
        _add_empty_factor_cols(df)
        return df

    logger.info(f"Computing factors for {len(df)} events")

    # 0. Reporting window (1 = single-day board, 3 = 三日榜 or mixed).
    #    Factors that mix seat amounts (window-cumulated) with single-day
    #    summary/kline amounts are only valid for window == 1.
    reason = df.get('summary_selected_reason',df.get('lhb_reason',pd.Series('',index=df.index)))
    df["lhb_window_days"] = _detect_window_days(reason)
    single_day = df["lhb_window_days"] == 1
    if 'seat_window_days' in df:
        single_day &= df.seat_window_days.eq(1)
    if 'disclosure_kind' in df:
        single_day &= df.disclosure_kind.eq('ordinary')

    # 1. buy_sell_ratio = lhb_buy_total / lhb_sell_total
    #    (both sides come from the summary → same window, always valid)
    df["buy_sell_ratio"] = _safe_divide(
        df.get("lhb_buy_total", pd.Series(dtype=float)),
        df.get("lhb_sell_total", pd.Series(dtype=float)),
    )

    # 2. buy1_concentration = buy1_amount / Σ(buy1..5_amount)
    #    Seat-internal ratio — all seats share the same reporting window, so
    #    this stays valid for 3-day boards too. (Step 5a computes the same
    #    definition; recomputed here so the factor exists even when the
    #    pivot step was skipped.)
    buy_cols = [f"buy{i}_amount" for i in range(1, 6)]
    existing_buy = [c for c in buy_cols if c in df.columns]
    if existing_buy:
        buy_sum = df[existing_buy].sum(axis=1, skipna=True)
        df["buy1_concentration"] = _safe_divide(
            df.get("buy1_amount", pd.Series(dtype=float)), buy_sum,
        )
    else:
        df["buy1_concentration"] = np.nan

    # 2b. buy1_to_lhb_buy_ratio = buy1_amount / lhb_buy_total
    #     Mixes seat amount with summary amount → single-day boards only.
    df["buy1_to_lhb_buy_ratio"] = _safe_divide(
        df.get("buy1_amount", pd.Series(dtype=float)),
        df.get("lhb_buy_total", pd.Series(dtype=float)),
    ).where(single_day)

    # 3. lhb_turnover_ratio = lhb_turnover / amount  (fallback when API value
    #    missing; single-day only — lhb_turnover is window-cumulated)
    if "lhb_turnover_ratio" not in df.columns or df["lhb_turnover_ratio"].isna().all():
        df["lhb_turnover_ratio"] = _safe_divide(
            df.get("lhb_turnover", pd.Series(dtype=float)),
            df.get("amount", pd.Series(dtype=float)),
        ).where(single_day)

    # 4. net_buy_ratio = lhb_net_buy / amount  (fallback, single-day only)
    if "net_buy_ratio" not in df.columns or df["net_buy_ratio"].isna().all():
        df["net_buy_ratio"] = _safe_divide(
            df.get("lhb_net_buy", pd.Series(dtype=float)),
            df.get("amount", pd.Series(dtype=float)),
        ).where(single_day)

    # 5. net_buy_float_mcap_ratio = lhb_net_buy / float_market_cap
    #    (market cap is a stock variable, ratio stays interpretable)
    df["net_buy_float_mcap_ratio"] = _safe_divide(
        df.get("lhb_net_buy", pd.Series(dtype=float)),
        df.get("float_market_cap", pd.Series(dtype=float)),
    )

    # 6. buy1_to_daily_amount_ratio = buy1_amount / amount
    #    (seat vs single-day kline amount → single-day boards only)
    df["buy1_to_daily_amount_ratio"] = _safe_divide(
        df.get("buy1_amount", pd.Series(dtype=float)),
        df.get("amount", pd.Series(dtype=float)),
    ).where(single_day)

    # 7. sell1_concentration = sell1_amount / Σ(sell1..5_amount)
    #    (seat-internal, mirrors buy1_concentration)
    sell_cols_c = [f"sell{i}_amount" for i in range(1, 6)]
    existing_sell_c = [c for c in sell_cols_c if c in df.columns]
    if existing_sell_c:
        sell_sum = df[existing_sell_c].sum(axis=1, skipna=True)
        df["sell1_concentration"] = _safe_divide(
            df.get("sell1_amount", pd.Series(dtype=float)), sell_sum,
        )
    else:
        df["sell1_concentration"] = np.nan

    # 8. top5_buy_avg = mean of buy1..buy5 amounts
    buy_cols = [f"buy{i}_amount" for i in range(1, 6)]
    existing_buy = [c for c in buy_cols if c in df.columns]
    if existing_buy:
        df["top5_buy_avg"] = df[existing_buy].mean(axis=1, skipna=True)
    else:
        df["top5_buy_avg"] = np.nan

    # 9. top5_sell_avg = mean of sell1..sell5 amounts
    sell_cols = [f"sell{i}_amount" for i in range(1, 6)]
    existing_sell = [c for c in sell_cols if c in df.columns]
    if existing_sell:
        df["top5_sell_avg"] = df[existing_sell].mean(axis=1, skipna=True)
    else:
        df["top5_sell_avg"] = np.nan

    # 10. Seat-quality counts from broker names (机构专用 / 沪深股通).
    #     NaN when broker detail is missing for the event (not zero — absence
    #     of data must not be confused with absence of institutions).
    _add_seat_quality_counts(df)

    # A percentage threshold inferred from today's stock name encodes future
    # ST changes. Executability uses validated dated price limits instead.
    df = df.drop(columns=['limit_up_threshold'],errors='ignore')

    logger.info("Factor computation complete")

    if save_path is not None or event_path is not None or event_df is None:
        out = save_path or str(_PROJECT_ROOT / "data" / "factor" / "lhb_event_factors.parquet")
        save_parquet(df, out)
        logger.info(f"Factors saved to {out}")

    return df


def _add_seat_quality_counts(df: pd.DataFrame) -> None:
    """Count institutional (机构专用) and northbound (沪/深股通) seats.

    Adds:
    - ``inst_buy_count``  / ``inst_sell_count``  — 机构专用 seats in top-5
    - ``northbound_buy``  — 1.0 when a 沪股通/深股通 seat is among top-5 buyers
    - ``inst_net_seats``  — inst_buy_count − inst_sell_count

    Values are NaN for events without broker detail.
    """
    buy_name_cols = [c for c in (f"buy{i}_broker" for i in range(1, 6)) if c in df.columns]
    sell_name_cols = [c for c in (f"sell{i}_broker" for i in range(1, 6)) if c in df.columns]

    def _count(cols: list[str], pattern: str) -> pd.Series:
        if not cols:
            return pd.Series(np.nan, index=df.index)
        hits = pd.DataFrame({
            c: df[c].astype(str).str.contains(pattern, na=False) for c in cols
        })
        has_any_data = pd.DataFrame({c: df[c].notna() & (df[c].astype(str) != "") for c in cols}).any(axis=1)
        return hits.sum(axis=1).astype(float).where(has_any_data)

    df["inst_buy_count"] = _count(buy_name_cols, "机构专用")
    df["inst_sell_count"] = _count(sell_name_cols, "机构专用")
    nb = _count(buy_name_cols, "股通")
    df["northbound_buy"] = (nb > 0).astype(float).where(nb.notna())
    df["inst_net_seats"] = df["inst_buy_count"] - df["inst_sell_count"]


def _add_empty_factor_cols(df: pd.DataFrame) -> None:
    """Add empty factor columns to an empty DataFrame for schema consistency."""
    factor_cols = [
        "buy_sell_ratio", "buy1_concentration", "buy1_to_lhb_buy_ratio",
        "lhb_turnover_ratio",
        "net_buy_ratio", "net_buy_float_mcap_ratio", "buy1_to_daily_amount_ratio",
        "sell1_concentration", "top5_buy_avg", "top5_sell_avg",
        "lhb_window_days",
    ]
    for col in factor_cols:
        if col not in df.columns:
            df[col] = pd.Series(dtype=float)
