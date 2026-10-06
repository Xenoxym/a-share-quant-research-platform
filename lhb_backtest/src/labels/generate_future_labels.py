"""Generate forward-looking labels for LHB events.

For every event ``(stock_code, trade_date)`` the module attaches labels that
describe what happened to the stock over the *N* trading days **after** the
event.  Labels include future returns, maximum intraday upside / drawdown,
limit-up continuation flags, and directional indicators.

Two return bases are produced:

- ``future_return_Nd`` — research view, based on the event-day close.
  NOT directly tradable: LHB data is published after the close.
- ``entry_open_return_Nd`` — price-label view, based on the T+1 open (the
  earliest intended entry). N=1 violates T+1 and is diagnostic only. Combine with ``next_day_untradable`` to drop
  events that could not actually be bought (T+1 opened at limit-up, or the
  stock was suspended on T+1).

Limit-up detection uses the exact ``high_limit`` price when available
(SimTradeData pre-computes it) and returns unknown when reliable limit prices are missing.

When the SimTradeData benchmark table is available, excess-return labels
``alpha_Nd = future_return_Nd - benchmark_return_Nd`` are added.

Forward bars are gathered only for event rows to keep full-history memory bounded.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.io import load_parquet, save_parquet
from src.utils.logging import get_logger

logger = get_logger("generate_future_labels")

_DEFAULT_EVENT_PATH = "data/factor/lhb_event_factors.parquet"
_DEFAULT_KLINE_PATH = "data/clean/daily_kline.parquet"
_DEFAULT_SAVE_PATH = "data/factor/lhb_event_labeled.parquet"
_DEFAULT_HORIZONS = [1, 2, 3, 5, 10]

# Price tolerance when comparing close against the exact limit price
# (guards against float representation of two-decimal prices).
_LIMIT_EPS = 1e-4


# ── Internal helpers ─────────────────────────────────────────────────


def _limit_up_flag(
    close: pd.Series,
    high_limit: pd.Series,
    pct_chg: pd.Series,
    pct_threshold: pd.Series,
) -> np.ndarray:
    """Per-day limit-up flag (1.0 / 0.0 / NaN).

    Uses exact price equality ``abs(close - high_limit) <= 1e-4`` where the limit price
    is known; returns NaN for unknown or internally inconsistent limits.
    """
    close = pd.to_numeric(close, errors="coerce")
    high_limit = pd.to_numeric(high_limit, errors="coerce")
    pct_chg = pd.to_numeric(pct_chg, errors="coerce")

    exact_known = high_limit.gt(0) & close.gt(0) & close.le(high_limit + _LIMIT_EPS)
    exact_hit = (close - high_limit).abs() <= _LIMIT_EPS

    return np.where(exact_known, exact_hit.astype(float), np.nan)


def _event_forward_data(kline: pd.DataFrame, events: pd.DataFrame, max_horizon: int) -> pd.DataFrame:
    """Gather only event rows; memory scales with events × horizons, not all bars."""
    keys = ["stock_code", "trade_date"]
    k = kline.sort_values(keys).reset_index(drop=True)
    base = events[keys].drop_duplicates().reset_index(drop=True)
    fields = {}
    if k.empty:
        positions = np.full(len(base), -1, dtype=int)
        codes = np.array([], dtype=int)
    else:
        index = pd.MultiIndex.from_frame(k[keys])
        positions = index.get_indexer(pd.MultiIndex.from_frame(base[keys]))
        codes = k.stock_code.astype("category").cat.codes.to_numpy()

    # Address bars by market date, not row count. A suspension/missing quote
    # must not move every subsequent horizon, or turn an old bar into T-1.
    from src.utils.calendar import get_trading_dates
    dated_positions = {0: positions}
    if not k.empty and not base.empty:
        start = min(k.trade_date.min(), base.trade_date.min())
        end = max(k.trade_date.max(), base.trade_date.max())
        calendar = get_trading_dates(start, end)
        for offset in [-1, *range(1, max_horizon + 1)]:
            if offset == -1:
                mapping = dict(zip(calendar[1:], calendar[:-1]))
            else:
                mapping = dict(zip(calendar[:-offset], calendar[offset:]))
            target_keys = base.assign(trade_date=base.trade_date.map(mapping))
            dated_positions[offset] = index.get_indexer(pd.MultiIndex.from_frame(target_keys))

    def take(column: str, offset: int) -> pd.Series:
        if k.empty or column not in k:
            return pd.Series(np.nan, index=base.index)
        targets = dated_positions.get(offset, np.full(len(base), -1, dtype=int))
        safe = targets.clip(0, len(k) - 1)
        valid = ((positions >= 0) & (targets >= 0) & (targets < len(k))
                 & (codes[safe] == codes[positions.clip(0, len(k) - 1)]))
        return pd.Series(k[column].to_numpy()[safe], index=base.index).where(valid)

    fields["close"] = take("close", 0)
    for name in ("pct_chg", "close", "high_limit"):
        fields[f"_{name}_prev"] = take(name, -1)
    fields["_high_limit_0"] = take("high_limit", 0)
    for i in range(1, max_horizon + 1):
        for name in ("close", "high", "low", "pct_chg", "open", "high_limit", "low_limit", "volume"):
            fields[f"_{name}_fwd_{i}"] = take(name, i)
        fields[f"_date_fwd_{i}"] = take("trade_date", i)
    return pd.concat([base, pd.DataFrame(fields)], axis=1)


def _merge_benchmark_returns(
    result: pd.DataFrame,
    horizons: list[int],
) -> pd.DataFrame:
    """Attach ``_bench_ret_{n}`` columns (benchmark close-to-close returns).

    Silently returns *result* unchanged when the benchmark table is missing.
    """
    from src.data_sources.simtradedata_metadata import load_benchmark

    bench = load_benchmark()
    if bench is None or bench.empty:
        logger.warning("Benchmark unavailable — alpha labels skipped")
        return result

    if bench["trade_date"].duplicated().any():
        raise ValueError("Duplicate benchmark trading dates")
    bench = bench.set_index("trade_date")
    result = result.copy()
    event_close = result["trade_date"].map(bench["close"])
    entry_open = result["label_end_date_1d"].map(bench["open"]) if "open" in bench else None
    for n in horizons:
        exit_close = result[f"label_end_date_{n}d"].map(bench["close"])
        result[f"_bench_ret_{n}"] = exit_close / event_close - 1.0
        if entry_open is not None:
            result[f"_bench_entry_ret_{n}"] = exit_close / entry_open - 1.0
    if entry_open is not None:
        result["_bench_oo_ret"] = result["label_end_date_2d"].map(bench["open"]) / entry_open - 1.0
    return result


def _next_trading_day_map(dates: pd.Series) -> dict[str, str]:
    """Map each event date to the next trading day (calendar-based)."""
    from src.utils.calendar import get_trading_dates

    uniq = sorted(dates.dropna().astype(str).unique())
    if not uniq:
        return {}
    # Extend well past the last event date to find its next trading day.
    end = (pd.Timestamp(uniq[-1]) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    all_days = get_trading_dates(uniq[0], end)
    nxt: dict[str, str] = {}
    idx = 0
    for d in uniq:
        while idx < len(all_days) and all_days[idx] <= d:
            idx += 1
        if idx < len(all_days):
            nxt[d] = all_days[idx]
    return nxt


# ── Public API ───────────────────────────────────────────────────────


def generate_future_labels(
    event_df: Optional[pd.DataFrame] = None,
    event_path: Optional[str] = None,
    kline_df: Optional[pd.DataFrame] = None,
    kline_path: Optional[str] = None,
    horizons: Optional[list[int]] = None,
    save_path: Optional[str] = None,
    include_alpha: bool = True,
    include_suspension: bool = True,
) -> pd.DataFrame:
    """Generate forward-looking labels for each LHB event.

    Parameters
    ----------
    event_df : pd.DataFrame | None
        Pre-loaded event table.  Takes precedence over *event_path*.
    event_path : str | None
        Path to the event factors parquet.
    kline_df : pd.DataFrame | None
        Pre-loaded daily kline data.
    kline_path : str | None
        Path to the clean daily kline parquet.
    horizons : list[int] | None
        Forward horizons in trading days.  Defaults to ``[1, 2, 3, 5, 10]``.
    save_path : str | None
        Output parquet path.
    include_alpha : bool
        Attach ``alpha_Nd`` excess-return labels when the SimTradeData
        benchmark table is available.
    include_suspension : bool
        Attach ``suspended_next_day`` from SimTradeData stock_status
        (HALT) when available; also feeds ``next_day_untradable``.

    Returns
    -------
    pd.DataFrame
        The event table with all label columns appended.  Events with
        insufficient future kline data receive ``NaN`` labels.
    """
    # ── Load inputs ─────────────────────────────────────────────
    if event_df is not None:
        events = event_df.copy()
    else:
        events = load_parquet(event_path or _DEFAULT_EVENT_PATH)

    if kline_df is not None:
        kline = kline_df.copy()
    else:
        kline = load_parquet(kline_path or _DEFAULT_KLINE_PATH)

    if horizons is None:
        horizons = list(_DEFAULT_HORIZONS)

    if not horizons or any(not isinstance(n, int) or n < 1 for n in horizons):
        raise ValueError("horizons must contain positive integers")
    max_horizon = max(2, max(horizons))
    logger.info(
        "Generating labels for {:,} events, horizons={}, max={}",
        len(events), horizons, max_horizon,
    )

    # ── Normalise dates ─────────────────────────────────────────
    events["trade_date"] = pd.to_datetime(events["trade_date"], format="mixed").dt.strftime("%Y-%m-%d")
    kline["trade_date"] = pd.to_datetime(kline["trade_date"], format="mixed").dt.strftime("%Y-%m-%d")
    for name, frame in (("events", events), ("K-lines", kline)):
        keys = ["stock_code", "trade_date"]
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"Null or duplicate stock/date {name}; clean input before generating labels")

    # Ensure numeric types on key kline columns
    numeric_kline_cols = ["close", "high", "low", "pct_chg", "open"]
    if "high_limit" in kline.columns:
        numeric_kline_cols.append("high_limit")
    for col in numeric_kline_cols:
        if col in kline.columns:
            kline[col] = pd.to_numeric(kline[col], errors="coerce")

    # ── Filter kline to event stocks only (memory optimisation) ─
    event_stocks = set(events["stock_code"].unique())
    kline = kline[kline["stock_code"].isin(event_stocks)].copy()
    logger.info("Filtered kline to {:,} rows for {:,} stocks", len(kline), len(event_stocks))

    # Gather the future values only for events, avoiding multi-GB wide K-lines.
    kline_merge = _event_forward_data(kline, events, max_horizon)

    result = events.merge(
        kline_merge.rename(columns={"close": "_event_close"}),
        on=["stock_code", "trade_date"],
        how="left",
    )

    # Verify that row shifts really mean market trading days. A missing row
    # must not quietly change an N-day label into an N-quotation label.
    from src.utils.calendar import get_trading_dates
    for i in range(1, max_horizon + 1):
        result[f"label_end_date_{i}d"] = result[f"_date_fwd_{i}"]
    result = result.copy()  # consolidate gathered blocks before adding labels
    result["label_schema_version"] = 2

    event_close = pd.to_numeric(result["_event_close"], errors="coerce")
    safe_close = event_close.replace(0, np.nan)

    # ── (a–c) Close-based return labels per horizon ─────────────
    for n in horizons:
        close_n = result[f"_close_fwd_{n}"]

        # a) future_return_Nd
        result[f"future_return_{n}d"] = (close_n - event_close) / safe_close

        # b) future_max_return_Nd — best intraday high over 1..N
        high_cols = [f"_high_fwd_{i}" for i in range(1, n + 1)]
        max_high = result[high_cols].max(axis=1).where(result[high_cols].notna().all(axis=1))
        result[f"future_max_return_{n}d"] = (max_high - event_close) / safe_close

        # c) future_max_drawdown_Nd — worst intraday low over 1..N
        low_cols = [f"_low_fwd_{i}" for i in range(1, n + 1)]
        min_low = result[low_cols].min(axis=1).where(result[low_cols].notna().all(axis=1))
        result[f"future_max_drawdown_{n}d"] = (min_low - event_close) / safe_close

    # ── Tradable entry labels (T+1 open basis) ──────────────────
    open_1 = pd.to_numeric(result["_open_fwd_1"], errors="coerce")
    safe_open_1 = open_1.replace(0, np.nan)

    # Opening gap: T+1 open vs event close
    result["entry_open_gap"] = (open_1 - event_close) / safe_close

    for n in horizons:
        close_n = result[f"_close_fwd_{n}"]
        result[f"entry_open_return_{n}d"] = (close_n - open_1) / safe_open_1

    # Minimum LEGAL holding under the A-share T+1 rule: shares bought at the
    # T+1 open cannot be sold before T+2, so entry_open_return_1d (exit at
    # T+1 close) is NOT executable — kept for reference only.  The shortest
    # tradable round trip is buy T+1 open → sell T+2 open:
    if max_horizon >= 2:
        open_2 = pd.to_numeric(result["_open_fwd_2"], errors="coerce")
        result["entry_open_exit_open_1d"] = (open_2 - open_1) / safe_open_1
    else:
        result["entry_open_exit_open_1d"] = np.nan

    # T+1 opened at (or above) the limit price → cannot fill a buy order.
    hl_1 = pd.to_numeric(result["_high_limit_fwd_1"], errors="coerce")
    open_at_limit = np.where(
        open_1.isna(), np.nan,
        np.where(hl_1.notna(), (open_1 >= hl_1 - _LIMIT_EPS).astype(float), np.nan),
    )
    result["next_day_open_at_limit"] = open_at_limit

    # ── Suspension (HALT) on the next trading day ───────────────
    suspended_next = pd.Series(np.nan, index=result.index)
    if include_suspension:
        from src.data_sources.simtradedata_metadata import load_status_lookup

        halt_lookup = load_status_lookup("HALT")
        if halt_lookup is not None:
            next_day = result["trade_date"].map(_next_trading_day_map(result["trade_date"]))
            pairs = zip(result["stock_code"].astype(str), next_day)
            suspended_next = pd.Series(
                [
                    float(code in halt_lookup[nd]) if nd in halt_lookup else np.nan
                    for code, nd in pairs
                ],
                index=result.index,
            )
    result["suspended_next_day"] = suspended_next

    # Untradable = T+1 kline missing, or T+1 opened at limit, or T+1 halted
    blocked = (open_1.isna() | open_1.le(0) | result["_volume_fwd_1"].le(0)
               | result["next_day_open_at_limit"].eq(1)
               | result["suspended_next_day"].eq(1))
    known = hl_1.notna() & result["_volume_fwd_1"].notna()
    if include_suspension:
        known &= result["suspended_next_day"].notna()
    result["next_day_untradable"] = np.where(blocked, 1.0, np.where(known, 0.0, np.nan))

    # ── Limit-up flags (exact price when available) ─────────────
    threshold = pd.Series(np.nan, index=result.index)  # no percentage fallback

    event_pct = pd.to_numeric(result.get("pct_chg"), errors="coerce")
    prev_pct = pd.to_numeric(result.get("_pct_chg_prev"), errors="coerce")

    event_lu = _limit_up_flag(event_close, result["_high_limit_0"], event_pct, threshold)
    prev_lu = _limit_up_flag(
        result["_close_prev"], result["_high_limit_prev"], prev_pct, threshold,
    )
    result["event_day_limit_up"] = event_lu
    # 首板 requires a known non-limit-up previous bar; unknown remains unknown.
    result["is_first_limit_up_board"] = np.where(
        np.isnan(event_lu) | np.isnan(prev_lu), np.nan,
        ((event_lu == 1.0) & (prev_lu == 0.0)).astype(float),
    )

    # Per-day limit-up flags (NaN-safe)
    for i in range(1, max_horizon + 1):
        result[f"_lu_d{i}"] = _limit_up_flag(
            result[f"_close_fwd_{i}"],
            result[f"_high_limit_fwd_{i}"],
            result[f"_pct_chg_fwd_{i}"],
            threshold,
        )

    # d) next_day_limit_up
    result["next_day_limit_up"] = result["_lu_d1"]

    # e) within_Nd_limit_up  (N ∈ {2, 3, 5})
    for n in (2, 3, 5):
        if n > max_horizon:
            continue
        lu_cols = [f"_lu_d{i}" for i in range(1, n + 1)]
        lu_vals = result[lu_cols]
        any_lu = (lu_vals == 1.0).any(axis=1)
        any_na = lu_vals.isna().any(axis=1)
        result[f"within_{n}d_limit_up"] = np.where(
            any_lu, 1.0, np.where(any_na, np.nan, 0.0),
        )

    # f) next_day_positive
    ret1 = result.get("future_return_1d")
    if ret1 is not None:
        result["next_day_positive"] = np.where(ret1.isna(), np.nan, (ret1 > 0).astype(float))

    # g) within_Nd_positive  (N ∈ {3, 5})
    for n in (3, 5):
        col = f"future_return_{n}d"
        if col in result.columns:
            ret_n = result[col]
            result[f"within_{n}d_positive"] = np.where(
                ret_n.isna(), np.nan, (ret_n > 0).astype(float),
            )

    # h) continue_board_1d
    result["continue_board_1d"] = result["_lu_d1"]

    # i) continue_board_2d
    if max_horizon >= 2:
        lu1 = result["_lu_d1"]
        lu2 = result["_lu_d2"]
        result["continue_board_2d"] = np.where(
            (lu1 == 0) | (lu2 == 0), 0.0,
            np.where((lu1 == 1) & (lu2 == 1), 1.0, np.nan),
        )

    # j) continue_board_3d
    if max_horizon >= 3:
        lu3 = result["_lu_d3"]
        result["continue_board_3d"] = np.where(
            (lu1 == 0) | (lu2 == 0) | (lu3 == 0), 0.0,
            np.where((lu1 == 1) & (lu2 == 1) & (lu3 == 1), 1.0, np.nan),
        )

    # k) max_continue_board_days  (vectorised running count)
    count = pd.Series(0, index=result.index, dtype=int)
    active = pd.Series(True, index=result.index)
    censored = pd.Series(False, index=result.index)
    for i in range(1, max_horizon + 1):
        col = f"_lu_d{i}"
        if col not in result.columns:
            break
        day_lu = (result[col] == 1.0) & result[col].notna()
        censored |= active & result[col].isna()
        active = active & day_lu
        count = count + active.astype(int)

    d1_exists = result["_lu_d1"].notna()
    result["observed_continue_board_days"] = np.where(d1_exists, count, np.nan)
    result["continue_board_censored"] = censored
    result["max_continue_board_days"] = np.where(d1_exists & ~censored, count, np.nan)

    # ── Benchmark excess returns (alpha) ────────────────────────
    if include_alpha:
        result = _merge_benchmark_returns(result, horizons).copy()
        alpha_cols = {}
        for n in horizons:
            bench_col = f"_bench_ret_{n}"
            if bench_col in result.columns:
                alpha_cols[f"alpha_{n}d"] = result[f"future_return_{n}d"] - result[bench_col]
                alpha_cols[f"research_close_alpha_{n}d"] = alpha_cols[f"alpha_{n}d"]
            entry_bench = f"_bench_entry_ret_{n}"
            if n >= 2 and entry_bench in result:
                alpha_cols[f"entry_alpha_{n}d"] = result[f"entry_open_return_{n}d"] - result[entry_bench]
        if "_bench_oo_ret" in result:
            alpha_cols["entry_oo_alpha_1d"] = result["entry_open_exit_open_1d"] - result["_bench_oo_ret"]
        if alpha_cols:
            result = pd.concat([result, pd.DataFrame(alpha_cols, index=result.index)], axis=1)

    # ── Clean up temporary columns ──────────────────────────────
    tmp_cols = [c for c in result.columns if c.startswith("_")]
    result = result.drop(columns=tmp_cols)

    # ── Save ────────────────────────────────────────────────────
    # Only persist when the caller passed an explicit save_path or is running
    # the pipeline (loading from disk). In-memory calls (unit tests, ad-hoc
    # research) must NOT silently overwrite the production labelled parquet.
    pipeline_mode = event_df is None or kline_df is None
    if save_path is not None or pipeline_mode:
        out_path = save_path or _DEFAULT_SAVE_PATH
        save_parquet(result, out_path)
        logger.info(
            "Saved labelled events → {} ({:,} rows × {} cols)",
            out_path, len(result), len(result.columns),
        )
    return result
