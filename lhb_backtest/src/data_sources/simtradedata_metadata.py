"""Loaders for SimTradeData metadata/ tables.

Provides:
- Historical ST membership lookup (stock_status.parquet, status_type="ST")
- Historical suspension lookup (stock_status.parquet, status_type="HALT")
- Benchmark index daily closes (benchmark.parquet)

All lookups are keyed by "YYYY-MM-DD" date strings and ".SH/.SZ/.BJ"
normalised stock codes (SimTradeData stores Shanghai codes as ".SS").
"""

from __future__ import annotations

from pathlib import Path
import os
import json

import pandas as pd
import yaml

from src.utils.logging import get_logger

logger = get_logger("simtradedata_metadata")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_export_dir(export_dir: str | Path | None = None) -> Path | None:
    """Resolve directory (arg > isolated-run environment override > config)."""
    if export_dir:
        return Path(export_dir)
    if os.environ.get('LHB_SIMTRADEDATA_EXPORT_DIR'):
        return Path(os.environ['LHB_SIMTRADEDATA_EXPORT_DIR'])
    config_path = _PROJECT_ROOT / "config" / "config.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        rel = cfg.get("simtradedata", {}).get("export_dir", "")
        if not rel:
            return None
        p = Path(rel)
        if not p.is_absolute():
            p = (_PROJECT_ROOT / p).resolve()
        return p
    except Exception:
        return None


def _normalize_date_series(s: pd.Series) -> pd.Series:
    """Normalise dates like '20150105' or timestamps to 'YYYY-MM-DD'."""
    return pd.to_datetime(s.astype(str), errors="coerce").dt.strftime("%Y-%m-%d")


def _normalize_symbol(symbol: str) -> str:
    """'600072.SS' → '600072.SH'; other suffixes unchanged."""
    s = str(symbol).strip()
    return s[:-3] + ".SH" if s.endswith(".SS") else s


def load_status_lookup(
    status_type: str,
    export_dir: str | Path | None = None,
) -> dict[str, set[str]] | None:
    """Load stock_status.parquet into ``{date: {stock_code, ...}}`` for one status.

    Parameters
    ----------
    status_type : str
        ``"ST"`` (special treatment) or ``"HALT"`` (suspended that day).
    export_dir : str, Path, or None
        SimTradeData export/cn directory. Resolved from config when None.

    Returns
    -------
    dict or None
        None when the metadata file is unavailable (callers should fall back
        to name-based heuristics).
    """
    root = resolve_export_dir(export_dir)
    if root is None:
        return None
    path = root / "metadata" / "stock_status.parquet"
    if not path.exists():
        logger.warning(f"stock_status.parquet not found: {path}")
        return None

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning(f"Failed to read stock_status.parquet: {exc}")
        return None

    df = df[df["status_type"] == status_type]
    dates = _normalize_date_series(df["date"])
    state_path=root/'.source_state.json'
    if state_path.exists():
        state=json.loads(state_path.read_text(encoding='utf-8'))
        if state.get('status_policy')=='dated-flags-v1':
            verified=dates.between(state['status_history_start'],state['status_history_end'])
            df=df.loc[verified]
            dates=dates.loc[verified]

    lookup: dict[str, set[str]] = {}
    for date, symbols in zip(dates, df["symbols"]):
        if pd.isna(date):
            continue
        lookup[date] = {_normalize_symbol(s) for s in symbols}

    logger.info(
        f"Loaded {status_type} status lookup: {len(lookup)} dates "
        f"({sum(len(v) for v in lookup.values()):,} memberships)"
    )
    return lookup


def load_benchmark(export_dir: str | Path | None = None) -> pd.DataFrame | None:
    """Load benchmark.parquet → DataFrame(trade_date, close) sorted by date.

    Returns None when unavailable.
    """
    root = resolve_export_dir(export_dir)
    if root is None:
        return None
    path = root / "metadata" / "benchmark.parquet"
    if not path.exists():
        logger.warning(f"benchmark.parquet not found: {path}")
        return None

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        logger.warning(f"Failed to read benchmark.parquet: {exc}")
        return None

    out = pd.DataFrame({
        "trade_date": _normalize_date_series(df["date"]),
        "close": pd.to_numeric(df["close"], errors="coerce"),
    }).dropna()
    if "open" in df:
        out["open"] = pd.to_numeric(df.loc[out.index, "open"], errors="coerce")
    if out.empty:
        return None
    out = out.sort_values("trade_date").reset_index(drop=True)
    logger.info(
        f"Loaded benchmark: {len(out)} days "
        f"[{out['trade_date'].iloc[0]} ~ {out['trade_date'].iloc[-1]}]"
    )
    return out
