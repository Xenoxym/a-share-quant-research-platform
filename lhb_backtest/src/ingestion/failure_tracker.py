"""Retry utility and failure-tracking CSV for LHB ingestion steps (Step 2/3).

Design
------
- with_retry(): wrap any callable with 3 attempts and 1s/2s/4s back-off.
  No fallback to any alternative source — failures are recorded and surfaced.

- FailureTracker: append-then-upsert failure CSV per ingestion step.
  Primary keys: trade_date (summary) or (trade_date, stock_code) (detail).
  Status values: "failed" | "resolved".

CSV layout (both files)
-----------------------
    trade_date, [stock_code,] status, error_msg, retry_count, last_attempt_at

Restore flow
------------
    pending = tracker.get_pending()
    for each row: re-fetch → if success: tracker.resolve(key) else tracker.record(key, err)
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("failure_tracker")

# Default wait schedule between retry attempts (seconds)
DEFAULT_WAITS: tuple[float, ...] = (1.0, 2.0, 4.0)

# Truncate error messages stored to CSV to keep files small
_MAX_ERR_LEN = 400


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

def with_retry(
    fn: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    waits: Sequence[float] = DEFAULT_WAITS,
    **kwargs: Any,
) -> Any:
    """Call ``fn(*args, **kwargs)`` with up to ``max_retries`` attempts.

    Parameters
    ----------
    fn : callable
        The function to call.
    *args :
        Positional arguments forwarded to fn.
    max_retries : int
        Total number of attempts (first call + retries). Default 3.
    waits : sequence of float
        Sleep durations **between** attempts. ``waits[i]`` is used before
        attempt ``i+2`` (i.e. before the 2nd, 3rd, … tries).
        Defaults to ``(1, 2, 4)`` seconds.
    **kwargs :
        Keyword arguments forwarded to fn.

    Returns
    -------
    Any
        Return value of fn on success.

    Raises
    ------
    Exception
        The last exception raised after all attempts are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = float(waits[attempt - 1]) if attempt - 1 < len(waits) else float(waits[-1])
                logger.debug(
                    f"Attempt {attempt}/{max_retries} failed ({type(exc).__name__}: {exc}). "
                    f"Retrying in {wait:.0f}s…"
                )
                time.sleep(wait)
            else:
                logger.debug(f"All {max_retries} attempts exhausted: {exc}")

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FailureTracker
# ---------------------------------------------------------------------------

class FailureTracker:
    """Track failed fetch tasks in a CSV file with upsert semantics.

    Parameters
    ----------
    csv_path : str or Path
        Path to the failure CSV (created automatically if absent).
    primary_keys : list[str]
        Columns that together form the unique record key.
        Summary: ``["trade_date"]``
        Detail:  ``["trade_date", "stock_code"]``
    """

    _FIXED_COLS = ["status", "error_msg", "retry_count", "last_attempt_at"]

    def __init__(self, csv_path: str | Path, primary_keys: list[str]) -> None:
        self._path = Path(csv_path)
        self._pk = primary_keys
        self._all_cols = primary_keys + self._FIXED_COLS
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, key: dict[str, str], error_msg: str) -> None:
        """Upsert a failure record (status → ``"failed"``, retry_count += 1)."""
        df = self._load()
        now = _now()
        truncated = str(error_msg)[:_MAX_ERR_LEN]
        mask = self._mask(df, key)

        if mask.any():
            cur_count = pd.to_numeric(df.loc[mask, "retry_count"], errors="coerce").fillna(0)
            df.loc[mask, "retry_count"] = (cur_count + 1).astype(int).astype(str)
            df.loc[mask, "status"] = "failed"
            df.loc[mask, "error_msg"] = truncated
            df.loc[mask, "last_attempt_at"] = now
        else:
            new_row = {**key, "status": "failed", "error_msg": truncated,
                       "retry_count": "1", "last_attempt_at": now}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        self._save(df)

    def resolve(self, key: dict[str, str]) -> None:
        """Mark a record as ``"resolved"`` (suppressed in future restore runs)."""
        df = self._load()
        mask = self._mask(df, key)
        if mask.any():
            df.loc[mask, "status"] = "resolved"
            df.loc[mask, "last_attempt_at"] = _now()
            self._save(df)

    def get_pending(self) -> pd.DataFrame:
        """Return all records with ``status == "failed"``."""
        df = self._load()
        if df.empty:
            return df
        pending = df[df["status"] == "failed"].reset_index(drop=True)
        return pending

    def count_pending(self) -> int:
        """Number of unresolved failures."""
        return len(self.get_pending())

    def summary_line(self) -> str:
        """Human-readable failure summary."""
        df = self._load()
        if df.empty:
            return "no failures recorded"
        total = len(df)
        failed = (df["status"] == "failed").sum()
        resolved = (df["status"] == "resolved").sum()
        return f"{total} records: {failed} pending, {resolved} resolved → {self._path.name}"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        if self._path.exists() and self._path.stat().st_size > 0:
            try:
                return pd.read_csv(self._path, dtype=str)
            except Exception as exc:
                logger.warning(f"Could not read failure CSV {self._path}: {exc}")
        return pd.DataFrame(columns=self._all_cols)

    def _save(self, df: pd.DataFrame) -> None:
        # Ensure all expected columns exist before writing
        for col in self._all_cols:
            if col not in df.columns:
                df[col] = ""
        df[self._all_cols].to_csv(self._path, index=False)

    def _mask(self, df: pd.DataFrame, key: dict[str, str]) -> pd.Series:
        if df.empty:
            return pd.Series([], dtype=bool)
        mask = pd.Series([True] * len(df), index=df.index)
        for col, val in key.items():
            if col in df.columns:
                mask = mask & (df[col].astype(str) == str(val))
            else:
                mask = mask & pd.Series([False] * len(df), index=df.index)
        return mask


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

def summary_tracker(failures_dir: str | Path) -> FailureTracker:
    """Return a FailureTracker for LHB summary (PK: trade_date)."""
    return FailureTracker(
        csv_path=Path(failures_dir) / "failures_lhb_summary.csv",
        primary_keys=["trade_date"],
    )


def detail_tracker(failures_dir: str | Path) -> FailureTracker:
    """Return a FailureTracker for LHB detail (PK: trade_date + stock_code)."""
    return FailureTracker(
        csv_path=Path(failures_dir) / "failures_lhb_detail.csv",
        primary_keys=["trade_date", "stock_code"],
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
