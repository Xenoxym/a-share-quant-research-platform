"""I/O utilities for reading and writing data (Parquet, CSV, DuckDB)."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile
from typing import Any, Optional, Union

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.utils.logging import get_logger

logger = get_logger("io")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


def _load_config() -> dict:
    """Load the main config file."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_path(path: Union[str, Path]) -> Path:
    """Resolve a path, treating relative paths as relative to project root."""
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the Path.

    Parameters
    ----------
    path : str or Path
        Directory path to ensure exists.

    Returns
    -------
    Path
        The resolved directory path.
    """
    p = _resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(
    df: pd.DataFrame,
    path: Union[str, Path],
    partition_cols: Optional[list[str]] = None,
) -> None:
    """Save a DataFrame to Parquet format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str or Path
        Destination file or directory path.
    partition_cols : list[str] or None
        Columns to partition by (creates directory-based partitioning).
    """
    p = _resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df)

    if partition_cols:
        pq.write_to_dataset(
            table,
            root_path=str(p),
            partition_cols=partition_cols,
        )
    else:
        fd, temporary = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=p.parent)
        os.close(fd)
        try:
            pq.write_table(table, temporary, compression="snappy")
            os.replace(temporary, p)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    logger.debug(f"Saved parquet: {p} ({len(df)} rows)")


def load_parquet(
    path: Union[str, Path],
    filters: Optional[list[tuple[str, str, Any]]] = None,
) -> pd.DataFrame:
    """Load a Parquet file or partitioned dataset into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to parquet file or partitioned directory.
    filters : list of tuples or None
        PyArrow filter expressions, e.g. [("date", ">=", "2023-01-01")].

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    p = _resolve_path(path)

    if p.is_dir():
        dataset = pq.ParquetDataset(str(p), filters=filters)
        table = dataset.read()
    else:
        table = pq.read_table(str(p), filters=filters)

    df = table.to_pandas()
    logger.debug(f"Loaded parquet: {p} ({len(df)} rows)")
    return df


def save_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Save a DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    path : str or Path
        Destination file path.
    """
    p = _resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(p), index=False, encoding="utf-8-sig")
    logger.debug(f"Saved CSV: {p} ({len(df)} rows)")


def load_csv(path: Union[str, Path], **kwargs: Any) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to CSV file.
    **kwargs
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    p = _resolve_path(path)
    df = pd.read_csv(str(p), encoding="utf-8-sig", **kwargs)
    logger.debug(f"Loaded CSV: {p} ({len(df)} rows)")
    return df


def get_duckdb_conn(db_path: Optional[Union[str, Path]] = None) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection.

    Parameters
    ----------
    db_path : str, Path, or None
        Path to DuckDB database file. If None, uses the path from config.
        Use ":memory:" for an in-memory database.

    Returns
    -------
    duckdb.DuckDBPyConnection
        Active DuckDB connection.
    """
    if db_path is None:
        cfg = _load_config()
        db_path = cfg.get("storage", {}).get("duckdb_path", "data/lhb.duckdb")

    p = _resolve_path(db_path) if str(db_path) != ":memory:" else ":memory:"

    if isinstance(p, Path):
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(p))
    else:
        conn = duckdb.connect(p)

    logger.debug(f"DuckDB connection opened: {p}")
    return conn


def query_parquet(sql: str, parquet_path: Union[str, Path]) -> pd.DataFrame:
    """Run a SQL query directly on a Parquet file using DuckDB.

    The parquet file is accessible in SQL as 'data'.

    Parameters
    ----------
    sql : str
        SQL query string. Reference the parquet data as 'data'.
        Example: "SELECT * FROM data WHERE amount > 10000"
    parquet_path : str or Path
        Path to the parquet file.

    Returns
    -------
    pd.DataFrame
        Query result as DataFrame.
    """
    p = _resolve_path(parquet_path)
    with duckdb.connect(":memory:") as conn:
        conn.read_parquet(str(p)).create_view("data")
        result = conn.execute(sql).fetchdf()
    logger.debug(f"DuckDB query on {p}: {len(result)} rows returned")
    return result


def register_parquet_as_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    parquet_path: Union[str, Path],
) -> None:
    """Register a Parquet file as a named table/view in DuckDB.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    table_name : str
        Name to register the table as.
    parquet_path : str or Path
        Path to the parquet file or directory.
    """
    p = _resolve_path(parquet_path)
    conn.read_parquet(str(p)).create_view(table_name, replace=True)
    logger.debug(f"Registered parquet as table '{table_name}': {p}")
