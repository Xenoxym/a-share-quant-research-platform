"""Prevent an incremental export from dropping historical keys or known cells."""
from pathlib import Path
from datetime import date, datetime
import shutil
import pandas as pd
from src.utils.io import save_parquet


def merge_history(previous: pd.DataFrame, current: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    old, new = previous.copy(), current.copy()
    if not set(keys).issubset(old) or not set(keys).issubset(new):
        raise ValueError(f'Snapshot merge requires keys {keys}')
    for frame in (old, new):
        if 'date' in keys:
            # Status exports use compact strings; daily exports use timestamps.
            frame['date'] = pd.to_datetime(frame['date'].astype(str), format='mixed').astype('datetime64[ns]')
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError('Snapshot contains null or duplicate keys')
    old, new = old.set_index(keys), new.set_index(keys)
    # Current non-null values win; an absent row/column/cell cannot erase history.
    result = new.combine_first(old).sort_index().reset_index()
    if 'date' in keys:
        sample = previous['date'].dropna()
        first = sample.iloc[0] if len(sample) else None
        if isinstance(first,str):
            fmt = '%Y%m%d' if first.isdigit() else '%Y-%m-%d'
            result['date'] = result['date'].dt.strftime(fmt)
        elif isinstance(first,date) and not isinstance(first,datetime):
            result['date'] = result['date'].dt.date
    return result


def preserve_snapshot_history(source: Path, staging: Path) -> dict:
    report = {'copied_files': 0, 'merged_files': 0}
    metadata_keys = {'stock_status.parquet': ['date', 'status_type'],
                     'index_constituents.parquet': ['date', 'index_code'],
                     'stock_metadata.parquet': ['symbol'],
                     'trade_days.parquet': ['date'], 'benchmark.parquet': ['date']}
    for folder in ('stocks', 'valuation', 'fundamentals', 'exrights', 'metadata'):
        for old_path in (source / folder).glob('*.parquet'):
            if folder == 'metadata' and old_path.name == 'version.parquet':
                continue
            target = staging / folder / old_path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(old_path, target)
                report['copied_files'] += 1
                continue
            if old_path.stat().st_size == target.stat().st_size and old_path.read_bytes() == target.read_bytes():
                continue
            keys = metadata_keys.get(old_path.name) if folder == 'metadata' else ['date']
            if keys is None:
                raise ValueError(f'Unknown metadata merge contract: {old_path.name}')
            result = merge_history(pd.read_parquet(old_path), pd.read_parquet(target), keys)
            save_parquet(result, target)
            report['merged_files'] += 1
    return report
