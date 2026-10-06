"""Read-only, per-file content comparison against a retained vendor baseline."""
from pathlib import Path
import argparse
from collections import Counter
import hashlib
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(old_root, new_root):
    report = {}
    for folder in ['stocks', 'valuation', 'exrights', 'fundamentals', 'metadata']:
        old_files = {p.name: p for p in (old_root / folder).glob('*.parquet')}
        new_files = {p.name: p for p in (new_root / folder).glob('*.parquet')}
        counts = Counter()
        null_losses, value_changes, old_nulls, new_nulls = Counter(), Counter(), Counter(), Counter()
        exceptions, examples, schemas = [], [], []
        missing_files = sorted(old_files.keys() - new_files.keys())
        for num, (name, old_path) in enumerate(sorted(old_files.items())):
            if num % 1000 == 0:
                print(folder, num, '/', len(old_files), flush=True)
            old_meta = pq.ParquetFile(old_path)
            counts['old_rows'] += old_meta.metadata.num_rows
            if name not in new_files:
                counts['missing_old_rows'] += old_meta.metadata.num_rows
                continue
            new_path = new_files[name]
            new_meta = pq.ParquetFile(new_path)
            counts['new_rows_in_common_files'] += new_meta.metadata.num_rows
            if digest(old_path) == digest(new_path):
                counts['identical_files'] += 1
                continue
            old_schema = {f.name: str(f.type) for f in old_meta.schema_arrow}
            new_schema = {f.name: str(f.type) for f in new_meta.schema_arrow}
            if old_schema != new_schema:
                schemas.append({'file': name, 'old': old_schema, 'new': new_schema})
            old, new = pd.read_parquet(old_path), pd.read_parquet(new_path)
            if name == 'version.parquet':
                continue  # Version metadata is expected to change.
            keys = ['date']
            if name == 'stock_metadata.parquet':
                keys = ['symbol']
            elif name == 'stock_status.parquet':
                keys = ['date', 'status_type']
            elif name == 'index_constituents.parquet':
                keys = ['date', 'index_code']
            if 'date' in keys:
                for frame in (old, new):
                    frame['date'] = pd.to_datetime(frame.date.astype(str), format='mixed').dt.strftime('%Y-%m-%d')
            if old.duplicated(keys).any() or new.duplicated(keys).any():
                exceptions.append({'file': name, 'error': 'duplicate keys',
                                   'old': int(old.duplicated(keys).sum()), 'new': int(new.duplicated(keys).sum())})
                continue
            old, new = old.set_index(keys), new.set_index(keys)
            missing = old.index.difference(new.index)
            counts['missing_old_rows'] += len(missing)
            counts['new_keys'] += len(new.index.difference(old.index))
            if len(missing) and len(examples) < 30:
                examples.append({'file': name, 'missing_keys': [str(v) for v in missing[:5]]})
            common = old.index.intersection(new.index)
            for col in old.columns:
                label = f'{name}:{col}' if folder == 'metadata' else col
                old_nulls[label] += int(old[col].isna().sum())
                if col not in new:
                    null_losses[label] += int(old[col].notna().sum())
                    continue
                new_nulls[label] += int(new[col].isna().sum())
                a, b = old.loc[common, col], new.loc[common, col]
                lost = a.notna() & b.isna()
                null_losses[label] += int(lost.sum())
                valid = a.notna() & b.notna()
                av, bv = a[valid], b[valid]
                if pd.api.types.is_numeric_dtype(av) and pd.api.types.is_numeric_dtype(bv):
                    changed = ~np.isclose(av.to_numpy(dtype=float), bv.to_numpy(dtype=float), rtol=1e-7, atol=1e-8)
                else:
                    def canonical(x):
                        if isinstance(x, (list, np.ndarray)):
                            return tuple(sorted(map(str, x)))
                        return str(x)
                    changed = av.map(canonical).to_numpy() != bv.map(canonical).to_numpy()
                value_changes[label] += int(changed.sum())
                if changed.any() and len(examples) < 30:
                    pos = np.flatnonzero(changed)[0]
                    examples.append({'file': name, 'column': col, 'key': str(av.index[pos]),
                                     'old': str(av.iloc[pos]), 'new': str(bv.iloc[pos])})
        added = sorted(new_files.keys() - old_files.keys())
        counts['new_rows_in_new_files'] = sum(pq.ParquetFile(new_files[n]).metadata.num_rows for n in added)
        report[folder] = {'old_files': len(old_files), 'new_files': len(new_files),
                          'missing_files': missing_files, 'added_files': added, **dict(counts),
                          'nonnull_to_null_common_keys': dict(+null_losses),
                          'changed_values_common_keys': dict(+value_changes),
                          'old_nulls_changed_files': dict(+old_nulls), 'new_nulls_changed_files': dict(+new_nulls),
                          'schema_changes': schemas, 'exceptions': exceptions, 'examples': examples}
        print(folder, json.dumps(dict(counts)), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.old, args.new)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
