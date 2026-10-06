"""Verify retained keys/known cells in bounded file batches, without rewriting data."""
from collections import Counter
from pathlib import Path
import tempfile
import duckdb
from src.utils.logging import get_logger

logger = get_logger('history_quality')


def _compare_files(old: list[Path], new: list[Path], keys: list[str], tmp: str) -> dict:
    with duckdb.connect() as conn:
        conn.execute("SET memory_limit='512MB'")
        conn.execute('SET threads=2')
        conn.execute('SET temp_directory=?', [tmp])
        for alias, paths in [('old_data', old), ('new_data', new)]:
            if not paths:
                continue
            conn.read_parquet([str(p) for p in paths], union_by_name=True, filename=True).create_view(alias+'_raw')
            # Materialize each small batch once; a view would rescan thousands
            # of Parquet footers for every DESCRIBE, join and duplicate check.
            conn.execute(f'''CREATE TEMP TABLE {alias} AS SELECT *,
                regexp_extract(replace(filename,chr(92),'/'),'([^/]+)$',1) AS _file
                FROM {alias}_raw''')
        result = {'missing_rows': 0, 'lost_known_cells': {}, 'missing_columns': [], 'duplicate_keys': 0}
        if not new:
            result['missing_rows'] = conn.execute('SELECT count(*) FROM old_data').fetchone()[0]
            fields = sorted(set(conn.table('old_data').columns)-set(keys)-{'filename','_file'})
            if fields:
                counts = conn.execute('SELECT '+', '.join(f'count("{c}")' for c in fields)+' FROM old_data').fetchone()
                result['lost_known_cells'] = {c:int(n) for c,n in zip(fields,counts) if n}
            return result
        new_cols = set(conn.table('new_data').columns)
        if not set(keys).issubset(new_cols):
            result['missing_columns'] = sorted(set(keys)-new_cols)
            return result
        group_keys = ','.join(f'"{k}"' for k in keys)
        result['duplicate_keys'] = conn.execute(f'SELECT count(*) FROM (SELECT {group_keys} FROM new_data GROUP BY {group_keys} HAVING count(*)>1)').fetchone()[0]
        if not old:
            return result
        old_cols = set(conn.table('old_data').columns)
        result['missing_columns'] = sorted(old_cols-new_cols-{'filename'})
        shared = sorted((old_cols & new_cols)-set(keys)-{'filename','_file'})
        on = ' AND '.join(f'o."{k}"=n."{k}"' for k in keys)
        checks = [f'count(*) FILTER (WHERE o."{col}" IS NOT NULL AND n."{col}" IS NULL)' for col in shared]
        query = f'SELECT count(*) FILTER (WHERE n."{keys[-1]}" IS NULL)'
        if checks:
            query += ', '+', '.join(checks)
        row = conn.execute(query+f' FROM old_data o LEFT JOIN new_data n ON {on}').fetchone()
        result['missing_rows'] = int(row[0])
        result['lost_known_cells'] = {col:int(count) for col,count in zip(shared,row[1:]) if count}
        return result


def verify_history_preserved(baseline: Path, candidate: Path, *, batch_size: int = 64) -> dict:
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    report = {}
    metadata_keys = {'stock_status.parquet':['date','status_type'],
                     'stock_metadata.parquet':['symbol'], 'benchmark.parquet':['date'],
                     'trade_days.parquet':['date'], 'index_constituents.parquet':['date','index_code']}
    pairs = [(f,baseline/f/'*.parquet',candidate/f/'*.parquet',['_file','date'])
             for f in ('stocks','valuation','fundamentals','exrights')]
    pairs += [(f'metadata/{name}',baseline/'metadata'/name,candidate/'metadata'/name,keys)
              for name,keys in metadata_keys.items()]
    with tempfile.TemporaryDirectory(prefix='lhb_history_check_') as tmp:
        for label, old_pattern, new_pattern, keys in pairs:
            old = {p.name:p for p in old_pattern.parent.glob(old_pattern.name)}
            new = {p.name:p for p in new_pattern.parent.glob(new_pattern.name)}
            if not old and not new:
                continue
            names = sorted(old.keys() | new.keys())
            total = {'missing_rows':0, 'lost_known_cells':Counter(), 'missing_columns':set(), 'duplicate_keys':0}
            logger.info('History comparison: {} ({} files, batches of {})', label,len(names),batch_size)
            for offset in range(0,len(names),batch_size):
                group = names[offset:offset+batch_size]
                part = _compare_files([old[n] for n in group if n in old],
                                      [new[n] for n in group if n in new],keys,tmp)
                total['missing_rows'] += part['missing_rows']
                total['duplicate_keys'] += part['duplicate_keys']
                total['lost_known_cells'].update(part['lost_known_cells'])
                total['missing_columns'].update(part['missing_columns'])
            total['lost_known_cells'] = dict(total['lost_known_cells'])
            total['missing_columns'] = sorted(total['missing_columns'])
            report[label] = total
            logger.info('History comparison complete: {}: {}',label,total)
    ok = all(not any(v.values()) for v in report.values())
    return {'status':'pass' if ok else 'fail','datasets':report,
            'scope':'Retained keys and known cells only; does not attest changed-value correctness or new-period completeness.'}
