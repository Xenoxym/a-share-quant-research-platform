"""Reject unreviewed historical OHLC changes using bounded file batches."""
from collections import Counter
from pathlib import Path
import duckdb


def verify_historical_prices(previous: Path, current: Path, *, batch_size: int = 64) -> dict:
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    old = {p.name:p for p in (previous/'stocks').glob('*.parquet')}
    new = {p.name:p for p in (current/'stocks').glob('*.parquet')}
    common = sorted(old.keys() & new.keys())
    if not common:
        return {'status':'fail','error':'No comparable stock files'}
    changes = Counter()
    for offset in range(0,len(common),batch_size):
        group = common[offset:offset+batch_size]
        with duckdb.connect() as conn:
            conn.execute("SET memory_limit='512MB'")
            conn.execute('SET threads=2')
            columns={}
            for name,paths in [('prior',old),('candidate',new)]:
                rel=conn.read_parquet([str(paths[n]) for n in group],filename=True,union_by_name=True)
                rel.create_view(name+'_raw')
                columns[name]=set(rel.columns)
            fields=[c for c in ('open','high','low','close') if c in columns['prior'] & columns['candidate']]
            if not fields:
                return {'status':'fail','error':'No comparable OHLC columns'}
            for name in ('prior','candidate'):
                conn.execute(f'''CREATE TEMP TABLE {name} AS SELECT date,{','.join(fields)},
                    regexp_extract(replace(filename,chr(92),'/'),'([^/]+)$',1) AS stock_file
                    FROM {name}_raw''')
            checks=','.join(f'count(*) FILTER (WHERE abs(p.{c}-n.{c})>0.00000001)' for c in fields)
            row=conn.execute(f'SELECT {checks} FROM prior p JOIN candidate n USING(stock_file,date)').fetchone()
            changes.update({field:int(count) for field,count in zip(fields,row) if count})
    return {'status':'fail' if changes else 'pass','changed_historical_prices':dict(changes),
            'scope':'OHLC only; source-confirmed corrections need explicit review before replacing published history.'}
