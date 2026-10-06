"""Atomically merge K-line history without full pandas history copies."""
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import duckdb
import pandas as pd


def merge_kline_file(previous: Path, current: pd.DataFrame) -> None:
    previous = previous.resolve()
    temporary = previous.with_name(previous.name + '.' + uuid4().hex + '.tmp')
    quote = lambda name: '"' + name.replace('"', '""') + '"'
    literal = lambda value: "'" + str(value).replace("'", "''") + "'"
    keys = ['stock_code', 'trade_date']
    try:
        with TemporaryDirectory(prefix='kline_merge_', dir=previous.parent) as spill:
            with duckdb.connect() as connection:
                connection.execute("SET memory_limit='512MB'")
                connection.execute('SET threads=2')
                connection.execute('SET preserve_insertion_order=false')
                connection.execute('SET temp_directory=' + literal(spill))
                connection.register('incoming', current)
                connection.execute('CREATE VIEW old AS SELECT * REPLACE '
                    '(CAST(CAST(trade_date AS DATE) AS VARCHAR) AS trade_date) '
                    'FROM read_parquet(' + literal(previous) + ')')
                connection.execute('CREATE VIEW new AS SELECT * REPLACE '
                    '(CAST(CAST(trade_date AS DATE) AS VARCHAR) AS trade_date) FROM incoming')
                columns = {}
                for table in ('old', 'new'):
                    columns[table] = {row[0] for row in connection.execute('DESCRIBE ' + table).fetchall()}
                    if not set(keys).issubset(columns[table]):
                        raise ValueError('K-line merge requires stock_code and trade_date')
                    invalid = connection.execute(f'SELECT 1 FROM {table} GROUP BY stock_code,trade_date '
                        'HAVING count(*)>1 OR stock_code IS NULL OR trade_date IS NULL LIMIT 1').fetchone()
                    if invalid:
                        raise ValueError('K-line history contains null or duplicate keys')
                expressions = []
                for name in keys + sorted((columns['old'] | columns['new']) - set(keys)):
                    column = quote(name)
                    sources = [f'{alias}.{column}' for table, alias in [('new', 'n'), ('old', 'o')]
                               if name in columns[table]]
                    value = sources[0] if len(sources) == 1 else 'COALESCE(' + ','.join(sources) + ')'
                    expressions.append(value + ' AS ' + column)
                query = ('SELECT ' + ','.join(expressions) + ' FROM old o FULL OUTER JOIN new n '
                         'ON o.stock_code=n.stock_code AND o.trade_date=n.trade_date '
                         'ORDER BY stock_code,trade_date')
                connection.execute('COPY (' + query + ') TO ' + literal(temporary) +
                                   " (FORMAT PARQUET, COMPRESSION ZSTD)")
        temporary.replace(previous)
    finally:
        temporary.unlink(missing_ok=True)
