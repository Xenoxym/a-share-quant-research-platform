"""Seed official DuckDB tables from a Parquet baseline before incremental download.

Fills absent keys and missing first-bar preclose from the trusted baseline.
Existing non-null database values remain authoritative. This is a one-time
baseline migration, not a daily post-export overlay.
The caller must initialize the schema with SimTradeData's DuckDBWriter first.
"""
from pathlib import Path
import duckdb


def seed_database_from_snapshot(database: Path, snapshot: Path) -> dict:
    report = {}
    with duckdb.connect(str(database)) as conn:
        conn.execute("SET memory_limit='2GB'")
        conn.execute('SET threads=2')
        tables = {r[0] for r in conn.execute('SHOW TABLES').fetchall()}
        mapping = {t: snapshot / t / '*.parquet' for t in ('stocks', 'valuation', 'fundamentals', 'exrights')}
        mapping.update({t: snapshot / 'metadata' / (t + '.parquet') for t in
                        ('stock_metadata', 'stock_status', 'index_constituents', 'trade_days', 'benchmark')})
        conn.execute('BEGIN TRANSACTION')
        try:
            for table, path in mapping.items():
                if table not in tables or not list(path.parent.glob(path.name)):
                    continue
                target = {r[0]: r[1] for r in conn.execute(f'DESCRIBE "{table}"').fetchall()}
                source = {r[0]: r[1] for r in conn.execute(
                    'DESCRIBE SELECT * FROM read_parquet(?, union_by_name=true)', [str(path)]).fetchall()}
                expressions = []
                if 'symbol' in target and 'symbol' not in source:
                    expressions.append("regexp_extract(replace(filename, chr(92), '/'), '([^/]+)\\.parquet$', 1) AS symbol")
                for col in target.keys() & source.keys():
                    expression = f'"{col}"'
                    if col == 'symbols' and source[col].endswith('[]') and target[col] == 'VARCHAR':
                        expression = f'to_json("{col}") AS "{col}"'
                    expressions.append(expression)
                before = conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
                conn.execute(f'INSERT OR IGNORE INTO "{table}" BY NAME SELECT {", ".join(expressions)} '
                             'FROM read_parquet(?, union_by_name=true, filename=true)', [str(path)])
                after = conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
                report[table] = {'before': before, 'inserted': after - before, 'after': after}
                if table == 'stocks' and 'preclose' in target and 'preclose' in source:
                    restored = conn.execute(r'''
                        WITH first_dates AS (SELECT symbol, min(date) AS date FROM stocks GROUP BY symbol),
                        baseline AS (
                            SELECT regexp_extract(replace(filename,chr(92),'/'),
                                '([^/]+)\.parquet$',1) AS symbol, date, preclose
                            FROM read_parquet(?, union_by_name=true, filename=true)
                        )
                        UPDATE stocks s SET preclose=b.preclose
                        FROM baseline b JOIN first_dates f USING(symbol,date)
                        WHERE s.symbol=b.symbol AND s.date=b.date
                          AND s.preclose IS NULL AND b.preclose IS NOT NULL
                        RETURNING s.symbol
                    ''', [str(path)]).fetchall()
                    report[table]['restored_first_preclose'] = len(restored)
            conn.execute('COMMIT')
        except Exception:
            conn.execute('ROLLBACK')
            raise
    return report
