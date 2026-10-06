"""Keep the baseline BaoStock index precision and share-volume unit on updates."""
from pathlib import Path
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo',required=True,type=Path)
    parser.add_argument('--db',required=True,type=Path)
    parser.add_argument('--end',required=True)
    args = parser.parse_args()
    sys.path.insert(0,str(args.repo.resolve()))
    from simtradedata.fetchers.unified_fetcher import UnifiedDataFetcher
    from simtradedata.writers.duckdb_writer import DuckDBWriter
    fetcher = UnifiedDataFetcher()
    fetcher.login()
    try:
        frame = fetcher.fetch_index_data('000300.SS','2005-01-01',args.end)
        if frame.empty:
            raise RuntimeError('Official BaoStock benchmark refresh returned no rows')
        with_writer = DuckDBWriter(db_path=str(args.db))
        try:
            with_writer.write_benchmark(frame)
        finally:
            with_writer.close()
    finally:
        fetcher.logout()


if __name__=='__main__':
    main()
