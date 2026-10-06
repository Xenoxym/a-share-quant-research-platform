"""Cache real repair inputs using SimTradeData's official UnifiedDataFetcher.

No live snapshot/database changes. Each process has its own BaoStock session.
"""
from pathlib import Path
import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import pandas as pd

FETCHER = None


def initialize(repo):
    global FETCHER
    sys.path.insert(0, repo)
    from simtradedata.fetchers.unified_fetcher import UnifiedDataFetcher
    FETCHER = UnifiedDataFetcher()
    FETCHER.login()
    atexit.register(FETCHER.logout)


def fetch(job):
    symbol, start, end, destination = job
    path = Path(destination) / f'{symbol}.parquet'
    if path.exists():
        df = pd.read_parquet(path)
        if len(df) and {'date', 'isST', 'tradestatus', 'turn'}.issubset(df):
            return {'symbol': symbol, 'status': 'cached', 'rows': len(df)}
    try:
        df = FETCHER.fetch_unified_daily_data(symbol, start, end)
        if df.empty:
            return {'symbol': symbol, 'status': 'empty', 'rows': 0}
        if df[['isST', 'tradestatus']].isna().any().any():
            raise ValueError('Unknown status fields returned by vendor')
        tmp = path.with_suffix('.tmp')
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
        return {'symbol': symbol, 'status': 'downloaded', 'rows': len(df),
                'first': str(df.date.min()), 'last': str(df.date.max())}
    except Exception as exc:
        return {'symbol': symbol, 'status': 'error', 'error': str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', default='2026-05-23')
    parser.add_argument('--end', default='2026-09-22')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Match the SH/SZ A-share scope of the baseline BaoStock valuation files.
    stocks = sorted(p.stem for p in (args.snapshot / 'stocks').glob('*.parquet')
                    if (p.stem.endswith('.SS') and p.stem.startswith('6'))
                    or (p.stem.endswith('.SZ') and p.stem.startswith(('00', '30'))))
    jobs = [(s, args.start, args.end, str(args.output.resolve())) for s in stocks]
    report = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(str(args.repo.resolve()),)) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for future in as_completed(futures):
            report.append(future.result())
            if len(report) % 100 == 0 or len(report) == len(jobs):
                counts = pd.Series([r['status'] for r in report]).value_counts().to_dict()
                print(f'{len(report)}/{len(jobs)} {counts}', flush=True)
                (args.output / 'progress.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'request.json').write_text(json.dumps({'start': args.start, 'end': args.end,
        'symbols': stocks, 'source': 'SimTradeData UnifiedDataFetcher / BaoStock',
        'results': report}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
