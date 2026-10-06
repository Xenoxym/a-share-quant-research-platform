"""Apply cached official repair inputs to an isolated candidate, with coverage evidence."""
from pathlib import Path
import json
import sys
import shutil
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, r'D:\Projects\SimTradeDataRepo')
from simtradedata.writers.duckdb_writer import DuckDBWriter
from simtradedata.writers.duckdb_writer import _FUNDAMENTAL_WRITE_COLUMNS
from simtradedata.utils.code_utils import convert_to_ptrade_code
from simtradedata.processors.data_splitter import DataSplitter
from src.data_sources.snapshot_merge import merge_history, preserve_snapshot_history

ROOT = PROJECT / 'data/quality_repair/20260923'
CANDIDATE = ROOT / 'candidate'


def main():
    request = json.loads((ROOT/'baostock/request.json').read_text(encoding='utf-8'))
    start, end = request['start'], request['end']
    writer = DuckDBWriter(db_path=str(ROOT/'cn.repair.duckdb'))
    splitter = DataSplitter()
    statuses = []
    valuations = []
    failures = [r for r in request['results'] if r['status'] == 'error']
    writer.begin()
    try:
        df = writer.conn.execute(r'''SELECT * EXCLUDE(filename),
            regexp_extract(replace(filename,chr(92),'/'),'([^/]+)\.parquet$',1) AS symbol
            FROM read_parquet(?,filename=true,union_by_name=true)''',
            [str(ROOT/'baostock/*.parquet')]).fetchdf()
        split = splitter.split_data(df)
        valuation = split['valuation'].reset_index()
        valuation['symbol'] = df.symbol.to_numpy()
        valuations.append(valuation)
        status = split['status'].copy()
        status['symbol'] = df.symbol.to_numpy()
        statuses.append(status)
        print('Loaded all cached repair rows',len(df),flush=True)
        fund = pd.read_parquet(ROOT/'financials_2026q2.parquet')
        valuation_batch = pd.concat(valuations,ignore_index=True)
        writer.conn.register('_repair_valuations',valuation_batch)
        writer.conn.execute('''INSERT INTO valuation BY NAME SELECT * FROM _repair_valuations
            ON CONFLICT(symbol,date) DO UPDATE SET
            pe_ttm=coalesce(EXCLUDED.pe_ttm,valuation.pe_ttm), pb=coalesce(EXCLUDED.pb,valuation.pb),
            ps_ttm=coalesce(EXCLUDED.ps_ttm,valuation.ps_ttm), pcf=coalesce(EXCLUDED.pcf,valuation.pcf),
            turnover_rate=coalesce(EXCLUDED.turnover_rate,valuation.turnover_rate)''')
        writer._record_symbol_changes('valuation',valuation_batch)
        print('Bulk valuation write complete',len(valuation_batch),flush=True)
        fund_batch = fund.rename(columns={'end_date':'date'}).copy()
        fund_batch['symbol'] = fund_batch.code.map(lambda code:convert_to_ptrade_code(str(code),'qstock'))
        fund_batch['publ_date'] = pd.to_datetime(fund_batch.publ_date).dt.strftime('%Y%m%d')
        fields = [c for c in _FUNDAMENTAL_WRITE_COLUMNS if c in fund_batch]
        fund_batch = fund_batch[fields]
        writer.conn.register('_repair_fundamentals',fund_batch)
        assignments = ','.join(f'{c}=coalesce(EXCLUDED.{c},fundamentals.{c})' for c in fields if c not in ('symbol','date'))
        writer.conn.execute('INSERT INTO fundamentals BY NAME SELECT * FROM _repair_fundamentals '
                            'ON CONFLICT(symbol,date) DO UPDATE SET '+assignments)
        writer._record_symbol_changes('fundamentals',fund_batch)
        print('Bulk financial write complete',len(fund_batch),flush=True)
        writer.commit()
    except Exception:
        writer.rollback()
        writer.close()
        raise
    status = pd.concat(statuses, ignore_index=True)
    status['date'] = pd.to_datetime(status.date).dt.strftime('%Y-%m-%d')
    by_date = {day: group for day, group in status.groupby('date')}
    present = {day: set(group.symbol) for day, group in by_date.items()}
    expected = {}
    quote_keys = writer.conn.execute(r'''SELECT CAST(date AS DATE)::VARCHAR AS day,
        regexp_extract(replace(filename,chr(92),'/'),'([^/]+)\.parquet$',1) AS symbol
        FROM read_parquet(?,filename=true) WHERE date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)''',
        [str(CANDIDATE/'stocks/*.parquet'),start,end]).fetchdf()
    quote_keys = quote_keys[quote_keys.symbol.isin(request['symbols'])]
    expected = {day:set(group.symbol) for day,group in quote_keys.groupby('day')}
    missing = {day: sorted(codes - present.get(day,set())) for day,codes in expected.items()
               if codes - present.get(day,set())}
    # Independent terminal universe check, beyond symbols seen in OHLCV exports.
    terminal = pd.read_parquet(ROOT/'status_universe'/f'{end}.parquet')
    terminal = terminal[terminal.code.str.match(r'^(sh\.(60|68)\d{4}|sz\.(00|30)\d{4})$')]
    terminal_codes = set(terminal.code.map(lambda c: convert_to_ptrade_code(c,'baostock')))
    missing_terminal = sorted(terminal_codes - present.get(end,set()))
    new_status = []
    for day, group in by_date.items():
        if day in missing:
            continue  # A partial symbol query must never establish a complete day.
        for kind, col, value in [('ST','isST',1),('HALT','tradestatus',0)]:
            symbols = sorted(group.loc[pd.to_numeric(group[col]).eq(value),'symbol'].unique())
            new_status.append({'date': day.replace('-',''), 'status_type':kind,'symbols':symbols})
    # For older no-HALT-record days, only vendor-confirmed daily universes establish an empty set.
    for path in (ROOT/'status_universe').glob('*.parquet'):
        if path.stem >= start:
            continue
        frame = pd.read_parquet(path)
        halted = frame.loc[frame.tradeStatus.eq('0'),'code'].map(lambda c: convert_to_ptrade_code(c,'baostock')).dropna().tolist()
        new_status.append({'date':path.stem.replace('-',''),'status_type':'HALT','symbols':halted})
    old_status = pd.read_parquet(CANDIDATE/'metadata/stock_status.parquet')
    # The previous release contains vendor status history. Current TDX-derived
    # HALT lists changed historical memberships; retain the richer baseline
    # before the actual vendor-refetched window, rather than treating those
    # heuristic changes as confirmed corrections.
    baseline_status = pd.read_parquet('D:/Projects/SimTradeData/data/export/cn.previous/metadata/stock_status.parquet')
    baseline_status = baseline_status[pd.to_datetime(baseline_status.date.astype(str),format='mixed') < pd.Timestamp(start)]
    old_status = merge_history(old_status,baseline_status,['date','status_type'])
    merged = merge_history(old_status,pd.DataFrame(new_status),['date','status_type'])
    merged.to_parquet(CANDIDATE/'metadata/stock_status.parquet',index=False)
    benchmark_path = ROOT/'benchmark_baostock.parquet'
    if benchmark_path.exists():
        benchmark = pd.read_parquet(benchmark_path).rename(columns={'amount':'money'})
        if 'date' not in benchmark and isinstance(benchmark.index,pd.DatetimeIndex):
            benchmark = benchmark.reset_index().rename(columns={'index':'date'})
        previous = pd.read_parquet(CANDIDATE/'metadata/benchmark.parquet')
        merge_history(previous,benchmark,['date']).to_parquet(CANDIDATE/'metadata/benchmark.parquet',index=False)
        writer.write_benchmark(benchmark.set_index('date'))
    constituents_path = ROOT/'index_constituents_new.parquet'
    if constituents_path.exists():
        previous = pd.read_parquet(CANDIDATE/'metadata/index_constituents.parquet')
        merge_history(previous,pd.read_parquet(constituents_path),['date','index_code']).to_parquet(CANDIDATE/'metadata/index_constituents.parquet',index=False)
    for row in new_status:
        writer.write_stock_status(row['date'],row['status_type'],row['symbols'])
    # Preserve the baseline's actual provider statuses in the repaired database too.
    status_seed = merged.copy()
    status_seed['symbols'] = status_seed.symbols.map(lambda xs:json.dumps(list(xs)))
    writer.conn.execute('INSERT OR REPLACE INTO stock_status SELECT date,status_type,symbols FROM status_seed')
    writer.compute_derived_fundamentals()
    export = ROOT/'auxiliary_export'
    for folder in ('valuation','fundamentals'):
        (export/folder).mkdir(parents=True,exist_ok=True)
    writer._export_fundamentals_batch(export/'fundamentals',market='cn')
    writer._export_valuation_batch(export/'valuation',market='cn')
    writer.close()
    # Keep old known cells when the current vendor calculation has no value.
    preserve_snapshot_history(CANDIDATE,export)
    for rel in ['manifest.json','metadata/version.parquet']:
        if (CANDIDATE/rel).exists():
            shutil.copy2(CANDIDATE/rel,export/rel)
    # export now includes the preserved quote/metadata trees; this is the final candidate.
    report = {'start':start,'end':end,'download_errors':failures,
              'missing_stock_dates':missing,'missing_terminal_universe':missing_terminal,
              'status_rows_added':len(new_status),'financial_rows_downloaded':len(fund),
              'complete':not failures and not missing and not missing_terminal,
              'output':str(export)}
    (ROOT/'repair_application.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['missing_stock_dates']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
