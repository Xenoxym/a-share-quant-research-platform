"""Finish the isolated repair and record its publication gates."""
from pathlib import Path
import json
import sys
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, r'D:\Projects\SimTradeDataRepo')
from simtradedata.writers.duckdb_writer import DuckDBWriter
from src.data_sources.snapshot_merge import merge_history
from src.data_sources.snapshot_quality import require_snapshot_quality
from src.data_sources.history_quality import verify_history_preserved

ROOT = PROJECT/'data/quality_repair/20260923'
EXPORT = ROOT/'auxiliary_export'
BASELINE = Path('D:/Projects/SimTradeData/data/export/cn.previous')


def main():
    application = json.loads((ROOT/'repair_application.json').read_text(encoding='utf-8'))
    if not application['complete']:
        raise RuntimeError('Per-date download coverage is incomplete; inspect repair_application.json')
    # Preserve the Release metadata date types, including Arrow date32.
    for source in (BASELINE/'metadata').glob('*.parquet'):
        target = EXPORT/'metadata'/source.name
        if source.name == 'version.parquet' or not target.exists():
            continue
        table = pq.read_table(target)
        for field in pq.read_schema(source):
            if pa.types.is_date32(field.type):
                values = pd.to_datetime(table[field.name].to_pandas().astype(str),format='mixed').dt.date
                table = table.set_column(table.schema.get_field_index(field.name),field.name,pa.array(values,type=field.type))
        pq.write_table(table,target,compression='zstd')
    writer = DuckDBWriter(db_path=str(ROOT/'cn.repair.duckdb'))
    fixes = json.loads((PROJECT/'research/audit_20260923/critical_quote_resolution.json').read_text(encoding='utf-8'))
    for fix in fixes:
        field = fix['field']
        assert field in ('close','high') and fix['matches_old']
        writer.conn.execute(f'UPDATE stocks SET {field}=? WHERE symbol=? AND date=CAST(? AS DATE)',
                            [fix['fresh_vendor'],fix['symbol'],fix['date']])
    writer.write_trade_days(pd.read_parquet(EXPORT/'metadata/trade_days.parquet'))
    writer.write_stock_metadata(pd.read_parquet(EXPORT/'metadata/stock_metadata.parquet'))
    for row in pd.read_parquet(EXPORT/'metadata/index_constituents.parquet').itertuples():
        writer.write_index_constituents(str(row.date),row.index_code,list(row.symbols))
    for symbol in sorted({r['symbol'] for r in fixes if r['field']=='close'}):
        target = EXPORT/'valuation'/f'{symbol}.parquet'
        previous = pd.read_parquet(target)
        temporary = ROOT/f'{symbol}.valuation.parquet'
        writer._export_valuation_enriched(symbol,temporary)
        merge_history(previous,pd.read_parquet(temporary),['date']).to_parquet(target,index=False)
        temporary.unlink()
    writer.close()
    manifest = json.loads((EXPORT/'manifest.json').read_text(encoding='utf-8'))
    manifest['description'] = f'SimTradeData export ({len(list((EXPORT/"stocks").glob("*.parquet")))} stocks); repaired history and auxiliary coverage'
    manifest['export_date'] = datetime.now().isoformat(timespec='seconds')
    manifest['repair_evidence'] = 'lhb_history_quality.json; lhb_quality_report.json'
    (EXPORT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Candidate finalized; checking coverage',flush=True)
    quality = require_snapshot_quality(EXPORT,'2026-09-22','2019-01-02')
    print(json.dumps(quality,ensure_ascii=False),flush=True)
    history = verify_history_preserved(BASELINE,EXPORT)
    (EXPORT/'lhb_history_quality.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    (PROJECT/'research/audit_20260923/final_history_quality.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    print(json.dumps(history),flush=True)
    if history['status'] != 'pass':
        raise RuntimeError('Historical preservation gate failed')


if __name__=='__main__':
    main()
