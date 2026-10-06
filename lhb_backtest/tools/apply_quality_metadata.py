"""Apply cached provider metadata without inventing listing dates or industries."""
from pathlib import Path
import json
import sys
import pandas as pd
sys.path.insert(0,r'D:\Projects\SimTradeDataRepo')
from scripts.repair_cn_industry import parse_zjhhy, build_blocks
from simtradedata.utils.code_utils import convert_to_ptrade_code
from simtradedata.writers.duckdb_writer import DuckDBWriter

root=Path(__file__).resolve().parents[1]/'data/quality_repair/20260923'
path=root/'auxiliary_export/metadata/stock_metadata.parquet'
metadata=pd.read_parquet(path).set_index('symbol')
basic=pd.read_parquet(root/'all_stock_basic.parquet')
basic['symbol']=basic.code.map(lambda x:convert_to_ptrade_code(x,'baostock'))
basic=basic.set_index('symbol')
for source,target in [('code_name','stock_name'),('ipoDate','listed_date'),('outDate','de_listed_date'),('type','security_type'),('status','listing_status')]:
    valid=basic[source].replace('',None).dropna()
    symbols=metadata.index.intersection(valid.index)
    metadata.loc[symbols,target]=valid.loc[symbols]
industry=pd.read_parquet(root/'industry_latest.parquet')
count=0
for row in industry.itertuples():
    symbol=convert_to_ptrade_code(row.code,'baostock')
    parsed=parse_zjhhy(row.industry)
    if symbol in metadata.index and parsed:
        existing=metadata.loc[symbol,'blocks']
        metadata.loc[symbol,'blocks']=build_blocks(None if pd.isna(existing) else existing,*parsed)
        count+=1
metadata=metadata.reset_index()
metadata.to_parquet(path,index=False)
writer=DuckDBWriter(db_path=str(root/'cn.repair.duckdb'))
writer.write_stock_metadata(metadata)
writer.close()
result={'industry_rows_filled':count,'basic_rows':len(basic),
        'unverified_codes':['001246.SZ','301569.SZ','301660.SZ','301716.SZ'],
        'unverified_policy':'Retain metadata records; do not fabricate listing dates or classify unconfirmed securities as active A shares.',
        'industry_asof':'2026-09-22','industry_scope':'Current classification only, not historical point-in-time industry membership.'}
(root/'metadata_application.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False),flush=True)
