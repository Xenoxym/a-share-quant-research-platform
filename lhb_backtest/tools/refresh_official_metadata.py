"""Refresh CN basic/industry metadata using the endpoints in SimTradeData's repair tools."""
from pathlib import Path
import argparse
import sys
import pandas as pd


def read_result(result):
    if result.error_code!='0':
        raise RuntimeError(result.error_msg)
    rows=[]
    while result.next():
        rows.append(result.get_row_data())
    if result.error_code!='0':
        raise RuntimeError(result.error_msg)
    return pd.DataFrame(rows,columns=result.fields)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--end',required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.repo))
    from simtradedata.fetchers.baostock_fetcher import BaoStockFetcher
    from simtradedata.writers.duckdb_writer import DuckDBWriter
    from simtradedata.utils.code_utils import convert_to_ptrade_code
    from scripts.repair_cn_industry import parse_zjhhy,build_blocks
    import baostock as bs
    fetcher=BaoStockFetcher()
    fetcher.login()
    try:
        basic=read_result(bs.query_stock_basic())
        industry=read_result(bs.query_stock_industry(code='',date=args.end))
    finally:
        fetcher.logout()
    if len(basic)<5000 or len(industry)<4000:
        raise RuntimeError('Provider metadata response is incomplete')
    writer=DuckDBWriter(db_path=str(args.db))
    try:
        metadata=writer.conn.execute('SELECT * FROM stock_metadata').fetchdf().set_index('symbol')
        basic['symbol']=basic.code.map(lambda x:convert_to_ptrade_code(x,'baostock'))
        basic=basic.set_index('symbol')
        for source,target in [('code_name','stock_name'),('ipoDate','listed_date'),('outDate','de_listed_date'),('type','security_type'),('status','listing_status')]:
            valid=basic[source].replace('',None).dropna()
            symbols=metadata.index.intersection(valid.index)
            metadata.loc[symbols,target]=valid.loc[symbols]
        for row in industry.itertuples():
            symbol=convert_to_ptrade_code(row.code,'baostock')
            parsed=parse_zjhhy(row.industry)
            if symbol in metadata.index and parsed:
                existing=metadata.loc[symbol,'blocks']
                metadata.loc[symbol,'blocks']=build_blocks(None if pd.isna(existing) else existing,*parsed)
        writer.write_stock_metadata(metadata.reset_index())
    finally:
        writer.close()


if __name__=='__main__':
    main()
