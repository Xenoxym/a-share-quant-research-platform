import json
import pytest
import pandas as pd
from src.data_sources.snapshot_quality import inspect_active_universe

@pytest.mark.parametrize('encoding',['json','array'])
def test_missing_active_share_is_not_hidden_by_other_updated_or_halted_stocks(tmp_path,encoding):
 (tmp_path/'metadata').mkdir();(tmp_path/'stocks').mkdir()
 symbols=['000001.SZ','000002.SZ','000003.SZ','000004.SZ','000005.SZ','150001.SZ']
 pd.DataFrame({'symbol':symbols,'listed_date':['2000-01-01']*4+['2026-09-24','2000-01-01'],
  'de_listed_date':['2900-01-01']*3+['2026-09-23','2900-01-01','2900-01-01']}).to_parquet(tmp_path/'metadata/stock_metadata.parquet')
 pd.DataFrame({'date':['20260923'],'status_type':['HALT'],'symbols':[json.dumps(['000003.SZ']) if encoding=='json' else ['000003.SZ']]}).to_parquet(tmp_path/'metadata/stock_status.parquet')
 bar=pd.DataFrame({'date':pd.to_datetime(['2026-09-23'])})
 bar.to_parquet(tmp_path/'stocks/000001.SZ.parquet')
 result=inspect_active_universe(tmp_path,'2026-09-23')
 assert result=={'expected_active_symbols':3,'active_symbols_with_bars':1,'halted_without_bars':['000003.SZ'],'unexplained_missing_symbols':['000002.SZ']}
 bar.to_parquet(tmp_path/'stocks/000002.SZ.parquet')
 assert inspect_active_universe(tmp_path,'2026-09-23')['unexplained_missing_symbols']==[]
