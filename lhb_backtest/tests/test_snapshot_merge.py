import pandas as pd
import pytest
from src.data_sources.snapshot_merge import merge_history
from src.ingestion.load_kline_simtradedata import load_kline_simtradedata


def test_new_partial_row_cannot_erase_known_history():
    old = pd.DataFrame({'date': ['20260102', '20260105'], 'close': [10., 11.], 'turnover_rate': [1., 2.]})
    new = pd.DataFrame({'date': ['2026-01-05', '2026-01-06'], 'close': [11.1, 12.], 'turnover_rate': [None, 3.]})
    result = merge_history(old, new, ['date']).set_index('date')
    assert len(result) == 3
    assert result.loc['20260105', 'close'] == 11.1
    assert result.loc['20260105', 'turnover_rate'] == 2.
    assert result.loc['20260102', 'close'] == 10.


def test_duplicate_history_keys_abort_merge():
    bad = pd.DataFrame({'date': ['20260102', '20260102'], 'v': [1, 2]})
    with pytest.raises(ValueError, match='duplicate'):
        merge_history(bad, bad.iloc[:1], ['date'])


def test_refresh_repairs_auxiliary_cells_on_already_complete_dates(tmp_path, monkeypatch):
    old = pd.DataFrame({'trade_date': ['2026-01-05'], 'stock_code': ['000001.SZ'],
                        'close': [11.], 'turnover_rate': [None], 'amount': [123.]})
    path = tmp_path / 'daily.parquet'
    old.to_parquet(path, index=False)
    monkeypatch.setattr('src.utils.calendar.get_trading_dates', lambda *args: ['2026-01-05'])
    updated = old.copy()
    updated['turnover_rate'] = .12
    updated['amount'] = None
    monkeypatch.setattr('src.data_sources.simtradedata_loader.load_stocks_parquet', lambda **kwargs: updated)
    result = load_kline_simtradedata('2026-01-05', '2026-01-05', export_dir=tmp_path,
                                     save_path=str(path), refresh_existing=True)
    assert result.turnover_rate.iloc[0] == .12
    assert result.amount.iloc[0] == 123.
    assert result.attrs['update_report']['rows_added'] == 0


def test_seed_baseline_preserves_database_and_status_lists(tmp_path):
    import duckdb
    import json
    from src.data_sources.snapshot_seed import seed_database_from_snapshot
    db = tmp_path / 'cn.duckdb'
    with duckdb.connect(str(db)) as c:
        c.execute('CREATE TABLE valuation(symbol VARCHAR, date DATE, pb DOUBLE, PRIMARY KEY(symbol,date))')
        c.execute("INSERT INTO valuation VALUES ('000001.SZ','2026-01-05',2)")
        c.execute('CREATE TABLE stock_status(date VARCHAR, status_type VARCHAR, symbols VARCHAR, PRIMARY KEY(date,status_type))')
    root = tmp_path / 'snapshot'
    (root / 'valuation').mkdir(parents=True)
    (root / 'metadata').mkdir()
    pd.DataFrame({'date': pd.to_datetime(['2026-01-05','2026-01-06']), 'pb': [1.,3.]}).to_parquet(root/'valuation/000001.SZ.parquet', index=False)
    pd.DataFrame({'date':['20260105'], 'status_type':['ST'], 'symbols':[['000001.SZ']]}).to_parquet(root/'metadata/stock_status.parquet', index=False)
    report = seed_database_from_snapshot(db, root)
    assert report['valuation']['inserted'] == 1
    assert seed_database_from_snapshot(db, root)['valuation']['inserted'] == 0
    with duckdb.connect(str(db)) as c:
        assert c.execute('SELECT pb FROM valuation ORDER BY date').fetchall() == [(2.,),(3.,)]
        assert json.loads(c.execute('SELECT symbols FROM stock_status').fetchone()[0]) == ['000001.SZ']


def test_baostock_pagination_retains_all_pages_and_rejects_partial_error():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('vendor_compat', Path(__file__).resolve().parents[1] / 'compat/sitecustomize.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Result:
        fields = ['code']
        error_code = '0'
        error_msg = ''
        data = [['a'], ['b']]
        calls = 0
        def next(self):
            self.calls += 1
            if self.calls == 1:
                self.data = [['c']]
                return True
            return False
    assert module.collect_result_data(Result()).code.tolist() == ['a','b','c']
    bad = Result()
    def fail():
        bad.error_code = 'network-error'
        return False
    bad.next = fail
    with pytest.raises(RuntimeError, match='pagination failed'):
        module.collect_result_data(bad)


def test_history_gate_rejects_missing_rows_and_known_cell_loss(tmp_path):
    from src.data_sources.history_quality import verify_history_preserved
    old, new = tmp_path/'old', tmp_path/'new'
    for root in (old,new):
        (root/'stocks').mkdir(parents=True)
    a = pd.DataFrame({'date':pd.to_datetime(['2026-01-05','2026-01-06']), 'close':[1.,2.]})
    a.to_parquet(old/'stocks/000001.SZ.parquet',index=False)
    b = a.iloc[:1].copy()
    b['close'] = None
    b.to_parquet(new/'stocks/000001.SZ.parquet',index=False)
    report = verify_history_preserved(old,new)
    assert report['status'] == 'fail'
    assert report['datasets']['stocks']['missing_rows'] == 1
    a.to_parquet(new/'stocks/000001.SZ.parquet',index=False)
    assert verify_history_preserved(old,new)['status'] == 'pass'


def test_one_fresh_stock_cannot_hide_missing_market_valuation(tmp_path):
    from src.data_sources.snapshot_quality import inspect_snapshot
    for folder in ('stocks','valuation','exrights','metadata'):
        (tmp_path/folder).mkdir()
    day = pd.Timestamp('2026-01-05')
    for name in ['trade_days','benchmark']:
        pd.DataFrame({'date':[day]}).to_parquet(tmp_path/f'metadata/{name}.parquet',index=False)
    pd.DataFrame({'date':['20260105']*2,'status_type':['ST','HALT'],'symbols':[[],[]]}).to_parquet(tmp_path/'metadata/stock_status.parquet',index=False)
    for code in ('000001.SZ','000002.SZ'):
        pd.DataFrame({'date':[day],'volume':[100]}).to_parquet(tmp_path/f'stocks/{code}.parquet',index=False)
    pd.DataFrame({'date':[day],'turnover_rate':[1.]}).to_parquet(tmp_path/'valuation/000001.SZ.parquet',index=False)
    pd.DataFrame({'date':[day]}).to_parquet(tmp_path/'exrights/000001.SZ.parquet',index=False)
    report = inspect_snapshot(tmp_path,'2026-01-05','2026-01-05')
    assert report['status'] == 'fail'
    assert report['coverage']['latest_trading_symbols_without_valuation'] == 1


def test_publication_does_not_delete_retained_release(tmp_path):
    from src.ingestion.update_simtradedata import _publish_snapshot
    live, prior, stage = tmp_path/'cn', tmp_path/'cn.previous', tmp_path/'cn.staging'
    for path in (live,prior,stage):
        path.mkdir()
    (prior/'reference.txt').write_text('baseline')
    (live/'current.txt').write_text('current')
    (stage/'new.txt').write_text('new')
    backup = _publish_snapshot(stage,live)
    assert (prior/'reference.txt').read_text() == 'baseline'
    assert backup != prior
    assert (backup/'current.txt').read_text() == 'current'


def test_unreadable_existing_kline_is_not_silently_replaced(tmp_path):
    path = tmp_path/'daily.parquet'
    path.write_bytes(b'not a valid parquet')
    with pytest.raises(RuntimeError,match='refusing to overwrite'):
        load_kline_simtradedata('2026-01-05','2026-01-05',export_dir=tmp_path,save_path=str(path))
    assert path.read_bytes() == b'not a valid parquet'


def test_metadata_date32_does_not_silently_become_text(tmp_path):
    from datetime import date
    import pyarrow.parquet as pq
    import pyarrow as pa
    old = pd.DataFrame({'date':[date(2026,1,5)],'close':[1.]})
    new = pd.DataFrame({'date':[date(2026,1,6)],'close':[2.]})
    path=tmp_path/'benchmark.parquet'
    merge_history(old,new,['date']).to_parquet(path,index=False)
    assert pq.read_schema(path).field('date').type == pa.date32()


def test_historical_price_regression_is_not_a_successful_update(tmp_path):
    from src.data_sources.quote_quality import verify_historical_prices
    old,new=tmp_path/'old',tmp_path/'new'
    for root in (old,new):
        (root/'stocks').mkdir(parents=True)
    pd.DataFrame({'date':pd.to_datetime(['2026-01-05']),'close':[16.69]}).to_parquet(old/'stocks/000004.SZ.parquet',index=False)
    pd.DataFrame({'date':pd.to_datetime(['2026-01-05','2026-01-06']),'close':[16.21,17.]}).to_parquet(new/'stocks/000004.SZ.parquet',index=False)
    report=verify_historical_prices(old,new)
    assert report['status']=='fail' and report['changed_historical_prices']=={'close':1}

def test_seed_restores_first_bar_preclose_without_overwriting_known_values(tmp_path):
    import duckdb
    from src.data_sources.snapshot_seed import seed_database_from_snapshot
    db=tmp_path/'cn.duckdb'
    with duckdb.connect(str(db)) as c:
        c.execute('CREATE TABLE stocks(symbol VARCHAR,date DATE,preclose DOUBLE,PRIMARY KEY(symbol,date))')
        c.execute("INSERT INTO stocks VALUES ('000001.SZ','2020-01-02',NULL),('000001.SZ','2020-01-03',20)")
    root=tmp_path/'baseline'
    (root/'stocks').mkdir(parents=True)
    pd.DataFrame({'date':pd.to_datetime(['2020-01-02','2020-01-03']),'preclose':[10.,99.]}).to_parquet(root/'stocks/000001.SZ.parquet')
    assert seed_database_from_snapshot(db,root)['stocks']['restored_first_preclose']==1
    assert seed_database_from_snapshot(db,root)['stocks']['restored_first_preclose']==0
    with duckdb.connect(str(db)) as c:
        assert c.execute('SELECT preclose FROM stocks ORDER BY date').fetchall()==[(10.,),(20.,)]
