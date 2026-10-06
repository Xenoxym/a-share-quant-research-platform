"""Safety contracts for complete official acquisition and bounded publication."""
from pathlib import Path
import json
import pytest
from src.ingestion.update_contract import (publish_managed, recover_publication,
    update_lock)


def stage(root, text):
    path = root / 'cn.staging'
    path.mkdir()
    (path / 'value').write_text(text)
    return path


def test_only_one_managed_rollback_and_release_untouched(tmp_path):
    live = tmp_path / 'cn'
    live.mkdir()
    (live/'value').write_text('first')
    release = tmp_path/'cn.previous'
    release.mkdir()
    (release/'value').write_text('release')
    publish_managed(stage(tmp_path, 'second'), live)
    publish_managed(stage(tmp_path, 'third'), live)
    assert (live/'value').read_text() == 'third'
    assert (tmp_path/'cn.update-rollback/value').read_text() == 'second'
    assert (release/'value').read_text() == 'release'
    assert len(list(tmp_path.glob('cn.update-rollback*'))) == 1


def test_publish_failure_restores_live(tmp_path, monkeypatch):
    live = tmp_path/'cn'
    live.mkdir()
    (live/'value').write_text('baseline')
    candidate = stage(tmp_path, 'new')
    rename = Path.rename
    def fail(self, target):
        if self == candidate:
            raise OSError('simulated publication failure')
        return rename(self, target)
    monkeypatch.setattr(Path, 'rename', fail)
    with pytest.raises(OSError):
        publish_managed(candidate, live)
    assert (live/'value').read_text() == 'baseline'


def test_crash_gap_is_recovered(tmp_path):
    live = tmp_path/'cn'
    backup = tmp_path/'cn.update-rollback'
    backup.mkdir()
    (backup/'value').write_text('old')
    (tmp_path/'cn.update-owner.json').write_text(json.dumps({'target':str(live.resolve())}))
    recover_publication(live)
    assert (live/'value').read_text() == 'old'


def test_unmanaged_backup_is_never_deleted(tmp_path):
    backup = tmp_path/'cn.update-rollback'
    backup.mkdir()
    with pytest.raises(RuntimeError, match='unmanaged'):
        publish_managed(stage(tmp_path, 'new'), tmp_path/'cn')
    assert backup.exists()


def test_concurrent_updater_is_rejected_and_lock_released(tmp_path):
    path = tmp_path/'lock'
    with update_lock(path):
        with pytest.raises((RuntimeError, BlockingIOError)):
            with update_lock(path):
                pass
    with update_lock(path):
        pass


def test_upstream_failure_stops_menu_import(monkeypatch, tmp_path):
    import tools.launcher as menu
    import src.ingestion.update_simtradedata as updater
    import src.ingestion.load_kline_simtradedata as loader
    import src.ingestion.fetch_lhb_summary as lhb
    monkeypatch.setattr(menu, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(menu, 'ask', lambda *args: '2026-09-22')
    def fail(*args, **kwargs):
        raise RuntimeError('upstream incomplete')
    def forbidden(*args, **kwargs):
        pytest.fail('dependent import ran after upstream failure')
    monkeypatch.setattr(updater, 'update_simtradedata', fail)
    monkeypatch.setattr(loader, 'load_kline_simtradedata', forbidden)
    monkeypatch.setattr(lhb, 'fetch_lhb_summary', forbidden)
    menu.action_update()

def test_failed_history_gate_keeps_live_export_and_records_failure(tmp_path, monkeypatch):
    import src.ingestion.update_simtradedata as updater
    import src.data_sources.snapshot_quality as coverage
    import src.data_sources.history_quality as history
    import src.data_sources.quote_quality as prices
    import sys
    repo=tmp_path/'repo'
    (repo/'data').mkdir(parents=True)
    (repo/'data/cn.duckdb').write_bytes(b'database')
    (repo/'scripts').mkdir()
    (repo/'scripts/download_mootdx.py').write_text('# qualified test provider')
    live=tmp_path/'cn'
    live.mkdir()
    (live/'sentinel').write_text('known good')
    monkeypatch.setattr(updater,'_PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(updater,'_load_config',lambda: {})
    monkeypatch.setattr(updater,'_validate_upstream_checkout',lambda *args: None)
    def resolve_under_lock(requested, *args):
        with pytest.raises((RuntimeError, BlockingIOError)):
            with update_lock(repo/'data/lhb_update.lock'):
                pytest.fail('Calendar request opened before acquiring the update lock')
        return requested
    monkeypatch.setattr(updater,'_resolve_update_date',resolve_under_lock)
    monkeypatch.setattr(updater,'_download_tdx_package',lambda dest: repo/'archive.zip')
    (repo/'archive.zip').write_bytes(b'zip')
    calls=[]
    def fake_run(command,**kwargs):
        calls.append([str(x) for x in command])
        if any(str(x).endswith('export_parquet.py') for x in command):
            stocks=tmp_path/'cn.staging/stocks'
            stocks.mkdir(parents=True)
            (stocks/'000001.SZ.parquet').write_bytes(b'candidate')
        if any(str(x).endswith('check_integrity.py') for x in command):
            (tmp_path/'data/update_reports/official_integrity.json').write_text('{"status":"pass"}')
    monkeypatch.setattr(updater,'_run',fake_run)
    monkeypatch.setattr(coverage,'inspect_snapshot',lambda *args: {'status':'pass'})
    monkeypatch.setattr(history,'verify_history_preserved',lambda *args: {'status':'fail','missing_rows':1})
    monkeypatch.setattr(prices,'verify_historical_prices',lambda *args: {'status':'pass'})
    with pytest.raises(RuntimeError,match='history'):
        updater.update_simtradedata('2026-09-22',project_dir=repo,export_dir=live,python_executable=sys.executable)
    assert (live/'sentinel').read_text()=='known good'
    report=json.loads((tmp_path/'data/update_reports/latest.json').read_text(encoding='utf-8'))
    assert report['status']=='fail'
    assert report['history']['missing_rows']==1
    download=next(c for c in calls if any(v.endswith('download.py') for v in c))
    assert '--skip-fundamentals' not in download
    assert '--skip-mootdx-ohlcv' in download

def test_batched_history_checks_missing_files_cells_and_new_duplicates(tmp_path):
    import pandas as pd
    from src.data_sources.history_quality import verify_history_preserved
    old,new=tmp_path/'old',tmp_path/'new'
    (old/'stocks').mkdir(parents=True)
    (new/'stocks').mkdir(parents=True)
    pd.DataFrame({'date':['20260101','20260102'],'close':[1.,2.]}).to_parquet(old/'stocks/A.parquet')
    pd.DataFrame({'date':['20260101','20260102'],'close':[3.,4.]}).to_parquet(old/'stocks/B.parquet')
    pd.DataFrame({'date':['20260101'],'close':[None]}).to_parquet(new/'stocks/A.parquet')
    pd.DataFrame({'date':['20260101','20260101'],'close':[5.,5.]}).to_parquet(new/'stocks/C.parquet')
    one=verify_history_preserved(old,new,batch_size=1)
    together=verify_history_preserved(old,new,batch_size=64)
    assert one==together
    assert one['status']=='fail'
    assert one['datasets']['stocks']['missing_rows']==3
    assert one['datasets']['stocks']['lost_known_cells']['close']==4
    assert one['datasets']['stocks']['duplicate_keys']==1

def test_price_change_counts_are_independent_of_batch_size(tmp_path):
    import pandas as pd
    from src.data_sources.quote_quality import verify_historical_prices
    old,new=tmp_path/'old',tmp_path/'new'
    (old/'stocks').mkdir(parents=True)
    (new/'stocks').mkdir(parents=True)
    for name in ('A','B'):
        frame=pd.DataFrame({'date':['20260101'],'open':[10.], 'high':[12.], 'low':[9.], 'close':[11.]})
        frame.to_parquet(old/'stocks'/f'{name}.parquet')
        if name=='A':
            frame['close']=11.5
        frame.to_parquet(new/'stocks'/f'{name}.parquet')
    one=verify_historical_prices(old,new,batch_size=1)
    together=verify_historical_prices(old,new,batch_size=64)
    assert one==together
    assert one['status']=='fail'
    assert one['changed_historical_prices']=={'close':1}


def test_project_import_failure_is_reported_after_source_publish(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import tools.launcher as menu
    import src.ingestion.update_simtradedata as updater
    import src.ingestion.load_kline_simtradedata as loader
    monkeypatch.setattr(menu,'PROJECT_ROOT',tmp_path)
    monkeypatch.setattr(menu,'ask',lambda *args:'2026-09-22')
    monkeypatch.setattr(updater,'update_simtradedata',lambda *args:SimpleNamespace(
        target_date='2026-09-22',export_dir=tmp_path/'cn',database_path=tmp_path/'cn.duckdb'))
    def fail(*args,**kwargs):raise RuntimeError('existing K-line cannot be read')
    monkeypatch.setattr(loader,'load_kline_simtradedata',fail)
    with pytest.raises(RuntimeError,match='cannot be read'):menu.action_update()
    report=json.loads((tmp_path/'data/update_reports/menu2_latest.json').read_text(encoding='utf-8'))
    assert report['status']=='fail' and report['phase']=='project_import'
    assert report['source_published'] is True


def test_cli_update_failure_returns_nonzero_exit(monkeypatch):
    import tools.launcher as menu
    monkeypatch.setattr(menu.sys,'argv',['launcher.py','update'])
    monkeypatch.setitem(menu.ACTIONS,'2',lambda:False)
    with pytest.raises(SystemExit) as error:menu.main()
    assert error.value.code==1
