"""Publish only the validated 20260923 repair, retaining recoverable backups."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.ingestion.update_simtradedata import _publish_snapshot


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def main():
    repair = PROJECT/'data/quality_repair/20260923'
    candidate = repair/'auxiliary_export'
    validation = PROJECT/'data/validation/quality_repair_20260923_validated'
    report = json.loads((validation/'validation.json').read_text(encoding='utf-8'))
    for name in ('lhb_quality_report.json','lhb_history_quality.json','lhb_quote_quality.json'):
        assert json.loads((candidate/name).read_text(encoding='utf-8'))['status']=='pass', name
    assert json.loads((repair/'repair_application.json').read_text(encoding='utf-8'))['complete']
    assert report['pipeline_invariants']=='pass' and report['snapshot_readiness']['status']=='pass'
    assert json.loads((PROJECT/'research/audit_20260923/official_repair_integrity.json').read_text(encoding='utf-8'))['status']=='pass'
    assert Path(report['vendor_export_dir']).resolve()==candidate.resolve()
    for name,expected in report['source_sha256'].items():
        assert digest(PROJECT/'data/raw'/name)==expected,'Raw inputs changed after validation'
    live = Path('D:/Projects/SimTradeData/data/export/cn').resolve()
    db = Path('D:/Projects/SimTradeDataRepo/data/cn.duckdb').resolve()
    assert live.parent==Path('D:/Projects/SimTradeData/data/export').resolve()
    assert db.parent==Path('D:/Projects/SimTradeDataRepo/data').resolve()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    staged = live.with_name('cn.quality-staging-'+stamp)
    backup = PROJECT/'data/backups'/('before_quality_repair_'+stamp)
    backup.mkdir(parents=True,exist_ok=False)
    files = [validation/'raw/daily_kline.parquet']
    files += sorted(p for folder in ('clean','factor') for p in (validation/folder).glob('*.parquet'))
    assert len(files)==7
    records = []
    for source in files:
        relative = source.relative_to(validation)
        target = PROJECT/'data'/relative
        old = backup/relative
        old.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            shutil.copy2(target,old)
        records.append({'path':str(relative),'sha256':digest(source),'had_original':target.exists()})
    print('Preparing verified vendor copy',flush=True)
    shutil.copytree(candidate,staged)
    for source in candidate.rglob('*'):
        if source.is_file():
            assert digest(source)==digest(staged/source.relative_to(candidate)),str(source)
    db_staged = db.with_name('cn.quality-staging-'+stamp+'.duckdb')
    shutil.copy2(repair/'cn.repair.duckdb',db_staged)
    assert digest(db_staged)==digest(repair/'cn.repair.duckdb')
    db_backup = db.with_name('cn.before-quality-'+stamp+'.duckdb')
    assert not db.with_suffix('.duckdb.wal').exists(),'Live database has an active WAL'
    # Confirm the live database can be opened exclusively before replacing it.
    import duckdb
    connection = duckdb.connect(str(db))
    connection.close()
    vendor_backup = None
    moved_db = False
    changed = []
    try:
        db.rename(db_backup)
        moved_db = True
        db_staged.rename(db)
        vendor_backup = _publish_snapshot(staged,live)
        for source,record in zip(files,records):
            target = PROJECT/'data'/record['path']
            temporary = target.with_suffix('.quality-tmp')
            assert not temporary.exists()
            shutil.copy2(source,temporary)
            os.replace(temporary,target)
            changed.append(record)
            assert digest(target)==record['sha256']
    except Exception:
        for record in reversed(changed):
            target = PROJECT/'data'/record['path']
            if record['had_original']:
                shutil.copy2(backup/record['path'],target)
            else:
                target.unlink(missing_ok=True)
        if vendor_backup is not None:
            live.rename(live.with_name('cn.failed-quality-'+stamp))
            vendor_backup.rename(live)
        if moved_db:
            if db.exists():
                db.rename(db.with_name('cn.failed-quality-'+stamp+'.duckdb'))
            db_backup.rename(db)
        raise
    result = {'published':True,'vendor':str(live),'database':str(db),
              'vendor_backup':str(vendor_backup),'database_backup':str(db_backup),
              'project_backup':str(backup),'validation':str(validation),'files':records,
              'old_release_deleted':False,
              'retention_reason':'Historical corporate-action and derived-price value differences still require reconciliation; coverage gates alone do not prove these values.'}
    for target in (backup/'migration.json',PROJECT/'research/audit_20260923/quality_publication.json'):
        target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
