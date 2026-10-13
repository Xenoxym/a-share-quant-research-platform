"""Evidence copy verification is independent of source disk and model loading."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
import pytest
from src.researchops.backup import backup_learning_experiment,verify_learning_backup
from src.technical.artifacts import digest,write_json
from tests.test_completed_learning_identity import fixture_record as identity_fixture
from src.researchops.store import Store,encode


def fixture_record(tmp_path):
    ops,ex,job=identity_fixture(tmp_path)
    store=Store(ops.root)
    with store.connection(True) as db:
        db.execute("INSERT INTO experiments VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (ex["id"],ex["task_id"],"backup-fixture",encode(ex["proposal"]),"completed",1.,1.,ex["pid"],None,encode(ex["result"]),None))
    ops.store=Store(ops.root,readonly=True)
    return ops,ops.store.experiment(ex["id"]),job


def test_backup_survives_unavailable_source_and_cli_needs_no_database(tmp_path,monkeypatch):
    ops,ex,job=fixture_record(tmp_path);folder=ops.project/'research/backup'
    import joblib
    monkeypatch.setattr(joblib,'load',lambda *a:pytest.fail('never load checkpoints'))
    result=backup_learning_experiment(ops,ex['id'],folder,max_bytes=100_000_000)
    assert result['verified'] and not result['activation_allowed'] and result['model_loads']==0
    before=ops.store.experiment(ex['id']);assert before==ex
    # Copy the backup away, then make the original project unavailable. No
    # backup code should follow frozen absolute input.json paths during verify.
    recovered=tmp_path/'recovered';shutil.copytree(folder,recovered)
    ops.project.rename(tmp_path/'offline-original')
    assert verify_learning_backup(recovered)==result
    command=[sys.executable,str(Path(__file__).resolve().parents[1]/'tools/execution.py'),
        '--project',str(tmp_path/'nonexistent'),'verify-backup','--folder',str(recovered)]
    process=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert process.returncode==0,process.stderr+process.stdout
    assert json.loads(process.stdout)['data']==result


@pytest.mark.parametrize('fault',['bytes','missing','unlisted','traversal','database','ready'])
def test_corrupted_or_incomplete_copy_rejected(tmp_path,fault):
    ops,ex,_=fixture_record(tmp_path);folder=ops.project/'research/backup'
    backup_learning_experiment(ops,ex['id'],folder,max_bytes=100_000_000)
    manifest=json.loads((folder/'manifest.json').read_text())
    target=folder/next(n for n in manifest['files'] if n.startswith('project/'))
    if fault=='bytes':target.write_bytes(b'corrupt')
    elif fault=='missing':target.unlink()
    elif fault=='unlisted':(folder/'extra.txt').write_text('unexpected')
    elif fault=='ready':(folder/'ready.json').unlink()
    else:
        if fault=='traversal':manifest['files']['project/../../outside']=manifest['files'].pop(target.relative_to(folder).as_posix())
        else:
            database=folder/'metadata/research.sqlite3';database.write_bytes(b'not a database')
            manifest['files']['metadata/research.sqlite3']={'sha256':digest(database),'bytes':database.stat().st_size}
            manifest['total_bytes']=sum(r['bytes'] for r in manifest['files'].values())
        write_json(folder/'manifest.json',manifest)
        write_json(folder/'ready.json',{'schema':'learning-backup-ready-v1','manifest_sha256':digest(folder/'manifest.json')})
    with pytest.raises((ValueError,OSError, __import__('sqlite3').DatabaseError)):verify_learning_backup(folder)


def test_budget_output_and_copy_interruption_preserve_originals(tmp_path,monkeypatch):
    ops,ex,_=fixture_record(tmp_path);folder=ops.project/'research/backup'
    with pytest.raises(ValueError,match='budget'):backup_learning_experiment(ops,ex['id'],folder,max_bytes=1024)
    assert not folder.exists()
    with pytest.raises(ValueError,match='escaped'):backup_learning_experiment(ops,ex['id'],tmp_path/'outside',max_bytes=100_000_000)
    monkeypatch.setattr(shutil,'copyfile',lambda *a:(_ for _ in ()).throw(OSError('copy interrupted')))
    with pytest.raises(OSError,match='interrupted'):backup_learning_experiment(ops,ex['id'],folder,max_bytes=100_000_000)
    assert folder.exists() and not (folder/'ready.json').exists() and ops.store.experiment(ex['id'])==ex
    with pytest.raises(ValueError,match='already exists'):backup_learning_experiment(ops,ex['id'],folder,max_bytes=100_000_000)


def test_budget_includes_manifest_and_ready_records(tmp_path):
    ops,ex,_=fixture_record(tmp_path);first=ops.project/'research/first'
    result=backup_learning_experiment(ops,ex['id'],first,max_bytes=100_000_000)
    assert result['directory_bytes']==sum(p.stat().st_size for p in first.rglob('*') if p.is_file())
    assert result['directory_bytes']>result['bytes']
    capped=ops.project/'research/capped'
    with pytest.raises(ValueError,match='budget'):
        backup_learning_experiment(ops,ex['id'],capped,max_bytes=result['bytes'])
    assert not (capped/'ready.json').exists()
