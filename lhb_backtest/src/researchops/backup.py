"""Portable, offline-verifiable learning evidence copies; never activate workers.

This is a selected experiment's artifact backup, not a whole-market backup or
an automatically executable relocated Research database. Models are not loaded.
"""
from contextlib import closing
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3

from .completed_learning_identity import verify_completed_learning_identity
from .store import Store
from ..technical.artifacts import content_id, digest, write_json


def _ordinary(path, root):
    path=Path(path);root=Path(root).resolve()
    if not path.resolve().is_relative_to(root):raise ValueError('Backup path escaped declared root')
    for item in (path,*path.parents):
        if item.is_symlink() or (hasattr(item,'is_junction') and item.is_junction()):
            raise ValueError('Linked backup path refused')
        if item==root:break
    return path


def verify_learning_backup(folder):
    """Check the copied bytes and SQLite record without reading original paths."""
    folder=Path(folder)
    if not folder.is_dir():raise ValueError('Backup folder missing')
    _ordinary(folder,folder)
    ready=json.loads(_ordinary(folder/'ready.json',folder).read_text(encoding='utf-8'))
    manifest=json.loads(_ordinary(folder/'manifest.json',folder).read_text(encoding='utf-8'))
    if (set(ready)!={'schema','manifest_sha256'} or ready['schema']!='learning-backup-ready-v1'
            or ready['manifest_sha256']!=digest(folder/'manifest.json')):
        raise ValueError('Backup ready record differs')
    required={'schema','experiment_id','source_project','source_research_root','experiment_record_id',
        'learning_identity_id','files','total_bytes','activation_allowed','scope'}
    if (set(manifest)!=required or manifest['schema']!='learning-evidence-backup-v1'
            or manifest['activation_allowed'] is not False or not isinstance(manifest['files'],dict)):
        raise ValueError('Strict offline learning backup required')
    listed=set(manifest['files'])|{'ready.json','manifest.json'}
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual!=listed:raise ValueError('Backup file inventory differs')
    total=0
    for name,record in manifest['files'].items():
        relative=PurePosixPath(name)
        if (not isinstance(name,str) or '\\' in name or relative.is_absolute()
                or relative.as_posix()!=name or any(x in {'.','..'} for x in relative.parts)
                or not (name.startswith('project/') or name in {'metadata/research.sqlite3','experiment.json'})
                or not isinstance(record,dict) or set(record)!={'sha256','bytes'}):
            raise ValueError('Unsafe backup file record')
        path=_ordinary(folder/name,folder)
        if not path.is_file() or path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256']:
            raise ValueError('Backup artifact differs: '+name)
        total+=record['bytes']
    if total!=manifest['total_bytes']:raise ValueError('Backup byte count differs')
    if not {'metadata/research.sqlite3','experiment.json'}<=set(manifest['files']):
        raise ValueError('Backup metadata records missing')
    database=_ordinary(folder/'metadata/research.sqlite3',folder)
    with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Backup SQLite integrity failed')
    ex=Store(folder/'metadata',readonly=True).experiment(manifest['experiment_id'])
    record=json.loads((folder/'experiment.json').read_text(encoding='utf-8'))
    if content_id(ex)!=manifest['experiment_record_id'] or ex!=record or ex['status']!='completed':
        raise ValueError('Copied database and experiment record differ')
    return dict(verified=True,experiment_id=ex['id'],files=len(manifest['files']),bytes=total,
        activation_allowed=False,model_loads=0,model_fits=0,predict_calls=0,
        directory_bytes=total+(folder/'manifest.json').stat().st_size+(folder/'ready.json').stat().st_size,
        scope=manifest['scope'])


def backup_learning_experiment(research,experiment_id,folder,*,max_bytes):
    """Copy inputs, frozen code, model/outputs/logs and one SQLite snapshot.

    The database can include unrelated records; their artifacts are not copied.
    An incomplete output is retained and has no ready record. No overwriting.
    """
    if type(max_bytes) is not int or not 1024<=max_bytes<=100_000_000_000:
        raise ValueError('Finite backup byte cap required')
    project=research.project.resolve();folder=Path(folder)
    if not folder.is_absolute():folder=project/folder
    folder=_ordinary(folder,project/'research')
    if folder.exists():raise ValueError('Backup destination already exists; verify it, do not overwrite')
    identity=verify_completed_learning_identity(research,experiment_id)
    ex=research.store.experiment(experiment_id);record_id=content_id(ex)
    job=_ordinary(research.root/'worker_jobs'/experiment_id,project)
    sources=set()
    for item in ex['proposal']['inputs']:sources.add(_ordinary(project/item['path'],project))
    for item in job.rglob('*'):
        _ordinary(item,project)
        if item.is_file() and '__pycache__' not in item.parts:sources.add(item)
    if folder.resolve().is_relative_to(job.resolve()) or folder.resolve() in {p.resolve() for p in sources}:
        raise ValueError('Backup destination overlaps source artifact directory')
    if sum(p.stat().st_size for p in sources)+research.store.path.stat().st_size>max_bytes:
        raise ValueError('Backup input exceeds byte budget')
    folder.mkdir(parents=True,exist_ok=False);files={}
    for source in sorted(sources):
        name='project/'+source.relative_to(project).as_posix();target=folder/name
        target.parent.mkdir(parents=True,exist_ok=True)
        before=digest(source);shutil.copyfile(source,target)
        if digest(target)!=before or digest(source)!=before:raise ValueError('Source changed during backup')
        files[name]=dict(sha256=before,bytes=target.stat().st_size)
    database=folder/'metadata/research.sqlite3';database.parent.mkdir()
    with research.store.connection() as source,closing(sqlite3.connect(database)) as dest:
        source.backup(dest)
        # A standalone snapshot must not recreate unlisted WAL/SHM sidecars
        # when opened read-only. Change only the copy, never the live database.
        dest.execute("PRAGMA journal_mode=DELETE")
    if content_id(Store(database.parent,readonly=True).experiment(experiment_id))!=record_id:
        raise ValueError('Experiment changed before backup snapshot')
    write_json(folder/'experiment.json',ex)
    for name in ['metadata/research.sqlite3','experiment.json']:
        target=folder/name;files[name]=dict(sha256=digest(target),bytes=target.stat().st_size)
    total=sum(x['bytes'] for x in files.values())
    if total>max_bytes:raise ValueError('Actual backup exceeds byte budget; incomplete copy retained')
    if content_id(research.store.experiment(experiment_id))!=record_id:
        raise ValueError('Original experiment changed during backup')
    verify_completed_learning_identity(research,experiment_id)
    manifest=dict(schema='learning-evidence-backup-v1',experiment_id=experiment_id,
        source_project=str(project),source_research_root=str(research.root),
        experiment_record_id=record_id,learning_identity_id=identity['identity_id'],files=files,total_bytes=total,
        activation_allowed=False,scope='Selected completed learning inputs, frozen code, model, predictions, logs and metadata; other experiments/market history and executable relocation excluded')
    write_json(folder/'manifest.json',manifest)
    ready=dict(schema='learning-backup-ready-v1',manifest_sha256=digest(folder/'manifest.json'))
    ready_bytes=len(json.dumps(ready,ensure_ascii=False,indent=2,allow_nan=False).replace('\n',os.linesep).encode('utf-8'))
    if total+(folder/'manifest.json').stat().st_size+ready_bytes>max_bytes:
        raise ValueError('Backup metadata exceeds byte budget; incomplete copy retained')
    write_json(folder/'ready.json',ready)
    return verify_learning_backup(folder)
