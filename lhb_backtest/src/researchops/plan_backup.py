"""Selected completed plan evidence, copied for offline inspection, never activated."""
from contextlib import closing
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import sqlite3
from .backup import _ordinary
from .workflow import ExecutionPlans
from .alpha_execution import verify_completed
from .account_experiments import native_folder
from .store import Store
from ..technical.artifacts import content_id,digest,write_json


def backup_plan_evidence(research,plan_id,folder,*,max_bytes):
    if type(max_bytes) is not int or not 1024<=max_bytes<=100_000_000_000:raise ValueError('Finite plan backup byte cap required')
    project=research.project.resolve();folder=Path(folder)
    if not folder.is_absolute():folder=project/folder
    _ordinary(folder,project/'research')
    if folder.exists():raise ValueError('Existing plan backup refused')
    plans=ExecutionPlans(research);state=plans.status(plan_id)
    if not state['steps'] or any(s['status']!='completed' or not s['experiment_id'] for s in state['steps']):
        raise ValueError('Every selected plan step must be completed')
    ids=list(dict.fromkeys(s['experiment_id'] for s in state['steps']));records={};sources=set();directories=[]
    for eid in ids:
        ex=research.store.experiment(eid)
        if ex['status']!='completed':raise ValueError('Completed original experiments required')
        verify_completed(research,ex);records[eid]=ex
        for item in ex['proposal']['inputs']:sources.add(_ordinary(project/item['path'],project))
        job=_ordinary(research.root/'worker_jobs'/eid,project);directories.append(job)
        if ex['proposal'].get('kind')=='alpha_account':
            cfg=json.loads((job/'input.json').read_text(encoding='utf-8'))
            directories.extend([_ordinary(native_folder(cfg),project),_ordinary(research.root/'snapshots'/ex['proposal']['spec']['snapshot_id'],project)])
    for directory in directories:
        if folder.resolve().is_relative_to(directory.resolve()):raise ValueError('Backup overlaps source directory')
        for path in directory.rglob('*'):
            _ordinary(path,project)
            if path.is_file() and '__pycache__' not in path.parts:sources.add(path)
    if sum(p.stat().st_size for p in sources)+research.store.path.stat().st_size>max_bytes:
        raise ValueError('Plan evidence exceeds byte cap before copying')
    folder.mkdir(parents=True,exist_ok=False);files={}
    for source in sorted(sources):
        name='project/'+source.relative_to(project).as_posix();target=folder/name
        target.parent.mkdir(parents=True,exist_ok=True);expected=digest(source);shutil.copyfile(source,target)
        if digest(target)!=expected or digest(source)!=expected:raise ValueError('Plan source changed during copy')
        files[name]=dict(sha256=expected,bytes=target.stat().st_size)
    database=folder/'metadata/research.sqlite3';database.parent.mkdir()
    with research.store.connection() as source,closing(sqlite3.connect(database)) as dest:
        source.backup(dest);dest.execute('PRAGMA journal_mode=DELETE')
    write_json(folder/'plan.json',state);write_json(folder/'experiments.json',records)
    for name in ['metadata/research.sqlite3','plan.json','experiments.json']:
        p=folder/name;files[name]=dict(sha256=digest(p),bytes=p.stat().st_size)
    for ex in records.values():verify_completed(research,ex)
    if plans.status(plan_id)!=state or any(research.store.experiment(eid)!=ex for eid,ex in records.items()):
        raise ValueError('Selected plan changed during backup')
    manifest=dict(schema='execution-evidence-backup-v1',plan_id=plan_id,plan_state_id=content_id(state),
        experiment_records_id=content_id(records),experiment_ids=ids,files=files,total_bytes=sum(x['bytes'] for x in files.values()),
        activation_allowed=False,scope='Selected plan inputs, frozen jobs, native accounts, snapshots and SQLite; unrelated experiments and provenance ancestors outside this plan excluded; inspection only, no relocated execution')
    write_json(folder/'manifest.json',manifest)
    ready=dict(schema='execution-evidence-ready-v1',manifest_sha256=digest(folder/'manifest.json'))
    metadata=len(json.dumps(ready,ensure_ascii=False,indent=2,allow_nan=False).replace('\n',os.linesep).encode('utf-8'))
    if manifest['total_bytes']+(folder/'manifest.json').stat().st_size+metadata>max_bytes:
        raise ValueError('Plan backup metadata exceeds byte cap; incomplete copy retained')
    write_json(folder/'ready.json',ready)
    return verify_plan_backup(folder)


def verify_plan_backup(folder):
    """Read only copied bytes and copied SQLite; original project may be absent."""
    folder=Path(folder);_ordinary(folder,folder)
    read=lambda name:json.loads(_ordinary(folder/name,folder).read_text(encoding='utf-8'))
    manifest=read('manifest.json');ready=read('ready.json')
    if (ready!=dict(schema='execution-evidence-ready-v1',manifest_sha256=digest(folder/'manifest.json'))
            or manifest.get('schema')!='execution-evidence-backup-v1' or manifest.get('activation_allowed') is not False):
        raise ValueError('Strict completed plan backup required')
    required={'schema','plan_id','plan_state_id','experiment_records_id','experiment_ids','files','total_bytes','activation_allowed','scope'}
    if set(manifest)!=required or not isinstance(manifest['files'],dict):raise ValueError('Strict plan backup manifest required')
    files=manifest['files'];actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual!=set(files)|{'manifest.json','ready.json'}:raise ValueError('Plan backup inventory differs')
    total=0
    for name,entry in files.items():
        rel=PurePosixPath(name)
        if ('\\' in name or rel.is_absolute() or rel.as_posix()!=name or '..' in rel.parts
                or not (name.startswith('project/') or name in {'metadata/research.sqlite3','plan.json','experiments.json'})):
            raise ValueError('Unsafe plan backup path')
        path=_ordinary(folder/name,folder)
        if entry!=dict(sha256=digest(path),bytes=path.stat().st_size):raise ValueError('Plan backup artifact differs')
        total+=entry['bytes']
    if total!=manifest['total_bytes']:raise ValueError('Plan backup byte count differs')
    state=read('plan.json');records=read('experiments.json')
    if content_id(state)!=manifest['plan_state_id'] or content_id(records)!=manifest['experiment_records_id']:
        raise ValueError('Plan backup record differs')
    database=folder/'metadata/research.sqlite3'
    with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Copied SQLite integrity failed')
        db.row_factory=sqlite3.Row
        row=db.execute('SELECT * FROM execution_plans WHERE id=?',(manifest['plan_id'],)).fetchone()
        steps=[dict(s) for s in db.execute('SELECT name,experiment_id,status FROM execution_steps WHERE plan_id=?',(manifest['plan_id'],))]
    if row is None or {s['name']:s for s in steps}!={s['name']:{k:s[k] for k in ('name','experiment_id','status')} for s in state['steps']}:
        raise ValueError('Copied SQLite and plan step records differ')
    if row is not None:
        definition=json.loads(row['definition'])
        header=dict(plan_id=manifest['plan_id'],task_id=row['task_id'],starts=row['starts'],max_starts=definition['max_starts'],began=row['began'],deadline=row['deadline'])
        if any(state.get(k)!=v for k,v in header.items()):raise ValueError('Copied plan header differs')
    if set(records)!=set(manifest['experiment_ids']) or any(s['status']!='completed' for s in steps):raise ValueError('Completed chain records required')
    for record in records.values():
        for item in record['proposal']['inputs']:
            entry=files.get('project/'+item['path'])
            if entry!=dict(sha256=item['sha256'],bytes=item['bytes']):raise ValueError('Copied original registered input differs')
    store=Store(folder/'metadata',readonly=True)
    if any(store.experiment(eid)!=record or record['status']!='completed' for eid,record in records.items()):
        raise ValueError('Copied experiment records differ')
    return dict(plan_id=manifest['plan_id'],experiments=len(records),files=len(files),bytes=total,
        offline_verified=True,activation_allowed=False,scope=manifest['scope'])
