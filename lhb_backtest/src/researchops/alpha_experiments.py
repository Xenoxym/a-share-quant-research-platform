"""First-class frozen feature-batch registration and bounded E12 dispatch.

Registration, numerical execution, and output identity audits have separate
scopes. Neither feature execution nor hashes prove vendor history/account returns.
"""
from datetime import date
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import time

import pandas as pd

from .store import text
from ..alpharesearch.batch import AlphaBatchSpec
from ..alpharesearch.features.base import FeatureBlock
from ..alpharesearch.registry import FeatureRegistry,definitions_for_block,verify_materialization,SHA
from ..alpharesearch.operators import ExpressionEvaluator,LeafBinding
from ..technical.artifacts import content_id,digest,write_json,verify_artifacts

ROLE=re.compile(r'^[a-z][a-z0-9_-]{0,79}$')


def _path(project,name):
    if not isinstance(name,str) or '\\' in name:raise ValueError('Portable project-relative input path required')
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts or any(p.startswith('.') for p in relative.parts):raise ValueError('Input path must stay in named project data/research')
    if relative.parts[0] not in {'data','research'}:raise ValueError('Batch inputs must be within project data/research')
    project=Path(project).resolve();candidate=project/relative
    for p in [candidate,*candidate.parents]:
        if p==project:break
        if p.is_symlink() or (hasattr(p,'is_junction') and p.is_junction()):raise ValueError('Linked batch input path refused')
    if not candidate.resolve().is_relative_to(project) or not candidate.is_file():raise ValueError('Input file missing or escaped project')
    return candidate


def verify_inputs(project,records,max_bytes):
    if not isinstance(records,list) or not 1<=len(records)<=512:raise ValueError('Frozen input records required')
    total=0;roles={}
    for record in records:
        if not isinstance(record,dict) or set(record)!={'role','path','sha256','bytes'}:raise ValueError('Strict input record fields required')
        role=record['role']
        if not isinstance(role,str) or not ROLE.fullmatch(role) or role in roles or not isinstance(record['sha256'],str) or not SHA.fullmatch(record['sha256']) or type(record['bytes']) is not int or record['bytes']<0:raise ValueError('Unique input role with size/SHA required')
        total+=record['bytes']
        if total>max_bytes:raise ValueError('Frozen input byte budget exceeded')
        path=_path(project,record['path']);before=path.stat()
        if before.st_size!=record['bytes'] or digest(path)!=record['sha256']:raise ValueError('Frozen input content changed')
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Input changed while verifying')
        roles[role]=path
    if len({p.resolve() for p in roles.values()})!=len(roles):raise ValueError('Distinct input roles cannot alias the same file')
    return roles


def inspect_inputs(project,proposal,*,include_evaluator=False):
    spec_raw=proposal['spec']
    if not isinstance(spec_raw,dict) or not isinstance(spec_raw.get('budget'),dict) or type(spec_raw['budget'].get('max_input_bytes')) is not int or spec_raw['budget']['max_input_bytes']<1:raise ValueError('Bounded input-byte budget required before reading')
    budget=spec_raw['budget'];roles=verify_inputs(project,proposal['inputs'],budget['max_input_bytes'])
    registry=FeatureRegistry.from_dict(json.loads(roles['registry'].read_text(encoding='utf-8')));spec=AlphaBatchSpec.from_dict(spec_raw,registry)
    if set(roles)!=spec.input_roles:raise ValueError('Input roles must exactly cover the frozen protocol')
    doc=spec.to_dict();blocks={}
    for name,record in doc['blocks'].items():
        meta=json.loads(roles[record['definition']].read_text(encoding='utf-8'));units=meta.pop('units')
        block=FeatureBlock(pd.read_parquet(roles[record['values']]),pd.read_parquet(roles[record['missing']]),units,meta).validate();blocks[name]=block
        current={d.definition_id:d for d in definitions_for_block(block)}
        receipt=verify_materialization(json.loads(roles[record['receipt']].read_text(encoding='utf-8')))
        artifact_expected={roles[record[r]].name:digest(roles[record[r]]) for r in ('values','missing')}
        if dict(receipt['artifacts'])!=artifact_expected or set(receipt['definition_ids'])!=set(current) or receipt['rows']!=len(block.values) or receipt['source_id']!=meta['source_id'] or receipt['source_version']!=meta['version_id'] or receipt['vintage']!=meta['vintage']:raise ValueError('Materialization receipt does not bind actual block artifacts/definitions')
        for key,binding in doc['bindings'].items():
            if binding['block']!=name:continue
            definition=current.get(binding['definition_id'])
            if definition is None or definition.key!=key or definition.key.split('.',1)[1]!=binding['column'] or binding['materialization_id']!=receipt['materialization_id']:raise ValueError('Bound definition/column/receipt does not match frozen block')
    leaves={key:LeafBinding(blocks[b['block']],b['column'],b['definition_id'],b['materialization_id']) for key,b in doc['bindings'].items()}
    calendar=json.loads(roles['calendar'].read_text(encoding='utf-8'))
    if not isinstance(calendar,dict) or set(calendar)!={'trade_dates'}:raise ValueError('Explicit full calendar document required')
    full_days=calendar['trade_dates']
    if not isinstance(full_days,list) or not full_days or any(not isinstance(d,str) or date.fromisoformat(d).isoformat()!=d for d in full_days) or full_days!=sorted(set(full_days)):raise ValueError('Canonical unique sorted full calendar required')
    r=doc['ranges'];days=[d for d in full_days if r['warmup_start']<=d<=r['development_end']]
    p=pd.read_parquet(roles['membership']);p=p.loc[p.trade_date.between(r['warmup_start'],r['development_end'])].copy()
    clocks=pd.read_parquet(roles['decision_clocks']);clocks=clocks.loc[clocks.trade_date.between(r['warmup_start'],r['development_end'])].copy()
    if not p.trade_date.isin(days).all() or not clocks.trade_date.isin(days).all():raise ValueError('Development membership/clock date missing from frozen calendar')
    evaluator=ExpressionEvaluator(registry,leaves,calendar=days,membership=p,decision_clocks=clocks,universe_id=doc['universe_id'],allow_weak_vintage=doc['allow_weak_vintage'],max_grid_cells=budget['max_grid_cells'],max_estimated_buffer_bytes=budget['max_buffer_bytes'])
    leaf_bytes=sum(len(evaluator.days)*(len(evaluator.stocks) if registry.resolve(key,leaves[key].definition_id).domain=='stock_day' else 1)*9 for key in leaves)
    if leaf_bytes>budget['max_buffer_bytes']:raise ValueError('Referenced numerical leaf buffer budget exceeded before preparation')
    # Validate only referenced leaves, with source clocks/version gates, no formula fit.
    for key,binding in leaves.items():evaluator._leaf(key,binding.definition_id)
    if not any(r['development_start']<=d<=r['development_end'] for d in p.trade_date):raise ValueError('No development decision rows in the frozen input')
    info={'execution_days':days,'execution_rows':len(p),'holdout_rows_executed':0,'source_files_by_reference':True,'historical_source_certified':False}
    if include_evaluator:info['_evaluator']=evaluator
    return spec,registry,info


def register(research,session,proposal):
    required={'kind','hypothesis','expected_observation','falsification','spec','inputs'}
    if not isinstance(proposal,dict) or not required<=set(proposal) or set(proposal)-required-{'timeout_seconds','resources'} or proposal['kind']!='alpha_batch':raise ValueError('Strict alpha_batch registration fields required')
    with research.store.connection() as db:research.store.owned(db,session)
    from .resources import WorkerResources
    resources=WorkerResources.from_dict(proposal.get('resources',WorkerResources().to_dict())).to_dict()
    spec,registry,inspection=inspect_inputs(research.project,proposal)
    timeout=proposal.get('timeout_seconds',1200)
    if type(timeout) is not int or not 30<=timeout<=7200:raise ValueError('Bounded batch execution timeout required')
    sources=sorted(p for p in (research.project/'src').rglob('*.py') if '__pycache__' not in p.parts)
    if not sources or not (research.project/'src/alpharesearch/batch.py').is_file():raise ValueError('Missing registered batch implementation')
    for source in sources:
        if source.is_symlink() or not source.resolve().is_relative_to(research.project):raise ValueError('Source implementation escapes project')
        for parent in source.parents:
            if parent==research.project:break
            if parent.is_symlink() or (hasattr(parent,'is_junction') and parent.is_junction()):raise ValueError('Linked source implementation refused')
    inventory={p.relative_to(research.project).as_posix():digest(p) for p in sources}
    environment={'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ('numpy','pandas','pyarrow','tzdata','psutil')}}
    p={k:text(proposal[k],k) for k in ('hypothesis','expected_observation','falsification')}
    p.update(kind='alpha_batch',spec=spec.to_dict(),inputs=proposal['inputs'],timeout_seconds=timeout,registry_version=registry.version_id,engine_inventory=inventory,engine_hash=content_id(inventory),environment=environment,inspection=inspection,submitted_by=session['worker'],evaluation_scope='retrospective_time_split',candidate_count=len(spec.planned_attempts),planned_model_fits=0,planned_accounts=0,resources=resources)
    canonical=json.loads(json.dumps(p));canonical['spec'].pop('name')
    for key in ('hypothesis','expected_observation','falsification','timeout_seconds','submitted_by'):canonical.pop(key)
    # File roles are sets by meaning; descriptive order must not consume another try.
    canonical['inputs']=sorted(canonical['inputs'],key=lambda x:x['role'])
    experiment=research.store.register(session,p,content_id(canonical));job=research.root/'worker_jobs'/experiment['id']
    if (job/'ready.json').is_file():verify_registered(research,experiment);return experiment
    if job.exists():raise ValueError('Previous freeze is incomplete; preserve the failed try')
    proposal=experiment['proposal'];code=job/'code'
    try:
        code.mkdir(parents=True)
        for name,sha in proposal['engine_inventory'].items():
            target=code/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(research.project/name,target)
            if digest(target)!=sha:raise ValueError('Code changed while freezing batch')
        write_json(job/'code_manifest.json',{'artifacts':{name:sha for name,sha in proposal['engine_inventory'].items()}})
        write_json(job/'input.json',dict(proposal,project=str(research.project),root=str(research.root),task_id=session['task_id'],experiment_id=experiment['id']))
        write_json(job/'candidate_manifest.json',{'schema':'planned-alpha-attempts-v1','attempts':spec.planned_attempts,'candidate_count':len(spec.planned_attempts),'fit_budget':0,'account_budget':0,'counting_rule':'All registered candidates count; identical formulas with distinct names are still counted; failed/cancelled batch budget is not refunded'})
        # Verify source files again after code/input freeze; reference does not copy them.
        verify_inputs(research.project,proposal['inputs'],proposal['spec']['budget']['max_input_bytes'])
        write_json(job/'ready.json',{'kind':'alpha_batch','input_hash':digest(job/'input.json'),'code_manifest_hash':digest(job/'code_manifest.json'),'candidate_manifest_hash':digest(job/'candidate_manifest.json')})
    except Exception as exc:
        with research.store.connection(True) as db:
            db.execute("UPDATE experiments SET status='failed',error=?,updated=? WHERE id=?",(str(exc),time.time(),experiment['id']))
            research.store.event(db,session['task_id'],session['worker'],'freeze_failed',{'experiment_id':experiment['id'],'error':str(exc)})
        raise
    return experiment


def verify_registered(research,experiment):
    if experiment['proposal'].get('kind')!='alpha_batch':raise ValueError('Not an alpha batch')
    job=research.root/'worker_jobs'/experiment['id'];ready=json.loads((job/'ready.json').read_text(encoding='utf-8'))
    expected={'kind':'alpha_batch','input_hash':digest(job/'input.json'),'code_manifest_hash':digest(job/'code_manifest.json'),'candidate_manifest_hash':digest(job/'candidate_manifest.json')}
    if ready!=expected:raise ValueError('Frozen batch manifests changed')
    inventory=json.loads((job/'code_manifest.json').read_text(encoding='utf-8'))
    cfg=json.loads((job/'input.json').read_text(encoding='utf-8'));p=experiment['proposal']
    if inventory['artifacts']!=p['engine_inventory'] or content_id(inventory['artifacts'])!=p['engine_hash'] or any(cfg.get(k)!=v for k,v in p.items()) or cfg['experiment_id']!=experiment['id'] or cfg['task_id']!=experiment['task_id'] or cfg['project']!=str(research.project) or cfg['root']!=str(research.root):raise ValueError('Frozen batch does not match registration')
    if set(inventory)!={'artifacts'}:raise ValueError('Strict frozen code manifest required')
    for name in inventory['artifacts']:
        path=job/'code'/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.resolve().is_relative_to(job/'code'):raise ValueError('Frozen implementation path escaped code root')
    code_root=job/'code';actual_paths=list(code_root.rglob('*'))
    if any(path.is_symlink() or (hasattr(path,'is_junction') and path.is_junction()) for path in actual_paths):raise ValueError('Frozen code file set contains linked paths')
    actual_files={path.relative_to(code_root).as_posix() for path in actual_paths if path.is_file()}
    if actual_files!=set(inventory['artifacts']):raise ValueError('Frozen code file set contains missing or unregistered import files')
    verify_artifacts(code_root,inventory)
    verify_inputs(cfg['project'],p['inputs'],p['spec']['budget']['max_input_bytes'])
    if 'resources' in p:
        from .resources import WorkerResources
        WorkerResources.from_dict(p['resources'])
    registry_record=next(x for x in p['inputs'] if x['role']=='registry');registry=FeatureRegistry.from_dict(json.loads(_path(cfg['project'],registry_record['path']).read_text(encoding='utf-8')))
    spec=AlphaBatchSpec.from_dict(p['spec'],registry);plan=json.loads((job/'candidate_manifest.json').read_text(encoding='utf-8'))
    if plan!={'schema':'planned-alpha-attempts-v1','attempts':spec.planned_attempts,'candidate_count':len(spec.planned_attempts),'fit_budget':0,'account_budget':0,'counting_rule':'All registered candidates count; identical formulas with distinct names are still counted; failed/cancelled batch budget is not refunded'} or p['candidate_count']!=len(spec.planned_attempts) or registry.version_id!=p['registry_version']:raise ValueError('Frozen candidate/registry plan changed')
    return {'verified':True,'kind':'alpha_batch_registration','candidate_count':len(spec.planned_attempts),'executed_candidates':0,'fits':0,'accounts':0,'scope':'registration and input/code identity only; not execution or strategy validity'}


def execute(research,session,eid):
    from .alpha_execution import execute as execute_worker
    return execute_worker(research,session,eid)


def verify_completed(research,experiment):
    from .alpha_execution import verify_completed as verify_worker
    return verify_worker(research,experiment)


def recover(research,experiment,receipt=None):
    from .alpha_execution import recover as recover_worker
    return recover_worker(research,experiment,receipt)
