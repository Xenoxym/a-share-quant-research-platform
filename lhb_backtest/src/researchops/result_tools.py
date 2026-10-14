"""Read-only registered-result tools. Never register, fit, predict or backtest."""
from datetime import datetime,timezone
import copy
import json
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from .completed_learning_identity import verify_completed_learning_identity
from .alpha_experiments import verify_inputs
from .alpha_execution import verify_completed
from .account_experiments import native_folder
from .backup import _ordinary
from ..alpharesearch.learning import KEYS,CLOCKS,LABELS,_fingerprint
from ..alpharesearch.split_diagnostics import diagnose_splits
from ..technical.artifacts import content_id,digest,write_json


def write_result_report(research,output,value,*,max_bytes):
    path=__import__('pathlib').Path(output)
    if not path.is_absolute():path=research.project/path
    _ordinary(path,research.project/'research')
    if path.exists():raise ValueError('Existing result report refused')
    if type(max_bytes) is not int or max_bytes<1:raise ValueError('Finite report byte cap required')
    import os
    encoded=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).replace('\n',os.linesep).encode('utf-8')
    if len(encoded)>max_bytes:raise ValueError('Report byte cap exceeded')
    path.parent.mkdir(parents=True,exist_ok=True);write_json(path,value)
    return dict(path=str(path),report_id=content_id(value),model_fits=0,predict_calls=0,account_backtests=0)


def diagnose_registered_learning(research,experiment_id,*,max_rows,segments=None,include_train=False):
    if type(max_rows) is not int or max_rows<1:raise ValueError('Finite diagnostic rows required')
    if type(include_train) is not bool:raise ValueError('Explicit include_train flag required')
    identity=verify_completed_learning_identity(research,experiment_id)
    ex=research.store.experiment(experiment_id);settings=ex['proposal']['spec']['learning']
    folder=research.root/'worker_jobs'/experiment_id/'learn_result'
    pred=folder/'predictions.parquet'
    if pq.ParquetFile(pred).metadata.num_rows>max_rows:raise ValueError('Prediction diagnostic row cap exceeded')
    paths=verify_inputs(research.project,ex['proposal']['inputs'],ex['proposal']['spec']['budget']['max_input_bytes'])
    source=json.loads((folder/'learning_receipt.json').read_text(encoding='utf-8'))
    if source['schema']=='causal-learning-receipt-v2':
        target=dict(kind='raw_return',unit='decimal_return',predicted_score_is_raw_return=True)
    elif source['schema']=='causal-learning-receipt-v3':
        t=source['training_target_receipt']
        target=dict(kind=t['spec']['kind'],unit=t['unit'],predicted_score_is_raw_return=t['predicted_score_is_raw_return'])
        if source['score_unit']!=target['unit'] or source['predicted_score_is_raw_return']!=target['predicted_score_is_raw_return']:
            raise ValueError('Completed score target metadata differs')
    else:raise ValueError('Known causal score target receipt required')
    selected=(ds.field('trade_date')>=settings['predict_start']) & (ds.field('trade_date')<=settings['predict_end'])
    if ds.dataset(paths['labels'],format='parquet').count_rows(filter=selected)>max_rows:
        raise ValueError('Label diagnostic row cap exceeded before loading')
    scores=pd.read_parquet(pred)
    labels=pd.read_parquet(paths['labels'],columns=KEYS+LABELS,
        filters=[('trade_date','>=',settings['predict_start']),('trade_date','<=',settings['predict_end'])])
    if segments is None:
        segments=[dict(name='test',start=settings['predict_start'],end=settings['predict_end'],
            role='post_fit_retrospective',used_for_selection=True)]
    artifact_path=folder/'training_artifact_receipt.json';trace_path=folder/'training_trace.parquet'
    artifact=json.loads(artifact_path.read_text(encoding='utf-8')) if artifact_path.is_file() else {}
    available=artifact.get('train_scores_saved') is True and trace_path.is_file()
    train_status='not_requested' if not include_train else 'not_saved'
    training_error=None;trace_hash=None
    if include_train and available:
        manifest_path=folder/'manifest.json'
        if digest(manifest_path)!=identity['output_manifest_hash']:raise ValueError('Training trace output manifest differs')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        for name in ['training_trace.parquet','training_artifact_receipt.json']:
            if digest(folder/name)!=manifest['artifacts'].get(name):raise ValueError('Saved training artifact differs')
        footer=pq.ParquetFile(trace_path);needed=KEYS+CLOCKS+['raw_target_return','label_start','label_end','label_known_at','target_reason','target_return_used','weight_used','train_score']
        if not set(needed)<=set(footer.schema.names):raise ValueError('Legacy trace lacks raw targets/clocks; explicit legacy adapter required')
        if footer.metadata.num_rows+len(scores)>max_rows:raise ValueError('Combined train/test diagnostic row cap exceeded before loading')
        if footer.metadata.num_rows!=artifact['rows'] or artifact['rows']!=source['mature_training_rows']:
            raise ValueError('Saved training trace mature row count differs')
        trace=pd.read_parquet(trace_path,columns=needed)
        if not trace.target_reason.eq('present').all():raise ValueError('Saved training trace has unusable mature targets')
        training_scores=trace[KEYS+CLOCKS].copy();training_scores['score']=trace.train_score
        training_labels=trace[KEYS+['label_start','label_end','label_known_at']].copy()
        training_labels['target_return']=trace.raw_target_return;training_labels['label_observed']=True
        scores=pd.concat([training_scores,scores[KEYS+CLOCKS+['score']]],ignore_index=True)
        labels=pd.concat([training_labels[KEYS+LABELS],labels],ignore_index=True)
        if not any(part['name']=='train' for part in segments):
            segments=[dict(name='train',start=settings['train_start'],end=settings['train_end'],role='in_sample_replay',used_for_selection=True)]+segments
        from sklearn.metrics import mean_squared_error
        training_error=dict(weighted_residual_mse=float(mean_squared_error(trace.target_return_used,trace.train_score,sample_weight=trace.weight_used)),
            target='actual saved target_return_used, with actual row weights; regularization and optimizer scaling excluded',account_backtest=False)
        trace_hash=manifest['artifacts']['training_trace.parquet'];train_status='included_actual_saved_fit_scores'
    report=diagnose_splits(scores,labels,segments,fit_cutoff=settings['fit_cutoff'],
        as_of=datetime.now(timezone.utc).isoformat(),max_rows=max_rows,score_target=target)
    if (digest(pred)!=identity['predictions_sha256'] or digest(folder/'learning_receipt.json')!=identity['learning_receipt_sha256']
            or research.store.experiment(experiment_id)!=ex):
        raise ValueError('Completed source changed during diagnostics')
    if trace_hash is not None and digest(trace_path)!=trace_hash:raise ValueError('Training trace changed during diagnostics')
    verify_inputs(research.project,ex['proposal']['inputs'],ex['proposal']['spec']['budget']['max_input_bytes'])
    return dict(schema='registered-score-diagnostics-v1',experiment_id=experiment_id,
        completed_learning_identity=identity['identity_id'],diagnostics=report,
        saved_train_scores_available=available,training_diagnostic_status=train_status,training_residual=training_error,
        selection_history='Retrospective research; not a claim of previously unseen data')


def account_comparison_index(research,experiment_ids):
    """Group only exactly matching declared account conditions; no global winner."""
    if (not isinstance(experiment_ids,list) or not 1<=len(experiment_ids)<=100
            or len(set(experiment_ids))!=len(experiment_ids)):raise ValueError('One to 100 distinct account IDs required')
    groups={}
    for eid in experiment_ids:
        ex=research.store.experiment(eid)
        if ex['status']!='completed' or ex['proposal'].get('kind')!='alpha_account':raise ValueError('Completed scored accounts required')
        verify_completed(research,ex)
        p=ex['proposal']
        cfg=json.loads((research.root/'worker_jobs'/eid/'input.json').read_text(encoding='utf-8'))
        native=native_folder(cfg)
        read=lambda name:json.loads((native/name).read_text(encoding='utf-8'))
        portfolio=read('alpha_portfolio_receipt.json');receipt=read('alpha_account_receipt.json');result=read('result.json')
        policy=copy.deepcopy(p['spec']['portfolio']);policy.pop('name',None)
        account=copy.deepcopy(p['spec']['account']);account.get('strategy',{}).pop('name',None)
        decisions=pd.read_parquet(native/'decisions.parquet',columns=['trade_date','stock_code','reason'])
        eligible=decisions.loc[decisions.reason.eq('eligible_external_score'),['trade_date','stock_code']].sort_values(['trade_date','stock_code']).reset_index(drop=True)
        if eligible.duplicated().any():raise ValueError('Duplicate eligible account keys')
        bundle=read('alpha_scoring_code.json')
        code=bundle.get('consumer_files') if bundle.get('verification_mode')=='reuse_completed_identity_v1' else bundle.get('files')
        if not isinstance(code,dict) or not code:raise ValueError('Complete archived scoring source required')
        account_code={name:value['sha256'] for name,value in code.items()}
        conditions=dict(snapshot_id=p['spec']['snapshot_id'],snapshot_manifest_hash=p['spec']['snapshot_manifest_hash'],
            portfolio=policy,account=account,scenarios=p['spec']['scenarios'],
            eligible_membership=_fingerprint(eligible),references=portfolio['reference_inputs'],
            calendar_id=portfolio['calendar_id'],planned_horizons=portfolio['planned_horizons'],
            account_code=account_code,native_code_hash=result['code_hash'],portfolio_implementation=portfolio['implementation_hashes'],
            holding_policy=p['spec'].get('holding_policy'))
        group=content_id(conditions)
        groups.setdefault(group,dict(group_id=group,conditions=conditions,accounts=[]))['accounts'].append(dict(
            experiment_id=eid,score_source=p['spec']['score_source'],native_run=native.name,
            code_bundle_id=receipt['code_bundle_id'],result_sha256=digest(native/'result.json'),metrics=[x['metrics'] for x in result['portfolios']],
            unseen_claim=False))
        if research.store.experiment(eid)!=ex:raise ValueError('Account changed during index')
    return dict(schema='account-comparison-index-v1',groups=list(groups.values()),
        scope='Compare within a group only; historical protocols in other groups have no common winner. Existing results remain retrospective.',
        model_fits=0,predict_calls=0,account_backtests=0)
