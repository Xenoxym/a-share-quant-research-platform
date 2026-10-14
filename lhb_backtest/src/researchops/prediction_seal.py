"""Prospective local score bundles. No backdating, model loading or outcome reads.

This is a local reproducibility record, not a trusted external timestamp service.
"""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from .backup import _ordinary
from .completed_learning_identity import verify_completed_learning_identity
from ..alpharesearch.learning import KEYS,CLOCKS
from ..alpharesearch.contracts import timestamp,timestamps
from ..alpharesearch.portfolio import ScorePortfolioSpec
from ..alpharesearch.account import _account_spec
from ..technical.contracts import ResearchSpec
from ..technical.artifacts import content_id,digest,write_json


def _rules(protocol):
    if (not isinstance(protocol,dict) or set(protocol)!={'schema','portfolio','account','scenarios','evaluation_end'}
            or protocol['schema']!='prospective-account-rules-v1'
            or protocol['scenarios']!=['zero_transaction_cost','configured']):
        raise ValueError('Fixed prospective account rules required')
    policy=ScorePortfolioSpec.from_dict(protocol['portfolio'])
    _account_spec(ResearchSpec.from_dict(protocol['account']),policy.to_dict())
    timestamp(protocol['evaluation_end'])
    return policy.to_dict()


def _check_clocks(scores,protocol,sealed_at):
    policy=_rules(protocol);now=timestamp(sealed_at)
    if (scores.empty or not set(KEYS+CLOCKS+['score','fit_id','prediction_id'])==set(scores)
            or scores[KEYS].isna().any().any() or scores.duplicated(KEYS).any()
            or scores.sample_id.duplicated().any()):raise ValueError('Complete unique scored membership required')
    if not np.isfinite(scores.score.to_numpy(dtype=float)).all():raise ValueError('Finite sealed scores required')
    frame=scores.copy()
    for key in CLOCKS:frame[key]=timestamps(frame[key])
    if (frame.decision_at.gt(now)|frame.execution_at.le(now)|frame.target_end_at.le(frame.execution_at)).any():
        raise ValueError('Prospective sealing requires decision <= actual seal < execution < target end for every row')
    if (not frame.trade_date.between(policy['start'],policy['end']).all()
            or frame.target_end_at.gt(timestamp(protocol['evaluation_end'])).any()):
        raise ValueError('Scores exceed predeclared evaluation interval')
    return frame


def _copy_seal(predictions,receipt,identity,protocol,folder,*,max_bytes,max_rows,clock):
    if type(max_bytes) is not int or max_bytes<1 or type(max_rows) is not int or max_rows<1:
        raise ValueError('Finite seal byte/row budgets required')
    if folder.exists():raise ValueError('Existing prediction seal refused')
    if predictions.stat().st_size+receipt.stat().st_size>max_bytes:raise ValueError('Seal input byte budget exceeded')
    rows=pq.ParquetFile(predictions).metadata.num_rows
    if rows>max_rows:raise ValueError('Seal row budget exceeded before loading')
    scores=pd.read_parquet(predictions);_check_clocks(scores,protocol,clock())
    if digest(predictions)!=identity['predictions_sha256'] or digest(receipt)!=identity['learning_receipt_sha256']:
        raise ValueError('Predictions differ from completed registered producer')
    folder.mkdir(parents=True,exist_ok=False)
    expected={'predictions.parquet':identity['predictions_sha256'],'learning_receipt.json':identity['learning_receipt_sha256']}
    for source,name in [(predictions,'predictions.parquet'),(receipt,'learning_receipt.json')]:
        shutil.copyfile(source,folder/name)
        if digest(source)!=expected[name] or digest(folder/name)!=expected[name]:
            raise ValueError('Source changed from completed identity during prediction seal')
    write_json(folder/'protocol.json',protocol)
    sealed_at=clock().isoformat()
    _check_clocks(scores,protocol,sealed_at)
    files={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in folder.iterdir()}
    record=dict(schema='prospective-prediction-seal-v1',sealed_at=sealed_at,experiment_id=identity['experiment_id'],
        learning_identity_id=identity['identity_id'],rows=rows,files=files,protocol_id=content_id(protocol),
        additional_fits=0,additional_predict_calls=0,outcomes_read=False,
        timestamp_authority='local system clock; not external notarization; no proof against deliberate local rewriting')
    if digest(predictions)!=expected['predictions.parquet'] or digest(receipt)!=expected['learning_receipt.json']:
        raise ValueError('Source changed before prediction seal publication')
    record['seal_id']=content_id(record)
    encoded=json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False).replace('\n',__import__('os').linesep).encode('utf-8')
    if sum(x['bytes'] for x in files.values())+len(encoded)>max_bytes:
        raise ValueError('Seal metadata exceeds byte budget; incomplete copy retained')
    write_json(folder/'seal.json',record)
    return record


def seal_registered_predictions(research,experiment_id,protocol,folder,*,max_bytes,max_rows):
    """Seal the complete original prediction file before its first execution.

    Old predictions are rejected as a whole, never silently trimmed to a subset.
    The public API has no caller-controlled clock or backdate argument.
    """
    folder=Path(folder)
    if not folder.is_absolute():folder=research.project/folder
    folder=_ordinary(folder,research.project/'research')
    if folder.exists():raise ValueError('Existing prediction seal refused')
    identity=verify_completed_learning_identity(research,experiment_id)
    source=research.root/'worker_jobs'/experiment_id/'learn_result'
    return _copy_seal(source/'predictions.parquet',source/'learning_receipt.json',identity,protocol,folder,
        max_bytes=max_bytes,max_rows=max_rows,clock=lambda:datetime.now(timezone.utc))


def verify_prediction_seal(folder,*,max_bytes,max_rows):
    """Offline byte/clock check; does not certify the local wall clock's honesty."""
    folder=Path(folder);_ordinary(folder,folder)
    if {p.name for p in folder.iterdir()}!={'predictions.parquet','learning_receipt.json','protocol.json','seal.json'}:
        raise ValueError('Complete prediction seal inventory required')
    if type(max_bytes) is not int or max_bytes<1 or type(max_rows) is not int or max_rows<1:
        raise ValueError('Finite offline budgets required')
    if sum(_ordinary(p,folder).stat().st_size for p in folder.iterdir())>max_bytes:raise ValueError('Offline byte cap exceeded')
    record=json.loads((folder/'seal.json').read_text(encoding='utf-8'));copy=dict(record);identity=copy.pop('seal_id',None)
    if record.get('schema')!='prospective-prediction-seal-v1' or content_id(copy)!=identity:
        raise ValueError('Prediction seal identity differs')
    if set(record['files'])!={'predictions.parquet','learning_receipt.json','protocol.json'}:
        raise ValueError('Prediction seal file list differs')
    for name,entry in record['files'].items():
        path=_ordinary(folder/name,folder)
        if entry!=dict(sha256=digest(path),bytes=path.stat().st_size):raise ValueError('Sealed artifact differs')
    if pq.ParquetFile(folder/'predictions.parquet').metadata.num_rows>max_rows:raise ValueError('Offline row cap exceeded')
    scores=pd.read_parquet(folder/'predictions.parquet');protocol=json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    if content_id(protocol)!=record['protocol_id'] or len(scores)!=record['rows']:raise ValueError('Sealed protocol/rows differ')
    _check_clocks(scores,protocol,record['sealed_at'])
    return record
