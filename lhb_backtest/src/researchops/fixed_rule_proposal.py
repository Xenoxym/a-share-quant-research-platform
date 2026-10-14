"""Prepare fixed-rule score files and account proposals; no worker or fit."""
import copy,json,os,shutil
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from .matrix_materialization import _output,_budget
from .alpha_experiments import _path,verify_inputs
from .account_experiments import protocol,ROLES
from .account_proposal import PUBLIC_FIELDS,REQUIRED_FIELDS
from ..alpharesearch.fixed_rules import build_fixed_rule_result
from ..technical.artifacts import digest,write_json

FILES={'predictions':'predictions.parquet','learning_receipt':'fixed_rule_receipt.json','rule_inputs':'rule_inputs.parquet'}


def save_fixed_rule_scores(research,input_path,spec,output,*,max_rows,max_input_bytes,max_output_bytes):
    _budget(max_rows=max_rows,max_input_bytes=max_input_bytes,max_output_bytes=max_output_bytes)
    source=_path(research.project,input_path);folder=_output(research.project,output)
    if source.stat().st_size>max_input_bytes or pq.ParquetFile(source).metadata.num_rows>max_rows:
        raise ValueError('Fixed-rule input byte/row budget exceeded before loading')
    expected=digest(source);result=build_fixed_rule_result(pd.read_parquet(source),spec,max_rows=max_rows)
    if digest(source)!=expected:raise ValueError('Fixed-rule input changed during reading')
    folder.mkdir(parents=True,exist_ok=False)
    result.predictions.to_parquet(folder/FILES['predictions'],index=False)
    shutil.copyfile(source,folder/FILES['rule_inputs'])
    if digest(folder/FILES['rule_inputs'])!=expected or digest(source)!=expected:
        raise ValueError('Fixed-rule input changed during copy')
    encoded=json.dumps(result.receipt,ensure_ascii=False,indent=2,allow_nan=False).replace('\n',os.linesep).encode('utf-8')
    if sum(p.stat().st_size for p in folder.iterdir())+len(encoded)>max_output_bytes:
        raise ValueError('Fixed-rule output byte cap exceeded; incomplete files retained')
    write_json(folder/FILES['learning_receipt'],result.receipt)
    return dict(folder=output,rule_id=result.receipt['rule_id'],rows=len(result.predictions),model_fits=0,predict_calls=0,registered_experiments=0)


def account_from_fixed_rules(research,folder,template):
    if (not isinstance(template,dict) or not REQUIRED_FIELDS<=set(template) or set(template)-PUBLIC_FIELDS
            or template['kind']!='alpha_account'):raise ValueError('Public account template required')
    protocol(template['spec'])
    inputs=template['inputs']
    if not isinstance(inputs,list) or any(not isinstance(x,dict) for x in inputs):raise ValueError('Distinct account input roles required')
    roles=[x.get('role') for x in inputs]
    if len(roles)!=len(set(roles)) or set(roles) not in (ROLES,ROLES|{'rule_inputs'}):raise ValueError('Distinct account input roles required')
    path=_output(research.project,folder,allow_existing=True)
    if not path.is_dir() or {p.name for p in path.iterdir()}!=set(FILES.values()):raise ValueError('Complete original fixed-rule score files required')
    sources={role:_path(research.project,(path/name).relative_to(research.project).as_posix()) for role,name in FILES.items()}
    if sum(source.stat().st_size for source in sources.values())>template['spec']['budget']['max_input_bytes']:
        raise ValueError('Fixed-rule source byte cap exceeded before receipt loading')
    receipt=json.loads(sources['learning_receipt'].read_text(encoding='utf-8'))
    if receipt.get('schema')!='fixed-rule-score-receipt-v1' or receipt.get('model_fits')!=0:raise ValueError('Unfitted fixed-rule receipt required')
    value=copy.deepcopy(template);value['spec']['schema']='registered-alpha-account-v4'
    value['spec']['score_source']=dict(kind='fixed_rule',rule_id=receipt['rule_id'])
    value['inputs']=[item for item in value['inputs'] if item['role'] not in FILES]
    for role,name in FILES.items():
        source=sources[role]
        value['inputs'].append(dict(role=role,path=source.relative_to(research.project).as_posix(),sha256=digest(source),bytes=source.stat().st_size))
    value['hypothesis']='Fixed-rule account: '+receipt['rule_identity']['spec']['kind']+'; no model fitting'
    value['expected_observation']='Account outputs under the declared membership, costs and holding policy'
    value['falsification']='Future-known rule value, source mismatch, or different account conditions'
    verify_inputs(research.project,value['inputs'],value['spec']['budget']['max_input_bytes']);protocol(value['spec'])
    return value
