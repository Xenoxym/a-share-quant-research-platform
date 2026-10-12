"""Train-only expression output handoff; no formula, label or model execution.

The registered wrapper verifies frozen output identities, not vendor history or
formula mathematics. The pure constructor accepts declared blocks for testing.
"""
import json
from pathlib import Path
import pandas as pd
from .assembly import FeatureAssembly
from .contracts import VintagePolicy,timestamps,same_timestamp_instants
from .features.base import FeatureBlock
from .registry import FeatureRegistry,FeatureDefinition,implementation_hashes,SHA
from .prepared_assembly import verify_pointwise_expression
from ..technical.artifacts import content_id,digest,verify_artifacts
KEYS=['sample_id','trade_date','stock_code']
CODE=['src/alpharesearch/expression_assembly.py','src/alpharesearch/registry.py','src/alpharesearch/features/base.py','src/alpharesearch/contracts.py']
def _require(ok,message):
    if not ok:raise ValueError(message)
def _hash_map(value):
    _require(isinstance(value,dict) and value and all(isinstance(k,str) and k and isinstance(v,str) and SHA.fullmatch(v) for k,v in value.items()),'Expression artifact hashes required')
    return dict(value)
def assemble_declared_expression_blocks(membership,items,registry,*,universe_id,provenance,max_rows,max_matrix_cells):
    """Assemble supplied (candidate,block,artifact_hashes) items; no authenticity granted."""
    _require(isinstance(registry,FeatureRegistry),'Original expression registry required')
    _require(type(max_rows) is int and type(max_matrix_cells) is int and max_rows>0 and max_matrix_cells>0,'Positive expression assembly budgets required')
    _require(isinstance(membership,pd.DataFrame) and set(KEYS+['decision_at'])<=set(membership) and 0<len(membership)<=max_rows,'Bounded expression membership required')
    p=membership[KEYS+['decision_at']].copy().reset_index(drop=True)
    _require(not p.isna().any().any() and not p.sample_id.duplicated().any() and not p.duplicated(['trade_date','stock_code']).any(),'Expression membership unique complete keys required')
    p['decision_at']=timestamps(p.decision_at)
    _require(p.decision_at.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d').eq(p.trade_date).all(),'Expression decision local date differs')
    keys=p[KEYS];value_columns={};reason_columns={};units={};bindings=[];definitions=[];weak=False
    implementation=implementation_hashes(CODE)
    for candidate,block,artifacts in items:
        _require(isinstance(candidate,dict) and {'candidate_id','expression','hypothesis'}<=set(candidate),'Declared expression candidate required')
        cid=candidate['candidate_id'];key='expr.'+cid
        _require(key not in units and len(p)*(len(units)+1)<=max_matrix_cells,'Duplicate expression or assembly cell budget exceeded')
        verified=verify_pointwise_expression(candidate['expression'],registry)
        _require(isinstance(block,FeatureBlock),'Typed expression block required');block.validate();m=block.metadata
        _require(block.units=={'score':'ratio'} and m.get('schema')=='alpha-expression-values-v1' and m.get('key_columns')==KEYS and m.get('universe_id')==universe_id and m.get('expression_id')==verified.expression_id,'Expression block definition/unit/universe differs')
        _require(block.values[KEYS].reset_index(drop=True).equals(keys) and block.missing[KEYS].reset_index(drop=True).equals(keys),'Expression output must retain complete membership and row order')
        for clock in ('known_at','observed_end'):
            _require(same_timestamp_instants(timestamps(block.values[clock]).reset_index(drop=True),p.decision_at),'Expression output clock differs from conservative decision clock')
        hashes=_hash_map(artifacts)
        deps=m.get('dependencies');_require(isinstance(deps,list) and all(isinstance(x,dict) and {'key','definition_id','materialization_id','source_id','version_id','vintage'}<=set(x) for x in deps) and sorted(x['definition_id'] for x in deps)==sorted(verified.dependencies) and all(registry.resolve(x['key'],x['definition_id']).definition_id==x['definition_id'] for x in deps),'Expression source dependency identities differ')
        vintage=VintagePolicy(m['vintage']);weak|=vintage in {VintagePolicy.LATEST_ONLY,VintagePolicy.UNKNOWN}
        definition=FeatureDefinition(key=key,unit='ratio',domain='stock_day',description=candidate['hypothesis'],dependencies=tuple(verified.dependencies),parameters_json=json.dumps(dict(schema='registered-expression-feature-definition-v1',expression=candidate['expression'],source_registry_version=registry.version_id),sort_keys=True),implementation_hashes=tuple(implementation),timing_rule='Original decision-date expression output clocks retained; no added history or publication authentication',missing_rule='Original score missing reason retained; no blanket zero fill',limitations=('Source vintage and original calculation limits remain','Registered output identity does not reproduce formula mathematics or prove returns'))
        definitions.append(definition);units[key]='ratio';value_columns[key]=block.values.score.reset_index(drop=True).copy();reason_columns[key]=block.missing.score.reset_index(drop=True).copy()
        bindings.append(dict(namespace='expr',materialization_id=content_id(hashes),source_id=m['source_id'],source_version=m['version_id'],vintage=m['vintage'],matched_rows=len(p),definition_ids=[definition.definition_id],expression_id=verified.expression_id,original_dependencies=deps,output_artifacts=hashes,definition_binding='Expression algorithm declaration plus supplied output identities; original source limitations retained'))
    _require(bool(units),'At least one complete expression output required')
    new_registry=FeatureRegistry(definitions);membership_id=content_id(list(map(list,p[KEYS].itertuples(index=False,name=None))))
    identity=dict(registry_version=new_registry.version_id,source_bindings=bindings,source_provenance=provenance,membership_id=membership_id,universe_id=universe_id)
    values=pd.concat([keys,pd.DataFrame(value_columns)],axis=1);missing=pd.concat([keys,pd.DataFrame(reason_columns)],axis=1)
    values['observed_end']=p.decision_at;values['known_at']=p.decision_at
    meta=dict(schema='assembled-feature-matrix-v1',key_columns=KEYS,source_id='registered_expression_output_handoff',version_id=content_id(identity),vintage=VintagePolicy.LATEST_ONLY.value if weak else VintagePolicy.MARKET.value,universe_id=universe_id,membership_id=membership_id,registry_version=new_registry.version_id,source_bindings=bindings,source_provenance=provenance,clock='Conservative original expression decision-time availability',missing='Original score reasons retained',historical_execution_certified=False,independent_reproduction_claimed=False,registered_batch_verified_by_wrapper=False)
    return FeatureAssembly(FeatureBlock(values,missing,units,meta).validate(),new_registry,tuple(bindings))
def assemble_registered_expression_batch(research,experiment,membership,*,max_rows,max_matrix_cells):
    """Verify and reuse registered partition outputs, without formula evaluation."""
    from ..researchops.service import Research
    _require(isinstance(research,Research),'Actual Research DB interface required; pure fixtures cannot authenticate registration')
    from ..researchops.alpha_execution import verify_completed
    from ..researchops.alpha_experiments import verify_inputs
    from .feature_storage import read_shared_index,read_compact_candidate
    actual=research.store.experiment(experiment['id'])
    _require(actual==experiment and actual['status']=='completed' and actual['proposal']['kind']=='alpha_batch' and isinstance(actual.get('result'),dict) and actual['result'].get('verified') is True,'Actual completed registered batch identity required')
    proposal=actual['proposal'];spec=proposal['spec']
    _require(spec['schema'] in ('alpha-feature-batch-v4','alpha-feature-batch-v6') and spec['storage_layout']=='shared_index_v1','Prepared development/forecast compact batch required')
    audit=verify_completed(research,actual);_require(audit['verified'] is True and audit['failed_candidates']==0 and audit['completed_candidates']==len(spec['candidates']) and audit['fits']==audit['accounts']==0,'Complete registered expression batch audit required')
    folder=research.root/'worker_jobs'/actual['id']/'alpha_result'
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'));_require(digest(folder/'manifest.json')==audit['manifest_hash'],'Audited registered manifest identity differs');verify_artifacts(folder,manifest)
    roles=verify_inputs(research.project,proposal['inputs'],spec['budget']['max_input_bytes']);registry=FeatureRegistry.from_dict(json.loads(roles['registry'].read_text(encoding='utf8')))
    attempts=json.loads((folder/'attempts.json').read_text(encoding='utf8'));completed={r['candidate_id'] for r in attempts if r['status']=='completed'}
    original=pd.read_parquet(roles['membership'])
    _require(isinstance(membership,pd.DataFrame) and membership[KEYS].reset_index(drop=True).equals(original[KEYS].reset_index(drop=True)) and same_timestamp_instants(timestamps(membership.decision_at).reset_index(drop=True),timestamps(original.decision_at).reset_index(drop=True)),'Caller membership differs from frozen full registered source; no label eligibility gate')
    from .batch_partition import execution_partition, is_forecast_batch
    part=execution_partition(spec)
    scoped=original.loc[original.trade_date.between(part['start'],part['end'])].reset_index(drop=True)
    _require(audit['rows_per_completed_candidate']==len(scoped) and audit['holdout_rows_executed']==(len(scoped) if is_forecast_batch(spec) else 0),'Audited partition membership count differs')
    _require(len(completed)==len(spec['candidates']) and completed=={c['candidate_id'] for c in spec['candidates']},'All declared candidates must complete; no implicit failed-output selection')
    limit=proposal['resources']['max_output_bytes'];index=read_shared_index(folder,max_bytes=limit)
    _require(index.keys.equals(scoped[KEYS]),'Registered batch shared keys must equal all declared partition decisions')
    def items():
        for candidate in spec['candidates']:
            cid=candidate['candidate_id']
            _require(cid in completed,'Missing declared completed candidate')
            block=read_compact_candidate(index,cid,max_bytes=limit)
            hashes={name:digest(folder/cid/name) for name in ('values.parquet','missing.parquet','definition.json','storage.json')}
            yield candidate,block,hashes
    provenance=dict(schema='registered-expression-batch-source-v1',experiment_id=actual['id'],task_id=actual['task_id'],batch_fingerprint=actual['fingerprint'],forecast_producer=spec.get('forecast_source'),batch_engine_hash=proposal['engine_hash'],output_manifest_sha256=digest(folder/'manifest.json'),batch_audit_content_id=content_id(audit),source_registry_version=registry.version_id,all_declared_candidates=len(spec['candidates']),complete_candidates=len(completed),excluded_failed_candidates=[c['candidate_id'] for c in spec['candidates'] if c['candidate_id'] not in completed],selection_rule='All completed outputs, no labels or return selection in handoff',execution_partition=part['name'],formula_math_reexecution=False,vendor_history_certified=False)
    result=assemble_declared_expression_blocks(scoped,items(),registry,universe_id=spec['universe_id'],provenance=provenance,max_rows=max_rows,max_matrix_cells=max_matrix_cells)
    result.block.metadata['registered_batch_verified_by_wrapper']=True
    _require(research.store.experiment(actual['id'])==actual,'Completed registered DB identity changed during assembly')
    _require(digest(folder/'manifest.json')==provenance['output_manifest_sha256'],'Registered batch manifest changed during assembly');verify_artifacts(folder,manifest)
    return result


def merge_prepared_expression_assemblies(base, development, forecast, membership, source_receipt,
    prepared_receipt, *, artifact_hashes, expected_expression_keys, max_rows, max_matrix_cells):
    """Exact-key merge adapted from the existing R33 constructor; no formula evaluation.

    Original base columns/bindings survive byte-for-value; expression definition IDs
    must agree across train/forecast while each materialization binding is retained.
    """
    from .prepared_assembly import prepared_assembly_receipt
    _require(all(isinstance(x,FeatureAssembly) for x in (base,development,forecast)), 'Typed prepared and expression assemblies required')
    expected=prepared_assembly_receipt(base.block,base.registry,membership,source_receipt,artifact_hashes=artifact_hashes)
    _require(_same_json(prepared_receipt,expected),'Exact original prepared receipt required for merge')
    p=membership[KEYS+['decision_at']].copy().reset_index(drop=True)
    _require(type(max_rows) is int and 0<len(p)<=max_rows and type(max_matrix_cells) is int and max_matrix_cells>0,'Finite merge budgets required')
    p['decision_at']=timestamps(p.decision_at)
    units=dict(development.block.units)
    _require(isinstance(expected_expression_keys,(list,tuple)) and 1<=len(expected_expression_keys)<=10000 and list(units)==list(expected_expression_keys) and len(set(expected_expression_keys))==len(expected_expression_keys) and units==forecast.block.units and not (set(units)&set(base.block.units)), 'All declared common expression keys required')
    _require(len(p)*(len(base.block.units)+len(units))<=max_matrix_cells,'Merged matrix cell budget exceeded')
    _require(development.registry.to_dict()==forecast.registry.to_dict(),'Training and forecast expression definition identities differ')
    dev_source=development.block.metadata.get('source_provenance',{})
    forecast_source=forecast.block.metadata.get('source_provenance',{})
    producer=forecast_source.get('forecast_producer')
    _require(isinstance(producer,dict) and dev_source.get('schema')=='registered-expression-batch-source-v1'
        and producer.get('producer_experiment_id')==dev_source.get('experiment_id')
        and producer.get('producer_task_id')==dev_source.get('task_id')
        and producer.get('producer_fingerprint')==dev_source.get('batch_fingerprint'),
        'Forecast must reuse this exact development producer')
    _require(base.registry.version_id==dev_source.get('source_registry_version'),
        'Original prepared registry differs from development producer')
    def dependencies(assembly):
        result={}
        for binding in assembly.source_bindings:
            _require(isinstance(binding.get('original_dependencies'),list), 'Original expression dependencies required')
            for did in binding['definition_ids']:
                _require(did not in result, 'Duplicate expression dependency binding')
                result[did]=(binding.get('expression_id'),binding['original_dependencies'])
        return result
    dev_dependencies=dependencies(development)
    _require(_same_json(dev_dependencies,dependencies(forecast)),
        'Training and forecast original source materializations differ')
    for _, deps in dev_dependencies.values():
        _require(all(d.get('materialization_id')==prepared_receipt['adapter_id']
            and d.get('source_id')==base.block.metadata['source_id']
            and d.get('version_id')==base.block.metadata['version_id'] for d in deps),
            'Expression source differs from original prepared base')

    for x in (development,forecast):
        x.block.validate()
        _require(x.block.metadata.get('universe_id')==base.block.metadata['universe_id'] and x.block.metadata.get('registry_version')==x.registry.version_id
            and x.block.metadata.get('source_bindings')==list(x.source_bindings),'Expression merge universe/registry/bindings differ')
        x.registry.resolve_bound_definitions(x.source_bindings,units)
    expr_values=pd.concat([development.block.values,forecast.block.values],ignore_index=True)
    expr_missing=pd.concat([development.block.missing,forecast.block.missing],ignore_index=True)
    _require(not expr_values.sample_id.duplicated().any() and not expr_values.duplicated(['trade_date','stock_code']).any()
        and len(expr_values)==len(p),'Expression partitions overlap, omit or add membership')
    joined=p.merge(expr_values,on=KEYS,how='left',validate='one_to_one',indicator=True,sort=False)
    _require(joined.pop('_merge').eq('both').all(),'Expression partition union must retain full original membership')
    for field in ('known_at','observed_end'):
        _require(same_timestamp_instants(timestamps(joined[field]).reset_index(drop=True),p.decision_at),'Merged expression clock differs')
    why=p[KEYS].merge(expr_missing,on=KEYS,how='left',validate='one_to_one',sort=False)
    values=base.block.values.copy();missing=base.block.missing.copy()
    for key in units:
        values[key]=joined[key].to_numpy(copy=True);missing[key]=why[key].to_numpy(copy=True)
    registry=FeatureRegistry(list(base.registry.definitions)+list(development.registry.definitions))
    bindings=tuple(base.source_bindings)+tuple(development.source_bindings)+tuple(forecast.source_bindings)
    registry.resolve_bound_definitions(bindings,list(base.block.units)+list(units))
    identity=dict(schema='prepared-expression-merge-v1',prepared_adapter_id=prepared_receipt['adapter_id'],
        registry_version=registry.version_id,source_bindings=list(bindings),
        development_version=development.block.metadata['version_id'],forecast_version=forecast.block.metadata['version_id'])
    weak=any(x.block.metadata.get('vintage') in {VintagePolicy.LATEST_ONLY.value,VintagePolicy.UNKNOWN.value} for x in (base,development,forecast))
    meta=dict(base.block.metadata,source_id='prepared_expression_partition_merge',version_id=content_id(identity),
        registry_version=registry.version_id,source_bindings=list(bindings),
        vintage=VintagePolicy.LATEST_ONLY.value if weak else VintagePolicy.MARKET.value,
        source_provenance=dict(schema='prepared-expression-merge-source-v1',original_prepared=prepared_receipt,
            original_base_metadata=base.block.metadata,development=development.block.metadata,forecast=forecast.block.metadata),
        registered_batch_verified_by_wrapper=False,independent_reproduction_claimed=False,historical_execution_certified=False)
    return FeatureAssembly(FeatureBlock(values,missing,dict(base.block.units,**units),meta).validate(),registry,bindings)


def _same_json(left, right):
    return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(right, sort_keys=True, separators=(",", ":"), allow_nan=False)
