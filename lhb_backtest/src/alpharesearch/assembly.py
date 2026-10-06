"""Registered inputs assembled at explicit decisions; no model or hidden universe filter."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import MissingReason as M, VintagePolicy, timestamp, required_id
from .features.base import FeatureBlock
from .registry import FeatureRegistry, SCHEMAS, SHA, definitions_for_block, implementation_hashes
from ..technical.artifacts import content_id


@dataclass
class FeatureAssembly:
    block: FeatureBlock
    registry: FeatureRegistry
    source_bindings: tuple


def assemble_features(membership, blocks, *, universe_id, block_ids,
                      allow_weak_vintage=False, selected_keys=None):
    """Left join stock/day or broadcast day blocks; require actual materialization IDs.

    Caller verifies immutable artifacts and supplies their content identities.
    IDs and declared provenance are not authenticated by trusting metadata alone.
    Raw report-domain rows cannot be broadcast; use a causal event projection.
    """
    required_id(universe_id,'universe_id');blocks=tuple(blocks)
    if not blocks:raise ValueError('At least one feature block required')
    keys=['sample_id','trade_date','stock_code']
    if not set(keys+['decision_at'])<=set(membership):raise ValueError('Decision membership keys/clocks required')
    p=membership[keys+['decision_at']].copy().reset_index(drop=True)
    if p[keys].isna().any().any() or p.sample_id.duplicated().any() or p.duplicated(['trade_date','stock_code']).any():raise ValueError('Unique non-null decision keys required')
    if not p.stock_code.astype(str).str.fullmatch(r'\d{6}\.(SH|SZ|BJ)').all():raise ValueError('Exchange-qualified stocks required')
    if not p.trade_date.astype(str).str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():raise ValueError('Canonical decision dates required')
    p['decision_at']=pd.to_datetime(p.decision_at.map(timestamp),utc=True).astype('datetime64[ns, UTC]')
    if not p.decision_at.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d').eq(p.trade_date).all():raise ValueError('Decision local date differs from membership')
    values={k:p[k] for k in keys};values.update(observed_end=p.decision_at,known_at=p.decision_at)
    reasons={k:p[k] for k in keys};units={};registry=FeatureRegistry();bindings=[];seen=set();used=set()
    selected=None if selected_keys is None else tuple(selected_keys)
    if selected is not None and (not selected or len(set(selected))!=len(selected)):raise ValueError('Unique nonempty selected feature keys required')
    strong={VintagePolicy.HISTORICAL.value,VintagePolicy.MARKET.value};weak_used=False
    for block in blocks:
        definitions=definitions_for_block(block)
        namespace,domain,_=SCHEMAS[block.metadata['schema']]
        if namespace in seen:raise ValueError('Choose one materialized version per feature namespace')
        seen.add(namespace)
        if namespace not in block_ids or not isinstance(block_ids[namespace],str) or not SHA.fullmatch(block_ids[namespace]):raise ValueError('Every block needs a verified materialization content ID')
        chosen=[x for x in definitions if selected is None or x.key in selected]
        if not chosen:continue
        if domain=='report':raise ValueError('Report-domain data need a causal stock/day projection')
        join_keys=list(block.metadata['key_columns'])
        allowed=(['trade_date'] if domain=='market_day' else ['trade_date','stock_code'])
        if set(join_keys) not in (set(allowed),set(keys)) or (domain=='market_day' and join_keys!=['trade_date']):raise ValueError('Unexpected feature key domain')
        if 'sample_id' in join_keys:
            possible=p[['sample_id','trade_date','stock_code']].merge(block.values[['sample_id','trade_date','stock_code']],on=['trade_date','stock_code'],suffixes=('_decision','_feature'))
            if not possible.sample_id_decision.eq(possible.sample_id_feature).all():raise ValueError('Feature decision identity differs from supplied stock/date sample')
        names=[x.key.split('.',1)[1] for x in chosen]
        right=block.values[join_keys+names+['observed_end','known_at']]
        joined=p.merge(right,on=join_keys,how='left',validate='many_to_one',indicator=True,sort=False)
        why=p[join_keys].merge(block.missing[join_keys+names],on=join_keys,how='left',validate='many_to_one',sort=False)
        matched=joined._merge.eq('both')
        known=pd.to_datetime(joined.known_at,utc=True).astype('datetime64[ns, UTC]')
        observed=pd.to_datetime(joined.observed_end,utc=True).astype('datetime64[ns, UTC]')
        if (matched & (~known.le(p.decision_at)|~observed.le(p.decision_at))).any():raise ValueError('Feature was not available by the decision cutoff')
        finite=np.isfinite(joined[names].to_numpy(dtype=float)).any(axis=1)
        weak=block.metadata['vintage'] not in strong and bool(finite.any())
        if weak and not allow_weak_vintage:raise ValueError('Weak historical feature vintage requires explicit opt-in')
        weak_used|=weak
        for definition,name in zip(chosen,names):
            registry.add(definition);used.add(definition.key)
            values[definition.key]=joined[name].astype(float)
            reasons[definition.key]=why[name].where(matched,M.UNCOVERED.value)
            units[definition.key]=definition.unit
        bindings.append({'namespace':namespace,'materialization_id':block_ids[namespace],
            'source_id':block.metadata['source_id'],'source_version':block.metadata['version_id'],
            'vintage':block.metadata['vintage'],'matched_rows':int(matched.sum()),
            'rows_with_selected_finite_values':int(finite.sum()),'weak_vintage_used':weak,
            'definition_ids':[x.definition_id for x in chosen],
            'definition_binding':'current registered declaration; original calculation not certified'})
    if selected is not None and used!=set(selected):raise ValueError('Selected features were not supplied by registered blocks')
    if set(block_ids)!=seen:raise ValueError('Materialization identities do not match supplied block namespaces')
    membership_id=content_id([list(row[:3])+[row[3].isoformat()] for row in p[keys+['decision_at']].itertuples(index=False,name=None)])
    code=implementation_hashes(['src/alpharesearch/assembly.py','src/alpharesearch/registry.py','src/alpharesearch/contracts.py','src/alpharesearch/features/base.py'])
    identity={'registry_version':registry.version_id,'source_bindings':bindings,'universe_id':universe_id,'membership_id':membership_id,
              'allow_weak_vintage':allow_weak_vintage,'assembly_implementation_hashes':code}
    block=FeatureBlock(pd.DataFrame(values),pd.DataFrame(reasons),units,{
        'schema':'assembled-feature-matrix-v1','key_columns':keys,'source_id':'registered_feature_inputs',
        'version_id':content_id(identity),'vintage':VintagePolicy.LATEST_ONLY.value if weak_used else VintagePolicy.MARKET.value,
        'universe_id':universe_id,'membership_id':membership_id,'registry_version':registry.version_id,
        'source_bindings':bindings,'weak_vintage_allowed':allow_weak_vintage,'clock':'decision-time transformation; source clocks checked before assembly',
        'implementation_hashes':code,'input_artifact_verification':'performed by caller, not granted by metadata',
        'missing':'source reasons retained; absent joined row source_uncovered; no blanket numeric zero fill',
        'membership_rows':len(p),'historical_execution_certified':False,'independent_reproduction_claimed':False,
    }).validate()
    return FeatureAssembly(block,registry,tuple(bindings))
