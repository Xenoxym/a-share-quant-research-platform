"""Frozen prepared-assembly leaves for decision-date, pointwise expressions only.

This adapter preserves prior definitions and conservative assembly clocks. It
checks supplied file/content bindings, not original vendor truth or raw clocks.
No source feature, label, model or account is recomputed here.
"""
from dataclasses import dataclass
import hashlib
import json

import pandas as pd

from .contracts import timestamps, same_timestamp_instants
from .features.base import FeatureBlock
from .operators import ExpressionEvaluator, LeafBinding
from .registry import FeatureRegistry, SHA
from .dsl import verify_compiled
from ..technical.artifacts import content_id

KEYS = ['sample_id', 'trade_date', 'stock_code']
ARTIFACTS = {'values', 'missing', 'definition', 'registry', 'source_receipt', 'membership'}
FORBIDDEN = {'lag', 'delta', 'ts_mean', 'ts_sum', 'ts_std', 'ts_min', 'ts_max',
             'ts_rank', 'ts_corr', 'ts_cov'}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _frame_id(frame):
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype='uint64')
    return content_id(dict(columns=list(frame), dtypes=[str(x) for x in frame.dtypes],
                           rows=len(frame), content_sha256=hashlib.sha256(values.tobytes()).hexdigest()))


def _source_contract(block, registry, membership, source_receipt, artifacts):
    _require(isinstance(block, FeatureBlock) and isinstance(registry, FeatureRegistry),
             'Typed prepared assembly and frozen registry required')
    block.validate()
    meta = block.metadata
    _require(meta.get('schema') == 'assembled-feature-matrix-v1' and meta.get('key_columns') == KEYS,
             'Prepared source must retain assembled decision keys')
    _require(meta.get('registry_version') == registry.version_id and isinstance(meta.get('source_bindings'), list),
             'Prepared source registry/bindings differ')
    definitions = registry.resolve_bound_definitions(meta['source_bindings'], block.units)
    _require({d.key for d in registry.definitions} == set(block.units)
             and len(registry.definitions) == len(block.units), 'Complete unambiguous prepared registry required')
    _require(all(d.unit == block.units[k] and d.domain in {'stock_day', 'market_day'}
                 for k, d in definitions.items()), 'Prepared leaf unit/domain differs')
    _require(isinstance(artifacts, dict) and set(artifacts) == ARTIFACTS
             and all(isinstance(v, str) and SHA.fullmatch(v) for v in artifacts.values()),
             'Exact prepared source artifact SHA bindings required')
    _require(isinstance(membership, pd.DataFrame) and set(KEYS + ['decision_at']) <= set(membership)
             and not membership.empty, 'Full prepared decision membership required')
    p = membership[KEYS + ['decision_at']].copy().reset_index(drop=True)
    _require(not p[KEYS + ['decision_at']].isna().any().any() and not p.sample_id.duplicated().any()
             and not p.duplicated(['trade_date', 'stock_code']).any(), 'Prepared membership keys differ')
    _require(block.values[KEYS].reset_index(drop=True).equals(p[KEYS]),
             'Prepared assembly must preserve all original membership rows/order')
    p['decision_at'] = timestamps(p.decision_at)
    _require(p.decision_at.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d').eq(p.trade_date).all(),
             'Prepared decision local date differs')
    _require(p.groupby('trade_date').decision_at.nunique().eq(1).all(),
             'One prepared decision clock per date required')
    for field in ('known_at', 'observed_end'):
        actual = timestamps(block.values[field]).reset_index(drop=True)
        _require(same_timestamp_instants(actual, p.decision_at),
                 'Prepared assembly must retain conservative decision-time ' + field)
    expected = dict(schema='screen-assembly-reference-v1',
        artifacts=dict(assembly_values=artifacts['values'], assembly_missing=artifacts['missing'],
                       assembly_definition=artifacts['definition']),
        registry_version=registry.version_id, assembly_version=meta['version_id'],
        source_bindings=meta['source_bindings'], rows=len(block.values), upstream_execution_certified=False)
    expected['reference_id'] = content_id(expected)
    _require(_canonical(source_receipt) == _canonical(expected), 'Original assembly source receipt differs')
    return definitions, p


def prepared_assembly_receipt(block, registry, membership, source_receipt, *, artifact_hashes):
    """Bind verified caller files and current frames; do not authenticate old calculations."""
    definitions, p = _source_contract(block, registry, membership, source_receipt, artifact_hashes)
    value = dict(schema='prepared-assembly-leaf-receipt-v1', artifact_hashes=dict(artifact_hashes),
        registry_version=registry.version_id, source_version=block.metadata['version_id'],
        source_reference_id=source_receipt['reference_id'], universe_id=block.metadata['universe_id'],
        source_bindings_id=content_id(block.metadata['source_bindings']),
        definitions={k: d.definition_id for k, d in sorted(definitions.items())},
        domains={k: d.domain for k, d in sorted(definitions.items())}, units=dict(block.units),
        values_id=_frame_id(block.values), missing_id=_frame_id(block.missing), membership_id=_frame_id(p),
        rows=len(p), vintage=block.metadata['vintage'],
        clock_policy='conservative_assembly_decision_time; original_raw_clocks_not_reconstructed',
        temporal_policy='decision_date_pointwise_only; no_additional_session_history',
        historical_source_certified=False, original_calculation_reproduced=False)
    return dict(value, adapter_id=content_id(value))


@dataclass(frozen=True)
class PreparedAssemblyLeaves:
    leaves: dict
    stock_block: FeatureBlock
    market_block: FeatureBlock | None
    receipt: dict


def adapt_prepared_assembly(block, registry, membership, source_receipt, receipt, *, artifact_hashes):
    """Share one stock block and one checked market block across all supplied leaves."""
    expected = prepared_assembly_receipt(block, registry, membership, source_receipt,
                                         artifact_hashes=artifact_hashes)
    _require(_canonical(receipt) == _canonical(expected), 'Prepared leaf receipt/content/source changed')
    definitions = registry.resolve_bound_definitions(block.metadata['source_bindings'], block.units)
    stock = [k for k, d in definitions.items() if d.domain == 'stock_day']
    market = [k for k, d in definitions.items() if d.domain == 'market_day']
    _require(bool(stock), 'Prepared assembly needs stock-day leaves')
    meta = dict(block.metadata, schema='prepared-decision-stock-leaves-v1',
                source_prepared_adapter_id=receipt['adapter_id'], original_raw_clocks_reconstructed=False)
    stock_block = FeatureBlock(block.values[KEYS + stock + ['observed_end', 'known_at']].copy(),
        block.missing[KEYS + stock].copy(), {k: block.units[k] for k in stock}, meta).validate()
    market_block = None
    if market:
        for day, index in block.values.groupby('trade_date', sort=True).groups.items():
            fields = market + ['observed_end', 'known_at']
            _require(block.values.loc[index, fields].nunique(dropna=False).le(1).all()
                     and block.missing.loc[index, market].nunique(dropna=False).le(1).all(),
                     'Market prepared values/reasons/availability disagree within date: ' + day)
        take = ~block.values.trade_date.duplicated()
        market_meta = dict(block.metadata, schema='prepared-decision-market-leaves-v1', key_columns=['trade_date'],
            source_prepared_adapter_id=receipt['adapter_id'], original_raw_clocks_reconstructed=False,
            projection='exact same-date value/reason/availability consistency; explicit DSL broadcast required')
        market_block = FeatureBlock(block.values.loc[take, ['trade_date'] + market + ['observed_end', 'known_at']].copy(),
            block.missing.loc[take, ['trade_date'] + market].copy(),
            {k: block.units[k] for k in market}, market_meta).validate()
    leaves = {k: LeafBinding(stock_block if d.domain == 'stock_day' else market_block,
                             k, d.definition_id, receipt['adapter_id']) for k, d in definitions.items()}
    return PreparedAssemblyLeaves(leaves, stock_block, market_block, dict(receipt))


def verify_pointwise_expression(compiled, registry):
    verified = verify_compiled(compiled.to_dict() if hasattr(compiled, 'to_dict') else compiled, registry)
    def visit(node):
        _require(node['op'] not in FORBIDDEN, 'Prepared decision leaves forbid all lag/delta/TS operators')
        for child in node['args']:
            visit(child)
    visit(json.loads(verified.ast_json))
    _require(verified.additional_history_sessions == 0, 'Prepared decision leaves have zero additional history')
    return verified


class PreparedAssemblyEvaluator(ExpressionEvaluator):
    """Existing evaluator with an explicit sparse-decision pointwise gate."""
    def evaluate(self, compiled):
        return super().evaluate(verify_pointwise_expression(compiled, self.registry))
