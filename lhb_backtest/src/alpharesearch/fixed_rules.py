"""Deterministic, unfitted score sources for the shared portfolio/account path."""
from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
from .contracts import required_id,timestamps
from .learning import KEYS,CLOCKS,_keys,_fingerprint
from .registry import implementation_hashes
from ..technical.artifacts import content_id


@dataclass
class FixedRuleResult:
    predictions: pd.DataFrame
    receipt: dict
    rule_inputs: pd.DataFrame


def rule_spec(value):
    if (not isinstance(value,dict) or set(value)!={'schema','kind','seed','universe_id','value_definition'}
            or value['schema']!='fixed-rule-v1'
            or value['kind'] not in {'negative_reported_market_cap_proxy','stable_stock_hash'}):
        raise ValueError('Strict fixed-rule definition required')
    required_id(value['universe_id'],'Fixed-rule universe')
    if value['kind']=='stable_stock_hash':
        if type(value['seed']) is not int or not 0<=value['seed']<2**32 or value['value_definition'] is not None:
            raise ValueError('Fixed stock hash requires a bounded integer seed and no value definition')
    elif value['seed'] is not None or value['value_definition']!='financial.reported_market_cap_proxy':
        raise ValueError('Market-cap proxy rule requires its actual value definition and no random seed')
    return dict(value)


def build_fixed_rule_result(inputs,spec,*,max_rows):
    """No labels, fit, tuning or filtering. Invalid/missing values reject the source.

    value_known_at is the caller's declared availability, not vendor PIT proof.
    A market-cap proxy is not certified total/free-float current capitalization.
    """
    cfg=rule_spec(spec)
    if type(max_rows) is not int or max_rows<1 or not isinstance(inputs,pd.DataFrame) or inputs.empty or len(inputs)>max_rows:
        raise ValueError('Finite nonempty fixed-rule input row budget required')
    _keys(inputs,'Fixed rule inputs')
    required=KEYS+CLOCKS+(['value','value_known_at'] if cfg['kind']=='negative_reported_market_cap_proxy' else [])
    if set(inputs)!=set(required):raise ValueError('Strict fixed-rule input columns required')
    clocks={k:timestamps(inputs[k]) for k in CLOCKS}
    if (clocks['execution_at'].le(clocks['decision_at'])|clocks['target_end_at'].le(clocks['execution_at'])).any():
        raise ValueError('Fixed-rule decision/execution/target clocks differ')
    if not clocks['decision_at'].dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d').eq(inputs.trade_date).all():
        raise ValueError('Fixed-rule decision date differs')
    if cfg['kind']=='negative_reported_market_cap_proxy':
        if (not pd.api.types.is_numeric_dtype(inputs.value.dtype) or pd.api.types.is_complex_dtype(inputs.value.dtype)
                or not np.isfinite(inputs.value.to_numpy(dtype=float)).all() or inputs.value.le(0).any()):
            raise ValueError('Finite positive market-cap proxy required for every source row; no silent filtering')
        if timestamps(inputs.value_known_at).gt(clocks['decision_at']).any():raise ValueError('Market-cap proxy was not known at decision')
        score=-inputs.value.to_numpy(dtype=float)
    else:
        score=np.asarray([int(hashlib.sha256(f"{cfg['seed']}:{stock}".encode()).hexdigest()[:13],16)/16**13 for stock in inputs.stock_code],dtype=float)
    code=dict(implementation_hashes(['src/alpharesearch/fixed_rules.py','src/alpharesearch/contracts.py','src/alpharesearch/learning.py','src/technical/artifacts.py']))
    identity=dict(schema='fixed-rule-identity-v1',spec=cfg,implementation_hashes=code)
    rule_id=content_id(identity);membership=_fingerprint(inputs[KEYS+CLOCKS].reset_index(drop=True));source=_fingerprint(inputs)
    prediction_id=content_id(dict(rule_id=rule_id,rule_inputs=source,prediction_membership=membership))
    output=inputs[KEYS+CLOCKS].copy();output['score']=score;output['rule_id']=rule_id;output['prediction_id']=prediction_id
    receipt=dict(schema='fixed-rule-score-receipt-v1',rule_id=rule_id,rule_identity=identity,prediction_id=prediction_id,
        prediction_rows=len(output),prediction_membership=membership,prediction_outputs=_fingerprint(output),rule_inputs=source,
        implementation_hashes=code,universe_id=cfg['universe_id'],model_fits=0,account_results=False,historical_execution_certified=False,
        scope='Deterministic unfitted rule; declared input availability only, not vendor point-in-time authentication')
    return FixedRuleResult(output,receipt,inputs.copy())


def validate_fixed_rule_result(result,*,max_rows):
    if type(result) is not FixedRuleResult or not isinstance(result.receipt,dict):raise ValueError('Typed unfitted rule result required')
    expected=build_fixed_rule_result(result.rule_inputs,result.receipt.get('rule_identity',{}).get('spec'),max_rows=max_rows)
    if content_id(result.receipt)!=content_id(expected.receipt):raise ValueError('Fixed-rule source receipt differs from actual inputs and rule')
    try:pd.testing.assert_frame_equal(result.predictions,expected.predictions,check_exact=True)
    except AssertionError as exc:raise ValueError('Fixed-rule scores differ from actual rule inputs') from exc
    return result.receipt
