"""Non-executable, version-bound Alpha AST with dimensions and causal window limits.

Compilation validates a proposed expression. It does not evaluate data, establish
historical source truth, or demonstrate a profitable strategy.
"""
from dataclasses import dataclass
import json
import math

from .contracts import Unit
from .registry import FeatureRegistry, SHA
from ..technical.artifacts import content_id

DIMENSIONS={Unit.PRICE.value:(1,-1,0),Unit.AMOUNT.value:(1,0,0),
            Unit.SHARES.value:(0,1,0),Unit.SECONDS.value:(0,0,1),
            **{u.value:(0,0,0) for u in (Unit.RATIO,Unit.LOG_RETURN,Unit.COUNT,Unit.BINARY)}}
UNARY={'neg','abs','log','exp','cs_rank','cs_zscore','broadcast','lag','delta',
       'ts_mean','ts_sum','ts_std','ts_min','ts_max','ts_rank','signed_power'}
BINARY={'add','sub','mul','div','minimum','maximum','gt','lt','ts_corr','ts_cov'}
OPS=UNARY|BINARY|{'ref','const','where'}
ROLLING={'ts_mean','ts_sum','ts_std','ts_min','ts_max','ts_rank','ts_corr','ts_cov'}


def _json(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple=()
    parameters_json: str='{}'

    def __post_init__(self):
        if self.op not in OPS:raise ValueError('Unknown expression operator')
        if not isinstance(self.args,(tuple,list)) or any(not isinstance(x,Expr) for x in self.args):raise ValueError('Expression children must be typed AST nodes')
        object.__setattr__(self,'args',tuple(self.args))
        params=json.loads(self.parameters_json)
        if not isinstance(params,dict):raise ValueError('Expression parameters must be a JSON object')
        object.__setattr__(self,'parameters_json',_json(params))

    @classmethod
    def ref(cls,key,definition_id=None):
        return cls('ref',parameters_json=_json({'key':key,'definition_id':definition_id}))

    @classmethod
    def const(cls,value,unit=Unit.RATIO):
        return cls('const',parameters_json=_json({'value':value,'unit':Unit(unit).value}))

    @classmethod
    def call(cls,op,*args,**parameters):return cls(op,tuple(args),_json(parameters))

    def to_dict(self):return {'op':self.op,'args':[x.to_dict() for x in self.args],'parameters':json.loads(self.parameters_json)}

    @classmethod
    def from_dict(cls,value,*,max_nodes=127,max_depth=12):
        """Bounded JSON loader; never parse/eval a Python source expression."""
        count=0
        def load(node,depth):
            nonlocal count
            count+=1
            if count>max_nodes or depth>max_depth:raise ValueError('Expression JSON complexity budget exceeded')
            if not isinstance(node,dict) or set(node)!={'op','args','parameters'} or not isinstance(node['args'],list):raise ValueError('Strict structured AST dictionary required')
            return cls(node['op'],tuple(load(x,depth+1) for x in node['args']),_json(node['parameters']))
        return load(value,1)


@dataclass(frozen=True)
class ExpressionLimits:
    max_nodes: int=127
    max_depth: int=12
    max_additional_history_sessions: int=252
    max_constant_magnitude: float=1e12

    def __post_init__(self):
        if any(type(x) is not int or x<1 for x in (self.max_nodes,self.max_depth,self.max_additional_history_sessions)):raise ValueError('Expression budgets must be positive integers')
        if isinstance(self.max_constant_magnitude,bool) or not isinstance(self.max_constant_magnitude,(int,float)) or not math.isfinite(self.max_constant_magnitude) or self.max_constant_magnitude<=0:raise ValueError('Finite positive literal bound required')


@dataclass(frozen=True)
class CompiledExpression:
    ast_json: str
    dimension: tuple
    domain: str
    additional_history_sessions: int
    nodes: int
    depth: int
    dependencies: tuple
    registry_version: str
    limits_json: str
    purpose: str

    @property
    def expression_id(self):return content_id(self.to_dict(include_id=False))

    def to_dict(self,include_id=True):
        value={'schema':'typed-alpha-ast-v1','ast':json.loads(self.ast_json),'dimension':self.dimension,
            'domain':self.domain,'additional_history_sessions':self.additional_history_sessions,
            'nodes':self.nodes,'depth':self.depth,'dependencies':self.dependencies,
            'registry_version':self.registry_version,'limits':json.loads(self.limits_json),'purpose':self.purpose}
        if include_id:value['expression_id']=self.expression_id
        return value


def compile_expression(expr: Expr, registry: FeatureRegistry, *, limits=None, require_alpha=True):
    """Bind source versions, dimensions, domains and only-backward time operators.

    Lookback is additional to precomputed leaf generation, not total raw history.
    Market-day data require an explicit broadcast before a stock-day score.
    """
    if not isinstance(expr,Expr) or not isinstance(registry,FeatureRegistry):raise ValueError('Typed AST and registry required')
    if type(require_alpha) is not bool:raise ValueError('Explicit alpha-purpose Boolean required')
    limits=limits or ExpressionLimits()
    if not isinstance(limits,ExpressionLimits):raise ValueError('Typed expression limits required')
    nodes=0;depth_used=0;dependencies=set()

    def visit(node,depth):
        nonlocal nodes,depth_used
        nodes+=1;depth_used=max(depth_used,depth)
        if nodes>limits.max_nodes or depth>limits.max_depth:raise ValueError('Expression complexity budget exceeded')
        params=json.loads(node.parameters_json);op=node.op
        arity=0 if op in {'ref','const'} else (1 if op in UNARY else (2 if op in BINARY else 3))
        if len(node.args)!=arity:raise ValueError('Incorrect operator arity: '+op)
        allowed={'key','definition_id'} if op=='ref' else ({'value','unit'} if op=='const' else ({'periods'} if op in {'lag','delta'} else ({'window'} if op in ROLLING else ({'power'} if op=='signed_power' else set()))))
        if set(params)!=allowed:raise ValueError('Missing or extra operator parameters: '+op)
        if op=='ref':
            key=params['key'];definition_id=params['definition_id']
            if not isinstance(key,str) or (definition_id is not None and (not isinstance(definition_id,str) or not SHA.fullmatch(definition_id))):raise ValueError('Explicit registered feature reference required')
            definition=registry.resolve(key,definition_id);dependencies.add(definition.definition_id)
            bound={'op':'ref','args':[],'parameters':{'key':key,'definition_id':definition.definition_id}}
            return bound,DIMENSIONS[definition.unit],definition.domain,0
        if op=='const':
            value=params['value'];unit=Unit(params['unit']).value
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or abs(value)>limits.max_constant_magnitude:raise ValueError('Finite bounded numeric literal required')
            if unit==Unit.COUNT.value and (value<0 or int(value)!=value):raise ValueError('Count literal must be a nonnegative integer')
            if unit==Unit.BINARY.value and value not in (0,1):raise ValueError('Binary literal must be zero or one')
            return node.to_dict(),DIMENSIONS[unit],'scalar',0
        children=[visit(child,depth+1) for child in node.args]
        dims=[x[1] for x in children];domains={x[2] for x in children if x[2]!='scalar'}
        if len(domains)>1:raise ValueError('Different data domains need an explicit supported conversion')
        domain=next(iter(domains),'scalar');history=max(x[3] for x in children);dimension=dims[0]
        if op=='broadcast':
            if domain!='market_day':raise ValueError('Only market-day inputs can broadcast to stock-day')
            domain='stock_day'
        elif op in {'cs_rank','cs_zscore'}:
            if domain!='stock_day':raise ValueError('Cross-sectional normalization needs stock-day data')
            dimension=(0,0,0)
        elif op in {'lag','delta'}:
            periods=params['periods']
            if type(periods) is not int or periods<0:raise ValueError('Future references and non-integer lag are forbidden')
            if domain not in {'stock_day','market_day'}:raise ValueError('Session lag needs daily data, not raw reports or constants')
            history+=periods
        elif op in ROLLING:
            window=params['window']
            if type(window) is not int or window<2:raise ValueError('Full rolling windows require integer window >=2')
            if domain not in {'stock_day','market_day'}:raise ValueError('Rolling session operators need daily data')
            if any(child[2]!=domain for child in children):raise ValueError('Every rolling operand must have the same daily domain; scalar rolling pairs are forbidden')
            history+=window-1
            if op in {'ts_rank','ts_corr'}:dimension=(0,0,0)
            if op=='ts_cov':dimension=tuple(a+b for a,b in zip(dims[0],dims[1]))
        elif op in {'log','exp','signed_power'}:
            if dimension!=(0,0,0):raise ValueError('Nonlinear log/exp/power requires dimensionless data')
            if op=='signed_power':
                if isinstance(params['power'],bool) or params['power'] not in (.5,1.,2.,3.):raise ValueError('Signed power requires a whitelisted exponent')
        elif op in {'add','sub','minimum','maximum','gt','lt'}:
            if dims[0]!=dims[1]:raise ValueError('Addition/comparison cannot mix physical dimensions')
            if op in {'gt','lt'}:dimension=(0,0,0)
        elif op in {'mul','div'}:
            direction=1 if op=='mul' else -1
            dimension=tuple(a+direction*b for a,b in zip(dims[0],dims[1]))
        elif op=='where':
            if dims[0]!=(0,0,0) or dims[1]!=dims[2]:raise ValueError('Conditional needs dimensionless condition and matching branch dimensions')
            dimension=dims[1]
        # neg/abs preserve dimensions and domain; no executable text is accepted.
        if history>limits.max_additional_history_sessions:raise ValueError('Additional historical-session budget exceeded')
        ast={'op':op,'args':[x[0] for x in children],'parameters':params}
        return ast,dimension,domain,history

    ast,dimension,domain,history=visit(expr,1)
    if require_alpha and (dimension!=(0,0,0) or domain!='stock_day'):raise ValueError('Alpha root must be a dimensionless stock-day score')
    limits_json=_json({'max_nodes':limits.max_nodes,'max_depth':limits.max_depth,'max_additional_history_sessions':limits.max_additional_history_sessions,'max_constant_magnitude':limits.max_constant_magnitude})
    return CompiledExpression(_json(ast),dimension,domain,history,nodes,depth_used,tuple(sorted(dependencies)),registry.version_id,limits_json,'alpha_score' if require_alpha else 'intermediate_feature')


def verify_compiled(value, registry):
    """Recompile a saved proposal; detect changed registry, AST, types or content ID."""
    if not isinstance(value,dict) or value.get('schema')!='typed-alpha-ast-v1':raise ValueError('Unknown compiled expression schema')
    limits=ExpressionLimits(**value['limits'])
    ast=Expr.from_dict(value['ast'],max_nodes=limits.max_nodes,max_depth=limits.max_depth)
    if value['purpose'] not in {'alpha_score','intermediate_feature'}:raise ValueError('Unknown expression purpose')
    compiled=compile_expression(ast,registry,limits=limits,require_alpha=value['purpose']=='alpha_score')
    if _json(compiled.to_dict())!=_json(value):raise ValueError('Compiled expression identity or contract changed')
    return compiled
