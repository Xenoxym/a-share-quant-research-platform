"""Bounded, as-of numerical Alpha operators on an explicit full session grid.

No fitting or labels. A passed materialization ID binds declared inputs; this
module does not independently authenticate historical vendor files or execution.
"""
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from .contracts import MissingReason as M, VintagePolicy, required_id, timestamp
from .dsl import CompiledExpression, verify_compiled
from .features.base import FeatureBlock
from .registry import FeatureRegistry, SHA, implementation_hashes
from ..technical.artifacts import content_id

REASONS=tuple(m.value for m in M);CODES={name:i for i,name in enumerate(REASONS)}
PRESENT=CODES[M.PRESENT.value]


@dataclass(frozen=True)
class LeafBinding:
    block: FeatureBlock
    column: str
    definition_id: str
    materialization_id: str

    def __post_init__(self):
        if not isinstance(self.block,FeatureBlock) or self.column not in self.block.units:raise ValueError('Declared feature block column required')
        if any(not isinstance(v,str) or not SHA.fullmatch(v) for v in (self.definition_id,self.materialization_id)):raise ValueError('Definition and materialization SHA256 identities required')


@dataclass
class _Array:
    values: np.ndarray
    reasons: np.ndarray
    domain: str


class ExpressionEvaluator:
    """One finite panel; daily cross-sections use only supplied membership.

    Source leaves may include non-member days needed for stock history. They are
    not silently selected as trading candidates. Rolling windows use calendar
    sessions, including missing observations. No event/market/stock-domain join
    is inferred; market-to-stock expansion requires the compiled broadcast node.
    """
    def __init__(self,registry: FeatureRegistry,leaves: dict,*,calendar,membership,
                 decision_clocks,universe_id,allow_weak_vintage=False,
                 max_grid_cells=1_000_000,max_estimated_buffer_bytes=512_000_000):
        if not isinstance(registry,FeatureRegistry) or not isinstance(leaves,dict) or not leaves:raise ValueError('Registry and declared leaf bindings required')
        if type(allow_weak_vintage) is not bool:raise ValueError('Explicit vintage opt-in required')
        if any(type(x) is not int or x<1 for x in (max_grid_cells,max_estimated_buffer_bytes)):raise ValueError('Finite positive resource budgets required')
        required_id(universe_id,'universe_id')
        days=list(calendar)
        if not days or any(not isinstance(d,str) or len(d)!=10 for d in days) or days!=sorted(set(days)):raise ValueError('Unique sorted full trading calendar required')
        pd.to_datetime(days,format='%Y-%m-%d',errors='raise')
        p=membership.copy();fields=['sample_id','trade_date','stock_code','decision_at']
        if not set(fields)<=set(p) or p.empty or p[fields].isna().any().any() or p.sample_id.duplicated().any() or p.duplicated(['trade_date','stock_code']).any():raise ValueError('Unique non-null daily membership and sample IDs required')
        if not p.stock_code.map(lambda x:isinstance(x,str) and bool(x.strip())).all() or not p.trade_date.isin(days).all():raise ValueError('Membership stock IDs and bounded calendar dates required')
        clocks=decision_clocks[['trade_date','decision_at']].copy()
        if clocks.trade_date.duplicated().any() or set(clocks.trade_date)!=set(days):raise ValueError('One explicit decision clock per calendar session required')
        clocks['decision_at']=pd.to_datetime(clocks.decision_at.map(timestamp),utc=True)
        if clocks.decision_at.dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d').ne(clocks.trade_date).any():raise ValueError('Decision clock must belong to its local trading date')
        self.clocks=clocks.set_index('trade_date').reindex(days).decision_at
        p['decision_at']=pd.to_datetime(p.decision_at.map(timestamp),utc=True)
        if not p.decision_at.reset_index(drop=True).equals(p.trade_date.map(self.clocks).reset_index(drop=True)):raise ValueError('Membership and session decision clocks differ')
        stocks=sorted(p.stock_code.unique());cells=len(days)*len(stocks)
        if cells>max_grid_cells:raise ValueError('Calendar grid resource budget exceeded before allocation')
        self.registry=registry;self.leaves=leaves.copy();self.days=days;self.stocks=stocks;self.membership=p[fields].reset_index(drop=True)
        self.shape=(len(days),len(stocks));self.index=pd.MultiIndex.from_product([days,stocks],names=['trade_date','stock_code'])
        self.day_index={d:i for i,d in enumerate(days)};self.stock_index={c:i for i,c in enumerate(stocks)}
        self.row_i=p.trade_date.map(self.day_index).to_numpy();self.col_i=p.stock_code.map(self.stock_index).to_numpy()
        self.member=np.zeros(self.shape,dtype=bool);self.member[self.row_i,self.col_i]=True
        self.universe_id=universe_id;self.allow_weak=allow_weak_vintage;self.max_buffer=max_estimated_buffer_bytes;self._leaf_cache={};self._validated_blocks=set()
        self.context_id=content_id({'schema':'expression-context-v1','calendar':days,'membership':self.membership[['sample_id','trade_date','stock_code']].to_dict('records'),
            'decision_clocks':[(d,t.isoformat()) for d,t in self.clocks.items()],'universe_id':universe_id,'allow_weak_vintage':allow_weak_vintage})

    def _leaf(self,key,definition_id):
        cache_key=(key,definition_id)
        if cache_key in self._leaf_cache:return self._leaf_cache[cache_key]
        if key not in self.leaves or not isinstance(self.leaves[key],LeafBinding):raise ValueError('Missing typed leaf binding: '+key)
        binding=self.leaves[key];definition=self.registry.resolve(key,definition_id)
        if binding.definition_id!=definition_id or binding.block.units[binding.column]!=definition.unit:raise ValueError('Leaf version or unit does not match compiled reference')
        block=binding.block
        if id(block) not in self._validated_blocks:block.validate();self._validated_blocks.add(id(block))
        keys=block.metadata['key_columns']
        if definition.domain=='report':raise ValueError('Report-domain leaves require a prior decision projection')
        if 'trade_date' not in keys or ('stock_code' in keys)!=(definition.domain=='stock_day'):raise ValueError('Leaf keys do not match the registered data domain')
        k=['trade_date','stock_code'] if definition.domain=='stock_day' else ['trade_date']
        if block.values.duplicated(k).any():raise ValueError('Leaf has ambiguous rows per daily domain')
        source=block.values[k+[binding.column,'known_at']].copy();source['reason']=block.missing[binding.column].to_numpy()
        if definition.domain=='stock_day':
            source=source.set_index(k).reindex(self.index);shape=self.shape;limits=np.repeat(self.clocks.to_numpy(),len(self.stocks))
        else:
            source=source.set_index(k).reindex(self.days);shape=(len(self.days),1);limits=self.clocks.to_numpy()
        values=source[binding.column].to_numpy(dtype=float,copy=True);reasons=source.reason.fillna(M.UNCOVERED.value).map(CODES).to_numpy(dtype=np.uint8,copy=True)
        known=pd.to_datetime(source.known_at,utc=True);future=known.notna().to_numpy() & known.gt(pd.to_datetime(limits,utc=True)).to_numpy()
        values[future]=np.nan;reasons[future]=CODES[M.NOT_KNOWN.value]
        vintage=VintagePolicy(block.metadata['vintage'])
        if vintage in (VintagePolicy.LATEST_ONLY,VintagePolicy.UNKNOWN) and np.isfinite(values).any() and not self.allow_weak:raise ValueError('Weak source vintage requires explicit opt-in')
        result=_Array(values.reshape(shape),reasons.reshape(shape),definition.domain);self._leaf_cache[cache_key]=result;return result

    @staticmethod
    def _clean(values,reasons,domain):
        values=np.array(values,dtype=float,copy=True);reasons=np.array(np.broadcast_to(reasons,values.shape),dtype=np.uint8,copy=True)
        bad=~np.isfinite(values);values[bad]=np.nan
        reasons[bad & (reasons==PRESENT)]=CODES[M.UNDEFINED.value];reasons[~bad]=PRESENT
        return _Array(values,reasons,domain)

    def _rolling(self,op,children,window):
        x=children[0];df=pd.DataFrame(x.values);rolling=df.rolling(window,min_periods=window)
        with np.errstate(all='ignore'):
            if op=='ts_rank':values=rolling.rank(method='average',pct=True).to_numpy()
            elif op=='ts_std':values=rolling.std(ddof=1).to_numpy()
            elif op in {'ts_corr','ts_cov'}:
                other=pd.DataFrame(children[1].values)
                values=(rolling.corr(other,pairwise=False) if op=='ts_corr' else rolling.cov(other,pairwise=False,ddof=1)).to_numpy()
            else:values=getattr(rolling,op.removeprefix('ts_'))().to_numpy()
        paired=np.isfinite(x.values)
        if len(children)==2:paired=paired & np.isfinite(children[1].values)
        count=pd.DataFrame(paired.astype(float)).rolling(window,min_periods=1).sum().to_numpy()
        reasons=np.full(values.shape,CODES[M.UNDEFINED.value],dtype=np.uint8)
        reasons[count<window]=CODES[M.INCOMPLETE.value];reasons[:window-1]=CODES[M.HISTORY.value]
        return self._clean(values,reasons,x.domain)

    def _visit(self,node):
        op=node['op'];params=node['parameters']
        if op=='ref':return self._leaf(params['key'],params['definition_id'])
        if op=='const':return _Array(np.array([[params['value']]],dtype=float),np.zeros((1,1),dtype=np.uint8),'scalar')
        args=[self._visit(child) for child in node['args']];x=args[0]
        domains=[a.domain for a in args if a.domain!='scalar'];domain=domains[0] if domains else 'scalar'
        if op=='broadcast':return _Array(np.broadcast_to(x.values,self.shape),np.broadcast_to(x.reasons,self.shape),'stock_day')
        if op in {'cs_rank','cs_zscore'}:
            values=np.where(self.member,x.values,np.nan);frame=pd.DataFrame(values)
            if op=='cs_rank':values=frame.rank(axis=1,method='average',pct=True).to_numpy()
            else:
                with np.errstate(all='ignore'):values=((frame.sub(frame.mean(axis=1),axis=0)).div(frame.std(axis=1,ddof=1),axis=0)).to_numpy()
            reasons=x.reasons.copy();reasons[~self.member]=CODES[M.UNCOVERED.value]
            return self._clean(values,reasons,'stock_day')
        if op in {'lag','delta'}:
            periods=params['periods'];values=np.full(x.values.shape,np.nan);reasons=np.full(x.values.shape,CODES[M.HISTORY.value],dtype=np.uint8)
            if periods==0:values=x.values.copy();reasons=x.reasons.copy()
            elif periods<len(self.days):values[periods:]=x.values[:-periods];reasons[periods:]=x.reasons[:-periods]
            lagged=_Array(values,reasons,x.domain)
            if op=='lag':return lagged
            with np.errstate(all='ignore'):delta=x.values-lagged.values
            why=np.where(x.reasons!=PRESENT,x.reasons,lagged.reasons)
            return self._clean(delta,why,domain)
        if op.startswith('ts_'):return self._rolling(op,args,params['window'])
        if op=='where':
            condition,true,false=args;shape=np.broadcast_shapes(*(a.values.shape for a in args));cv=np.broadcast_to(condition.values,shape)
            values=np.where(cv!=0,true.values,false.values);why=np.where(cv!=0,true.reasons,false.reasons)
            absent=~np.isfinite(cv);values=np.where(absent,np.nan,values);why=np.where(absent,condition.reasons,why)
            return self._clean(values,why,domain)
        reasons=x.reasons
        for other in args[1:]:reasons=np.where(reasons!=PRESENT,reasons,other.reasons)
        with np.errstate(all='ignore'):
            if op=='neg':values=-x.values
            elif op=='abs':values=np.abs(x.values)
            elif op=='log':values=np.where(x.values>0,np.log(x.values),np.nan)
            elif op=='exp':values=np.exp(x.values)
            elif op=='signed_power':values=np.sign(x.values)*np.abs(x.values)**params['power']
            elif op=='add':values=x.values+args[1].values
            elif op=='sub':values=x.values-args[1].values
            elif op=='mul':values=x.values*args[1].values
            elif op=='div':values=np.where(args[1].values!=0,x.values/args[1].values,np.nan)
            elif op=='minimum':values=np.minimum(x.values,args[1].values)
            elif op=='maximum':values=np.maximum(x.values,args[1].values)
            elif op in {'gt','lt'}:
                y=args[1].values;values=(x.values>y if op=='gt' else x.values<y).astype(float)
                values=np.where(np.isfinite(x.values)&np.isfinite(y),values,np.nan)
            else:raise ValueError('Unsupported compiled numerical operator')
        return self._clean(values,reasons,domain)

    def evaluate(self,compiled: CompiledExpression):
        if not isinstance(compiled,CompiledExpression):raise ValueError('Verified compiled expression required')
        verified=verify_compiled(compiled.to_dict(),self.registry)
        if verified.purpose!='alpha_score':raise ValueError('Numerical evaluator exports only dimensionless stock-day scores')
        # Conservative preflight rather than silently running an unbounded panel.
        estimated=np.prod(self.shape)*9*(verified.nodes+len(verified.dependencies)+4)
        if estimated>self.max_buffer:raise ValueError('Estimated numerical buffer budget exceeded')
        result=self._visit(json.loads(verified.ast_json));assert result.domain=='stock_day'
        values=self.membership[['sample_id','trade_date','stock_code']].copy();missing=values.copy()
        values['score']=result.values[self.row_i,self.col_i];missing['score']=np.asarray(REASONS,dtype=object)[result.reasons[self.row_i,self.col_i]]
        values['observed_end']=self.membership.decision_at;values['known_at']=self.membership.decision_at
        dependencies=[];weak=False
        for definition_id in verified.dependencies:
            definition=next(d for d in self.registry.definitions if d.definition_id==definition_id);binding=self.leaves[definition.key]
            dependencies.append({'key':definition.key,'definition_id':definition_id,'materialization_id':binding.materialization_id,
                'source_id':binding.block.metadata['source_id'],'version_id':binding.block.metadata['version_id'],'vintage':binding.block.metadata['vintage']})
            weak=weak or binding.block.metadata['vintage'] in (VintagePolicy.LATEST_ONLY.value,VintagePolicy.UNKNOWN.value)
        implementation=implementation_hashes(('src/alpharesearch/operators.py','src/alpharesearch/dsl.py','src/alpharesearch/contracts.py','src/alpharesearch/features/base.py'))
        identity={'expression_id':verified.expression_id,'context_id':self.context_id,'dependencies':dependencies,'implementation':implementation}
        meta={'schema':'alpha-expression-values-v1','key_columns':['sample_id','trade_date','stock_code'],'source_id':'typed_alpha_expression','version_id':content_id(identity),
            'vintage':VintagePolicy.LATEST_ONLY.value if weak else VintagePolicy.MARKET.value,'expression_id':verified.expression_id,'context_id':self.context_id,
            'universe_id':self.universe_id,'dependencies':dependencies,'implementation_hashes':implementation,'availability':'computed as-of explicit per-session decision clocks; future leaves masked',
            'calendar_sessions':len(self.days),'grid_stocks':len(self.stocks),'grid_cells':int(np.prod(self.shape)),'estimated_buffer_bytes':int(estimated),
            'rolling':'full session windows; sample std/cov ddof=1; rank average ties / valid count',
            'cross_section':'only declared membership; missing leaves excluded, not zero-filled','condition':'nonzero true; missing condition unknown; unused branch does not contaminate result',
            'source_file_authentication':'passed materialization identities are caller declarations; independent file/execution certification not provided',
            'historical_execution_certified':False,'independent_reproduction':False}
        return FeatureBlock(values,missing,{'score':'ratio'},meta).validate()
