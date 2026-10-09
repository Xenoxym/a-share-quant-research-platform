"""Strict, finite feature batch protocol. Models/accounts are separate later kinds."""
from dataclasses import dataclass
from datetime import date
import json
import re

from .dsl import verify_compiled
from .contracts import required_id
from .registry import FeatureRegistry,SHA

NAME=re.compile(r'^[a-z][a-z0-9_-]{0,79}$')


def _json(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


@dataclass(frozen=True)
class AlphaBatchSpec:
    document_json: str

    @classmethod
    def from_dict(cls,value,registry: FeatureRegistry):
        keys={'schema','name','dataset_id','universe_id','ranges','candidates','blocks','bindings','budget','allow_weak_vintage','evaluation_scope','selection_rule','primary_objectives'}
        if not isinstance(value,dict):raise ValueError('Strict alpha feature batch schema required')
        if value.get('schema')=='alpha-feature-batch-v2':
            keys=keys|{'storage_layout'}
            if value.get('storage_layout')!='shared_index_v1':raise ValueError('Explicit supported feature storage layout required')
        if set(value)!=keys or value.get('schema') not in ('alpha-feature-batch-v1','alpha-feature-batch-v2'):raise ValueError('Strict alpha feature batch schema required')
        doc=json.loads(_json(value))
        for name in ('name','dataset_id','universe_id'):required_id(doc[name],name)
        if doc['evaluation_scope']!='retrospective_time_split' or doc['selection_rule']!='all_candidates_no_selection' or doc['primary_objectives']!=['account_return','max_drawdown']:raise ValueError('Feature batches cannot claim selection or final strategy success')
        if type(doc['allow_weak_vintage']) is not bool:raise ValueError('Explicit vintage opt-in required')
        ranges=doc['ranges'];range_keys={'warmup_start','development_start','development_end','holdout_start','holdout_end'}
        if not isinstance(ranges,dict) or set(ranges)!=range_keys:raise ValueError('Explicit warmup/development/holdout dates required')
        for d in ranges.values():
            if not isinstance(d,str) or date.fromisoformat(d).isoformat()!=d:raise ValueError('Canonical frozen date required')
        if not ranges['warmup_start']<=ranges['development_start']<=ranges['development_end']<ranges['holdout_start']<=ranges['holdout_end']:raise ValueError('Development and holdout ranges overlap or are reversed')
        budget=doc['budget'];budget_keys={'max_candidates','max_input_bytes','max_grid_cells','max_buffer_bytes','max_model_fits','max_accounts'}
        if not isinstance(budget,dict) or set(budget)!=budget_keys or any(type(budget[k]) is not int or budget[k]<1 for k in budget_keys-{'max_model_fits','max_accounts'}):raise ValueError('Finite positive execution/input budgets required')
        if budget['max_model_fits']!=0 or type(budget['max_model_fits']) is not int or budget['max_accounts']!=0 or type(budget['max_accounts']) is not int:raise ValueError('Feature-only batches have zero fit/account budgets')
        candidates=doc['candidates']
        if not isinstance(candidates,list) or not 1<=len(candidates)<=budget['max_candidates'] or len(candidates)>10_000:raise ValueError('Explicit finite candidate list required')
        ids=[]
        for candidate in candidates:
            if not isinstance(candidate,dict) or set(candidate)!={'candidate_id','hypothesis','family','expression'}:raise ValueError('Candidate fields are strict; no free executable model/code')
            if not isinstance(candidate['candidate_id'],str) or not NAME.fullmatch(candidate['candidate_id']):raise ValueError('Candidate identifier required')
            ids.append(candidate['candidate_id']);required_id(candidate['hypothesis'],'candidate hypothesis');required_id(candidate['family'],'candidate family')
            compiled=verify_compiled(candidate['expression'],registry)
            if compiled.purpose!='alpha_score':raise ValueError('Candidate must be a stock-day alpha score')
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate candidate names; repeated expressions need distinct counted candidates')
        blocks=doc['blocks'];bindings=doc['bindings']
        if not isinstance(blocks,dict) or not blocks or not isinstance(bindings,dict) or not bindings:raise ValueError('Explicit source blocks and leaf bindings required')
        roles=set()
        for name,block in blocks.items():
            if not NAME.fullmatch(name) or not isinstance(block,dict) or set(block)!={'values','missing','definition','receipt'}:raise ValueError('Named blocks need exact value/missing/definition roles')
            for role in block.values():
                if not isinstance(role,str) or not NAME.fullmatch(role):raise ValueError('Explicit input role required')
                roles.add(role)
        if len(roles)!=len(blocks)*4 or roles & {'registry','membership','calendar','decision_clocks'}:raise ValueError('Input block roles must be distinct')
        for key,binding in bindings.items():
            if not isinstance(binding,dict) or set(binding)!={'block','column','definition_id','materialization_id'} or binding['block'] not in blocks:raise ValueError('Named leaf binding fields required')
            if not isinstance(binding['column'],str) or not binding['column'].strip() or any(not isinstance(binding[k],str) or not SHA.fullmatch(binding[k]) for k in ('definition_id','materialization_id')):raise ValueError('Bound definition/materialization identities required')
            registry.resolve(key,binding['definition_id'])
        used=set()
        for candidate in candidates:
            for definition_id in candidate['expression']['dependencies']:
                definition=next(d for d in registry.definitions if d.definition_id==definition_id);used.add(definition.key)
                if definition.key not in bindings or bindings[definition.key]['definition_id']!=definition_id:raise ValueError('Candidate reference lacks an exact leaf binding')
        if used!=set(bindings):raise ValueError('Bind only explicitly used inputs; unused feature channels cannot be hidden in a batch')
        if {b['block'] for b in bindings.values()}!=set(blocks):raise ValueError('Unused source block')
        return cls(_json(doc))

    def to_dict(self):return json.loads(self.document_json)

    @property
    def input_roles(self):
        doc=self.to_dict();return {'registry','membership','calendar','decision_clocks'}|{r for b in doc['blocks'].values() for r in b.values()}

    @property
    def planned_attempts(self):
        return [{'candidate_id':c['candidate_id'],'expression_id':c['expression']['expression_id'],'family':c['family'],'status':'planned'} for c in self.to_dict()['candidates']]
