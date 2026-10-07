"""Versioned feature definitions and separate, honest materialization receipts.

A definition describes an algorithm. Registering today's algorithm cannot prove
that a previously cached matrix was calculated by that algorithm.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .contracts import Unit, VintagePolicy, required_id
from .features.base import FeatureBlock
from ..technical.artifacts import content_id

DOMAINS = {'stock_day', 'market_day', 'report'}
KEY = re.compile(r'^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$')
SHA = re.compile(r'^[0-9a-f]{64}$')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hashes(records):
    if isinstance(records,dict):records=records.items()
    pairs=tuple(sorted(tuple(pair) for pair in records))
    if any(len(pair)!=2 for pair in pairs):raise ValueError('Artifact hashes need name/hash pairs')
    if len(set(name for name,_ in pairs))!=len(pairs):raise ValueError('Duplicate implementation/artifact identifier')
    for name,sha in pairs:
        required_id(name,'artifact identifier')
        if not isinstance(sha,str) or not SHA.fullmatch(sha):raise ValueError('Artifact needs a SHA256 hash')
    return pairs


@dataclass(frozen=True)
class FeatureDefinition:
    key: str
    unit: str
    domain: str
    description: str
    dependencies: tuple
    parameters_json: str
    implementation_hashes: tuple
    timing_rule: str
    missing_rule: str
    limitations: tuple = ()

    def __post_init__(self):
        if not isinstance(self.key,str) or not KEY.fullmatch(self.key):raise ValueError('Qualified feature key required')
        object.__setattr__(self,'unit',Unit(self.unit).value)
        if self.domain not in DOMAINS:raise ValueError('Unknown feature domain')
        for name in ('description','timing_rule','missing_rule'):required_id(getattr(self,name),name)
        for name in ('dependencies','limitations'):
            items=tuple(getattr(self,name))
            if isinstance(getattr(self,name),str) or len(items)!=len(set(items)):raise ValueError('Unique descriptor items required')
            for value in items:required_id(value,name)
            object.__setattr__(self,name,items)
        if not self.dependencies:raise ValueError('A feature needs declared dependencies')
        params=json.loads(self.parameters_json)
        if not isinstance(params,dict):raise ValueError('Feature parameters must be a JSON object')
        object.__setattr__(self,'parameters_json',_json(params))
        hashes=_hashes(self.implementation_hashes)
        if not hashes:raise ValueError('Definition needs implementation hashes')
        object.__setattr__(self,'implementation_hashes',hashes)

    @property
    def definition_id(self):return content_id(asdict(self))

    def to_dict(self):return dict(asdict(self),definition_id=self.definition_id)

    @classmethod
    def from_dict(cls,value):
        value=dict(value);expected=value.pop('definition_id');result=cls(**value)
        if result.definition_id!=expected:raise ValueError('Feature definition identity changed')
        return result


class FeatureRegistry:
    def __init__(self,definitions=()):
        self._definitions={}
        for definition in definitions:self.add(definition)

    def add(self,definition):
        if not isinstance(definition,FeatureDefinition):raise ValueError('Typed feature definition required')
        old=self._definitions.get(definition.definition_id)
        if old is not None and old!=definition:raise ValueError('Feature identity collision')
        self._definitions[definition.definition_id]=definition
        return definition.definition_id

    def resolve(self,key,definition_id=None):
        candidates=[x for x in self._definitions.values() if x.key==key and (definition_id is None or x.definition_id==definition_id)]
        if len(candidates)!=1:raise ValueError('Unknown or ambiguous feature; select an explicit definition version')
        return candidates[0]

    def resolve_bound_definitions(self, source_bindings, keys):
        """Build one call-local lookup; never cache across registry changes.

        Duplicate bindings to one ID remain unambiguous. Unregistered IDs in
        unrelated source records are ignored as by the former selected-key scan;
        every requested key still needs exactly one registered bound version.
        Metadata is a declaration, not proof of the source calculation.
        """
        requested = set(keys)
        by_id = {d.definition_id: d for d in self.definitions}
        bound = {key: set() for key in requested}
        for binding in source_bindings:
            for definition_id in binding["definition_ids"]:
                definition = by_id.get(definition_id) if isinstance(definition_id, str) else None
                if definition is not None and definition.key in requested:
                    bound[definition.key].add(definition_id)
        if any(len(ids) != 1 for ids in bound.values()):
            raise ValueError("Unknown or ambiguous bound feature definition; select an explicit version")
        return {key: by_id[next(iter(ids))] for key, ids in bound.items()}

    @property
    def version_id(self):return content_id([x.to_dict() for x in self.definitions])

    @property
    def definitions(self):return tuple(sorted(self._definitions.values(),key=lambda x:(x.key,x.definition_id)))

    def to_dict(self):return {'schema':'feature-registry-v1','version_id':self.version_id,'definitions':[x.to_dict() for x in self.definitions]}

    @classmethod
    def from_dict(cls,value):
        if set(value)!={'schema','version_id','definitions'} or value['schema']!='feature-registry-v1':raise ValueError('Unknown registry schema')
        definitions=[FeatureDefinition.from_dict(x) for x in value['definitions']]
        if len(set(x.definition_id for x in definitions))!=len(definitions):raise ValueError('Duplicate serialized definition')
        result=cls(definitions)
        if result.version_id!=value['version_id']:raise ValueError('Registry identity changed')
        return result


SCHEMAS = {
    'daily-observation-primitives-v1': ('daily','stock_day',['src/alpharesearch/features/primitives.py']),
    'legacy-price-volume16-v1': ('pv16','stock_day',['src/mlresearch/dataset.py','src/mlresearch/contracts.py']),
    'daily-structure-boards-v1': ('boards','stock_day',['src/alpharesearch/features/boards.py']),
    'lhb-report-features-v1': ('lhb_report','report',['src/alpharesearch/features/lhb.py','src/alpharesearch/panel.py']),
    'lhb-latest-decision-features-v1': ('lhb_latest','stock_day',['src/alpharesearch/features/lhb.py','src/alpharesearch/panel.py']),
    'financial-proxies-v1': ('financial','stock_day',['src/alpharesearch/features/financial.py']),
    'market-context-v1': ('market','market_day',['src/alpharesearch/features/market.py']),
}


def implementation_hashes(paths):
    """Normalized source text identity, portable across Git LF/CRLF checkout."""
    app=Path(__file__).resolve().parents[2];result={}
    for name in paths:
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or path.suffix!='.py':raise ValueError('Project-relative Python implementation required')
        target=app/path
        if target.is_symlink() or not target.resolve().is_relative_to(app):raise ValueError('Implementation escapes project')
        result[name]=hashlib.sha256(target.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    return _hashes(result.items())


def _description(namespace,name):
    from ..mlresearch.contracts import FEATURES
    if namespace=='daily':
        descriptions={'open':'供应商未复权日开盘价','high':'供应商未复权日最高价','low':'供应商未复权日最低价','close':'供应商未复权日收盘价','pre_close':'供应商当日参考前收价；不猜作昨日实际收盘','high_limit':'明确有效源涨停价','low_limit':'明确有效源跌停价','volume':'源声明的成交股数，允许合法0','amount':'源声明的人民币成交金额，允许合法0','source_limit_valid':'源价格限制有效性标记；0不代表市场没有涨跌幅限制','positive_volume_observed':'观察成交量是否正；不证明订单可成交'}
        return descriptions[name]
    if namespace=='pv16':return FEATURES[name]+'；保留原算法兼容口径'
    if namespace=='lhb_latest':
        match=re.fullmatch(r'lhb_w(0|1|3|10|30)_(.*)',name)
        if match is None:raise ValueError('Unknown LHB projection feature')
        window,field=match.groups()
        inner={'report_present':'已知最新普通报告存在标记；确定0需要全覆盖声明','age_sessions':'距选中原报告多少真实交易日'}.get(field)
        return ('未知' if window=='0' else window+'日')+'窗口最新普通报告：'+(inner or _description('lhb_report',field))
    if namespace=='lhb_report':
        side=re.fullmatch(r'(buy|sell)_(.*)',name)
        if side:
            label,field=side.groups()
            meanings={'disclosed_amount':'披露名单本侧金额之和；不重建全交易者合并总额','disclosed_rows':'披露源行数量；同名匿名行不合并','top1_share':'最大一行本侧金额/本侧名单金额','top2_share':'前两行本侧金额/本侧名单金额','hhi':'本侧名单金额占比平方和Σp²','entropy_log5':'本侧名单熵−Σp×log(p)/log(5)','institution_labeled_rows':'名称恰为机构专用的披露行数','institution_amount_share':'机构专用标签本侧金额/本侧名单金额','northbound_labeled_rows':'名称恰为沪股通专用或深股通专用的披露行数','northbound_amount_share':'沪深股通标签本侧金额/本侧名单金额'}
            return ('买入' if label=='buy' else '卖出')+'列表：'+meanings[field]
        meanings={'reported_buy_total':'原报告汇总买入额','reported_sell_total':'原报告汇总卖出额','reported_turnover':'原报告汇总成交额','net_from_reported_totals':'原报告汇总买额−汇总卖额','reported_net_imbalance':'(汇总买额−汇总卖额)/(汇总买额+汇总卖额)','reported_buy_sell_ratio':'汇总买额/汇总卖额；分母须正','single_day_net_to_reported_daily_amount':'仅单日普通榜：汇总净额/源表单日成交额','single_day_turnover_to_reported_daily_amount':'仅单日普通榜：汇总榜成交额/源表单日成交额','window_days':'报告实际声明窗口天数；未知不猜1','window_known':'原报告窗口是否能明确识别','is_ordinary':'普通席位披露标记','is_margin_buy':'融资买入披露标记','is_short_sell':'融券卖出披露标记','is_investor_category':'可能重叠投资者类别披露标记'}
        return meanings[name]
    if namespace=='financial':
        meanings={'reported_shares_proxy':'最新已知且未陈旧季度报告股本；非实时股本登记','reported_market_cap_proxy':'当日原价收盘×最新已公布且未陈旧报告股本','annual_parent_profit':'最新已知且未陈旧12月年度归母净利润','annual_total_equity':'最新已知且未陈旧12月年度总股东权益','annual_total_assets':'最新已知且未陈旧12月年度总资产','annual_ep':'年度归母净利润/报告股本市值代理','annual_bp':'年度总权益/报告股本市值代理','annual_profitability_proxy':'年度归母净利润/正的年度总权益；不是精确归母ROE'}
        if name in meanings:return meanings[name]
        match=re.fullmatch(r'(quarter|annual)_(period_age_days|publication_age_days|stale)',name)
        if match is None:raise ValueError('Unknown financial feature')
        return ('季度' if match[1]=='quarter' else '年度')+{'period_age_days':'报告期末至决策日期的自然日数','publication_age_days':'公告日期至决策日期的自然日数','stale':'报告期末年龄超过明确阈值的标记'}[match[2]]
    if namespace=='market':
        meanings={'market_declared_members':'调用方声明的市场代理成员数；非交易所全覆盖证明','market_valid_members':'有限正价格/参考价及正成交量的成员数','market_valid_fraction':'有效成员数/声明成员数','market_daily_mean_log_return':'成员log(收盘/参考前收)的横截面平均','market_daily_breadth':'有效成员中当日价格回报为正的比例','market_daily_cross_section_dispersion':'成员日对数回报的横截面样本标准差'}
        if name in meanings:return meanings[name]
        match=re.fullmatch(r'market_(return|volatility|breadth)_(\d+)',name)
        if match is None:raise ValueError('Unknown market feature')
        return match[2]+'个完整交易日：'+{'return':'日平均对数回报之和，非账户收益','volatility':'日平均对数回报的时间样本标准差','breadth':'每日上涨成员比例的时间均值'}[match[1]]
    if namespace=='boards':
        meanings={'upper_wick_fraction':'(最高−max(开盘,收盘))/(最高−最低)','lower_wick_fraction':'(min(开盘,收盘)−最低)/(最高−最低)','absolute_body_fraction':'abs(收盘−开盘)/(最高−最低)','body_direction':'sign(收盘−开盘)','zero_range_bar':'有效交易行情的最高价是否等于最低价','overnight_gap_ratio':'开盘/参考前收−1','quote_gap_sessions':'本次与前次行情之间缺少的市场交易日数','hit_both_limits':'同日OHLC同时触及两端；不推断顺序','upper_touch_close_inside':'触及涨停但未收涨停','lower_touch_close_inside':'触及跌停但未收跌停','upper_open_lower_close':'开涨停且收跌停；不证明盘中路径','lower_open_upper_close':'开跌停且收涨停；不证明盘中路径','upper_limit_distance_close':'(涨停价−收盘)/参考前收','lower_limit_distance_close':'(收盘−跌停价)/参考前收','limit_span_ratio':'(涨停价−跌停价)/参考前收'}
        if name in meanings:return meanings[name]
        match=re.fullmatch(r'(close|open|touched|one_price)_(upper|lower)_limit',name)
        if match:return {'close':'收盘等于','open':'开盘等于','touched':'最高/最低触及','one_price':'开高低收全部等于'}[match[1]]+('涨停价' if match[2]=='upper' else '跌停价')+'；需要有效历史价格限制和正成交量'
        match=re.fullmatch(r'(upper|lower)_close_streak(_lower_bound)?',name)
        if match:return ('收涨停' if match[1]=='upper' else '收跌停')+('已观察连续次数下界；未知起点不猜完整' if match[2] else '连续真实交易日次数；必须有已知非命中起点')
        match=re.fullmatch(r'age_since_observed_(upper|lower)_close',name)
        if match:return '距最近已观察到的'+('收涨停' if match[1]=='upper' else '收跌停')+'事件的交易日数；不认证历史事件无遗漏'
    raise ValueError('Feature needs an explicit readable definition')


def definitions_for_block(block: FeatureBlock):
    """Describe the current implementation; never certify a cached execution.

    Parameters are actual block declarations, excluding data-instance identities.
    A new computation's executor must separately freeze/verify implementation hashes.
    """
    block.validate();schema=block.metadata['schema']
    if schema not in SCHEMAS:raise ValueError('Feature schema needs an explicit definition adapter')
    namespace,domain,code=SCHEMAS[schema]
    common=['src/alpharesearch/contracts.py','src/alpharesearch/features/base.py','src/alpharesearch/registry.py']
    if namespace=='pv16':code=code+['src/alpharesearch/features/legacy.py','src/technical/foundation_data.py']
    hashes=implementation_hashes(code+common)
    instance={'source_id','version_id','key_columns','vintage','filing_vintage','quote_vintage','quote_source_id','quote_version_id',
              'universe_id','rows','feature_count','membership_rows','weak_vintage_allowed'}
    parameters={k:v for k,v in block.metadata.items() if k not in instance}
    timing=str(block.metadata.get('timing',block.metadata.get('clock',block.metadata.get('availability',block.metadata.get('availability_basis','declared per-row clocks')))))
    missing=str(block.metadata.get('missing','explicit reasons; finite values exactly match present'))
    limitations={
        'daily':('Unadjusted prices may contain corporate-action gaps; zero-volume quotes do not prove tradability',),
        'pv16':('Legacy quote-row windows preserved; not a new strict-calendar algorithm','Flat-range close location is 0.5; cumulative index fills missing daily returns with zero internally'),
        'boards':('Daily OHLC cannot reveal intraday ordering or queue position',),
        'lhb_report':('Disclosed labels are not beneficial-owner IDs; report windows may overlap',),
        'lhb_latest':('Latest ordinary report per window is a chosen representation; no overlap sum',),
        'financial':('Reported size and profitability are proxies; latest snapshot does not restore historical revisions',),
        'market':('Mean log-return proxy is not a tradable index account return',),
    }
    dependencies={
        'daily':('daily.reported_ohlcv_amount_reference_close','daily.declared_valid_price_limits'),
        'pv16':('daily.ohlcv_amount_reference_close','market.calendar'),
        'boards':('daily.ohlcv_reference_close','daily.validated_price_limits','market.calendar'),
        'lhb_report':('lhb.original_report_headers','lhb.original_disclosed_occurrences'),
        'lhb_latest':('lhb.original_report_features','decision.original_report_links','decision.membership'),
        'financial':('financial.published_filings','daily.current_close_and_clocks','decision.membership'),
        'market':('daily.current_close_reference_close_volume','market.declared_membership','market.calendar'),
    }
    return tuple(FeatureDefinition(namespace+'.'+name,unit,domain,_description(namespace,name),
        dependencies[namespace],_json(parameters),hashes,timing,missing,limitations[namespace]) for name,unit in sorted(block.units.items()))


def coverage_diagnostics(block: FeatureBlock):
    """Retrospective coverage, not model normalization, factor selection or earnings."""
    block.validate();columns={}
    for name in block.units:
        values=block.values[name].to_numpy(dtype=float);good=np.isfinite(values)
        columns[name]={'present':int(good.sum()),'rows':len(values),'missing_reasons':block.missing[name].value_counts().to_dict(),
                       'min':float(values[good].min()) if good.any() else None,'max':float(values[good].max()) if good.any() else None}
    return {'schema':'feature-coverage-v1','rows':len(block.values),'columns':columns,'fits':0,'selection_applied':False}


def materialization_receipt(block: FeatureBlock, definitions, *, artifact_hashes,
                            execution_hashes=None, execution_ref=None, input_hashes=()):
    """Bind declared definitions and actual artifact/input hashes separately.

    Matching declared execution hashes is a consistency check, not authentication
    that an external executor ran them or an independent reproduction occurred.
    """
    block.validate();definitions=tuple(definitions)
    if len(definitions)!=len(block.units) or len(set(x.key for x in definitions))!=len(definitions):raise ValueError('One definition per feature required')
    if {x.key.split('.',1)[1] for x in definitions}!=set(block.units) or len({x.key.split('.',1)[0] for x in definitions})!=1:
        raise ValueError('Definitions do not cover one materialized namespace')
    for definition in definitions:
        name=definition.key.split('.',1)[1]
        if name not in block.units or definition.unit!=block.units[name]:raise ValueError('Definition does not match materialized feature unit/name')
    artifacts=_hashes(artifact_hashes)
    if not artifacts:raise ValueError('Materialization needs artifact hashes')
    execution=() if execution_hashes is None else _hashes(execution_hashes)
    expected={}
    for definition in definitions:
        for name,sha in definition.implementation_hashes:
            if name in expected and expected[name]!=sha:raise ValueError('Conflicting definition implementations')
            expected[name]=sha
    if execution_ref is not None:required_id(execution_ref,'execution_ref')
    matches=bool(execution) and all(dict(execution).get(name)==sha for name,sha in expected.items())
    if execution_ref is None and execution:raise ValueError('Declared execution hashes need an execution reference')
    payload={'schema':'feature-materialization-v1','definition_ids':sorted(x.definition_id for x in definitions),
        'source_id':block.metadata['source_id'],'source_version':block.metadata['version_id'],'vintage':VintagePolicy(block.metadata['vintage']).value,
        'rows':len(block.values),'artifacts':artifacts,'input_hashes':_hashes(input_hashes),
        'declared_execution_hashes':execution,'execution_ref':execution_ref,'declared_implementation_matches':matches,
        'historical_execution_certified':False,'independent_reproduction_claimed':False,
        'status':'declared_implementation_matches' if matches else 'execution_not_verified_against_definition'}
    return dict(payload,materialization_id=content_id(payload))


def verify_materialization(receipt):
    value=dict(receipt);expected=value.pop('materialization_id')
    if value.get('schema')!='feature-materialization-v1' or content_id(value)!=expected:raise ValueError('Materialization identity changed')
    for name in ('artifacts','input_hashes','declared_execution_hashes'):_hashes(value[name])
    if value['historical_execution_certified'] or value['independent_reproduction_claimed']:raise ValueError('Registry cannot grant execution/reproduction certification')
    return receipt
