"""Fixed breadth library contract; compilation only, never data evaluation."""
from collections import Counter
from src.alpharesearch.meaningful_library import meaningful64, ALPHA_GROUPS
from src.alpharesearch.registry import FeatureDefinition, FeatureRegistry


def test_fixed_library_consumes_all68_with_eight_real_cross_information_families():
    names=[]
    names += ['pv16.'+x for x in ('amount_ratio_5_20','amount_volatility_20','bias_20','bias_5','bias_60','body_1','close_location','range_1','return_1','return_20','return_5','return_60','volatility_20','volatility_5','volatility_60','volume_ratio_5_20')]
    names += ['alpha.'+x for group in ALPHA_GROUPS for x in group]
    names += ['boards.'+x for x in ('overnight_gap_ratio','upper_close_streak','lower_close_streak','upper_open_lower_close','lower_open_upper_close','one_price_upper_limit','one_price_lower_limit','quote_gap_sessions')]
    names += ['lhb_latest.lhb_w1_'+x for x in ('report_present','age_sessions','reported_net_imbalance','single_day_net_to_reported_daily_amount','buy_hhi','sell_hhi','buy_top1_share','sell_top1_share','buy_institution_amount_share','sell_institution_amount_share','buy_northbound_amount_share','sell_northbound_amount_share')]
    names += ['financial.'+x for x in ('reported_market_cap_proxy','reported_shares_proxy','annual_ep','annual_bp','annual_profitability_proxy','annual_publication_age_days','quarter_publication_age_days','quarter_stale')]
    names += ['market.'+x for x in ('market_breadth_20','market_breadth_60','market_return_20','market_volatility_20')]
    definitions=[]
    for key in names:
        unit='cny' if key.endswith('market_cap_proxy') else ('shares' if key.endswith('shares_proxy') else 'ratio')
        definitions.append(FeatureDefinition(key,unit,'market_day' if key.startswith('market.') else 'stock_day',
            'Synthetic fixed-library definition',('synthetic.source',),'{}',(('tests/test_meaningful_library.py','a'*64),),
            'conservative decision clock','unknown remains missing'))
    registry=FeatureRegistry(definitions)
    before=registry.to_dict();library=meaningful64(registry)
    assert len(library)==64 and len({x['expression']['expression_id'] for x in library})==64
    assert set(Counter(x['family'] for x in library).values())=={8}
    assert len({x['family'] for x in library})==8
    assert all(x['expression']['additional_history_sessions']==0 for x in library)
    assert {d for x in library for d in x['expression']['dependencies']}=={d.definition_id for d in registry.definitions}
    assert registry.to_dict()==before
    # Every market-dependent expression must explicitly broadcast its market leaf.
    def walk(node,parent=None):
        if node['op']=='ref' and node['parameters']['key'].startswith('market.'):
            assert parent=='broadcast'
        for child in node['args']:walk(child,node['op'])
    for candidate in library:walk(candidate['expression']['ast'])
