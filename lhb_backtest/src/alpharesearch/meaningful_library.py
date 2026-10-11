"""Exactly 64 predeclared cross-information expressions; no data or model access.

All directions, pairings and market thresholds are fixed hypotheses. Compile
against the original complete68 registry; this does not evaluate or select them.
"""
from .dsl import Expr, compile_expression
from .prepared_assembly import verify_pointwise_expression

ALPHA_GROUPS = (
    ('boards_price_00_negative', 'boards_price_05_positive', 'boards_price_06_positive'),
    ('higher_order_00_negative', 'higher_order_15_negative'),
    ('lhb_balance_00_negative', 'lhb_balance_10_positive', 'lhb_balance_11_negative'),
    ('lhb_seat_00_negative', 'lhb_seat_07_positive'),
    ('market_regime_00_negative', 'market_regime_02_negative', 'market_regime_14_negative'),
    ('price_liquidity_00_negative', 'price_liquidity_11_positive'),
    ('price_risk_00_negative', 'price_risk_15_positive'),
    ('value_risk_00_negative', 'value_risk_06_negative', 'value_risk_14_negative'),
)


def meaningful64(registry):
    """Return eight families of eight distinct, zero-additional-history ASTs."""
    out = []
    call = Expr.call
    const = Expr.const
    def ref(key):
        definition = registry.resolve(key)
        node = Expr.ref(key, definition.definition_id)
        return call('broadcast', node) if definition.domain == 'market_day' else node
    def rank(key):
        if registry.resolve(key).domain != 'stock_day':
            raise ValueError('Market constants must gate states, never cross-sectional rank')
        return call('cs_rank', ref(key))
    def centered(key):
        return call('sub', rank(key), const(.5))
    def add(family, label, expression, hypothesis):
        compiled = compile_expression(expression, registry)
        verify_pointwise_expression(compiled, registry)
        out.append(dict(candidate_id=family+'_'+str(sum(x['family']==family for x in out)).zfill(2),
                        hypothesis=label+': '+hypothesis, family=family, expression=compiled.to_dict()))
    def event(expression):
        # Confirmed absence does not synthesize a zero flow. Its original net leaf
        # stays missing; unknown presence propagates the condition missing reason.
        return call('where', call('gt', ref('lhb_latest.lhb_w1_report_present'), const(.5)),
                    expression, ref('lhb_latest.lhb_w1_reported_net_imbalance'))
    pv = 'pv16.'
    for horizon in (1, 5, 20, 60):
        for liquidity in ('volume_ratio_5_20', 'amount_ratio_5_20'):
            add('momentum_liquidity', str(horizon)+'_'+liquidity,
                call('mul', centered(pv+'return_'+str(horizon)), rank(pv+liquidity)),
                'Relative momentum is amplified by independently observed trading activity')
    board_pairs = (
        ('overnight_gap_ratio', 'volume_ratio_5_20', 1),
        ('upper_close_streak', 'return_20', 1),
        ('lower_close_streak', 'return_5', -1),
        ('upper_open_lower_close', 'body_1', -1),
        ('lower_open_upper_close', 'body_1', 1),
        ('one_price_upper_limit', 'amount_ratio_5_20', 1),
        ('one_price_lower_limit', 'return_1', -1),
    )
    for board, price, sign in board_pairs:
        expr = call('mul', rank('boards.'+board), centered(pv+price))
        if sign < 0:expr = call('neg', expr)
        add('board_price', board, expr, 'Board structure conditions a separate price or activity signal')
    add('board_price', 'quote_gap_risk', call('neg', call('add', rank('boards.quote_gap_sessions'), rank(pv+'volatility_20'))),
        'Penalize quote discontinuity together with realized price risk')
    net_keys = ('lhb_latest.lhb_w1_reported_net_imbalance',
                'lhb_latest.lhb_w1_single_day_net_to_reported_daily_amount')
    for net in net_keys:
        for price in ('return_5', 'return_20', 'amount_ratio_5_20', 'volatility_20'):
            add('net_price', net.split('.')[-1]+'_'+price,
                event(call('mul', centered(net), centered(pv+price))),
                'Net disclosed flow interacts with price confirmation, activity or risk')
    for side in ('buy', 'sell'):
        for field in ('hhi', 'top1_share', 'institution_amount_share', 'northbound_amount_share'):
            seat = rank('lhb_latest.lhb_w1_'+side+'_'+field)
            if side == 'sell':seat = call('sub', const(1.), seat)
            add('seat_net', side+'_'+field, event(call('mul', centered(net_keys[0]), seat)),
                'Buy-side concentration/support and sell-side pressure modify net flow separately')
    for net in net_keys:
        decay = call('div', ref(net), call('add', const(1.), ref('lhb_latest.lhb_w1_age_sessions')))
        for context in (pv+'return_5', pv+'volume_ratio_5_20', 'financial.annual_ep', pv+'volatility_20'):
            weight = rank(context)
            if context.endswith('volatility_20'):weight = call('sub', const(1.), weight)
            add('event_age_context', net.split('.')[-1]+'_'+context.split('.')[-1], event(call('mul', decay, weight)),
                'Session-age decay is combined with a separate price, activity, value or risk context')
    financial = ('reported_market_cap_proxy', 'reported_shares_proxy', 'annual_ep', 'annual_bp',
                 'annual_profitability_proxy', 'annual_publication_age_days',
                 'quarter_publication_age_days', 'quarter_stale')
    for field in financial:
        value = centered('financial.'+field)
        if field in ('reported_market_cap_proxy', 'reported_shares_proxy') or 'age_days' in field or field=='quarter_stale':
            value = call('neg', value)
        add('value_publication_price', field, call('add', value, centered(pv+'return_20')),
            'Value/size or disclosed publication freshness combines with a fixed price trend')
    regimes = (('market_breadth_20', .5), ('market_breadth_60', .5),
               ('market_return_20', 0.), ('market_volatility_20', .01))
    for market, threshold in regimes:
        gate = call('gt', ref('market.'+market), const(threshold))
        for stock in (pv+'return_5', net_keys[0]):
            signal = centered(stock)
            expr = call('where', gate, signal, call('neg', signal))
            if stock in net_keys:expr = event(expr)
            add('market_stock_state', market+'_'+stock.split('.')[-1], expr,
                'A fixed market-state threshold reverses the stock signal, rather than rescaling every stock equally')
    contexts = ('bias_5', 'bias_20', 'bias_60', 'close_location', 'range_1',
                'amount_volatility_20', 'volatility_5', 'volatility_60')
    for group, context in zip(ALPHA_GROUPS, contexts):
        terms = [centered('alpha.'+key) for key in group]
        total = terms[0]
        for term in terms[1:]:total = call('add', total, term)
        consensus = call('div', total, const(float(len(terms))))
        add('prior_alpha_context', group[0].split('_00')[0], call('mul', consensus, centered(pv+context)),
            'Frozen prior-family consensus interacts with a separately observed price representation; prior selection lineage retained')
    ids = [x['expression']['expression_id'] for x in out]
    if len(out)!=64 or len(set(ids))!=64:
        raise ValueError('Exactly64 distinct expressions required')
    used = {d for x in out for d in x['expression']['dependencies']}
    if used != {d.definition_id for d in registry.definitions} or len(registry.definitions)!=68:
        raise ValueError('Fixed library must consume exactly the original complete68 definitions')
    return out
