"""Independent selection check, given upstream ledger-reconstructed close state.

This function does not reconstruct the ledger itself and never calls the
production holding selector. Native chronological integration is still needed.
"""
import hashlib
import json
import math


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=True, allow_nan=False).encode('utf8')).hexdigest()


def _selection_check(score_rows, reconstructed_state, policy, selection, *, expected_only=False):
    state = reconstructed_state
    _require(state['schema'] == 'signal-close-actual-state-v1', 'Unknown reconstructed state schema')
    _require(policy['schema'] == 'actual-holding-rank-buffer-policy-v1', 'Unknown holding policy schema')
    _require(selection['schema'] == 'actual-holding-selection-v1', 'Unknown selection schema')
    for k in ('max_positions','new_entry_rank','retain_rank'):
        _require(type(policy[k]) is int and 1 <= policy[k] <= 5000, 'Invalid declared holding rank/limit')
    k, enter, keep = policy['max_positions'], policy['new_entry_rank'], policy['retain_rank']
    _require(enter == k and keep >= enter and 0 < policy['allocation'] <= 1,
             'Invalid declared holding policy')
    sd, ed = selection['signal_date'], selection['execution_date']
    _require(sd == state['asof'] and sd < ed, 'Selection is not signal-close causal')
    _require(selection['scenario'] == state['scenario']
             and state['scenario'] in {'configured','zero_transaction_cost'},
             'Selection fee scenario differs from reconstructed state')
    held = state['holdings'];names = [h['stock_code'] for h in held]
    _require(names == sorted(set(names)) and len(names) <= k, 'Invalid actual holdings membership')
    _require(all(type(h['shares']) is int and h['shares'] > 0 and h['quote_date'] <= sd
                 and h['last_buy'] <= sd and math.isfinite(h['mark']) and h['mark'] > 0 for h in held),
             'Invalid actual shares or future holding state')
    money = [state['cash'],state['receivable'],state['equity']]
    _require(all(math.isfinite(x) for x in money) and state['cash'] >= -1e-6
             and state['receivable'] >= -1e-6 and state['equity'] > 0, 'Invalid reconstructed cash state')
    total = state['cash'] + state['receivable'] + sum(h['shares']*h['mark'] for h in held)
    _require(abs(total-state['equity']) <= max(.001,state['equity']*1e-9),
             'Reconstructed close equity identity differs')
    records = score_rows.to_dict('records')
    _require(all(r['trade_date'] == sd and r['execution_date'] == ed for r in records),
             'Scored signal/execution membership differs')
    _require(len({r['stock_code'] for r in records}) == len(records), 'Duplicate decision scores')
    ordered = sorted((r for r in records if r['reason'] == 'eligible_external_score'),
                     key=lambda r: (-r['score'],r['stock_code']))
    _require(all(math.isfinite(r['score']) and math.isfinite(r['close']) and r['close'] > 0
                 for r in ordered), 'Invalid eligible score/reference close')
    number = {r['stock_code']:i+1 for i,r in enumerate(ordered)}
    old = set(names)
    survivors = [h['stock_code'] for h in held if h['stock_code'] in number
                 and number[h['stock_code']] <= keep]
    newcomers = []
    for r in ordered:
        if number[r['stock_code']] > enter:
            break
        if r['stock_code'] not in old and len(newcomers) < k-len(survivors):
            newcomers.append(r['stock_code'])
    target_names = set(survivors+newcomers)
    expected_targets = [dict(stock_code=r['stock_code'], rank=number[r['stock_code']],
                             reference_close=float(r['close']), target_weight=float(policy['allocation'])/k,
                             reason='retain_actual_holding' if r['stock_code'] in survivors else 'new_entry')
                        for r in ordered if r['stock_code'] in target_names]
    expected_exits = [dict(stock_code=h['stock_code'],actual_shares=h['shares'],
                           reason='outside_retain_rank' if h['stock_code'] in number else 'no_eligible_score')
                      for h in held if h['stock_code'] not in survivors]
    # JSON normalization accounts only for tuple/list serialization, not values.
    if expected_only:
        return expected_targets, expected_exits
    actual_targets = json.loads(json.dumps(selection['targets'], allow_nan=False))
    actual_exits = json.loads(json.dumps(selection['exits'], allow_nan=False))
    _require(actual_targets == expected_targets, 'Actual holding targets/entry/retention reason differs')
    _require(actual_exits == expected_exits, 'Actual holding exits or residual shares differs')
    _require(selection['state_id'] == _hash(state), 'Selection actual-state identity differs')
    _require(selection['policy_id'] == _hash(policy), 'Selection holding-policy identity differs')
    return dict(actual_state_checked=True, production_selector_called=False,
                selected_targets=len(expected_targets),exit_intents=len(expected_exits),
                ledger_state_reconstruction_checked_here=False,
                scope='Pure selection conditional on independently reconstructed upstream state; not native audit alone')


def independently_expected_selection(score_rows, state, policy, *, signal_date, execution_date):
    selection = dict(schema='actual-holding-selection-v1', signal_date=signal_date,
                     execution_date=execution_date, scenario=state['scenario'])
    return _selection_check(score_rows, state, policy, selection, expected_only=True)


def audit_actual_selection(score_rows, reconstructed_state, policy, selection):
    return _selection_check(score_rows, reconstructed_state, policy, selection)


def audit_holding_history(choices, decisions, schedule, ledger, fills, quotes,
                          actions, status, metadata, spec, scenario, sessions, calendar, policy):
    """Replay holdings and signal states without saved positions or selector calls.

    Ledger cash/entitlements and fill validity are checked by the enclosing
    financial audit. This replay uses only those primitive events and source
    quotes/actions; stored state, selection, targets and positions are not inputs
    to its expected holdings. Floating money comparison allows 0.001 absolute.
    """
    import pandas as pd
    import numpy as np
    _require(set(choices) == {'signal_date','execution_date','scenario','state_json','selection_json'},
             'Strict per-scenario holding evidence required')
    executed = {entry:sd for sd,entry in schedule.items() if entry in sessions}
    _require(not choices.execution_date.duplicated().any()
             and set(choices.execution_date) == set(executed)
             and choices.scenario.eq(scenario).all(), 'Holding choice schedule/scenario differs')
    evidence = {r.execution_date:r for r in choices.itertuples()}
    action_map = {(r.stock_code,r.trade_date):r for r in actions.itertuples()}
    # Scope quote lookups from primitive share movements, never saved positions.
    # This avoids expanding the entire historical snapshot into Python objects.
    quote_keys=[];share_balances={}
    ledger_days={d:g for d,g in ledger.groupby('date')}
    for current in sessions:
        for r in ledger_days.get(current,ledger.iloc[:0]).itertuples():
            share_balances[r.stock_code]=share_balances.get(r.stock_code,0)+r.shares_delta
        quote_keys.extend((code,current) for code,qty in share_balances.items() if qty>0)
    needed=pd.MultiIndex.from_tuples(quote_keys,names=['stock_code','trade_date'])
    scoped=quotes.loc[quotes.index.intersection(needed)].reset_index()
    qmap={(r.stock_code,r.trade_date):r for r in scoped.itertuples(index=False)}
    halts = {r.trade_date:{str(x).replace('.SS','.SH') for x in r.symbols}
             for r in status.loc[status.status_type.eq('HALT')].itertuples()}
    delisted = dict(zip(metadata.stock_code, metadata.de_listed_date.astype(str)))
    fg = {d:g for d,g in fills.groupby('date')}
    lg = {d:g for d,g in ledger.groupby('date')}
    dg = {d:g for d,g in decisions.groupby('trade_date')}
    held = {};cash=float(spec.execution.initial_cash);receivable=0.;planned=[];members={}
    for index, current in enumerate(sessions):
        if current in executed:
            sd = executed[current]
            ci=calendar.index(current)
            _require(ci>0 and calendar[ci-1]==sd and (index==0 or sessions[index-1]==sd),
                     'Holding state is not immediately preceding completed signal close')
            value=0.
            for h in held.values():value += h['shares']*h['mark']
            state=dict(schema='signal-close-actual-state-v1',asof=sd,scenario=scenario,
                       cash=cash,receivable=receivable,equity=cash+receivable+value,
                       holdings=[dict(stock_code=code,**held[code]) for code in sorted(held)])
            row=evidence[current];stored=json.loads(row.state_json);selected=json.loads(row.selection_json)
            _require(row.signal_date==sd and stored.keys()==state.keys(), 'Holding state keys or signal date differ')
            _require(stored['schema']==state['schema'] and stored['asof']==sd and stored['scenario']==scenario,
                     'Holding state causal/scenario identity differs')
            _require(len(stored['holdings'])==len(state['holdings']), 'Reconstructed actual holdings membership differs')
            for actual,expected in zip(stored['holdings'],state['holdings']):
                _require(set(actual)==set(expected), 'Holding state fields differ')
                for k in ('stock_code','shares','quote_date','last_buy'):
                    _require(actual[k]==expected[k], 'Reconstructed actual holding '+k+' differs')
                _require(math.isfinite(actual['mark']) and abs(actual['mark']-expected['mark'])<=1e-9,
                         'Reconstructed actual holding mark differs')
            for k in ('cash','receivable','equity'):
                _require(math.isfinite(stored[k]) and abs(stored[k]-state[k])<=.001,
                         'Reconstructed signal-close '+k+' differs')
            _require(selected['signal_date']==sd and selected['execution_date']==current,
                     'Holding selection clocks differ from evidence')
            audit_actual_selection(dg[sd],stored,policy,selected)
            expected_targets, _ = independently_expected_selection(dg[sd],state,policy,
                                           signal_date=sd,execution_date=current)
            members[sd]={r['stock_code'] for r in expected_targets}
            for target in expected_targets:
                planned.append(dict(signal_date=sd,execution_date=current,stock_code=target['stock_code'],
                    equity_used=state['equity'],reference_close=target['reference_close'],target_weight=target['target_weight'],
                    target_shares=np.floor(state['equity']*target['target_weight']/target['reference_close']/100)*100,
                    rank=target['rank']))
        # Entitlements and adjusted carrying marks occur after the frozen choice.
        for code,h in held.items():
            a=action_map.get((code,current))
            if a is not None:
                h['shares'] += int(np.floor(h['shares']*a.allotted_ps+1e-5))
                h['mark']=(h['mark']-a.dividend+a.rationed_ps*a.rationed_px)/(1+a.allotted_ps+a.rationed_ps)
        today=fg.get(current,fills.iloc[:0])
        for side in ('sell','buy'):
            for r in today.loc[today.side.eq(side)].itertuples():
                code=r.stock_code;qty=int(r.filled_shares)
                if side=='sell':
                    _require(code in held and qty<=held[code]['shares'], 'Holding replay sells unavailable shares')
                    held[code]['shares'] -= qty
                    if held[code]['shares']==0:del held[code]
                else:
                    h=held.setdefault(code,dict(shares=0,mark=float(r.price),quote_date=current,last_buy=current))
                    h['shares']+=qty;h['last_buy']=current
        for r in lg.get(current,ledger.iloc[:0]).itertuples():
            cash+=float(r.cash_flow);receivable+=float(r.receivable_flow)
        for code,h in held.items():
            q=qmap.get((code,current))
            if current>=delisted.get(code,'9999-12-31') or code in halts.get(current,set()):continue
            if q is not None and np.isfinite(q.close) and q.close>0:
                h['mark']=float(q.close);h['quote_date']=current
    return planned,members,dict(schema='independent-holding-history-audit-v1',
            choices=len(choices),primitive_ledger_state_reconstruction=True,production_selector_called=False,
            stored_positions_used_for_expected_state=False,scenario=scenario)
