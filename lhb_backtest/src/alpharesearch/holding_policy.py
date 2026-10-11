"""Actual signal-close holdings selection; no prices/orders from execution date.

Pure interface draft. Native integration and independent state reconstruction
are separate requirements; calling this module alone does not run an account.
"""
from dataclasses import dataclass, asdict
from datetime import date
import hashlib
import json
import math
from numbers import Integral, Real


def require(condition, message):
    if not condition:
        raise ValueError(message)


def money(value):
    require(isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(float(value)), 'State money must be finite numeric')
    return float(value)


def day(value):
    require(isinstance(value, str), 'State date must be ISO date')
    try:
        result = date.fromisoformat(value)
    except ValueError:
        raise ValueError('State date must be ISO date') from None
    require(result.isoformat() == value, 'State date must be canonical ISO date')
    return value


def identity(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=True, allow_nan=False).encode('utf8')
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ActualHolding:
    stock_code: str
    shares: int
    mark: float
    quote_date: str
    last_buy: str

    def __post_init__(self):
        require(isinstance(self.stock_code, str) and self.stock_code
                and not any(x in self.stock_code for x in ('/', '\\')),
                'Holding stock code is invalid')
        require(isinstance(self.shares, Integral) and not isinstance(self.shares, bool)
                and self.shares > 0, 'Actual holdings require positive integer shares')
        require(money(self.mark) > 0, 'Holding mark must be positive')
        day(self.quote_date); day(self.last_buy)


@dataclass(frozen=True)
class SignalCloseState:
    asof: str
    scenario: str
    cash: float
    receivable: float
    equity: float
    holdings: tuple[ActualHolding, ...]

    def __post_init__(self):
        day(self.asof)
        require(self.scenario in {'configured', 'zero_transaction_cost'},
                'State fee scenario is not explicitly supported')
        require(type(self.holdings) is tuple
                and all(type(h) is ActualHolding for h in self.holdings),
                'State holdings must be immutable ActualHolding tuple')
        codes = tuple(h.stock_code for h in self.holdings)
        require(codes == tuple(sorted(set(codes))),
                'State holdings must be unique and sorted by stock code')
        require(all(h.quote_date <= self.asof and h.last_buy <= self.asof for h in self.holdings),
                'State contains future holding quote or purchase')
        cash, receivable, equity = map(money, (self.cash, self.receivable, self.equity))
        require(cash >= -1e-6 and receivable >= -1e-6 and equity > 0,
                'State cash/receivable/equity is invalid')
        expected = cash + receivable + math.fsum(money(h.mark)*int(h.shares) for h in self.holdings)
        require(abs(equity-expected) <= max(.001, equity*1e-9),
                'State equity differs from cash plus actual holdings and receivable')

    def to_dict(self):
        return dict(schema='signal-close-actual-state-v1', asof=self.asof,
                    scenario=self.scenario, cash=float(self.cash),
                    receivable=float(self.receivable), equity=float(self.equity),
                    holdings=[dict(stock_code=h.stock_code, shares=int(h.shares),
                                   mark=float(h.mark), quote_date=h.quote_date,
                                   last_buy=h.last_buy) for h in self.holdings])

    @property
    def state_id(self):
        return identity(self.to_dict())


@dataclass(frozen=True)
class ActualHoldingPolicy:
    max_positions: int = 100
    new_entry_rank: int = 100
    retain_rank: int = 150
    allocation: float = .98

    def __post_init__(self):
        for value in (self.max_positions, self.new_entry_rank, self.retain_rank):
            require(type(value) is int and 1 <= value <= 5000,
                    'Policy ranks/position limit must be positive bounded integers')
        require(self.new_entry_rank == self.max_positions
                and self.retain_rank >= self.new_entry_rank,
                'Entry rank equals position limit; retention must cover entry rank')
        require(0 < money(self.allocation) <= 1, 'Policy allocation must be in (0,1]')

    @classmethod
    def from_dict(cls, value):
        require(isinstance(value, dict) and set(value) ==
                {'schema', 'max_positions', 'new_entry_rank', 'retain_rank', 'allocation'}
                and value['schema'] == 'actual-holding-rank-buffer-policy-v1',
                'Strict actual holding policy fields required')
        return cls(**{k:v for k,v in value.items() if k != 'schema'})

    def to_dict(self):
        return dict(schema='actual-holding-rank-buffer-policy-v1', **asdict(self))


@dataclass(frozen=True)
class ChosenTarget:
    stock_code: str
    rank: int
    reference_close: float
    target_weight: float
    reason: str


@dataclass(frozen=True)
class HoldingExit:
    stock_code: str
    actual_shares: int
    reason: str


@dataclass(frozen=True)
class ActualHoldingSelection:
    signal_date: str
    execution_date: str
    scenario: str
    state_id: str
    policy_id: str
    targets: tuple[ChosenTarget, ...]
    exits: tuple[HoldingExit, ...]

    def to_dict(self):
        return dict(schema='actual-holding-selection-v1', **asdict(self))


def select_actual_holdings(rows, state, policy, *, signal_date, execution_date):
    """rows: common score decisions for one signal date; no future market state.

    Rank eligible scores anew, descending score then ascending stock code.
    Exit intent does not create an execution vacancy. The native engine must
    enforce actual residual-holding slot and cash limits for every buy fill.
    """
    require(type(state) is SignalCloseState and type(policy) is ActualHoldingPolicy,
            'Explicit immutable actual-state and holding policy required')
    day(signal_date); day(execution_date)
    require(state.asof == signal_date and signal_date < execution_date,
            'Selection must use completed signal-date state before execution')
    require(len(state.holdings) <= policy.max_positions,
            'Actual holdings exceed declared position limit')
    require({'stock_code','trade_date','execution_date','score','close','reason'}.issubset(rows.columns),
            'Scored decision columns are incomplete')
    raw = rows.to_dict('records')
    require(all(r['trade_date'] == signal_date and r['execution_date'] == execution_date
                for r in raw), 'Scores contain wrong signal or execution date')
    codes = [r['stock_code'] for r in raw]
    require(all(isinstance(c,str) and c and not any(x in c for x in ('/', '\\')) for c in codes),
            'Scored stock codes are invalid')
    require(len(codes) == len(set(codes)), 'Duplicate scored stock in signal date')
    eligible = []
    for r in raw:
        if r['reason'] != 'eligible_external_score':
            continue
        require(math.isfinite(money(r['score'])) and money(r['close']) > 0,
                'Eligible score/reference close must be finite and positive close')
        eligible.append(r)
    eligible.sort(key=lambda r: (-float(r['score']), r['stock_code']))
    ranks = {r['stock_code']:i+1 for i,r in enumerate(eligible)}
    by_code = {r['stock_code']:r for r in eligible}
    held = {h.stock_code:h for h in state.holdings}
    retained = {code for code in held if ranks.get(code, math.inf) <= policy.retain_rank}
    exits = tuple(HoldingExit(code, int(h.shares),
                             'outside_retain_rank' if code in ranks else 'no_eligible_score')
                  for code,h in sorted(held.items()) if code not in retained)
    entries = [r['stock_code'] for r in eligible[:policy.new_entry_rank]
               if r['stock_code'] not in held][:policy.max_positions-len(retained)]
    chosen = retained | set(entries)
    targets = tuple(ChosenTarget(code, ranks[code], float(by_code[code]['close']),
                                 float(policy.allocation)/policy.max_positions,
                                 'retain_actual_holding' if code in retained else 'new_entry')
                    for code in sorted(chosen, key=lambda c: (ranks[c], c)))
    return ActualHoldingSelection(signal_date, execution_date, state.scenario,
                                  state.state_id, identity(policy.to_dict()), targets, exits)
