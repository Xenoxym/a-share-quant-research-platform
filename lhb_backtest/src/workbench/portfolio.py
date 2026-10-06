"""Sequential cash-account simulator with explicit daily-bar limitations.

Orders are sized before the session. Missing exits remain positions; no future
tradability filter is allowed to remove a signal. Raw prices and entitlements
are accounted separately. This is an execution model, not observed fills.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np
import pandas as pd

from .data import BAR_COLS, KEYS, check_keys


class Market:
    def __init__(self, bars, calendar, actions=None, status=None, metadata=None, missing_actions=()):
        check_keys(bars, "行情")
        self.columns = [c for c in BAR_COLS if c not in KEYS]
        indexed = bars.set_index(KEYS)
        self.index = indexed.index
        self.values = indexed[self.columns].to_numpy(dtype=float)
        self.calendar = calendar
        self.day_index = {d: i for i, d in enumerate(calendar)}
        self.actions = {} if actions is None else {
            (r.stock_code, r.trade_date): r._asdict() for r in actions.itertuples(index=False)}
        self.halts = {}
        if status is not None:
            self.halts = {r.trade_date: {str(s).replace(".SS", ".SH") for s in r.symbols}
                          for r in status.loc[status.status_type.eq("HALT")].itertuples(index=False)}
        self.delisted = {} if metadata is None else dict(zip(metadata.stock_code, metadata.de_listed_date.astype(str)))
        self.missing_actions = set(missing_actions)

    def quote(self, code, date):
        try:
            idx = self.index.get_loc((code, date))
        except KeyError:
            return None
        return dict(zip(self.columns, self.values[idx]))

    def fill_block(self, code, date, side):
        row = self.quote(code, date)
        if code in self.halts.get(date, set()):
            return "halt", row
        if row is None:
            return "missing_quote", row
        if not all(np.isfinite(row[k]) and row[k] > 0 for k in ("open", "high", "low", "close")):
            return "invalid_quote", row
        if not np.isfinite(row["volume"]) or row["volume"] <= 0:
            return "zero_volume", row
        if side == "buy" and row["is_st"] != 0:
            return "st_or_unknown", row
        row["limit_origin"] = "validated_clean_limits"
        if row["limit_price_valid"] != 1 or not all(np.isfinite(row[k]) for k in ("high_limit", "low_limit")):
            from .rules import st_exit_limits
            limits = st_exit_limits(code, date, row) if side == "sell" else None
            if limits is None:
                return "unknown_price_limits", row
            row["high_limit"], row["low_limit"] = limits
            row["limit_origin"] = "dated_st_rule_model"
        if row["open"] >= row["high_limit"] - 1e-6:
            return "open_at_upper_limit", row
        if row["open"] <= row["low_limit"] + 1e-6:
            return "open_at_lower_limit", row
        return "", row


def fees(value, side, day, card):
    if value <= 0:
        return {"commission": 0., "other_fee": 0., "stamp_tax": 0., "slippage": 0., "total": 0.}
    commission = max(card.minimum_commission, value * card.commission_rate)
    other = value * card.other_fee_rate
    stamp = value * (0.001 if day < "2023-08-28" else 0.0005) if side == "sell" else 0.
    slippage = value * card.slippage_bps / 10000
    return {"commission": commission, "other_fee": other, "stamp_tax": stamp,
            "slippage": slippage, "total": commission + other + stamp + slippage}


def _lot_quantity(budget, price, day, card):
    qty = max(0, math.floor(budget / (price * 100)) * 100)
    while qty and price * qty + fees(price * qty, "buy", day, card)["total"] > budget + 1e-8:
        qty -= 100
    return qty


@dataclass
class Position:
    stock_code: str
    event_id: str
    entry_date: str
    due_index: int
    shares: int
    original_shares: int
    invested: float
    last_mark: float
    last_quote_date: str
    proceeds: float = 0.
    dividend: float = 0.
    deferred_days: int = 0


def simulate(signals, market: Market, sessions, card, name="institutional", fee_multiplier=1.0):
    if not sessions:
        raise ValueError("组合模拟需要至少一个交易日")
    if not np.isfinite(fee_multiplier) or fee_multiplier < 1:
        raise ValueError("费用压力倍数必须为有限数且不小于 1")

    def costs(value, side, day):
        result = fees(value, side, day, card)
        for key in ("commission", "other_fee", "slippage"):
            result[key] *= fee_multiplier
        result["total"] = sum(result[key] for key in ("commission", "other_fee", "stamp_tax", "slippage"))
        return result

    def affordable(budget, price, day):
        qty = _lot_quantity(budget, price, day, card)
        while qty > 0 and qty * price + costs(qty * price, "buy", day)["total"] > budget + 1e-8:
            qty -= 100
        return qty
    cash = float(card.initial_cash)
    receivable = 0.
    equity_before = cash
    positions: dict[str, Position] = {}
    orders, ledger, fills, closed, equity, position_rows, warnings = [], [], [], [], [], [], []
    warning_keys = set()
    grouped = {d: f.to_dict("records") for d, f in signals.groupby("entry_date", sort=False, dropna=True)}

    def warn(day, code, kind, event_id):
        key = (day, code, kind)
        if key not in warning_keys:
            warning_keys.add(key)
            warnings.append({"date": day, "stock_code": code, "kind": kind, "event_id": event_id})

    def book(day, code, event_id, kind, cash_flow=0., receivable_flow=0., shares=0, **extra):
        nonlocal cash, receivable
        cash += cash_flow
        receivable += receivable_flow
        ledger.append({"date": day, "stock_code": code, "event_id": event_id, "kind": kind,
                       "cash_flow": cash_flow, "receivable_flow": receivable_flow, "shares_delta": shares, **extra})

    def order(day, event_id, code, side, requested, filled, reason, price=None, cost=None, limit_origin=None):
        status = "filled" if filled == requested and filled > 0 else "partial" if filled > 0 else "deferred" if side == "sell" else "rejected"
        if reason == "beyond_observation_window":
            status = "pending"
        record = {"order_id": f"{name}-{len(orders) + 1:07d}", "date": day, "event_id": event_id,
                  "stock_code": code, "side": side, "requested_shares": requested,
                  "filled_shares": filled, "unfilled_shares": requested - filled,
                  "status": status, "reason": reason or "model_fill", "price": price, "limit_origin": limit_origin,
                  **(cost or {"commission": 0., "other_fee": 0., "stamp_tax": 0., "slippage": 0., "total": 0.})}
        orders.append(record)
        if filled:
            fills.append(record.copy())

    for day in sessions:
        day_idx = market.day_index[day]
        # Plans are frozen before today's sales or prices are observed.
        available_cash = cash
        available_slots = max(0, card.max_positions - len(positions))
        plans = []
        daily_signals = grouped.get(day, [])
        if name.startswith("institutional"):
            daily_signals.sort(key=lambda r: (-r["inst_ratio"], r["stock_code"]))
        else:
            daily_signals.sort(key=lambda r: hashlib.sha256(f"{card.seed}|{r['event_id']}".encode()).hexdigest())
        for signal in daily_signals:
            code, eid = signal["stock_code"], signal["event_id"]
            if signal["trade_date"] >= day:
                raise ValueError("信号未早于执行日")
            if code in positions:
                order(day, eid, code, "buy", 0, 0, "already_held")
                continue
            if available_slots <= 0:
                order(day, eid, code, "buy", 0, 0, "position_limit")
                continue
            # Conservative main-board reserve; not a next-day quote or inferred fill.
            reserve_price = float(signal["close"]) * 1.1
            budget = min(equity_before * card.position_fraction, available_cash)
            qty = affordable(budget, reserve_price, day)
            prior_capacity = math.floor(signal["avg_volume_20"] * card.max_participation / 100) * 100
            qty = min(qty, prior_capacity)
            if qty <= 0:
                order(day, eid, code, "buy", 0, 0, "prior_liquidity" if prior_capacity < 100 else "cash_or_lot_size")
                continue
            reserved = qty * reserve_price + costs(qty * reserve_price, "buy", day)["total"]
            available_cash -= reserved
            available_slots -= 1
            plans.append((signal, qty, reserved))

        # Entitlements belong to positions held across the ex-date, before new buys.
        for code, pos in list(positions.items()):
            action = market.actions.get((code, day))
            if action:
                old_qty = pos.shares
                dividend = old_qty * action["dividend"] * (1 - card.dividend_tax_reserve)
                bonus_qty = math.floor(old_qty * action["allotted_ps"] + 1e-5)
                pos.dividend += dividend
                pos.shares += bonus_qty
                book(day, code, pos.event_id, "corporate_action", receivable_flow=dividend, shares=bonus_qty,
                     gross_dividend=old_qty * action["dividend"], bonus_shares=bonus_qty,
                     rights_ratio=action["rationed_ps"], policy="dividend_receivable;bonus_exdate;no_rights")
                pos.last_mark = (pos.last_mark - action["dividend"] + action["rationed_ps"] * action["rationed_px"]) / (1 + action["allotted_ps"] + action["rationed_ps"])
                if bonus_qty or action["rationed_ps"]:
                    warn(day, code, "corporate_timing_assumption", pos.event_id)
            row = market.quote(code, day)
            previous_session = market.calendar[day_idx - 1] if day_idx else None
            if row is not None and pos.last_quote_date == previous_session and not action:
                if np.isfinite(row["pre_close"]) and abs(row["pre_close"] - pos.last_mark) > .011:
                    warn(day, code, "unexplained_price_reference_change", pos.event_id)

        # Exits are retried until filled; no fictional end-of-period liquidation.
        for code, pos in list(positions.items()):
            if day_idx < pos.due_index:
                continue
            if day_idx <= market.day_index[pos.entry_date]:
                raise AssertionError("T+1 violation")
            reason, row = market.fill_block(code, day, "sell")
            if row is not None and row.get("limit_origin") == "dated_st_rule_model":
                warn(day, code, "dated_st_exit_model", pos.event_id)
            if reason == "unknown_price_limits":
                warn(day, code, "unresolved_exit_regime", pos.event_id)
            requested = pos.shares
            capacity = 0 if row is None or not np.isfinite(row["volume"]) else math.floor(row["volume"] * card.max_participation)
            qty = 0 if reason else min(requested, capacity // 100 * 100)
            if not reason and requested <= capacity:
                qty = requested  # Final odd lot created by a corporate action can exit.
            if qty <= 0:
                pos.deferred_days += 1
                order(day, pos.event_id, code, "sell", requested, 0, reason or "daily_liquidity", limit_origin=row.get("limit_origin") if row else None)
                continue
            value = qty * row["open"]
            cost = costs(value, "sell", day)
            net = value - cost["total"]
            book(day, code, pos.event_id, "sell", cash_flow=net, shares=-qty, fee=cost["total"])
            pos.proceeds += net
            pos.shares -= qty
            if qty < requested:
                pos.deferred_days += 1
            order(day, pos.event_id, code, "sell", requested, qty, "daily_liquidity" if qty < requested else "", row["open"], cost, row.get("limit_origin"))
            if pos.shares == 0:
                closed.append({"stock_code": code, "event_id": pos.event_id, "entry_date": pos.entry_date,
                               "exit_date": day, "invested": pos.invested, "net_proceeds": pos.proceeds,
                               "dividend_receivable": pos.dividend, "pnl": pos.proceeds + pos.dividend - pos.invested,
                               "return": (pos.proceeds + pos.dividend) / pos.invested - 1,
                               "deferred_days": pos.deferred_days})
                del positions[code]

        for signal, requested, reserved in plans:
            code, eid = signal["stock_code"], signal["event_id"]
            reason, row = market.fill_block(code, day, "buy")
            if reason:
                order(day, eid, code, "buy", requested, 0, reason)
                continue
            capacity = math.floor(row["volume"] * card.max_participation / 100) * 100
            qty = min(requested, capacity, affordable(min(reserved, cash), row["open"], day))
            if qty <= 0:
                order(day, eid, code, "buy", requested, 0, "daily_liquidity" if capacity < 100 else "cash_or_gap")
                continue
            value = qty * row["open"]
            cost = costs(value, "buy", day)
            invested = value + cost["total"]
            book(day, code, eid, "buy", cash_flow=-invested, shares=qty, fee=cost["total"])
            positions[code] = Position(code, eid, day, day_idx + card.holding_days, qty, qty,
                                       invested, row["open"], day)
            order(day, eid, code, "buy", requested, qty, "partial_capacity_or_cash" if qty < requested else "", row["open"], cost)
            if code in market.missing_actions:
                warn(day, code, "missing_corporate_action_file", eid)

        market_value = 0.
        unresolved_value = 0.
        stale_count = 0
        for code, pos in positions.items():
            row = market.quote(code, day)
            stale_reason = ""
            if day >= market.delisted.get(code, "9999-12-31"):
                stale_reason = "delisted_unsettled"
            elif row is None or not np.isfinite(row["close"]) or row["close"] <= 0:
                stale_reason = "halt_mark" if code in market.halts.get(day, set()) else "missing_mark"
            else:
                pos.last_mark = row["close"]
                pos.last_quote_date = day
            if stale_reason:
                stale_count += 1
                warn(day, code, stale_reason, pos.event_id)
            value = pos.shares * pos.last_mark
            market_value += value
            if stale_reason in {"delisted_unsettled", "missing_mark"}:
                unresolved_value += value
            position_rows.append({"date": day, "stock_code": code, "event_id": pos.event_id,
                                  "shares": pos.shares, "mark": pos.last_mark, "market_value": value,
                                  "entry_date": pos.entry_date, "due_session_index": pos.due_index,
                                  "deferred_days": pos.deferred_days, "stale_reason": stale_reason})
        total = cash + receivable + market_value
        if cash < -1e-6 or any(p.shares <= 0 for p in positions.values()):
            raise AssertionError("现金或持仓数量不守恒")
        equity.append({"date": day, "cash": cash, "receivable": receivable, "market_value": market_value,
                       "equity": total, "nav": total / card.initial_cash, "positions": len(positions),
                       "stale_positions": stale_count, "exposure": market_value / total if total else 0.})
        equity[-1]["unresolved_market_value"] = unresolved_value
        equity[-1]["equity_zero_unresolved"] = total - unresolved_value
        equity_before = total

    pending = signals.loc[signals.entry_date.isna() | ~signals.entry_date.isin(sessions)]
    for row in pending.itertuples(index=False):
        order(str(row.entry_date) if pd.notna(row.entry_date) else None, row.event_id,
              row.stock_code, "buy", 0, 0, "beyond_observation_window")
    eq = pd.DataFrame(equity)
    initial = card.initial_cash
    ledger_cash = initial + math.fsum(r["cash_flow"] for r in ledger)
    ledger_receivable = math.fsum(r["receivable_flow"] for r in ledger)
    share_balances = {}
    for entry in ledger:
        share_balances[entry["stock_code"]] = share_balances.get(entry["stock_code"], 0) + entry["shares_delta"]
    if any(q != (positions[c].shares if c in positions else 0) for c, q in share_balances.items()):
        raise AssertionError("股份流水与持仓不一致")
    pnl = math.fsum(t["pnl"] for t in closed) + math.fsum(
        p.proceeds + p.dividend + p.shares * p.last_mark - p.invested for p in positions.values())
    errors = {"cash_error": abs(cash - ledger_cash), "receivable_error": abs(receivable - ledger_receivable),
              "pnl_error": abs(eq.equity.iloc[-1] - initial - pnl), "share_balances_match": True}
    if max(errors["cash_error"], errors["receivable_error"], errors["pnl_error"]) > max(.001, initial * 1e-9):
        raise AssertionError(f"账户对账失败: {errors}")
    running_max = eq.nav.cummax().clip(lower=1)
    eq["drawdown"] = eq.nav / running_max - 1
    daily = eq.nav.pct_change().fillna(eq.nav.iloc[0] - 1)
    orders_df = pd.DataFrame(orders, columns=list(orders[0]) if orders else ["order_id", "date", "event_id", "stock_code", "side", "requested_shares", "filled_shares", "status", "reason", "total"])
    metrics = {"name": name, "initial_cash": initial, "final_equity": float(eq.equity.iloc[-1]),
               "total_return": float(eq.nav.iloc[-1] - 1), "max_drawdown": float(eq.drawdown.min()),
               "annualized_return": float(eq.nav.iloc[-1] ** (252 / len(eq)) - 1) if len(eq) >= 252 and eq.nav.iloc[-1] > 0 else None,
               "sharpe_zero_rf": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if len(eq) >= 252 and daily.std(ddof=1) > 0 else None,
               "average_exposure": float(eq.exposure.mean()), "closed_trades": len(closed),
               "win_rate": float(np.mean([t["pnl"] > 0 for t in closed])) if closed else None,
               "open_positions": len(positions), "receivable": receivable,
               "unresolved_market_value": float(eq.unresolved_market_value.iloc[-1]),
               "return_zero_unresolved": float(eq.equity_zero_unresolved.iloc[-1] / initial - 1),
               "buy_fills": sum(f["side"] == "buy" for f in fills),
               "cost": math.fsum(f["total"] for f in fills),
               "rejection_reasons": orders_df.loc[orders_df.side.eq("buy") & orders_df.filled_shares.eq(0), "reason"].value_counts().to_dict(),
               "deferred_exit_reasons": orders_df.loc[orders_df.side.eq("sell") & orders_df.filled_shares.eq(0), "reason"].value_counts().to_dict(),
               "warnings": pd.Series([w["kind"] for w in warnings], dtype=str).value_counts().to_dict(),
               "accounting": errors}
    def frame(rows, columns):
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)

    return {"metrics": metrics, "equity": eq, "orders": orders_df,
            "fills": frame(fills, list(orders_df.columns)),
            "ledger": frame(ledger, ["date", "stock_code", "event_id", "kind", "cash_flow", "receivable_flow", "shares_delta"]),
            "trades": frame(closed, ["stock_code", "event_id", "entry_date", "exit_date", "invested", "net_proceeds", "dividend_receivable", "pnl", "return", "deferred_days"]),
            "positions": frame(position_rows, ["date", "stock_code", "event_id", "shares", "mark", "market_value", "entry_date", "due_session_index", "deferred_days", "stale_reason"]),
            "warnings": frame(warnings, ["date", "stock_code", "kind", "event_id"])}
