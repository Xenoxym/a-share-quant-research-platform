"""Long-only target-share execution, independent of the signal formula.

All scenarios recompute cash and quantities. No future sale proceeds size a buy.
Slippage is an explicit cash charge on a raw opening-price fill, counted once.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .contracts import SCENARIOS


COST_KEYS = ("commission", "other_fee", "stamp_tax", "slippage")


def costs(value, side, day, execution, scenario):
    if scenario not in SCENARIOS:
        raise ValueError("未知费用情景")
    if value <= 0 or scenario == "zero_transaction_cost":
        return {key: 0. for key in (*COST_KEYS, "total")}
    c = {"commission": max(execution.minimum_commission, value*execution.commission_rate),
         "other_fee": value*execution.other_fee_rate,
         "stamp_tax": value*(.001 if day < "2023-08-28" else .0005) if side == "sell" else 0.,
         "slippage": value*execution.slippage_bps/10000 if scenario == "configured" else 0.}
    return {**c, "total": math.fsum(c.values())}


def simulate(decisions, schedule, market, sessions, spec, scenario):
    e, s = spec.execution, spec.strategy
    if not sessions:
        raise ValueError("交易日为空")
    selected = {d: rows.loc[rows.selected].sort_values("rank").to_dict("records")
                for d, rows in decisions.groupby("trade_date")}
    execute_on = {nxt: d for d, nxt in schedule.items()}
    cash, receivable, equity_before, realized = float(e.initial_cash), 0., float(e.initial_cash), 0.
    positions, targets, target_ranks = {}, {}, {}
    equity, orders, fills, ledger, holdings, warnings, target_rows = [], [], [], [], [], [], []

    def fee(value, side, day):
        return costs(value, side, day, e, scenario)

    def affordable(budget, price, day):
        n = max(0, math.floor(budget / (100*price))*100)
        while n and n*price + fee(n*price, "buy", day)["total"] > budget+1e-8:
            n -= 100
        return n

    def book(day, code, kind, cash_flow=0., receivable_flow=0., shares_delta=0, **extra):
        nonlocal cash, receivable
        cash += cash_flow
        receivable += receivable_flow
        ledger.append(dict(date=day, stock_code=code, kind=kind, cash_flow=cash_flow,
                           receivable_flow=receivable_flow, shares_delta=shares_delta, **extra))

    def record(day, code, side, requested, planned, qty, reason, price=None, limit_origin=None):
        c = fee(qty*price, side, day) if qty else {k: 0. for k in (*COST_KEYS, "total")}
        row = dict(date=day, stock_code=code, side=side, requested_shares=requested,
                   submitted_shares=planned, filled_shares=qty, reason=reason, price=price,
                   limit_origin=limit_origin, **c)
        orders.append(row)
        if qty:
            fills.append(row.copy())
        return c

    def warn(day, code, kind):
        warnings.append(dict(date=day, stock_code=code, kind=kind))

    for day in sessions:
        idx = market.day_index[day]
        prev = market.calendar[idx-1] if idx else None
        if day in execute_on:
            signal_day = execute_on[day]
            if signal_day >= day:
                raise AssertionError("信号时点未早于执行")
            candidates = selected.get(signal_day, [])
            targets = {r["stock_code"]: math.floor(equity_before*r["target_weight"]/r["close"]/100)*100 for r in candidates}
            target_ranks = {r["stock_code"]: r["rank"] for r in candidates}
            for r in candidates:
                target_rows.append(dict(signal_date=signal_day, execution_date=day, stock_code=r["stock_code"],
                    equity_used=equity_before, reference_close=r["close"], target_weight=r["target_weight"],
                    target_shares=targets[r["stock_code"]], rank=r["rank"]))
        # Ex-date entitlements only apply to positions held overnight.
        for code in sorted(set(positions) | set(targets)):
            action = market.actions.get((code, day))
            if action and code in targets:
                targets[code] = math.floor(targets[code]*(1+action["allotted_ps"])+1e-5)
            if code not in positions:
                continue
            pos = positions[code]
            if action:
                gross = pos["shares"]*action["dividend"]
                bonus = math.floor(pos["shares"]*action["allotted_ps"]+1e-5)
                pos["shares"] += bonus
                book(day, code, "corporate_action", receivable_flow=gross*(1-e.dividend_tax_reserve),
                     shares_delta=bonus, gross_dividend=gross, dividend_tax_reserve=gross*e.dividend_tax_reserve)
                pos["mark"] = (pos["mark"]-action["dividend"]+action["rationed_ps"]*action["rationed_px"])/(1+action["allotted_ps"]+action["rationed_ps"])
                if bonus or action["rationed_ps"]:
                    warn(day, code, "corporate_timing_or_rights_assumption")
            quote = market.quote(code, day)
            if not action and quote and pos["quote_date"] == prev and np.isfinite(quote["pre_close"]) and abs(quote["pre_close"]-pos["mark"]) > .011:
                warn(day, code, "unexplained_price_reference_change")
        # Prepare purchases from cash already available before today's sales.
        available = cash
        plans = []
        for code in sorted(targets, key=lambda c: (target_ranks.get(c, 1e9), c)):
            requested = max(0, targets[code]-positions.get(code, {}).get("shares", 0))
            if requested < 100:
                continue
            q = market.quote(code, prev) if prev else None
            if not q or not np.isfinite(q["avg_volume_20"]) or q["close"] <= 0:
                record(day, code, "buy", requested, 0, 0, "missing_prior_quote_or_volume")
                continue
            prior_price = q["close"]
            action = market.actions.get((code, day))
            if action:
                prior_price = (prior_price-action["dividend"]+action["rationed_ps"]*action["rationed_px"])/(1+action["allotted_ps"]+action["rationed_ps"])
            if prior_price <= 0:
                record(day, code, "buy", requested, 0, 0, "invalid_action_reference")
                continue
            reserve_price = prior_price*1.1
            planned = min(requested//100*100, math.floor(q["avg_volume_20"]*e.max_participation/100)*100,
                          affordable(available, reserve_price, day))
            if planned <= 0:
                record(day, code, "buy", requested, 0, 0, "prior_cash_or_capacity")
                continue
            reserved = planned*reserve_price + fee(planned*reserve_price, "buy", day)["total"]
            available -= reserved
            plans.append((code, requested, planned, reserved))
        # Exits/reductions precede purchases; T+1 shares are all prior-day holdings.
        for code, pos in list(positions.items()):
            requested = max(0, pos["shares"]-targets.get(code, 0))
            if not requested:
                continue
            if pos["last_buy"] >= day:
                raise AssertionError("T+1 violation")
            reason, quote = market.fill_block(code, day, "sell")
            if quote and quote.get("limit_origin") == "dated_st_rule_model":
                warn(day, code, "dated_st_exit_model")
            cap = 0 if not quote or not np.isfinite(quote["volume"]) else math.floor(quote["volume"]*e.max_participation)
            qty = 0 if reason else min(requested, cap)//100*100
            if not reason and requested == pos["shares"] and requested <= cap:
                qty = requested
            if qty and qty*quote["open"]-fee(qty*quote["open"], "sell", day)["total"]+cash < -1e-8:
                qty, reason = 0, "sale_fee_cash_shortfall"
            if not qty:
                record(day, code, "sell", requested, requested, 0, reason or "volume_or_odd_lot")
                continue
            c = record(day, code, "sell", requested, requested, qty, "filled" if qty == requested else "partial_capacity",
                       quote["open"], quote.get("limit_origin"))
            removed_basis = pos["basis"]*qty/pos["shares"]
            net = qty*quote["open"]-c["total"]
            pnl = net-removed_basis
            realized += pnl
            book(day, code, "sell", cash_flow=net, shares_delta=-qty, fee=c["total"], realized_pnl=pnl)
            pos["basis"] -= removed_basis
            pos["shares"] -= qty
            if not pos["shares"]:
                del positions[code]
        for code, requested, planned, reserved in plans:
            position_limit = 2*s.top_k if s.family == "allocation" else s.top_k
            if code not in positions and len(positions) >= position_limit:
                record(day, code, "buy", requested, planned, 0, "position_limit_with_pending_exits")
                continue
            reason, quote = market.fill_block(code, day, "buy")
            if reason:
                record(day, code, "buy", requested, planned, 0, reason)
                continue
            cap = math.floor(quote["volume"]*e.max_participation/100)*100
            qty = min(planned, cap, affordable(min(reserved, cash), quote["open"], day))
            if not qty:
                record(day, code, "buy", requested, planned, 0, "opening_cash_or_volume")
                continue
            c = record(day, code, "buy", requested, planned, qty, "filled" if qty == requested else "partial_capacity_or_cash",
                       quote["open"], quote.get("limit_origin"))
            invested = qty*quote["open"]+c["total"]
            book(day, code, "buy", cash_flow=-invested, shares_delta=qty, fee=c["total"])
            pos = positions.setdefault(code, dict(shares=0, basis=0., mark=quote["open"], quote_date=day, last_buy=day))
            pos["shares"] += qty
            pos["basis"] += invested
            pos["last_buy"] = day
            if code in market.missing_actions:
                warn(day, code, "missing_action_source")
        mv, basis, unresolved = 0., 0., 0.
        for code, pos in positions.items():
            quote = market.quote(code, day)
            stale = ""
            if day >= market.delisted.get(code, "9999-12-31"):
                stale = "delisted_unsettled"
            elif code in market.halts.get(day, set()):
                # Vendor halt bars can retain the pre-action price. Preserve the
                # entitlement-adjusted carrying mark until a tradable quote returns.
                stale = "halt_mark"
            elif not quote or not np.isfinite(quote["close"]) or quote["close"] <= 0:
                stale = "missing_mark"
            else:
                pos["mark"], pos["quote_date"] = quote["close"], day
            value = pos["mark"]*pos["shares"]
            mv += value
            basis += pos["basis"]
            if stale:
                if stale != "halt_mark":
                    unresolved += value
                warn(day, code, stale)
            holdings.append(dict(date=day, stock_code=code, shares=pos["shares"], target_shares=targets.get(code, 0),
                mark=pos["mark"], market_value=value, cost_basis=pos["basis"], unrealized_pnl=value-pos["basis"], stale_reason=stale))
        total = cash+receivable+mv
        identity = e.initial_cash+realized+(mv-basis)+receivable
        if cash < -1e-6 or abs(total-identity) > max(.001, total*1e-9):
            raise AssertionError("账户损益或现金不守恒")
        equity.append(dict(date=day, cash=cash, receivable=receivable, market_value=mv, equity=total,
            nav=total/e.initial_cash, positions=len(positions), exposure=mv/total if total else 0.,
            cumulative_realized_pnl=realized, unrealized_pnl=mv-basis, unresolved_market_value=unresolved,
            equity_zero_unresolved=total-unresolved, accounting_error=abs(total-identity)))
        equity_before = total
    eq = pd.DataFrame(equity)
    eq["drawdown"] = eq.nav/eq.nav.cummax().clip(lower=1)-1
    columns = ["date", "stock_code", "side", "requested_shares", "submitted_shares", "filled_shares", "reason", "price", "limit_origin", *COST_KEYS, "total"]
    orders_df, fills_df = pd.DataFrame(orders, columns=columns), pd.DataFrame(fills, columns=columns)
    ledger_df = pd.DataFrame(ledger, columns=None if ledger else ["date", "stock_code", "kind", "cash_flow", "receivable_flow", "shares_delta", "fee"])
    cash_error = abs(cash-e.initial_cash-math.fsum(ledger_df.cash_flow))
    receipt_error = abs(receivable-math.fsum(ledger_df.receivable_flow))
    shares = ledger_df.groupby("stock_code").shares_delta.sum().to_dict()
    if any(v != positions.get(k, {}).get("shares", 0) for k, v in shares.items()) or max(cash_error, receipt_error) > .001:
        raise AssertionError("流水和期末余额不一致")
    components = {k: float(fills_df[k].sum()) for k in COST_KEYS}
    metrics = dict(scenario=scenario, label=SCENARIOS[scenario], final_equity=float(eq.equity.iloc[-1]),
        total_return=float(eq.nav.iloc[-1]-1), max_drawdown=float(eq.drawdown.min()),
        annualized_return=float(eq.nav.iloc[-1]**(252/len(eq))-1) if len(eq) >= 252 and eq.nav.iloc[-1] > 0 else None,
        total_cost=sum(components.values()), cost_components=components, buy_fills=int(fills_df.side.eq("buy").sum()),
        sell_fills=int(fills_df.side.eq("sell").sum()), turnover_notional=float((fills_df.filled_shares*fills_df.price).sum()),
        average_exposure=float(eq.exposure.mean()), receivable=receivable, realized_pnl=realized,
        unrealized_pnl=float(eq.unrealized_pnl.iloc[-1]), unresolved_market_value=float(eq.unresolved_market_value.iloc[-1]),
        final_positions=len(positions), cash_error=cash_error, receivable_error=receipt_error,
        max_accounting_error=float(eq.accounting_error.max()),
        rejected=orders_df.loc[orders_df.filled_shares.eq(0), "reason"].value_counts().to_dict(),
        warnings=pd.Series([w["kind"] for w in warnings], dtype=str).value_counts().to_dict())
    return {"metrics": metrics, "equity": eq, "orders": orders_df, "fills": fills_df, "ledger": ledger_df,
            "positions": pd.DataFrame(holdings, columns=None if holdings else ["date", "stock_code", "shares", "mark", "market_value"]),
            "warnings": pd.DataFrame(warnings, columns=["date", "stock_code", "kind"]),
            "targets": pd.DataFrame(target_rows, columns=None if target_rows else ["signal_date", "execution_date", "stock_code", "target_shares"])}
