"""Independent score selection and accounting checks; never fits or simulates."""
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import timestamp, timestamps
from .learning import CLOCKS, KEYS, LearningResult, _fingerprint, _keys
from .portfolio import ScorePortfolio, ScorePortfolioSpec
from .models import ModelSpec
from ..technical.artifacts import content_id
from ..technical.contracts import ResearchSpec


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _equal_frame(actual, expected, keys):
    try:
        pd.testing.assert_frame_equal(
            actual.sort_values(keys).reset_index(drop=True),
            expected.sort_values(keys).reset_index(drop=True),
            check_dtype=False, check_exact=False, rtol=0, atol=1e-12,
        )
    except (AssertionError, KeyError) as exc:
        raise ValueError("Independent expected table differs from archived output") from exc


def _schedule(calendar, cfg):
    def period(day):
        parsed = date.fromisoformat(day)
        return (parsed.year, parsed.month) if cfg["rebalance"] == "monthly" else parsed.isocalendar()[:2]
    _require(isinstance(calendar, list) and calendar == sorted(set(calendar)), "Invalid audit calendar")
    return {day: nxt for day, nxt in zip(calendar[:-1], calendar[1:])
            if cfg["start"] <= day <= cfg["end"] and period(day) != period(nxt)}


def audit_score_portfolio(result, references, calendar, portfolio):
    """Rebuild schedule/ranks/weights without calling the portfolio builder."""
    _require(isinstance(result, LearningResult) and isinstance(portfolio, ScorePortfolio),
             "Typed learning and portfolio outputs required")
    cfg = ScorePortfolioSpec.from_dict(portfolio.receipt["spec"]).to_dict()
    score, source, receipt = result.predictions, result.receipt, portfolio.receipt
    _keys(score, "Audit scores")
    _keys(references, "Audit references")
    _require(source.get("schema") == "causal-learning-receipt-v2"
             and source.get("prediction_outputs") == _fingerprint(score)
             and source.get("prediction_membership") == _fingerprint(score[KEYS + CLOCKS])
             and source.get("prediction_rows") == len(score)
             and source.get("prediction_inputs", {}).get("rows") == len(score)
             and source.get("prediction_missing", {}).get("rows") == len(score),
             "Learning score provenance differs")
    fit = source.get("fit_identity", {})
    fitted = source.get("fitted_model", {})
    excluded = fit.get("excluded_train_empty", [])
    retained = [key for key in fit["settings"]["features"] if key not in excluded]
    expected_prediction = content_id(dict(fit_id=source["fit_id"], protocol_id=source["protocol_id"],
        prediction_inputs=source["prediction_inputs"], prediction_missing=source["prediction_missing"],
        prediction_membership=source["prediction_membership"]))
    _require(fit.get("schema")=="causal-fit-identity-v1"
             and source.get("account_results") is False
             and fitted.get("schema")=="fitted-tabular-model-v1"
             and fitted.get("training_rows")==source.get("mature_training_rows")==fit["training_content"]["rows"]
             and fit["training_missing"]["rows"]==source["mature_training_rows"]
             and fitted.get("fit_attempts")==1 and fitted.get("account_results") is False
             and fitted.get("environment")==fit.get("environment")
             and fitted.get("implementation_hashes", {}).get("src/alpharesearch/models.py")
                ==fit["implementation_hashes"].get("src/alpharesearch/models.py")
             and isinstance(excluded, list) and len(set(excluded))==len(excluded)
             and all(key in fit["settings"]["features"] for key in excluded),
             "Learning mature-row/model metadata differs from bound fit")
    _require(source["fit_id"]==content_id(fit) and source["prediction_id"]==expected_prediction
             and source["excluded_train_empty"]==excluded and source["retained_features"]==retained
             and fitted["feature_names"]==retained and fitted["transformed_features"]==2*len(retained)
             and fitted["model_id"]==source["model_id"]==fit["model_id"]==ModelSpec.from_dict(fitted["spec"]).model_id
             and timestamp(source["latest_training_label_known_at"])<=timestamp(fit["settings"]["fit_cutoff"]),
             "Learning fit/prediction metadata identity differs")
    _require(receipt.get("schema") == "score-portfolio-receipt-v1"
             and receipt.get("source_learning_receipt_id") == content_id(source)
             and receipt.get("source_score_outputs") == _fingerprint(score)
             and receipt.get("reference_inputs") == _fingerprint(references)
             and receipt.get("calendar_id") == content_id(calendar)
             and receipt.get("source_fit_id") == source["fit_id"]
             and receipt.get("source_prediction_id") == source["prediction_id"],
             "Portfolio source provenance differs")
    _require(score.fit_id.eq(source["fit_id"]).all()
             and score.prediction_id.eq(source["prediction_id"]).all()
             and np.isfinite(score.score.astype(float)).all(), "Invalid source score identity")
    _require(len(references) == len(score), "Complete reference pool required")
    joined = score.merge(references, on=KEYS, how="left", indicator=True, validate="one_to_one")
    _require(joined.pop("_merge").eq("both").all(), "Reference pool keys differ")
    _require(joined.eligible.map(lambda x: isinstance(x, (bool, np.bool_))).all(), "Boolean eligibility required")
    for key in CLOCKS + ["eligibility_known_at"]:
        joined[key] = timestamps(joined[key]).astype("datetime64[ns, UTC]")
    _require(joined.decision_at.gt(timestamp(fit["settings"]["fit_cutoff"])).all()
             and joined.execution_at.gt(joined.decision_at).all()
             and joined.target_end_at.gt(joined.execution_at).all()
             and joined.eligibility_known_at.le(joined.decision_at).all()
             and joined.decision_at.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d").eq(joined.trade_date).all(),
             "Audit source decision/eligibility clocks differ")
    schedule = _schedule(calendar, cfg)
    _require(schedule and portfolio.schedule == schedule
             and receipt["schedule_id"] == content_id(schedule), "Independent account schedule differs")
    rows = joined.loc[joined.trade_date.isin(schedule)].copy()
    _require(set(rows.trade_date) == set(schedule), "Scheduled score pool missing")
    # Independently spell the documented account-board policy, not engine's regex.
    board = rows.stock_code.str.fullmatch(r"(?:60[0-9]{4}\.SH|00[0-9]{4}\.SZ)")
    _require(cfg["account_universe"] == "main_board_subset" or board.all(), "Undeclared account subset")
    eligible = board & rows.eligible
    entry = pd.to_datetime(rows.trade_date.map(schedule)+"T09:30:00+08:00", utc=True)
    _require(rows.execution_at.eq(entry).all(), "Audit next-session opening entry differs")
    full = _schedule(calendar, dict(cfg, start=calendar[0], end=calendar[-1]))
    days = sorted(full)
    ends = {day:full[days[i+1]] for i,day in enumerate(days[:-1])}
    expected_end = pd.to_datetime(rows.trade_date.map(ends)+"T09:30:00+08:00", utc=True)
    mismatch = int((eligible & ~rows.target_end_at.eq(expected_end)).sum())
    _require(receipt["target_mismatch_eligible_rows"]==mismatch
             and (cfg["horizon_policy"]!="require_next_rebalance_open" or mismatch==0),
             "Audit target/account horizon differs")
    reference = rows.loc[eligible]
    observed = timestamps(reference.reference_close_observed_at)
    known = timestamps(reference.reference_close_known_at)
    close = reference.reference_close.to_numpy(dtype=float)
    _require(np.isfinite(close).all() and (close>0).all()
             and observed.eq(pd.to_datetime(reference.trade_date+"T15:00:00+08:00", utc=True)).all()
             and known.ge(observed).all() and known.le(reference.decision_at).all(),
             "Audit raw reference availability differs")
    rows["reason"] = ["outside_native_main_board" if not b else
                      "ineligible_declared" if not e else "eligible_external_score"
                      for b, e in zip(board, rows.eligible)]
    rows = rows.sort_values(["trade_date", "score", "stock_code"], ascending=[True, False, True])
    ranks, selected, weights = [], [], []
    counts = {}
    for row in rows.itertuples():
        usable = row.reason == "eligible_external_score"
        rank = counts.get(row.trade_date, 0) + 1 if usable else np.nan
        if usable:
            counts[row.trade_date] = rank
        choose = usable and rank <= cfg["top_k"]
        ranks.append(rank); selected.append(choose)
        weights.append(cfg["allocation"] / cfg["top_k"] if choose else 0.)
    rows["rank"], rows["selected"], rows["target_weight"] = ranks, selected, weights
    rows["close"] = rows.reference_close
    rows["execution_date"] = rows.trade_date.map(schedule)
    rows["known_at_assumed"] = timestamps(rows.decision_at).map(lambda x: x.isoformat())
    columns = ["stock_code", "trade_date", "close", "score", "rank", "selected", "reason",
               "target_weight", "execution_date", "known_at_assumed"]
    expected = rows[columns].reset_index(drop=True)
    _equal_frame(portfolio.decisions, expected, ["trade_date", "stock_code"])
    _require(receipt.get("decision_outputs") == _fingerprint(portfolio.decisions)
             and receipt.get("selected_rows") == int(expected.selected.sum())
             and receipt.get("reason_counts") == expected.reason.value_counts().to_dict(),
             "Portfolio output summary differs")
    return dict(schema="independent-score-audit-v1", rows=len(expected),
                selected_rows=int(expected.selected.sum()), schedule_id=content_id(schedule),
                source_learning_receipt_id=content_id(source), numeric_model_reproduction=False)


def _close(actual, expected, message, tolerance=1e-8):
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    _require(a.shape == e.shape and np.isfinite(a).all() and np.isfinite(e).all()
             and np.allclose(a, e, rtol=0, atol=tolerance), message)
    return float(np.max(np.abs(a-e))) if a.size else 0.


def _audit_daily_state(eq, fills, positions, targets, bars, actions, status, metadata, spec, sessions):
    """Independent chronological targets, quantities, basis and snapshot marks."""
    quote_map = {(r.stock_code, r.trade_date):r for r in bars.itertuples(index=False)}
    _require(not actions.duplicated(["stock_code", "trade_date"]).any(), "Duplicate company actions")
    action_map = {(r.stock_code, r.trade_date):r for r in actions.itertuples(index=False)}
    halts = {r.trade_date:{str(c).replace(".SS", ".SH") for c in r.symbols}
             for r in status.loc[status.status_type.eq("HALT")].itertuples()}
    delisted = dict(zip(metadata.stock_code, metadata.de_listed_date.astype(str)))
    active, held, daily, realized = {}, {}, [], 0.
    grouped_targets = {d:g for d,g in targets.groupby("execution_date")}
    grouped_fills = {d:g for d,g in fills.groupby("date")}
    _require(not fills.duplicated(["date", "stock_code", "side"]).any(), "Duplicate daily fill")
    for i, day in enumerate(sessions):
        if day in grouped_targets:
            active = dict(zip(grouped_targets[day].stock_code, grouped_targets[day].target_shares))
        # A declared rebalance with no selected stocks clears the previous target.
        elif day in eq.attrs.get("scheduled_entries", set()):
            active = {}
        for code in set(active) | set(held):
            a = action_map.get((code, day))
            if a is None:
                continue
            if code in active:
                active[code] = np.floor(active[code]*(1+a.allotted_ps)+1e-5)
            if code in held:
                old = held[code]
                old["shares"] += np.floor(old["shares"]*a.allotted_ps+1e-5)
                old["mark"] = (old["mark"]-a.dividend+a.rationed_ps*a.rationed_px)/(1+a.allotted_ps+a.rationed_ps)
        today = grouped_fills.get(day, fills.iloc[:0])
        before = float(spec.execution.initial_cash if i==0 else eq.cash.iloc[i-1])
        buying_cost = float((today.loc[today.side.eq("buy"), "price"]*
                             today.loc[today.side.eq("buy"), "filled_shares"]+
                             today.loc[today.side.eq("buy"), "total"]).sum())
        _require(buying_cost <= before+.001, "Buying uses same-day sale proceeds")
        for side in ("sell", "buy"):
            for row in today.loc[today.side.eq(side)].itertuples():
                code = row.stock_code
                old_shares = held.get(code, {}).get("shares", 0)
                desired = active.get(code, 0)
                gap = max(0, desired-old_shares) if side=="buy" else max(0, old_shares-desired)
                _require(row.filled_shares <= gap and gap>0, "Fill exceeds active target gap/excess")
                _require(row.requested_shares==gap and row.filled_shares<=row.submitted_shares<=gap,
                         "Fill requested/submitted quantity differs from active target")
                q = quote_map.get((code, day))
                _require(q is not None and q.volume>0 and code not in halts.get(day, set())
                         and day<delisted.get(code, "9999-12-31"), "Fill during absent/halted/delisted quote")
                cap = np.floor(q.volume*spec.execution.max_participation)
                _require(row.filled_shares<=cap, "Fill exceeds source daily participation")
                if side=="buy":
                    _require(q.is_st==0 and q.limit_price_valid==1 and q.open<q.high_limit-1e-6,
                             "Buy violates declared native source limits/status")
                elif q.limit_price_valid==1:
                    _require(q.open>q.low_limit+1e-6, "Sell violates declared native lower limit")
                if side=="sell":
                    item = held[code]
                    removed = item["basis"]*row.filled_shares/item["shares"]
                    realized += row.price*row.filled_shares-row.total-removed
                    item["basis"] -= removed; item["shares"] -= row.filled_shares
                    if not item["shares"]:
                        del held[code]
                else:
                    item = held.setdefault(code, dict(shares=0, basis=0., mark=row.price))
                    item["shares"] += row.filled_shares
                    item["basis"] += row.price*row.filled_shares+row.total
        actual = positions.loc[positions.date.eq(day)].set_index("stock_code")
        _require(set(actual.index)==set(held), "Chronological holdings membership differs")
        unresolved, mv, basis = 0., 0., 0.
        for code, item in held.items():
            q = quote_map.get((code, day))
            stale = ""
            if day>=delisted.get(code, "9999-12-31"):
                stale = "delisted_unsettled"
            elif code in halts.get(day, set()):
                stale = "halt_mark"
            elif q is None or not np.isfinite(q.close) or q.close<=0:
                stale = "missing_mark"
            else:
                item["mark"] = q.close
            row = actual.loc[code]
            _close([row.shares, row.mark, row.market_value, row.cost_basis, row.unrealized_pnl],
                   [item["shares"], item["mark"], item["shares"]*item["mark"], item["basis"],
                    item["shares"]*item["mark"]-item["basis"]], "Snapshot-relative holdings/mark/basis differs", 1e-6)
            _require(row.stale_reason==stale and row.target_shares==active.get(code, 0),
                     "Holding stale/target policy differs")
            value = item["shares"]*item["mark"]
            mv += value; basis += item["basis"]
            if stale and stale!="halt_mark":
                unresolved += value
        _require(len(held)<=spec.strategy.top_k, "Native position limit exceeded")
        daily.append(dict(market_value=mv, basis=basis, realized=realized, unresolved=unresolved))
    return pd.DataFrame(daily)


def audit_account_tables(output, bars, spec, scenario, sessions, *, decisions, schedule, actions, status, metadata):
    """Recalculate daily balances, fill/ledger links, targets and return metrics."""
    _require(isinstance(spec, ResearchSpec) and scenario in {"zero_transaction_cost", "configured"},
             "Validated native spec and explicit audited cost scenario required")
    ex = spec.execution
    eq, fills, ledger, positions, targets = [output[k].copy() for k in
                                           ("equity", "fills", "ledger", "positions", "targets")]
    _require(eq.date.tolist() == sessions and sessions == sorted(set(sessions)) and len(sessions)>0,
             "Account sessions differ")
    for frame in (fills, ledger, positions):
        _require(frame.date.isin(sessions).all(), "Account row outside declared sessions")
    _require(not positions.duplicated(["date", "stock_code"]).any(), "Duplicate holdings")
    for frame, fields in ((ledger, ["cash_flow", "receivable_flow", "shares_delta"]),
                           (positions, ["shares", "mark", "market_value"])):
        _require(np.isfinite(frame[fields].to_numpy(dtype=float)).all(), "Nonfinite account ledger/positions")
    cash = ledger.groupby("date").cash_flow.sum().reindex(sessions, fill_value=0).cumsum()+ex.initial_cash
    receivable = ledger.groupby("date").receivable_flow.sum().reindex(sessions, fill_value=0).cumsum()
    mv = positions.groupby("date").market_value.sum().reindex(sessions, fill_value=0)
    _close(positions.market_value, positions.shares*positions.mark, "Position valuation arithmetic differs")
    errors = dict(cash=_close(eq.cash, cash, "Daily cash differs", .001),
                  receivable=_close(eq.receivable, receivable, "Daily receivable differs", .001),
                  market_value=_close(eq.market_value, mv, "Daily market value differs", .001),
                  equity=_close(eq.equity, cash+receivable+mv, "Daily equity differs", .001))
    _require(eq.cash.ge(-1e-6).all() and positions.shares.ge(0).all(), "Negative cash/shares")
    balances = ledger.groupby(["date", "stock_code"]).shares_delta.sum().unstack(fill_value=0)
    balances = balances.reindex(sessions, fill_value=0).cumsum()
    observed = positions.pivot(index="date", columns="stock_code", values="shares")
    columns = sorted(set(balances.columns) | set(observed.columns))
    balances = balances.reindex(columns=columns, fill_value=0)
    observed = observed.reindex(index=sessions, columns=columns).fillna(0)
    _close(observed, balances, "Daily share ledger differs")
    _require(fills.side.isin(["buy", "sell"]).all() and fills.filled_shares.gt(0).all()
             and (fills.filled_shares % 1 == 0).all(), "Invalid fill side/shares")
    buys = fills.side.eq("buy")
    _require((fills.loc[buys, "filled_shares"] % 100 == 0).all(), "Buy lot differs")
    _require(not fills.groupby(["date", "stock_code"]).side.nunique().gt(1).any(), "Same-day buy/sell violates T+1")
    quotes = bars.set_index(["stock_code", "trade_date"])
    _require(quotes.index.is_unique, "Duplicate audit quotes")
    keys = pd.MultiIndex.from_arrays([fills.stock_code, fills.date])
    _close(fills.price, quotes.open.reindex(keys), "Fill is not raw opening price", 1e-10)
    notion = fills.price.astype(float)*fills.filled_shares
    commission = np.maximum(ex.minimum_commission, notion*ex.commission_rate)
    other = notion*ex.other_fee_rate
    stamp = notion*np.where(fills.date.lt("2023-08-28"), .001, .0005)*~buys
    slip = notion*ex.slippage_bps/10000
    if scenario == "zero_transaction_cost":
        commission = other = stamp = slip = np.zeros(len(fills))
    for key, expected in dict(commission=commission, other_fee=other, stamp_tax=stamp,
                              slippage=slip, total=commission+other+stamp+slip).items():
        _close(fills[key], expected, "Independent fill fee differs: "+key)
    trade = fills[["date", "stock_code", "side"]].rename(columns={"side":"kind"}).copy()
    trade["cash_flow"] = np.where(buys, -notion-fills.total, notion-fills.total)
    trade["receivable_flow"] = 0.
    trade["shares_delta"] = np.where(buys, fills.filled_shares, -fills.filled_shares)
    trade["fee"] = fills.total
    actual = ledger.loc[ledger.kind.isin(["buy", "sell"]), list(trade)]
    _equal_frame(actual, trade, ["date", "stock_code", "kind"])
    _require(ledger.kind.isin(["buy", "sell", "corporate_action"]).all(), "Unknown ledger kind")
    prior = balances.shift(1).fillna(0)
    corporate = ledger.loc[ledger.kind.eq("corporate_action")].copy()
    _require(not corporate.duplicated(["date", "stock_code"]).any(), "Duplicate corporate ledger")
    action_keys = set(zip(actions.trade_date, actions.stock_code))
    expected_actions = {(day, code) for day in sessions for code in columns
                        if prior.loc[day, code] > 0 and (day, code) in action_keys}
    _require(set(zip(corporate.date, corporate.stock_code)) == expected_actions, "Corporate entitlement keys differ")
    for row in corporate.itertuples():
        source = actions.loc[actions.trade_date.eq(row.date) & actions.stock_code.eq(row.stock_code)]
        _require(len(source)==1, "Corporate source missing/duplicate")
        a = source.iloc[0]; old = prior.loc[row.date, row.stock_code]
        _close([row.cash_flow, row.receivable_flow, row.shares_delta],
               [0., old*a.dividend*(1-ex.dividend_tax_reserve), np.floor(old*a.allotted_ps+1e-5)],
               "Corporate entitlement differs")
    for row in fills.loc[~buys].itertuples():
        bonus = corporate.loc[corporate.date.eq(row.date) & corporate.stock_code.eq(row.stock_code), "shares_delta"].sum()
        _require(row.filled_shares <= prior.loc[row.date, row.stock_code]+bonus, "Sold unavailable shares")
    # Targets are rebuilt from scored decisions and previous day's equity, not copied.
    planned = []
    expected_values = (cash+receivable+mv).to_numpy()
    for day, entry in schedule.items():
        if entry not in sessions:
            continue
        idx = sessions.index(entry); equity_before = ex.initial_cash if idx==0 else expected_values[idx-1]
        for row in decisions.loc[decisions.trade_date.eq(day) & decisions.selected].itertuples():
            planned.append(dict(signal_date=day, execution_date=entry, stock_code=row.stock_code,
                                equity_used=equity_before, reference_close=row.close, target_weight=row.target_weight,
                                target_shares=np.floor(equity_before*row.target_weight/row.close/100)*100, rank=row.rank))
    expected_targets = pd.DataFrame(planned, columns=["signal_date", "execution_date", "stock_code",
                                   "equity_used", "reference_close", "target_weight", "target_shares", "rank"])
    if len(expected_targets):
        _equal_frame(targets, expected_targets, ["execution_date", "stock_code"])
    else:
        _require(targets.empty, "Unplanned targets")
    for row in fills.loc[buys].itertuples():
        active = [day for day, entry in schedule.items() if entry <= row.date]
        _require(active and row.stock_code in set(decisions.loc[
            decisions.trade_date.eq(max(active)) & decisions.selected, "stock_code"]), "Buy outside latest scored targets")
    eq.attrs["scheduled_entries"] = set(schedule.values())
    state = _audit_daily_state(eq, fills, positions, expected_targets, bars, actions, status, metadata, spec, sessions)
    _close(eq.market_value, state.market_value, "Independent snapshot mark total differs", .001)
    _close(eq.cumulative_realized_pnl, state.realized, "Reported cumulative realized PnL differs", .001)
    _close(eq.unrealized_pnl, state.market_value-state.basis, "Reported unrealized PnL differs", .001)
    _close(eq.unresolved_market_value, state.unresolved, "Reported unresolved source valuation differs", .001)
    identity = ex.initial_cash+state.realized+state.market_value-state.basis+receivable.to_numpy()
    _close(eq.accounting_error, np.abs(expected_values-identity), "Reported accounting identity error differs", .001)
    nav = expected_values/ex.initial_cash
    drawdown = nav/np.maximum(1., np.maximum.accumulate(nav))-1
    _close(eq.nav, nav, "Reported NAV differs")
    _close(eq.drawdown, drawdown, "Reported drawdown differs")
    _close(eq.exposure, mv.to_numpy()/expected_values, "Reported exposure differs")
    _close(eq.positions, observed.gt(0).sum(axis=1), "Reported position count differs")
    _require(eq.unresolved_market_value.ge(0).all() and eq.unresolved_market_value.le(eq.market_value+.001).all(),
             "Invalid unresolved valuation")
    _close(eq.equity_zero_unresolved, expected_values-eq.unresolved_market_value, "Unresolved equity differs")
    metrics = output["metrics"]
    expected_metrics = dict(final_equity=expected_values[-1], total_return=nav[-1]-1,
                            max_drawdown=drawdown.min(), total_cost=float(fills.total.sum()),
                            average_exposure=float((mv.to_numpy()/expected_values).mean()),
                            turnover_notional=float(notion.sum()), receivable=receivable.iloc[-1],
                            buy_fills=int(buys.sum()), sell_fills=int((~buys).sum()),
                            final_positions=int((observed.iloc[-1]>0).sum()),
                            unresolved_market_value=float(state.unresolved.iloc[-1]),
                            realized_pnl=float(state.realized.iloc[-1]),
                            unrealized_pnl=float(state.market_value.iloc[-1]-state.basis.iloc[-1]),
                            max_accounting_error=float(np.abs(expected_values-identity).max()),
                            cash_error=float(abs(eq.cash.iloc[-1]-ex.initial_cash-float(ledger.cash_flow.sum()))),
                            receivable_error=float(abs(eq.receivable.iloc[-1]-float(ledger.receivable_flow.sum()))))
    _require(metrics.get("scenario")==scenario, "Metric cost scenario differs")
    for key, value in expected_metrics.items():
        _close([metrics[key]], [value], "Reported metric differs: "+key)
    components = {key:float(fills[key].sum()) for key in ("commission", "other_fee", "stamp_tax", "slippage")}
    _require(set(metrics["cost_components"])==set(components), "Cost component keys differ")
    for key,value in components.items():
        _close([metrics["cost_components"][key]], [value], "Reported cost component differs: "+key)
    annual = nav[-1]**(252/len(nav))-1 if len(nav)>=252 and nav[-1]>0 else None
    if annual is None:
        _require(metrics.get("annualized_return") is None, "Short account must not annualize")
    else:
        _close([metrics["annualized_return"]], [annual], "Reported annualized return differs")
    return dict(schema="independent-account-table-audit-v1", scenario=scenario,
                sessions=len(sessions), fills=len(fills), daily_errors=errors,
                recalculated_metrics=expected_metrics, annualized_return=annual,
                limits="Independent target/quantity, snapshot-relative closing/carrying marks and PnL arithmetic; vendor truth, queue fills and economic realism of declared policies are not authenticated.")


def audit_score_account_run(folder):
    """Verify immutable archive, score decisions, native inputs and financial tables."""
    from .account import read_score_account_archive
    from ..technical.runner import verify
    folder = Path(folder)
    manifest, snapshot, sm = verify(folder)
    data = read_score_account_archive(folder, manifest, snapshot, sm)
    selection = audit_score_portfolio(data["learning"], data["references"], data["calendar"], data["portfolio"])
    result = json.loads((folder/"result.json").read_text(encoding="utf-8"))
    spec = ResearchSpec.from_dict(manifest["spec"])
    _require(result["spec"] == spec.to_dict() and result["snapshot_id"]==sm["snapshot_id"],
             "Native result source/spec differs")
    sessions = [d for d in data["calendar"] if spec.experiment.start_date<=d<=spec.experiment.end_date]
    scenarios = {}
    listed = [p["metrics"]["scenario"] for p in result["portfolios"]]
    _require(listed and len(listed)==len(set(listed)), "Empty/duplicate account scenarios")
    for portfolio in result["portfolios"]:
        scenario = portfolio["metrics"]["scenario"]
        _require(scenario in {"zero_transaction_cost", "configured"}, "Unaudited account scenario")
        tables = {k:pd.read_parquet(folder/scenario/(k+".parquet")) for k in
                  ("equity", "fills", "ledger", "positions", "targets")}
        _equal_frame(pd.DataFrame(portfolio["equity"]), tables["equity"], ["date"])
        tables["metrics"] = portfolio["metrics"]
        scenarios[scenario] = audit_account_tables(tables, data["bars"], spec, scenario, sessions,
            decisions=data["portfolio"].decisions, schedule=data["portfolio"].schedule, actions=data["actions"],
            status=data["status"], metadata=data["metadata"])
    return dict(schema="independent-score-account-run-audit-v1", run_id=folder.name,
                selection=selection, scenarios=scenarios, snapshot_id=sm["snapshot_id"],
                signal_provenance="Declared learning/feature provenance; may contain LHB and other sources.",
                numeric_model_reproduction=False, vendor_authentication=False)
