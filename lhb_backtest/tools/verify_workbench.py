"""Independent checks against frozen inputs and accounting outputs (no network)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.workbench.artifacts import digest, write_json
from src.workbench.report import PORTFOLIOS
from src.workbench.runner import verify_run


def verify(folder, comparison=None):
    folder = Path(folder).resolve()
    manifest, snapshot, _ = verify_run(folder)
    card = manifest["card"]
    initial = card["initial_cash"]
    events = pd.read_parquet(folder / "events.parquet").set_index("event_id")
    pairs = pd.read_parquet(folder / "pairs.parquet")
    calendar = json.loads((snapshot / "calendar.json").read_text(encoding="utf-8"))
    day_index = {d: i for i, d in enumerate(calendar)}
    results = {}
    for name in sorted(PORTFOLIOS):
        target = folder / name
        eq = pd.read_parquet(target / "equity.parquet").set_index("date")
        ledger = pd.read_parquet(target / "ledger.parquet")
        orders = pd.read_parquet(target / "orders.parquet")
        positions = pd.read_parquet(target / "positions.parquet")
        trades = pd.read_parquet(target / "trades.parquet")
        cash = ledger.groupby("date").cash_flow.sum().reindex(eq.index, fill_value=0).cumsum() + initial
        receivable = ledger.groupby("date").receivable_flow.sum().reindex(eq.index, fill_value=0).cumsum()
        pos_value = positions.groupby("date").market_value.sum().reindex(eq.index, fill_value=0)
        assert np.allclose(cash, eq.cash, rtol=0, atol=1e-5), name
        assert np.allclose(receivable, eq.receivable, rtol=0, atol=1e-5), name
        assert np.allclose(pos_value, eq.market_value, rtol=0, atol=1e-5), name
        assert np.allclose(cash + receivable + pos_value, eq.equity, rtol=0, atol=1e-5), name
        assert eq.cash.ge(-1e-6).all() and eq.positions.le(card["max_positions"]).all()
        assert np.allclose(positions.shares * positions.mark, positions.market_value)
        if not ledger.empty:
            share_delta = ledger.pivot_table(index="date", columns="stock_code", values="shares_delta", aggfunc="sum", fill_value=0)
            share_balance = share_delta.reindex(eq.index, fill_value=0).cumsum()
            daily_positions = positions.pivot_table(index="date", columns="stock_code", values="shares", aggfunc="sum", fill_value=0).reindex(index=eq.index, columns=share_balance.columns, fill_value=0)
            assert np.array_equal(share_balance.to_numpy(), daily_positions.to_numpy()), name
        filled = orders.loc[orders.filled_shares.gt(0)]
        buys = filled.loc[filled.side.eq("buy")]
        assert buys.filled_shares.mod(100).eq(0).all()
        assert buys.requested_shares.mod(100).eq(0).all()
        assert buys.date.eq(buys.event_id.map(events.entry_date)).all()
        assert buys.date.gt(buys.event_id.map(events.trade_date)).all()
        assert buys.filled_shares.le(buys.event_id.map(events.avg_volume_20) * card["max_participation"]).all()
        if not trades.empty:
            assert (trades.exit_date.map(day_index) - trades.entry_date.map(day_index)).ge(card["holding_days"]).all()
        assert filled.filled_shares.le(filled.requested_shares).all()
        # Match every modeled fill to an actual opening price from the frozen quote table.
        bars = pd.read_parquet(snapshot / "bars.parquet", columns=["trade_date", "stock_code", "open", "volume"])
        matched = filled.merge(bars, left_on=["stock_code", "date"], right_on=["stock_code", "trade_date"], how="left", validate="many_to_one")
        assert np.allclose(matched.price, matched.open)
        assert matched.filled_shares.le(np.floor(matched.volume * card["max_participation"])).all()
        value = filled.price * filled.filled_shares
        expected_stamp = np.where(filled.side.eq("sell"), value * np.where(filled.date.lt("2023-08-28"), .001, .0005), 0)
        assert np.allclose(filled.stamp_tax, expected_stamp)
        assert np.allclose(filled.total, filled[["commission", "other_fee", "stamp_tax", "slippage"]].sum(axis=1))
        results[name] = {"days": len(eq), "fills": len(filled), "ledger_entries": len(ledger),
                         "max_cash_error": float(np.abs(cash - eq.cash).max()),
                         "max_equity_error": float(np.abs(cash + receivable + pos_value - eq.equity).max()),
                         "checks": "daily cash/receivable/shares/equity, lots, T+1/holding dates, prior capacity, source openings, fees"}
    assert pairs.treated_event_id.is_unique and pairs.control_event_id.is_unique
    assert pairs.trade_date.eq(pairs.treated_event_id.map(events.trade_date)).all()
    assert pairs.trade_date.eq(pairs.control_event_id.map(events.trade_date)).all()
    assert pairs.treated_event_id.map(events.signal).all()
    assert not pairs.control_event_id.map(events.signal).any()
    assert pairs.distance.le(card["match_caliper"]).all()
    replay = None
    if comparison:
        previous, _, _ = verify_run(comparison)
        assert previous["snapshot_id"] == manifest["snapshot_id"]
        assert previous["code_hash"] == manifest["code_hash"]
        tables = [name for name in manifest["artifacts"] if name.endswith(".parquet")]
        differences = [name for name in tables if digest(folder / name) != digest(Path(comparison) / name)]
        assert not differences, differences
        replay = {"compared_with": str(comparison), "identical_tables": len(tables), "differences": differences}
    return {"status": "pass", "run_id": folder.name, "portfolios": results, "pairs": len(pairs), "replay": replay}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir, args.compare)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
