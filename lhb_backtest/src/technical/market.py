"""Daily quotes and side-specific limit assumptions; no strategy selection."""
from __future__ import annotations
import numpy as np
from .signals import BAR_COLS, KEYS, check_keys

class Market:
    def __init__(self, bars, calendar, actions=None, status=None, metadata=None, missing_actions=()):
        check_keys(bars, "行情")
        self.columns = [c for c in BAR_COLS if c not in KEYS] + ["avg_volume_20"]
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
        if date >= self.delisted.get(code, "9999-12-31"):
            return "delisted_unsettled", row
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
        if side == "buy" and row["open"] >= row["high_limit"] - 1e-6:
            return "open_at_upper_limit", row
        if side == "sell" and row["open"] <= row["low_limit"] + 1e-6:
            return "open_at_lower_limit", row
        return "", row

