"""Past-session windows with availability masks; no fit, fill or selection.

This initial adapter accepts a bounded in-memory partition. It does not claim to
stream an arbitrary source file, certify vendor timestamps, or build labels.
"""
from dataclasses import dataclass
from datetime import date
from itertools import islice
import re

import numpy as np
import pandas as pd

from .contracts import MissingReason, VintagePolicy, timestamps
from .features.base import FeatureBlock

REASON_NAMES = tuple(reason.value for reason in MissingReason)
REASON_CODES = {name: i for i, name in enumerate(REASON_NAMES)}
SAMPLE_KEYS = ["trade_date", "stock_code", "decision_at"]


def _dates(values):
    output = []
    for value in values:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("Canonical calendar dates required")
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError("Canonical calendar dates required")
        output.append(value)
    return output


@dataclass(frozen=True)
class WindowSpec:
    sessions: int = 20
    max_batch_samples: int = 512
    max_batch_cells: int = 2_000_000
    max_source_rows: int = 500_000
    max_source_cells: int = 32_000_000

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError("Positive finite integer limits required")
        bounds = {"sessions": 2520, "max_batch_samples": 100000,
                  "max_batch_cells": 100000000, "max_source_rows": 10000000,
                  "max_source_cells": 100000000}
        if any(getattr(self, name) > upper for name, upper in bounds.items()):
            raise ValueError("Window limit exceeds finite implementation bound")


@dataclass(frozen=True)
class WindowBatch:
    samples: pd.DataFrame
    dates: np.ndarray
    features: tuple
    values: np.ndarray
    observed: np.ndarray
    reason_codes: np.ndarray
    source_id: str
    version_id: str
    vintage: str
    reason_names: tuple = REASON_NAMES


class CausalWindowDataset:
    """Own source arrays once; materialize only explicitly requested windows.

    Calendar slots remain calendar slots. An unavailable observation is indistinguishable from
    absent source coverage; its future existence cannot enter a past mask. No target
    or model preprocessing is accepted by this adapter.
    """

    def __init__(self, block, samples, calendar, features, *, spec=None,
                 allow_weak_vintage=False):
        self.spec = WindowSpec() if spec is None else spec
        if not isinstance(self.spec, WindowSpec) or not isinstance(block, FeatureBlock):
            raise ValueError("Typed window spec and feature block required")
        if type(allow_weak_vintage) is not bool:
            raise ValueError("Explicit Boolean vintage policy required")
        if len(block.values) > self.spec.max_source_rows or len(block.values)*len(block.units) > self.spec.max_source_cells:
            raise ValueError("Source partition exceeds declared budget")
        if len(block.values) == 0:
            raise ValueError("Nonempty source partition required")
        block.validate()
        if set(block.metadata["key_columns"]) != {"trade_date", "stock_code"}:
            raise ValueError("Daily stock/date source keys required")
        if block.metadata.get("frequency") != "daily":
            raise ValueError("Declared daily source frequency required")
        vintage = VintagePolicy(block.metadata["vintage"])
        if vintage not in {VintagePolicy.HISTORICAL, VintagePolicy.MARKET} and not allow_weak_vintage:
            raise ValueError("Weak source vintage needs explicit allowance")
        if not isinstance(features, (list, tuple)) or not features or len(set(features)) != len(features) or any(type(f) is not str or f not in block.units for f in features):
            raise ValueError("Unique declared feature axis required")
        days = _dates(calendar)
        if not days or days != sorted(set(days)):
            raise ValueError("Ordered unique trading calendar required")
        self._calendar = tuple(days)
        self._calendar_index = {d: i for i, d in enumerate(days)}
        source_dates = _dates(block.values.trade_date)
        if any(d not in self._calendar_index for d in source_dates):
            raise ValueError("Source day is outside declared calendar")
        if not isinstance(samples, pd.DataFrame) or samples.empty or set(samples) != set(SAMPLE_KEYS) or samples.columns.duplicated().any():
            raise ValueError("Samples require only explicit stock/date/decision keys")
        if samples[SAMPLE_KEYS].isna().any().any() or samples.duplicated(["trade_date", "stock_code"]).any():
            raise ValueError("Unique non-null sample keys required")
        sample_dates = _dates(samples.trade_date)
        if any(d not in self._calendar_index for d in sample_dates):
            raise ValueError("Sample day is outside declared calendar")
        for stock in (block.values.stock_code, samples.stock_code):
            if not stock.map(lambda x: isinstance(x, str) and re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", x) is not None).all():
                raise ValueError("Exchange-qualified stock codes required")
        clocks = timestamps(samples.decision_at).astype("datetime64[ns, UTC]")
        if not clocks.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d").eq(samples.trade_date).all():
            raise ValueError("Sample trading date differs from decision clock")
        ends = timestamps(block.values.observed_end).astype("datetime64[ns, UTC]")
        if not ends.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d").eq(block.values.trade_date).all():
            raise ValueError("Source trading date differs from observation clock")
        self.features = tuple(features)
        self._samples = samples[SAMPLE_KEYS].reset_index(drop=True).copy()
        self._samples["decision_at"] = clocks.reset_index(drop=True).copy(deep=True)
        self._cutoff = clocks.astype("int64").to_numpy(copy=True)
        self._known = timestamps(block.values.known_at).astype("datetime64[ns, UTC]").astype("int64").to_numpy(copy=True)
        self._end = ends.astype("int64").to_numpy(copy=True)
        # Match FeatureBlock numeric normalization, including object pd.NA and
        # nullable numeric dtypes. Keep only one temporary feature column.
        self._values = np.empty((len(block.values), len(features)), dtype=float)
        for j, feature in enumerate(features):
            numeric = pd.to_numeric(block.values[feature], errors="raise")
            self._values[:, j] = numeric.to_numpy(dtype=float, na_value=np.nan, copy=True)
        self._reasons = np.column_stack([block.missing[f].map(REASON_CODES).to_numpy(dtype=np.uint8) for f in features])
        self._index = pd.MultiIndex.from_frame(block.values[["stock_code", "trade_date"]]).copy(deep=True)
        for array in (self._cutoff, self._known, self._end, self._values, self._reasons):
            array.flags.writeable = False
        self.source_id, self.version_id, self.vintage = block.metadata["source_id"], block.metadata["version_id"], vintage.value

    def __len__(self):
        return len(self._samples)

    def batch(self, indices):
        t, d = self.spec.sessions, len(self.features)
        allowed = min(self.spec.max_batch_samples, self.spec.max_batch_cells // (t*d))
        # A huge/infinite iterator cannot be materialized before admission.
        ids = list(islice(iter(indices), allowed+1))
        if len(ids) > allowed:
            raise ValueError("Window batch exceeds declared budget")
        if not ids or any(type(i) is not int or i < 0 or i >= len(self) for i in ids):
            raise ValueError("Explicit valid integer sample indices required")
        n = len(ids)
        if n > self.spec.max_batch_samples or n*t*d > self.spec.max_batch_cells:
            raise ValueError("Window batch exceeds declared budget")
        samples = self._samples.iloc[ids].reset_index(drop=True).copy()
        values = np.full((n, t, d), np.nan, dtype=float)
        reasons = np.full((n, t, d), REASON_CODES[MissingReason.HISTORY.value], dtype=np.uint8)
        dates = np.full((n, t), None, dtype=object)
        for i, row in enumerate(samples.itertuples(index=False)):
            stop = self._calendar_index[row.trade_date]+1
            days = self._calendar[max(0, stop-t):stop]
            left = t-len(days)
            dates[i, left:] = days
            reasons[i, left:] = REASON_CODES[MissingReason.UNCOVERED.value]
            query = pd.MultiIndex.from_arrays([[row.stock_code]*len(days), days], names=self._index.names)
            positions = self._index.get_indexer(query)
            exists = positions >= 0
            slots = np.flatnonzero(exists)+left
            positions = positions[exists]
            cutoff = self._cutoff[ids[i]]
            known = (self._known[positions] <= cutoff) & (self._end[positions] <= cutoff)
            # Do not reveal that an as-yet unknown record will later exist.
            # Unavailable records retain the same code as absent source coverage.
            slots, positions = slots[known], positions[known]
            values[i, slots] = self._values[positions]
            reasons[i, slots] = self._reasons[positions]
        observed = reasons == REASON_CODES[MissingReason.PRESENT.value]
        return WindowBatch(samples, dates, self.features, values, observed, reasons,
                           self.source_id, self.version_id, self.vintage)
