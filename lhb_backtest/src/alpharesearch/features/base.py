"""Column-oriented feature contracts; no per-row Python value objects required."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..contracts import MissingReason, Unit, VintagePolicy, timestamps, required_id


@dataclass
class FeatureBlock:
    values: pd.DataFrame
    missing: pd.DataFrame
    units: dict
    metadata: dict

    def validate(self):
        keys = self.metadata['key_columns']
        required_id(self.metadata['source_id'], 'source_id')
        required_id(self.metadata['version_id'], 'version_id')
        VintagePolicy(self.metadata['vintage'])
        if len(self.values) != len(self.missing) or (self.values.columns.duplicated().any() or self.missing.columns.duplicated().any()):
            raise ValueError('Feature/reason shape mismatch')
        if self.values[keys].isna().any().any() or self.values.duplicated(keys).any():
            raise ValueError('Feature keys must be non-null and unique')
        if not self.values[keys].reset_index(drop=True).equals(self.missing[keys].reset_index(drop=True)):
            raise ValueError('Reason keys do not match feature rows')
        if set(self.missing) != set(keys) | set(self.units):
            raise ValueError('Every feature needs exactly one missing reason column')
        if set(self.values) != set(keys) | set(self.units) | {'observed_end', 'known_at'}:
            raise ValueError('Undeclared columns or labels cannot enter a feature block')
        end = timestamps(self.values.observed_end)
        known = timestamps(self.values.known_at)
        if known.lt(end).any():
            raise ValueError('Feature cannot be available before observation ends')
        for name, unit in self.units.items():
            unit = Unit(unit)
            value = pd.to_numeric(self.values[name], errors='raise').to_numpy(dtype=float, na_value=np.nan)
            reasons = self.missing[name]
            if not reasons.isin([m.value for m in MissingReason]).all():
                raise ValueError('Unknown missing reason: ' + name)
            present = reasons.eq(MissingReason.PRESENT.value).to_numpy()
            if not np.array_equal(np.isfinite(value), present) or np.isinf(value).any():
                raise ValueError('Values and missingness disagree: ' + name)
            if unit == Unit.BINARY and not np.isin(value[present], [0., 1.]).all():
                raise ValueError('Binary values must be zero or one')
            if unit == Unit.COUNT and ((value[present] < 0).any() or (value[present] != np.floor(value[present])).any()):
                raise ValueError('Count values must be non-negative integers')
            if unit == Unit.PRICE and (value[present] <= 0).any():
                raise ValueError('Price values must be positive')
        return self
