"""Call-local reuse of pairwise ranks with EXACT identical finite membership.

No cache survives a call or changes score selection. Fixed cache cell budget bounds
rank storage; differing missing sets get separate ranks or uncached computation.
"""
from collections import OrderedDict
from itertools import combinations, islice
import numpy as np
import pandas as pd


def redundancy_pairs(values, names, minimum, max_pairs, *, max_cached_cells=16_000_000):
    if (type(minimum) is not int or minimum < 3 or type(max_pairs) is not int
            or max_pairs < 0 or type(max_cached_cells) is not int or max_cached_cells < 0):
        raise ValueError("Finite pair and cache cell bounds required")
    if len(names) != len(set(names)) or not set(names) <= set(values):
        raise ValueError("Unique declared numeric feature names required")
    for name in names:
        if (not pd.api.types.is_numeric_dtype(values[name])
                or pd.api.types.is_complex_dtype(values[name])):
            raise ValueError("Numeric noncomplex features required")
    if max_pairs == 0 or len(names) < 2:
        return []
    columns = {name: values[name].copy(deep=True) for name in names}
    finite = {name: column.notna().to_numpy(dtype=bool) for name, column in columns.items()}
    if any(np.isinf(x.to_numpy(dtype=float, na_value=np.nan)).any() for x in columns.values()):
        raise ValueError("Infinite values are not valid feature values")
    cache = OrderedDict(); used = 0
    def ranked(name, common, mask_identity):
        nonlocal used
        key = (name, mask_identity)
        if key in cache:
            item = cache.pop(key); cache[key] = item; return item
        ranks = columns[name].iloc[common].rank().to_numpy()
        item = (ranks, bool(ranks.min() == ranks.max()))
        if len(ranks) <= max_cached_cells:
            while cache and used + len(ranks) > max_cached_cells:
                _, evicted = cache.popitem(last=False); used -= len(evicted[0])
            cache[key] = item; used += len(ranks)
        return item
    results = []
    for left, right in islice(combinations(names, 2), max_pairs):
        common = finite[left] & finite[right]
        count = int(common.sum())
        if count < minimum:
            correlation, reason = None, "insufficient_pairs"
        else:
            mask_identity = np.packbits(common).tobytes()
            lr, lc = ranked(left, common, mask_identity)
            rr, rc = ranked(right, common, mask_identity)
            if lc or rc:
                correlation, reason = None, "constant_score_or_target"
            else:
                correlation = float(pd.Series(lr).corr(pd.Series(rr)))
                reason = "present"
        results.append(dict(left=left, right=right, common_rows=count,
                            pooled_rank_correlation=correlation, reason=reason))
    return results
