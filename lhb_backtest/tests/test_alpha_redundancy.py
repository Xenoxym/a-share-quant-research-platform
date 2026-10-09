"""Counterexamples and operational bounds for exact pair-common rank reuse."""
from itertools import combinations, islice
import numpy as np
import pandas as pd
import pytest
from src.alpharesearch.redundancy import redundancy_pairs
from src.alpharesearch.screening import _rank_ic


def reference(frame, minimum=3, maximum=1000):
    out = []
    for left, right in islice(combinations(list(frame), 2), maximum):
        corr, n, why = _rank_ic(frame[left], frame[right], minimum)
        out.append(dict(left=left, right=right, common_rows=n,
                        pooled_rank_correlation=corr, reason=why))
    return out


def test_missing_pair_ranks_are_not_whole_column_ranks():
    f = pd.DataFrame(dict(x=[1., 2., 100., 4.], y=[1., np.nan, 0., 2.],
                          z=[np.nan, 10., 20., 0.]))
    result = redundancy_pairs(f, list(f), 3, 3)
    assert result == reference(f)
    assert result[0]["pooled_rank_correlation"] == -.5
    wrong = f.x.rank().corr(f.y.rank())
    assert wrong != result[0]["pooled_rank_correlation"]


@pytest.mark.parametrize("cell_cap", [0, 1, 8, 1000])
def test_eviction_and_uncached_fallback_match_pairwise_reference(cell_cap):
    f = pd.DataFrame(dict(a=[1., 1., 3., 4., 5., 6.],
       b=[1., np.nan, 4., 3., 6., 5.], c=[6., 5., np.nan, 3., 2., 1.],
       d=[1., 1., 1., 1., 1., 1.], e=[np.nan]*6))
    assert redundancy_pairs(f, list(f), 3, 10, max_cached_cells=cell_cap) == reference(f)


def test_complete32columns_use32_rank_calls_and_exact496results(monkeypatch):
    rng = np.random.default_rng(71)
    f = pd.DataFrame(rng.integers(-9, 10, size=(160, 32)),
                     columns=["f"+str(i) for i in range(32)])
    original = pd.Series.rank; calls = 0
    def tracked(series, *args, **kwargs):
        nonlocal calls
        calls += 1; return original(series, *args, **kwargs)
    monkeypatch.setattr(pd.Series, "rank", tracked)
    expected = reference(f); assert calls == 992
    calls = 0
    actual = redundancy_pairs(f, list(f), 3, 496)
    assert calls == 32 and len(actual) == 496 and actual == expected


def test_no_cross_call_reuse_and_no_input_mutation():
    f = pd.DataFrame(dict(a=[1., 2., 3., 4.], b=[4., 2., 3., 1.]),
                     index=[9, 4, 7, 1]); before = f.copy(deep=True)
    first = redundancy_pairs(f, list(f), 3, 1)
    pd.testing.assert_frame_equal(f, before)
    changed = f.copy(); changed["b"] = changed.a
    second = redundancy_pairs(changed, list(changed), 3, 1)
    assert first == reference(f) and second == reference(changed)
    assert first != second


def test_nullable_numeric_missingness_preserved():
    f = pd.DataFrame(dict(a=pd.Series([1., 2., pd.NA, 4., 5.], dtype="Float64"),
                          b=pd.Series([5., 4., 3., 2., 1.], dtype="Float64")))
    assert redundancy_pairs(f, list(f), 3, 1) == reference(f)


@pytest.mark.parametrize("minimum,maximum", [(3,0), (4,1), (5,3)])
def test_pair_cap_and_thin_common_rows(minimum, maximum):
    f = pd.DataFrame(dict(a=[1., 2., 3., 4.], b=[4., 3., np.nan, 1.], c=[1., 3., 2., 4.]))
    assert redundancy_pairs(f, list(f), minimum, maximum) == reference(f, minimum, maximum)


@pytest.mark.parametrize("field,value", [("minimum",True),("minimum",2),
    ("max_pairs",-1),("max_pairs",True),("max_cached_cells",-1),("max_cached_cells",True)])
def test_finite_parameter_guards(field, value):
    f = pd.DataFrame(dict(a=[1.,2.,3.],b=[3.,2.,1.]))
    kw = dict(minimum=3,max_pairs=1,max_cached_cells=10); kw[field]=value
    with pytest.raises(ValueError): redundancy_pairs(f,list(f),**kw)


@pytest.mark.parametrize("bad", [pd.Series([1+1j,2+0j,3+0j]),
    pd.Series(["1","2","3"]),pd.Series([1.,2.,np.inf])])
def test_invalid_numeric_inputs_rejected(bad):
    f=pd.DataFrame(dict(a=bad,b=[3.,2.,1.]))
    with pytest.raises(ValueError): redundancy_pairs(f,list(f),3,1)


def test_large_integer_order_is_not_rounded_by_float_conversion():
    f=pd.DataFrame(dict(a=np.array([2**53,2**53+1,2**53+2,2**53+3],dtype=np.uint64),
                        b=np.array([1,4,2,3],dtype=np.uint64)))
    assert redundancy_pairs(f,list(f),3,1)==reference(f)


def test_integrated_screen_outputs_match_original_pairwise_reference(monkeypatch):
    from tests.test_alpha_screening import example
    import src.alpharesearch.screening as screening
    # Two tiny synthetic screens; no model fitting or expression generation.
    data = example()
    fast = screening.screen(*data)
    def old_pairs(values, names, minimum, maximum):
        return reference(values[names], minimum, maximum)
    monkeypatch.setattr(screening, "redundancy_pairs", old_pairs)
    old = screening.screen(*data)
    for name in ["daily", "selections", "windows", "trials", "redundancy"]:
        pd.testing.assert_frame_equal(getattr(fast, name), getattr(old, name), check_exact=True)
    assert fast.receipt == old.receipt
