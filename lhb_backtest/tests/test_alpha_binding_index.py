"""Binding semantics and growth counterexamples, with no new model fit."""
import json
from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from src.alpharesearch.features.base import FeatureBlock
from src.alpharesearch.registry import FeatureDefinition, FeatureRegistry


def definition(key="test.signal", version=1):
    return FeatureDefinition(key, "ratio", "stock_day", "synthetic",
        ("raw.signal",), '{"version":' + str(version) + '}',
        (("synthetic.py", "a"*64),), "before decision", "explicit")


def former_lookup(registry, source_bindings, keys):
    """The previous selected-key scan, retained only as a bounded reference."""
    result = {}
    for key in keys:
        ids = {i for binding in source_bindings for i in binding["definition_ids"]
               if any(d.definition_id == i and d.key == key for d in registry.definitions)}
        if len(ids) != 1:
            raise ValueError("Unknown or ambiguous bound feature definition")
        result[key] = registry.resolve(key, ids.pop())
    return result


@pytest.mark.parametrize("versions,bound,selected,accepted", [
    ([1], [1], ["test.signal"], True),
    ([1], [1, 1], ["test.signal"], True),
    ([1, 2], [1], ["test.signal"], True),
    ([1, 2], [2], ["test.signal"], True),
    ([1, 2], [1, 2], ["test.signal"], False),
    ([1], [], ["test.signal"], False),
    ([1], [999], ["test.signal"], False),
    ([1], [1, 999], ["test.signal"], True),
    ([1], [1], ["test.absent"], False),
    ([1], [1], [], True),
])
def test_explicit_versions_and_repeated_or_unregistered_bindings(versions, bound, selected, accepted):
    defs = {v: definition(version=v) for v in versions}
    registry = FeatureRegistry(defs.values())
    bindings = ({"definition_ids": [defs[v].definition_id if v in defs else "f"*64 for v in bound]},)
    if not accepted:
        with pytest.raises(ValueError, match="ambiguous"):
            registry.resolve_bound_definitions(bindings, selected)
        with pytest.raises(ValueError):
            former_lookup(registry, bindings, selected)
    else:
        new = registry.resolve_bound_definitions(bindings, selected)
        assert new == former_lookup(registry, bindings, selected)
        for d in new.values():
            assert d is defs[int(json.loads(d.parameters_json)["version"])]


def test_call_local_lookup_sees_registry_and_binding_changes_without_global_cache():
    one, two = definition(), definition(version=2)
    registry = FeatureRegistry([one]); bindings = [{"definition_ids": [one.definition_id]}]
    assert registry.resolve_bound_definitions(bindings, ["test.signal"])["test.signal"] is one
    registry.add(two)
    bindings[0]["definition_ids"].append(two.definition_id)
    with pytest.raises(ValueError, match="ambiguous"):
        registry.resolve_bound_definitions(bindings, ["test.signal"])
    bindings[0]["definition_ids"] = [two.definition_id]
    assert registry.resolve_bound_definitions(bindings, ["test.signal"])["test.signal"] is two
    bindings[0]["definition_ids"].clear()
    with pytest.raises(ValueError):
        registry.resolve_bound_definitions(bindings, ["test.signal"])


def test_multi_source_conditions_and_subset_order_do_not_mutate_registry():
    first, condition, unused = [definition(key="test."+n) for n in ("signal", "condition", "unused")]
    registry = FeatureRegistry([first, condition, unused])
    bindings = ({"definition_ids": [unused.definition_id, first.definition_id]},
                {"definition_ids": [condition.definition_id, first.definition_id, "f"*64]})
    before, saved_bindings = registry.to_dict(), deepcopy(bindings)
    for selected in (["test.signal", "test.condition"], ["test.condition", "test.signal"], ["test.signal"]):
        assert registry.resolve_bound_definitions(bindings, selected) == former_lookup(registry, bindings, selected)
    assert registry.to_dict() == before and bindings == saved_bindings


def test_hash_and_sort_calls_grow_with_definitions_not_feature_binding_product(monkeypatch):
    defs = [definition(key=f"test.signal_{i:03}") for i in range(128)]
    registry = FeatureRegistry(defs); bindings = ({"definition_ids": [d.definition_id for d in defs]},)
    reads = {"hash": 0, "definitions": 0}
    original_hash = FeatureDefinition.definition_id.fget
    original_definitions = FeatureRegistry.definitions.fget

    def tracked_hash(self):
        reads["hash"] += 1
        return original_hash(self)

    def tracked_definitions(self):
        reads["definitions"] += 1
        return original_definitions(self)

    monkeypatch.setattr(FeatureDefinition, "definition_id", property(tracked_hash))
    monkeypatch.setattr(FeatureRegistry, "definitions", property(tracked_definitions))
    result = registry.resolve_bound_definitions(bindings, [d.key for d in defs])
    assert set(result) == {d.key for d in defs}
    assert reads["definitions"] == 1 and reads["hash"] <= 3*len(defs)


def assert_prepared_equal(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b)
        elif isinstance(a, pd.Series):
            pd.testing.assert_series_equal(a, b)
        elif isinstance(a, FeatureBlock):
            pd.testing.assert_frame_equal(a.values, b.values)
            pd.testing.assert_frame_equal(a.missing, b.missing)
            assert a.units == b.units and a.metadata == b.metadata
        else:
            assert a == b


@pytest.mark.parametrize("path", ["screening", "learning"])
def test_official_preparation_matches_former_scan_without_fitting(monkeypatch, path):
    if path == "screening":
        from tests.test_alpha_screening import example
        from src.alpharesearch.screening import _prepare
    else:
        from tests.test_alpha_learning import example
        from src.alpharesearch.learning import _prepare
    data = example()
    actual = _prepare(*data)
    monkeypatch.setattr(FeatureRegistry, "resolve_bound_definitions", former_lookup)
    expected = _prepare(*data)
    assert_prepared_equal(actual, expected)


@pytest.mark.parametrize("path", ["screening", "learning"])
def test_official_preparation_rejects_two_bound_versions_before_fit(path):
    if path == "screening":
        from tests.test_alpha_screening import example
        from src.alpharesearch.screening import _prepare
        data = list(example()); assembly = data[2]
    else:
        from tests.test_alpha_learning import example
        from src.alpharesearch.learning import _prepare
        data = list(example()); assembly = data[3]
    cfg = data[0].to_dict()
    selected_key = cfg["channels"][0]["feature_key"] if path == "screening" else cfg["features"][0]
    first = assembly.registry.resolve(selected_key)
    extra = replace(first, parameters_json='{"version":"other"}')
    assembly.registry.add(extra)
    assembly.source_bindings[0]["definition_ids"].append(extra.definition_id)
    assembly.block.metadata.update(registry_version=assembly.registry.version_id,
                                   source_bindings=list(assembly.source_bindings))
    with pytest.raises(ValueError, match="ambiguous"):
        _prepare(*data)
