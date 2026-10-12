"""Small protocol/source counterexamples; no market execution, fit or account."""
from copy import deepcopy
import hashlib
import json

import pytest

from src.alpharesearch.batch import AlphaBatchSpec
from src.alpharesearch.batch_partition import (
    execution_partition, forecast_binding, is_forecast_batch, is_prepared_batch, validate_forecast_source,
)
from src.alpharesearch.dsl import Expr, compile_expression
from src.alpharesearch.registry import FeatureDefinition, FeatureRegistry


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def case():
    leaf = FeatureDefinition("stock.x", "ratio", "stock_day", "Synthetic fixed input",
        ("synthetic.source",), "{}", (("tests/test_all_candidate_forecast.py", "a" * 64),),
        "decision-time availability", "unknown remains missing")
    registry = FeatureRegistry([leaf])
    expression = compile_expression(Expr.ref("stock.x"), registry).to_dict()
    producer_spec = dict(schema="alpha-feature-batch-v4", name="tiny_producer", dataset_id="tiny_dataset",
        universe_id="tiny_mainboard", ranges=dict(warmup_start="2024-01-02", development_start="2024-01-05",
            development_end="2024-01-12", holdout_start="2024-01-15", holdout_end="2024-01-19"),
        candidates=[dict(candidate_id=cid, hypothesis="Fixed toy input", family="tiny_family", expression=deepcopy(expression))
                    for cid in ("fixed_a", "fixed_b")],
        blocks=dict(prepared=dict(values="assembly_values", missing="assembly_missing", definition="assembly_definition",
            receipt="prepared_receipt", source_receipt="assembly_receipt")),
        bindings={"stock.x": dict(block="prepared", column="stock.x", definition_id=leaf.definition_id,
            materialization_id="b" * 64)},
        budget=dict(max_candidates=64, max_input_bytes=1000000, max_grid_cells=1000,
            max_buffer_bytes=1000000, max_model_fits=0, max_accounts=0), allow_weak_vintage=False,
        evaluation_scope="retrospective_time_split", selection_rule="all_candidates_no_selection",
        primary_objectives=["account_return", "max_drawdown"], storage_layout="shared_index_v1",
        leaf_source="prepared_assembly_v1", temporal_policy="decision_date_pointwise_only")
    base_roles = AlphaBatchSpec.from_dict(producer_spec, registry).input_roles
    inputs = [dict(role=role, path="research/toy/" + role + ".json", sha256=sha(role), bytes=20)
              for role in sorted(base_roles)]
    producer = dict(kind="alpha_batch", experiment_id="exp-toy-producer", task_id="task-toy-producer",
        spec=producer_spec, inputs=inputs, engine_hash="c" * 64, registry_version=registry.version_id)
    manifest = dict(schema="registered-alpha-feature-manifest-v2", experiment_id=producer["experiment_id"],
        task_id=producer["task_id"], engine_hash=producer["engine_hash"], input_hash=sha(producer),
        candidate_manifest_hash="d" * 64, artifacts={"attempts.json": sha("attempts"), "result.json": sha("result")})
    audit = dict(schema="alpha-frozen-output-audit-v1", verified=True, experiment_id=producer["experiment_id"],
        candidate_count=2, completed_candidates=2, failed_candidates=0, fits=0, accounts=0,
        holdout_rows_executed=0, manifest_hash=sha(manifest))
    forecast = deepcopy(producer_spec)
    forecast.update(schema="alpha-feature-batch-v6", name="tiny_forecast", execution_partition="forecast_features",
        forecast_warmup_start=producer_spec["ranges"]["holdout_start"], forecast_source=dict(
            producer_experiment_id=producer["experiment_id"], producer_task_id=producer["task_id"],
            producer_fingerprint="e" * 64, input_role="producer_input", audit_role="producer_audit",
            manifest_role="producer_manifest"))
    hashes = {record["role"]: record["sha256"] for record in inputs}
    hashes.update(producer_input=sha(producer), producer_audit=sha(audit), producer_manifest=sha(manifest))
    return registry, forecast, producer, audit, manifest, hashes


def validate(values):
    _, forecast, producer, audit, manifest, hashes = values
    return validate_forecast_source(forecast, producer, audit, manifest, input_hashes=hashes)


def test_all_candidates_keep_expression_ids_and_three_extra_roles():
    values = case()
    registry, forecast, producer, _, _, _ = values
    spec = AlphaBatchSpec.from_dict(forecast, registry)
    assert spec.input_roles == AlphaBatchSpec.from_dict(producer["spec"], registry).input_roles | {
        "producer_input", "producer_audit", "producer_manifest"}
    assert spec.planned_attempts == AlphaBatchSpec.from_dict(producer["spec"], registry).planned_attempts
    assert execution_partition(forecast) == dict(name="forecast_features", warmup_start="2024-01-15",
        start="2024-01-15", end="2024-01-19")
    result = validate(values)
    assert result["candidate_count"] == 2 and result["selection_rule"] == "all_candidates_no_selection"
    assert result["historical_source_certified"] is False
    assert result["actual_registered_producer_verified"] is False
    assert is_prepared_batch(forecast) and is_forecast_batch(forecast)


def test_producer_name_and_execution_budgets_may_change():
    values = case()
    values[1]["name"] = "different_forecast_name"
    values[1]["budget"]["max_input_bytes"] *= 2
    assert validate(values)["candidate_count"] == 2


@pytest.mark.parametrize("field,value", [
    ("forecast_warmup_start", "2024-01-12"), ("execution_partition", "development"),
    ("leaf_source", "raw"), ("temporal_policy", "daily_history"), ("storage_layout", "unknown"),
])
def test_v6_requires_explicit_pointwise_forecast_contract(field, value):
    registry, forecast, *_ = case()
    forecast[field] = value
    with pytest.raises(ValueError):
        AlphaBatchSpec.from_dict(forecast, registry)


def test_time_series_expression_cannot_enter_prepared_forecast():
    registry, forecast, *_ = case()
    forecast["candidates"][0]["expression"] = compile_expression(
        Expr.call("lag", Expr.ref("stock.x"), periods=1), registry).to_dict()
    with pytest.raises(ValueError, match="forbid"):
        AlphaBatchSpec.from_dict(forecast, registry)


@pytest.mark.parametrize("change", ["missing", "reordered", "ast", "family", "binding", "weak_vintage", "ranges"])
def test_full_ordered_library_and_frozen_source_scope_cannot_change(change):
    values = case()
    forecast = values[1]
    if change == "missing":
        forecast["candidates"].pop()
    elif change == "reordered":
        forecast["candidates"].reverse()
    elif change == "ast":
        forecast["candidates"][0]["expression"]["ast"]["op"] = "abs"
    elif change == "family":
        forecast["candidates"][0]["family"] = "changed_family"
    elif change == "binding":
        forecast["bindings"]["stock.x"]["materialization_id"] = "f" * 64
    elif change == "weak_vintage":
        forecast["allow_weak_vintage"] = True
    else:
        forecast["ranges"]["development_end"] = "2024-01-11"
    with pytest.raises(ValueError, match="ordered producer"):
        validate(values)


@pytest.mark.parametrize("field,value", [
    ("verified", False), ("candidate_count", 1), ("completed_candidates", 1), ("failed_candidates", 1),
    ("fits", 1), ("accounts", 1), ("holdout_rows_executed", 1), ("completed_candidates", True),
])
def test_incomplete_or_non_feature_only_saved_audit_is_rejected(field, value):
    values = case()
    values[3][field] = value
    with pytest.raises(ValueError):
        validate(values)


@pytest.mark.parametrize("change", ["manifest_hash", "input_hash", "engine", "manifest_task", "producer", "role_sha"])
def test_saved_hashes_and_producer_identity_are_not_success_flags(change):
    values = case()
    _, _, producer, _, manifest, hashes = values
    if change == "manifest_hash":
        hashes["producer_manifest"] = "0" * 64
    elif change == "input_hash":
        manifest["input_hash"] = "0" * 64
    elif change == "engine":
        manifest["engine_hash"] = "0" * 64
    elif change == "manifest_task":
        manifest["task_id"] = "task-other"
    elif change == "producer":
        producer["experiment_id"] = "exp-other"
    else:
        hashes["membership"] = "0" * 64
    with pytest.raises(ValueError):
        validate(values)


@pytest.mark.parametrize("change", ["duplicate", "base_alias", "extra", "missing", "fingerprint"])
def test_source_descriptor_is_strict_and_roles_are_distinct(change):
    values = case()
    registry, forecast, *_ = values
    source = forecast["forecast_source"]
    if change == "duplicate":
        source["audit_role"] = source["input_role"]
    elif change == "base_alias":
        source["input_role"] = "registry"
    elif change == "extra":
        source["screen_role"] = "screen"
    elif change == "missing":
        source.pop("manifest_role")
    else:
        source["producer_fingerprint"] = "not_a_hash"
    with pytest.raises(ValueError):
        AlphaBatchSpec.from_dict(forecast, registry)
    with pytest.raises(ValueError):
        validate(values)


def test_duplicate_or_missing_producer_roles_are_rejected():
    for change in ("duplicate", "missing"):
        values = case()
        records = values[2]["inputs"]
        if change == "duplicate":
            records.append(deepcopy(records[0]))
        else:
            records.pop()
        with pytest.raises(ValueError, match="exact producer"):
            validate(values)


def test_v4_keeps_development_partition_and_refuses_forecast_fields():
    registry, _, producer, *_ = case()
    document = producer["spec"]
    assert is_prepared_batch(document) and not is_forecast_batch(document)
    assert execution_partition(document) == dict(name="development", warmup_start="2024-01-02",
        start="2024-01-05", end="2024-01-12")
    document["forecast_source"] = {}
    with pytest.raises(ValueError, match="Strict alpha feature batch schema"):
        AlphaBatchSpec.from_dict(document, registry)


def test_forecast_binding_preserves_distinct_evidence_meanings():
    _, forecast, producer, _, _, hashes = case()
    assert forecast_binding(forecast, hashes) == {"forecast_source_sha256": hashes["producer_input"]}
    assert forecast_binding(producer["spec"], hashes) == {}
    legacy = {"schema": "alpha-feature-batch-v3", "training_selection": {"receipt_role": "receipt"}}
    assert forecast_binding(legacy, {"receipt": "f" * 64}) == {"training_selection_sha256": "f" * 64}
