"""Explicit forecast partitions and frozen train-window selection identities.

No target arrays, fitting or formula evaluation. Identity checks do not prove
vendor history, numerical selection reproduction or an unseen holdout.
"""
import json
import re
from .contracts import timestamp, required_id
from .registry import SHA


def is_prepared_batch(document):
    return document.get("schema") in {"alpha-feature-batch-v4", "alpha-feature-batch-v6"}


def is_forecast_batch(document):
    return document.get("schema") in {"alpha-feature-batch-v3", "alpha-feature-batch-v6"}


def forecast_binding(document, input_hashes):
    """Return the partition's evidence hash without inventing Screen selection."""
    if document.get("schema") == "alpha-feature-batch-v3":
        return {"training_selection_sha256": input_hashes[document["training_selection"]["receipt_role"]]}
    if document.get("schema") == "alpha-feature-batch-v6":
        return {"forecast_source_sha256": input_hashes[document["forecast_source"]["input_role"]]}
    return {}


def _forecast_source_roles(source):
    fields = {"producer_experiment_id", "producer_task_id", "producer_fingerprint",
              "input_role", "audit_role", "manifest_role"}
    if not isinstance(source, dict) or set(source) != fields:
        raise ValueError("Strict all-candidate forecast source fields required")
    for field in ("producer_experiment_id", "producer_task_id"):
        required_id(source[field], field)
    if not isinstance(source["producer_fingerprint"], str) or not SHA.fullmatch(source["producer_fingerprint"]):
        raise ValueError("Producer fingerprint must be a SHA256 identity")
    roles = [source[field] for field in ("input_role", "audit_role", "manifest_role")]
    if any(not isinstance(role, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,79}", role) for role in roles) or len(set(roles)) != 3:
        raise ValueError("Three distinct forecast source roles required")
    return tuple(roles)


def execution_partition(document):
    if is_forecast_batch(document):
        return dict(name="forecast_features", warmup_start=document["forecast_warmup_start"],
                    start=document["ranges"]["holdout_start"], end=document["ranges"]["holdout_end"])
    return dict(name="development", warmup_start=document["ranges"]["warmup_start"],
                start=document["ranges"]["development_start"], end=document["ranges"]["development_end"])


def validate_forecast_selection(document, receipt, summary, library, source_receipt, *, input_hashes):
    if document.get("schema") != "alpha-feature-batch-v3":
        raise ValueError("Forecast selection verification requires explicit v3")
    fields = {"schema", "dataset_id", "universe_id", "development_start", "development_end",
              "training_label_cutoff", "selected_at", "historical_selection_certified",
              "summary_sha256", "library_sha256", "source_receipt_sha256", "selected_candidate_ids"}
    if not isinstance(receipt, dict) or set(receipt) != fields or receipt["schema"] != "train-window-formula-selection-v1":
        raise ValueError("Strict frozen train-window selection receipt required")
    if receipt["historical_selection_certified"] is not False:
        raise ValueError("Retrospective selection cannot certify historical selection or a blind holdout")
    for field in ("dataset_id", "universe_id"):
        if receipt[field] != document[field]:
            raise ValueError("Selection dataset or universe differs")
    for field in ("development_start", "development_end"):
        if receipt[field] != document["ranges"][field]:
            raise ValueError("Training selection date window differs")
    for field in ("summary_sha256", "library_sha256", "source_receipt_sha256"):
        value = receipt[field]
        if not isinstance(value, str) or not SHA.fullmatch(value) or value != input_hashes[field]:
            raise ValueError("Frozen selection source identity differs")
    selected_at = timestamp(receipt["selected_at"])
    cutoff = timestamp(receipt["training_label_cutoff"])
    if selected_at < cutoff:
        raise ValueError("Actual selection time precedes declared training information")
    if cutoff >= timestamp(document["ranges"]["holdout_start"] + "T00:00:00+08:00"):
        raise ValueError("Training information cutoff reaches forecast dates")
    if not isinstance(source_receipt, dict) or source_receipt.get("schema") != "formula-training-leaves-v1":
        raise ValueError("Frozen training input receipt required")
    if source_receipt.get("universe_id") != document["universe_id"] or any(
            source_receipt.get(field) != document["ranges"][field]
            for field in ("development_start", "development_end")):
        raise ValueError("Training source receipt scope differs")
    if timestamp(source_receipt["fit_cutoff"]) != cutoff or source_receipt.get("label_filter_for_membership") is not False:
        raise ValueError("Training source cutoff or eligibility declaration differs")
    if not isinstance(library, list) or not 1 <= len(library) <= 10_000:
        raise ValueError("Finite frozen candidate library required")
    records = {}
    for candidate in library:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("candidate_id"), str) or candidate["candidate_id"] in records:
            raise ValueError("Unique source-library candidate identities required")
        records[candidate["candidate_id"]] = candidate
    if not isinstance(summary, dict) or any(type(summary.get(k)) is not int or summary[k] != 0 for k in ("new_fits", "new_accounts")):
        raise ValueError("Formula training summary cannot contain model/account claims")
    if selected_at < timestamp(summary["at"]):
        raise ValueError("Actual selection time precedes its source summary")
    trials = summary.get("all_trials")
    if not isinstance(trials, list) or len(trials) != len(records):
        raise ValueError("Complete training attempts required, including rejected channels")
    trial_ids = set()
    promoted = set()
    for trial in trials:
        if not isinstance(trial, dict) or trial.get("channel_id") not in records or trial["channel_id"] in trial_ids:
            raise ValueError("Training attempts differ from the complete candidate library")
        cid = trial["channel_id"]
        trial_ids.add(cid)
        if trial.get("family") != records[cid]["family"] or type(trial.get("promoted")) is not bool:
            raise ValueError("Training candidate family or promotion type differs")
        if trial["promoted"]:
            if trial.get("quality_passed") is not True or trial.get("proxy_complete") is not True:
                raise ValueError("Incomplete training candidate cannot enter the frozen shortlist")
            promoted.add(cid)
    ids = receipt["selected_candidate_ids"]
    if not isinstance(ids, list) or not ids or any(not isinstance(cid, str) for cid in ids) or ids != sorted(set(ids)) or set(ids) != promoted:
        raise ValueError("Receipt must preserve the full training-promoted shortlist")
    declared = document["candidates"]
    families = {candidate["family"] for candidate in declared}
    required = {cid for cid in promoted if records[cid]["family"] in families}
    if {candidate["candidate_id"] for candidate in declared} != required:
        raise ValueError("Forecast batch must retain all selected candidates in its declared families")
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    for candidate in declared:
        source = records[candidate["candidate_id"]]
        if canonical(candidate) != canonical({key: source[key] for key in ("candidate_id", "hypothesis", "family", "expression")}):
            raise ValueError("Forecast candidate definition differs from the frozen training library")
    return dict(selection_scope="retrospective_train_window_only", selected_candidates=len(ids),
                batch_candidates=len(declared), historical_selection_certified=False,
                future_labels_read=False, numerical_selection_reproduction=False)


def validate_forecast_source(document, producer_input, audit, manifest, *, input_hashes):
    """Check frozen all-candidate declarations; authenticate no DB or vendor history.

    The caller verifies exact raw-file SHA identities and the actual producer DB
    fingerprint. JSON objects cannot independently authenticate their raw bytes.
    No labels, Screen selection, formula execution, fits or accounts are read.
    """
    def require(ok, message):
        if not ok:
            raise ValueError(message)

    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    require(isinstance(document, dict) and document.get("schema") == "alpha-feature-batch-v6",
            "All-candidate source verification requires explicit v6")
    source = document.get("forecast_source")
    roles = _forecast_source_roles(source)
    require(isinstance(input_hashes, dict) and all(isinstance(role, str) and isinstance(sha, str)
            and SHA.fullmatch(sha) for role, sha in input_hashes.items())
            and set(roles) <= set(input_hashes), "Frozen forecast input SHA identities required")
    require(all(isinstance(value, dict) for value in (producer_input, audit, manifest)),
            "Frozen producer input, audit and manifest JSON required")
    producer_id, task_id = source["producer_experiment_id"], source["producer_task_id"]
    require(producer_input.get("kind") == "alpha_batch"
            and producer_input.get("experiment_id") == producer_id and producer_input.get("task_id") == task_id,
            "Frozen producer input identity differs")
    spec = producer_input.get("spec")
    require(isinstance(spec, dict) and spec.get("schema") == "alpha-feature-batch-v4"
            and spec.get("leaf_source") == "prepared_assembly_v1"
            and spec.get("temporal_policy") == "decision_date_pointwise_only"
            and spec.get("storage_layout") == "shared_index_v1", "Original prepared development v4 required")
    fields = ("dataset_id", "universe_id", "ranges", "allow_weak_vintage", "candidates", "bindings", "blocks",
              "storage_layout", "leaf_source", "temporal_policy", "evaluation_scope", "selection_rule", "primary_objectives")
    require(all(field in spec and field in document and canonical(spec[field]) == canonical(document[field])
            for field in fields), "Full ordered producer candidates, leaves or original scope differ")
    require(document.get("execution_partition") == "forecast_features"
            and document.get("forecast_warmup_start") == spec["ranges"].get("holdout_start"),
            "Original forecast partition and zero-history warmup required")
    candidates = spec["candidates"]
    require(isinstance(candidates, list) and 1 <= len(candidates) <= 10_000
            and all(isinstance(candidate, dict) and isinstance(candidate.get("candidate_id"), str)
                    for candidate in candidates)
            and len({candidate["candidate_id"] for candidate in candidates}) == len(candidates)
            and spec["selection_rule"] == "all_candidates_no_selection", "Complete unique all-candidate library required")
    require(audit.get("schema") == "alpha-frozen-output-audit-v1" and audit.get("verified") is True
            and audit.get("experiment_id") == producer_id and audit.get("task_id", task_id) == task_id,
            "Verified saved producer audit identity required")
    counts = {"candidate_count": len(candidates), "completed_candidates": len(candidates),
              "failed_candidates": 0, "fits": 0, "accounts": 0, "holdout_rows_executed": 0}
    require(all(type(audit.get(field)) is int and audit[field] == count for field, count in counts.items()),
            "Producer audit must complete every declared candidate without fits, accounts or holdout")
    require(audit.get("manifest_hash") == input_hashes[source["manifest_role"]],
            "Saved producer manifest hash differs from audit")
    engine_hash = producer_input.get("engine_hash")
    require(isinstance(engine_hash, str) and SHA.fullmatch(engine_hash)
            and manifest.get("experiment_id") == producer_id and manifest.get("task_id") == task_id
            and manifest.get("engine_hash") == engine_hash
            and manifest.get("input_hash") == input_hashes[source["input_role"]],
            "Producer manifest experiment, task, engine or input identity differs")
    records = producer_input.get("inputs")
    require(isinstance(records, list) and 1 <= len(records) <= 512
            and all(isinstance(record, dict) and set(record) == {"role", "path", "sha256", "bytes"}
                    and isinstance(record["role"], str) and isinstance(record["sha256"], str)
                    and SHA.fullmatch(record["sha256"]) and type(record["bytes"]) is int and record["bytes"] >= 0
                    for record in records), "Complete frozen producer input-role identities required")
    record_roles = [record["role"] for record in records]
    base_roles = {"registry", "membership", "calendar", "decision_clocks"}
    require(isinstance(spec["blocks"], dict) and all(isinstance(block, dict) for block in spec["blocks"].values()),
            "Prepared producer source block roles required")
    base_roles |= {role for block in spec["blocks"].values() for role in block.values()}
    require(len(set(record_roles)) == len(record_roles) and set(record_roles) == base_roles
            and not (set(roles) & base_roles), "Distinct exact producer and forecast source roles required")
    require(all(input_hashes.get(record["role"]) == record["sha256"] for record in records),
            "Original producer source input role SHA changed")
    return dict(producer_experiment_id=producer_id, producer_task_id=task_id,
                producer_fingerprint=source["producer_fingerprint"], candidate_count=len(candidates),
                selection_rule="all_candidates_no_selection", historical_source_certified=False,
                actual_registered_producer_verified=False, future_labels_read=False,
                source_input_sha256=input_hashes[source["input_role"]],
                source_audit_sha256=input_hashes[source["audit_role"]],
                source_manifest_sha256=input_hashes[source["manifest_role"]])
