"""Explicit forecast partitions and frozen train-window selection identities.

No target arrays, fitting or formula evaluation. Identity checks do not prove
vendor history, numerical selection reproduction or an unseen holdout.
"""
import json
from .contracts import timestamp
from .registry import SHA


def execution_partition(document):
    if document.get("schema") == "alpha-feature-batch-v3":
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
