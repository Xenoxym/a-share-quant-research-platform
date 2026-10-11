"""Synthetic v2 native archive: one fixture, two fee paths, no fitted model.

The identity here is an interface fixture. It never authenticates a registered
completed learning experiment. All negative cases reuse the single archive.
"""
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from .test_alpha_account_audit import inputs, resign_archive
from .test_alpha_portfolio import resign_scores
from .test_score_reuse_versions import identity_for
from src.alpharesearch.account import _code_bundle, prepare_score_account
from src.alpharesearch.audit import audit_score_account_run
from src.alpharesearch.portfolio import build_score_portfolio
from src.technical.artifacts import content_id, write_json
from src.technical.runner import run


@pytest.fixture(name="tmp_path")
def stable_case_path(tmp_path_factory, request):
    # Ordinary, deterministic short directories: no mutable pytest current alias.
    key = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]
    folder = tmp_path_factory.getbasetemp()/("case_" + key)
    folder.mkdir()
    return folder


@pytest.fixture(scope="module")
def v2_native_archive(tmp_path_factory):
    root = tmp_path_factory.getbasetemp()/"v2arc"
    root.mkdir()
    data, snapshot, account = inputs(root)
    learning = data[0]
    original_scores = learning.predictions.score.copy(deep=True)
    identity = identity_for(learning)
    name = "src/alpharesearch/models.py"
    historic = identity["producer_files"][name]["text"] + "\n# synthetic historical producer fixture, not registered learning\n"
    h = hashlib.sha256(historic.encode()).hexdigest()
    identity["producer_files"][name] = dict(text=historic, sha256=h)
    learning.receipt["fit_identity"]["implementation_hashes"][name] = h
    learning.receipt["fitted_model"]["implementation_hashes"][name] = h
    learning.receipt["fit_id"] = content_id(learning.receipt["fit_identity"])
    learning.predictions["fit_id"] = learning.receipt["fit_id"]
    resign_scores(learning, membership=True)
    identity.update(fit_id=learning.receipt["fit_id"], prediction_id=learning.receipt["prediction_id"],
        learning_receipt_id=content_id(learning.receipt), prediction_outputs=learning.receipt["prediction_outputs"])
    identity["identity_id"] = content_id({k:v for k,v in identity.items() if k != "identity_id"})
    assert learning.predictions.score.equals(original_scores)
    portfolio = build_score_portfolio(*data)
    with pytest.raises(ValueError, match="installed implementation"):
        _code_bundle(learning, portfolio)
    prepared = prepare_score_account(*data, snapshot, account, reuse_identity=identity)
    assert prepared["extras"]["alpha_account_receipt.json"]["schema"] == "score-account-input-receipt-v2"
    folder = run(root, account, root=root, prepared=prepared,
        scenarios=["zero_transaction_cost", "configured"], publish=False,
        research_context={"role":"screen", "source":"synthetic_v2_archive_interface_fixture",
                          "registered_learning_authenticated":False})
    audited = audit_score_account_run(folder)  # Legal positive baseline before mutations.
    return dict(root=root, folder=folder, prepared=prepared, audited=audited,
                scores=original_scores, identity=identity)


def test_v2_native_archive_keeps_old_producer_and_new_consumer_separate(v2_native_archive):
    value = v2_native_archive
    folder = value["folder"]
    read = lambda name:json.loads((folder/name).read_text(encoding="utf8"))
    receipt, bundle = read("alpha_account_receipt.json"), read("alpha_scoring_code.json")
    assert receipt["schema"] == "score-account-input-receipt-v2"
    assert bundle["schema"] == "score-account-code-bundle-v2"
    assert receipt["verification_mode"] == "reuse_completed_identity_v1"
    assert receipt["reuse_identity_id"] == value["identity"]["identity_id"]
    assert "synthetic historical producer fixture" in bundle["producer_identity"]["producer_files"]["src/alpharesearch/models.py"]["text"]
    assert bundle["producer_identity"]["numeric_model_reproduction"] is False
    assert receipt["numeric_model_reproduction"] is False
    assert pd.read_parquet(folder/"alpha_predictions.parquet").score.equals(value["scores"])
    assert not (value["root"]/"latest.json").exists()


def test_v2_archive_independently_reconciles_both_fee_accounts(v2_native_archive):
    audit = v2_native_archive["audited"]
    assert set(audit["scenarios"]) == {"zero_transaction_cost", "configured"}
    assert audit["selection"]["selected_rows"] == 4
    assert audit["numeric_model_reproduction"] is False and audit["vendor_authentication"] is False
    for item in audit["scenarios"].values():
        assert item["annualized_return"] is None
        assert item["recalculated_metrics"]["buy_fills"] > 0
        assert item["recalculated_metrics"]["sell_fills"] > 0
        assert max(item["daily_errors"].values()) < .001
    folder = v2_native_archive["folder"]
    assert pd.read_parquet(folder/"configured/equity.parquet").receivable.max() > 0
    fills = pd.read_parquet(folder/"configured/fills.parquet")
    assert (fills.loc[fills.side.eq("sell"), "filled_shares"] % 100 != 0).any()


@pytest.mark.parametrize("change,reason", [
    ("receipt_mode", "Archived score reuse receipt"),
    ("receipt_identity", "Archived score reuse receipt"),
    ("bundle_mode", "Strict versioned score reuse bundle"),
    ("bundle_schema", "Strict versioned score reuse bundle"),
    ("producer_text", "Archived source role content"),
    ("consumer_text", "Archived source role content"),
    ("producer_fit", "Reuse model/prediction identity"),
    ("producer_receipt", "Reuse model/prediction identity"),
    ("producer_predictions", "Reuse model/prediction identity"),
    ("producer_schema", "Completed reuse identity"),
])
def test_v2_archive_rejects_resigned_role_and_receipt_changes(change, reason, v2_native_archive, tmp_path):
    value = v2_native_archive
    local = tmp_path/"r"
    shutil.copytree(value["root"]/"snapshots", local/"snapshots")
    folder = local/"runs"/value["folder"].name
    shutil.copytree(value["folder"], folder)
    def read(name):return json.loads((folder/name).read_text(encoding="utf8"))
    bundle = read("alpha_scoring_code.json")
    receipt = read("alpha_account_receipt.json")
    identity = bundle["producer_identity"]
    if change == "receipt_mode":receipt["verification_mode"] = "trust_success"
    elif change == "receipt_identity":receipt["reuse_identity_id"] = "0"*64
    elif change == "bundle_mode":bundle["verification_mode"] = "trust_success"
    elif change == "bundle_schema":bundle["schema"] = "score-account-code-bundle-v1"
    elif change == "producer_text":next(iter(identity["producer_files"].values()))["text"] += "\n# mutation\n"
    elif change == "consumer_text":next(iter(bundle["consumer_files"].values()))["text"] += "\n# mutation\n"
    elif change == "producer_fit":identity["fit_id"] = "0"*64
    elif change == "producer_receipt":identity["learning_receipt_id"] = "0"*64
    elif change == "producer_predictions":identity["prediction_outputs"] = dict(rows=999)
    else:identity["producer_schema"] = "synthetic-not-registered"
    # Deliberately refresh envelope IDs/checksums: semantic validation must reject
    # the specific mutation instead of failing only an outer byte checksum.
    identity["identity_id"] = content_id({k:v for k,v in identity.items() if k != "identity_id"})
    if change != "receipt_identity":receipt["reuse_identity_id"] = identity["identity_id"]
    receipt["code_bundle_id"] = content_id(bundle)
    write_json(folder/"alpha_scoring_code.json", bundle)
    write_json(folder/"alpha_account_receipt.json", receipt)
    resign_archive(folder)
    with pytest.raises(ValueError, match=reason):audit_score_account_run(folder)


@pytest.mark.parametrize("change,reason", [("cash", "Daily cash differs"), ("target", "Independent expected table"), ("score_member", "Learning score provenance differs"), ("fee_component", "Reported cost component differs")])
def test_v2_archive_checks_specific_cash_targets_membership_and_cost_reasons(change, reason, v2_native_archive, tmp_path):
    value = v2_native_archive
    local = tmp_path/"r"
    shutil.copytree(value["root"]/"snapshots", local/"snapshots")
    folder = local/"runs"/value["folder"].name
    shutil.copytree(value["folder"], folder)
    if change == "fee_component":
        doc = json.loads((folder/"result.json").read_text(encoding="utf8"))
        chosen = next(v for v in doc["portfolios"] if v["metrics"]["scenario"] == "configured")
        chosen["metrics"]["cost_components"]["commission"] += 100.
        write_json(folder/"result.json", doc)
    else:
        name, column = {"cash":("configured/equity.parquet", "cash"),
                        "target":("configured/targets.parquet", "target_shares"),
                        "score_member":("alpha_predictions.parquet", "sample_id")}[change]
        frame = pd.read_parquet(folder/name)
        frame.loc[0, column] = "replaced-source-member" if change == "score_member" else frame.loc[0, column] + 100.
        frame.to_parquet(folder/name, index=False)
        if change == "cash":
            # Keep the two archived equity copies identical. The independent
            # ledger reconstruction, rather than duplicate copies, must reject.
            doc = json.loads((folder/"result.json").read_text(encoding="utf8"))
            chosen = next(v for v in doc["portfolios"] if v["metrics"]["scenario"] == "configured")
            row = next(v for v in chosen["equity"] if v["date"] == frame.loc[0, "date"])
            row["cash"] = float(frame.loc[0, "cash"])
            write_json(folder/"result.json", doc)
    resign_archive(folder)
    with pytest.raises(ValueError, match=reason):audit_score_account_run(folder)
