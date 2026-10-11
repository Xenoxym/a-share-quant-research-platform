"""One synthetic registered worker; semantic clock audit shares its saved outputs.

This fixture is synthetic engineering evidence only. Each test reseals output
hashes and the worker receipt, but does not alter the registered proposal, source
inputs or DB result. audit_output is the semantic audit under test; the saved
completion certificate is restored rather than falsely re-certified.
"""
from contextlib import contextmanager
import copy
import json

import pandas as pd
import pytest

from src.alpharesearch.contracts import timestamps
from src.researchops.alpha_execution import _receipt
from src.researchops.alpha_experiments import verify_completed
from src.researchops.alpha_output_audit import audit_output
from src.technical.artifacts import digest, write_json
from tests.test_alpha_batch_registration import setup


@pytest.fixture(scope="module")
def registered_timestamp_output(tmp_path_factory):
    ops, session, proposal, _, source = setup(
        tmp_path_factory.mktemp("r32t_native"), max_experiments=1)
    proposal["spec"]["candidates"].append(dict(
        copy.deepcopy(proposal["spec"]["candidates"][0]),
        candidate_id="identical_counted"))
    # Bind the actual us Parquet inputs before registration. They are never
    # changed after the experiment is registered.
    for filename in ("membership.parquet", "decision_clocks.parquet"):
        path = source / filename
        frame = pd.read_parquet(path)
        frame["decision_at"] = timestamps(frame.decision_at).dt.as_unit("us")
        frame.to_parquet(path, index=False)
        stored = pd.read_parquet(path)
        assert str(stored.decision_at.dtype) == "datetime64[us, UTC]"
        role = "membership" if filename == "membership.parquet" else "decision_clocks"
        record = next(item for item in proposal["inputs"] if item["role"] == role)
        record.update(sha256=digest(path), bytes=path.stat().st_size)
    experiment = ops.register(session, proposal)
    # Exactly one real feature-worker execution for all three tests below.
    final = ops.execute(session, experiment["id"])
    assert final["status"] == "completed", final
    assert final["result"]["candidate_count"] == 2
    assert final["result"]["fits"] == final["result"]["accounts"] == 0
    assert len(ops.store.get(session["task_id"])["experiments"]) == 1
    baseline = verify_completed(ops, final)
    assert baseline["verified"] and baseline["completed_candidates"] == 2
    job = ops.root / "worker_jobs" / final["id"]
    return ops, final, job


@contextmanager
def restored_output(registered_timestamp_output):
    ops, experiment, job = registered_timestamp_output
    folder = job / "alpha_result"
    originals = {p.relative_to(folder): p.read_bytes()
                 for p in folder.rglob("*") if p.is_file()}
    receipt = (job / "receipt.json").read_bytes()
    # A valid positive baseline is required independently for every mutation.
    assert verify_completed(ops, experiment)["verified"]
    try:
        yield ops, experiment, job, folder
    finally:
        for path in folder.rglob("*"):
            if path.is_file() and path.relative_to(folder) not in originals:
                path.unlink()
        for name, data in originals.items():
            (folder / name).write_bytes(data)
        (job / "receipt.json").write_bytes(receipt)
        assert {p.relative_to(folder): p.read_bytes()
                for p in folder.rglob("*") if p.is_file()} == originals
        assert (job / "receipt.json").read_bytes() == receipt
        assert verify_completed(ops, experiment)["verified"]


def reseal_outputs_and_receipt(ops, experiment, job, folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"] = {
        p.relative_to(folder).as_posix(): digest(p)
        for p in folder.rglob("*")
        if p.is_file() and p != folder / "manifest.json"}
    write_json(folder / "manifest.json", manifest)
    receipt = json.loads((job / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "completed"
    receipt["result_manifest_hash"] = digest(folder / "manifest.json")
    write_json(job / "receipt.json", receipt)
    # Prove the coherently re-signed envelope is accepted; a stale SHA or
    # receipt must not mask the semantic rejection being tested.
    _receipt(ops, experiment, receipt)


def test_registered_native_audit_accepts_equal_ns_us_clocks(registered_timestamp_output):
    with restored_output(registered_timestamp_output) as (ops, experiment, job, folder):
        for cid in ("relative_close_rank", "identical_counted"):
            path = folder / cid / "values.parquet"
            values = pd.read_parquet(path)
            values["known_at"] = timestamps(values.known_at).dt.as_unit("ns")
            values["observed_end"] = timestamps(values.observed_end).dt.as_unit("us")
            values.to_parquet(path, index=False)
            actual = pd.read_parquet(path)
            role = next(r for r in experiment["proposal"]["inputs"] if r["role"] == "membership")
            members = pd.read_parquet(ops.project / role["path"])
            keys = ["sample_id", "trade_date", "stock_code"]
            joined = actual[keys + ["known_at", "observed_end"]].merge(
                members[keys + ["decision_at"]], on=keys, validate="one_to_one")
            assert str(joined.known_at.dtype) == "datetime64[ns, UTC]"
            assert str(joined.observed_end.dtype) == "datetime64[us, UTC]"
            assert str(joined.decision_at.dtype) == "datetime64[us, UTC]"
            assert not joined.known_at.equals(joined.decision_at)
            assert joined.known_at.eq(joined.decision_at).all()
            assert joined.observed_end.eq(joined.decision_at).all()
        reseal_outputs_and_receipt(ops, experiment, job, folder)
        result = audit_output(ops, experiment)
        assert result["verified"] and result["completed_candidates"] == 2
        assert result["rows_per_completed_candidate"] == 4
        assert result["fits"] == result["accounts"] == 0
        assert not result["numeric_output_reproduction"]


@pytest.mark.parametrize("offset_ns", [-1, 1])
def test_registered_native_audit_rejects_resealed_actual_ns_change(
        registered_timestamp_output, offset_ns):
    with restored_output(registered_timestamp_output) as (ops, experiment, job, folder):
        path = folder / "relative_close_rank" / "values.parquet"
        values = pd.read_parquet(path)
        # Move both clocks together so FeatureBlock's observation/availability
        # order remains valid and the actual decision-time check must reject.
        for field in ("known_at", "observed_end"):
            values[field] = timestamps(values[field]).dt.as_unit("ns")
            values.loc[0, field] += pd.Timedelta(nanoseconds=offset_ns)
        values.to_parquet(path, index=False)
        actual = pd.read_parquet(path)
        assert actual.known_at.eq(actual.observed_end).all()
        reseal_outputs_and_receipt(ops, experiment, job, folder)
        with pytest.raises(ValueError, match="Numerical output clocks differ from declared development decisions"):
            audit_output(ops, experiment)
