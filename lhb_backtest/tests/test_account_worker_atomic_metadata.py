"""Resource sampling tolerates atomic JSON saves without hiding output errors."""
import os
from pathlib import Path
import pytest
from src.researchops import account_worker


def output_roots(tmp_path, monkeypatch):
    job = tmp_path / "job"
    folder = job / "account_result"
    folder.mkdir(parents=True)
    native = tmp_path / "native"
    native.mkdir()
    monkeypatch.setattr(account_worker, "native_folder", lambda cfg: native)
    return job, folder, native


def interrupt_metadata(monkeypatch, target, action):
    original = Path.lstat
    fired = False

    def observation(path, *args, **kwargs):
        nonlocal fired
        if path == target and not fired:
            fired = True
            action()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", observation)


def test_atomic_json_save_can_finish_during_output_sampling(tmp_path, monkeypatch):
    job, folder, _ = output_roots(tmp_path, monkeypatch)
    target = folder / ("attempts.json." + "a" * 32 + ".tmp")
    target.write_bytes(b"done")
    final = folder / "attempts.json"
    interrupt_metadata(monkeypatch, target, lambda: os.replace(target, final))
    account_worker.output_bytes(job, {})
    assert final.read_bytes() == b"done"
    assert account_worker.output_bytes(job, {}) == 4


@pytest.mark.parametrize("name", ["attempts.json", "attempts.json.tmp", "attempts.json.bad.tmp", "trace.parquet." + "a" * 32 + ".tmp"])
def test_canonical_or_unrecognized_disappearance_is_fatal(tmp_path, monkeypatch, name):
    job, folder, _ = output_roots(tmp_path, monkeypatch)
    target = folder / name
    target.write_bytes(b"evidence")
    interrupt_metadata(monkeypatch, target, target.unlink)
    with pytest.raises(FileNotFoundError):
        account_worker.output_bytes(job, {})


def test_permission_failure_is_not_treated_as_atomic_save(tmp_path, monkeypatch):
    job, folder, _ = output_roots(tmp_path, monkeypatch)
    target = folder / ("attempts.json." + "a" * 32 + ".tmp")
    target.write_bytes(b"evidence")
    def denied():
        raise PermissionError("output unreadable")
    interrupt_metadata(monkeypatch, target, denied)
    with pytest.raises(PermissionError):
        account_worker.output_bytes(job, {})


def test_still_present_temp_after_transient_error_is_fatal(tmp_path, monkeypatch):
    job, folder, _ = output_roots(tmp_path, monkeypatch)
    target = folder / ("attempts.json." + "a" * 32 + ".tmp")
    target.write_bytes(b"evidence")
    def false_missing():
        raise FileNotFoundError("entry still exists")
    interrupt_metadata(monkeypatch, target, false_missing)
    with pytest.raises(FileNotFoundError):
        account_worker.output_bytes(job, {})


@pytest.mark.parametrize("broken", [False, True])
def test_linked_output_is_refused(tmp_path, monkeypatch, broken):
    job, folder, _ = output_roots(tmp_path, monkeypatch)
    destination = tmp_path / "destination"
    if not broken:
        destination.write_bytes(b"evidence")
    link = folder / ("attempts.json." + "a" * 32 + ".tmp")
    try:
        link.symlink_to(destination)
    except OSError as exc:
        pytest.skip("Host does not permit symlink creation: " + str(exc))
    with pytest.raises(ValueError, match="Linked account output refused"):
        account_worker.output_bytes(job, {})


def test_counts_ordinary_files_in_both_output_roots(tmp_path, monkeypatch):
    job, folder, native = output_roots(tmp_path, monkeypatch)
    (folder / "attempts.json").write_bytes(b"abc")
    nested = native / "configured"
    nested.mkdir()
    (nested / "equity.parquet").write_bytes(b"1234567")
    assert account_worker.output_bytes(job, {}) == 10
