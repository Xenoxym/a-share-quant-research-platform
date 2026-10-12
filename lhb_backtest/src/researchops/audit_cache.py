"""Reuse numerical audits only when their complete file/implementation binding matches.

This saves mathematical recomputation, not content verification. Cache entries
live in the existing SQLite store; historical receipts are never rewritten.
"""

import hashlib
import importlib.metadata
import json
import marshal
import sys
import time

from ..technical.artifacts import content_id, digest
from .store import encode

FOLDERS = {
    "alpha_batch": "alpha_result",
    "alpha_screen": "screen_result",
    "alpha_learn": "learn_result",
    "alpha_account": "account_result",
}


def _files(folder):
    # Check each directory before descending, including Windows junctions.
    files = []
    pending = [folder]
    while pending:
        path = pending.pop()
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("Linked audit evidence refused")
        if path.is_dir():
            pending.extend(path.iterdir())
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError("Audit evidence folder/file missing")
    return files


def _binding(research, experiment):
    proposal = experiment["proposal"]
    job = research.root / "worker_jobs" / experiment["id"]
    paths = _files(job / "code") + _files(job / FOLDERS[proposal["kind"]])
    for name in (
        "input.json",
        "ready.json",
        "code_manifest.json",
        "candidate_manifest.json",
        "receipt.json",
        "launch.json",
    ):
        path = job / name
        if path.exists():
            paths.append(path)
    from .alpha_experiments import _path

    paths.extend(_path(research.project, record["path"]) for record in proposal["inputs"])
    if proposal["kind"] == "alpha_account":
        from .account_experiments import native_folder

        cfg = json.loads((job / "input.json").read_text(encoding="utf-8"))
        paths.extend(_files(native_folder(cfg)))
        paths.extend(_files(research.root / "snapshots" / proposal["spec"]["snapshot_id"]))
    # Changes to any current Python audit dependency conservatively invalidate.
    paths.extend(
        p for p in (research.project / "src").rglob("*.py") if "__pycache__" not in p.parts
    )
    audit_tool = research.project / "tools/verify_technical.py"
    if audit_tool.exists():
        paths.append(audit_tool)
    unique = sorted(set(paths), key=str)
    files = {}
    for path in unique:
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("Linked audit input refused")
        resolved = path.resolve()
        if not (
            resolved.is_relative_to(research.project) or resolved.is_relative_to(research.root)
        ):
            raise ValueError("Audit input escaped registered project/root")
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Audit input changed while hashing")
        files[str(path)] = {"sha256": sha, "bytes": before.st_size}
    versions = {"python": sys.version, "executable": sys.executable}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "duckdb", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "schema": "numerical-audit-binding-v1",
        "experiment_id": experiment["id"],
        "proposal": proposal,
        "files": files,
        "environment": versions,
    }


def numerical_audit(research, experiment, auditor, *, force=False):
    """Only a real successful auditor creates a reusable result."""
    with research.store.connection() as db:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='execution_audits'").fetchone()
    if not exists and not research.store.readonly:
        with research.store.connection(True) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS execution_audits(
                experiment_id TEXT PRIMARY KEY, binding_hash TEXT NOT NULL,
                audit TEXT NOT NULL, checked_at REAL NOT NULL)""")
        exists = True
    before = _binding(research, experiment)
    runtime_code = getattr(auditor, "__code__", None)
    before["auditor"] = dict(
        module=auditor.__module__,
        name=auditor.__qualname__,
        code_hash=hashlib.sha256(marshal.dumps(runtime_code)).hexdigest()
        if runtime_code is not None
        else None,
    )
    key = content_id(before)
    row = None
    if exists:
        with research.store.connection() as db:
            row = db.execute(
                "SELECT * FROM execution_audits WHERE experiment_id=?", (experiment["id"],)
            ).fetchone()
    if not force and row and row["binding_hash"] == key:
        after = _binding(research, experiment)
        after["auditor"] = before["auditor"]
        if after != before:
            raise ValueError("Audit evidence changed during cached verification")
        return json.loads(row["audit"])
    audit = auditor(research, experiment)
    if not isinstance(audit, dict) or audit.get("verified") is not True:
        raise ValueError("Only a verified numerical audit may be cached")
    after = _binding(research, experiment)
    after["auditor"] = before["auditor"]
    if after != before:
        raise ValueError("Audit evidence changed during numerical validation")
    if research.store.readonly:
        return audit
    with research.store.connection(True) as db:
        db.execute(
            """INSERT INTO execution_audits VALUES(?,?,?,?)
                   ON CONFLICT(experiment_id) DO UPDATE SET
                   binding_hash=excluded.binding_hash,audit=excluded.audit,
                   checked_at=excluded.checked_at""",
            (experiment["id"], key, encode(audit), time.time()),
        )
    return audit
