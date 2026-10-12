"""Retire old one-off stage scripts into a verified archive; never discard evidence."""

import argparse
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path

APP = Path(__file__).resolve().parents[1]


def retire(directory, apply=False):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(APP / "research") or not directory.name.startswith(
        "autonomous_delivery_"
    ):
        raise ValueError("Only historical autonomous_delivery research directories are supported")
    protected = set()
    checkpoint = directory / "CHECKPOINT.json"
    if checkpoint.exists():
        text = checkpoint.read_text(encoding="utf-8")
        info = json.loads(text)
        handoff = info.get("current_handoff")
        if isinstance(handoff, str):
            path = directory / handoff
            if path.resolve().parent != directory:
                raise ValueError("Handoff escaped stage directory")
            if path.exists():
                text += path.read_text(encoding="utf-8")
        protected |= set(re.findall(r"[A-Za-z0-9_-]+\.py\b", text))
    files = [
        p
        for p in sorted(directory.glob("*.py"))
        if p.name not in protected and not p.name.lower().startswith(("r32sm_", "r32sc_"))
    ]
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError("Linked stage scripts refused")
    archive = directory / "retired_stage_scripts.zip"
    index = directory / "RETIRED_STAGE_SCRIPTS.json"
    if archive.exists() or index.exists():
        raise ValueError("Archive already exists; preserve it and inspect index, never overwrite")
    rows = [
        dict(name=p.name, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        for p in files
    ]
    if not apply:
        return dict(
            candidates=len(rows),
            bytes=sum(r["bytes"] for r in rows),
            protected_names=sorted(protected),
            active_prefixes=["r32sm_", "r32sc_"],
        )
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.name)
    with archive.open("r+b") as stream:
        os.fsync(stream.fileno())
    # Read each stored original and its live source before removing any source.
    with zipfile.ZipFile(archive) as z:
        for row in rows:
            if (
                hashlib.sha256(z.read(row["name"])).hexdigest() != row["sha256"]
                or hashlib.sha256((directory / row["name"]).read_bytes()).hexdigest()
                != row["sha256"]
            ):
                raise ValueError("Archive or live script changed; no deletion allowed")
    index.write_text(
        json.dumps(
            dict(
                schema="retired-stage-scripts-v1",
                archive=archive.name,
                files=rows,
                protected_names=sorted(protected),
                restore=(
                    "Extract a specific original after checking sha256; "
                    "never overwrite a modified file."
                ),
                reason=(
                    "Replaced one-off execution wrappers; "
                    "bytes retained for historical reproduction"
                ),
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with index.open("r+b") as stream:
        os.fsync(stream.fileno())
    for row in rows:
        p = directory / row["name"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Script changed after archive validation; stop retirement")
        p.unlink()
    return dict(
        retired=len(rows),
        original_bytes=sum(r["bytes"] for r in rows),
        archive_bytes=archive.stat().st_size,
        archive=str(archive),
        index=str(index),
        remaining_scripts=len(list(directory.glob("*.py"))),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    print(json.dumps(retire(a.directory, a.apply), ensure_ascii=False))
