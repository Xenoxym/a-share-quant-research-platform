"""Capture source content once, before importing project implementations."""
from pathlib import Path as _SourcePath
from hashlib import sha256 as _source_sha256

_source_folder = _SourcePath(__file__).resolve().parent
_SOURCE_SHA256 = {
    p.relative_to(_source_folder).as_posix(): _source_sha256(p.read_bytes()).hexdigest()
    for p in _source_folder.rglob("*.py")
    if "__pycache__" not in p.parts and p.resolve().is_relative_to(_source_folder)
}
