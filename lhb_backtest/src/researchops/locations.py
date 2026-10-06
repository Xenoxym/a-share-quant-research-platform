"""Explicit relocation aliases for evidence access, separate from immutable content identity.

The local registry is a machine setting; it never rewrites evidence or database rows.
Only exact historical roots registered for the current store are accepted.
"""
from pathlib import Path, PurePosixPath
import json
import re

REGISTRY_NAME = 'location_aliases.json'


def previous_roots(root):
    root = Path(root).resolve()
    path = root / REGISTRY_NAME
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding='utf-8'))
    if (value.get('schema_version') != 1 or
            not isinstance(value.get('current_root'), str) or
            Path(value['current_root']) != root or
            not isinstance(value.get('previous_roots'), list)):
        raise ValueError('Invalid location alias registry for this research store')
    roots = []
    for previous in value['previous_roots']:
        if not isinstance(previous, str) or not Path(previous).is_absolute():
            raise ValueError('Historical research roots must be absolute paths')
        path = Path(previous)
        if '..' in path.parts or path == root or path in roots:
            raise ValueError('Ambiguous historical research root')
        roots.append(path)
    return roots


def evidence_folder(root, recorded_folder, fingerprint):
    """Return only the expected content-addressed folder, after location authorization."""
    if not isinstance(fingerprint, str) or not re.fullmatch(r'[0-9a-f]{64}', fingerprint):
        raise ValueError('Invalid evidence fingerprint')
    root = Path(root).resolve()
    relative = Path('analysis_evidence') / fingerprint
    expected = root / relative
    if not isinstance(recorded_folder, str) or not Path(recorded_folder).is_absolute():
        raise ValueError('Invalid recorded evidence location')
    recorded = Path(recorded_folder)
    if '..' in recorded.parts:
        raise ValueError('Evidence location may not contain parent traversal')
    if recorded != expected and not any(recorded == old / relative for old in previous_roots(root)):
        raise ValueError('Evidence location differs without an authorized relocation')
    if expected.resolve() != expected:
        raise ValueError('Evidence folder may not resolve through a link')
    return expected


def verify_evidence_paths(folder, artifacts):
    """Imported evidence cannot use absolute, traversal or linked artifact paths."""
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError('Evidence artifact inventory is empty')
    folder = Path(folder).resolve()
    for name, digest in artifacts.items():
        if not isinstance(name, str) or not name or '\\' in name:
            raise ValueError('Evidence artifact name is invalid')
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or ':' in name or
                not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
            raise ValueError('Evidence artifact identity or relative path is invalid')
        artifact = folder / name
        if artifact.resolve() != artifact or not artifact.resolve().is_relative_to(folder):
            raise ValueError('Evidence artifact may not resolve through a link')
