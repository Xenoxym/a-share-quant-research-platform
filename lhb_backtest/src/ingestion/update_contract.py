"""Publication and exclusion contracts for the official acquisition workflow."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


@contextmanager
def update_lock(path: Path):
    """OS lock is released on crash; existence of the file is not a lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b'0')
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError('Another SimTradeData update is running') from exc
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def recover_publication(target: Path) -> None:
    backup = target.with_name(target.name + '.update-rollback')
    marker = target.with_name(target.name + '.update-owner.json')
    if not marker.exists():
        return
    owner = json.loads(marker.read_text(encoding='utf-8'))
    if owner.get('target') != str(target.resolve()):
        raise RuntimeError('Update rollback belongs to a different target')
    if not target.exists() and backup.is_dir():
        backup.rename(target)


def publish_managed(staging: Path, target: Path) -> Path | None:
    """One managed rollback; never delete user Release or legacy backups.

    Directory renames are individually atomic. A crash between them can leave
    target absent; the next call recovers the old export before continuing.
    """
    target, staging = target.resolve(), staging.resolve()
    if staging.parent != target.parent or staging.name != target.name + '.staging':
        raise ValueError('Staging must be the designated sibling of the live export')
    if not staging.is_dir():
        raise FileNotFoundError(staging)
    backup = target.with_name(target.name + '.update-rollback')
    marker = target.with_name(target.name + '.update-owner.json')
    if backup.exists() and not marker.exists():
        raise RuntimeError('Refusing to delete an unmanaged rollback directory')
    recover_publication(target)
    if backup.exists():
        # Checked resolved parent and reserved name; reject links/junctions.
        if backup.is_symlink() or (hasattr(backup, 'is_junction') and backup.is_junction()):
            raise RuntimeError('Rollback must not be a link/junction')
        if backup.resolve().parent != target.parent:
            raise RuntimeError('Rollback escapes export directory')
        shutil.rmtree(backup)
    write_report(marker, {'target': str(target)})
    moved = False
    try:
        if target.exists():
            target.rename(backup)
            moved = True
        staging.rename(target)
    except BaseException:
        if moved and backup.exists() and not target.exists():
            backup.rename(target)
        raise
    return backup if moved else None
