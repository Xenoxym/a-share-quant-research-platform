"""Persist claim credentials inside the claim transaction, without overwriting a file."""

import json
import os
from pathlib import Path


def claim_to_file(store, task, worker, path, ttl=1800):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:

            def persist(session):
                json.dump(session, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())

            session = store.claim(task, worker, ttl, persist=persist)
        return session
    except BaseException:
        path.unlink(missing_ok=True)
        raise
