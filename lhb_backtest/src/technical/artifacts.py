"""Content-addressed inputs and atomic, inspectable experiment artifacts."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pandas as pd


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(paths):
    def one(path):
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"数据读取期间发生变更，请等待更新完成: {path}")
        return {"path": str(path.resolve()), "size": after.st_size,
                "mtime_ns": after.st_mtime_ns, "sha256": sha}
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, sorted(set(paths))))


def content_id(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verify_inventory(inventory):
    for record in inventory:
        stat = Path(record["path"]).stat()
        if stat.st_size != record["size"] or stat.st_mtime_ns != record["mtime_ns"]:
            raise RuntimeError(f"输入文件已变化，本次结果未发布: {record['path']}")


@contextmanager
def exclusive_run(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".running.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"已有研究任务运行；如进程异常退出，核对 {path} 中 PID 后移除此锁") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "started_at": utc_now()}, f)
        yield
    finally:
        path.unlink(missing_ok=True)


def verify_artifacts(folder: Path, manifest: dict):
    for name, expected in manifest["artifacts"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"快照完整性检查失败: {name}")


def records(frame):
    return jsonable(frame.to_dict("records"))
