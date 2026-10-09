"""Lossless shared row keys for feature batches; no evaluation or model fitting."""
from dataclasses import dataclass
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from .features.base import FeatureBlock
from ..technical.artifacts import digest, write_json

KEYS = ["sample_id", "trade_date", "stock_code"]
NAME = re.compile(r"^[a-z][a-z0-9_-]{0,79}$")
INDEX_FILES = {"shared_keys.parquet", "shared_keys.json"}
CANDIDATE_FILES = {"values.parquet", "missing.parquet", "definition.json", "storage.json"}


def _plain(path):
    path = Path(path)
    if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()) for p in (path, *path.parents)):
        raise ValueError("Linked feature storage path refused")
    return path


def _budget(value):
    if type(value) is not int or not 1024 <= value <= 100_000_000_000:
        raise ValueError("Finite positive feature storage byte budget required")
    return value


def _size(root):
    total = 0
    for path in root.rglob("*"):
        _plain(path)
        if path.is_file(): total += path.stat().st_size
    return total


def _keys(frame):
    if (not isinstance(frame, pd.DataFrame) or list(frame) != KEYS or frame.empty
            or frame.isna().any().any() or frame.duplicated(KEYS).any()
            or frame.sample_id.duplicated().any()
            or frame.duplicated(["trade_date", "stock_code"]).any()):
        raise ValueError("Unique nonempty exact shared feature keys required")
    if not all(frame[k].map(lambda x: isinstance(x, str) and bool(x)).all() for k in KEYS):
        raise ValueError("Shared key fields must be nonempty strings")
    return frame.reset_index(drop=True)


@dataclass(frozen=True)
class SharedFeatureIndex:
    directory: Path
    keys: pd.DataFrame
    parquet_sha256: str
    index_sha256: str


def write_shared_index(directory, keys, *, max_bytes):
    limit = _budget(max_bytes); directory = _plain(directory)
    directory.mkdir(parents=True, exist_ok=True)
    keys = _keys(keys).copy()
    if any((directory / n).exists() for n in INDEX_FILES):
        raise ValueError("Shared feature index already exists; never overwrite")
    temporary = directory / "shared_keys.parquet.partial"
    if temporary.exists(): raise ValueError("Partial shared index retained; no overwrite")
    keys.to_parquet(temporary, index=False, compression="zstd")
    if _size(directory) > limit: raise ValueError("Shared index exceeds byte budget; partial retained")
    temporary.replace(directory / "shared_keys.parquet")
    record = dict(schema="shared-feature-index-v1", rows=len(keys), columns=KEYS,
                  keys_sha256=digest(directory / "shared_keys.parquet"),
                  keys_bytes=(directory / "shared_keys.parquet").stat().st_size,
                  row_order="explicit uint32 position within this exact immutable index")
    write_json(directory / "shared_keys.json", record)
    if _size(directory) > limit: raise ValueError("Shared index metadata exceeds byte budget")
    return SharedFeatureIndex(directory, keys, record["keys_sha256"], digest(directory / "shared_keys.json"))


def read_shared_index(directory, *, max_bytes):
    limit = _budget(max_bytes); directory = _plain(directory)
    p = _plain(directory / "shared_keys.json"); q = _plain(directory / "shared_keys.parquet")
    if p.stat().st_size > 16_384 or q.stat().st_size + p.stat().st_size > limit:
        raise ValueError("Shared index exceeds input byte budget")
    record = json.loads(p.read_text(encoding="utf8"))
    expected = {"schema", "rows", "columns", "keys_sha256", "keys_bytes", "row_order"}
    if (set(record) != expected or record["schema"] != "shared-feature-index-v1"
            or record["columns"] != KEYS or type(record["rows"]) is not int
            or not 1 <= record["rows"] <= 10_000_000 or type(record["keys_bytes"]) is not int
            or record["keys_bytes"] != q.stat().st_size or record["keys_sha256"] != digest(q)
            or record["row_order"] != "explicit uint32 position within this exact immutable index"):
        raise ValueError("Shared index record/bytes identity differs")
    keys = _keys(pd.read_parquet(q))
    if len(keys) != record["rows"]: raise ValueError("Shared index row count differs")
    return SharedFeatureIndex(directory, keys, record["keys_sha256"], digest(p))


def write_compact_candidate(block, index, candidate_id, *, max_bytes):
    limit = _budget(max_bytes)
    if not isinstance(index, SharedFeatureIndex) or not isinstance(candidate_id, str) or not NAME.fullmatch(candidate_id):
        raise ValueError("Explicit shared index and candidate identity required")
    block.validate(); root = _plain(index.directory)
    if (block.metadata["key_columns"] != KEYS or block.units != {"score": "ratio"}
            or not block.values[KEYS].reset_index(drop=True).equals(index.keys)
            or digest(root / "shared_keys.parquet") != index.parquet_sha256
            or digest(root / "shared_keys.json") != index.index_sha256):
        raise ValueError("Candidate keys differ from its immutable shared index")
    target = _plain(root / candidate_id); target.mkdir(exist_ok=False)
    for name, frame in (("values", block.values[["score", "observed_end", "known_at"]]), ("missing", block.missing.drop(columns=KEYS))):
        frame = frame.reset_index(drop=True).copy()
        frame.insert(0, "row_index", np.arange(len(frame), dtype=np.uint32))
        temporary = target / (name + ".parquet.partial")
        frame.to_parquet(temporary, index=False, compression="zstd")
        if _size(root) > limit: raise ValueError("Compact batch byte budget exceeded; partial retained")
        temporary.replace(target / (name + ".parquet"))
    write_json(target / "definition.json", dict(block.metadata, units=block.units))
    write_json(target / "storage.json", dict(schema="shared-feature-candidate-v1", candidate_id=candidate_id,
        rows=len(index.keys), keys_sha256=index.parquet_sha256, index_sha256=index.index_sha256,
        value_columns=["score", "observed_end", "known_at"], missing_columns=["score"],
        position_dtype="uint32"))
    if _size(root) > limit: raise ValueError("Compact batch metadata exceeds byte budget")
    return target


def read_compact_candidate(index, candidate_id, *, max_bytes):
    limit = _budget(max_bytes)
    if not isinstance(index, SharedFeatureIndex) or not isinstance(candidate_id, str) or not NAME.fullmatch(candidate_id):
        raise ValueError("Explicit shared index and candidate identity required")
    root = _plain(index.directory); target = _plain(root / candidate_id)
    if ({p.name for p in target.iterdir()} != CANDIDATE_FILES
            or any(not _plain(p).is_file() for p in target.iterdir())
            or sum(p.stat().st_size for p in target.iterdir()) > limit):
        raise ValueError("Strict bounded compact candidate file set required")
    for name in ("storage.json", "definition.json"):
        if (target / name).stat().st_size > 1_000_000: raise ValueError("Compact metadata too large")
    record = json.loads((target / "storage.json").read_text(encoding="utf8"))
    expected = dict(schema="shared-feature-candidate-v1", candidate_id=candidate_id,
        rows=len(index.keys), keys_sha256=index.parquet_sha256, index_sha256=index.index_sha256,
        value_columns=["score", "observed_end", "known_at"], missing_columns=["score"], position_dtype="uint32")
    if (record != expected or digest(root / "shared_keys.parquet") != index.parquet_sha256
            or digest(root / "shared_keys.json") != index.index_sha256):
        raise ValueError("Compact candidate shared index binding differs")
    frames = []
    for name, columns in (("values", expected["value_columns"]), ("missing", ["score"])):
        frame = pd.read_parquet(target / (name + ".parquet"))
        if (list(frame) != ["row_index"] + columns or len(frame) != len(index.keys)
                or str(frame.row_index.dtype) != "uint32"
                or not np.array_equal(frame.row_index.to_numpy(), np.arange(len(index.keys), dtype=np.uint32))):
            raise ValueError("Compact candidate row order/count/schema differs")
        frames.append(pd.concat([index.keys.copy(), frame.drop(columns="row_index")], axis=1))
    metadata = json.loads((target / "definition.json").read_text(encoding="utf8")); units = metadata.pop("units")
    return FeatureBlock(frames[0], frames[1], units, metadata).validate()
