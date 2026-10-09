"""Bounded local sklearn checkpoints. Not a loader for downloaded models."""
from pathlib import Path
import hashlib
import json
import re
import joblib
from .models import ModelSpec, TabularRegressor
from .contracts import timestamp
from .registry import implementation_hashes

FILES = {"model.joblib", "manifest.json"}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _limit(value):
    if type(value) is not int or not 1024 <= value <= 50000000:
        raise ValueError("Checkpoint budget must be1024..50000000 bytes")
    return value


class _BoundedWriter:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.bytes = stream, limit, 0
    def write(self, data):
        size = memoryview(data).nbytes
        if self.bytes + size > self.limit:
            raise ValueError("Serialized model exceeds byte budget")
        self.bytes += size
        return self.stream.write(data)
    def tell(self):
        return self.stream.tell()
    def flush(self):
        return self.stream.flush()


def save_checkpoint(adapter, directory, training_context, *, max_bytes=10000000):
    """Caller supplies training identities; model metadata cannot authenticate them.

    The fitted pipeline includes learned imputation/scaling/model parameters. This
    does not save training data or predictions; the future worker must save those.
    Partial failures remain in the newlycreated directory and cannot be overwritten.
    """
    limit = _limit(max_bytes)
    if not isinstance(adapter, TabularRegressor) or adapter.state != "fitted":
        raise ValueError("Successfully fitted adapter required")
    fields = {"fit_id", "fit_cutoff", "training_inputs_sha256", "training_targets_sha256", "training_weights_sha256", "target_spec_id"}
    if not isinstance(training_context, dict) or set(training_context) != fields:
        raise ValueError("Explicit training context identities required")
    context = dict(training_context)
    timestamp(context["fit_cutoff"])
    for field in fields - {"fit_cutoff"}:
        if not isinstance(context[field], str) or not re.fullmatch("[0-9a-f]{64}", context[field]):
            raise ValueError("External training identity must be SHA256")
    meta = adapter.metadata()
    if dict(meta["implementation_hashes"]) != dict(implementation_hashes(["src/alpharesearch/models.py"])):
        raise ValueError("Current model adapter code differs from fitted adapter")
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Checkpoint directory cannot be a symlink")
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "model.joblib").open("xb") as f:
        joblib.dump(adapter._model, _BoundedWriter(f, limit), compress=0, protocol=5)
    manifest = dict(schema="local-tabular-checkpoint-v1", metadata=meta,
                    training_context=context, model_sha256=_sha(directory / "model.joblib"),
                    model_bytes=(directory / "model.joblib").stat().st_size,
                    trust_scope="Caller-generated localjoblib only; hashisnot a sandbox",
                    row_weight_values_saved=False, training_predictions_saved=False,
                    refit_on_load=False)
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) + manifest["model_bytes"] > limit:
        raise ValueError("Checkpoint total byte budget exceeded")
    with (directory / "manifest.json").open("xb") as f:
        f.write(raw)
    return dict(manifest_sha256=_sha(directory / "manifest.json"), model_sha256=manifest["model_sha256"], model_bytes=manifest["model_bytes"])


def load_checkpoint(directory, *, expected_manifest_sha256, trusted_local=False,
                    max_bytes=10000000):
    """Require expected identity from a caller-held receipt, before deserialization.

    joblib canexecute Python; trustisexplicit, hashes authenticatebytes only. Read
    bounded complete files into owned memory aftervalidation toavoid laterfile swaps.
    """
    limit = _limit(max_bytes)
    if trusted_local is not True:
        raise ValueError("Only explicitly trusted caller-generated localcheckpoints")
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch("[0-9a-f]{64}", expected_manifest_sha256):
        raise ValueError("External expected manifest SHA256 required")
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir() or {p.name for p in directory.iterdir()} != FILES:
        raise ValueError("Strict complete checkpoint directory required")
    for name in FILES:
        f = directory / name
        if f.is_symlink() or not f.is_file():
            raise ValueError("Checkpoint files must be regular files")
    if sum((directory / n).stat().st_size for n in FILES) > limit:
        raise ValueError("Checkpoint input byte budget exceeded")
    with (directory / "manifest.json").open("rb") as f:
        raw = f.read(limit + 1)
    if len(raw) > limit or hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("Manifest bytes differ from externalexpectedidentity")
    doc = json.loads(raw)
    if doc.get("schema") != "local-tabular-checkpoint-v1":
        raise ValueError("Unsupported checkpoint schema")
    meta = doc["metadata"]; spec = ModelSpec.from_dict(meta["spec"])
    if meta.get("schema") != "fitted-tabular-model-v1" or meta["model_id"] != spec.model_id or meta["fit_attempts"] != 1 or meta["account_results"] is not False:
        raise ValueError("Fitted checkpoint metadata inconsistent")
    expected = dict(implementation_hashes(["src/alpharesearch/models.py"]))
    if dict(meta["implementation_hashes"]) != expected:
        raise ValueError("Loading adapter code differs from fitted code")
    adapter = TabularRegressor(spec)
    # Dependency compatibility is exact; loadingcannot silentlyupgrade a model.
    from .learning import _environment
    if meta["environment"] != _environment():
        raise ValueError("Model dependency versions differ")
    budget = limit - len(raw)
    with (directory / "model.joblib").open("rb") as f:
        payload = f.read(budget + 1)
    if len(payload) > budget or len(payload) != doc["model_bytes"] or hashlib.sha256(payload).hexdigest() != doc["model_sha256"]:
        raise ValueError("Model bytes differ from frozenmanifest")
    import io
    model = joblib.load(io.BytesIO(payload))
    if model.named_steps["regressor"].n_features_in_ != meta["transformed_features"] or meta["transformed_features"] != 2 * len(meta["feature_names"]):
        raise ValueError("Fitted feature width differs")
    adapter._model, adapter._metadata = model, meta
    adapter.feature_names = tuple(meta["feature_names"])
    adapter.state, adapter.initialization_attempts = "fitted", 0
    adapter.fit_attempts, adapter.successful_fits = 0, 0
    return adapter
