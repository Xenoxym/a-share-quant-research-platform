"""Temp-name migration preserves saved producer identity; no model calls."""
import copy
import hashlib
from pathlib import Path

import pytest

from src.alpharesearch.score_reuse import reuse_code_bundle, validate_reuse_bundle
from src.technical.artifacts import content_id
from .test_alpha_portfolio import scored_example, build, resign_scores

APP = Path(__file__).resolve().parents[1]
NAME = "src/technical/artifacts.py"


def old_producer():
    data = scored_example()
    learning = data[0]
    current = (APP / NAME).read_bytes().replace(b"\r\n", b"\n").decode()
    historic = current.replace('prefix=".j-", suffix=".tmp"', 'prefix="", suffix=""')
    assert historic != current
    sha = hashlib.sha256(historic.encode()).hexdigest()
    learning.receipt["fit_identity"]["implementation_hashes"][NAME] = sha
    learning.receipt["fit_id"] = content_id(learning.receipt["fit_identity"])
    learning.predictions["fit_id"] = learning.receipt["fit_id"]
    resign_scores(learning, membership=True)
    # Synthetic fixture represents an already checked old producer; production
    # authentication still requires the registered frozen receipt and audit.
    files = {n: dict(sha256=h, text=(APP/n).read_bytes().replace(b"\r\n", b"\n").decode())
             for n,h in learning.receipt["fit_identity"]["implementation_hashes"].items()}
    files[NAME] = dict(sha256=sha, text=historic)
    v = dict(schema="completed-learning-identity-v1", verification_mode="reuse_completed_identity_v1",
             producer_schema="registered-alpha-learn-v3", fit_id=learning.receipt["fit_id"],
             prediction_id=learning.receipt["prediction_id"], learning_receipt_id=content_id(learning.receipt),
             prediction_outputs=learning.receipt["prediction_outputs"], producer_files=files,
             additional_model_fits=0, additional_predict_calls=0, numeric_model_reproduction=False)
    identity = dict(v, identity_id=content_id(v))
    return data, identity


def test_old_tempname_bundle_preserves_every_producer_field():
    data, identity = old_producer()
    original = copy.deepcopy(identity)
    original_scores = data[0].predictions.copy(deep=True)
    portfolio = build(data)
    bundle = reuse_code_bundle(data[0], portfolio, identity)
    assert bundle["producer_identity"] == original
    assert data[0].predictions.equals(original_scores)
    assert bundle["producer_identity"]["producer_files"][NAME]["sha256"] != bundle["consumer_files"][NAME]["sha256"]
    validate_reuse_bundle(bundle, data[0], portfolio)


@pytest.mark.parametrize("mutation", [
    ('prefix=".j-"', 'prefix="anything"'),
    ('suffix=".tmp"', 'suffix=""'),
    ('prefix=".j-", ', ''),
    ('prefix=".j-", ', 'prefix=".j-", prefix=".j-", '),
    ('encoding="utf-8"', 'encoding="ascii"'),
    ('allow_nan=False', 'allow_nan=True'),
    ('indent=2', 'indent=4'),
    ('timezone.utc', 'None'),
    ('h.hexdigest()', '"changed"'),
    ('dir=path.parent', 'dir=None'),
    ('os.fsync(stream.fileno())', 'pass'),
    ('os.replace(temp, path)', 'os.rename(temp, path)'),
    ('temp.unlink(missing_ok=True)', 'pass'),
    ('stream.write(data)', 'stream.write(data + "x")'),
    ('data = json.dumps', 'tempfile.NamedTemporaryFile()\n    data = json.dumps'),
])
def test_other_consumer_semantic_change_is_rejected(mutation):
    data, identity = old_producer()
    portfolio = build(data)
    bundle = reuse_code_bundle(data[0], portfolio, identity)
    old, new = mutation
    text = bundle["consumer_files"][NAME]["text"]
    assert old in text
    text = text.replace(old, new)
    from src.alpharesearch.score_reuse import _tempname_only_compatible
    assert not _tempname_only_compatible(identity["producer_files"][NAME]["text"], text)
    bundle["consumer_files"][NAME] = dict(text=text, sha256=hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValueError):
        validate_reuse_bundle(bundle, data[0], portfolio)


def test_unbound_text_still_rejected():
    data, identity = old_producer()
    identity["producer_files"][NAME]["text"] += "\n# unbound change"
    identity["identity_id"] = content_id({k:v for k,v in identity.items() if k != "identity_id"})
    with pytest.raises(ValueError, match="content changed"):
        reuse_code_bundle(data[0], build(data), identity)


@pytest.mark.parametrize("old,new", [(('prefix="a"', 'suffix="b"'), ('prefix=".j-"', 'suffix=".tmp"')),
                                      (('prefix=".j-"', 'suffix=".tmp"'), ('prefix=""', 'suffix=""'))])
def test_unapproved_or_reverse_migration_rejected(old,new):
    from src.alpharesearch.score_reuse import _tempname_only_compatible
    template = (APP / NAME).read_text(encoding="utf8")
    def text(pair):
        return template.replace('prefix=".j-"', pair[0]).replace('suffix=".tmp"', pair[1])
    assert not _tempname_only_compatible(text(old), text(new))
