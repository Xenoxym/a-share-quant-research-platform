"""Versioned old-learning/new-account source roles for saved-score reuse."""
import hashlib
from pathlib import Path
import re
from .audit import _require
from .registry import implementation_hashes
from ..technical.artifacts import content_id

MODE = "reuse_completed_identity_v1"
APP = Path(__file__).resolve().parents[2]
# These shared data semantics must match. Learning/model implementations may
# differ: only their unchanged, previously audited saved outputs are consumed.
SHARED = {"src/alpharesearch/contracts.py", "src/technical/artifacts.py"}


def _files(files):
    _require(isinstance(files, dict) and files, "Source role requires nonempty files")
    for name, value in files.items():
        _require(isinstance(name, str) and name.startswith("src/") and ".." not in Path(name).parts
                 and "\\" not in name and isinstance(value, dict) and set(value) == {"sha256", "text"}
                 and isinstance(value["text"], str) and isinstance(value["sha256"], str)
                 and re.fullmatch(r"[a-f0-9]{64}", value["sha256"]), "Strict source role file required")
        _require(hashlib.sha256(value["text"].encode("utf-8")).hexdigest() == value["sha256"],
                 "Archived source role content changed")


def _tempname_only_compatible(producer_text, consumer_text):
    """One directed writer-name migration; all other module semantics match.

    This never replaces the declared source hashes. Both original texts are
    still independently checked against their archived identities.
    """
    import ast

    def normalized(text, expected):
        tree = ast.parse(text)
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "write_json"]
        if len(funcs) != 1:
            return None
        calls = [n for n in ast.walk(funcs[0]) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id == "tempfile" and n.func.attr == "NamedTemporaryFile"]
        if len(calls) != 1:
            return None
        options = [k for k in calls[0].keywords if k.arg in ("prefix", "suffix")]
        if [k.arg for k in options] != ["prefix", "suffix"]:
            return None
        for option, value in zip(options, expected):
            if not (isinstance(option.value, ast.Constant) and type(option.value.value) is str
                    and option.value.value == value):
                return None
            option.value.value = ""
        return ast.dump(tree, include_attributes=False)

    try:
        old = normalized(producer_text, ("", ""))
        new = normalized(consumer_text, (".j-", ".tmp"))
        return old is not None and old == new
    except (SyntaxError, ValueError, TypeError):
        return False


def _shared_compatible(name, producer, consumer):
    return producer["sha256"] == consumer["sha256"] or (
        name == "src/technical/artifacts.py"
        and _tempname_only_compatible(producer["text"], consumer["text"]))


def validate_reuse_identity(identity, learning):
    _require(isinstance(identity, dict) and identity.get("schema") == "completed-learning-identity-v1"
             and identity.get("verification_mode") == MODE
             and identity.get("producer_schema") in {"registered-alpha-learn-v2", "registered-alpha-learn-v3"}
             and identity.get("identity_id") == content_id({k:v for k,v in identity.items() if k != "identity_id"})
             and type(identity.get("additional_model_fits")) is int and identity["additional_model_fits"] == 0
             and type(identity.get("additional_predict_calls")) is int and identity["additional_predict_calls"] == 0
             and identity.get("numeric_model_reproduction") is False,
             "Completed reuse identity differs")
    _require(identity["fit_id"] == learning.receipt["fit_id"]
             and identity["prediction_id"] == learning.receipt["prediction_id"]
             and identity["learning_receipt_id"] == content_id(learning.receipt)
             and identity["prediction_outputs"] == learning.receipt["prediction_outputs"], "Reuse model/prediction identity differs")
    _files(identity["producer_files"])
    expected = learning.receipt["fit_identity"]["implementation_hashes"]
    _require(set(identity["producer_files"]) == set(expected)
             and all(identity["producer_files"][n]["sha256"] == h for n,h in expected.items()),
             "Reuse source differs from declared producer implementation")


def reuse_code_bundle(learning, portfolio, identity, *, holding_policy=None):
    validate_reuse_identity(identity, learning)
    required = dict(portfolio.receipt["implementation_hashes"])
    required.update(implementation_hashes(["src/alpharesearch/account.py", "src/alpharesearch/audit.py",
                                          "src/alpharesearch/score_reuse.py", *sorted(SHARED)]))
    if holding_policy is not None:
        required.update(implementation_hashes(['src/alpharesearch/holding_policy.py',
                                              'src/alpharesearch/holding_policy_audit.py']))
    current = dict(implementation_hashes(list(required)))
    _require(required == current, "Current consumer source differs from declared portfolio")
    files = {n:dict(sha256=h,text=(APP/n).read_bytes().replace(b"\r\n",b"\n").decode("utf-8")) for n,h in current.items()}
    bundle = dict(schema="score-account-code-bundle-v2", verification_mode=MODE,
                  producer_identity=identity, consumer_files=files)
    if holding_policy is not None:
        bundle.update(schema='score-account-code-bundle-v3', holding_policy=holding_policy.to_dict())
    validate_reuse_bundle(bundle, learning, portfolio)
    return bundle


def validate_reuse_bundle(bundle, learning, portfolio):
    dynamic = isinstance(bundle,dict) and bundle.get('schema') == 'score-account-code-bundle-v3'
    fields = {'schema','verification_mode','producer_identity','consumer_files'} | ({'holding_policy'} if dynamic else set())
    _require(isinstance(bundle, dict) and set(bundle) == fields
             and bundle['schema'] in {'score-account-code-bundle-v2','score-account-code-bundle-v3'} and bundle['verification_mode'] == MODE,
             "Strict versioned score reuse bundle required")
    identity = bundle["producer_identity"]
    validate_reuse_identity(identity, learning)
    files = bundle["consumer_files"]
    _files(files)
    expected = portfolio.receipt["implementation_hashes"]
    required = set(expected) | {"src/alpharesearch/account.py", "src/alpharesearch/audit.py", "src/alpharesearch/score_reuse.py"} | SHARED
    if dynamic:
        from .holding_policy import ActualHoldingPolicy
        policy = ActualHoldingPolicy.from_dict(bundle['holding_policy'])
        cfg=portfolio.receipt['spec']
        _require(policy.max_positions==cfg['top_k'] and policy.allocation==cfg['allocation'], 'Archived dynamic policy differs from portfolio limits')
        required |= {'src/alpharesearch/holding_policy.py','src/alpharesearch/holding_policy_audit.py'}
    _require(set(files) == required and all(files[n]["sha256"] == h for n,h in expected.items()),
             "Archived consumer source differs from portfolio declaration")
    # Keep input serialization/time/value definitions compatible while permitting
    # independent producer and consumer algorithms. No stored receipt is rewritten.
    _require(SHARED <= set(identity["producer_files"]) & set(files)
             and all(_shared_compatible(n, identity["producer_files"][n], files[n]) for n in SHARED),
             "Producer/consumer shared data contract differs")
    return files
