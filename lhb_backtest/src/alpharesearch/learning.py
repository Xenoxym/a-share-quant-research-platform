"""Causal, bounded in-memory learning protocol; not a durable experiment executor."""
from dataclasses import dataclass
from datetime import date
import hashlib
import importlib.metadata
import json
import platform

import numpy as np
import pandas as pd

from .assembly import FeatureAssembly
from .contracts import VintagePolicy, required_id, timestamp, timestamps
from .features.base import FeatureBlock
from .models import ModelSpec, TabularRegressor
from .registry import implementation_hashes
from ..technical.artifacts import content_id

KEYS = ["sample_id", "trade_date", "stock_code"]
CLOCKS = ["decision_at", "execution_at", "target_end_at"]
LABELS = ["target_return", "label_start", "label_end", "label_known_at", "label_observed"]
CODE = [
    "src/alpharesearch/learning.py", "src/alpharesearch/models.py",
    "src/alpharesearch/contracts.py", "src/alpharesearch/assembly.py",
    "src/alpharesearch/registry.py", "src/alpharesearch/features/base.py",
    "src/technical/artifacts.py",
]


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Canonical YYYY-MM-DD date required")
    return value


@dataclass(frozen=True)
class LearningSpec:
    document_json: str

    @classmethod
    def from_dict(cls, value):
        fields = {
            "schema", "name", "dataset_id", "universe_id", "features",
            "train_start", "train_end", "predict_start", "predict_end", "fit_cutoff",
            "target", "empty_feature_policy", "allow_weak_vintage", "weighting",
            "negative_control", "control_seed", "min_train_rows", "max_input_rows",
        }
        if not isinstance(value, dict) or set(value) != fields or value["schema"] != "causal-learning-v1":
            raise ValueError("Strict causal learning fields required")
        for key in ("name", "dataset_id", "universe_id"):
            required_id(value[key], key)
        features = value["features"]
        if not isinstance(features, list) or not features or len(features) != len(set(features)):
            raise ValueError("Unique nonempty feature list required")
        for key in features:
            required_id(key, "feature")
        for key in ("train_start", "train_end", "predict_start", "predict_end"):
            _day(value[key])
        if not (value["train_start"] <= value["train_end"] < value["predict_start"] <= value["predict_end"]):
            raise ValueError("Training must precede a nonoverlapping prediction range")
        cutoff = timestamp(value["fit_cutoff"]).isoformat()
        if value["target"] != "execution_window_return_decimal":
            raise ValueError("Only declared execution-window decimal returns supported in v1")
        if value["empty_feature_policy"] not in ("reject", "exclude_train_empty"):
            raise ValueError("Explicit train-empty feature policy required")
        if type(value["allow_weak_vintage"]) is not bool:
            raise ValueError("Weak vintage opt-in must be boolean")
        if value["weighting"] not in ("uniform_rows", "equal_decision_date"):
            raise ValueError("Unsupported training weighting")
        if value["negative_control"] not in ("none", "within_date_label_permutation"):
            raise ValueError("Unsupported negative control")
        _integer(value["control_seed"], 0, 2**32 - 1, "control_seed")
        _integer(value["min_train_rows"], 2, 10000000, "min_train_rows")
        _integer(value["max_input_rows"], 2, 10000000, "max_input_rows")
        if value["min_train_rows"] > value["max_input_rows"]:
            raise ValueError("Training minimum exceeds input row cap")
        return cls(json.dumps(dict(value, fit_cutoff=cutoff), sort_keys=True,
                              separators=(",", ":"), allow_nan=False))

    def to_dict(self):
        return json.loads(self.document_json)

    @property
    def protocol_id(self):
        value = self.to_dict()
        value.pop("name")
        return content_id(value)


def learning_spec(*, dataset_id, universe_id, features, train, predict, fit_cutoff,
                  name="causal_learning", empty_feature_policy="reject",
                  allow_weak_vintage=False, weighting="uniform_rows",
                  negative_control="none", control_seed=11,
                  min_train_rows=2, max_input_rows=2000000):
    return LearningSpec.from_dict(dict(
        schema="causal-learning-v1", name=name, dataset_id=dataset_id,
        universe_id=universe_id, features=list(features),
        train_start=train[0], train_end=train[1], predict_start=predict[0],
        predict_end=predict[1], fit_cutoff=fit_cutoff,
        target="execution_window_return_decimal", empty_feature_policy=empty_feature_policy,
        allow_weak_vintage=allow_weak_vintage, weighting=weighting,
        negative_control=negative_control, control_seed=control_seed,
        min_train_rows=min_train_rows, max_input_rows=max_input_rows,
    ))


def _keys(frame, name):
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any() or not set(KEYS) <= set(frame):
        raise ValueError(f"{name} requires unique named key columns")
    if (frame[KEYS].isna().any().any() or frame.sample_id.duplicated().any()
            or frame.duplicated(["trade_date", "stock_code"]).any()):
        raise ValueError(f"{name} requires unique non-null stock/date and sample keys")
    if not frame.sample_id.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Sample IDs must be nonempty strings")
    if not frame.stock_code.map(lambda x: isinstance(x, str)).all() or not frame.stock_code.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)").all():
        raise ValueError("Exchange-qualified stock codes required")
    for value in frame.trade_date:
        _day(value)


def _fingerprint(frame):
    """Scoped content identity; environment version is recorded separately."""
    return dict(columns=list(frame.columns), dtypes=[str(d) for d in frame.dtypes],
                rows=len(frame), sha256=hashlib.sha256(
                    pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
                ).hexdigest())


def _environment():
    return {"python": platform.python_version(), "packages": {
        key: importlib.metadata.version(key) for key in
        ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")
    }}


def _prepare(spec, model, decisions, assembly, labels):
    cfg = spec.to_dict()
    if not isinstance(assembly, FeatureAssembly):
        raise ValueError("Registered FeatureAssembly required")
    block = assembly.block
    if not isinstance(block, FeatureBlock) or block.metadata.get("schema") != "assembled-feature-matrix-v1":
        raise ValueError("An assembled feature matrix is required")
    if block.metadata.get("universe_id") != cfg["universe_id"]:
        raise ValueError("Declared universe differs from feature assembly")
    if (block.metadata.get("registry_version") != assembly.registry.version_id
            or list(assembly.source_bindings) != block.metadata.get("source_bindings")):
        raise ValueError("Assembly registry/source binding identity differs")
    for frame in (decisions, block.values, block.missing, labels):
        if not isinstance(frame, pd.DataFrame) or len(frame) > cfg["max_input_rows"]:
            raise ValueError("Declared input row cap exceeded or invalid table")
    _keys(decisions, "Decisions")
    _keys(block.values, "Features")
    _keys(block.missing, "Missing reasons")
    _keys(labels, "Labels")
    if set(decisions) != set(KEYS + CLOCKS) or set(labels) != set(KEYS + LABELS):
        raise ValueError("Strict decision and label table fields required")
    if block.metadata.get("key_columns") != KEYS:
        raise ValueError("Feature key domain differs from decision sample keys")

    p = decisions.loc[
        decisions.trade_date.between(cfg["train_start"], cfg["train_end"])
        | decisions.trade_date.between(cfg["predict_start"], cfg["predict_end"])
    ].copy().sort_values(KEYS[1:] + KEYS[:1]).reset_index(drop=True)
    for key in CLOCKS:
        p[key] = timestamps(p[key]).astype("datetime64[ns, UTC]")
    if not p.decision_at.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d").eq(p.trade_date).all():
        raise ValueError("Decision clock must match its Shanghai trade date")
    if (~p.execution_at.gt(p.decision_at) | ~p.target_end_at.gt(p.execution_at)).any():
        raise ValueError("Execution must follow decision; target end must follow entry")
    train = p.trade_date.between(cfg["train_start"], cfg["train_end"])
    pred = p.trade_date.between(cfg["predict_start"], cfg["predict_end"])
    if not train.any() or not pred.any():
        raise ValueError("Both training and prediction decisions are required")
    cutoff = pd.Timestamp(cfg["fit_cutoff"])
    if (p.loc[train, "decision_at"] > cutoff).any() or (p.loc[pred, "decision_at"] <= cutoff).any():
        raise ValueError("Fit cutoff must follow all training decisions and precede prediction decisions")

    # Validate only the scoped rows/columns; later values cannot choose training inputs.
    names = cfg["features"]
    limits = model.to_dict()
    if len(names) > limits["max_features"] or len(p) * len(names) > limits["max_matrix_cells"]:
        raise ValueError("Declared scoped feature join/matrix budget exceeded")
    if not set(names) <= set(block.units):
        raise ValueError("Selected feature was not supplied by assembly")
    definitions = {}
    for key in names:
        ids = {definition_id for binding in assembly.source_bindings
               for definition_id in binding["definition_ids"]
               if any(d.definition_id == definition_id and d.key == key
                      for d in assembly.registry.definitions)}
        if len(ids) != 1:
            raise ValueError("Select an unambiguous bound feature definition")
        definition = assembly.registry.resolve(key, ids.pop())
        if definition.unit != block.units[key]:
            raise ValueError("Registered feature unit differs from assembled value")
        definitions[key] = definition.definition_id
    scoped_values = p[KEYS].merge(
        block.values[KEYS + names + ["observed_end", "known_at"]],
        on=KEYS, how="left", validate="one_to_one", indicator=True, sort=False)
    scoped_missing = p[KEYS].merge(block.missing[KEYS + names], on=KEYS,
                                  how="left", validate="one_to_one", sort=False)
    if not scoped_values.pop("_merge").eq("both").all():
        raise ValueError("A prediction/training decision lacks its assembled feature row")
    scoped = FeatureBlock(scoped_values, scoped_missing, {key: block.units[key] for key in names},
                          {**block.metadata, "key_columns": KEYS}).validate()
    for key in ("observed_end", "known_at"):
        scoped.values[key] = timestamps(scoped.values[key]).astype("datetime64[ns, UTC]")
        if scoped.values[key].gt(p.decision_at).any():
            raise ValueError("Feature input was not known by its decision cutoff")
    if block.metadata["vintage"] not in (VintagePolicy.MARKET.value, VintagePolicy.HISTORICAL.value) and not cfg["allow_weak_vintage"]:
        raise ValueError("Weak vintage requires explicit learning protocol opt-in")
    for key in names:
        if (not pd.api.types.is_numeric_dtype(scoped.values[key].dtype)
                or pd.api.types.is_complex_dtype(scoped.values[key].dtype)):
            raise ValueError("Features must have numeric, noncomplex dtypes")

    # A genuinely absent label is allowed; a supplied label with conflicting keys is not.
    train_keys = p.loc[train, KEYS]
    joined_keys = train_keys.merge(labels[KEYS], on=["trade_date", "stock_code"],
                                  suffixes=("_decision", "_label"), validate="one_to_one")
    if not joined_keys.sample_id_decision.eq(joined_keys.sample_id_label).all():
        raise ValueError("Training label sample identity conflicts with stock/date keys")
    joined_ids = train_keys.merge(labels[KEYS], on="sample_id",
                                 suffixes=("_decision", "_label"), validate="one_to_one")
    if not (joined_ids.trade_date_decision.eq(joined_ids.trade_date_label)
            & joined_ids.stock_code_decision.eq(joined_ids.stock_code_label)).all():
        raise ValueError("Training label stock/date keys conflict with sample identity")

    # Prediction labels are never inspected or used to select the prediction pool.
    t = p.loc[train].merge(labels, on=KEYS, how="left", validate="one_to_one",
                          indicator=True, sort=False)
    supplied = t._merge.eq("both")
    observed = supplied & t.label_observed.eq(True)
    if supplied.any():
        supplied_observed = t.loc[supplied, "label_observed"]
        if not supplied_observed.map(lambda x: isinstance(x, (bool, np.bool_))).all():
            raise ValueError("Observed outcome flag must be boolean")
        for key in ("label_start", "label_end"):
            t[key] = t[key].astype(object)
            t.loc[supplied, key] = timestamps(t.loc[supplied, key]).astype(object)
        start = pd.to_datetime(t.label_start, utc=True)
        end = pd.to_datetime(t.label_end, utc=True)
        if (supplied & (~start.eq(t.execution_at) | ~end.eq(t.target_end_at))).any():
            raise ValueError("Label interval must exactly match declared entry/target end")
        if not t.loc[supplied & ~observed, "label_known_at"].isna().all():
            raise ValueError("Unobserved outcome cannot have a known timestamp")
        if not t.loc[supplied & ~observed, "target_return"].isna().all():
            raise ValueError("Unobserved outcome cannot carry a numeric target")
    known = pd.Series(pd.NaT, index=t.index, dtype="datetime64[ns, UTC]")
    if observed.any():
        known.loc[observed] = timestamps(t.loc[observed, "label_known_at"]).astype("datetime64[ns, UTC]")
        if (known.loc[observed] < pd.to_datetime(t.loc[observed, "label_end"], utc=True)).any():
            raise ValueError("Outcome cannot be known before label end")
        if not t.loc[observed, "target_return"].map(
                lambda x: isinstance(x, (int, float, np.integer, np.floating))
                and not isinstance(x, (bool, np.bool_))).all():
            raise ValueError("Training targets must be numeric")
        if not np.isfinite(t.loc[observed, "target_return"].to_numpy(dtype=float)).all():
            raise ValueError("Observed training targets must be finite")
    mature = observed & known.le(cutoff)
    if int(mature.sum()) < cfg["min_train_rows"]:
        raise ValueError("Insufficient mature training rows")
    t = t.loc[mature].drop(columns="_merge").copy().reset_index(drop=True)
    t["label_known_at"] = known.loc[mature].reset_index(drop=True)
    x_train = t[KEYS].merge(scoped.values[KEYS + names], on=KEYS, validate="one_to_one", sort=False)
    x_pred = p.loc[pred, KEYS].merge(scoped.values[KEYS + names], on=KEYS, validate="one_to_one", sort=False)
    excluded = [key for key in names if x_train[key].isna().all()]
    if excluded and cfg["empty_feature_policy"] == "reject":
        raise ValueError("Entirely missing training feature requires explicit exclusion policy")
    retained = [key for key in names if key not in excluded]
    if not retained:
        raise ValueError("No usable training features remain")
    m = model.to_dict()
    if (len(t) > m["max_train_rows"] or len(retained) > m["max_features"]
            or max(len(t), len(x_pred)) * len(retained) > m["max_matrix_cells"]):
        raise ValueError("Declared model matrix budget exceeded")
    index = pd.MultiIndex.from_frame(t[KEYS])
    X = pd.DataFrame(x_train[retained].to_numpy(dtype=float), columns=retained, index=index)
    P = pd.DataFrame(x_pred[retained].to_numpy(dtype=float), columns=retained,
                     index=pd.MultiIndex.from_frame(x_pred[KEYS]))
    y = pd.Series(t.target_return.to_numpy(dtype=float), index=index)
    if cfg["negative_control"] == "within_date_label_permutation":
        rng = np.random.default_rng(cfg["control_seed"])
        for positions in t.groupby("trade_date", sort=True).indices.values():
            y.iloc[positions] = rng.permutation(y.iloc[positions].to_numpy())
    weights = None
    if cfg["weighting"] == "equal_decision_date":
        count = t.groupby("trade_date").trade_date.transform("size").to_numpy(dtype=float)
        values = 1 / count
        weights = pd.Series(values / values.mean(), index=index)

    code, environment = dict(implementation_hashes(CODE)), _environment()
    settings = {key: cfg[key] for key in (
        "dataset_id", "universe_id", "features", "train_start", "train_end", "fit_cutoff",
        "target", "empty_feature_policy", "allow_weak_vintage", "weighting",
        "negative_control", "control_seed", "min_train_rows", "max_input_rows",
    )}
    train_content = t[KEYS + CLOCKS + ["label_start", "label_end", "label_known_at"]].copy()
    train_content[retained] = X.to_numpy()
    train_content["target_return_used"] = y.to_numpy()
    train_content["weight_used"] = 1.0 if weights is None else weights.to_numpy()
    reason_content = t[KEYS].merge(scoped.missing[KEYS + names], on=KEYS,
                                  validate="one_to_one", sort=False)
    fit_identity = dict(
        schema="causal-fit-identity-v1", settings=settings, model_id=model.model_id,
        feature_definitions=definitions, feature_units={key: block.units[key] for key in names},
        feature_vintage=block.metadata["vintage"], excluded_train_empty=excluded,
        training_content=_fingerprint(train_content), training_missing=_fingerprint(reason_content),
        training_feature_clocks=_fingerprint(t[KEYS].merge(
            scoped.values[KEYS + ["observed_end", "known_at"]], on=KEYS,
            validate="one_to_one", sort=False)),
        implementation_hashes=code, environment=environment,
    )
    pred_content = p.loc[pred, KEYS + CLOCKS].reset_index(drop=True).copy()
    pred_content[retained] = P.to_numpy()
    pred_content = pred_content.merge(scoped.values[KEYS + ["observed_end", "known_at"]],
                                      on=KEYS, validate="one_to_one", sort=False)
    receipt = dict(
        schema="causal-learning-receipt-v2", protocol_id=spec.protocol_id,
        fit_id=content_id(fit_identity), fit_identity=fit_identity,
        model_id=model.model_id, retained_features=retained, excluded_train_empty=excluded,
        training_decisions=int(train.sum()), mature_training_rows=len(t),
        omitted_or_unobserved_labels=int((~observed).sum()),
        observed_but_immature_labels=int((observed & ~mature).sum()),
        latest_training_label_known_at=t.label_known_at.max().isoformat(),
        prediction_rows=len(P), prediction_inputs=_fingerprint(pred_content),
        prediction_membership=_fingerprint(p.loc[pred, KEYS + CLOCKS].reset_index(drop=True)),
        prediction_missing=_fingerprint(x_pred[KEYS].merge(
            scoped.missing[KEYS + retained], on=KEYS, validate="one_to_one", sort=False)),
        full_assembly_version=block.metadata["version_id"],
        full_assembly_source_bindings=json.loads(json.dumps(
            list(assembly.source_bindings), allow_nan=False)),
        prediction_selection="declared decisions; prediction labels ignored",
        input_artifact_verification="caller responsibility; metadata is not authentication",
        historical_execution_certified=False, account_results=False,
        scope="in-memory learning only; no durable registration or account execution",
    )
    receipt["prediction_id"] = content_id({
        "fit_id": receipt["fit_id"], "protocol_id": spec.protocol_id,
        "prediction_inputs": receipt["prediction_inputs"],
        "prediction_missing": receipt["prediction_missing"],
        "prediction_membership": receipt["prediction_membership"],
    })
    return X, y, weights, P, p.loc[pred].reset_index(drop=True), receipt


@dataclass
class LearningResult:
    predictions: pd.DataFrame
    receipt: dict


class LearningRunner:
    """Finite calls and new fit intents; failed intents are retained without retry.

    This ledger lives in one process. Registered jobs must persist it and artifacts;
    it is not a cross-process quota or crash-recovery mechanism.
    """

    def __init__(self, *, max_calls, max_fit_intents):
        _integer(max_calls, 1, 10000, "max_calls")
        _integer(max_fit_intents, 1, 1000, "max_fit_intents")
        self.max_calls, self.max_fit_intents = max_calls, max_fit_intents
        self.calls = 0
        self.fit_intents = 0
        self.ledger = []
        self._fits = {}

    def run(self, spec, model, decisions, assembly, labels):
        if self.calls >= self.max_calls:
            raise ValueError("Learning call budget exhausted")
        self.calls += 1
        event = {"call": self.calls, "state": "preparing", "fit_attempts": 0}
        self.ledger.append(event)
        try:
            if not isinstance(spec, LearningSpec) or not isinstance(model, ModelSpec):
                raise ValueError("Validated LearningSpec and ModelSpec required")
            spec = LearningSpec.from_dict(spec.to_dict())
            model = ModelSpec.from_dict(model.to_dict())
            X, y, weights, P, decision_rows, receipt = _prepare(
                spec, model, decisions, assembly, labels)
            fit_id = receipt["fit_id"]
            event["fit_id"] = fit_id
            cached = self._fits.get(fit_id)
            if cached is not None and cached["state"] == "failed":
                raise ValueError("Previous fit intent failed; implicit retry is forbidden")
            reused = cached is not None
            if cached is None:
                if self.fit_intents >= self.max_fit_intents:
                    raise ValueError("New fit intent budget exhausted")
                self.fit_intents += 1
                cached = {"state": "fitting", "adapter": None}
                self._fits[fit_id] = cached
                event["new_fit_intent"] = self.fit_intents
                try:
                    cached["adapter"] = TabularRegressor(model)
                    cached["adapter"].fit(X, y, sample_weight=weights)
                except Exception:
                    cached["state"] = "failed"
                    event["fit_attempts"] = 0 if cached["adapter"] is None else cached["adapter"].fit_attempts
                    raise
                cached["state"] = "fitted"
                event["fit_attempts"] = cached["adapter"].fit_attempts
            score = cached["adapter"].predict(P)
            output = decision_rows.copy()
            output["score"] = score.to_numpy()
            output["fit_id"] = fit_id
            output["prediction_id"] = receipt["prediction_id"]
            receipt.update(
                fit_reused=reused, call=self.calls, fit_intents_used=self.fit_intents,
                actual_fit_attempts_this_call=event["fit_attempts"],
                prediction_outputs=_fingerprint(output),
                fitted_model=cached["adapter"].metadata(),
            )
            event["state"] = "succeeded"
            json.dumps(receipt, allow_nan=False)
            return LearningResult(output, receipt)
        except Exception as exc:
            event["state"] = "failed"
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise
