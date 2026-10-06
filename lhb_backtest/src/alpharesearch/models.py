"""Finite tabular model adapters (E15), separate from time and account protocols.

Only explicitly supplied training rows fit preprocessing. This layer cannot infer
whether a caller supplied causal features or mature labels; E16 checks that.
"""
from dataclasses import dataclass
import importlib.metadata
import json
import math
import platform
import warnings

import numpy as np
import pandas as pd

from .contracts import required_id
from .registry import implementation_hashes
from ..technical.artifacts import content_id


DEFAULTS = {
    "ridge": {"alpha": 10.0, "max_iter": 2000, "tol": 1e-6},
    "elastic_net": {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 2000, "tol": 1e-5},
    "huber": {"alpha": 0.0001, "epsilon": 1.35, "max_iter": 200, "tol": 1e-5},
    "random_forest": {
        "n_estimators": 64, "max_depth": 6, "min_samples_leaf": 20, "max_features": 1.0,
    },
    "extra_trees": {
        "n_estimators": 64, "max_depth": 6, "min_samples_leaf": 20, "max_features": 1.0,
    },
    "hist_gbdt": {
        "max_iter": 80, "max_leaf_nodes": 7, "max_depth": 6, "min_samples_leaf": 20,
        "learning_rate": 0.05, "l2_regularization": 10.0,
    },
}
LINEAR = frozenset({"ridge", "elastic_net", "huber"})


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _number(value, low, high, name, *, lower_inclusive=True):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value > high or (value < low if lower_inclusive else value <= low):
        raise ValueError(f"{name} outside its finite parameter range")


@dataclass(frozen=True)
class ModelSpec:
    """Canonical configuration; a display name does not change model identity."""
    document_json: str

    @classmethod
    def from_dict(cls, value):
        keys = {
            "schema", "name", "family", "params", "seed", "threads",
            "max_train_rows", "max_features", "max_matrix_cells",
        }
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError("Strict tabular model fields required")
        if value["schema"] != "tabular-model-v1":
            raise ValueError("Unsupported model schema")
        required_id(value["name"], "model name")
        family = value["family"]
        if not isinstance(family, str) or family not in DEFAULTS:
            raise ValueError("Unsupported model family; no arbitrary estimator code")
        supplied = value["params"]
        if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS[family]):
            raise ValueError("Unknown model parameter")
        params = {**DEFAULTS[family], **supplied}
        for key, item in params.items():
            if key in {"max_iter", "n_estimators", "min_samples_leaf", "max_depth", "max_leaf_nodes"}:
                bounds = {
                    "max_iter": (1, 1000 if family == "hist_gbdt" else 10000),
                    "n_estimators": (1, 512), "min_samples_leaf": (1, 1000000),
                    "max_depth": (1, 32), "max_leaf_nodes": (2, 64),
                }
                _integer(item, *bounds[key], key)
            elif key in {"alpha", "l2_regularization"}:
                _number(item, 0, 1000000, key)
            elif key == "l1_ratio":
                _number(item, 0, 1, key)
            elif key == "epsilon":
                _number(item, 1, 10, key)
            elif key == "max_features":
                _number(item, 0, 1, key, lower_inclusive=False)
            elif key == "learning_rate":
                _number(item, 0, 1, key, lower_inclusive=False)
            elif key == "tol":
                _number(item, 0, 1, key, lower_inclusive=False)
        for key in params:
            if key not in {"max_iter", "n_estimators", "min_samples_leaf", "max_depth", "max_leaf_nodes"}:
                params[key] = float(params[key])
        _integer(value["seed"], 0, 2**32 - 1, "seed")
        _integer(value["threads"], 1, 2, "threads")
        _integer(value["max_train_rows"], 2, 10000000, "max_train_rows")
        _integer(value["max_features"], 1, 4096, "max_features")
        _integer(value["max_matrix_cells"], 1, 100000000, "max_matrix_cells")
        doc = {**value, "params": params}
        return cls(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False))

    def to_dict(self):
        return json.loads(self.document_json)

    @property
    def model_id(self):
        doc = self.to_dict()
        doc.pop("name")
        return content_id(doc)


def model_spec(family, *, name=None, params=None, seed=11, threads=2,
               max_train_rows=2000000, max_features=512, max_matrix_cells=32000000):
    """Convenience constructor; explicit serialized spec is used in registration."""
    return ModelSpec.from_dict(dict(
        schema="tabular-model-v1", name=name or family, family=family,
        params={} if params is None else params, seed=seed, threads=threads,
        max_train_rows=max_train_rows, max_features=max_features,
        max_matrix_cells=max_matrix_cells,
    ))


def _pipeline(spec):
    # Keep research dependencies optional for users of the account-only package.
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import MissingIndicator, SimpleImputer
    from sklearn.linear_model import ElasticNet, HuberRegressor, Ridge
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = spec.to_dict()
    family, params, seed = cfg["family"], cfg["params"], cfg["seed"]
    if family == "ridge":
        estimator = Ridge(**params, solver="lsqr", fit_intercept=True)
    elif family == "elastic_net":
        estimator = ElasticNet(**params, fit_intercept=True, selection="cyclic",
                               random_state=seed)
    elif family == "huber":
        estimator = HuberRegressor(**params, fit_intercept=True)
    elif family in {"random_forest", "extra_trees"}:
        factory = RandomForestRegressor if family == "random_forest" else ExtraTreesRegressor
        estimator = factory(**params, bootstrap=family == "random_forest",
                            random_state=seed, n_jobs=1, warm_start=False, oob_score=False)
    else:
        estimator = HistGradientBoostingRegressor(
            **params, random_state=seed, early_stopping=False,
            warm_start=False, categorical_features=None,
        )
    # Every input has a missing flag, even if it is never missing in training.
    # A fixed 2D width avoids using prediction missingness to define new columns.
    inputs = FeatureUnion([
        ("values", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("missing", MissingIndicator(features="all", error_on_new=False)),
    ], n_jobs=1)
    steps = [("inputs", inputs)]
    if family in LINEAR:
        steps.append(("scale", StandardScaler()))
    steps.append(("regressor", estimator))
    return Pipeline(steps)


class TabularRegressor:
    """One adapter = at most one sklearn fit attempt; no implicit retries/refits."""

    def __init__(self, spec):
        if not isinstance(spec, ModelSpec):
            raise ValueError("An explicitly validated ModelSpec is required")
        # Validate even a directly constructed dataclass.
        self.spec = ModelSpec.from_dict(spec.to_dict())
        self.state = "created"
        self.initialization_attempts = 0
        self.fit_attempts = 0
        self.successful_fits = 0
        self.feature_names = ()
        self._model = None
        self._metadata = None
        self.error = None

    def _matrix(self, X, *, training):
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("Nonempty named numeric DataFrame required")
        if X.columns.duplicated().any() or not X.index.is_unique or any(pd.isna(X.index.get_level_values(i)).any() for i in range(X.index.nlevels)):
            raise ValueError("Unique feature columns and non-null row index required")
        if any(not isinstance(c, str) or not c.strip() for c in X.columns):
            raise ValueError("Feature names must be nonempty strings")
        cfg = self.spec.to_dict()
        if len(X.columns) > cfg["max_features"] or X.size > cfg["max_matrix_cells"]:
            raise ValueError("Declared feature/matrix budget exceeded")
        if training and len(X) > cfg["max_train_rows"]:
            raise ValueError("Declared training row budget exceeded")
        if not training and tuple(X.columns) != self.feature_names:
            raise ValueError("Prediction feature names/order differ from fitted inputs")
        if any(not pd.api.types.is_numeric_dtype(d) or pd.api.types.is_complex_dtype(d)
               for d in X.dtypes):
            raise ValueError("Numeric features only; no object/categorical/complex coercion")
        values = X.to_numpy(dtype=float, na_value=np.nan, copy=True)
        if np.isinf(values).any():
            raise ValueError("Infinite feature input")
        if training and np.isnan(values).all(axis=0).any():
            raise ValueError("Entirely missing training feature; declare exclusion in E16")
        return values

    @staticmethod
    def _vector(vector, X, name, *, positive=False):
        if not isinstance(vector, pd.Series) or not vector.index.equals(X.index):
            raise ValueError(f"{name} must have the exact training row index/order")
        if not pd.api.types.is_numeric_dtype(vector.dtype) or pd.api.types.is_complex_dtype(vector.dtype):
            raise ValueError(f"{name} must be numeric")
        values = vector.to_numpy(dtype=float, na_value=np.nan, copy=True)
        if not np.isfinite(values).all() or (positive and not (values > 0).all()):
            raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
        return values

    def fit(self, X, y, *, sample_weight=None):
        if self.state != "created":
            raise ValueError("Adapter cannot be refitted or retried; create a counted new instance")
        values = self._matrix(X, training=True)
        if len(X) < 2:
            raise ValueError("At least two training rows required")
        target = self._vector(y, X, "Target")
        weight = None if sample_weight is None else self._vector(
            sample_weight, X, "Sample weight", positive=True)
        self.feature_names = tuple(X.columns)
        self.state = "initializing"
        self.initialization_attempts = 1
        try:
            from sklearn.exceptions import ConvergenceWarning
            from threadpoolctl import threadpool_limits

            self._model = _pipeline(self.spec)
            with warnings.catch_warnings(), threadpool_limits(limits=self.spec.to_dict()["threads"]):
                warnings.simplefilter("error", ConvergenceWarning)
                self.state = "fitting"
                self.fit_attempts += 1
                kwargs = {} if weight is None else {"regressor__sample_weight": weight}
                self._model.fit(values, target, **kwargs)
            if self._model.named_steps["regressor"].n_features_in_ != 2 * len(self.feature_names):
                raise ValueError("Fitted preprocessing changed feature width")
            cfg = self.spec.to_dict()
            imputer = dict(self._model.named_steps["inputs"].transformer_list)["values"]
            self._metadata = dict(
                schema="fitted-tabular-model-v1", model_id=self.spec.model_id,
                spec=cfg, feature_names=list(self.feature_names), training_rows=len(X),
                transformed_features=2 * len(self.feature_names),
                imputation_medians=imputer.statistics_.tolist(),
                preprocessing="train median plus missing flags for every feature",
                scaling="train-only mean/std" if cfg["family"] in LINEAR else "none",
                preprocessing_weighting="unweighted; optional weights apply to regressor",
                sample_weight_used=weight is not None,
                early_stopping=False, forest_jobs=1, initialization_attempts=1, fit_attempts=1,
                implementation_hashes=implementation_hashes(["src/alpharesearch/models.py"]),
                environment={"python": platform.python_version(), "packages": {
                    n: importlib.metadata.version(n) for n in
                    ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "threadpoolctl")
                }},
                scope="numeric adapter only; E16 must validate features and mature labels",
                account_results=False,
            )
            if cfg["family"] in LINEAR:
                scaler = self._model.named_steps["scale"]
                self._metadata["scaler_mean"] = scaler.mean_.tolist()
                self._metadata["scaler_scale"] = scaler.scale_.tolist()
            json.dumps(self._metadata, allow_nan=False)
        except Exception as exc:
            self.state = "failed"
            self.error = f"{type(exc).__name__}: {exc}"
            self._metadata = None
            raise
        self.state = "fitted"
        self.successful_fits = 1
        return self

    def predict(self, X):
        if self.state != "fitted":
            raise ValueError("Prediction requires a successfully fitted adapter")
        values = self._matrix(X, training=False)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=self.spec.to_dict()["threads"]):
            prediction = np.asarray(self._model.predict(values), dtype=float)
        if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
            raise ValueError("Prediction must contain one finite score per supplied row")
        return pd.Series(prediction, index=X.index, name="score")

    def metadata(self):
        if self.state != "fitted":
            raise ValueError("Fitted metadata requires successful training")
        return json.loads(json.dumps(self._metadata, allow_nan=False))
