"""Training-only target transforms; no feature fit, predictor or account."""
from dataclasses import dataclass
import json
import re
from datetime import datetime
import numpy as np
import pandas as pd
from .learning import KEYS, CLOCKS, LABELS, _keys, _day, _fingerprint
from ..technical.artifacts import content_id


def _clock(value):
    # datetime.fromisoformat loses submicrosecond ISO fractions. Preserveaccepted
    # UTC nanosecondprecision here without changing the old production contract.
    if isinstance(value, str):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})", value):
            raise ValueError("Explicit aware ISOclock withatmostnanosecond precision required")
    elif not isinstance(value, datetime):
        raise ValueError("Aware datetime or ISOclock required")
    try:
        parsed = pd.Timestamp(value)
        if pd.isna(parsed) or parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("Timestamp cannotbe missingor naive")
        return parsed.tz_convert("UTC")
    except (TypeError, OverflowError) as exc:
        raise ValueError("Unsupported clock") from exc


def _clocks(values):
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        if values.isna().any():
            raise ValueError("Timestamp cannotbe missing")
        return values.dt.tz_convert("UTC")
    return pd.to_datetime(values.map(_clock), utc=True)

KINDS = ("raw_return", "date_centered", "date_zscore", "date_rank")

@dataclass(frozen=True)
class TrainingTargetSpec:
    kind: str
    train_start: str
    train_end: str
    fit_cutoff: str
    min_date_rows: int = 2
    max_input_rows: int = 1000000
    scale_floor: float = 1e-12

    def validated(self):
        if self.kind not in KINDS:
            raise ValueError("Unknown training target")
        if _day(self.train_start) > _day(self.train_end):
            raise ValueError("Training interval reversed")
        cutoff = _clock(self.fit_cutoff)
        if cutoff.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") < self.train_end:
            raise ValueError("Fit cutoff precedes declared training end")
        if type(self.min_date_rows) is not int or not 2 <= self.min_date_rows <= 100000:
            raise ValueError("Finite minimum date count required")
        if type(self.max_input_rows) is not int or not 2 <= self.max_input_rows <= 1000000:
            raise ValueError("Finite input row cap required")
        if type(self.scale_floor) not in (int, float) or not np.isfinite(self.scale_floor) or not 0 < self.scale_floor <= 1:
            raise ValueError("Positive finite scale floor required")
        return TrainingTargetSpec(self.kind, self.train_start, self.train_end,
                                  cutoff.isoformat(), self.min_date_rows,
                                  self.max_input_rows, float(self.scale_floor))

    @property
    def target_id(self):
        cfg = self.validated()
        return content_id(dict(schema="mature-training-target-v1", **vars(cfg)))

@dataclass
class TrainingTargets:
    rows: pd.DataFrame
    receipt: dict


def build_training_targets(decisions, labels, spec):
    """Retain every declared training decision; unavailable targets remain unknown.

    Transforms use the mature, observed subset per decision date, not all A shares.
    Prediction labels are never joined. Raw Y and all clocks are retained separately.
    File authenticity/features and later account conversion remain caller duties.
    """
    if not isinstance(spec, TrainingTargetSpec):
        raise ValueError("Explicit TrainingTargetSpec required")
    cfg = spec.validated()
    for frame, fields in ((decisions, KEYS + CLOCKS), (labels, KEYS + LABELS)):
        if not isinstance(frame, pd.DataFrame) or set(frame) != set(fields) or len(frame) > cfg.max_input_rows:
            raise ValueError("Strict bounded decision/label tables required")
        _keys(frame, "Target input")
    d = decisions.loc[decisions.trade_date.between(cfg.train_start, cfg.train_end)].copy()
    if d.empty:
        raise ValueError("No declared training decisions")
    for clock in CLOCKS:
        d[clock] = _clocks(d[clock]).astype("datetime64[ns, UTC]")
    cutoff = pd.Timestamp(cfg.fit_cutoff)
    if (not d.decision_at.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d").eq(d.trade_date).all()
            or d.decision_at.gt(cutoff).any() or d.execution_at.le(d.decision_at).any()
            or d.target_end_at.le(d.execution_at).any()):
        raise ValueError("Invalid training decision clocks")
    # Reject conflicting identities within training, without using future numeric Y.
    l = labels.loc[labels.trade_date.between(cfg.train_start, cfg.train_end)].copy()
    by_stock = d[KEYS].merge(l[KEYS], on=["trade_date", "stock_code"], suffixes=("_d", "_l"), validate="one_to_one")
    by_id = d[KEYS].merge(l[KEYS], on="sample_id", suffixes=("_d", "_l"), validate="one_to_one")
    if (not by_stock.sample_id_d.eq(by_stock.sample_id_l).all()
            or not by_id.trade_date_d.eq(by_id.trade_date_l).all()
            or not by_id.stock_code_d.eq(by_id.stock_code_l).all()):
        raise ValueError("Training label identity mismatch")
    out = d.merge(l, on=KEYS, how="left", validate="one_to_one", indicator=True, sort=False)
    supplied = out._merge.eq("both")
    if not out.loc[supplied, "label_observed"].map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("Observed flags must be boolean")
    observed = supplied & out.label_observed.eq(True)
    absent = supplied & ~observed
    if out.loc[absent, ["target_return", "label_known_at"]].notna().any().any():
        raise ValueError("Unknown label cannot carry value/known clock")
    for clock in ("label_start", "label_end", "label_known_at"):
        vals = out.loc[observed, clock]
        converted = _clocks(vals).astype("datetime64[ns, UTC]")
        out[clock] = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
        out.loc[observed, clock] = converted
    if (not out.loc[observed, "label_start"].eq(out.loc[observed, "execution_at"]).all()
            or not out.loc[observed, "label_end"].eq(out.loc[observed, "target_end_at"]).all()
            or out.loc[observed, "label_known_at"].lt(out.loc[observed, "label_end"]).any()):
        raise ValueError("Observed target clocks differ from execution interval")
    mature = observed & out.label_known_at.le(cutoff)
    vals = out.loc[mature, "target_return"]
    if not vals.map(lambda x: isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))).all():
        raise ValueError("Mature Y must be real numeric")
    raw = vals.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(raw).all():
        raise ValueError("Mature Y must be finite")
    # Keep raw mature Y only: even the diagnostic output does not disclose immature Y.
    out["raw_target"] = np.nan
    out.loc[mature, "raw_target"] = raw
    out["training_target"] = np.nan
    out["target_reason"] = "label_absent"
    out.loc[absent, "target_reason"] = "label_unobserved"
    out.loc[observed & ~mature, "target_reason"] = "label_immature"
    out["mature_date_rows"] = 0
    for positions in out.loc[mature].groupby("trade_date", sort=True).groups.values():
        y = out.loc[positions, "raw_target"]
        n = len(y); out.loc[positions, "mature_date_rows"] = n
        if cfg.kind != "raw_return" and n < cfg.min_date_rows:
            out.loc[positions, "target_reason"] = "thin_date_group"; continue
        if cfg.kind == "raw_return":
            z = y
        elif cfg.kind == "date_centered":
            z = y - y.mean()
        elif cfg.kind == "date_zscore":
            scale = y.std(ddof=0)
            if scale <= cfg.scale_floor:
                out.loc[positions, "target_reason"] = "constant_date_group"; continue
            z = (y - y.mean()) / scale
        else:
            if y.max() - y.min() <= cfg.scale_floor:
                out.loc[positions, "target_reason"] = "constant_date_group"; continue
            z = (y.rank(method="average") - 1) / (n - 1) - .5
        if not np.isfinite(np.asarray(z, dtype=float)).all():
            raise ValueError("Transformed target is not finite")
        out.loc[positions, "training_target"] = z
        out.loc[positions, "target_reason"] = "present"
    columns = KEYS + CLOCKS + ["label_start", "label_end", "label_known_at", "raw_target", "training_target", "target_reason", "mature_date_rows"]
    result = out[columns].reset_index(drop=True)
    receipt = dict(schema="training-target-receipt-v1", target_id=cfg.target_id,
                   spec=vars(cfg), rows=len(result), usable_rows=int(result.target_reason.eq("present").sum()),
                   missing_counts=result.target_reason.value_counts().to_dict(), content=_fingerprint(result), usable_content=_fingerprint(result.loc[result.target_reason.eq("present")].reset_index(drop=True)),
                   unit="decimal_return" if cfg.kind in ("raw_return", "date_centered") else "dimensionless",
                   group_denominator="Observed labels known byfitcutoff within declaredtrainingdecisions",
                   rank_ties="average", zscore_ddof=0, prediction_labels_used=False,
                   predicted_score_is_raw_return=cfg.kind == "raw_return", account_results=False)
    json.dumps(receipt, allow_nan=False)
    return TrainingTargets(result, receipt)
