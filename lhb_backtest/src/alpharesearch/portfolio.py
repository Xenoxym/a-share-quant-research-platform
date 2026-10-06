"""Declared model-score portfolios for the existing next-session opening account."""
from dataclasses import dataclass
import json
import math

import numpy as np
import pandas as pd

from .contracts import required_id, timestamp, timestamps
from .learning import CLOCKS, KEYS, LearningResult, _day, _fingerprint, _keys
from .models import ModelSpec
from .registry import implementation_hashes
from ..technical.artifacts import content_id
from ..technical.signals import MAIN_BOARD, decision_dates

REFERENCES = [
    "reference_close", "reference_close_observed_at", "reference_close_known_at",
    "eligible", "eligibility_known_at",
]


@dataclass(frozen=True)
class ScorePortfolioSpec:
    document_json: str

    @classmethod
    def from_dict(cls, value):
        fields = {"schema", "name", "start", "end", "top_k", "allocation", "rebalance",
                  "account_universe", "horizon_policy", "max_prediction_rows"}
        if not isinstance(value, dict) or set(value) != fields or value["schema"] != "score-portfolio-v1":
            raise ValueError("Strict score portfolio fields required")
        required_id(value["name"], "portfolio name")
        if len(value["name"]) > 100:
            raise ValueError("Account display name exceeds 100 characters")
        _day(value["start"])
        _day(value["end"])
        if value["end"] < value["start"] or value["start"] < "2019-01-01":
            raise ValueError("Unsupported account date range")
        if type(value["top_k"]) is not int or not 1 <= value["top_k"] <= 100:
            raise ValueError("top_k must be an integer in [1, 100]")
        allocation = value["allocation"]
        if type(allocation) not in (int, float) or not math.isfinite(allocation) or not .01 <= allocation <= 1:
            raise ValueError("allocation must be finite in [.01, 1]")
        if value["rebalance"] not in ("weekly", "monthly"):
            raise ValueError("Native account bridge supports weekly/monthly schedules")
        if value["account_universe"] not in ("require_main_board", "main_board_subset"):
            raise ValueError("Explicit native account universe policy required")
        if value["horizon_policy"] not in ("require_next_rebalance_open", "allow_forecast_mismatch"):
            raise ValueError("Explicit target/account horizon policy required")
        cap = value["max_prediction_rows"]
        if type(cap) is not int or not 1 <= cap <= 10000000:
            raise ValueError("Prediction row cap must be an integer in [1, 10000000]")
        return cls(json.dumps(dict(value, allocation=float(allocation)), sort_keys=True,
                              separators=(",", ":"), allow_nan=False))

    def to_dict(self):
        return json.loads(self.document_json)

    @property
    def portfolio_id(self):
        doc = self.to_dict()
        doc.pop("name")
        return content_id(doc)


def score_portfolio_spec(*, start, end, name="model_score_portfolio", top_k=100,
                         allocation=.98, rebalance="weekly",
                         account_universe="require_main_board",
                         horizon_policy="require_next_rebalance_open",
                         max_prediction_rows=2000000):
    return ScorePortfolioSpec.from_dict(dict(
        schema="score-portfolio-v1", name=name, start=start, end=end, top_k=top_k,
        allocation=allocation, rebalance=rebalance, account_universe=account_universe,
        horizon_policy=horizon_policy, max_prediction_rows=max_prediction_rows,
    ))


@dataclass
class ScorePortfolio:
    decisions: pd.DataFrame
    schedule: dict
    receipt: dict


def build_score_portfolio(result, references, calendar, spec):
    """Validate the whole source score table, then explicitly select account dates.

    References are caller-verified, decision-time eligibility and raw closing data.
    Hash consistency is not source authentication. This function makes no trades.
    """
    if not isinstance(result, LearningResult) or not isinstance(spec, ScorePortfolioSpec):
        raise ValueError("LearningResult and validated ScorePortfolioSpec required")
    spec = ScorePortfolioSpec.from_dict(spec.to_dict())
    cfg = spec.to_dict()
    score, source = result.predictions, result.receipt
    if not isinstance(source, dict):
        raise ValueError("Learning receipt must be a mapping")
    if not isinstance(score, pd.DataFrame) or not isinstance(references, pd.DataFrame):
        raise ValueError("Scores and reference data must be DataFrames")
    if max(len(score), len(references)) > cfg["max_prediction_rows"]:
        raise ValueError("Declared prediction row cap exceeded")
    _keys(score, "Scores")
    _keys(references, "Reference data")
    if set(score) != set(KEYS + CLOCKS + ["score", "fit_id", "prediction_id"]):
        raise ValueError("Unexpected source score columns")
    if set(references) != set(KEYS + REFERENCES):
        raise ValueError("Strict reference/eligibility columns required")
    if source.get("schema") != "causal-learning-receipt-v2" or source.get("account_results") is not False:
        raise ValueError("Causal learning receipt required")
    if source.get("prediction_rows") != len(score) or source.get("prediction_outputs") != _fingerprint(score):
        raise ValueError("Source prediction output identity/row count changed")
    if (source.get("prediction_inputs", {}).get("rows") != len(score)
            or source.get("prediction_missing", {}).get("rows") != len(score)
            or source.get("prediction_membership") != _fingerprint(score[KEYS + CLOCKS])):
        raise ValueError("Source prediction input/missing/membership identity differs from complete output")
    fit_identity = source.get("fit_identity")
    if not isinstance(fit_identity, dict) or source.get("fit_id") != content_id(fit_identity):
        raise ValueError("Source fit identity is inconsistent")
    fitted = source.get("fitted_model")
    if (fit_identity.get("schema") != "causal-fit-identity-v1"
            or not isinstance(fitted, dict) or fitted.get("schema") != "fitted-tabular-model-v1"
            or fitted.get("model_id") != fit_identity.get("model_id")
            or source.get("model_id") != fit_identity.get("model_id")
            or fitted.get("training_rows") != source.get("mature_training_rows")
            or fitted.get("fit_attempts") != 1 or fitted.get("account_results") is not False):
        raise ValueError("Fitted model metadata does not match the source fit receipt")
    excluded = fit_identity.get("excluded_train_empty")
    declared = fit_identity["settings"]["features"]
    if (not isinstance(excluded, list) or any(key not in declared for key in excluded)
            or len(set(excluded)) != len(excluded)
            or source.get("excluded_train_empty") != excluded):
        raise ValueError("Source excluded features differ from bound fit identity")
    retained = [key for key in declared if key not in excluded]
    if (fitted.get("feature_names") != retained or source.get("retained_features") != retained
            or fitted.get("transformed_features") != 2 * len(retained)
            or ModelSpec.from_dict(fitted.get("spec")).model_id != source["model_id"]
            or fitted.get("environment") != fit_identity.get("environment")
            or dict(fitted.get("implementation_hashes", {})).get("src/alpharesearch/models.py")
               != fit_identity["implementation_hashes"].get("src/alpharesearch/models.py")
            or fit_identity["training_content"].get("rows") != source.get("mature_training_rows")
            or fit_identity["training_missing"].get("rows") != source.get("mature_training_rows")
            or timestamp(source["latest_training_label_known_at"]) > timestamp(fit_identity["settings"]["fit_cutoff"])):
        raise ValueError("Source fitted parameters/features or mature-label summary are inconsistent")
    prediction_id = content_id({
        "fit_id": source["fit_id"], "protocol_id": source["protocol_id"],
        "prediction_inputs": source["prediction_inputs"],
        "prediction_missing": source["prediction_missing"],
        "prediction_membership": source["prediction_membership"],
    })
    if source.get("prediction_id") != prediction_id:
        raise ValueError("Source prediction identity is inconsistent")
    if (not score.fit_id.eq(source["fit_id"]).all()
            or not score.prediction_id.eq(prediction_id).all()):
        raise ValueError("Score rows belong to different fit/prediction receipts")
    if (not pd.api.types.is_numeric_dtype(score.score.dtype)
            or pd.api.types.is_complex_dtype(score.score.dtype)
            or not np.isfinite(score.score.to_numpy(dtype=float)).all()):
        raise ValueError("Every supplied score must be finite numeric")
    if len(references) != len(score):
        raise ValueError("Every source score needs its exact reference row")
    joined = score.merge(references, on=KEYS, how="left", validate="one_to_one",
                         indicator=True, sort=False)
    if not joined.pop("_merge").eq("both").all():
        raise ValueError("Source scores and reference sample keys differ")
    if (not joined.eligible.map(lambda x: isinstance(x, (bool, np.bool_))).all()):
        raise ValueError("Eligibility must be boolean, declared before selection")
    if (not isinstance(calendar, list) or len(calendar) < 2
            or calendar != sorted(set(calendar))):
        raise ValueError("Full sorted unique market calendar required")
    for day in calendar:
        _day(day)
    if cfg["start"] < calendar[0] or cfg["end"] > calendar[-1]:
        raise ValueError("Account interval exceeds the supplied calendar")
    for day in joined.trade_date:
        _day(day)
    if not joined.trade_date.isin(calendar).all():
        raise ValueError("Prediction date is outside the market calendar")
    if not joined.stock_code.map(lambda x: isinstance(x, str)).all() or not joined.stock_code.str.fullmatch(r"\d{6}\.(SH|SZ|BJ)").all():
        raise ValueError("Exchange-qualified stock codes required")
    for key in CLOCKS + ["eligibility_known_at"]:
        joined[key] = timestamps(joined[key]).astype("datetime64[ns, UTC]")
    if not joined.decision_at.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d").eq(joined.trade_date).all():
        raise ValueError("Prediction decision clock and local date differ")
    cutoff = pd.Timestamp(fit_identity["settings"]["fit_cutoff"])
    if (joined.decision_at.le(cutoff) | joined.execution_at.le(joined.decision_at)
            | joined.target_end_at.le(joined.execution_at)
            | joined.eligibility_known_at.gt(joined.decision_at)).any():
        raise ValueError("Score/eligibility/fit timing is inconsistent")

    schedule = decision_dates(calendar, cfg["start"], cfg["end"], cfg["rebalance"])
    if not schedule:
        raise ValueError("No planned rebalance decisions in account interval")
    rows = joined.loc[joined.trade_date.isin(schedule)].copy()
    if set(rows.trade_date) != set(schedule):
        raise ValueError("Scheduled decision date lacks its scored candidate pool")
    main_board = rows.stock_code.str.fullmatch(MAIN_BOARD)
    if cfg["account_universe"] == "require_main_board" and not main_board.all():
        raise ValueError("Broader learning universe needs explicit main-board account subset")
    usable = main_board & rows.eligible
    reference = rows.loc[usable]
    if (not pd.api.types.is_numeric_dtype(reference.reference_close.dtype)
            or pd.api.types.is_complex_dtype(reference.reference_close.dtype)):
        raise ValueError("Raw reference closing prices must be numeric")
    if not np.isfinite(reference.reference_close.to_numpy(dtype=float)).all() or reference.reference_close.le(0).any():
        raise ValueError("Eligible account candidates need positive raw reference close")
    expected_observed = pd.to_datetime(rows.loc[usable, "trade_date"] + "T15:00:00+08:00", utc=True)
    observed = timestamps(reference.reference_close_observed_at)
    known = timestamps(reference.reference_close_known_at)
    if (~observed.eq(expected_observed) | known.lt(observed) | known.gt(reference.decision_at)).any():
        raise ValueError("Raw reference close was not available at decision time")
    expected_entry = pd.to_datetime(rows.trade_date.map(schedule) + "T09:30:00+08:00", utc=True)
    if not rows.execution_at.eq(expected_entry).all():
        raise ValueError("Native account requires next-session 09:30 opening entry")
    all_schedule = decision_dates(calendar, calendar[0], calendar[-1], cfg["rebalance"])
    all_dates = sorted(all_schedule)
    next_entries = {day: all_schedule[all_dates[i + 1]] for i, day in enumerate(all_dates[:-1])}
    expected_end = pd.to_datetime(rows.trade_date.map(next_entries) + "T09:30:00+08:00", utc=True)
    aligned = rows.target_end_at.eq(expected_end)
    if cfg["horizon_policy"] == "require_next_rebalance_open" and (usable & ~aligned).any():
        raise ValueError("Learning target differs from, or lacks, next planned rebalance open")

    rows["reason"] = np.where(~main_board, "outside_native_main_board",
                              np.where(~rows.eligible, "ineligible_declared", "eligible_external_score"))
    rows = rows.sort_values(["trade_date", "score", "stock_code"],
                            ascending=[True, False, True], kind="stable")
    eligible = rows.reason.eq("eligible_external_score")
    rows["rank"] = np.nan
    rows.loc[eligible, "rank"] = rows.loc[eligible].groupby("trade_date").cumcount() + 1
    rows["selected"] = rows["rank"].le(cfg["top_k"])
    rows["target_weight"] = np.where(rows.selected, cfg["allocation"] / cfg["top_k"], 0.)
    rows["close"] = rows.reference_close
    rows["execution_date"] = rows.trade_date.map(schedule)
    rows["known_at_assumed"] = rows.decision_at.map(lambda x: x.isoformat())
    columns = ["stock_code", "trade_date", "close", "score", "rank", "selected",
               "reason", "target_weight", "execution_date", "known_at_assumed"]
    decisions = rows[columns].reset_index(drop=True)
    horizons = joined.loc[joined.trade_date.isin(schedule), KEYS + ["target_end_at"]].copy()
    horizons["account_next_rebalance_open"] = expected_end.to_numpy()
    horizons["target_aligned"] = aligned.to_numpy()
    receipt = dict(
        schema="score-portfolio-receipt-v1", portfolio_id=spec.portfolio_id,
        spec=cfg, source_fit_id=source["fit_id"], source_prediction_id=prediction_id,
        source_learning_receipt_id=content_id(source),
        source_score_outputs=source["prediction_outputs"], reference_inputs=_fingerprint(references),
        calendar_id=content_id(calendar), schedule_id=content_id(schedule),
        scored_universe_id=fit_identity["settings"]["universe_id"],
        account_eligible_membership=_fingerprint(rows.loc[eligible, KEYS].reset_index(drop=True)),
        source_scored_rows=len(score), account_decision_rows=len(rows),
        scored_rows_not_account_dates=len(score) - len(rows),
        reason_counts=rows.reason.value_counts().to_dict(), selected_rows=int(rows.selected.sum()),
        target_allocation_rule="allocation/top_k per selected stock; unfilled slots stay cash",
        decision_outputs=_fingerprint(decisions), planned_horizons=_fingerprint(horizons),
        target_mismatch_eligible_rows=int((usable & ~aligned).sum()),
        schedule_after_account_end=sum(day > cfg["end"] for day in schedule.values()),
        terminal_policy="mark holdings on last account session; no forced terminal liquidation",
        account_universe_policy=cfg["account_universe"],
        implementation_hashes=dict(implementation_hashes([
            "src/alpharesearch/portfolio.py", "src/alpharesearch/learning.py",
            "src/technical/signals.py", "src/technical/portfolio.py",
            "src/technical/contracts.py", "src/technical/market.py",
        ])),
        input_artifact_verification="caller responsibility; declared eligibility is not vendor certification",
        account_results=False, scope="portfolio decisions only; no account executed by this bridge",
    )
    # This is an independent copy, so later edits cannot alter archived receipt metadata.
    receipt = json.loads(json.dumps(receipt, allow_nan=False))
    return ScorePortfolio(decisions, schedule, receipt)
