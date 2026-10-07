"""Bounded development screening; endpoint gross proxies are never account results."""
from dataclasses import dataclass
from itertools import combinations, islice
import json

import numpy as np
import pandas as pd

from .assembly import FeatureAssembly
from .contracts import VintagePolicy, required_id, timestamp, timestamps
from .features.base import FeatureBlock
from .learning import KEYS, CLOCKS, LABELS, _day, _integer, _keys, _fingerprint, _environment
from .registry import implementation_hashes
from ..technical.artifacts import content_id

CODE = ["src/alpharesearch/screening.py", "src/alpharesearch/learning.py",
        "src/alpharesearch/assembly.py", "src/alpharesearch/contracts.py",
        "src/alpharesearch/features/base.py", "src/alpharesearch/registry.py",
        "src/technical/artifacts.py"]


def _number(value, name):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not np.isfinite(value)):
        raise ValueError(name + " requires a finite real number")


@dataclass(frozen=True)
class ScreenSpec:
    document_json: str

    @classmethod
    def from_dict(cls, value):
        fields = {"schema", "name", "dataset_id", "universe_id", "development_start",
                  "development_end", "holdout_start", "diagnostic_cutoff", "target",
                  "allow_weak_vintage", "channels", "top_k", "min_pairs",
                  "min_coverage", "min_condition_rows", "min_proxy_windows",
                  "max_input_rows", "max_matrix_cells", "max_channels",
                  "max_redundancy_pairs", "family_quotas"}
        if not isinstance(value, dict) or set(value) != fields or value["schema"] != "development-screen-v1":
            raise ValueError("Strict development screening fields required")
        for key in ("name", "dataset_id", "universe_id"):
            required_id(value[key], key)
        for key in ("development_start", "development_end", "holdout_start"):
            _day(value[key])
        if not value["development_start"] <= value["development_end"] < value["holdout_start"]:
            raise ValueError("Development must precede declared holdout")
        cutoff = pd.Timestamp(timestamp(value["diagnostic_cutoff"]))
        if (cutoff.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") < value["development_end"]
                or cutoff.tz_convert("Asia/Shanghai").strftime("%Y-%m-%d") >= value["holdout_start"]):
            raise ValueError("Diagnostic cutoff must follow development and precede holdout date")
        if value["target"] != "execution_window_return_decimal":
            raise ValueError("Declared execution-window decimal returns required")
        if type(value["allow_weak_vintage"]) is not bool:
            raise ValueError("Explicit weak vintage Boolean required")
        for key, low, high in (("top_k", 1, 500), ("min_pairs", 3, 100000),
                              ("min_condition_rows", 1, 10000000), ("min_proxy_windows", 1, 100000),
                              ("max_input_rows", 1, 10000000), ("max_matrix_cells", 1, 100000000),
                              ("max_channels", 1, 10000), ("max_redundancy_pairs", 0, 100000)):
            _integer(value[key], low, high, key)
        _number(value["min_coverage"], "min_coverage")
        if not 0 <= value["min_coverage"] <= 1:
            raise ValueError("Coverage must be within [0, 1]")
        channels = value["channels"]
        if not isinstance(channels, list) or not channels or len(channels) > value["max_channels"]:
            raise ValueError("Nonempty bounded channel list required")
        ids, families = set(), set()
        for c in channels:
            if not isinstance(c, dict) or set(c) != {"channel_id", "feature_key", "family",
                    "direction", "condition", "route", "exploration_reason", "hypothesis"}:
                raise ValueError("Strict declared channel fields required")
            for key in ("channel_id", "feature_key", "family", "hypothesis"):
                required_id(c[key], key)
            if c["channel_id"] in ids:
                raise ValueError("Duplicate channel identity")
            ids.add(c["channel_id"]); families.add(c["family"])
            if type(c["direction"]) is not int or c["direction"] not in (-1, 1):
                raise ValueError("Direction must be declared as +1 or -1; this is not short selling")
            if c["route"] not in ("merit", "exploration"):
                raise ValueError("Declared merit/exploration route required")
            if not isinstance(c["exploration_reason"], str):
                raise ValueError("Exploration reason must be text")
            if c["route"] == "exploration" and not c["exploration_reason"].strip():
                raise ValueError("Exploration route requires an advance reason")
            cond = c["condition"]
            if cond is not None:
                if not isinstance(cond, dict) or set(cond) != {"feature_key", "op", "threshold"}:
                    raise ValueError("Condition must reference an already-known numeric feature")
                required_id(cond["feature_key"], "condition feature")
                if cond["op"] not in ("gt", "ge", "lt", "le", "eq"):
                    raise ValueError("Unsupported condition comparator")
                _number(cond["threshold"], "condition threshold")
        quotas = value["family_quotas"]
        if not isinstance(quotas, dict) or set(quotas) != families:
            raise ValueError("Every declared family needs explicit quotas")
        for quota in quotas.values():
            if not isinstance(quota, dict) or set(quota) != {"merit", "exploration"}:
                raise ValueError("Separate merit/exploration quotas required")
            for n in quota.values():
                _integer(n, 0, value["max_channels"], "family quota")
        return cls(json.dumps(dict(value, diagnostic_cutoff=cutoff.isoformat()), sort_keys=True,
                              separators=(",", ":"), allow_nan=False))

    def to_dict(self):
        return json.loads(self.document_json)

    @property
    def protocol_id(self):
        return content_id(self.to_dict())


def _prepare(spec, decisions, assembly, labels):
    cfg = spec.to_dict()
    if not isinstance(assembly, FeatureAssembly):
        raise ValueError("FeatureAssembly required")
    block = assembly.block
    if (not isinstance(block, FeatureBlock)
            or block.metadata.get("schema") != "assembled-feature-matrix-v1"
            or block.metadata.get("universe_id") != cfg["universe_id"]
            or block.metadata.get("registry_version") != assembly.registry.version_id
            or block.metadata.get("source_bindings") != list(assembly.source_bindings)
            or block.metadata.get("key_columns") != KEYS):
        raise ValueError("Assembly identity differs from screening declaration")
    names = sorted({c["feature_key"] for c in cfg["channels"]}
                   | {c["condition"]["feature_key"] for c in cfg["channels"] if c["condition"] is not None})
    if not set(names) <= set(block.units):
        raise ValueError("A declared candidate/condition was not supplied")
    for frame in (decisions, block.values, block.missing, labels):
        if not isinstance(frame, pd.DataFrame) or len(frame) > cfg["max_input_rows"]:
            raise ValueError("Input row budget exceeded or invalid table")
        _keys(frame, "Screen input")
    if set(decisions) != set(KEYS + CLOCKS) or set(labels) != set(KEYS + LABELS):
        raise ValueError("Strict decision/label columns required")
    p = decisions.loc[decisions.trade_date.between(cfg["development_start"], cfg["development_end"])].copy()
    p = p.sort_values(["trade_date", "stock_code", "sample_id"]).reset_index(drop=True)
    if p.empty or len(p) * len(names) > cfg["max_matrix_cells"]:
        raise ValueError("Empty development decisions or scoped matrix budget exceeded")
    for key in CLOCKS:
        p[key] = timestamps(p[key]).astype("datetime64[ns, UTC]")
    if not p.decision_at.dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d").eq(p.trade_date).all():
        raise ValueError("Decision local date differs from declared date")
    if (~p.execution_at.gt(p.decision_at) | ~p.target_end_at.gt(p.execution_at)).any():
        raise ValueError("Execution/target clocks must strictly follow decision/entry")
    cutoff = pd.Timestamp(cfg["diagnostic_cutoff"])
    if p.decision_at.gt(cutoff).any():
        raise ValueError("Decision occurs after diagnostic cutoff")
    previous = None
    for _, group in p.groupby("trade_date", sort=True):
        if group.execution_at.nunique() != 1 or group.target_end_at.nunique() != 1:
            raise ValueError("All stocks need the same declared execution window on a decision date")
        entry, end = group.execution_at.iloc[0], group.target_end_at.iloc[0]
        if previous is not None and entry < previous:
            raise ValueError("Endpoint proxy windows must not overlap")
        previous = end
    values = p[KEYS].merge(block.values[KEYS + names + ["observed_end", "known_at"]],
                          on=KEYS, how="left", validate="one_to_one", indicator=True, sort=False)
    missing = p[KEYS].merge(block.missing[KEYS + names], on=KEYS, how="left",
                           validate="one_to_one", sort=False)
    if not values.pop("_merge").eq("both").all():
        raise ValueError("Decision lacks its feature row")
    scoped = FeatureBlock(values, missing, {n: block.units[n] for n in names}, block.metadata).validate()
    for key in ("known_at", "observed_end"):
        scoped.values[key] = timestamps(scoped.values[key]).astype("datetime64[ns, UTC]")
        if scoped.values[key].gt(p.decision_at).any():
            raise ValueError("Candidate or condition was not known at decision")
    if block.metadata["vintage"] not in (VintagePolicy.HISTORICAL.value, VintagePolicy.MARKET.value) and not cfg["allow_weak_vintage"]:
        raise ValueError("Weak vintage requires explicit screening opt-in")
    bound_definitions = assembly.registry.resolve_bound_definitions(assembly.source_bindings, names)
    for name in names:
        if (not pd.api.types.is_numeric_dtype(scoped.values[name])
                or pd.api.types.is_complex_dtype(scoped.values[name])):
            raise ValueError("Numeric noncomplex candidate/condition values required")
        definition = bound_definitions[name]
        if definition.unit != block.units[name] or definition.domain not in ("stock_day", "market_day"):
            raise ValueError("Registered unit and causal stock/day or broadcast market/day domain required")
        if pd.api.types.is_bool_dtype(scoped.values[name]) and definition.unit != "binary":
            raise ValueError("Boolean cannot silently become a financial feature")
    # An omitted label is permitted; a supplied label cannot change key ownership.
    pairs = p[KEYS].merge(labels[KEYS], on=["trade_date", "stock_code"],
                         suffixes=("_decision", "_label"), validate="one_to_one")
    ids = p[KEYS].merge(labels[KEYS], on="sample_id",
                       suffixes=("_decision", "_label"), validate="one_to_one")
    if (not pairs.sample_id_decision.eq(pairs.sample_id_label).all()
            or not (ids.trade_date_decision.eq(ids.trade_date_label)
                    & ids.stock_code_decision.eq(ids.stock_code_label)).all()):
        raise ValueError("Label key ownership conflicts with decisions")
    y = p.merge(labels, on=KEYS, how="left", validate="one_to_one", indicator=True, sort=False)
    supplied = y._merge.eq("both")
    if not y.loc[supplied, "label_observed"].map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("Observed label flag must be Boolean")
    observed = supplied & y.label_observed.eq(True)
    known = pd.Series(pd.NaT, index=y.index, dtype="datetime64[ns, UTC]")
    if supplied.any():
        for key in ("label_start", "label_end"):
            clock = pd.to_datetime(y.loc[supplied, key].map(timestamp), utc=True)
            expected = y.loc[supplied, "execution_at" if key == "label_start" else "target_end_at"]
            if not clock.eq(expected).all():
                raise ValueError("Label interval differs from actual execution window")
        if (not y.loc[supplied & ~observed, "target_return"].isna().all()
                or not y.loc[supplied & ~observed, "label_known_at"].isna().all()):
            raise ValueError("Unobserved outcome cannot carry target/known timestamp")
    if observed.any():
        known.loc[observed] = timestamps(y.loc[observed, "label_known_at"]).astype("datetime64[ns, UTC]")
        if known.loc[observed].lt(y.loc[observed, "target_end_at"]).any():
            raise ValueError("Label cannot be known before target endpoint")
    mature = observed & known.le(cutoff)
    target = pd.Series(np.nan, index=y.index, dtype=float)
    if mature.any():
        actual = y.loc[mature, "target_return"]
        if not actual.map(lambda v: isinstance(v, (int, float, np.integer, np.floating))
                          and not isinstance(v, (bool, np.bool_))).all():
            raise ValueError("Mature targets must be numeric decimal returns")
        numeric = actual.to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric < -1).any():
            raise ValueError("Mature long-only target must be finite and at least -100%")
        target.loc[mature] = numeric
    reason = pd.Series("label_omitted", index=y.index, dtype=object)
    reason.loc[supplied & ~observed] = "outcome_unobserved"
    reason.loc[observed & ~mature] = "outcome_immature"
    reason.loc[mature] = "present"
    return cfg, p, scoped, target, reason


def _condition(c, values):
    if c["condition"] is None:
        return np.ones(len(values), dtype=bool), np.ones(len(values), dtype=bool)
    cond = c["condition"]; v = values[cond["feature_key"]].to_numpy(dtype=float)
    known = np.isfinite(v); threshold = cond["threshold"]
    masks = {"gt": lambda: v > threshold, "ge": lambda: v >= threshold,
             "lt": lambda: v < threshold, "le": lambda: v <= threshold,
             "eq": lambda: v == threshold}
    return masks[cond["op"]]() & known, known


def _rank_ic(x, y, minimum):
    pairs = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(pairs) < minimum:
        return None, len(pairs), "insufficient_pairs"
    if pairs.x.nunique() < 2 or pairs.y.nunique() < 2:
        return None, len(pairs), "constant_score_or_target"
    return float(pairs.x.rank().corr(pairs.y.rank())), len(pairs), "present"


@dataclass
class ScreenResult:
    daily: pd.DataFrame
    selections: pd.DataFrame
    windows: pd.DataFrame
    trials: pd.DataFrame
    redundancy: pd.DataFrame
    receipt: dict


def screen(spec, decisions, assembly, labels):
    """One bounded pure call. Selection rules depend only on X; Y is diagnostic."""
    if not isinstance(spec, ScreenSpec):
        raise ValueError("Validated ScreenSpec required")
    spec = ScreenSpec.from_dict(spec.to_dict())
    cfg, p, scoped, target, label_reason = _prepare(spec, decisions, assembly, labels)
    daily, selections, windows, trials = [], [], [], []
    dates = list(p.groupby("trade_date", sort=True).indices.items())
    for channel in cfg["channels"]:
        cid = channel["channel_id"]
        raw = scoped.values[channel["feature_key"]].to_numpy(dtype=float)
        score = raw * channel["direction"]
        cond, cond_known = _condition(channel, scoped.values)
        eligible = cond & np.isfinite(score)
        nav, peak, drawdown, path_known = 1.0, 1.0, 0.0, True
        ic_values = []
        for day, index in dates:
            index = np.asarray(index)
            active = index[eligible[index]]
            ic, pairs, ic_reason = _rank_ic(score[active], target.iloc[active].to_numpy(), cfg["min_pairs"])
            if ic is not None:
                ic_values.append(ic)
            ncond = int(cond[index].sum())
            daily.append(dict(channel_id=cid, trade_date=day, decision_rows=len(index),
                condition_true=ncond, condition_false=int((cond_known[index] & ~cond[index]).sum()),
                condition_unknown=int((~cond_known[index]).sum()), feature_present=int(np.isfinite(raw[index]).sum()),
                eligible_rows=len(active), conditional_coverage=None if not ncond else len(active)/ncond,
                paired_rows=pairs, rank_ic=ic, rank_ic_reason=ic_reason,
                constant_feature=bool(len(active) and np.unique(score[active]).size == 1)))
            # Rank selected IDs using only X; target missingness cannot choose them.
            order = pd.DataFrame({"position": active, "score": score[active],
                                  "stock_code": p.iloc[active].stock_code.to_numpy()})
            chosen = order.sort_values(["score", "stock_code"], ascending=[False, True], kind="stable").head(cfg["top_k"])
            positions = chosen.position.to_numpy(dtype=int)
            for rank, pos in enumerate(positions, 1):
                selections.append(dict(channel_id=cid, trade_date=day, sample_id=p.iloc[pos].sample_id,
                    stock_code=p.iloc[pos].stock_code, score=float(score[pos]), rank=rank, weight=1/cfg["top_k"]))
            unknown = int(target.iloc[positions].isna().sum())
            reason = "selected_outcome_unknown" if unknown else None
            with np.errstate(over="ignore", invalid="ignore"):
                ret = None if unknown else float(np.sum(target.iloc[positions].to_numpy()/cfg["top_k"]))
            if ret is not None and not np.isfinite(ret):
                ret, reason = None, "numeric_return_overflow"
            if ret is not None:
                # Known long-only Y >= -1 and fixed weights sum <= 1: exact
                # portfolio return cannot be below -1. Repair only roundoff.
                ret = max(-1.0, ret)
            if ret is None:
                path_known = False
            if path_known:
                nav *= 1 + ret
                if not np.isfinite(nav):
                    path_known, reason = False, "numeric_nav_overflow"
                else:
                    peak = max(peak, nav); drawdown = max(drawdown, 1-nav/peak)
            elif reason is None:
                reason = "prior_path_unknown"
            windows.append(dict(channel_id=cid, trade_date=day, execution_at=p.iloc[index[0]].execution_at,
                target_end_at=p.iloc[index[0]].target_end_at, selected_stocks=len(positions),
                cash_weight=1-len(positions)/cfg["top_k"], selected_unknown_targets=unknown,
                window_gross_return=ret, endpoint_nav=nav if path_known else None,
                endpoint_drawdown=drawdown if path_known else None,
                unknown_reason=reason))
        denominator = int(cond.sum()); neligible = int(eligible.sum())
        coverage = None if not denominator else neligible/denominator
        quality = (denominator >= cfg["min_condition_rows"]
                   and coverage is not None and coverage >= cfg["min_coverage"])
        complete = path_known and len(dates) >= cfg["min_proxy_windows"]
        trials.append(dict(channel_id=cid, feature_key=channel["feature_key"], family=channel["family"],
            direction=channel["direction"], route=channel["route"], exploration_reason=channel["exploration_reason"],
            condition=channel["condition"], status="evaluated", conditional_rows=denominator,
            eligible_rows=neligible, coverage=coverage, quality_passed=quality,
            rank_ic_mean=None if not ic_values else float(np.mean(ic_values)), rank_ic_dates=len(ic_values),
            proxy_complete=bool(complete), proxy_total_gross_return=nav-1 if complete else None,
            proxy_endpoint_max_drawdown=drawdown if complete else None, proxy_windows=len(dates),
            promoted=False, selection_reason="quality_not_met" if not quality else "pending_family_quota"))
    # Pareto comparison uses the same complete nonoverlapping window domain, not
    # each candidate's selectively available "good" dates. IC is not a hard gate.
    for family, quotas in cfg["family_quotas"].items():
        merit = [t for t in trials if t["family"] == family and t["route"] == "merit"
                 and t["quality_passed"] and t["proxy_complete"]]
        merit.sort(key=lambda t: (-t["proxy_total_gross_return"], t["proxy_endpoint_max_drawdown"], t["channel_id"]))
        front = []
        best_dd, best_return = float("inf"), None
        for t in merit:
            r, dd = t["proxy_total_gross_return"], t["proxy_endpoint_max_drawdown"]
            if dd < best_dd or (dd == best_dd and r == best_return):
                front.append(t)
                best_dd, best_return = dd, r
        for t in front[:quotas["merit"]]:
            t.update(promoted=True, selection_reason="complete_proxy_pareto_family_quota")
        explore = sorted([t for t in trials if t["family"] == family and t["route"] == "exploration"
                          and t["quality_passed"]], key=lambda t: t["channel_id"])
        for t in explore[:quotas["exploration"]]:
            t.update(promoted=True, selection_reason="predeclared_exploration_quota; efficacy_not_proved")
    for t in trials:
        if t["selection_reason"] == "pending_family_quota":
            t["selection_reason"] = ("incomplete_common_proxy_domain" if t["route"] == "merit" and not t["proxy_complete"]
                                     else "family_quota_or_proxy_dominance")
    names = sorted({c["feature_key"] for c in cfg["channels"]})
    pair_count = len(names)*(len(names)-1)//2
    redundancy = []
    for left, right in islice(combinations(names, 2), cfg["max_redundancy_pairs"]):
        r, n, why = _rank_ic(scoped.values[left], scoped.values[right], cfg["min_pairs"])
        redundancy.append(dict(left=left, right=right, common_rows=n, pooled_rank_correlation=r, reason=why))
    consumption = p.copy()
    consumption["target_used"] = target
    consumption["target_state"] = label_reason
    receipt = dict(schema="development-screen-receipt-v1", protocol_id=spec.protocol_id,
        spec=cfg, decision_inputs=_fingerprint(p), scoped_features=_fingerprint(scoped.values),
        scoped_missing=_fingerprint(scoped.missing), consumed_outcomes=_fingerprint(consumption),
        declared_registry_version=assembly.registry.version_id, declared_assembly_version=assembly.block.metadata["version_id"],
        candidate_direction_condition_intents=len(cfg["channels"]), development_decisions=len(p),
        redundancy_pairs_computed=len(redundancy), redundancy_pairs_omitted=pair_count-len(redundancy),
        implementation_hashes=dict(implementation_hashes(CODE)), environment=_environment(),
        outcome_scope="mature development only; future target numeric values not consumed",
        proxy="fixed-K equal slots; empty slots cash; nonoverlapping endpoint gross path only",
        account_results=False, annualized_results=False, complete_intraperiod_drawdown=False,
        historical_execution_certified=False, input_artifact_verification="caller responsibility; declarations are not authentication",
        durable_executor=False, ledger_scope="one process; E13B persistent registration/execution required")
    return ScreenResult(pd.DataFrame(daily), pd.DataFrame(selections, columns=[
        "channel_id", "trade_date", "sample_id", "stock_code", "score", "rank", "weight"]),
        pd.DataFrame(windows), pd.DataFrame(trials), pd.DataFrame(redundancy, columns=[
        "left", "right", "common_rows", "pooled_rank_correlation", "reason"]), receipt)


class ScreenRunner:
    """Process-local finite call/intents ledger; failed intents are not refunded."""

    def __init__(self, *, max_calls, max_channel_intents):
        _integer(max_calls, 1, 10000, "max_calls")
        _integer(max_channel_intents, 1, 100000, "max_channel_intents")
        self.max_calls, self.max_channel_intents = max_calls, max_channel_intents
        self.calls, self.channel_intents = 0, 0
        self.ledger = []

    def run(self, spec, decisions, assembly, labels):
        if self.calls >= self.max_calls:
            raise ValueError("Screen call budget exhausted")
        self.calls += 1
        event = dict(call=self.calls, state="preparing", channel_intents=0, channels=[])
        self.ledger.append(event)
        try:
            if not isinstance(spec, ScreenSpec):
                raise ValueError("Validated ScreenSpec required")
            spec = ScreenSpec.from_dict(spec.to_dict())
            channels = spec.to_dict()["channels"]
            event.update(protocol_id=spec.protocol_id, spec=spec.to_dict())
            event["channels"] = [dict(c, state="planned") for c in channels]
            if self.channel_intents + len(channels) > self.max_channel_intents:
                raise ValueError("Screen channel intent budget exhausted")
            self.channel_intents += len(channels)
            event["channel_intents"] = len(channels)
            result = screen(spec, decisions, assembly, labels)
            event["state"] = "succeeded"
            for row in event["channels"]:
                row["state"] = "evaluated"
            result.receipt.update(call=self.calls, channel_intents_used=self.channel_intents)
            return result
        except Exception as exc:
            event["state"] = "failed" if event["channel_intents"] else "rejected"
            event["error"] = f"{type(exc).__name__}: {exc}"
            for row in event["channels"]:
                row.update(state=event["state"], error=event["error"])
            raise
