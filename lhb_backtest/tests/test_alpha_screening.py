"""Hand counterexamples for screening, not market strategies or account backtests."""
import copy

import numpy as np
import pandas as pd
import pytest

from src.alpharesearch.assembly import FeatureAssembly
from src.alpharesearch.contracts import MissingReason as M
from src.alpharesearch.features.base import FeatureBlock
from src.alpharesearch.registry import FeatureDefinition, FeatureRegistry
from src.alpharesearch.screening import ScreenSpec, ScreenRunner, screen


def example():
    rows = []
    for day in ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-08", "2024-01-09"]:
        for i in range(4):
            rows.append(dict(sample_id=f"{day}-{i}", trade_date=day, stock_code=f"{600000+i:06}.SH",
                decision_at=f"{day}T09:00:00+08:00", execution_at=f"{day}T09:30:00+08:00",
                target_end_at=f"{day}T15:00:00+08:00"))
    decisions = pd.DataFrame(rows)
    names = ["test.first", "test.second", "test.interaction", "test.copy"]
    registry = FeatureRegistry([
        FeatureDefinition(n, "ratio", "stock_day", "synthetic", ("synthetic",), "{}",
            (("synthetic.py", "a"*64),), "by decision", "explicit") for n in names])
    keys = ["sample_id", "trade_date", "stock_code"]
    values = decisions[keys].copy()
    values[names[0]] = np.tile([-1., -1., 1., 1.], 5)
    values[names[1]] = np.tile([-1., 1., -1., 1.], 5)
    values[names[2]] = values[names[0]]*values[names[1]]
    values[names[3]] = values[names[0]]
    values["observed_end"] = values["known_at"] = decisions.decision_at
    missing = decisions[keys].copy()
    for n in names:
        missing[n] = M.PRESENT.value
    bindings = (dict(namespace="test", materialization_id="b"*64, source_id="synthetic",
        source_version="synthetic-v1", vintage="market_observation",
        definition_ids=[d.definition_id for d in registry.definitions]),)
    block = FeatureBlock(values, missing, {n: "ratio" for n in names},
        dict(schema="assembled-feature-matrix-v1", key_columns=keys, source_id="synthetic",
             version_id="c"*64, vintage="market_observation", universe_id="synthetic_universe",
             registry_version=registry.version_id, source_bindings=list(bindings)))
    assembly = FeatureAssembly(block, registry, bindings)
    labels = decisions[keys].copy()
    labels["target_return"] = values["test.interaction"]*.1
    labels["label_start"] = decisions.execution_at
    labels["label_end"] = labels["label_known_at"] = decisions.target_end_at
    labels["label_observed"] = True
    channels = [dict(channel_id=cid, feature_key=feature, family="core", direction=1, condition=None,
        route=route, exploration_reason="preserve possible interaction" if route=="exploration" else "",
        hypothesis="Synthetic XOR mechanics only") for cid, feature, route in [
            ("first_merit", "test.first", "merit"), ("second_merit", "test.second", "merit"),
            ("interaction_merit", "test.interaction", "merit"), ("first_explore", "test.first", "exploration")]]
    spec = ScreenSpec.from_dict(dict(schema="development-screen-v1", name="synthetic_screen",
        dataset_id="synthetic_dataset", universe_id="synthetic_universe", development_start="2024-01-02",
        development_end="2024-01-04", holdout_start="2024-01-08",
        diagnostic_cutoff="2024-01-07T23:00:00+08:00", target="execution_window_return_decimal",
        allow_weak_vintage=False, channels=channels, top_k=1, min_pairs=3, min_coverage=.5,
        min_condition_rows=1, min_proxy_windows=3, max_input_rows=1000, max_matrix_cells=10000,
        max_channels=32, max_redundancy_pairs=10, family_quotas={"core": {"merit":2, "exploration":1}}))
    return spec, decisions, assembly, labels


def changed(data, **fields):
    spec = data[0].to_dict(); spec.update(fields)
    return [ScreenSpec.from_dict(spec), *data[1:]]


def test_xor_marginals_interaction_and_predeclared_low_ic_exploration():
    result = screen(*example()); t = result.trials.set_index("channel_id")
    assert t.loc["first_merit", "rank_ic_mean"] == pytest.approx(0)
    assert t.loc["second_merit", "rank_ic_mean"] == pytest.approx(0)
    assert t.loc["interaction_merit", "rank_ic_mean"] == pytest.approx(1)
    assert set(t.loc[t.promoted].index) == {"interaction_merit", "first_explore"}
    assert t.loc["interaction_merit", "proxy_total_gross_return"] == pytest.approx(1.1**3-1)
    assert t.loc["first_merit", "proxy_endpoint_max_drawdown"] == pytest.approx(1-.9**3)
    assert result.receipt["candidate_direction_condition_intents"] == 4
    assert not result.receipt["account_results"] and not result.receipt["durable_executor"]
    assert not result.receipt["complete_intraperiod_drawdown"]
    assert set(result.selections.loc[result.selections.channel_id.eq("interaction_merit"), "stock_code"]) == {"600000.SH"}


def test_future_target_values_and_future_time_are_not_screen_inputs():
    data = list(example()); first = screen(*data)
    future = data[-1].trade_date.ge("2024-01-08")
    data[-1]["target_return"] = data[-1].target_return.astype(object)
    data[-1].loc[future, "target_return"] = "unread future target"
    data[-1].loc[future, "label_known_at"] = "unread future clock"
    second = screen(*data)
    for attr in ("daily", "trials", "selections", "windows", "redundancy"):
        pd.testing.assert_frame_equal(getattr(first, attr), getattr(second, attr))
    assert first.receipt["consumed_outcomes"] == second.receipt["consumed_outcomes"]


def test_immature_values_are_ignored_and_selected_unknown_means_unknown_window():
    data = list(example()); late = data[-1].trade_date.eq("2024-01-04")
    data[-1].loc[late, "label_known_at"] = "2024-01-10T15:00:00+08:00"
    first = screen(*data)
    data[-1]["target_return"] = data[-1].target_return.astype(object)
    data[-1].loc[late, "target_return"] = "unread immature return"
    second = screen(*data)
    pd.testing.assert_frame_equal(first.trials, second.trials)
    pd.testing.assert_frame_equal(first.selections, second.selections)
    w = first.windows.loc[first.windows.trade_date.eq("2024-01-04")]
    assert w.window_gross_return.isna().all() and w.endpoint_nav.isna().all()
    assert not first.trials.proxy_complete.any()


def test_omitted_selected_label_never_changes_selection_or_reweights_other_stock():
    data = list(changed(example(), top_k=2))
    first = screen(*data)
    data[-1] = data[-1].loc[~(data[-1].stock_code.eq("600000.SH") & data[-1].trade_date.eq("2024-01-03"))]
    second = screen(*data)
    pd.testing.assert_frame_equal(first.selections, second.selections)
    w = second.windows.loc[second.windows.channel_id.eq("interaction_merit") & second.windows.trade_date.eq("2024-01-03")].iloc[0]
    assert w.selected_unknown_targets == 1 and pd.isna(w.window_gross_return)
    assert pd.isna(w.endpoint_nav)
    t = second.trials.set_index("channel_id").loc["interaction_merit"]
    assert not t.promoted and t.selection_reason == "incomplete_common_proxy_domain"


def test_cash_slots_ties_and_declared_direction_are_not_implicit_shorting():
    result = screen(*changed(example(), top_k=5))
    assert np.allclose(result.windows.cash_weight, .2, rtol=0, atol=1e-15)
    assert result.windows.window_gross_return.eq(0).all()
    assert result.selections.weight.eq(.2).all()
    data = example(); cfg = data[0].to_dict()
    cfg["channels"][0]["direction"] = -1
    reverse = screen(ScreenSpec.from_dict(cfg), *data[1:])
    first = reverse.selections.loc[reverse.selections.channel_id.eq("first_merit")]
    assert set(first.stock_code) == {"600000.SH"}
    assert first.weight.eq(1).all()


def test_condition_unknown_denominator_is_not_no_event_false():
    data = list(example()); cfg = data[0].to_dict()
    cfg["channels"][0]["condition"] = dict(feature_key="test.second", op="gt", threshold=0)
    data[0] = ScreenSpec.from_dict(cfg)
    missing = data[2].block.values.stock_code.eq("600000.SH")
    data[2].block.values.loc[missing, "test.second"] = np.nan
    data[2].block.missing.loc[missing, "test.second"] = M.UNCOVERED.value
    result = screen(*data)
    d = result.daily.loc[result.daily.channel_id.eq("first_merit")]
    assert d.condition_true.eq(2).all() and d.condition_false.eq(1).all()
    assert d.condition_unknown.eq(1).all() and d.decision_rows.eq(4).all()
    assert d.eligible_rows.eq(2).all() and d.rank_ic.isna().all()


def test_redundancy_copy_budget_and_same_feature_channels_remain_counted():
    data = example(); cfg = data[0].to_dict()
    copy_channel = copy.deepcopy(cfg["channels"][0])
    copy_channel.update(channel_id="copy_merit", feature_key="test.copy")
    cfg["channels"].append(copy_channel)
    result = screen(ScreenSpec.from_dict(cfg), *data[1:])
    pair = result.redundancy.loc[(result.redundancy.left=="test.copy") & (result.redundancy.right=="test.first")].iloc[0]
    assert pair.pooled_rank_correlation == pytest.approx(1)
    assert result.receipt["candidate_direction_condition_intents"] == 5
    limited = screen(*changed([ScreenSpec.from_dict(cfg), *data[1:]], max_redundancy_pairs=0))
    assert limited.redundancy.empty and limited.receipt["redundancy_pairs_omitted"] == 6


@pytest.mark.parametrize("fault", ["feature_time", "mixed_window", "overlap", "label_key",
    "label_interval", "nonbool_observed", "future_known_before_end", "invalid_return",
    "weak_vintage", "registry_identity", "definition_unit", "missing_reason", "input_rows", "matrix_cells"])
def test_scoped_validation_and_budget_reject_invalid_inputs(fault):
    data = list(example())
    if fault=="feature_time":data[2].block.values.loc[0,"known_at"]="2024-01-02T10:00:00+08:00"
    if fault=="mixed_window":data[1].loc[0,"target_end_at"]="2024-01-02T14:00:00+08:00"
    if fault=="overlap":data[1].loc[data[1].trade_date.eq("2024-01-02"),"target_end_at"]="2024-01-03T15:00:00+08:00"
    if fault=="label_key":data[-1].loc[0,"sample_id"]="wrong-sample"
    if fault=="label_interval":data[-1].loc[0,"label_start"]="2024-01-02T10:00:00+08:00"
    if fault=="nonbool_observed":data[-1]["label_observed"]=data[-1].label_observed.astype(object);data[-1].loc[0,"label_observed"]="yes"
    if fault=="future_known_before_end":data[-1].loc[0,"label_known_at"]="2024-01-02T14:00:00+08:00"
    if fault=="invalid_return":data[-1].loc[0,"target_return"]=-1.01
    if fault=="weak_vintage":data[2].block.metadata["vintage"]="latest_only"
    if fault=="registry_identity":data[2].block.metadata["registry_version"]="d"*64
    if fault=="definition_unit":data[2].block.units["test.first"]="binary"
    if fault=="missing_reason":data[2].block.missing.loc[0,"test.first"]=M.UNCOVERED.value
    if fault=="input_rows":data=changed(data,max_input_rows=1)
    if fault=="matrix_cells":data=changed(data,max_matrix_cells=1)
    with pytest.raises(ValueError):screen(*data)


def test_unknown_constant_diagnostics_and_all_failed_intents_are_retained():
    data = list(example()); data[-1]["target_return"] = .1
    result = screen(*data)
    assert result.daily.rank_ic.isna().all()
    assert set(result.daily.rank_ic_reason) == {"constant_score_or_target"}
    bad = changed(example(), max_input_rows=1)
    runner = ScreenRunner(max_calls=2, max_channel_intents=4)
    with pytest.raises(ValueError,match="row budget"):runner.run(*bad)
    assert runner.channel_intents == 4 and runner.ledger[0]["state"]=="failed"
    assert len(runner.ledger[0]["channels"])==4
    with pytest.raises(ValueError,match="intent budget"):runner.run(*example())
    assert runner.channel_intents == 4 and runner.calls == 2
    assert len(runner.ledger)==2 and [e["state"] for e in runner.ledger]==["failed", "rejected"]
    with pytest.raises(ValueError,match="call budget"):runner.run(*example())


@pytest.mark.parametrize("fault", ["unknown", "top_k_bool", "direction_bool", "duplicate",
    "target", "condition_target", "explore_reason", "quota", "cutoff", "nan_threshold"])
def test_spec_rejects_ambiguous_or_unbounded_protocol(fault):
    value = example()[0].to_dict()
    if fault=="unknown":value["future_magic"]=True
    if fault=="top_k_bool":value["top_k"]=True
    if fault=="direction_bool":value["channels"][0]["direction"]=True
    if fault=="duplicate":value["channels"].append(copy.deepcopy(value["channels"][0]))
    if fault=="target":value["target"]="future_close_price"
    if fault=="condition_target":value["channels"][0]["condition"]={"target_return":">0"}
    if fault=="explore_reason":value["channels"][-1]["exploration_reason"]=""
    if fault=="quota":value["family_quotas"]={}
    if fault=="cutoff":value["diagnostic_cutoff"]="2024-01-08T23:00:00+08:00"
    if fault=="nan_threshold":value["channels"][0]["condition"]=dict(feature_key="test.first",op="gt",threshold=float("nan"))
    with pytest.raises(ValueError):ScreenSpec.from_dict(value)


def test_failed_and_rejected_ledgers_keep_complete_different_channel_definitions():
    runner = ScreenRunner(max_calls=3, max_channel_intents=8)
    left = changed(example(), max_input_rows=1)
    cfg = left[0].to_dict()
    cfg["channels"][0]["direction"] = -1
    cfg["channels"][0]["condition"] = dict(feature_key="test.second", op="ge", threshold=1)
    right = [ScreenSpec.from_dict(cfg), *left[1:]]
    for data in (left, right):
        with pytest.raises(ValueError, match="row budget"):
            runner.run(*data)
    with pytest.raises(ValueError, match="intent budget"):
        runner.run(*example())
    assert runner.channel_intents == 8
    assert [e["state"] for e in runner.ledger] == ["failed", "failed", "rejected"]
    assert runner.ledger[0]["protocol_id"] != runner.ledger[1]["protocol_id"]
    for event, data in zip(runner.ledger, (left, right, example())):
        assert event["spec"] == data[0].to_dict()
        assert event["protocol_id"] == data[0].protocol_id
        for row, declared in zip(event["channels"], data[0].to_dict()["channels"]):
            assert {key: row[key] for key in declared} == declared
    # Changing a caller's copy cannot alter stored protocol or nested conditions.
    cfg["channels"][0]["condition"]["threshold"] = 999
    assert runner.ledger[1]["channels"][0]["condition"]["threshold"] == 1


def test_empty_selection_and_disabled_redundancy_keep_output_columns():
    data = list(example()); cfg = data[0].to_dict()
    cfg["max_redundancy_pairs"] = 0
    for c in cfg["channels"]:
        c["condition"] = dict(feature_key="test.first", op="gt", threshold=2)
    result = screen(ScreenSpec.from_dict(cfg), *data[1:])
    assert result.selections.empty
    assert list(result.selections) == ["channel_id", "trade_date", "sample_id", "stock_code",
                                       "score", "rank", "weight"]
    assert result.redundancy.empty
    assert list(result.redundancy) == ["left", "right", "common_rows", "pooled_rank_correlation", "reason"]
    assert result.windows.cash_weight.eq(1).all()
    assert result.windows.window_gross_return.eq(0).all()
    assert not result.trials.promoted.any()


@pytest.mark.parametrize("huge,top_k", [(1e308, 4), (1e200, 1)])
def test_finite_large_returns_never_emit_infinity_and_overflow_has_reason(huge, top_k):
    import json

    data = list(changed(example(), top_k=top_k))
    data[-1]["target_return"] = huge
    result = screen(*data)
    first = result.windows.groupby("channel_id", sort=False).head(1)
    assert np.isfinite(first.window_gross_return).all()
    assert np.allclose(first.window_gross_return, huge, rtol=1e-14, atol=0)
    for channel, rows in result.windows.groupby("channel_id", sort=False):
        assert rows.iloc[1].unknown_reason == "numeric_nav_overflow"
        assert rows.iloc[2].unknown_reason == "prior_path_unknown"
        assert rows.iloc[1:].endpoint_nav.isna().all()
    for frame in (result.daily, result.selections, result.windows, result.trials, result.redundancy):
        # Strict JSON tolerates null; never NaN/Infinity values in numeric output.
        clean = frame.astype(object).where(pd.notna(frame), None)
        json.dumps(clean.to_dict("records"), default=str, allow_nan=False)
    assert not result.trials.proxy_complete.any()
    assert not result.trials.loc[result.trials.route.eq("merit"), "promoted"].any()


@pytest.mark.parametrize("domain,unit,bool_dtype,accepted", [
    ("report", "ratio", False, False),
    ("stock_day", "ratio", True, False),
    ("market_day", "ratio", False, True),
    ("stock_day", "binary", True, True),
])
def test_domain_projection_and_boolean_unit_contract(domain, unit, bool_dtype, accepted):
    from dataclasses import replace

    data = list(example()); assembly = data[2]
    definitions = [replace(d, domain=domain, unit=unit) if d.key=="test.first" else d
                   for d in assembly.registry.definitions]
    assembly.registry = FeatureRegistry(definitions)
    assembly.block.units["test.first"] = unit
    if bool_dtype:
        assembly.block.values["test.first"] = assembly.block.values["test.first"].gt(0)
    assembly.source_bindings = (dict(assembly.source_bindings[0],
        definition_ids=[d.definition_id for d in definitions]),)
    assembly.block.metadata.update(registry_version=assembly.registry.version_id,
                                   source_bindings=list(assembly.source_bindings))
    if accepted:
        screen(*data)
    else:
        with pytest.raises(ValueError, match="domain|Boolean"):
            screen(*data)


def test_full_loss_roundoff_never_creates_negative_nav_or_more_than_total_drawdown():
    data = list(changed(example(), top_k=20))
    for position in (1, 3):
        frame = data[position]
        pieces = []
        for duplicate in range(5):
            f = frame.copy()
            f["sample_id"] = f.sample_id + "-" + str(duplicate)
            f["stock_code"] = f.stock_code.map(lambda c: f"{int(c[:6])+duplicate*4:06}.SH")
            pieces.append(f)
        data[position] = pd.concat(pieces, ignore_index=True)
    for attr in ("values", "missing"):
        pieces = []
        frame = getattr(data[2].block, attr)
        for duplicate in range(5):
            f = frame.copy()
            f["sample_id"] = f.sample_id + "-" + str(duplicate)
            f["stock_code"] = f.stock_code.map(lambda c: f"{int(c[:6])+duplicate*4:06}.SH")
            pieces.append(f)
        setattr(data[2].block, attr, pd.concat(pieces, ignore_index=True))
    data[-1]["target_return"] = -1.
    data[-1].loc[data[-1].trade_date.gt("2024-01-02"), "target_return"] = 1e308
    result = screen(*data)
    for _, rows in result.windows.groupby("channel_id", sort=False):
        assert rows.iloc[0].window_gross_return == -1
        assert rows.endpoint_nav.eq(0).all()
        assert rows.endpoint_drawdown.eq(1).all()
    assert result.trials.proxy_complete.all()
    assert result.trials.proxy_total_gross_return.eq(-1).all()


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("container", ["native", "object", "categorical"])
def test_official_source_assembly_preserves_boolean_unit_contract(binary, container):
    from src.alpharesearch.assembly import assemble_features
    from src.alpharesearch.features.primitives import daily_primitives

    spec, decisions, _, labels = example()
    decisions = decisions.copy()
    days = pd.to_datetime(decisions.trade_date)
    decisions["decision_at"] = decisions.trade_date+"T16:00:00+08:00"
    next_days = (days+pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    decisions["execution_at"] = next_days+"T09:30:00+08:00"
    decisions["target_end_at"] = next_days+"T15:00:00+08:00"
    labels = labels.copy()
    labels["label_start"] = decisions.execution_at
    labels["label_end"] = labels["label_known_at"] = decisions.target_end_at
    bars = decisions[["trade_date","stock_code"]].assign(
        open=10., high=11., low=9., close=10., pre_close=10.,
        volume=100., amount=1000., high_limit=11., low_limit=9., limit_price_valid=True)
    source = daily_primitives(bars, source_id="synthetic_source", version_id="synthetic_v1")
    field = "positive_volume_observed" if binary else "close"
    source.values[field] = source.values[field].astype(bool)
    if container == "object":
        source.values[field] = source.values[field].astype(object)
    elif container == "categorical":
        source.values[field] = pd.Categorical(source.values[field],
            categories=pd.Index([True, 2.0], dtype=object))
    key = "daily."+field
    def run():
        assembly = assemble_features(decisions, [source], universe_id="synthetic_universe",
                                     block_ids={"daily":"e"*64}, selected_keys=[key])
        cfg = spec.to_dict()
        cfg["channels"] = [dict(cfg["channels"][0], channel_id="source_binary", feature_key=key)]
        return screen(ScreenSpec.from_dict(cfg), decisions, assembly, labels)
    if binary:
        result = run()
        assert len(result.trials) == 1 and result.trials.status.eq("evaluated").all()
    else:
        with pytest.raises(ValueError, match="Boolean"):
            run()


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("container", ["native", "object", "categorical"])
def test_official_source_rejects_complex_before_float_conversion(binary, container):
    from src.alpharesearch.assembly import assemble_features
    from src.alpharesearch.features.primitives import daily_primitives

    _, decisions, _, _ = example()
    decisions = decisions.assign(decision_at=decisions.trade_date+"T16:00:00+08:00")
    bars = decisions[["trade_date","stock_code"]].assign(
        open=10., high=11., low=9., close=10., pre_close=10.,
        volume=100., amount=1000., high_limit=11., low_limit=9., limit_price_valid=True)
    source = daily_primitives(bars, source_id="synthetic_source", version_id="synthetic_v1")
    field = "positive_volume_observed" if binary else "close"
    values = pd.Series([1.+2.j]*len(source.values))
    if container == "object":
        values = values.astype(object)
    elif container == "categorical":
        values = pd.Series(pd.Categorical(values))
    source.values[field] = values
    with pytest.raises(ValueError, match="Complex"):
        assemble_features(decisions, [source], universe_id="synthetic_universe",
                          block_ids={"daily":"e"*64}, selected_keys=["daily."+field])
