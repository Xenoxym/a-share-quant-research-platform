"""Leakage boundaries, unknown outcomes, and the real frozen ML executor."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import threading
import re
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from src.mlresearch.contracts import MLSpec, FEATURES, benchmark_proposal
from src.mlresearch.dataset import price_features, forward_labels, assign_splits
from src.mlresearch.evaluation import evaluate, SCORES
from src.mlresearch.runner import fit_models
from src.researchops.service import Research
from src.technical.artifacts import digest, write_json

PROJECT = Path(__file__).resolve().parents[1]


def quotes(calendar, code="000001.SZ", phase=0):
    t = np.arange(len(calendar))
    r = 0.0003 + 0.004 * np.sin(t / 6 + phase)
    close = 20 * np.exp(np.cumsum(r))
    return pd.DataFrame(dict(stock_code=code, trade_date=calendar, close=close,
        pre_close=close / np.exp(r), open=close * .998, high=close * 1.01,
        low=close * .99, volume=np.full(len(t), 2000000), amount=close * 2000000,
        is_st=np.zeros(len(t))))


def test_features_do_not_change_when_future_is_appended_or_changed():
    cal = pd.bdate_range("2020-01-01", periods=150).strftime("%Y-%m-%d").tolist()
    raw = quotes(cal)
    prefix = price_features(raw.iloc[:100], cal)
    full = price_features(raw, cal)
    changed = raw.copy()
    changed.loc[100:, ["close", "pre_close", "amount"]] *= 10
    future_changed = price_features(changed, cal)
    for comparison in [full.iloc[:100], future_changed.iloc[:100]]:
        pd.testing.assert_frame_equal(prefix[list(FEATURES)], comparison[list(FEATURES)])


def test_forward_label_uses_market_sessions_and_gap_is_unknown():
    cal = pd.bdate_range("2020-01-01", periods=80).strftime("%Y-%m-%d").tolist()
    raw = quotes(cal)
    labeled = forward_labels(price_features(raw, cal), cal, 5)
    assert labeled.loc[60, "label_return"] == pytest.approx(np.prod(raw.loc[61:65, "close"] / raw.loc[61:65, "pre_close"]) - 1)
    assert labeled.loc[60, "label_end"] == cal[65]
    missing = forward_labels(price_features(raw.drop(index=62), cal), cal, 5)
    assert np.isnan(missing.loc[missing.trade_date.eq(cal[60]), "label_return"].iloc[0])
    assert labeled.tail(5).label_return.isna().all()


def test_label_boundary_purge_not_arbitrary_panel_rows():
    spec = MLSpec(train_start="2020-01-01", train_end="2020-12-31", valid_start="2021-01-01", valid_end="2021-12-31", test_start="2022-01-01", test_end="2022-12-31")
    frame = pd.DataFrame({"trade_date": ["2020-12-25", "2020-12-30", "2021-12-30", "2022-12-30"],
        "label_end": ["2020-12-31", "2021-01-05", "2022-01-06", "2023-01-06"]})
    assert assign_splits(frame, spec).split.tolist() == ["train", "excluded", "excluded", "excluded"]


def test_preprocessing_and_fit_ignore_validation_and_future_label_columns():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(rng.normal(size=(140, len(FEATURES))), columns=list(FEATURES))
    frame["split"] = ["train"] * 120 + ["test"] * 20
    frame["label_observed"] = True
    frame["label_relative"] = frame.return_5 * .01
    frame["future_wealth"] = rng.normal(size=len(frame)) * 1000
    frame.loc[:5, "return_1"] = np.nan
    spec = MLSpec(tree_iterations=3, tree_min_samples=2)
    models = fit_models(frame, spec)
    changed = frame.copy()
    changed.loc[120:, list(FEATURES)] = 10000
    changed.loc[120:, "label_relative"] = -999
    changed["future_wealth"] = -frame.future_wealth
    again = fit_models(changed, spec)
    for name in models:
        assert len(models[name].steps[0][1].statistics_) == len(FEATURES)
        assert np.allclose(models[name].steps[0][1].statistics_, frame.iloc[:120][list(FEATURES)].median())
        assert np.allclose(models[name].predict(frame.iloc[:120][list(FEATURES)]), again[name].predict(frame.iloc[:120][list(FEATURES)]))


def test_unknown_outcome_stays_in_top_selection():
    frame = pd.DataFrame({"split": ["test"] * 10, "trade_date": ["2024-01-05"] * 10,
        "stock_code": [str(i) for i in range(10)], "label_return": np.arange(10) / 100,
        "label_relative": (np.arange(10) - 4.5) / 100, "label_observed": [True] * 10})
    for model in SCORES:
        frame[model] = np.arange(10)
    frame.loc[9, ["label_return", "label_relative"]] = np.nan
    frame.loc[9, "label_observed"] = False
    daily, _, summary, _ = evaluate(frame, MLSpec(top_k=2, bins=2))
    assert daily.top_count.eq(2).all()
    assert daily.top_label_count.eq(1).all()
    assert all(s["top_label_coverage"] == .5 for s in summary)


@pytest.fixture
def ml_research(tmp_path):
    project = tmp_path / "project"
    for package in ["technical", "mlresearch"]:
        shutil.copytree(PROJECT / "src" / package, project / "src" / package, ignore=shutil.ignore_patterns("__pycache__"))
    root = tmp_path / "store"
    sid = "a" * 24
    snapshot = root / "snapshots" / sid
    snapshot.mkdir(parents=True)
    cal = pd.bdate_range("2019-01-01", "2022-06-30").strftime("%Y-%m-%d").tolist()
    codes = [f"00000{i}.SZ" for i in range(1, 9)]
    pd.concat([quotes(cal, code, phase=i / 2) for i, code in enumerate(codes)]).to_parquet(snapshot / "bars.parquet", index=False)
    pd.DataFrame(dict(stock_code=codes, listed_date="2010-01-01", de_listed_date="9999-12-31")).to_parquet(snapshot / "metadata.parquet", index=False)
    reports = []
    for code in codes:
        for period in pd.date_range("2018-12-31", "2022-03-31", freq="QE"):
            reports.append(dict(stock_code=code, report_date=period.strftime("%Y-%m-%d"), publication_date=(period + pd.Timedelta(days=30)).strftime("%Y-%m-%d"), total_shares=1e8, np_parent_company_owners=1e7, total_shareholder_equity=1e8, total_assets=2e8))
    pd.DataFrame(reports).to_parquet(snapshot / "fundamentals.parquet", index=False)
    write_json(snapshot / "calendar.json", cal)
    write_json(snapshot / "manifest.json", {"snapshot_id": sid, "start": cal[0], "end": cal[-1], "counts": {"bars": len(cal) * len(codes), "fundamentals": len(reports)},
        "artifacts": {p.name: digest(p) for p in snapshot.iterdir()}})
    research = Research(project, root)
    task = research.create({"title": "ML冻结验收", "question": "时间隔离", "rationale": "工程测试", "success_criteria": ["数值核查通过"], "stop_criteria": ["一次"], "max_experiments": 2})
    session = research.store.claim(task["id"], "builder")
    proposal = benchmark_proposal(sid)
    proposal["spec"] = MLSpec(train_end="2020-12-31", valid_start="2021-01-01", valid_end="2021-12-31", test_start="2022-01-01", test_end="2022-06-30", min_avg_amount=0, tree_iterations=3, tree_min_samples=2, bins=2, top_k=2).to_dict()
    return research, session, proposal


def test_native_frozen_ml_executor_and_typed_submission(ml_research):
    research, session, proposal = ml_research
    experiment = research.register(session, proposal)
    assert research.register(session, {**proposal, "hypothesis": "描述不同不应重跑"})["id"] == experiment["id"]
    (research.project / "src/mlresearch/runner.py").write_text('raise RuntimeError("changed development code")')
    result = research.execute(session, experiment["id"])
    assert result["status"] == "completed", result
    assert result["result"]["kind"] == "ml"
    assert result["result"]["audit"]["prediction_sample_max_error"]["ridge"] < 1e-10
    from src.technical.server import make_server
    server = make_server(research.project, port=0, root=research.root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(url) as response:
            page = response.read().decode()
        assert 'data-view="ml"' in page
        boot = json.loads(re.search(r"window.TECHNICAL=(.*?);</script>", page).group(1))
        with urlopen(url + "/api/ml/plan") as response:
            assert json.load(response)["kind"] == "ml"
        with urlopen(url + "/api/ml/" + result["run_id"]) as response:
            assert json.load(response)["summaries"]
        with urlopen(url + "/api/ml/" + result["run_id"] + "/download?file=REPORT.md") as response:
            assert response.read()
        with pytest.raises(HTTPError):
            urlopen(url + "/api/ml/" + result["run_id"] + "/download?file=../../input.json")
        with pytest.raises(HTTPError) as unauthorized:
            urlopen(Request(url + "/api/ml/benchmark", data=b"{}"))
        assert unauthorized.value.code == 403
        with pytest.raises(HTTPError) as override:
            urlopen(Request(url + "/api/ml/benchmark", data=b'{"hidden":true}', headers={"X-Research-Token": boot["token"]}))
        assert override.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    submitted = research.submit(session, {"summary": "ML验收完成", "findings": ["核查通过"], "limitations": ["合成"], "next_steps": ["另一角色复核"]})
    assert submitted["submission"]["evidence_kind"] == "registered_ml_analysis"
    reviewed = research.review(session["task_id"], "reviewer", "stop", "工程验收，非投资结论")
    assert reviewed["status"] == "completed"
    child = research.create({"title": "ML接续", "question": "接力", "rationale": "已存在基准", "success_criteria": ["可读"], "stop_criteria": ["一次"], "baseline_runs": [result["run_id"]]})
    assert research.context(child["id"])["baselines"][0]["kind"] == "ml"


def test_tampered_ml_input_fails_before_launch(ml_research):
    research, session, proposal = ml_research
    experiment = research.register(session, proposal)
    path = research.root / "worker_jobs" / experiment["id"] / "input.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["spec"]["test_start"] = "2020-01-01"
    write_json(path, cfg)
    with pytest.raises(ValueError, match="冻结输入"):
        research.execute(session, experiment["id"])
    assert research.store.experiment(experiment["id"])["status"] == "failed"


def test_ml_rejects_cost_override_and_invalid_date_order(ml_research):
    research, session, proposal = ml_research
    with pytest.raises(ValueError):
        research.register(session, {**proposal, "scenarios": ["configured"]})
    with pytest.raises(ValueError):
        MLSpec(test_start="2020-01-01")


def test_completed_ml_manifest_cannot_be_replaced_even_with_valid_artifacts(ml_research):
    research, session, proposal = ml_research
    experiment = research.register(session, proposal)
    outcome = research.execute(session, experiment["id"])
    assert outcome["status"] == "completed"
    from src.researchops.ml_experiments import verify_completed
    verify_completed(research, outcome)
    path = research.root / "ml_runs" / outcome["run_id"] / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["extra_unregistered_note"] = "Artifact hashes still match, but receipt identity must not."
    write_json(path, manifest)
    with pytest.raises(ValueError, match="清单与已核验收据"):
        verify_completed(research, outcome)
