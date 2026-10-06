"""Tests for official SimTradeData orchestration and honest import reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import pandas as pd


@pytest.mark.parametrize('coverage_passes', [True, False])
def test_existing_parquet_reuse_never_downloads_or_exports(tmp_path, monkeypatch, coverage_passes):
    import src.ingestion.update_simtradedata as updater
    import src.data_sources.snapshot_quality as quality
    repo, live = tmp_path/'repo', tmp_path/'cn'
    live.mkdir()
    (live/'manifest.json').write_text(json.dumps({'version':'2026-09-24'}))
    (live/'.source_state.json').write_text(json.dumps({'valuation_policy':'publication-date-v1','status_policy':'dated-flags-v1'}))
    reports = tmp_path/'data/update_reports'
    reports.mkdir(parents=True)
    (reports/'latest.json').write_text(json.dumps({'status':'pass','target_date':'2026-09-24','export_dir':str(live)}))
    monkeypatch.setattr(updater, '_PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(updater, '_load_config', lambda: {})
    monkeypatch.setattr(updater, '_validate_upstream_checkout', lambda *args: None)
    monkeypatch.setattr(updater, '_completed_day_ceiling', lambda day: day)
    monkeypatch.setattr(quality, 'inspect_snapshot', lambda *args: {'status':'pass' if coverage_passes else 'fail'})
    def forbidden(*args, **kwargs):
        raise AssertionError('Acquisition must not run for current verified Parquet')
    monkeypatch.setattr(updater, '_run', forbidden)
    monkeypatch.setattr(updater, '_resolve_update_date', forbidden)
    if coverage_passes:
        result = updater.update_simtradedata('2026-09-24', project_dir=repo, export_dir=live)
        assert result.reused and not result.database_path.exists()
        assert json.loads((reports/'latest.json').read_text())['commands'] == []
    else:
        with pytest.raises(AssertionError):
            updater.update_simtradedata('2026-09-24', project_dir=repo, export_dir=live)

from src.ingestion.update_simtradedata import (
    _is_valid_zip,
    _overlay_auxiliary_snapshot,
    _merge_stock_status,
    _publish_snapshot,
    _validate_integrity_report,
)


def test_integrity_report_must_pass(tmp_path: Path) -> None:
    staging = tmp_path / "cn.staging"
    stocks = staging / "stocks"
    stocks.mkdir(parents=True)
    (stocks / "000001.SZ.parquet").write_bytes(b"fixture")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "fail"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="did not pass"):
        _validate_integrity_report(report, staging)


def test_integrity_report_requires_stock_files(tmp_path: Path) -> None:
    staging = tmp_path / "cn.staging"
    staging.mkdir()
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "pass"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="no stock parquet"):
        _validate_integrity_report(report, staging)


def test_publish_keeps_previous_snapshot(tmp_path: Path) -> None:
    target = tmp_path / "cn"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / "cn.staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")

    backup = _publish_snapshot(staging, target)

    assert backup == tmp_path / "cn.previous"
    assert (target / "new.txt").read_text(encoding="utf-8") == "new"
    assert (backup / "old.txt").read_text(encoding="utf-8") == "old"
    assert not staging.exists()


def test_tdx_zip_validation_rejects_antibot_html(tmp_path: Path) -> None:
    fake = tmp_path / "hsjday.zip"
    fake.write_text("<script>anti-bot challenge</script>", encoding="utf-8")
    assert not _is_valid_zip(fake)


def test_overlay_preserves_old_stock_files_missing_from_new_export(tmp_path: Path) -> None:
    live = tmp_path / "cn"
    staging = tmp_path / "cn.staging"
    (live / "stocks").mkdir(parents=True)
    (live / "valuation").mkdir()
    (staging / "stocks").mkdir(parents=True)
    (live / "stocks" / "old.parquet").write_bytes(b"old")
    (live / "valuation" / "v.parquet").write_bytes(b"valuation")
    (staging / "stocks" / "new.parquet").write_bytes(b"new")

    _overlay_auxiliary_snapshot(live, staging)

    assert (staging / "stocks" / "old.parquet").read_bytes() == b"old"
    assert (staging / "stocks" / "new.parquet").exists()
    assert (staging / "valuation" / "v.parquet").read_bytes() == b"valuation"


def test_stock_status_merge_keeps_old_st_and_prefers_new_halt(tmp_path: Path) -> None:
    previous = tmp_path / "previous.parquet"
    current = tmp_path / "current.parquet"
    pd.DataFrame(
        [
            {"date": "20260101", "status_type": "ST", "symbols": ["A"]},
            {"date": "20260101", "status_type": "HALT", "symbols": ["OLD"]},
        ]
    ).to_parquet(previous, index=False)
    pd.DataFrame(
        [{"date": "20260101", "status_type": "HALT", "symbols": ["NEW"]}]
    ).to_parquet(current, index=False)

    _merge_stock_status(previous, current)

    merged = pd.read_parquet(current).set_index("status_type")
    assert list(merged.loc["ST", "symbols"]) == ["A"]
    assert list(merged.loc["HALT", "symbols"]) == ["NEW"]
