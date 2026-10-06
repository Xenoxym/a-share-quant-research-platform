from types import SimpleNamespace
import pandas as pd
import src.ingestion.fetch_lhb_summary as summary


def test_failed_summary_update_keeps_file_and_reports_failure(tmp_path, monkeypatch):
    path = tmp_path / "summary.parquet"
    pd.DataFrame({"trade_date": ["2026-09-22"], "stock_code": ["000001.SZ"]}).to_parquet(path)
    before = path.read_bytes()
    monkeypatch.setattr(summary, "get_trading_dates", lambda *a: ["2026-09-22", "2026-09-23"])
    monkeypatch.setattr(summary, "_load_failures_dir", lambda: tmp_path / "failures")
    def fail(*a):
        raise ConnectionError("source unavailable")
    monkeypatch.setattr(summary, "_get_client", lambda *a: SimpleNamespace(fetch_lhb_summary=fail))
    monkeypatch.setattr(summary, "with_retry", lambda fn, *args: fn(*args))
    result = summary.fetch_lhb_summary("2026-09-22", "2026-09-23", save_path=str(path))
    assert path.read_bytes() == before
    assert result.attrs["update_report"]["status"] == "fail"
    assert result.attrs["update_report"]["unresolved_dates"] == ["2026-09-23"]
