"""Run the official SimTradeData updater and publish a verified snapshot.

This wraps README Option 2 using the locally repaired SimTradeData checkout.
Downloader, writer and Windows/Pandas compatibility fixes are recorded in the
audit patch. Native exports are validated without historical overlays. Failed quality checks never replace the research snapshot.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

import yaml
import duckdb
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("update_simtradedata")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TDX_URL = "https://data.tdx.com.cn/vipdoc/hsjday.zip"


@dataclass(frozen=True)
class SimTradeDataUpdateResult:
    project_dir: Path
    database_path: Path
    export_dir: Path
    previous_export_dir: Path | None
    target_date: str
    reused: bool = False


def update_simtradedata(
    target_date: str,
    *,
    project_dir: str | Path | None = None,
    export_dir: str | Path | None = None,
    python_executable: str | Path | None = None,
) -> SimTradeDataUpdateResult:
    """Locally repaired native pipeline; publish its verified export.

    DuckDB is a mutable acquisition workspace, not the published research
    snapshot. Failed updates may advance its cursors; Parquet stays unchanged.
    Baseline migration is a separate, explicit operation, never a daily overlay.
    """
    from datetime import date
    from src.ingestion.update_contract import (update_lock, write_report, publish_managed,
                                               recover_publication)
    from src.data_sources.snapshot_quality import inspect_snapshot
    from src.data_sources.history_quality import verify_history_preserved
    from src.data_sources.quote_quality import verify_historical_prices

    date.fromisoformat(target_date)
    cfg = _load_config()
    settings = cfg.get("simtradedata", {})
    repo = _resolve_path(project_dir or settings.get("project_dir", "../../SimTradeDataRepo"))
    live = _resolve_path(export_dir or settings.get("export_dir", "../../SimTradeData/data/export/cn"))
    python = _resolve_python(repo, python_executable or settings.get("python_executable"))
    _validate_upstream_checkout(repo, python)
    requested_date = target_date
    if settings.get("market_data_only") or settings.get("download_args"):
        raise ValueError("Only the complete official update is supported; remove market_data_only/download_args overrides")
    db = repo / "data/cn.duckdb"
    staging = live.with_name(live.name + ".staging")
    reports = _PROJECT_ROOT / "data/update_reports"
    report = {"status": "running", "target_date": target_date, "requested_date": requested_date,
              "project_dir": str(repo), "database_path": str(db),
              "export_dir": str(live), "staging_dir": str(staging),
              "method": "Reuse verified Parquet first; native Option 2 daily APIs for missing dates; selective Parquet export",
              "limitations": ["Sparse corporate-action completeness is not certified by coverage checks.",
                              "Official refresh may re-fetch recent financial quarters; daily updates do not read the TDX bulk archive.",
                              "DuckDB is acquisition state; only verified Parquet is published."],
              "commands": []}
    with update_lock(repo / "data/lhb_update.lock"):
        previous_report = _read_json(reports / "latest.json")
        write_report(reports / "latest.json", report)
        def run(args, label):
            report["phase"] = label
            report["commands"].append([str(x) for x in args])
            write_report(reports / "latest.json", report)
            _run(args, cwd=repo, label=label, extra_env={"SIMTRADE_END_DATE": target_date, "SIMTRADE_DAILY_MARKET": "1"})
        def reuse_if_current(day):
            if (previous_report.get('status') != 'pass' or previous_report.get('target_date') != day
                    or previous_report.get('export_dir') != str(live)
                    or _read_json(live/'manifest.json').get('version') != day
                    or _read_json(live/'.source_state.json').get('valuation_policy') != 'publication-date-v1'
                    or _read_json(live/'.source_state.json').get('status_policy') != 'dated-flags-v1'):
                return False
            coverage = inspect_snapshot(live,day,str(cfg.get('defaults',{}).get('start_date',day)))
            if coverage.get('status') != 'pass':
                return False
            report.update(status='pass',phase='reused_published_parquet',target_date=day,
                          coverage=coverage,reused=True,previous_export_dir=None)
            write_report(reports/'latest.json',report)
            logger.info('Existing verified Parquet covers {}; reusing it without acquisition or export',day)
            return True
        try:
            recover_publication(live)
            ceiling = _completed_day_ceiling(requested_date)
            if reuse_if_current(ceiling):
                return SimTradeDataUpdateResult(repo,db,live,None,ceiling,reused=True)
            # The calendar request also opens BaoStock: hold the update lock
            # before any network session, not just the download subprocess.
            report["phase"] = "Resolve completed trading day"
            write_report(reports / "latest.json", report)
            target_date = _resolve_update_date(requested_date, repo, python)
            report["target_date"] = target_date
            write_report(reports / "latest.json", report)
            if reuse_if_current(target_date):
                return SimTradeDataUpdateResult(repo,db,live,None,target_date,reused=True)
            if not db.exists():
                raise RuntimeError('Download database is missing; existing Parquet is retained. Initialize acquisition explicitly before requesting new dates.')
            # Pin import resolution to this checkout, including editable installs.
            run([python, "-c", "from pathlib import Path; import simtradedata; "
                 "from simtradedata.writers.duckdb_writer import DEFAULT_DB_PATH; "
                 "assert Path(simtradedata.__file__).resolve().is_relative_to(Path.cwd()); "
                 "assert Path(DEFAULT_DB_PATH).resolve()==Path('data/cn.duckdb').resolve()"],
                "Verify official module/database paths")
            required = max(int(db.stat().st_size * 0.8), 2 * 1024**3)
            if shutil.disk_usage(live.parent).free < required:
                raise RuntimeError(f"Insufficient disk space for export/checks: need {required:,} free bytes")
            run([python, repo / "scripts/download.py", "--skip-mootdx-ohlcv"],
                "Native daily incremental download (no bulk archive)")
            run([python, "-c", "from scripts.download_mootdx import MootdxDownloader; "
                 "d=MootdxDownloader(); "
                 f"d.require_xdxr_complete({target_date!r}); d.writer.close()"],
                "Verify successful corporate-action refresh for every active stock")
            if staging.exists():
                _safe_remove_tree(staging, expected_parent=live.parent)
            run([python, repo / "scripts/export_parquet.py", "--market", "cn",
                 "--db", db, "--output", staging, "--reuse-source", live], "Native export reusing unchanged Parquet")
            integrity = reports / "official_integrity.json"
            integrity.unlink(missing_ok=True)
            run([python, repo / "scripts/check_integrity.py", "--db-path", db,
                 "--market", "cn", "--target-date", target_date, "--export-dir", staging,
                 "--json-output", integrity, "--strict"], "Official strict integrity")
            _validate_integrity_report(integrity, staging)
            report["coverage"] = inspect_snapshot(staging, target_date,
                str(cfg.get("defaults", {}).get("start_date", target_date)))
            if live.is_dir():
                report["history"] = verify_history_preserved(live, staging)
                report["prices"] = verify_historical_prices(live, staging)
            failures = [key for key in ("coverage", "history", "prices")
                        if report.get(key, {}).get("status") == "fail"]
            if failures:
                raise RuntimeError("Publication refused by " + ", ".join(failures))
            backup = publish_managed(staging, live)
            report.update(status="pass", phase="published", previous_export_dir=str(backup) if backup else None)
            write_report(reports / "latest.json", report)
        except BaseException as exc:
            report.update(status="fail", error=f"{type(exc).__name__}: {exc}")
            write_report(reports / "latest.json", report)
            raise
    from src.utils.calendar import invalidate_cache
    invalidate_cache()
    return SimTradeDataUpdateResult(repo, db, live, backup, target_date)


def _load_config() -> dict:
    path = _PROJECT_ROOT / "config" / "config.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _completed_day_ceiling(requested: str, now=None) -> str:
    """Use Shanghai time; today's complete datasets are requested after 18:00."""
    from datetime import datetime, timedelta, timezone, date
    shanghai = timezone(timedelta(hours=8))
    now = (now or datetime.now(shanghai)).astimezone(shanghai)
    ceiling = now.date() if now.hour >= 18 else now.date() - timedelta(days=1)
    return min(date.fromisoformat(requested), ceiling).isoformat()


def _resolve_update_date(requested: str, repo: Path, python: Path) -> str:
    """Resolve weekends/holidays using SimTradeData's live BaoStock calendar."""
    from datetime import date, timedelta
    import tempfile
    ceiling = _completed_day_ceiling(requested)
    start = (date.fromisoformat(ceiling) - timedelta(days=35)).isoformat()
    with tempfile.TemporaryDirectory(prefix="lhb_calendar_") as tmp:
        result = Path(tmp) / "calendar.json"
        script = (
            "from simtradedata.fetchers.baostock_fetcher import BaoStockFetcher; "
            "from pathlib import Path; import json; f=BaoStockFetcher(); f.login(); "
            f"df=f.fetch_trade_calendar({start!r},{ceiling!r}); f.logout(); "
            "days=df.loc[df['is_trading_day'].astype(str).eq('1'),'calendar_date'].astype(str).tolist(); "
            f"Path({str(result)!r}).write_text(json.dumps(days),encoding='utf-8')"
        )
        _run([python, "-c", script], cwd=repo, label="Resolve completed trading day")
        days = sorted(d for d in json.loads(result.read_text(encoding="utf-8")) if start <= d <= ceiling)
    if not days:
        raise RuntimeError("SimTradeData calendar returned no completed trading day")
    logger.info("Requested through {}; updating through completed trading day {}", requested, days[-1])
    return days[-1]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _resolve_python(repo: Path, configured: str | Path | None) -> Path:
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = (repo / path).resolve()
        return path
    windows_venv = repo / ".venv" / "Scripts" / "python.exe"
    posix_venv = repo / ".venv" / "bin" / "python"
    if windows_venv.exists():
        return windows_venv
    if posix_venv.exists():
        return posix_venv
    return Path(sys.executable)


def _download_tdx_package(destination: Path) -> Path:
    """Download the official TDX bulk ZIP, handling its JS cookie challenge.

    SimTradeData's current stdlib downloader accepts the anti-bot HTML response
    as a 986-byte ZIP.  We only fix transport here; parsing/import remains the
    responsibility of SimTradeData's official ``--tdx-source`` path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = destination.with_suffix(".metadata.json")
    cookie = _tdx_challenge_cookie()
    remote = _tdx_remote_info(cookie)

    if destination.exists() and _is_valid_zip(destination):
        local_meta = _read_json(metadata_path)
        same_size = destination.stat().st_size == remote["size"]
        same_modified = local_meta.get("last_modified") == remote.get("last_modified")
        if same_size and same_modified:
            logger.info("TDX bulk package is current: {}", destination)
            return destination

    temp = destination.with_suffix(".download")
    if temp.exists():
        temp.unlink()
    request = Request(
        _TDX_URL,
        headers={"User-Agent": _user_agent(), "Cookie": cookie},
    )
    logger.info(
        "Downloading official TDX package ({:.1f} MB) to {}",
        remote["size"] / 1024 / 1024,
        destination,
    )
    with urlopen(request, timeout=120) as response, temp.open("wb") as handle:
        downloaded = 0
        next_report = 10
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            percent = int(downloaded * 100 / remote["size"])
            if percent >= next_report:
                logger.info("TDX package download: {}%", percent)
                next_report += 10
    if temp.stat().st_size != remote["size"] or not _is_valid_zip(temp):
        temp.unlink(missing_ok=True)
        raise RuntimeError("TDX download is incomplete or not a valid ZIP archive")
    temp.replace(destination)
    metadata_path.write_text(json.dumps(remote, ensure_ascii=False), encoding="utf-8")
    return destination


def _tdx_challenge_cookie() -> str:
    request = Request(
        _TDX_URL,
        headers={"User-Agent": _user_agent(), "Range": "bytes=0-4095"},
    )
    with urlopen(request, timeout=30) as response:
        body = response.read(4096)
        content_type = response.headers.get("Content-Type", "")
    if body.startswith(b"PK") or "application/zip" in content_type:
        return ""
    text = body.decode("utf-8", errors="replace")
    values = []
    for name in ("WTKkN", "bOYDu", "wyeCN"):
        match = re.search(rf"{name}:(\d+)", text)
        if not match:
            raise RuntimeError("TDX anti-bot challenge format is unsupported")
        values.append(int(match.group(1)))
    ssid = re.search(r"\(t,(\d{6,})\)", text)
    if not ssid:
        raise RuntimeError("TDX anti-bot session id was not found")
    return f"__tst_status={sum(values)}#; EO_Bot_Ssid={ssid.group(1)}"


def _tdx_remote_info(cookie: str) -> dict:
    headers = {"User-Agent": _user_agent()}
    if cookie:
        headers["Cookie"] = cookie
    request = Request(_TDX_URL, method="HEAD", headers=headers)
    with urlopen(request, timeout=30) as response:
        size = int(response.headers.get("Content-Length", "0"))
        content_type = response.headers.get("Content-Type", "")
        modified = response.headers.get("Last-Modified")
    if size < 100 * 1024 * 1024 or "zip" not in content_type.lower():
        raise RuntimeError(
            f"TDX endpoint did not expose a valid bulk ZIP (size={size}, type={content_type})"
        )
    return {"size": size, "last_modified": modified}


def _user_agent() -> str:
    return "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def _is_valid_zip(path: Path) -> bool:
    try:
        return zipfile.is_zipfile(path)
    except OSError:
        return False


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _file_fingerprint(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns
    except OSError:
        return None


def _validate_upstream_checkout(repo: Path, python: Path) -> None:
    required = [
        repo / "pyproject.toml",
        repo / "scripts" / "download.py",
        repo / "scripts" / "import_tdx_day.py",
        repo / "scripts" / "repair_cn_benchmark.py",
        repo / "scripts" / "export_parquet.py",
        repo / "scripts" / "check_integrity.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "SimTradeData完整仓库不可用，缺少：" + ", ".join(missing)
        )
    if not python.exists():
        raise FileNotFoundError(
            f"SimTradeData Python环境不存在：{python}。请先在 {repo} 创建.venv并安装依赖。"
        )


def _run(command: list[str | Path], *, cwd: Path, label: str, extra_env: dict | None = None) -> None:
    rendered = [str(part) for part in command]
    logger.info("{}: {}", label, " ".join(rendered))
    env = os.environ.copy()
    env.update(extra_env or {})
    compat_dir = str(_PROJECT_ROOT / "compat")
    prior_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [compat_dir, str(cwd.resolve())] + ([prior_pythonpath] if prior_pythonpath else [])
    )
    try:
        subprocess.run(rendered, cwd=str(cwd), env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{label} failed with exit code {exc.returncode}") from exc


def _validate_integrity_report(report_path: Path, staging: Path) -> None:
    if not report_path.exists():
        raise RuntimeError("SimTradeData integrity check produced no JSON report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise RuntimeError(
            f"SimTradeData integrity report did not pass: {report_path}"
        )
    stocks = staging / "stocks"
    if not stocks.is_dir() or not next(stocks.glob("*.parquet"), None):
        raise RuntimeError(f"Verified export has no stock parquet files: {stocks}")


def _overlay_auxiliary_snapshot(source: Path, staging: Path) -> None:
    """Preserve old rows and non-null fields across all exported datasets."""
    from src.data_sources.snapshot_merge import preserve_snapshot_history
    if source.is_dir():
        report = preserve_snapshot_history(source, staging)
        logger.info("Historical snapshot preservation: {}", report)


def _merge_stock_status(previous: Path, current: Path) -> None:
    """Keep historical ST rows while preferring newly exported HALT rows."""
    old = pd.read_parquet(previous)
    new = pd.read_parquet(current)
    required = {"date", "status_type", "symbols"}
    if not required.issubset(old.columns) or not required.issubset(new.columns):
        raise RuntimeError("Unexpected SimTradeData stock_status schema")
    merged = pd.concat([old, new], ignore_index=True)
    merged = merged.drop_duplicates(["date", "status_type"], keep="last")
    merged = merged.sort_values(["date", "status_type"], kind="stable")
    merged.to_parquet(current, index=False)


def _validate_daily_kline_snapshot(
    db_path: Path,
    staging: Path,
    target_date: str,
    *,
    min_latest_symbols: int = 3000,
) -> None:
    """Fail closed if the newly imported daily-K snapshot is stale or partial."""
    stocks = staging / "stocks"
    files = list(stocks.glob("*.parquet")) if stocks.is_dir() else []
    if not files:
        raise RuntimeError(f"Export has no stock parquet files: {stocks}")

    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        max_date = connection.execute("SELECT MAX(date) FROM stocks").fetchone()[0]
        parquet_glob = str(stocks / "*.parquet")
        latest_symbols = connection.execute(
            """
            SELECT COUNT(DISTINCT filename)
            FROM read_parquet(?, filename=true)
            WHERE CAST(date AS DATE) = CAST(? AS DATE)
            """,
            [parquet_glob, target_date],
        ).fetchone()[0]
    finally:
        connection.close()

    max_date_text = max_date.isoformat() if max_date is not None else None
    if max_date_text != target_date:
        raise RuntimeError(
            f"SimTradeData daily K is stale: expected {target_date}, actual {max_date_text}"
        )
    if latest_symbols < min_latest_symbols:
        raise RuntimeError(
            "SimTradeData daily K is partial on "
            f"{target_date}: only {latest_symbols:,} symbols, "
            f"minimum required {min_latest_symbols:,}"
        )
    logger.info(
        "Daily-K validation passed: {} symbols on {} ({} parquet files)",
        latest_symbols,
        target_date,
        len(files),
    )


def _publish_snapshot(staging: Path, target: Path) -> Path | None:
    """Atomically swap a verified staging directory into the live location."""
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(f"{target.name}.previous")
    if backup.exists():
        # A retained Release may be the only evidence for unresolved value
        # differences. Publishing an update must not silently delete it.
        from datetime import datetime
        backup = target.with_name(f"{target.name}.previous.{datetime.now():%Y%m%d_%H%M%S_%f}")

    moved_old = False
    try:
        if target.exists():
            target.rename(backup)
            moved_old = True
        staging.rename(target)
    except Exception:
        if moved_old and backup.exists() and not target.exists():
            backup.rename(target)
        raise
    return backup if moved_old else None


def _safe_remove_tree(path: Path, *, expected_parent: Path) -> None:
    resolved = path.resolve()
    parent = expected_parent.resolve()
    if resolved.parent != parent or resolved == parent:
        raise ValueError(f"Refusing to remove unexpected directory: {resolved}")
    shutil.rmtree(resolved)
