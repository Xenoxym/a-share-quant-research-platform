"""Per-dataset coverage gate for publishing a research snapshot."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


def inspect_snapshot(root: Path, target_date: str, start_date: str) -> dict:
    root = Path(root)
    report = {"target_date": target_date, "start_date": start_date, "errors": [], "coverage": {}}
    state_path=root/'.source_state.json'
    if state_path.exists():
        state=json.loads(state_path.read_text(encoding='utf-8'))
        if state.get('status_policy')=='unverified':
            report['errors'].append('Historical status flags have not passed source reconciliation')
        if state.get('status_policy')=='dated-flags-v1':
            if start_date < str(state.get('status_history_start')) or target_date > str(state.get('status_history_end')):
                report['errors'].append('Requested range exceeds verified historical status coverage')
    def dates(path: Path) -> set[str]:
        if not path.exists():
            report["errors"].append(f"Missing {path.relative_to(root)}")
            return set()
        df = pd.read_parquet(path)
        return set(pd.to_datetime(df["date"].astype(str), errors="coerce").dropna().dt.strftime("%Y-%m-%d"))
    calendar = dates(root / "metadata/trade_days.parquet")
    expected = {d for d in calendar if start_date <= d <= target_date}
    if target_date not in calendar:
        report["errors"].append("Trading calendar does not cover target date")
    benchmark = dates(root / "metadata/benchmark.parquet")
    report["coverage"]["benchmark_end"] = max(benchmark, default=None)
    if expected - benchmark:
        report["errors"].append(f"Benchmark missing {len(expected - benchmark)} requested days")
    path = root / "metadata/stock_status.parquet"
    status = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=["date", "status_type"])
    for kind in ("ST", "HALT"):
        part = status.loc[status.status_type.eq(kind), "date"]
        present = set(pd.to_datetime(part.astype(str), errors="coerce").dropna().dt.strftime("%Y-%m-%d"))
        missing = sorted(expected - present)
        report["coverage"][kind] = {"end": max(present, default=None), "missing_dates": missing}
        if missing:
            report["errors"].append(f"{kind} status unknown on {len(missing)} requested days")
    for folder in ("valuation", "exrights"):
        files = list((root / folder).glob("*.parquet"))
        if not files:
            report["errors"].append(f"Missing {folder} data")
            continue
        with duckdb.connect() as conn:
            row = conn.execute("SELECT min(date)::varchar, max(date)::varchar, count(*) FROM read_parquet(?, union_by_name=true)",
                               [str(root / folder / "*.parquet")]).fetchone()
        report["coverage"][folder] = {"start": row[0], "end": row[1], "rows": row[2]}
        # Exrights are sparse events (no event on a date is valid). A newest
        # event date cannot prove completeness; do not invent such a check.
        if folder == "valuation" and (row[1] is None or row[1][:10] < target_date):
            report["errors"].append("Valuation data is stale")
    # A single updated symbol cannot establish whole-market valuation freshness.
    stock_files = list((root / 'stocks').glob('*.parquet'))
    if not stock_files:
        report['errors'].append('Missing stocks data')
    valuation_files = list((root / 'valuation').glob('*.parquet'))
    if stock_files and valuation_files:
        with duckdb.connect() as conn:
            columns = {r[0] for r in conn.execute('DESCRIBE SELECT * FROM read_parquet(?, union_by_name=true)',
                        [str(root/'valuation/*.parquet')]).fetchall()}
            if 'turnover_rate' not in columns:
                report['errors'].append('Valuation schema has no turnover_rate')
            else:
                missing = conn.execute('''
                    WITH s AS (SELECT regexp_extract(replace(filename,chr(92),'/'),'([^/]+)$',1) symbol,
                               date, volume FROM read_parquet(?,filename=true,union_by_name=true)),
                         v AS (SELECT regexp_extract(replace(filename,chr(92),'/'),'([^/]+)$',1) symbol,
                               date, turnover_rate FROM read_parquet(?,filename=true,union_by_name=true))
                    SELECT count(*) FROM s LEFT JOIN v USING(symbol,date)
                    WHERE CAST(s.date AS DATE)=CAST(? AS DATE) AND s.volume>0
                    AND regexp_matches(s.symbol,'^(6[0-9]{5}\\.SS|(?:00|30)[0-9]{4}\\.SZ)\\.parquet$')
                    AND (v.date IS NULL OR v.turnover_rate IS NULL)
                ''',[str(root/'stocks/*.parquet'),str(root/'valuation/*.parquet'),target_date]).fetchone()[0]
                report['coverage']['latest_trading_symbols_without_valuation'] = missing
                if missing:
                    report['errors'].append(f'{missing} trading stocks lack target-day valuation/turnover')
    if stock_files:
        try:
            universe = inspect_active_universe(root,target_date)
            report['coverage']['active_universe'] = universe
            if universe['unexplained_missing_symbols']:
                report['errors'].append(f"{len(universe['unexplained_missing_symbols'])} active stocks lack bars without a halt explanation")
        except (ValueError,KeyError) as exc:
            report['errors'].append(str(exc))
    report["status"] = "fail" if report["errors"] else "pass"
    report["limitations"] = ["Coverage is not a vendor-level completeness attestation; sparse corporate actions require separate reconciliation."]
    return report


def inspect_active_universe(root: Path, target_date: str) -> dict:
    """Reconcile expected listed shares with bars and the dated halt list."""
    root = Path(root)
    metadata = root / 'metadata/stock_metadata.parquet'
    status = root / 'metadata/stock_status.parquet'
    if not metadata.exists() or not status.exists():
        raise ValueError('Stock metadata/status is required for active-universe reconciliation')
    with duckdb.connect() as conn:
        active = {row[0] for row in conn.execute(r"""
            SELECT symbol FROM read_parquet(?)
            WHERE TRY_CAST(listed_date AS DATE)<=CAST(? AS DATE)
              AND (TRY_CAST(de_listed_date AS DATE)>CAST(? AS DATE)
                   OR TRY_CAST(de_listed_date AS DATE) IS NULL)
              AND regexp_matches(symbol,'^(6[0-9]{5}\.SS|(?:00|30)[0-9]{4}\.SZ)$')
        """, [str(metadata),target_date,target_date]).fetchall()}
        available = {row[0] for row in conn.execute(r"""
            SELECT DISTINCT regexp_extract(replace(filename,chr(92),'/'),'([^/]+)\.parquet$',1)
            FROM read_parquet(?,filename=true,union_by_name=true)
            WHERE CAST(date AS DATE)=CAST(? AS DATE)
        """, [str(root/'stocks/*.parquet'),target_date]).fetchall()}
    statuses = pd.read_parquet(status)
    dates = pd.to_datetime(statuses['date'].astype(str),errors='coerce').dt.strftime('%Y-%m-%d')
    halt_rows = statuses.loc[dates.eq(target_date) & statuses.status_type.eq('HALT'),'symbols']
    if len(halt_rows)!=1:
        raise ValueError('Exactly one target-day HALT record is required')
    raw_halts = halt_rows.iloc[0]
    if isinstance(raw_halts,(str,bytes,bytearray)):
        raw_halts = json.loads(raw_halts)
    elif hasattr(raw_halts,'tolist'):
        raw_halts = raw_halts.tolist()
    if not isinstance(raw_halts,(list,tuple)) or not all(isinstance(x,str) for x in raw_halts):
        raise ValueError('HALT symbols must be an array of stock symbols')
    halted = set(raw_halts)
    missing = active - available
    return {'expected_active_symbols':len(active),'active_symbols_with_bars':len(active & available),
            'halted_without_bars':sorted(missing & halted),
            'unexplained_missing_symbols':sorted(missing-halted)}


def require_snapshot_quality(root: Path, target_date: str, start_date: str) -> dict:
    report = inspect_snapshot(root, target_date, start_date)
    (Path(root) / "lhb_quality_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["status"] != "pass":
        raise RuntimeError("Snapshot publication refused: " + "; ".join(report["errors"]))
    return report
