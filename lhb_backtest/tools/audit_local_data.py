"""Read-only full-file data audit; writes one small JSON report, never source data.

Usage: python tools/audit_local_data.py --output research/project_review_20260925/data_audit.json
Coverage checks and structural checks cannot establish vendor-level authenticity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import duckdb
import pandas as pd

from src.data_sources.simtradedata_metadata import resolve_export_dir
from src.data_sources.snapshot_quality import inspect_snapshot


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def audit():
    start = time.perf_counter()
    export = resolve_export_dir()
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    report = {"export_dir": str(export), "manifest_version": manifest["version"],
              "files": {}, "checks": {}, "errors": []}
    for folder in ("raw", "clean", "factor"):
        for path in (ROOT / "data" / folder).glob("*.parquet"):
            report["files"][str(path.relative_to(ROOT))] = {"bytes": path.stat().st_size,
                                                           "sha256": digest(path)}
    with tempfile.TemporaryDirectory(prefix="audit_", dir=ROOT / "data/update_work") as temporary:
        with duckdb.connect() as conn:
            conn.execute("SET memory_limit='768MB'")
            conn.execute("SET threads=2")
            conn.execute("SET temp_directory=?", [temporary])
            conn.read_parquet(str(export / "stocks/*.parquet"), filename=True).create_view("vendor")
            for name, relative in (("raw", "raw/daily_kline"), ("clean", "clean/daily_kline"),
                                   ("summary", "raw/lhb_summary"), ("factors", "factor/lhb_event_factors"),
                                   ("labels", "factor/lhb_event_labeled")):
                conn.read_parquet(str(ROOT / "data" / (relative + ".parquet"))).create_view(name)

            def check(name, sql, parameters=None):
                count = conn.execute(sql, parameters or []).fetchone()[0]
                report["checks"][name] = count
                if count:
                    report["errors"].append(f"{name}: {count}")

            for name, date, key, amount in (("vendor", "date", "filename", "money"),
                                          ("raw", "trade_date", "stock_code", "amount"),
                                          ("clean", "trade_date", "stock_code", "amount")):
                count, earliest, latest, stocks = conn.execute(
                    f"SELECT count(*), min({date})::VARCHAR, max({date})::VARCHAR, count(DISTINCT {key}) FROM {name}"
                ).fetchone()
                report[name] = {"rows": count, "start": earliest, "end": latest, "stocks": stocks}
                check(name + "_invalid_keys", f"SELECT count(*) FROM (SELECT {key},{date} FROM {name} GROUP BY ALL HAVING count(*)>1 OR {key} IS NULL OR {date} IS NULL)")
                invalid = " OR ".join(f"{c} IS NULL OR NOT isfinite({c}) OR {c}<=0" for c in ("open", "high", "low", "close"))
                check(name + "_invalid_prices", f"SELECT count(*) FROM {name} WHERE {invalid} OR high<greatest(open,close,low) OR low>least(open,close,high)")
                check(name + "_invalid_volume_amount", f"SELECT count(*) FROM {name} WHERE volume IS NULL OR {amount} IS NULL OR NOT isfinite(volume) OR NOT isfinite({amount}) OR volume<0 OR {amount}<0")

            # Verify every imported price and volume against its published source.
            conn.execute(r"""CREATE VIEW canonical_vendor AS SELECT *,
                replace(regexp_extract(replace(filename,chr(92),'/'),'([^/]+)\.parquet$',1),'.SS','.SH') AS stock_code,
                CAST(date AS DATE)::VARCHAR AS trade_date FROM vendor""")
            check("raw_without_vendor_key", "SELECT count(*) FROM raw r ANTI JOIN canonical_vendor v ON replace(r.stock_code,'.SS','.SH')=v.stock_code AND r.trade_date=v.trade_date")
            comparisons = [f"abs(r.{c}-v.{c})>1e-8" for c in ("open", "high", "low", "close", "volume")]
            comparisons.append("abs(r.amount-v.money)>1e-6")
            check("raw_vendor_value_differences", "SELECT count(*) FROM raw r JOIN canonical_vendor v ON replace(r.stock_code,'.SS','.SH')=v.stock_code AND r.trade_date=v.trade_date WHERE " + " OR ".join(comparisons))
            check("clean_missing_raw_keys", "SELECT count(*) FROM raw r ANTI JOIN clean c ON replace(r.stock_code,'.SS','.SH')=c.stock_code AND r.trade_date=c.trade_date")
            for name in ("factors", "labels"):
                fields = [row[0] for row in conn.execute(f"DESCRIBE {name}").fetchall()
                          if row[1] in ("DOUBLE", "FLOAT")]
                invalid = " OR ".join(f'("{c}" IS NOT NULL AND NOT isfinite("{c}"))' for c in fields)
                check(name + "_nonfinite_numbers", f"SELECT count(*) FROM {name} WHERE {invalid}")
            report["limit_price_status"] = dict(conn.execute("SELECT limit_price_status,count(*) FROM clean GROUP BY 1").fetchall())
            report["event_counts"] = dict(conn.execute("SELECT disclosure_kind,count(*) FROM factors GROUP BY 1").fetchall())
            report["event_coverage"] = conn.execute("SELECT count(*) AS events,count(*) FILTER(WHERE market_supported) AS supported,count(*) FILTER(WHERE market_supported AND NOT quote_available) AS missing_quotes,count(*) FILTER(WHERE market_supported AND is_st IS NULL) AS unknown_st FROM factors").fetchdf().iloc[0].to_dict()
            report["labels_available"] = {str(n): conn.execute(f"SELECT count(future_return_{n}d) FROM labels").fetchone()[0] for n in (1,2,3,5,10)}
            conn.read_parquet(str(export / "metadata/trade_days.parquet")).create_view("calendar_source")
            # Independent SQL recomputation for every event, using the full
            # exchange calendar rather than any stock's row offsets.
            for n in (1, 2, 3, 5, 10):
                conn.execute(f"""CREATE OR REPLACE TEMP VIEW audit_calendar AS
                    SELECT CAST(date AS DATE)::VARCHAR AS trade_date,
                           lead(CAST(date AS DATE)::VARCHAR,{n}) OVER(ORDER BY date) AS exit_date
                    FROM calendar_source""")
                check(f"label_return_{n}d_mismatch", f"""
                    WITH compared AS (
                        SELECT l.future_return_{n}d AS actual, c.close / b.close - 1 AS expected
                        FROM labels l LEFT JOIN audit_calendar a USING(trade_date)
                        LEFT JOIN clean b ON b.stock_code=l.stock_code AND b.trade_date=l.trade_date
                        LEFT JOIN clean c ON c.stock_code=l.stock_code AND c.trade_date=a.exit_date)
                    SELECT count(*) FROM compared
                    WHERE (actual IS NULL)<>(expected IS NULL) OR abs(actual-expected)>1e-10
                """)
    report["snapshot"] = inspect_snapshot(export, report["raw"]["end"], "2019-01-01")
    if report["snapshot"]["status"] != "pass":
        report["errors"].extend(report["snapshot"]["errors"])
    report["status"] = "fail" if report["errors"] else "pass"
    report["seconds"] = time.perf_counter() - start
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": result["errors"], "seconds": result["seconds"]}, ensure_ascii=False))
    raise SystemExit(result["status"] != "pass")
