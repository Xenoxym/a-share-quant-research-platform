"""Prepare an isolated official database and restore baseline auxiliary tables."""
from pathlib import Path
import json
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_sources.snapshot_seed import seed_database_from_snapshot

root = Path(__file__).resolve().parents[1] / 'data/quality_repair/20260923'
db = root / 'cn.repair.duckdb'
if db.exists():
    raise RuntimeError('Repair database already exists; refusing overwrite')
source = Path('D:/Projects/SimTradeDataRepo/data/cn.duckdb')
before = (source.stat().st_size, source.stat().st_mtime_ns)
shutil.copy2(source, db)
assert before == (source.stat().st_size, source.stat().st_mtime_ns), 'Source database changed during copy'
print('Database copy complete; seeding baseline', flush=True)
report = seed_database_from_snapshot(db, Path('D:/Projects/SimTradeData/data/export/cn.previous'))
(root / 'database_seed.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
