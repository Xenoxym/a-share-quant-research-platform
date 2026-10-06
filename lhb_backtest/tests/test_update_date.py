from datetime import datetime, timezone
from src.ingestion.update_simtradedata import _completed_day_ceiling

def test_shanghai_cutoff_uses_completed_day():
    # Computer timezone must not determine the China-market cutoff.
    assert _completed_day_ceiling('2026-09-24',datetime(2026,9,24,6,tzinfo=timezone.utc))=='2026-09-23'
    assert _completed_day_ceiling('2026-09-24',datetime(2026,9,24,10,tzinfo=timezone.utc))=='2026-09-24'
    assert _completed_day_ceiling('2026-09-20',datetime(2026,9,24,10,tzinfo=timezone.utc))=='2026-09-20'
