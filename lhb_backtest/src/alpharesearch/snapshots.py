"""Immutable, non-label snapshots of all locally archived LHB reports (E02).

A provider report identity is distinct from a stock/date event. This adapter does
not select one report, infer a missing side, deduplicate anonymous institutions,
or certify historical vendor vintages. Only explicit whitelists enter outputs.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq

from ..data_sources.reporting_windows import reporting_window_days
from ..technical.artifacts import content_id, digest, write_json

SCHEMA = 'lhb-original-reports-v1'
SUMMARY_AMOUNTS = ('lhb_buy_total', 'lhb_sell_total', 'lhb_net_buy',
                   'lhb_turnover', 'daily_amount')
SUMMARY_FIELDS = ('trade_date', 'stock_code', 'lhb_reason') + SUMMARY_AMOUNTS
SEAT_FIELDS = ('trade_date', 'stock_code', 'report_id', 'report_type',
               'report_reason', 'flag', 'broker_name', 'broker_code',
               'buy_amount', 'sell_amount', 'net_amount', 'source_version')
REPORT_KEYS = ['trade_date', 'stock_code', 'report_id', 'report_type', 'report_reason']
MATCH_KEYS = ['trade_date', 'stock_code', 'report_reason']
CATEGORY_NAMES = {'自然人', '中小投资者', '其他自然人'}
POLICY = {
    'timing': 'trade_date_proxy_next_civil_midnight_Asia_Shanghai',
    'publication_precision': 'trade_date_proxy_no_original_timestamp',
    'vintage': 'latest_snapshot_only',
    'absence': 'not_certified_by_file_presence_or_selected_detail_coverage',
    'units': 'amounts_cny_as_declared_by_existing_ingestion_contract',
    'matching': 'exact_stock_date_reason_unique_source_report_only',
    'anonymous_rows': 'preserve_all_source_occurrences',
}


def _keys(frame):
    frame = frame.copy()
    for column in ('trade_date', 'stock_code'):
        if column not in frame or frame[column].isna().any():
            raise ValueError('Missing event key: ' + column)
        frame[column] = frame[column].astype(str)
    if not frame.trade_date.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all():
        raise ValueError('Event date must be canonical YYYY-MM-DD')
    pd.to_datetime(frame.trade_date, format='%Y-%m-%d', errors='raise')
    if not frame.stock_code.str.fullmatch(r'\d{6}\.(SH|SZ|BJ)').all():
        raise ValueError('Stock code must include its declared exchange')
    return frame


def _json_row(values):
    return json.dumps([None if pd.isna(v) else v for v in values],
                      ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _hash_rows(frame, columns):
    return pd.Series([hashlib.sha256(_json_row(row).encode()).hexdigest()
                      for row in frame[columns].itertuples(index=False, name=None)],
                     index=frame.index)


def _numbers(frame, columns):
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors='coerce').astype(float)
        frame.loc[~np.isfinite(frame[column]), column] = np.nan
    return frame


def build_event_tables(summary, archive):
    """Build full report headers and ranked source occurrences, with no Y fields.

    Exact summary repetitions collapse with a count. Conflicting repetitions or
    ambiguous source-report matching reject the build rather than choose a value.
    Invalid/incomplete seat reports remain in the outputs with explicit status.
    A rank here is a derived within-report order, not a persistent broker identity.
    """
    required_summary = {'trade_date', 'stock_code', 'lhb_reason'}
    required_seats = set(SEAT_FIELDS) - {'broker_code', 'net_amount', 'source_version'}
    if not required_summary <= set(summary) or not required_seats <= set(archive):
        raise ValueError('Missing required original report fields')
    s = _keys(summary[[c for c in SUMMARY_FIELDS if c in summary]])
    s = _numbers(s, SUMMARY_AMOUNTS).rename(columns={'lhb_reason': 'report_reason'})
    a = _keys(archive[[c for c in SEAT_FIELDS if c in archive]])
    if a.empty:
        raise ValueError('No original seat reports; do not infer no-event coverage')
    for column in ('report_reason',):
        for table in (s, a):
            if table[column].isna().any() or table[column].astype(str).str.strip().eq('').any():
                raise ValueError('Missing report reason')
            table[column] = table[column].astype(str)
    for column in ('report_id', 'report_type'):
        if a[column].isna().any() or a[column].astype(str).str.strip().eq('').any():
            raise ValueError('Missing original provider report identity')
        a[column] = a[column].astype(str)
    if not a.flag.isin(['买入', '卖出', 'buy', 'sell']).all():
        raise ValueError('Unknown source side; no silent sell fallback')
    a['direction'] = a.flag.replace({'买入': 'buy', '卖出': 'sell'})
    a = _numbers(a, ('buy_amount', 'sell_amount', 'net_amount'))
    for column in ('broker_code', 'source_version'):
        if column not in a:
            a[column] = None
    a['broker_name'] = a.broker_name.fillna('').astype(str)
    # Identity hashes depend on economic report fields, not download order/path or Y.
    a['event_id'] = _hash_rows(a, REPORT_KEYS)
    s['_summary_value'] = _hash_rows(s, list(SUMMARY_AMOUNTS))
    repeated = s.groupby(MATCH_KEYS, dropna=False)._summary_value.nunique()
    if repeated.gt(1).any():
        raise ValueError('Conflicting summary versions: retain raw inputs, refuse publication')
    counts = s.groupby(MATCH_KEYS, dropna=False).size().rename('summary_occurrences')
    s = s.drop_duplicates(MATCH_KEYS).merge(counts, on=MATCH_KEYS, validate='one_to_one')
    s = s.drop(columns='_summary_value')
    headers = a[REPORT_KEYS + ['event_id']].drop_duplicates()
    if headers.duplicated(MATCH_KEYS).any():
        raise ValueError('Ambiguous report matching: multiple source identities for one reason')
    merged = headers.merge(s, on=MATCH_KEYS, how='outer', indicator=True, validate='one_to_one')
    if merged._merge.eq('right_only').any():
        raise ValueError('Original summary report has no archived seat report')
    merged['summary_status'] = np.where(merged._merge.eq('both'), 'matched', 'source_only')
    merged = merged.drop(columns='_merge')
    merged['summary_occurrences'] = merged.summary_occurrences.fillna(0).astype(int)
    merged['window_days'] = merged.report_reason.map(reporting_window_days).astype(int)
    merged['window_known'] = merged.window_days.gt(0)
    kind = pd.Series('ordinary', index=merged.index)
    kind.loc[merged.report_reason.str.contains('融资买入', regex=False)] = 'margin_buy'
    kind.loc[merged.report_reason.str.contains('融券卖出', regex=False)] = 'short_sell'
    categories = set(a.loc[a.broker_name.isin(CATEGORY_NAMES), 'event_id'])
    kind.loc[merged.event_id.isin(categories)] = 'investor_category'
    merged['disclosure_kind'] = kind
    a['side_amount'] = np.where(a.direction.eq('buy'), a.buy_amount, a.sell_amount)
    a['seat_amount_valid'] = a.side_amount.notna() & a.side_amount.ge(0)
    a['seat_name_valid'] = a.broker_name.str.strip().ne('')
    versions = a.groupby('event_id').source_version.nunique(dropna=False)
    if versions.gt(1).any():
        raise ValueError('Mixed ingestion versions inside a source report')
    side_counts = a.groupby(['event_id', 'direction']).size().unstack(fill_value=0)
    for side in ('buy', 'sell'):
        if side not in side_counts:
            side_counts[side] = 0
        merged[side + '_rows'] = merged.event_id.map(side_counts[side]).astype(int)
    valid = (a.seat_amount_valid & a.seat_name_valid).groupby(a.event_id).all()
    merged['seat_values_valid'] = merged.event_id.map(valid).astype(bool)
    within = merged.buy_rows.le(5) & merged.sell_rows.le(5)
    both = merged.buy_rows.ge(1) & merged.sell_rows.ge(1)
    margin = merged.disclosure_kind.eq('margin_buy') & merged.buy_rows.ge(1) & merged.sell_rows.eq(0)
    short = merged.disclosure_kind.eq('short_sell') & merged.sell_rows.ge(1) & merged.buy_rows.eq(0)
    good = within & (both | margin | short) & merged.seat_values_valid
    merged['seat_status'] = np.where(good, 'complete_disclosure', 'incomplete_or_invalid')
    merged.loc[merged.disclosure_kind.eq('investor_category'), 'seat_status'] = 'non_seat_disclosure'
    merged['source_version'] = merged.event_id.map(a.groupby('event_id').source_version.first())
    # Source row multiplicity is retained, including identical anonymous institutions.
    row_fields = ['direction', 'broker_name', 'broker_code', 'buy_amount', 'sell_amount', 'net_amount']
    a['source_row_hash'] = _hash_rows(a, row_fields)
    a = a.sort_values(['event_id', 'direction', 'side_amount', 'source_row_hash'],
                     ascending=[True, True, False, True], kind='stable', na_position='last')
    a['rank'] = a.groupby(['event_id', 'direction']).cumcount() + 1
    a['identical_occurrences'] = a.groupby(['event_id', 'direction', 'source_row_hash']).source_row_hash.transform('size')
    a['seat_ordinal_id'] = a.event_id + ':' + a.direction + ':' + a['rank'].astype(str)
    local_day = pd.to_datetime(merged.trade_date, format='%Y-%m-%d').dt.tz_localize('Asia/Shanghai')
    merged['observed_end'] = (local_day + pd.Timedelta(hours=15)).dt.tz_convert('UTC')
    merged['known_at'] = (local_day + pd.Timedelta(days=1)).dt.tz_convert('UTC')
    merged['publication_precision'] = POLICY['publication_precision']
    merged['vintage'] = POLICY['vintage']
    merged['timing_rule'] = POLICY['timing']
    merged['original_timestamp_available'] = False
    merged['summary_original_id_available'] = False
    merged['amount_unit'] = 'cny'
    output_seats = a[['event_id', 'seat_ordinal_id', 'direction', 'rank', 'broker_name',
                       'broker_code', 'buy_amount', 'sell_amount', 'net_amount',
                       'seat_amount_valid', 'seat_name_valid', 'source_row_hash',
                       'identical_occurrences']].reset_index(drop=True)
    merged = merged.sort_values(['trade_date', 'stock_code', 'event_id']).reset_index(drop=True)
    quality = {
        'summary_input_rows': len(summary), 'summary_reports': len(s),
        'summary_exact_repetitions_collapsed': len(summary) - len(s),
        'source_rows': len(a), 'source_reports': len(merged),
        'source_only_reports': int(merged.summary_status.eq('source_only').sum()),
        'windows': {str(k): int(v) for k, v in merged.window_days.value_counts().items()},
        'disclosure_kinds': merged.disclosure_kind.value_counts().to_dict(),
        'seat_statuses': merged.seat_status.value_counts().to_dict(),
        'anonymous_or_other_identical_occurrences_retained': int(a.identical_occurrences.gt(1).sum()),
        'future_or_interpretation_fields_in_outputs': [],
        'no_event_coverage_certified': False,
    }
    return merged, output_seats, quality


def _source_record(path, app):
    if path.is_symlink() or not path.resolve().is_relative_to(app):
        raise ValueError('Snapshot sources must be regular files inside the declared project')
    before = path.stat(); sha = digest(path); after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source changed while hashing')
    return {'path': path.relative_to(app).as_posix(), 'sha256': sha,
            'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns}


def _load_sources(app, records):
    summary_path = app / 'data/raw/lhb_summary_reports.parquet'
    summary = pd.read_parquet(summary_path, columns=[c for c in SUMMARY_FIELDS
                                                    if c in pq.ParquetFile(summary_path).schema_arrow.names])
    frames = []
    markers = []
    for record in records:
        path = app / record['path']
        if path.parent.name != 'lhb_detail_source' or path.suffix != '.parquet':
            continue
        marker_path = path.with_suffix('.json')
        marker = json.loads(marker_path.read_text(encoding='utf-8'))
        columns = pq.ParquetFile(path).schema_arrow.names
        frame = pd.read_parquet(path, columns=[c for c in SEAT_FIELDS if c in columns])
        if marker.get('schema') != 2 or marker.get('rows') != len(frame) or marker.get('bytes') != path.stat().st_size:
            raise ValueError('Archive marker counts/schema mismatch: ' + path.name)
        lo, hi, side = marker.get('start'), marker.get('end'), marker.get('side')
        if not isinstance(lo, str) or not isinstance(hi, str) or side not in {'BUY', 'SELL'}:
            raise ValueError('Invalid archive range marker')
        if not frame.trade_date.between(lo, hi).all() or lo > hi:
            raise ValueError('Archive rows outside declared range')
        expected = '买入' if side == 'BUY' else '卖出'
        if not frame.flag.eq(expected).all():
            raise ValueError('Archive side disagrees with marker')
        frames.append(frame);markers.append({'start': lo, 'end': hi, 'side': side, 'rows': len(frame)})
    if not frames:
        raise ValueError('No complete local source archives')
    return summary, pd.concat(frames, ignore_index=True), markers


def _check(folder, require_name):
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != SCHEMA or content_id(manifest['identity']) != manifest['snapshot_id']:
        raise ValueError('Snapshot identity changed')
    if manifest['identity'].get('artifacts') != manifest.get('artifacts') or manifest['identity'].get('quality') != manifest.get('quality'):
        raise ValueError('Snapshot identity no longer binds outputs and quality')
    if require_name and folder.name != manifest['snapshot_id']:
        raise ValueError('Snapshot directory identity differs')
    allowed = {'reports.parquet', 'seats.parquet', 'code/snapshots.py',
               'code/contracts.py', 'code/reporting_windows.py'}
    if set(manifest.get('artifacts', {})) != allowed:
        raise ValueError('Snapshot artifact names differ from the schema')
    for name, sha in manifest['artifacts'].items():
        p = folder / name
        if p.is_symlink() or p.resolve() != folder.resolve() / name or digest(p) != sha:
            raise ValueError('Snapshot artifact integrity failed: ' + name)
    reports = pd.read_parquet(folder / 'reports.parquet', columns=['event_id'])
    seats = pd.read_parquet(folder / 'seats.parquet', columns=['event_id', 'seat_ordinal_id'])
    if reports.event_id.duplicated().any() or seats.seat_ordinal_id.duplicated().any():
        raise ValueError('Duplicate immutable event/seat identities')
    if not seats.event_id.isin(reports.event_id).all():
        raise ValueError('Seat points outside report snapshot')
    if len(reports) != manifest['quality']['source_reports'] or len(seats) != manifest['quality']['source_rows']:
        raise ValueError('Snapshot row counts differ')
    return manifest


def verify_snapshot(folder):
    """Verify identity, safe artifact paths/hashes and report/seat referential keys."""
    return _check(Path(folder).resolve(), require_name=True)


def create_event_snapshot(project, root=None, progress=None):
    """Read local archives only; freeze without changing any prior raw/clean table."""
    app = Path(project).resolve()
    root = Path(root or app / 'data/alpharesearch/event_snapshots').resolve()
    sources = [app / 'data/raw/lhb_summary_reports.parquet']
    archives = sorted((app / 'data/raw/lhb_detail_source').glob('*.parquet'))
    for path in archives:
        sources += [path, path.with_suffix('.json')]
    records = [_source_record(p, app) for p in sources]
    code_paths = [Path(__file__), Path(__file__).with_name('contracts.py'),
                  Path(__file__).parents[1] / 'data_sources/reporting_windows.py']
    codes = [{'name': p.name, 'sha256': digest(p)} for p in code_paths]
    inputs = {'schema': SCHEMA, 'policy': POLICY,
                'sources': [{k: v for k, v in r.items() if k != 'mtime_ns'} for r in records],
                'code': codes, 'pandas': pd.__version__, 'pyarrow': pyarrow.__version__}
    for old in sorted(root.glob('*/manifest.json')):
        if old.parent.name.startswith('_building-'):
            continue
        previous = json.loads(old.read_text(encoding='utf-8'))
        if previous.get('identity', {}).get('inputs') == inputs:
            verify_snapshot(old.parent)
            if progress: progress('Verified and reused existing immutable event snapshot')
            return old.parent
    if progress: progress('Read all original report archives; no selected-report filter')
    summary, archive, markers = _load_sources(app, records)
    reports, seats, quality = build_event_tables(summary, archive)
    if progress: progress(f'Built {len(reports)} reports and {len(seats)} source occurrences')
    # Rehash after reads; a size/mtime check alone cannot detect all replacements.
    for record in records:
        current = _source_record(app / record['path'], app)
        if current != record:
            raise ValueError('Source changed during snapshot build; refusing publication')
    root.mkdir(parents=True, exist_ok=True)
    staging = root / ('_building-' + uuid.uuid4().hex);staging.mkdir()
    reports.to_parquet(staging / 'reports.parquet', index=False)
    seats.to_parquet(staging / 'seats.parquet', index=False)
    (staging / 'code').mkdir()
    for p in code_paths:
        shutil.copyfile(p, staging / 'code' / p.name)
    artifacts = {p.relative_to(staging).as_posix(): digest(p)
                 for p in sorted(staging.rglob('*')) if p.is_file()}
    if any(digest(staging / 'code' / c['name']) != c['sha256'] for c in codes):
        raise ValueError('Source code changed during snapshot build')
    identity = {'inputs': inputs, 'artifacts': artifacts, 'quality': quality}
    sid = content_id(identity);target = root / sid
    manifest = {'schema': SCHEMA, 'snapshot_id': sid, 'identity': identity,
                'frozen_at': datetime.now(timezone.utc).isoformat(), 'sources': records,
                'artifacts': artifacts, 'quality': quality, 'archive_range_markers': markers,
                'limits': ['Trade date is a publication-date proxy; exact timestamps absent',
                           'Latest archived snapshot does not certify historical revisions',
                           'Archive markers certify local row counts, not all-market absence',
                           'Amounts follow declared ingestion units; no new provider audit',
                           'Source-only/category/multi-day reports must not become ordinary daily seats']}
    write_json(staging / 'manifest.json', manifest);_check(staging, require_name=False)
    try:
        staging.rename(target)
    except FileExistsError:
        # Another builder may finish first. Verify it, retain our staging evidence.
        verify_snapshot(target)
    return target
