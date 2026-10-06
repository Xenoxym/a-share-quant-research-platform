"""Resumable source-complete LHB seat acquisition, retaining reporting windows.

Uses the same EastMoney reports as AKShare's per-stock adapter, but requests
bounded date ranges and verifies every page. No strategy filters at acquisition.
"""
from pathlib import Path
import json
import hashlib
import time
import pandas as pd
import requests
from src.utils.io import save_parquet
from src.data_sources.reporting_windows import reporting_window_days

URL = 'https://datacenter-web.eastmoney.com/api/data/v1/get'
FIELDS = {'SECUCODE':'stock_code','TRADE_DATE':'trade_date',
          'OPERATEDEPT_NAME':'broker_name','OPERATEDEPT_CODE':'broker_code',
          'BUY':'buy_amount','SELL':'sell_amount','NET':'net_amount',
          'TOTAL_BUYRIO':'buy_amount_ratio','TOTAL_SELLRIO':'sell_amount_ratio',
          'CHANGE_TYPE':'report_type','EXPLANATION':'report_reason','TRADE_ID':'report_id'}


def fetch_range(start, end, side, session=None, _symbol=None, _allow_empty=False):
    session = session or requests.Session()
    rows, total, pages = [], None, None
    source_pages=[]
    page = 1
    while pages is None or page <= pages:
        params = dict(reportName='RPT_BILLBOARD_DAILYDETAILS'+side, columns='ALL',
                      filter=f"(TRADE_DATE>='{start}')(TRADE_DATE<='{end}')",
                      pageNumber=str(page), pageSize='500',
                      sortTypes='1,1,1,1,1',
                      sortColumns='TRADE_DATE,SECURITY_CODE,CHANGE_TYPE,OPERATEDEPT_CODE,'+side,
                      source='WEB',client='WEB')
        if _symbol is not None:
            params['filter'] += f'(SECURITY_CODE="{_symbol}")'
        for attempt in range(3):
            try:
                response = session.get(URL,params=params,timeout=(10,45))
                response.raise_for_status()
                body = response.json()
                if _allow_empty and body.get('code')==9201 and body.get('message')=='返回数据为空':
                    # Only a smaller verification sub-query can be empty. Its
                    # parent must still reconcile to the independently reported
                    # total, so an incomplete source response cannot pass.
                    return pd.DataFrame(columns=list(FIELDS.values())+['flag','window_days','source_version'])
                if not body.get('success') or not isinstance(body.get('result'),dict):
                    raise RuntimeError(f'Unconfirmed source response: {str(body)[:240]}')
                result = body['result']
                n, p = int(result['count']), int(result['pages'])
                if total is not None and (total != n or pages != p):
                    raise RuntimeError('Source changed during pagination; refusing incomplete archive')
                chunk = result.get('data') or []
                if n and not chunk:
                    raise RuntimeError('Empty page within nonempty source result')
                break
            except (requests.RequestException, ValueError, KeyError, RuntimeError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        total, pages = n, p
        rows.extend(chunk)
        source_pages.extend([page]*len(chunk))
        page += 1
        time.sleep(.2)
    if len(rows) != total:
        raise RuntimeError(f'Pagination count mismatch {len(rows)} != {total}')
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError('Empty monthly report requires explicit source investigation')
    if not set(FIELDS).issubset(frame.columns):
        raise RuntimeError('Source schema changed')
    duplicated=frame.duplicated(keep=False)
    across_pages=False
    if duplicated.any() and pages>1:
        duplicates=pd.DataFrame({'row_hash':pd.util.hash_pandas_object(frame.loc[duplicated],index=False),
                                 'source_page':pd.Series(source_pages,index=frame.index).loc[duplicated]})
        across_pages=duplicates.groupby('row_hash').source_page.nunique().gt(1).any()
    if across_pages:
        # Anonymous institutions can legitimately have identical source rows.
        # Distinguish those from unstable page boundaries with smaller queries.
        if start < end:
            midpoint=(pd.Timestamp(start)+(pd.Timestamp(end)-pd.Timestamp(start))/2).date()
            parts=[fetch_range(start,str(midpoint),side,session,_allow_empty=True),
                   fetch_range(str(midpoint+pd.Timedelta(days=1)),end,side,session,_allow_empty=True)]
        elif _symbol is None:
            parts=[fetch_range(start,end,side,session,str(code))
                   for code in sorted(frame.SECURITY_CODE.unique())]
        else:
            raise RuntimeError('Ambiguous pagination even for a single stock and day')
        combined=pd.concat(parts,ignore_index=True)
        if len(combined)!=total:
            raise RuntimeError('Smaller source queries disagree with original count')
        return combined
    frame = frame.rename(columns=FIELDS)
    frame['trade_date'] = pd.to_datetime(frame.trade_date).dt.strftime('%Y-%m-%d')
    if not frame.trade_date.between(start,end).all():
        raise RuntimeError('Source returned dates outside requested interval')
    frame['stock_code'] = frame.stock_code.str.replace('.SS','.SH',regex=False)
    frame['flag'] = '买入' if side == 'BUY' else '卖出'
    frame['window_days'] = frame.report_reason.map(reporting_window_days)
    frame['source_version'] = 2
    return frame


def acquire_archive(start, end, directory):
    from src.ingestion.update_contract import update_lock
    with update_lock(Path(directory)/'.acquisition.lock'):
        return _acquire_archive(start,end,directory)


def _acquire_archive(start, end, directory):
    directory = Path(directory);directory.mkdir(parents=True,exist_ok=True)
    periods = pd.period_range(start,end,freq='M')
    with requests.Session() as session:
        for period in reversed(periods):
            base_lo=max(start,str(period.start_time.date()));hi=min(end,str(period.end_time.date()))
            for side in ('BUY','SELL'):
                lo=base_lo
                existing=[]
                for marker in directory.glob('*_'+side+'.json'):
                    meta=json.loads(marker.read_text(encoding='utf-8'))
                    cached=marker.with_suffix('.parquet')
                    if cached.exists() and meta.get('schema')==2 and meta.get('bytes')==cached.stat().st_size:
                        existing.append(meta)
                for meta in sorted(existing,key=lambda m:m['start']):
                    if meta['start']<=lo<=meta['end']:
                        lo=str((pd.Timestamp(meta['end'])+pd.Timedelta(days=1)).date())
                if lo>hi:
                    print(f'Reuse {base_lo}_{hi}_{side}',flush=True);continue
                name=f'{lo}_{hi}_{side}'
                path=directory/(name+'.parquet'); marker=directory/(name+'.json')
                if path.exists() and marker.exists():
                    meta=json.loads(marker.read_text(encoding='utf-8'))
                    if meta.get('bytes')==path.stat().st_size and meta.get('schema')==2:
                        print(f'Reuse {name}',flush=True);continue
                frame=fetch_range(lo,hi,side,session)
                save_parquet(frame,path)
                marker.write_text(json.dumps({'schema':2,'rows':len(frame),'bytes':path.stat().st_size,'start':lo,'end':hi,'side':side}),encoding='utf-8')
                print(f'Fetched {name}: {len(frame)} rows',flush=True)


def canonical_seats(frame, preferred_reports=None):
    """Choose one complete report per event without mixing reporting windows.

    All other reports remain preserved in the archive. Prefer single-day reports;
    the choice is explicit on each row for downstream factor eligibility.
    """
    frame=frame.copy()
    frame['window_days']=frame.report_reason.map(reporting_window_days)
    keys=['trade_date','stock_code','report_type']
    groups=frame.groupby(keys,dropna=False).agg(sides=('flag','nunique'),side=('flag','first'),window=('window_days','max'),reason=('report_reason','first')).reset_index()
    # Financing-monitoring disclosures publish one direction only. This is
    # a different report type, not a missing half of an ordinary LHB report.
    groups['disclosure_kind']='ordinary'
    groups.loc[groups.reason.str.contains('融资买入',na=False),'disclosure_kind']='margin_buy'
    groups.loc[groups.reason.str.contains('融券卖出',na=False),'disclosure_kind']='short_sell'
    category_keys=frame.loc[frame.broker_name.isin(['自然人','中小投资者','其他自然人']),keys].drop_duplicates()
    category_index=pd.MultiIndex.from_frame(category_keys)
    groups.loc[pd.MultiIndex.from_frame(groups[keys]).isin(category_index),'disclosure_kind']='investor_category'
    one_side=(groups.disclosure_kind.eq('margin_buy') & groups.side.eq('买入') | groups.disclosure_kind.eq('short_sell') & groups.side.eq('卖出')) & groups.sides.eq(1)
    valid=groups[groups.sides.eq(2)|one_side].copy()
    valid['_priority']=valid.window.where(valid.window.gt(0),999)
    valid['_preferred']=1
    if preferred_reports is not None and 'summary_selected_reason' in preferred_reports:
        valid=valid.merge(preferred_reports[['trade_date','stock_code','summary_selected_reason']],on=['trade_date','stock_code'],how='left',validate='many_to_one')
        valid['_preferred']=valid.reason.ne(valid.summary_selected_reason).astype(int)
    valid=valid.sort_values(['trade_date','stock_code','_preferred','_priority','report_type'])
    chosen=valid.drop_duplicates(['trade_date','stock_code'])
    result=frame.merge(chosen[keys+['disclosure_kind']],on=keys,validate='many_to_one')
    result=result.sort_values(keys+['flag','buy_amount','sell_amount','broker_code'],ascending=[True,True,True,True,False,False,True],na_position='last')
    # Rank each side by its own transaction amount, not API page order.
    result['_side_amount']=result.buy_amount.where(result.flag.eq('买入'),result.sell_amount)
    if result['_side_amount'].isna().any() or result['_side_amount'].lt(0).any() or result.broker_name.fillna('').str.strip().eq('').any():
        raise RuntimeError('Missing or invalid ranked seat amount/name; refusing publication')
    result=result.sort_values(keys+['flag','_side_amount','broker_code'],ascending=[True,True,True,True,False,True],na_position='last')
    result['rank']=result.groupby(keys+['flag']).cumcount()+1
    if result['rank'].max()>5:
        raise RuntimeError('More than five seats per source reporting type; investigate before publication')
    return result.drop(columns='_side_amount').reset_index(drop=True)


def refresh_detail_archive(summary_path, save_path, start=None, end=None):
    """Acquire all summary events; publish a usable table with explicit gaps."""
    from src.ingestion.update_contract import update_lock
    with update_lock(Path(save_path).parent/'lhb_detail_source'/'.publication.lock'):
        return _refresh_detail_archive(summary_path,save_path,start,end)


def _refresh_detail_archive(summary_path, save_path, start=None, end=None):
    summary_path=Path(summary_path).resolve()
    save_path=Path(save_path).resolve()
    summary=pd.read_parquet(summary_path)
    from src.cleaning.normalize_codes import normalize_stock_code
    summary['stock_code']=summary.stock_code.map(normalize_stock_code)
    start=max(start or summary.trade_date.min(),summary.trade_date.min())
    end=min(end or summary.trade_date.max(),summary.trade_date.max())
    universe=summary[summary.trade_date.between(start,end)][['trade_date','stock_code']].drop_duplicates()
    if universe.empty:
        raise ValueError('No summary events in requested interval; update/verify summary first')
    directory=Path(save_path).parent/'lhb_detail_source'
    acquire_archive(start,end,directory)
    report_path=Path(save_path).with_suffix('.coverage.json')
    inputs=[Path(summary_path)]+sorted(directory.glob('*.parquet'))+sorted(directory.glob('*_BUY.json'))+sorted(directory.glob('*_SELL.json'))
    signature=hashlib.sha256(json.dumps([(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in inputs]).encode()).hexdigest()
    if report_path.exists() and Path(save_path).exists():
        previous=json.loads(report_path.read_text(encoding='utf-8'))
        stat=Path(save_path).stat()
        if (previous.get('status')=='pass' and previous.get('policy')=='coherent-reports-v4'
                and previous.get('input_signature')==signature
                and previous.get('start')==start and previous.get('end')==end
                and previous.get('output_stamp')==[stat.st_size,stat.st_mtime_ns]):
            seats=pd.read_parquet(save_path)
            seats.attrs['update_report']={**previous,'reused':True}
            return seats
    frames=[]
    for path in directory.glob('*.parquet'):
        if path.with_suffix('.json').exists():
            frame=pd.read_parquet(path)
            meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            if meta.get('rows')!=len(frame) or meta.get('bytes')!=path.stat().st_size:
                raise RuntimeError(f'Archive integrity mismatch: {path}')
            frames.append(frame[frame.trade_date.between(start,end)])
    seats=canonical_seats(pd.concat(frames,ignore_index=True),summary)
    present=pd.MultiIndex.from_frame(seats[['trade_date','stock_code']].drop_duplicates())
    gaps=universe.loc[~pd.MultiIndex.from_frame(universe).isin(present)]
    report={'status':'pass' if gaps.empty else 'fail','start':start,'end':end,
            'scope':'all summary events, no strategy filters','expected_events':len(universe),
            'covered_events':len(universe)-len(gaps),'unresolved_events':gaps.to_dict('records')}
    events=seats.drop_duplicates(['trade_date','stock_code'])
    in_summary=pd.MultiIndex.from_frame(events[['trade_date','stock_code']]).isin(pd.MultiIndex.from_frame(universe))
    report['disclosure_kinds']=events.loc[in_summary,'disclosure_kind'].value_counts().to_dict()
    report['source_only_events']=events.loc[~in_summary,['trade_date','stock_code','report_type','report_reason','disclosure_kind']].to_dict('records')
    from src.ingestion.update_contract import write_report
    if not gaps.empty:
        write_report(report_path,report)
        raise RuntimeError(f'{len(gaps)} events lack complete source reports; see {report_path}')
    # Retain history outside a caller's requested interval.
    if Path(save_path).exists():
        old=pd.read_parquet(save_path)
        seats=pd.concat([old[~old.trade_date.between(start,end)],seats],ignore_index=True)
    save_parquet(seats,save_path)
    stat=Path(save_path).stat()
    report.update(policy='coherent-reports-v4',input_signature=signature,output_stamp=[stat.st_size,stat.st_mtime_ns],reused=False)
    write_report(report_path,report)
    seats.attrs['update_report']=report
    return seats
