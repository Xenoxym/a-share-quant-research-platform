"""Read-only contract checks from acquired rows through research tables."""
from pathlib import Path
import json,tempfile
import duckdb
from src.ingestion.update_contract import write_report

SUPPORTED=r'^(60[0-9]{4}\.SH|688[0-9]{3}\.SH|00[0-9]{4}\.SZ|30[0-9]{4}\.SZ)$'

def inspect_project_data(project_root: Path) -> dict:
    root=Path(project_root).resolve();data=root/'data';errors=[];checks={}
    files={
        'kline':data/'raw/daily_kline.parquet','summary':data/'raw/lhb_summary.parquet',
        'detail':data/'raw/lhb_broker_detail.parquet','clean_kline':data/'clean/daily_kline.parquet',
        'clean_summary':data/'clean/lhb_summary.parquet','clean_detail':data/'clean/lhb_broker_detail.parquet',
        'events':data/'factor/lhb_event_daily.parquet','factors':data/'factor/lhb_event_factors.parquet',
        'labels':data/'factor/lhb_event_labeled.parquet'}
    for name,path in files.items():
        if not path.exists():errors.append('Missing '+name)
    if errors:return {'status':'fail','errors':errors,'checks':checks}
    work=data/'update_work';work.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='quality_check_',dir=work) as temp:
        conn=duckdb.connect();conn.execute("SET memory_limit='768MB'");conn.execute('SET threads=2');conn.execute('SET temp_directory=?',[temp])
        try:
            for name,path in files.items():
                conn.read_parquet(str(path)).create_view(name)
                key='stock_code,trade_date'+(',flag,rank' if name=='detail' else ',direction,rank' if name=='clean_detail' else '')
                count,end=conn.execute(f'SELECT count(*),max(trade_date)::VARCHAR FROM {name}').fetchone()
                duplicates=conn.execute(f'SELECT count(*) FROM (SELECT {key} FROM {name} GROUP BY {key} HAVING count(*)>1)').fetchone()[0]
                checks[name]={'rows':count,'end':end,'duplicate_keys':duplicates}
                if duplicates:errors.append(f'{name}: duplicate keys')
            def check(name,query,params=None):
                count=conn.execute(query,params or []).fetchone()[0];checks[name]=count
                if count:errors.append(f'{name}: {count}')
            check('summary_events_without_seats','SELECT count(*) FROM summary s ANTI JOIN detail d USING(stock_code,trade_date)')
            check('supported_events_without_quotes',"SELECT count(*) FROM summary s ANTI JOIN (SELECT replace(stock_code,'.SS','.SH') stock_code,trade_date FROM kline) k USING(stock_code,trade_date) WHERE regexp_matches(s.stock_code,?)",[SUPPORTED])
            checks['unsupported_market_events']=conn.execute('SELECT count(*) FROM summary WHERE NOT regexp_matches(stock_code,?)',[SUPPORTED]).fetchone()[0]
            check('missing_ranked_seat_amounts',"SELECT count(*) FROM detail WHERE broker_name IS NULL OR trim(broker_name)='' OR CASE WHEN flag='买入' THEN buy_amount ELSE sell_amount END IS NULL")
            for name in ('clean_summary','events','factors','labels'):
                check(name+'_missing_events',f'SELECT count(*) FROM summary ANTI JOIN {name} USING(stock_code,trade_date)')
                check(name+'_extra_events',f'SELECT count(*) FROM {name} ANTI JOIN summary USING(stock_code,trade_date)')
            for name in ('events','factors'):
                columns=set(conn.table(name).columns)
                forbidden=sorted(columns & {'return_1d','return_2d','return_5d','return_10d','lhb_interpret','limit_up_threshold'})
                checks[name+'_forbidden_predictors']=forbidden
                if forbidden:errors.append(f'{name}: future/current-name-derived predictors {forbidden}')
                if 'disclosure_kind' in columns:
                    seat_fields=columns & {'buy1_amount','sell1_amount','buy1_concentration','sell1_concentration','inst_buy_count','inst_sell_count','top5_buy_avg','top5_sell_avg'}
                    if seat_fields:
                        check(name+'_category_used_as_seat',f"SELECT count(*) FROM {name} WHERE disclosure_kind='investor_category' AND ("+' OR '.join('"'+c+'" IS NOT NULL' for c in sorted(seat_fields))+')')
            check('event_report_mismatches','SELECT count(*) FROM events WHERE summary_selected_reason IS DISTINCT FROM seat_report_reason')
            check('supported_events_with_unknown_st','SELECT count(*) FROM events WHERE market_supported AND is_st IS NULL')
            check('unsupported_events_with_assumed_st','SELECT count(*) FROM events WHERE NOT market_supported AND is_st IS NOT NULL')
            latest=checks['kline']['end']
            for horizon in (1,2,3,5,10):
                check(f'latest_day_has_future_return_{horizon}d',f'SELECT count(*) FROM labels WHERE trade_date=? AND future_return_{horizon}d IS NOT NULL',[latest])
            from src.data_sources.simtradedata_metadata import resolve_export_dir
            export=resolve_export_dir()
            dependencies={'clean_kline':['kline'],'clean_summary':['summary'],'clean_detail':['detail'],
                          'events':['clean_kline','clean_summary','clean_detail'],'factors':['events'],'labels':['factors','clean_kline']}
            for name,sources in dependencies.items():
                if any(files[name].stat().st_mtime_ns<files[source].stat().st_mtime_ns for source in sources):errors.append(name+': stale inputs; rebuild required')
            if export:
                for name,source in [('clean_kline','stock_status.parquet'),('labels','benchmark.parquet')]:
                    path=export/'metadata'/source
                    if path.exists() and files[name].stat().st_mtime_ns<path.stat().st_mtime_ns:errors.append(name+': stale metadata; rebuild required')
            checks['source_range']={'start':conn.execute('SELECT min(trade_date) FROM summary').fetchone()[0],'end':checks['summary']['end']}
        finally:conn.close()
    report={'status':'fail' if errors else 'pass','errors':errors,'checks':checks,
            'scope':'Configured Shanghai/Shenzhen A-share research; other LHB markets are retained and explicitly unsupported',
            'limitations':['This contract gate complements source acquisition and publication audits; it is not a guarantee against vendor errors.']}
    write_report(data/'update_reports/project_quality_latest.json',report)
    return report
