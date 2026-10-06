# 项目代码与文件索引

对应快照：2026-10-06T07:53:36.269170+00:00。本索引辅助定位，不能替代逐行语义审查。

项目根：`<LOCAL_REPOSITORY>`。Git 状态来自同一快照；新增的审查资料单独保存。

## src Python 文件

| 文件 | 行数 | 职责 | 顶层定义（最多 16 项） | Git 状态 |
| --- | --- | --- | --- | --- |
| [src/__init__.py](../src/__init__.py) | 1 | 模块/包入口 | 初始化 / 常量 / 入口 | 已跟踪 |
| [src/backtest/__init__.py](../src/backtest/__init__.py) | 21 | 事件筛选与统计 | 初始化 / 常量 / 入口 | 已跟踪 |
| [src/backtest/filter_engine.py](../src/backtest/filter_engine.py) | 354 | 事件筛选与统计 | BacktestResult, run_lhb_backtest, apply_filters, _apply_date_filter, _apply_board_type_filter | 已跟踪 |
| [src/backtest/grid_search.py](../src/backtest/grid_search.py) | 301 | 事件筛选与统计 | format_grid_search_summary, _objective_count_column, grid_search | 已跟踪 |
| [src/backtest/statistics.py](../src/backtest/statistics.py) | 360 | 事件筛选与统计 | compute_backtest_statistics, compute_grouped_statistics, compute_quantile_statistics, compute_rolling_period_statistics, _safe_mean, _compute_profit_loss_ratio | 已跟踪 |
| [src/cleaning/__init__.py](../src/cleaning/__init__.py) | 1 | 规范化与清洗 | 初始化 / 常量 / 入口 | 已跟踪 |
| [src/cleaning/clean_kline.py](../src/cleaning/clean_kline.py) | 119 | 规范化与清洗 | clean_daily_kline | 已跟踪 |
| [src/cleaning/clean_lhb.py](../src/cleaning/clean_lhb.py) | 332 | 规范化与清洗 | _aggregate_multi_reason_rows, filter_lhb_events, _filter_remove_st, _filter_lhb_reason, _filter_first_appearance, clean_lhb_summary, clean_lhb_broker_detail | 已跟踪 |
| [src/cleaning/normalize_codes.py](../src/cleaning/normalize_codes.py) | 126 | 规范化与清洗 | normalize_stock_code, _infer_suffix, normalize_date, normalize_codes_in_df | 已跟踪 |
| [src/cleaning/price_limits.py](../src/cleaning/price_limits.py) | 53 | 规范化与清洗 | validate_price_limits | 已跟踪 |
| [src/data_sources/__init__.py](../src/data_sources/__init__.py) | 5 | 供应读取与质量 | 初始化 / 常量 / 入口 | 已跟踪 |
| [src/data_sources/akshare_client.py](../src/data_sources/akshare_client.py) | 395 | 供应读取与质量 | _add_exchange_suffix, AKShareClient, _empty_kline_df, _empty_lhb_summary_df, _empty_broker_detail_df | 已跟踪 |
| [src/data_sources/baostock_client.py](../src/data_sources/baostock_client.py) | 239 | 供应读取与质量 | _to_baostock_code, _to_standard_code, BaostockClient, _format_date, _empty_kline_df | 已跟踪 |
| [src/data_sources/history_quality.py](../src/data_sources/history_quality.py) | 89 | 供应读取与质量 | _compare_files, verify_history_preserved | 已跟踪 |
| [src/data_sources/parquet_merge.py](../src/data_sources/parquet_merge.py) | 52 | 供应读取与质量 | merge_kline_file | 已跟踪 |
| [src/data_sources/project_quality.py](../src/data_sources/project_quality.py) | 72 | 供应读取与质量 | inspect_project_data | 已跟踪 |
| [src/data_sources/quote_quality.py](../src/data_sources/quote_quality.py) | 37 | 供应读取与质量 | verify_historical_prices | 已跟踪 |
| [src/data_sources/reporting_windows.py](../src/data_sources/reporting_windows.py) | 25 | 供应读取与质量 | reporting_window_days | 已跟踪 |
| [src/data_sources/simtradedata_loader.py](../src/data_sources/simtradedata_loader.py) | 250 | 供应读取与质量 | load_stocks_parquet, _normalize_date_column, _build_date_filter, _compute_pct_chg, _join_valuation, _is_a_share_stock_code | 已跟踪 |
| [src/data_sources/simtradedata_metadata.py](../src/data_sources/simtradedata_metadata.py) | 148 | 供应读取与质量 | resolve_export_dir, _normalize_date_series, _normalize_symbol, load_status_lookup, load_benchmark | 已跟踪 |
| [src/data_sources/snapshot_merge.py](../src/data_sources/snapshot_merge.py) | 57 | 供应读取与质量 | merge_history, preserve_snapshot_history | 已跟踪 |
| [src/data_sources/snapshot_quality.py](../src/data_sources/snapshot_quality.py) | 141 | 供应读取与质量 | inspect_snapshot, inspect_active_universe, require_snapshot_quality | 已跟踪 |
| [src/data_sources/snapshot_seed.py](../src/data_sources/snapshot_seed.py) | 61 | 供应读取与质量 | seed_database_from_snapshot | 已跟踪 |
| [src/data_sources/tushare_client.py](../src/data_sources/tushare_client.py) | 352 | 供应读取与质量 | _load_token_from_config, _format_ts_date, _format_output_date, TushareClient, _empty_kline_df, _empty_daily_basic_df, _empty_lhb_df | 已跟踪 |
| [src/features/__init__.py](../src/features/__init__.py) | 9 | 事件特征构造 | 初始化 / 常量 / 入口 | 已跟踪 |
| [src/features/build_lhb_features.py](../src/features/build_lhb_features.py) | 205 | 事件特征构造 | build_lhb_event_table, _compute_historical_st, _pivot_broker_detail | 已跟踪 |
| [src/features/compute_lhb_factors.py](../src/features/compute_lhb_factors.py) | 229 | 事件特征构造 | _safe_divide, _detect_window_days, compute_lhb_factors, _add_seat_quality_counts, _add_empty_factor_cols | 已跟踪 |
| [src/ingestion/__init__.py](../src/ingestion/__init__.py) | 1 | 获取、归档与受控更新 | 初始化 / 常量 / 入口 | 已跟踪 |
| [src/ingestion/failure_tracker.py](../src/ingestion/failure_tracker.py) | 235 | 获取、归档与受控更新 | with_retry, FailureTracker, summary_tracker, detail_tracker, _now | 已跟踪 |
| [src/ingestion/fetch_daily_kline.py](../src/ingestion/fetch_daily_kline.py) | 161 | 获取、归档与受控更新 | fetch_daily_kline, _get_client, _get_stock_list | 已跟踪 |
| [src/ingestion/fetch_lhb_detail.py](../src/ingestion/fetch_lhb_detail.py) | 354 | 获取、归档与受控更新 | fetch_lhb_detail, restore_lhb_detail, _fetch_both_sides, _complete_pair_set, _resolve_events, _to_pair_set, _load_existing, _merge_and_save, _load_failures_dir, _get_client | 已跟踪 |
| [src/ingestion/fetch_lhb_summary.py](../src/ingestion/fetch_lhb_summary.py) | 359 | 获取、归档与受控更新 | fetch_lhb_summary, restore_lhb_summary, _load_existing, _get_missing_trading_dates, _merge_and_save, _aggregate_multi_reason_rows, _parse_range_key, _load_failures_dir, _get_client | 已跟踪 |
| [src/ingestion/lhb_detail_archive.py](../src/ingestion/lhb_detail_archive.py) | 247 | 获取、归档与受控更新 | fetch_range, acquire_archive, _acquire_archive, canonical_seats, refresh_detail_archive, _refresh_detail_archive | 已跟踪 |
| [src/ingestion/load_kline_simtradedata.py](../src/ingestion/load_kline_simtradedata.py) | 264 | 获取、归档与受控更新 | load_kline_simtradedata, _load_config, _resolve_export_dir, _get_missing_trading_dates, _missing_dates_from_counts | 已跟踪 |
| [src/ingestion/update_contract.py](../src/ingestion/update_contract.py) | 90 | 获取、归档与受控更新 | write_report, update_lock, recover_publication, publish_managed | 已跟踪 |
| [src/ingestion/update_simtradedata.py](../src/ingestion/update_simtradedata.py) | 490 | 获取、归档与受控更新 | SimTradeDataUpdateResult, update_simtradedata, _load_config, _completed_day_ceiling, _resolve_update_date, _resolve_path, _resolve_python, _download_tdx_package, _tdx_challenge_cookie, _tdx_remote_info, _user_agent, _is_valid_zip, _read_json, _file_fingerprint, _validate_upstream_checkout, _run | 已跟踪 |
| [src/labels/__init__.py](../src/labels/__init__.py) | 17 | 后验目标与事件检测 | 初始化 / 常量 / 入口 | 已跟踪 |
| [src/labels/generate_future_labels.py](../src/labels/generate_future_labels.py) | 477 | 后验目标与事件检测 | _limit_up_flag, _event_forward_data, _merge_benchmark_returns, _next_trading_day_map, generate_future_labels | 已跟踪 |
| [src/labels/limit_up_detector.py](../src/labels/limit_up_detector.py) | 121 | 后验目标与事件检测 | get_board_type, get_limit_up_threshold, detect_limit_up, detect_limit_up_series | 已跟踪 |
| [src/mlresearch/__init__.py](../src/mlresearch/__init__.py) | 1 | ML 数据、训练、评价与核查 | 初始化 / 常量 / 入口 | 未跟踪（需发布筛选） |
| [src/mlresearch/audit.py](../src/mlresearch/audit.py) | 112 | ML 数据、训练、评价与核查 | audit | 未跟踪（需发布筛选） |
| [src/mlresearch/blend_audit.py](../src/mlresearch/blend_audit.py) | 112 | ML 数据、训练、评价与核查 | reference_scores, reference_decisions, audit_blend | 未跟踪（需发布筛选） |
| [src/mlresearch/contracts.py](../src/mlresearch/contracts.py) | 111 | ML 数据、训练、评价与核查 | MLSpec, feature_names, benchmark_proposal, completion_proposal | 未跟踪（需发布筛选） |
| [src/mlresearch/dataset.py](../src/mlresearch/dataset.py) | 109 | ML 数据、训练、评价与核查 | price_features, forward_labels, assign_splits, build_panel | 未跟踪（需发布筛选） |
| [src/mlresearch/evaluation.py](../src/mlresearch/evaluation.py) | 70 | ML 数据、训练、评价与核查 | block_interval, evaluate | 未跟踪（需发布筛选） |
| [src/mlresearch/financial_audit.py](../src/mlresearch/financial_audit.py) | 164 | ML 数据、训练、评价与核查 | annual_reference, audit_market_calendar_scope, market_reference, audit_financial | 未跟踪（需发布筛选） |
| [src/mlresearch/financial_study.py](../src/mlresearch/financial_study.py) | 199 | ML 数据、训练、评价与核查 | financial_plan, financial_source_identity, financial_features, market_calendar_scope, market_features, run_financial | 未跟踪（需发布筛选） |
| [src/mlresearch/fixed_blend.py](../src/mlresearch/fixed_blend.py) | 166 | ML 数据、训练、评价与核查 | read, blend_plan, source_identity, scores, decisions, run_blend | 未跟踪（需发布筛选） |
| [src/mlresearch/runner.py](../src/mlresearch/runner.py) | 142 | ML 数据、训练、评价与核查 | fit_models, predict_panel, run | 未跟踪（需发布筛选） |
| [src/mlresearch/study.py](../src/mlresearch/study.py) | 305 | ML 数据、训练、评价与核查 | study_plan, Fits, run_study | 未跟踪（需发布筛选） |
| [src/mlresearch/study_audit.py](../src/mlresearch/study_audit.py) | 170 | ML 数据、训练、评价与核查 | read, verify_links, audit_study | 未跟踪（需发布筛选） |
| [src/mlresearch/study_data.py](../src/mlresearch/study_data.py) | 72 | ML 数据、训练、评价与核查 | open_labels, fixed_scores, size_buckets, balanced_score, score_decisions | 未跟踪（需发布筛选） |
| [src/mlresearch/study_evaluation.py](../src/mlresearch/study_evaluation.py) | 106 | ML 数据、训练、评价与核查 | signal_metrics, select_validation, exposure_diagnostics, account_statistics | 未跟踪（需发布筛选） |
| [src/mlresearch/study_report.py](../src/mlresearch/study_report.py) | 60 | ML 数据、训练、评价与核查 | render_report | 未跟踪（需发布筛选） |
| [src/mlresearch/universe.py](../src/mlresearch/universe.py) | 172 | ML 数据、训练、评价与核查 | universe_plan, pool_masks, weekly_random, run_universe | 未跟踪（需发布筛选） |
| [src/mlresearch/universe_audit.py](../src/mlresearch/universe_audit.py) | 104 | ML 数据、训练、评价与核查 | read, check_account_audit, audit_universe | 未跟踪（需发布筛选） |
| [src/reports/__init__.py](../src/reports/__init__.py) | 21 | 图表与旧研究报告 | 初始化 / 常量 / 入口 | 已跟踪 |
| [src/reports/make_report.py](../src/reports/make_report.py) | 393 | 图表与旧研究报告 | generate_html_report, _fig_to_base64, _format_pct, _format_number, _build_stats_rows, _build_key_metrics | 已跟踪 |
| [src/reports/plot_results.py](../src/reports/plot_results.py) | 508 | 图表与旧研究报告 | plot_return_distribution, plot_factor_vs_return, plot_grid_search_top_n, plot_grid_search_metric_lines, plot_grid_search_results, plot_grid_search_heatmap, plot_cumulative_returns, plot_monthly_stats, _save_figure | 已跟踪 |
| [src/researchops/__init__.py](../src/researchops/__init__.py) | 1 | 任务、冻结、控制与持续研究 | 初始化 / 常量 / 入口 | 已跟踪 |
| [src/researchops/controller.py](../src/researchops/controller.py) | 805 | 任务、冻结、控制与持续研究 | DispatchDeferred, obj, synthesis_notice_schema, validate, config, Campaigns, CodexTransport, Controller | 已跟踪 |
| [src/researchops/loop.py](../src/researchops/loop.py) | 501 | 任务、冻结、控制与持续研究 | read, quota_failure, Loops, Supervisor | 未跟踪（需发布筛选） |
| [src/researchops/ml_experiments.py](../src/researchops/ml_experiments.py) | 229 | 任务、冻结、控制与持续研究 | register, verify_completed, recover | 未跟踪（需发布筛选） |
| [src/researchops/notification.py](../src/researchops/notification.py) | 53 | 任务、冻结、控制与持续研究 | desktop_notice | 未跟踪（需发布筛选） |
| [src/researchops/quota.py](../src/researchops/quota.py) | 75 | 任务、冻结、控制与持续研究 | read_limits, quota_status | 未跟踪（需发布筛选） |
| [src/researchops/service.py](../src/researchops/service.py) | 574 | 任务、冻结、控制与持续研究 | Research | 已跟踪 |
| [src/researchops/store.py](../src/researchops/store.py) | 522 | 任务、冻结、控制与持续研究 | encode, text, items, task_spec, Store | 已跟踪 |
| [src/technical/__init__.py](../src/technical/__init__.py) | 2 | 快照、市场规则、组合账户、核查与 UI | 初始化 / 常量 / 入口 | 已跟踪 |
| [src/technical/artifacts.py](../src/technical/artifacts.py) | 101 | 快照、市场规则、组合账户、核查与 UI | utc_now, jsonable, write_json, digest, fingerprint, content_id, verify_inventory, exclusive_run, verify_artifacts, records | 已跟踪 |
| [src/technical/contracts.py](../src/technical/contracts.py) | 153 | 快照、市场规则、组合账户、核查与 UI | Strategy, Execution, Experiment, ResearchSpec, describe | 已跟踪 |
| [src/technical/data.py](../src/technical/data.py) | 128 | 快照、市场规则、组合账户、核查与 UI | dates, prepare_snapshot | 已跟踪 |
| [src/technical/diagnostics.py](../src/technical/diagnostics.py) | 225 | 快照、市场规则、组合账户、核查与 UI | contributions, period_stats, market_states, drawdown_episodes, summarize, load_account, analysis_for, execution_delays, execution_for, inspect_period, inspect_case | 已跟踪 |
| [src/technical/foundation_data.py](../src/technical/foundation_data.py) | 139 | 快照、市场规则、组合账户、核查与 UI | normalize_filings, prepare_foundation_snapshot, attach_reports | 已跟踪 |
| [src/technical/foundation_lab.py](../src/technical/foundation_lab.py) | 125 | 快照、市场规则、组合账户、核查与 UI | plan, run_suite | 已跟踪 |
| [src/technical/foundation_review.py](../src/technical/foundation_review.py) | 143 | 快照、市场规则、组合账户、核查与 UI | joint_block_intervals, build_evidence | 已跟踪 |
| [src/technical/foundation_signals.py](../src/technical/foundation_signals.py) | 122 | 快照、市场规则、组合账户、核查与 UI | definition, decide | 已跟踪 |
| [src/technical/laboratory.py](../src/technical/laboratory.py) | 137 | 快照、市场规则、组合账户、核查与 UI | plan, score_trial, select, run_batch | 已跟踪 |
| [src/technical/market.py](../src/technical/market.py) | 58 | 快照、市场规则、组合账户、核查与 UI | Market | 已跟踪 |
| [src/technical/portfolio.py](../src/technical/portfolio.py) | 243 | 快照、市场规则、组合账户、核查与 UI | costs, simulate | 已跟踪 |
| [src/technical/regime_lab.py](../src/technical/regime_lab.py) | 140 | 快照、市场规则、组合账户、核查与 UI | plan, review, run_suite | 已跟踪 |
| [src/technical/regimes.py](../src/technical/regimes.py) | 190 | 快照、市场规则、组合账户、核查与 UI | definition, market_features, gaussian_filter, infer_states, allocation_weights, combine_decisions, prepare_components, prepare_states | 已跟踪 |
| [src/technical/rules.py](../src/technical/rules.py) | 37 | 快照、市场规则、组合账户、核查与 UI | st_exit_limits | 已跟踪 |
| [src/technical/runner.py](../src/technical/runner.py) | 200 | 快照、市场规则、组合账户、核查与 UI | code_inventory, verify, run | 已跟踪 |
| [src/technical/server.py](../src/technical/server.py) | 430 | 快照、市场规则、组合账户、核查与 UI | make_server, serve | 已跟踪 |
| [src/technical/signals.py](../src/technical/signals.py) | 77 | 快照、市场规则、组合账户、核查与 UI | check_keys, features, decision_dates, decisions | 已跟踪 |
| [src/technical/web.py](../src/technical/web.py) | 11 | 快照、市场规则、组合账户、核查与 UI | render_page | 已跟踪 |
| [src/utils/__init__.py](../src/utils/__init__.py) | 38 | I/O、日历与日志 | 初始化 / 常量 / 入口 | 已跟踪 |
| [src/utils/calendar.py](../src/utils/calendar.py) | 333 | I/O、日历与日志 | _resolve_simtradedata_export_dir, _load_from_simtradedata, _fetch_trading_dates_akshare, _generate_fallback_dates, _load_trading_dates, invalidate_cache, _normalize_date, calendar_max_date, get_trading_dates, get_next_n_trading_dates, get_prev_n_trading_dates, is_trading_date, shift_trading_date | 已跟踪 |
| [src/utils/io.py](../src/utils/io.py) | 240 | I/O、日历与日志 | _load_config, _resolve_path, ensure_dir, save_parquet, load_parquet, save_csv, load_csv, get_duckdb_conn, query_parquet, register_parquet_as_table | 已跟踪 |
| [src/utils/logging.py](../src/utils/logging.py) | 102 | I/O、日历与日志 | _load_logging_config, setup_logging, get_logger | 已跟踪 |
| [src/workbench/__init__.py](../src/workbench/__init__.py) | 3 | 旧事件工作台 | 初始化 / 常量 / 入口 | 已跟踪 |
| [src/workbench/artifacts.py](../src/workbench/artifacts.py) | 101 | 旧事件工作台 | utc_now, jsonable, write_json, digest, fingerprint, content_id, verify_inventory, exclusive_run, verify_artifacts, records | 已跟踪 |
| [src/workbench/contracts.py](../src/workbench/contracts.py) | 105 | 旧事件工作台 | StrategyCard | 已跟踪 |
| [src/workbench/data.py](../src/workbench/data.py) | 229 | 旧事件工作台 | dates, check_keys, configured_paths, institutional_features, attach_features, _read_stock_tables, prepare_snapshot | 已跟踪 |
| [src/workbench/portfolio.py](../src/workbench/portfolio.py) | 351 | 旧事件工作台 | Market, fees, _lot_quantity, Position, simulate | 已跟踪 |
| [src/workbench/report.py](../src/workbench/report.py) | 73 | 旧事件工作台 | event_evidence, render_page, export_report | 已跟踪 |
| [src/workbench/research.py](../src/workbench/research.py) | 149 | 旧事件工作台 | select_cohorts, price_labels, block_interval, describe_research | 已跟踪 |
| [src/workbench/rules.py](../src/workbench/rules.py) | 37 | 旧事件工作台 | st_exit_limits | 已跟踪 |
| [src/workbench/runner.py](../src/workbench/runner.py) | 148 | 旧事件工作台 | code_inventory, verify_run, run_research | 已跟踪 |
| [src/workbench/server.py](../src/workbench/server.py) | 243 | 旧事件工作台 | make_server, serve | 已跟踪 |

## tools Python 文件

| 文件 | 行数 | 职责 | 顶层定义（最多 16 项） | Git 状态 |
| --- | --- | --- | --- | --- |
| [tools/apply_quality_metadata.py](../tools/apply_quality_metadata.py) | 40 | 操作、维护或验证入口 | 初始化 / 常量 / 入口 | 已跟踪 |
| [tools/apply_vendor_quality_repair.py](../tools/apply_vendor_quality_repair.py) | 152 | 操作、维护或验证入口 | main | 已跟踪 |
| [tools/audit_legacy_costs.py](../tools/audit_legacy_costs.py) | 97 | 操作、维护或验证入口 | audit | 已跟踪 |
| [tools/audit_local_data.py](../tools/audit_local_data.py) | 124 | 操作、维护或验证入口 | digest, audit | 已跟踪 |
| [tools/build_quality_candidate.py](../tools/build_quality_candidate.py) | 21 | 操作、维护或验证入口 | 初始化 / 常量 / 入口 | 已跟踪 |
| [tools/compare_vendor_snapshots.py](../tools/compare_vendor_snapshots.py) | 112 | 操作、维护或验证入口 | digest, compare | 已跟踪 |
| [tools/controller.py](../tools/controller.py) | 56 | 操作、维护或验证入口 | main | 已跟踪 |
| [tools/diagnose_signal_timing.py](../tools/diagnose_signal_timing.py) | 50 | 操作、维护或验证入口 | main | 已跟踪 |
| [tools/fetch_vendor_quality_repair.py](../tools/fetch_vendor_quality_repair.py) | 78 | 操作、维护或验证入口 | initialize, fetch, main | 已跟踪 |
| [tools/finalize_quality_candidate.py](../tools/finalize_quality_candidate.py) | 74 | 操作、维护或验证入口 | main | 已跟踪 |
| [tools/launcher.py](../tools/launcher.py) | 530 | 操作、维护或验证入口 | ask, ask_yn, ask_float, pause, action_health, action_update, _finish_project_update, action_detail, action_rebuild, action_backtest, action_grid, action_report, action_open, action_tests, action_workbench, main | 已跟踪 |
| [tools/ml_loop.py](../tools/ml_loop.py) | 62 | 操作、维护或验证入口 | main | 未跟踪（需发布筛选） |
| [tools/ml_research.py](../tools/ml_research.py) | 53 | 操作、维护或验证入口 | main | 未跟踪（需发布筛选） |
| [tools/prepare_repair_database.py](../tools/prepare_repair_database.py) | 20 | 操作、维护或验证入口 | 初始化 / 常量 / 入口 | 已跟踪 |
| [tools/publish_quality_repair.py](../tools/publish_quality_repair.py) | 112 | 操作、维护或验证入口 | digest, main | 已跟踪 |
| [tools/refresh_official_benchmark.py](../tools/refresh_official_benchmark.py) | 32 | 操作、维护或验证入口 | main | 已跟踪 |
| [tools/refresh_official_metadata.py](../tools/refresh_official_metadata.py) | 61 | 操作、维护或验证入口 | read_result, main | 已跟踪 |
| [tools/research.py](../tools/research.py) | 139 | 操作、维护或验证入口 | read, main | 已跟踪 |
| [tools/run_research_followups.py](../tools/run_research_followups.py) | 51 | 操作、维护或验证入口 | main | 已跟踪 |
| [tools/technical_research.py](../tools/technical_research.py) | 59 | 操作、维护或验证入口 | main | 已跟踪 |
| [tools/verify_controller_ui.py](../tools/verify_controller_ui.py) | 81 | 操作、维护或验证入口 | main | 已跟踪 |
| [tools/verify_examples.py](../tools/verify_examples.py) | 45 | 操作、维护或验证入口 | main | 已跟踪 |
| [tools/verify_foundations.py](../tools/verify_foundations.py) | 61 | 操作、维护或验证入口 | check | 已跟踪 |
| [tools/verify_foundations_ui.py](../tools/verify_foundations_ui.py) | 57 | 操作、维护或验证入口 | verify | 已跟踪 |
| [tools/verify_ml_completion_ui.py](../tools/verify_ml_completion_ui.py) | 78 | 操作、维护或验证入口 | main | 未跟踪（需发布筛选） |
| [tools/verify_ml_ui.py](../tools/verify_ml_ui.py) | 115 | 操作、维护或验证入口 | main | 未跟踪（需发布筛选） |
| [tools/verify_pipeline.py](../tools/verify_pipeline.py) | 462 | 操作、维护或验证入口 | check, section, verify_calendar, verify_raw, verify_clean, verify_factors, verify_labels, verify_filters, verify_statistics, main | 已跟踪 |
| [tools/verify_regimes_ui.py](../tools/verify_regimes_ui.py) | 67 | 操作、维护或验证入口 | verify | 已跟踪 |
| [tools/verify_research_batch.py](../tools/verify_research_batch.py) | 58 | 操作、维护或验证入口 | check | 已跟踪 |
| [tools/verify_research_e2e.py](../tools/verify_research_e2e.py) | 116 | 操作、维护或验证入口 | digest, main | 已跟踪 |
| [tools/verify_research_loop_ui.py](../tools/verify_research_loop_ui.py) | 106 | 操作、维护或验证入口 | verify, main | 已跟踪 |
| [tools/verify_research_tasks_ui.py](../tools/verify_research_tasks_ui.py) | 96 | 操作、维护或验证入口 | verify | 已跟踪 |
| [tools/verify_research_workers.py](../tools/verify_research_workers.py) | 137 | 操作、维护或验证入口 | main | 已跟踪 |
| [tools/verify_technical.py](../tools/verify_technical.py) | 65 | 操作、维护或验证入口 | audit | 已跟踪 |
| [tools/verify_technical_ui.py](../tools/verify_technical_ui.py) | 5 | 操作、维护或验证入口 | 初始化 / 常量 / 入口 | 已跟踪 |
| [tools/verify_workbench.py](../tools/verify_workbench.py) | 101 | 操作、维护或验证入口 | verify | 已跟踪 |
| [tools/verify_workbench_ui.py](../tools/verify_workbench_ui.py) | 123 | 操作、维护或验证入口 | Browser, verify | 已跟踪 |
| [tools/workbench.py](../tools/workbench.py) | 59 | 操作、维护或验证入口 | main | 已跟踪 |

## samples Python 文件

| 文件 | 行数 | 职责 | 顶层定义（最多 16 项） | Git 状态 |
| --- | --- | --- | --- | --- |
| [samples/_bootstrap.py](../samples/_bootstrap.py) | 19 | 用户示例 | 初始化 / 常量 / 入口 | 已跟踪 |
| [samples/sample_01_update_data.py](../samples/sample_01_update_data.py) | 55 | 用户示例 | main | 已跟踪 |
| [samples/sample_02_full_pipeline.py](../samples/sample_02_full_pipeline.py) | 66 | 用户示例 | main | 已跟踪 |
| [samples/sample_03_basic_backtest.py](../samples/sample_03_basic_backtest.py) | 79 | 用户示例 | main | 已跟踪 |
| [samples/sample_04_custom_filters.py](../samples/sample_04_custom_filters.py) | 78 | 用户示例 | main | 已跟踪 |
| [samples/sample_05_grid_search_oos.py](../samples/sample_05_grid_search_oos.py) | 64 | 用户示例 | main | 已跟踪 |
| [samples/sample_06_factor_quantiles.py](../samples/sample_06_factor_quantiles.py) | 68 | 用户示例 | main | 已跟踪 |
| [samples/sample_07_group_analysis.py](../samples/sample_07_group_analysis.py) | 73 | 用户示例 | main | 已跟踪 |
| [samples/sample_08_event_query.py](../samples/sample_08_event_query.py) | 69 | 用户示例 | main | 已跟踪 |
| [samples/sample_09_plots_report.py](../samples/sample_09_plots_report.py) | 75 | 用户示例 | main | 已跟踪 |
| [samples/sample_10_tradable_alpha.py](../samples/sample_10_tradable_alpha.py) | 69 | 用户示例 | describe, main | 已跟踪 |

## 测试

| 文件 | 逻辑容量 | Git 状态 |
| --- | --- | --- |
| [__init__.py](../tests/__init__.py) | 2 B · 0.00 MB | 已跟踪 |
| [test_active_universe.py](../tests/test_active_universe.py) | 1,310 B · 0.00 MB | 已跟踪 |
| [test_baostock_transport.py](../tests/test_baostock_transport.py) | 944 B · 0.00 MB | 已跟踪 |
| [test_controller.py](../tests/test_controller.py) | 11,456 B · 0.01 MB | 已跟踪 |
| [test_detail_integrity.py](../tests/test_detail_integrity.py) | 12,836 B · 0.01 MB | 已跟踪 |
| [test_factor_calculation.py](../tests/test_factor_calculation.py) | 16,986 B · 0.02 MB | 已跟踪 |
| [test_failure_tracker.py](../tests/test_failure_tracker.py) | 9,652 B · 0.01 MB | 已跟踪 |
| [test_filter_engine.py](../tests/test_filter_engine.py) | 8,363 B · 0.01 MB | 已跟踪 |
| [test_financial_context.py](../tests/test_financial_context.py) | 2,977 B · 0.00 MB | 未跟踪、未忽略 |
| [test_foundations.py](../tests/test_foundations.py) | 8,059 B · 0.01 MB | 已跟踪 |
| [test_incremental_semantics.py](../tests/test_incremental_semantics.py) | 4,881 B · 0.00 MB | 已跟踪 |
| [test_label_generation.py](../tests/test_label_generation.py) | 11,473 B · 0.01 MB | 已跟踪 |
| [test_market_calendar_scope.py](../tests/test_market_calendar_scope.py) | 2,983 B · 0.00 MB | 未跟踪、未忽略 |
| [test_ml_completion.py](../tests/test_ml_completion.py) | 5,638 B · 0.01 MB | 未跟踪、未忽略 |
| [test_ml_loop.py](../tests/test_ml_loop.py) | 27,494 B · 0.03 MB | 未跟踪、未忽略 |
| [test_ml_universe.py](../tests/test_ml_universe.py) | 4,136 B · 0.00 MB | 未跟踪、未忽略 |
| [test_mlresearch.py](../tests/test_mlresearch.py) | 11,231 B · 0.01 MB | 未跟踪、未忽略 |
| [test_parquet_merge.py](../tests/test_parquet_merge.py) | 1,520 B · 0.00 MB | 已跟踪 |
| [test_project_data_quality.py](../tests/test_project_data_quality.py) | 2,503 B · 0.00 MB | 已跟踪 |
| [test_project_review_regressions.py](../tests/test_project_review_regressions.py) | 14,219 B · 0.01 MB | 已跟踪 |
| [test_regimes.py](../tests/test_regimes.py) | 7,628 B · 0.01 MB | 已跟踪 |
| [test_research_loop.py](../tests/test_research_loop.py) | 5,502 B · 0.01 MB | 已跟踪 |
| [test_research_safety.py](../tests/test_research_safety.py) | 8,969 B · 0.01 MB | 已跟踪 |
| [test_researchops.py](../tests/test_researchops.py) | 11,894 B · 0.01 MB | 已跟踪 |
| [test_simtradedata_loader.py](../tests/test_simtradedata_loader.py) | 12,499 B · 0.01 MB | 已跟踪 |
| [test_simtradedata_updater.py](../tests/test_simtradedata_updater.py) | 5,357 B · 0.01 MB | 已跟踪 |
| [test_snapshot_merge.py](../tests/test_snapshot_merge.py) | 9,232 B · 0.01 MB | 已跟踪 |
| [test_summary_update_reporting.py](../tests/test_summary_update_reporting.py) | 1,052 B · 0.00 MB | 已跟踪 |
| [test_technical.py](../tests/test_technical.py) | 12,303 B · 0.01 MB | 已跟踪 |
| [test_unit_contract.py](../tests/test_unit_contract.py) | 11,574 B · 0.01 MB | 已跟踪 |
| [test_update_contract.py](../tests/test_update_contract.py) | 9,309 B · 0.01 MB | 已跟踪 |
| [test_update_date.py](../tests/test_update_date.py) | 540 B · 0.00 MB | 已跟踪 |
| [test_workbench.py](../tests/test_workbench.py) | 13,454 B · 0.01 MB | 已跟踪 |

## 配置

| 文件 | 逻辑容量 | Git 状态 |
| --- | --- | --- |
| [config.yaml](../config/config.yaml) | 3,683 B · 0.00 MB | 已跟踪 |
| [data_sources.yaml](../config/data_sources.yaml) | 3,473 B · 0.00 MB | 已跟踪 |
| [institutional_research.json](../config/institutional_research.json) | 500 B · 0.00 MB | 已跟踪 |
| [technical_batch.json](../config/technical_batch.json) | 639 B · 0.00 MB | 已跟踪 |
| [technical_research.json](../config/technical_research.json) | 635 B · 0.00 MB | 已跟踪 |

## 前端资源

| 文件 | 逻辑容量 | Git 状态 |
| --- | --- | --- |
| [app.js](../src/technical/assets/app.js) | 45,646 B · 0.05 MB | 已跟踪 |
| [chart.js](../src/technical/assets/chart.js) | 5,174 B · 0.01 MB | 已跟踪 |
| [controller.js](../src/technical/assets/controller.js) | 9,873 B · 0.01 MB | 已跟踪 |
| [foundations.js](../src/technical/assets/foundations.js) | 8,285 B · 0.01 MB | 已跟踪 |
| [index.html](../src/technical/assets/index.html) | 11,367 B · 0.01 MB | 已跟踪 |
| [ml.js](../src/technical/assets/ml.js) | 14,049 B · 0.01 MB | 未跟踪、未忽略 |
| [regimes.js](../src/technical/assets/regimes.js) | 12,216 B · 0.01 MB | 已跟踪 |
| [style.css](../src/technical/assets/style.css) | 20,377 B · 0.02 MB | 已跟踪 |
| [tasks.js](../src/technical/assets/tasks.js) | 14,325 B · 0.01 MB | 已跟踪 |

## 旧工作台资源

| 文件 | 逻辑容量 | Git 状态 |
| --- | --- | --- |
| [app.js](../src/workbench/web/app.js) | 25,792 B · 0.03 MB | 已跟踪 |
| [index.html](../src/workbench/web/index.html) | 7,532 B · 0.01 MB | 已跟踪 |
| [style.css](../src/workbench/web/style.css) | 6,395 B · 0.01 MB | 已跟踪 |

## 兼容代码

| 文件 | 逻辑容量 | Git 状态 |
| --- | --- | --- |
| [__init__.py](../compat/__init__.py) | 88 B · 0.00 MB | 已跟踪 |
| [fcntl.py](../compat/fcntl.py) | 1,011 B · 0.00 MB | 已跟踪 |
| [sitecustomize.py](../compat/sitecustomize.py) | 2,836 B · 0.00 MB | 已跟踪 |

## Notebook

| 文件 | 逻辑容量 | Git 状态 |
| --- | --- | --- |
| 01_data_check.ipynb（本机附件，不随公开源码提供） | 4,240 B · 0.00 MB | 已跟踪 |
| 02_feature_check.ipynb（本机附件，不随公开源码提供） | 2,955 B · 0.00 MB | 已跟踪 |
| 03_backtest_exploration.ipynb（本机附件，不随公开源码提供） | 5,555 B · 0.01 MB | 已跟踪 |
| data/output/grid_search_plots/heatmap_net_oo_return_1d__min_net_buy_ratio_vs_min_buy1_concentration.png（本机附件，不随公开源码提供） | 77,767 B · 0.08 MB | 未跟踪、未忽略 |
| data/output/grid_search_plots/heatmap_sample_count__min_net_buy_ratio_vs_min_buy1_concentration.png（本机附件，不随公开源码提供） | 98,782 B · 0.10 MB | 未跟踪、未忽略 |
| data/output/grid_search_plots/lines_net_oo_return_1d_vs_min_buy1_concentration.png（本机附件，不随公开源码提供） | 55,600 B · 0.06 MB | 未跟踪、未忽略 |
| data/output/grid_search_plots/lines_net_oo_return_1d_vs_min_net_buy_ratio.png（本机附件，不随公开源码提供） | 50,477 B · 0.05 MB | 未跟踪、未忽略 |
| data/output/grid_search_plots/top15_net_oo_return_1d.png（本机附件，不随公开源码提供） | 128,224 B · 0.13 MB | 未跟踪、未忽略 |

## 设计与报告文档

| 文件 | 逻辑容量 | Git 状态 |
| --- | --- | --- |
| [CODEX_PROJECT_DELIVERY_PLAN_20261005.html](CODEX_PROJECT_DELIVERY_PLAN_20261005.html) | 54,705 B · 0.05 MB | 未跟踪、未忽略 |
| [CODEX_PROJECT_DELIVERY_PLAN_20261005.md](CODEX_PROJECT_DELIVERY_PLAN_20261005.md) | 34,374 B · 0.03 MB | 未跟踪、未忽略 |
| [CONTROLLER_GUIDE.md](CONTROLLER_GUIDE.md) | 5,357 B · 0.01 MB | 已跟踪 |
| [DATA_CONTRACT.md](DATA_CONTRACT.md) | 5,998 B · 0.01 MB | 已跟踪 |
| [FOUNDATION_RESEARCH_V4.md](FOUNDATION_RESEARCH_V4.md) | 7,195 B · 0.01 MB | 已跟踪 |
| LOCAL_DELIVERY_20261006.html（本机附件，不随公开源码提供） | 6,576 B · 0.01 MB | 忽略 |
| LOCAL_DELIVERY_20261006.md（本机附件，不随公开源码提供） | 5,187 B · 0.01 MB | 忽略 |
| LOCAL_WORKSPACE.md（本机附件，不随公开源码提供） | 4,252 B · 0.00 MB | 忽略 |
| [ML_ACCOUNT_RESULTS.html](ML_ACCOUNT_RESULTS.html) | 1,362,724 B · 1.36 MB | 未跟踪、未忽略 |
| [ML_ACCOUNT_RESULTS.md](ML_ACCOUNT_RESULTS.md) | 20,644 B · 0.02 MB | 未跟踪、未忽略 |
| [ML_AUTO_LOOP.md](ML_AUTO_LOOP.md) | 6,665 B · 0.01 MB | 未跟踪、未忽略 |
| [ML_CODE_WALKTHROUGH.html](ML_CODE_WALKTHROUGH.html) | 539,728 B · 0.54 MB | 未跟踪、未忽略 |
| [ML_CODE_WALKTHROUGH.md](ML_CODE_WALKTHROUGH.md) | 208,609 B · 0.21 MB | 未跟踪、未忽略 |
| [ML_COMPLETION_CODE.html](ML_COMPLETION_CODE.html) | 179,530 B · 0.18 MB | 未跟踪、未忽略 |
| [ML_COMPLETION_CODE.md](ML_COMPLETION_CODE.md) | 7,316 B · 0.01 MB | 未跟踪、未忽略 |
| [ML_CYCLE_PROGRESS_20261005.html](ML_CYCLE_PROGRESS_20261005.html) | 13,445 B · 0.01 MB | 未跟踪、未忽略 |
| [ML_CYCLE_PROGRESS_20261005.md](ML_CYCLE_PROGRESS_20261005.md) | 9,895 B · 0.01 MB | 未跟踪、未忽略 |
| [ML_FROM_ZERO.html](ML_FROM_ZERO.html) | 17,639 B · 0.02 MB | 未跟踪、未忽略 |
| [ML_FROM_ZERO.md](ML_FROM_ZERO.md) | 12,422 B · 0.01 MB | 未跟踪、未忽略 |
| [ML_FROM_ZERO_TECHNICAL.html](ML_FROM_ZERO_TECHNICAL.html) | 53,501 B · 0.05 MB | 未跟踪、未忽略 |
| [ML_FROM_ZERO_TECHNICAL.md](ML_FROM_ZERO_TECHNICAL.md) | 39,871 B · 0.04 MB | 未跟踪、未忽略 |
| [ML_LOOP_LEARNING_UPDATE_20261004.html](ML_LOOP_LEARNING_UPDATE_20261004.html) | 9,668 B · 0.01 MB | 未跟踪、未忽略 |
| [ML_LOOP_LEARNING_UPDATE_20261004.md](ML_LOOP_LEARNING_UPDATE_20261004.md) | 7,599 B · 0.01 MB | 未跟踪、未忽略 |
| ML_PERFORMANCE_CYCLE_20261005.html（本机附件，不随公开源码提供） | 8,750 B · 0.01 MB | 未跟踪、未忽略 |
| ML_PERFORMANCE_CYCLE_20261005.md（本机附件，不随公开源码提供） | 6,677 B · 0.01 MB | 未跟踪、未忽略 |
| [ML_RESEARCH_GUIDE.md](ML_RESEARCH_GUIDE.md) | 9,625 B · 0.01 MB | 未跟踪、未忽略 |
| [ML_RISK_INFORMATION_DISCOVERY_20261005.html](ML_RISK_INFORMATION_DISCOVERY_20261005.html) | 9,532 B · 0.01 MB | 未跟踪、未忽略 |
| [ML_RISK_INFORMATION_DISCOVERY_20261005.md](ML_RISK_INFORMATION_DISCOVERY_20261005.md) | 6,988 B · 0.01 MB | 未跟踪、未忽略 |
| [ML_SYSTEM_BLUEPRINT_20261005.html](ML_SYSTEM_BLUEPRINT_20261005.html) | 191,716 B · 0.19 MB | 未跟踪、未忽略 |
| [ML_SYSTEM_BLUEPRINT_20261005.md](ML_SYSTEM_BLUEPRINT_20261005.md) | 62,181 B · 0.06 MB | 未跟踪、未忽略 |
| [ML_UNIVERSE_AND_ACCOUNT_FAQ.html](ML_UNIVERSE_AND_ACCOUNT_FAQ.html) | 941,381 B · 0.94 MB | 未跟踪、未忽略 |
| [ML_UNIVERSE_AND_ACCOUNT_FAQ.md](ML_UNIVERSE_AND_ACCOUNT_FAQ.md) | 19,702 B · 0.02 MB | 未跟踪、未忽略 |
| [PROJECT_CHARTER.html](PROJECT_CHARTER.html) | 32,417 B · 0.03 MB | 未跟踪、未忽略 |
| [PROJECT_CHARTER.md](PROJECT_CHARTER.md) | 25,192 B · 0.03 MB | 未跟踪、未忽略 |
| [PROJECT_CHARTER_CHANGELOG.html](PROJECT_CHARTER_CHANGELOG.html) | 3,953 B · 0.00 MB | 未跟踪、未忽略 |
| [PROJECT_CHARTER_CHANGELOG.md](PROJECT_CHARTER_CHANGELOG.md) | 862 B · 0.00 MB | 未跟踪、未忽略 |
| [PROJECT_CHARTER_FLOW.svg](PROJECT_CHARTER_FLOW.svg) | 3,643 B · 0.00 MB | 未跟踪、未忽略 |
| [PUBLIC_DELIVERY.md](PUBLIC_DELIVERY.md) | 3,851 B · 0.00 MB | 未跟踪、未忽略 |
| [REGIME_RESEARCH_V5.md](REGIME_RESEARCH_V5.md) | 2,439 B · 0.00 MB | 已跟踪 |
| [RESEARCH_RADAR.html](RESEARCH_RADAR.html) | 26,179 B · 0.03 MB | 未跟踪、未忽略 |
| [RESEARCH_RADAR.json](RESEARCH_RADAR.json) | 16,032 B · 0.02 MB | 未跟踪、未忽略 |
| [RESEARCH_RADAR.md](RESEARCH_RADAR.md) | 19,631 B · 0.02 MB | 未跟踪、未忽略 |
| RESEARCH_RESET_20260926.md（本机附件，不随公开源码提供） | 17,808 B · 0.02 MB | 已跟踪 |
| [RESEARCH_STAGE_TEMPLATE.html](RESEARCH_STAGE_TEMPLATE.html) | 6,747 B · 0.01 MB | 未跟踪、未忽略 |
| [RESEARCH_STAGE_TEMPLATE.md](RESEARCH_STAGE_TEMPLATE.md) | 2,839 B · 0.00 MB | 未跟踪、未忽略 |
| RESEARCH_WORKBENCH_V3.md（本机附件，不随公开源码提供） | 7,332 B · 0.01 MB | 已跟踪 |
| [TECHNICAL_RESEARCH_V2.md](TECHNICAL_RESEARCH_V2.md) | 13,230 B · 0.01 MB | 已跟踪 |
| WORKBENCH_V1.md（本机附件，不随公开源码提供） | 8,760 B · 0.01 MB | 已跟踪 |
| [WORKER_GUIDE.md](WORKER_GUIDE.md) | 10,173 B · 0.01 MB | 已跟踪 |

## 根与应用入口

| 文件 | 逻辑容量 | Git 状态 |
| --- | --- | --- |
| [.gitignore](../../.gitignore) | 534 B · 0.00 MB | 未跟踪、未忽略 |
| [AGENTS.md](../../AGENTS.md) | 4,549 B · 0.00 MB | 已跟踪 |
| [README.md](../../README.md) | 1,233 B · 0.00 MB | 未跟踪、未忽略 |
| [lhb_backtest/.gitignore](../.gitignore) | 935 B · 0.00 MB | 已跟踪 |
| [lhb_backtest/main.py](../main.py) | 18,238 B · 0.02 MB | 已跟踪 |
| lhb_backtest/PLAN.md（本机附件，不随公开源码提供） | 18,591 B · 0.02 MB | 已跟踪 |
| lhb_backtest/PROJECT_REPORT.md（本机附件，不随公开源码提供） | 13,177 B · 0.01 MB | 已跟踪 |
| [lhb_backtest/pyproject.toml](../pyproject.toml) | 1,368 B · 0.00 MB | 已跟踪 |
| [lhb_backtest/README.md](../README.md) | 5,238 B · 0.01 MB | 已跟踪 |
| lhb_backtest/REMEDIATION.md（本机附件，不随公开源码提供） | 7,825 B · 0.01 MB | 已跟踪 |
| [lhb_backtest/requirements.txt](../requirements.txt) | 229 B · 0.00 MB | 已跟踪 |
| lhb_backtest/TEST_REPORT.md（本机附件，不随公开源码提供） | 13,092 B · 0.01 MB | 已跟踪 |
| lhb_backtest/tmp_inspect.txt（本机附件，不随公开源码提供） | 3,070 B · 0.00 MB | 已跟踪 |
| lhb_backtest/一键回测.bat（本机附件，不随公开源码提供） | 254 B · 0.00 MB | 已跟踪 |
| lhb_backtest/一键更新数据.bat（本机附件，不随公开源码提供） | 248 B · 0.00 MB | 已跟踪 |
| lhb_backtest/启动器.bat（本机附件，不随公开源码提供） | 461 B · 0.00 MB | 已跟踪 |
| lhb_backtest/席位明细续传.bat（本机附件，不随公开源码提供） | 266 B · 0.00 MB | 已跟踪 |
| lhb_backtest/研究工作台.bat（本机附件，不随公开源码提供） | 278 B · 0.00 MB | 已跟踪 |
| lhb_backtest/网格搜索.bat（本机附件，不随公开源码提供） | 254 B · 0.00 MB | 已跟踪 |
| [.github/workflows/ci.yml](../../.github/workflows/ci.yml) | 1,057 B · 0.00 MB | 未跟踪、未忽略 |

## 历史研究目录

| 目录/文件 | 文件数 | 逻辑容量 | 跟踪 | 忽略 | 未跟踪未忽略 |
| --- | --- | --- | --- | --- | --- |
| lhb_backtest/research/RESEARCH_NOTES.md（本机附件，不随公开源码提供） | 1 | 8,685 B · 0.01 MB | 1 | 0 | 0 |
| lhb_backtest/research/__pycache__（本机附件，不随公开源码提供） | 4 | 26,574 B · 0.03 MB | 0 | 4 | 0 |
| lhb_backtest/research/_budget02_before_1791124470520210100（本机附件，不随公开源码提供） | 6 | 106,708 B · 0.11 MB | 0 | 0 | 6 |
| lhb_backtest/research/_contracts.py（本机附件，不随公开源码提供） | 1 | 1,325 B · 0.00 MB | 0 | 0 | 1 |
| lhb_backtest/research/audit_20260923（本机附件，不随公开源码提供） | 72 | 658,074 B · 0.66 MB | 0 | 41 | 31 |
| lhb_backtest/research/codex_delivery_plan_20261005（本机附件，不随公开源码提供） | 1138 | 24,866,327 B · 24.87 MB | 0 | 394 | 744 |
| lhb_backtest/research/controller_delivery_20260928（本机附件，不随公开源码提供） | 3 | 47,816 B · 0.05 MB | 3 | 0 | 0 |
| lhb_backtest/research/data_acceptance_20260925（本机附件，不随公开源码提供） | 87 | 5,736,392 B · 5.74 MB | 0 | 31 | 56 |
| lhb_backtest/research/exp01_gap_reversion.py（本机附件，不随公开源码提供） | 1 | 3,514 B · 0.00 MB | 1 | 0 | 0 |
| lhb_backtest/research/exp02_seat_quality.py（本机附件，不随公开源码提供） | 1 | 3,521 B · 0.00 MB | 1 | 0 | 0 |
| lhb_backtest/research/exp03_board_chasing.py（本机附件，不随公开源码提供） | 1 | 3,566 B · 0.00 MB | 1 | 0 | 0 |
| lhb_backtest/research/exp04_relay_decompose.py（本机附件，不随公开源码提供） | 1 | 4,591 B · 0.00 MB | 1 | 0 | 0 |
| lhb_backtest/research/exp05_relay_honest.py（本机附件，不随公开源码提供） | 1 | 4,544 B · 0.00 MB | 1 | 0 | 0 |
| lhb_backtest/research/exp06_seal_predictors.py（本机附件，不随公开源码提供） | 1 | 4,781 B · 0.00 MB | 1 | 0 | 0 |
| lhb_backtest/research/exp07_medium_horizon.py（本机附件，不随公开源码提供） | 1 | 4,631 B · 0.00 MB | 1 | 0 | 0 |
| lhb_backtest/research/exp08_inst_5d_robust.py（本机附件，不随公开源码提供） | 1 | 3,253 B · 0.00 MB | 1 | 0 | 0 |
| lhb_backtest/research/exp09_decision_tree.py（本机附件，不随公开源码提供） | 1 | 7,615 B · 0.01 MB | 1 | 0 | 0 |
| lhb_backtest/research/foundation_replication_20260927（本机附件，不随公开源码提供） | 10 | 96,129 B · 0.10 MB | 10 | 0 | 0 |
| lhb_backtest/research/independent_review_20261002（本机附件，不随公开源码提供） | 19 | 147,033 B · 0.15 MB | 0 | 3 | 16 |
| lhb_backtest/research/infrastructure_review_20261002（本机附件，不随公开源码提供） | 26 | 459,533 B · 0.46 MB | 0 | 9 | 17 |
| lhb_backtest/research/institutional_workflow_strategy_survey_20261002（本机附件，不随公开源码提供） | 41 | 1,175,352 B · 1.18 MB | 0 | 0 | 41 |
| lhb_backtest/research/local_delivery_20261006（本机附件，不随公开源码提供） | 787 | 8,536,435 B · 8.54 MB | 0 | 787 | 0 |
| lhb_backtest/research/menu2_update_20260924（本机附件，不随公开源码提供） | 181 | 9,638,833 B · 9.64 MB | 0 | 79 | 102 |
| lhb_backtest/research/ml_baseline_build_20261003（本机附件，不随公开源码提供） | 623 | 24,820,437 B · 24.82 MB | 0 | 575 | 48 |
| lhb_backtest/research/ml_beginner_20261003（本机附件，不随公开源码提供） | 328 | 12,974,155 B · 12.97 MB | 0 | 291 | 37 |
| lhb_backtest/research/ml_beginner_catalog_20261003.json（本机附件，不随公开源码提供） | 1 | 13,518 B · 0.01 MB | 0 | 0 | 1 |
| lhb_backtest/research/ml_beginner_tasks_20261003.json（本机附件，不随公开源码提供） | 1 | 44,895 B · 0.04 MB | 0 | 0 | 1 |
| lhb_backtest/research/ml_catalog_read_20261003.json（本机附件，不随公开源码提供） | 1 | 13,518 B · 0.01 MB | 0 | 0 | 1 |
| lhb_backtest/research/ml_completion_20261003（本机附件，不随公开源码提供） | 702 | 29,313,458 B · 29.31 MB | 0 | 585 | 117 |
| lhb_backtest/research/ml_completion_catalog_20261003.json（本机附件，不随公开源码提供） | 1 | 13,518 B · 0.01 MB | 0 | 0 | 1 |
| lhb_backtest/research/ml_completion_tasks_20261003.json（本机附件，不随公开源码提供） | 1 | 47,698 B · 0.05 MB | 0 | 0 | 1 |
| lhb_backtest/research/ml_cycle_summary_20261005（本机附件，不随公开源码提供） | 30 | 3,216,946 B · 3.22 MB | 0 | 1 | 29 |
| lhb_backtest/research/ml_explained_20261003（本机附件，不随公开源码提供） | 633 | 26,761,725 B · 26.76 MB | 0 | 580 | 53 |
| lhb_backtest/research/ml_loop_20261004（本机附件，不随公开源码提供） | 351 | 3,773,446 B · 3.77 MB | 0 | 2 | 349 |
| lhb_backtest/research/ml_loop_budget_fix_20261004（本机附件，不随公开源码提供） | 151 | 1,684,246 B · 1.68 MB | 0 | 20 | 131 |
| lhb_backtest/research/ml_loop_campaigns_20261004.json（本机附件，不随公开源码提供） | 1 | 33,679 B · 0.03 MB | 0 | 0 | 1 |
| lhb_backtest/research/ml_loop_catalog_20261004.json（本机附件，不随公开源码提供） | 1 | 14,189 B · 0.01 MB | 0 | 0 | 1 |
| lhb_backtest/research/ml_loop_context_20261004.json（本机附件，不随公开源码提供） | 1 | 70,548 B · 0.07 MB | 0 | 0 | 1 |
| lhb_backtest/research/ml_loop_permission_fix_20261004（本机附件，不随公开源码提供） | 19 | 70,637 B · 0.07 MB | 0 | 4 | 15 |
| lhb_backtest/research/ml_loop_resume_20261004（本机附件，不随公开源码提供） | 20 | 268,465 B · 0.27 MB | 0 | 2 | 18 |
| lhb_backtest/research/ml_loop_tasks_20261004.json（本机附件，不随公开源码提供） | 1 | 56,704 B · 0.06 MB | 0 | 0 | 1 |
| lhb_backtest/research/ml_notebook_review_20261003（本机附件，不随公开源码提供） | 27 | 276,760 B · 0.28 MB | 0 | 0 | 27 |
| lhb_backtest/research/ml_noteworthy_notification_fix_20261004（本机附件，不随公开源码提供） | 26 | 199,243 B · 0.20 MB | 0 | 3 | 23 |
| lhb_backtest/research/ml_noteworthy_stop_20261004（本机附件，不随公开源码提供） | 104 | 740,350 B · 0.74 MB | 0 | 19 | 85 |
| lhb_backtest/research/ml_notice_reference_recovery_20261005（本机附件，不随公开源码提供） | 45 | 555,991 B · 0.56 MB | 0 | 4 | 41 |
| lhb_backtest/research/ml_objective_catalog_20261003.json（本机附件，不随公开源码提供） | 1 | 13,518 B · 0.01 MB | 0 | 0 | 1 |
| lhb_backtest/research/ml_objective_tasks_20261003.json（本机附件，不随公开源码提供） | 1 | 47,698 B · 0.05 MB | 0 | 0 | 1 |
| lhb_backtest/research/ml_performance_cycle_20261005（本机附件，不随公开源码提供） | 11 | 88,006 B · 0.09 MB | 0 | 2 | 9 |
| lhb_backtest/research/ml_strategy_discussion_20261003（本机附件，不随公开源码提供） | 14 | 62,125 B · 0.06 MB | 0 | 0 | 14 |
| lhb_backtest/research/ml_system_blueprint_20261005（本机附件，不随公开源码提供） | 561 | 21,427,246 B · 21.43 MB | 0 | 489 | 72 |
| lhb_backtest/research/ml_tasks_read_20261003.json（本机附件，不随公开源码提供） | 1 | 41,860 B · 0.04 MB | 0 | 0 | 1 |
| lhb_backtest/research/ml_universe_20261003（本机附件，不随公开源码提供） | 680 | 29,176,117 B · 29.18 MB | 0 | 590 | 90 |
| lhb_backtest/research/ml_universe_catalog_20261003.json（本机附件，不随公开源码提供） | 1 | 13,518 B · 0.01 MB | 0 | 0 | 1 |
| lhb_backtest/research/ml_universe_context_20261003.json（本机附件，不随公开源码提供） | 1 | 160,259 B · 0.16 MB | 0 | 0 | 1 |
| lhb_backtest/research/ml_universe_tasks_20261003.json（本机附件，不随公开源码提供） | 1 | 52,057 B · 0.05 MB | 0 | 0 | 1 |
| lhb_backtest/research/project_charter_20261005（本机附件，不随公开源码提供） | 418 | 14,840,857 B · 14.84 MB | 0 | 330 | 88 |
| lhb_backtest/research/project_review_20260925（本机附件，不随公开源码提供） | 36 | 111,151 B · 0.11 MB | 0 | 26 | 10 |
| lhb_backtest/research/regime_allocation_20260927（本机附件，不随公开源码提供） | 12 | 613,987 B · 0.61 MB | 12 | 0 | 0 |
| lhb_backtest/research/research_loop_20260926（本机附件，不随公开源码提供） | 9 | 215,713 B · 0.22 MB | 9 | 0 | 0 |
| lhb_backtest/research/research_reset_20260926（本机附件，不随公开源码提供） | 113 | 23,110,248 B · 23.11 MB | 0 | 105 | 8 |
| lhb_backtest/research/state_recovery_20260928（本机附件，不随公开源码提供） | 39 | 2,317,177 B · 2.32 MB | 0 | 0 | 39 |
| lhb_backtest/research/storage_audit（本机附件，不随公开源码提供） | 26 | 997,365 B · 1.00 MB | 0 | 1 | 25 |
| lhb_backtest/research/structure_audit_20261006（本机附件，不随公开源码提供） | 1 | 6,105 B · 0.01 MB | 0 | 0 | 1 |
| lhb_backtest/research/update_fix_20260924（本机附件，不随公开源码提供） | 31 | 377,515 B · 0.38 MB | 0 | 6 | 25 |
| lhb_backtest/research/update_redesign_20260924（本机附件，不随公开源码提供） | 32 | 794,967 B · 0.79 MB | 0 | 7 | 25 |
| lhb_backtest/research/workbench_delivery_20260926（本机附件，不随公开源码提供） | 1 | 7,120 B · 0.01 MB | 0 | 0 | 1 |
| lhb_backtest/research/worker_platform_20260928（本机附件，不随公开源码提供） | 6 | 43,150 B · 0.04 MB | 6 | 0 | 0 |
| lhb_backtest/research/workspace_rename_20261006（本机附件，不随公开源码提供） | 78 | 2,297,967 B · 2.30 MB | 0 | 78 | 0 |

本次审查目录在快照后新增产物，随后已加本机 `.git/info/exclude`。原快照中该目录仅有最初盘点脚本。
