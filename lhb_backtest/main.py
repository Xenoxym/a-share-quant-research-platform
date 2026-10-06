"""
A-Share 龙虎榜 Event Backtesting System - Main Pipeline Entry Point

Usage:
    python main.py --step all                              # Full pipeline (SimTradeData K-line by default)
    python main.py --step fetch                            # Steps 1-3: Fetch raw data
    python main.py --step fetch --source akshare           # Step 1 via AKShare HTTP (override)
    python main.py --step clean                            # Step 4: Clean data
    python main.py --step features                         # Step 5: Build features + factors
    python main.py --step labels                           # Step 6: Generate labels
    python main.py --step backtest                         # Step 7: Run backtest
    python main.py --step grid_search                      # Step 8: Run grid search
    python main.py --step restore_lhb_summary              # Re-fetch failed LHB summary dates
    python main.py --step restore_lhb_detail               # Re-fetch failed LHB detail pairs
    python main.py --step restore_all                      # Re-fetch all pending failures

K-line sources (--source, affects Step 1 only):
    simtradedata  Read from pre-exported SimTradeData Parquet files (default, fast, no network)
    akshare       Per-stock HTTP fetch via AKShare (~25 min for full market)
    baostock      Per-stock fetch via BaoStock (unadjusted prices, volume in shares)
"""

from __future__ import annotations

import argparse
import os
from functools import wraps
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import setup_logging, get_logger


def load_config(config_path: str | None = None) -> dict:
    config_path = Path(config_path) if config_path else PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_end_date(config: dict, explicit: str | None = None) -> str:
    value = explicit or config.get('defaults', {}).get('end_date', 'latest')
    if value != 'latest':
        return value
    import json
    from datetime import date
    root = Path(config['simtradedata']['export_dir'])
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    value = str(manifest['version'])
    date.fromisoformat(value)
    return value


def _using_metadata_config(function):
    """Keep metadata/calendar resolution consistent with the supplied config."""
    @wraps(function)
    def wrapped(config, *args, **kwargs):
        from src.utils.calendar import invalidate_cache
        key = "LHB_SIMTRADEDATA_EXPORT_DIR"
        previous = os.environ.get(key)
        configured = config.get("simtradedata", {}).get("export_dir")
        if configured:
            os.environ[key] = str((PROJECT_ROOT / configured).resolve())
            invalidate_cache()
        try:
            return function(config, *args, **kwargs)
        finally:
            if configured:
                if previous is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = previous
                invalidate_cache()
    return wrapped


def _require_acquisition_success(frame, stage):
    report = frame.attrs.get("update_report", {})
    if report.get("status") != "pass":
        raise RuntimeError(f"{stage} did not pass acquisition validation: {report}")


@_using_metadata_config
def step_fetch_data(config: dict, start_date: str, end_date: str, kline_source: str | None = None):
    """Steps 1-3: Fetch raw daily kline, LHB summary, and LHB broker detail.

    Parameters
    ----------
    kline_source : str or None
        CLI override for K-line source (``--source`` flag).
        Priority: CLI arg → ``config.data_source.kline_source`` → ``"simtradedata"``.
        LHB steps always use ``config.data_source.lhb_source`` (default ``"akshare"``).
    """
    logger = get_logger("fetch")

    from src.ingestion.fetch_lhb_summary import fetch_lhb_summary
    from src.ingestion.fetch_lhb_detail import fetch_lhb_detail

    ds = config.get("data_source", {})
    # K-line source resolution: CLI flag > config.kline_source > fallback "simtradedata"
    resolved_kline = kline_source or ds.get("kline_source", "simtradedata")
    # LHB source: always from config (EastMoney data only via AKShare/Tushare)
    resolved_lhb = ds.get("lhb_source", "akshare")
    raw_dir = PROJECT_ROOT / config["paths"]["raw_dir"]

    logger.info("=" * 60)
    logger.info(f"Step 1: Fetching daily K-line data (source={resolved_kline})...")
    logger.info("=" * 60)

    if resolved_kline == "simtradedata":
        from src.ingestion.load_kline_simtradedata import load_kline_simtradedata
        quotes = load_kline_simtradedata(
            start_date=start_date,
            end_date=end_date,
            incremental=True,
            export_dir=os.environ.get("LHB_SIMTRADEDATA_EXPORT_DIR"),
            include_valuation=config.get("simtradedata", {}).get("include_valuation", True),
            save_path=str(raw_dir / "daily_kline.parquet"),
        )
    else:
        from src.ingestion.fetch_daily_kline import fetch_daily_kline
        quotes = fetch_daily_kline(
            start_date=start_date,
            end_date=end_date,
            source=resolved_kline,
            save_path=str(raw_dir / "daily_kline.parquet"),
        )
    _require_acquisition_success(quotes, "K-line")

    logger.info("=" * 60)
    logger.info(f"Step 2: Fetching LHB summary data (source={resolved_lhb})...")
    logger.info("=" * 60)
    summary = fetch_lhb_summary(
        start_date=start_date,
        end_date=end_date,
        source=resolved_lhb,
        save_path=str(raw_dir / "lhb_summary.parquet"),
    )
    _require_acquisition_success(summary, "LHB summary")

    logger.info("=" * 60)
    logger.info(f"Step 3: Fetching LHB broker detail data (source={resolved_lhb})...")
    logger.info("=" * 60)
    detail = fetch_lhb_detail(
        start_date=start_date,
        end_date=end_date,
        source=resolved_lhb,
        save_path=str(raw_dir / "lhb_broker_detail.parquet"),
        summary_path=str(raw_dir / "lhb_summary.parquet"),
    )
    _require_acquisition_success(detail, "LHB detail")


@_using_metadata_config
def step_clean_data(config: dict):
    """Step 4: Clean raw data."""
    logger = get_logger("clean")

    from src.cleaning.clean_kline import clean_daily_kline
    from src.cleaning.clean_lhb import clean_lhb_summary, clean_lhb_broker_detail

    raw_dir = PROJECT_ROOT / config["paths"]["raw_dir"]
    clean_dir = PROJECT_ROOT / config["paths"]["clean_dir"]

    logger.info("=" * 60)
    logger.info("Step 4a: Cleaning daily K-line data...")
    logger.info("=" * 60)
    clean_daily_kline(
        raw_path=str(raw_dir / "daily_kline.parquet"),
        save_path=str(clean_dir / "daily_kline.parquet"),
    )

    logger.info("Step 4b: Cleaning LHB summary data...")
    clean_lhb_summary(
        raw_path=str(raw_dir / "lhb_summary.parquet"),
        save_path=str(clean_dir / "lhb_summary.parquet"),
    )

    logger.info("Step 4c: Cleaning LHB broker detail data...")
    clean_lhb_broker_detail(
        raw_path=str(raw_dir / "lhb_broker_detail.parquet"),
        save_path=str(clean_dir / "lhb_broker_detail.parquet"),
    )


@_using_metadata_config
def step_build_features(config: dict):
    """Step 5: Build event feature table and compute factors."""
    logger = get_logger("features")

    from src.features.build_lhb_features import build_lhb_event_table
    from src.features.compute_lhb_factors import compute_lhb_factors

    clean_dir = PROJECT_ROOT / config["paths"]["clean_dir"]
    factor_dir = PROJECT_ROOT / config["paths"]["factor_dir"]

    logger.info("=" * 60)
    logger.info("Step 5a: Building LHB event table...")
    logger.info("=" * 60)
    build_lhb_event_table(
        clean_lhb_summary_path=str(clean_dir / "lhb_summary.parquet"),
        clean_lhb_broker_path=str(clean_dir / "lhb_broker_detail.parquet"),
        clean_kline_path=str(clean_dir / "daily_kline.parquet"),
        save_path=str(factor_dir / "lhb_event_daily.parquet"),
    )

    logger.info("Step 5b: Computing LHB factors...")
    compute_lhb_factors(
        event_path=str(factor_dir / "lhb_event_daily.parquet"),
        save_path=str(factor_dir / "lhb_event_factors.parquet"),
    )


@_using_metadata_config
def step_generate_labels(config: dict):
    """Step 6: Generate future labels."""
    logger = get_logger("labels")

    from src.labels.generate_future_labels import generate_future_labels

    clean_dir = PROJECT_ROOT / config["paths"]["clean_dir"]
    factor_dir = PROJECT_ROOT / config["paths"]["factor_dir"]

    horizons = config.get("analysis", {}).get("default_horizons", [1, 2, 3, 5, 10])

    logger.info("=" * 60)
    logger.info(f"Step 6: Generating future labels (horizons={horizons})...")
    logger.info("=" * 60)
    generate_future_labels(
        event_path=str(factor_dir / "lhb_event_factors.parquet"),
        kline_path=str(clean_dir / "daily_kline.parquet"),
        horizons=horizons,
        save_path=str(factor_dir / "lhb_event_labeled.parquet"),
    )


def step_run_backtest(config: dict, start_date=None, end_date=None):
    """Step 7: Run example backtest."""
    logger = get_logger("backtest")

    from src.backtest.filter_engine import run_lhb_backtest

    factor_dir = PROJECT_ROOT / config["paths"]["factor_dir"]
    output_dir = PROJECT_ROOT / config["paths"]["output_dir"]

    analysis = config.get("analysis", {})
    default_filters = analysis.get("default_filters", {})
    horizons = analysis.get("default_horizons", [1, 2, 3, 5, 10])
    round_trip_cost = analysis.get("round_trip_cost", 0.0)

    logger.info("=" * 60)
    logger.info(f"Step 7: Running backtest with filters: {default_filters}")
    logger.info("=" * 60)

    result = run_lhb_backtest(
        start_date=start_date, end_date=end_date,
        filters=default_filters,
        horizons=horizons,
        event_path=str(factor_dir / "lhb_event_labeled.parquet"),
        round_trip_cost=round_trip_cost,
    )

    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(result)

    result.to_csv(str(output_dir / "backtest_result.csv"))
    logger.info(f"Results saved to {output_dir / 'backtest_result.csv'}")


@_using_metadata_config
def step_restore_lhb_summary(config: dict):
    """Restore Step: re-fetch failed LHB summary dates from failures_lhb_summary.csv."""
    logger = get_logger("restore")
    from src.ingestion.fetch_lhb_summary import restore_lhb_summary

    ds = config.get("data_source", {})
    lhb_source = ds.get("lhb_source", "akshare")

    logger.info("=" * 60)
    logger.info(f"Restore: retrying failed LHB summary dates (source={lhb_source})...")
    logger.info("=" * 60)
    restore_lhb_summary(source=lhb_source,
                        save_path=str(PROJECT_ROOT / config["paths"]["raw_dir"] / "lhb_summary.parquet"))


@_using_metadata_config
def step_restore_lhb_detail(config: dict):
    """Restore Step: re-fetch failed LHB detail pairs from failures_lhb_detail.csv."""
    logger = get_logger("restore")
    from src.ingestion.fetch_lhb_detail import restore_lhb_detail

    ds = config.get("data_source", {})
    lhb_source = ds.get("lhb_source", "akshare")

    logger.info("=" * 60)
    logger.info(f"Restore: retrying failed LHB detail pairs (source={lhb_source})...")
    logger.info("=" * 60)
    restore_lhb_detail(source=lhb_source,
                       save_path=str(PROJECT_ROOT / config["paths"]["raw_dir"] / "lhb_broker_detail.parquet"))


def step_restore_all(config: dict):
    """Restore Step: re-fetch all pending failures (summary + detail)."""
    step_restore_lhb_summary(config)
    step_restore_lhb_detail(config)


def step_run_grid_search(config: dict, start_date=None, end_date=None):
    """Step 8: Run grid search over parameter combinations."""
    logger = get_logger("grid_search")

    from src.backtest.grid_search import grid_search

    factor_dir = PROJECT_ROOT / config["paths"]["factor_dir"]
    output_dir = PROJECT_ROOT / config["paths"]["output_dir"]

    analysis = config.get("analysis", {})
    default_filters = analysis.get("default_filters", {})
    oos_split_date = analysis.get("oos_split_date")
    round_trip_cost = analysis.get("round_trip_cost", 0.0)
    min_sample_count = analysis.get("min_sample_count", 30)

    # Thresholds follow the unit contract: all ratios are decimal fractions.
    # Ranges chosen around observed distributions (lhb_turnover_ratio median
    # ≈ 0.24, net_buy_ratio median ≈ 0.003, buy1_concentration median ≈ 0.33).
    param_grid = {
        "min_net_buy_ratio": [0.0, 0.01, 0.03, 0.05, 0.08],
        "min_lhb_turnover_ratio": [0.10, 0.20, 0.30, 0.40],
    }

    # Seat factors need broker-detail coverage; the detail backfill only
    # covers recent months until the long historical fetch completes.
    # Include the seat-concentration axis only when coverage is meaningful.
    try:
        import pandas as _pd
        _events = _pd.read_parquet(
            factor_dir / "lhb_event_labeled.parquet", columns=["buy1_concentration"],
        )
        seat_coverage = _events["buy1_concentration"].notna().mean()
    except Exception:
        seat_coverage = 0.0
    if seat_coverage >= 0.30:
        param_grid["min_buy1_concentration"] = [0.2, 0.3, 0.4, 0.5]
    else:
        logger.warning(
            f"buy1_concentration coverage {seat_coverage:.1%} < 30% — seat factor "
            "axis skipped (rerun grid_search after broker-detail backfill completes)"
        )

    # Rank on training labels only; OOS remains diagnostic.
    sort_by = "net_oo_return_1d"

    logger.info("=" * 60)
    logger.info(
        f"Step 8: Grid search over {len(param_grid)} parameters "
        f"(oos_split={oos_split_date}, cost={round_trip_cost}, sort_by={sort_by})..."
    )
    logger.info("=" * 60)

    results = grid_search(
        start_date=start_date, end_date=end_date,
        horizons=analysis.get("default_horizons", [1, 2, 3, 5, 10]),
        param_grid=param_grid,
        event_path=str(factor_dir / "lhb_event_labeled.parquet"),
        base_filters=default_filters,
        sort_by=sort_by,
        save_path=str(output_dir / "grid_search_result.csv"),
        plot_dir=str(output_dir / "grid_search_plots"),
        oos_split_date=oos_split_date,
        round_trip_cost=round_trip_cost,
        min_sample_count=min_sample_count,
    )

    from src.backtest.grid_search import format_grid_search_summary

    print(f"\nGrid search complete: {len(results)} combinations tested.")
    print(f"Results CSV : {output_dir / 'grid_search_result.csv'}")
    print(f"Plots folder: {output_dir / 'grid_search_plots'}")
    print("\n" + format_grid_search_summary(
        results,
        param_names=list(param_grid.keys()),
        sort_by=sort_by,
        top_n=10,
    ))


def main():
    parser = argparse.ArgumentParser(description="A-Share LHB Event Backtesting System")
    parser.add_argument(
        "--step",
        choices=[
            "all", "fetch", "clean", "features", "labels", "backtest", "grid_search",
            "restore_lhb_summary", "restore_lhb_detail", "restore_all",
        ],
        default="all",
        help="Which pipeline step to run",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--source",
        choices=["akshare", "baostock", "simtradedata"],
        default=None,
        help=(
            "K-line data source override. "
            "'simtradedata' reads from pre-exported Parquet files (fast, no network). "
            "Defaults to config.yaml data_source.primary."
        ),
    )

    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(
        level=config.get("logging", {}).get("level", "INFO"),
        log_file=config.get("logging", {}).get("log_file"),
    )

    logger = get_logger("main")

    start_date = args.start_date or config["defaults"]["start_date"]
    end_date = resolve_end_date(config, args.end_date)
    kline_source = args.source  # None → falls back to config inside step_fetch_data

    ds = config.get("data_source", {})
    effective_kline = kline_source or ds.get("kline_source", "simtradedata")
    effective_lhb = ds.get("lhb_source", "akshare")
    logger.info(
        f"LHB Backtest Pipeline - step={args.step}, "
        f"kline={effective_kline}, lhb={effective_lhb}, "
        f"range={start_date} to {end_date}"
    )

    t0 = time.time()

    step = args.step

    if step in ("all", "fetch"):
        step_fetch_data(config, start_date, end_date, kline_source=kline_source)

    if step in ("all", "clean"):
        step_clean_data(config)

    if step in ("all", "features"):
        step_build_features(config)

    if step in ("all", "labels"):
        step_generate_labels(config)

    if step in ("all", "backtest"):
        step_run_backtest(config, start_date=start_date, end_date=end_date)

    if step == "grid_search":
        step_run_grid_search(config, start_date=start_date, end_date=end_date)

    if step == "restore_lhb_summary":
        step_restore_lhb_summary(config)

    if step == "restore_lhb_detail":
        step_restore_lhb_detail(config)

    if step == "restore_all":
        step_restore_all(config)

    elapsed = time.time() - t0
    logger.info(f"Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
