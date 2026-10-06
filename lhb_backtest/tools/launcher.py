# -*- coding: utf-8 -*-
"""龙虎榜回测系统 —— 交互式启动器（面向不写代码的使用者）。

双击项目根目录的《启动器.bat》即可运行；也支持命令行直达：
    python tools/launcher.py            # 显示菜单
    python tools/launcher.py backtest   # 直接进入一键回测
    python tools/launcher.py update     # 直接增量更新数据
    python tools/launcher.py detail     # 席位明细断点续传
    python tools/launcher.py grid       # 网格搜索
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# 保持控制台原生编码，无法编码的字符替换为 '?'，避免崩溃
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

import datetime as _datetime

DATA_END_DEFAULT = _datetime.datetime.now(
    _datetime.timezone(_datetime.timedelta(hours=8))
).date().isoformat()
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

MENU = """
╔══════════════════════════════════════════════════╗
║        龙虎榜事件回测系统 —— 快捷启动器          ║
╠══════════════════════════════════════════════════╣
║  1. 数据体检（先看看数据是否就绪）               ║
║  2. 更新数据（行情、辅助数据及龙虎榜汇总/明细） ║
║  3. 单独续传席位明细（按日期分段复用）           ║
║  4. 重建特征与标签（每次更新数据后必做）         ║
║  5. 一键回测（问答式配置 → 出结果）              ║
║  6. 网格搜索（自动参数寻优 + 样本外验证）        ║
║  7. 生成 HTML 图表报告（可发给别人看）           ║
║  8. 打开结果文件夹                               ║
║  9. 运行全部单元测试（检查系统是否正常）         ║
║ 10. 打开技术研究（策略定义、费用及逐日决策）     ║
║  0. 退出                                         ║
╚══════════════════════════════════════════════════╝"""


# ── 输入辅助 ─────────────────────────────────────────────────

def ask(msg: str, default: str = "") -> str:
    tip = f"（直接回车 = {default}）" if default else ""
    val = input(f"  {msg}{tip}: ").strip()
    return val if val else default


def ask_yn(msg: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    val = input(f"  {msg} [{d}]: ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes", "1", "是")


def ask_float(msg: str, default: float) -> float:
    val = input(f"  {msg}（直接回车 = {default}）: ").strip()
    try:
        return float(val) if val else default
    except ValueError:
        print("    输入无效，使用默认值", default)
        return default


def pause() -> None:
    input("\n按回车返回菜单 …")


# ── 功能 1：数据体检 ─────────────────────────────────────────

def action_health() -> None:
    import datetime as _dt

    import pandas as pd

    print("\n—— 数据体检 ——\n")
    today = _dt.date.today().isoformat()
    max_dates: dict[str, str] = {}

    def _check(path: str, name: str, key: str) -> None:
        p = PROJECT_ROOT / path
        if not p.exists():
            print(f"  [缺失] {name}: {path} 不存在")
            return
        try:
            df = pd.read_parquet(p, columns=["trade_date"])
            dmax = df["trade_date"].astype(str).max()
            max_dates[key] = dmax
            print(f"  [OK] {name}: {len(df):,} 行，"
                  f"{df['trade_date'].astype(str).min()} ~ {dmax}")
        except Exception as exc:
            print(f"  [异常] {name}: {exc}")

    _check("data/raw/daily_kline.parquet", "日K线(raw)", "kline")
    _check("data/raw/lhb_summary.parquet", "龙虎榜汇总(raw)", "summary")
    _check("data/raw/lhb_broker_detail.parquet", "席位明细(raw)", "detail")
    summary_path = PROJECT_ROOT / 'data/raw/lhb_summary.parquet'
    detail_path = PROJECT_ROOT / 'data/raw/lhb_broker_detail.parquet'
    if summary_path.exists() and detail_path.exists():
        keys=['stock_code','trade_date']
        summary_keys=pd.read_parquet(summary_path,columns=keys).drop_duplicates()
        detail_keys=pd.read_parquet(detail_path,columns=keys).drop_duplicates()
        absent=~pd.MultiIndex.from_frame(summary_keys).isin(pd.MultiIndex.from_frame(detail_keys))
        print(f"  [{'缺失' if absent.any() else '覆盖通过'}] 席位事件覆盖: {len(summary_keys)-absent.sum():,}/{len(summary_keys):,}，缺 {absent.sum():,} 个事件")
        print("       单纯文件可读、日期最新不代表数据完整；完整验收记录见 data/update_reports/menu2_latest.json。")

    # 每日股票覆盖数检查：行数多不代表数据全（部分导出的快照会造成
    # 某些日期只有零星股票，回测在这些日期上会静默失真）
    kp = PROJECT_ROOT / "data/raw/daily_kline.parquet"
    if kp.exists():
        k = pd.read_parquet(kp, columns=["stock_code", "trade_date"])
        per_day = k.groupby(k["trade_date"].astype(str))["stock_code"].nunique()
        partial = per_day[per_day < per_day.median() * 0.5]
        if len(partial):
            print(f"  [警告] K线有 {len(partial)} 个交易日覆盖不足"
                  f"（{partial.index.min()} ~ {partial.index.max()}，"
                  f"最少仅 {partial.min()} 只，正常约 {per_day.median():.0f} 只）")
            print("         → SimTradeData 快照不完整，需先更新它再跑第 2 项（会自动重补这些日期）")

    # 基准指数覆盖（决定 alpha 标签的可用区间）
    try:
        from src.data_sources.simtradedata_metadata import load_benchmark
        bench = load_benchmark()
        if bench is not None and not bench.empty:
            bmax = str(bench["trade_date"].max())
            print(f"  [OK] 基准指数: 到 {bmax}（此日期之后的事件无 alpha 标签）")
    except Exception:
        pass

    # 新鲜度与跨源一致性（"能跑通"不等于"是新的"）
    print(f"\n  今天是 {today}：")
    kline_max = max_dates.get("kline", "")
    if kline_max:
        lag = (_dt.date.fromisoformat(today) - _dt.date.fromisoformat(kline_max)).days
        if lag > 7:
            print(f"  [注意] K线落后今天 {lag} 天（SimTradeData 本地快照只到 "
                  f"{kline_max}，需先更新 SimTradeData 仓库再跑第 2 项）")
        else:
            print(f"  [OK] K线新鲜度: 落后 {lag} 天")
    summary_max = max_dates.get("summary", "")
    if summary_max and kline_max and summary_max > kline_max:
        print(f"  [注意] 龙虎榜汇总({summary_max}) 比 K线({kline_max}) 新 —— "
              "这段事件缺行情，无法算收益标签，属正常过渡态")

    labeled = PROJECT_ROOT / "data/factor/lhb_event_labeled.parquet"
    if labeled.exists():
        df = pd.read_parquet(labeled)
        lab_max = df["trade_date"].astype(str).max()
        print(f"\n  [OK] 标签宽表: {len(df):,} 事件 × {len(df.columns)} 列（到 {lab_max}）")
        if summary_max and lab_max < summary_max:
            print(f"  [注意] 标签宽表落后于原始数据（{lab_max} < {summary_max}）"
                  "→ 跑第 4 项重建")
        print(f"       席位因子覆盖率: {df['buy1_concentration'].notna().mean():.1%}"
              "（低说明明细还没抓全，跑第 3 项可补）")
        if "alpha_1d" in df.columns:
            print(f"       超额收益覆盖率: {df['alpha_1d'].notna().mean():.1%}")
        print(f"       T+1 不可买占比: {df['next_day_untradable'].mean():.1%}")
    else:
        print("\n  [缺失] 标签宽表未生成 → 请先跑第 2 项再跑第 4 项")

    for fname, label in (
        ("failures_lhb_summary.csv", "汇总抓取失败记录"),
        ("failures_lhb_detail.csv", "明细抓取失败记录"),
    ):
        p = PROJECT_ROOT / "data" / "failures" / fname
        if p.exists():
            try:
                fails = pd.read_csv(p)
                pending = (fails["status"] == "failed").sum() if "status" in fails.columns else len(fails)
                if pending:
                    print(f"  [注意] {label}: {pending} 条待重试")
            except Exception:
                pass


# ── 功能 2/3：数据更新 ───────────────────────────────────────

def action_update() -> bool:
    from src.utils.logging import setup_logging
    from src.ingestion.load_kline_simtradedata import load_kline_simtradedata
    from src.ingestion.fetch_lhb_summary import fetch_lhb_summary
    from src.ingestion.update_simtradedata import update_simtradedata
    from src.ingestion.update_contract import write_report

    print("\n—— 数据更新（SimTradeData 下载、导出与质量验收）——")
    print("  按北京时间确定日期：18:00 前取上一交易日，休市日自动回退。")
    start = ask("起始日期 YYYY-MM-DD", "2019-01-01")
    end = ask("结束日期 YYYY-MM-DD", DATA_END_DEFAULT)
    if start > end:
        raise ValueError("起始日期不能晚于结束日期")
    menu_report_path = PROJECT_ROOT / "data/update_reports/menu2_latest.json"
    write_report(menu_report_path, {"status": "running", "phase": "source", "requested_date": end})

    setup_logging(level="INFO")
    print("\n[1/4] 更新 SimTradeData（下载器 → DuckDB → 校验后发布 Parquet）…")
    print("  优先复用已验收 Parquet；有缺失日期时调用 SimTradeData 日频接口。")
    print("  BaoStock 同次请求补行情/估值/状态；Mootdx 更新除权/财务。")
    print("  日常更新不读取行情压缩包；未变化的 Parquet 直接复用。")
    upstream_ok = False
    try:
        updated = update_simtradedata(end)
        end = updated.target_date
        upstream_ok = True
        print(f"  [OK] {'直接复用已有数据' if getattr(updated, 'reused', False) else '更新并发布数据'}，覆盖至 {end}: {updated.export_dir}")
        print(f"  下载数据库: {updated.database_path}")
    except Exception as exc:
        write_report(menu_report_path, {"status": "fail", "phase": "source",
                                       "requested_date": end, "error": str(exc)})
        print(f"  [失败] SimTradeData 上游更新未完成: {exc}")
        print("         旧快照已保留。详情见 data/update_reports/latest.json；本次更新终止。")
        return False

    return _finish_project_update(start, end, updated, menu_report_path)


def _finish_project_update(start, end, updated, menu_report_path):
    """Complete the same project-import steps after a verified source publication."""
    from src.ingestion.load_kline_simtradedata import load_kline_simtradedata
    from src.ingestion.fetch_lhb_summary import fetch_lhb_summary
    from src.ingestion.fetch_lhb_detail import fetch_lhb_detail
    from src.ingestion.update_contract import write_report
    upstream_ok = True
    write_report(menu_report_path, {"status": "running", "phase": "project_import",
                                   "target_date": end, "export_dir": str(updated.export_dir)})
    try:
        print("\n[2/4] 读取项目日K；有新记录时才合并 …")
        print(f"  项目数据目录: {PROJECT_ROOT / 'data/raw'}")
        kline = load_kline_simtradedata(start, end, incremental=True, refresh_existing=upstream_ok and not getattr(updated, "reused", False))
        report = kline.attrs.get("update_report", {})
        unresolved = report.get("unresolved_dates", [])
        print(
            f"  净新增 {report.get('rows_added', 0):,} 行；"
            f"仍缺失/不完整 {len(unresolved)} 个交易日。"
        )
        if unresolved:
            print(f"  [未完成] {unresolved[0]} ~ {unresolved[-1]} 仍不可用于可靠回测。")

        print("\n[3/4] 龙虎榜汇总（AKShare，需要网络）…")
        summary = fetch_lhb_summary(start_date=start, end_date=end, incremental=True)
        summary_report = summary.attrs.get("update_report", {})
        complete = upstream_ok and not unresolved and summary_report.get("status") == "pass"
        detail_report = {"status": "not_run"}
        if complete:
            print("\n[4/4] 龙虎榜席位明细（仅抓取未完成日期分段，核对汇总事件覆盖）…")
            detail = fetch_lhb_detail(start_date=start, end_date=end, incremental=True)
            detail_report = detail.attrs.get("update_report", {})
            complete = detail_report.get("status") == "pass"
        write_report(menu_report_path, {
            "status": "pass" if complete else "fail", "target_date": end,
            "export_dir": str(updated.export_dir), "kline": report, "lhb_summary": summary_report,
            "lhb_detail": detail_report,
        })
        if complete:
            print("\n行情、汇总和席位明细更新及覆盖校验通过。请接着跑第 4 项重建特征与标签。")
        else:
            print("\n更新未完整完成；日K或龙虎榜仍有缺失。详情见 data/update_reports/menu2_latest.json。")

        return complete
    except Exception as exc:
        write_report(menu_report_path, {"status": "fail", "phase": "project_import",
                                       "target_date": end, "export_dir": str(updated.export_dir),
                                       "source_published": True, "error": str(exc)})
        raise


def action_detail() -> None:
    from src.utils.logging import setup_logging
    from src.ingestion.fetch_lhb_detail import fetch_lhb_detail

    print("\n—— 席位明细续传 ——")
    print("  完整榜单按日期分段分页抓取；已有分段直接复用，首次历史核对需要较长时间。")
    print("  每完成一个分段保存；中断后从未完成分段继续，覆盖验收通过后发布明细。")
    if not ask_yn("确认开始？（若已有别的窗口在抓，请选 N 避免重复）", True):
        return
    start = ask("起始日期 YYYY-MM-DD", "2019-01-01")
    end = ask("结束日期 YYYY-MM-DD", DATA_END_DEFAULT)
    setup_logging(level="WARNING")   # 只显示进度条，减少刷屏
    fetch_lhb_detail(start_date=start, end_date=end, incremental=True)
    print("\n明细抓取结束（或已抓完）。请跑第 4 项重建特征与标签。")


# ── 功能 4：重建特征与标签 ───────────────────────────────────

def action_rebuild() -> None:
    from src.utils.logging import setup_logging
    from src.cleaning.clean_kline import clean_daily_kline
    from src.cleaning.clean_lhb import clean_lhb_summary, clean_lhb_broker_detail
    from src.features.build_lhb_features import build_lhb_event_table
    from src.features.compute_lhb_factors import compute_lhb_factors
    from src.labels.generate_future_labels import generate_future_labels

    setup_logging(level="INFO")
    print("\n[1/4] 清洗K线 …")
    clean_daily_kline()
    print("\n[2/4] 清洗龙虎榜 …")
    clean_lhb_summary()
    clean_lhb_broker_detail()
    print("\n[3/4] 特征与因子 …")
    build_lhb_event_table()
    compute_lhb_factors()
    print("\n[4/4] 标签 …")
    generate_future_labels(horizons=[1, 2, 3, 5, 10])
    from src.data_sources.project_quality import inspect_project_data
    quality=inspect_project_data(PROJECT_ROOT)
    if quality['status']!='pass':
        raise RuntimeError('重建后的数据验收未通过：'+'；'.join(quality['errors']))
    print("\n重建及数据契约验收通过，可以跑第 5 项回测了。")


# ── 功能 5：一键回测 ─────────────────────────────────────────

PRESETS = {
    "1": ("严格首板（推荐）", {
        "exclude_ST": True, "include_only_limit_up_event": True,
        "include_only_first_board": True, "exclude_consecutive_lhb_day": True,
        "only_single_day_board": True, "exclude_untradable_next_day": True,
        "exclude_lhb_reason": "无价格涨跌幅限制",
    }),
    "2": ("全部涨停上榜", {
        "exclude_ST": True, "include_only_limit_up_event": True,
        "exclude_consecutive_lhb_day": True,
        "only_single_day_board": True, "exclude_untradable_next_day": True,
        "exclude_lhb_reason": "无价格涨跌幅限制",
    }),
    "3": ("全部龙虎榜事件", {
        "exclude_ST": True, "only_single_day_board": True,
        "exclude_untradable_next_day": True,
    }),
}


def action_backtest() -> None:
    from src.utils.logging import setup_logging
    from src.backtest.filter_engine import run_lhb_backtest

    print("\n—— 一键回测：请回答几个问题（不确定就直接回车用默认值）——\n")

    start = ask("起始日期 YYYY-MM-DD", "2019-01-01")
    end = ask("结束日期 YYYY-MM-DD", DATA_END_DEFAULT)
    cost = ask_float("往返交易成本（0.002 = 0.2%，含佣金印花税滑点）", 0.002)

    print("\n  事件筛选方案：")
    for key, (name, _) in PRESETS.items():
        print(f"    {key} = {name}")
    choice = ask("选择方案 1/2/3", "1")
    name, filters = PRESETS.get(choice, PRESETS["1"])
    filters = dict(filters)

    if ask_yn("要加数值阈值吗？（净买占比/榜占比等）", False):
        v = ask_float("最低净买入占比（小数，0.01 = 1%；0 = 不限制）", 0.0)
        if v > 0:
            filters["min_net_buy_ratio"] = v
        v = ask_float("最低龙虎榜成交占比（小数，0.2 = 20%；0 = 不限制）", 0.0)
        if v > 0:
            filters["min_lhb_turnover_ratio"] = v
        v = ask_float("最低买一席位集中度（小数；0 = 不限制，需明细数据）", 0.0)
        if v > 0:
            filters["min_buy1_concentration"] = v

    print(f"\n  开始回测：方案「{name}」，{start} ~ {end}，成本 {cost:.2%} …\n")
    setup_logging(level="WARNING")
    result = run_lhb_backtest(
        start_date=start, end_date=end, filters=filters, round_trip_cost=cost,
    )

    s = result.summary.iloc[0]
    n = int(s["sample_count"])
    print("=" * 54)
    print(f"  样本数: {n:,}")
    if n == 0:
        print("  没有符合条件的事件——请放宽筛选或检查数据（菜单第 1 项）")
        return
    print("-" * 54)
    print("  【研究口径】上榜日收盘入场（实际买不到，仅参考）")
    print(f"    1日均收益 {s['avg_return_1d']:+.2%}    胜率 {s['win_rate_1d']:.1%}")
    print(f"    5日均收益 {s['avg_return_5d']:+.2%}    次日连板率 {s['next_day_limit_up_rate']:.1%}")
    print("  【持有期合规的价格标签】尚非组合成交收益")
    print(f"    1日均收益 {s.get('avg_oo_return_1d', float('nan')):+.2%}    费后 {s.get('net_oo_return_1d', float('nan')):+.2%}")
    print(f"    5日均收益 {s['avg_entry_return_5d']:+.2%}    费后 {s['net_entry_return_5d']:+.2%}")
    if "net_oo_alpha_1d" in s.index and s.notna().get("net_oo_alpha_1d", False):
        print(f"  【超额】1日 vs 基准指数 {s['net_oo_alpha_1d']:+.2%}")
    print("=" * 54)
    print(f"  明细 CSV: {OUTPUT_DIR / 'backtest_result_detail.csv'}")

    if ask_yn("生成 HTML 图表报告并打开？", True):
        import matplotlib
        matplotlib.use("Agg")
        from src.reports.make_report import generate_html_report
        html = generate_html_report(result, title=f"龙虎榜回测：{name}")
        print(f"  报告: {html}")
        try:
            os.startfile(str(PROJECT_ROOT / html) if not Path(html).is_absolute() else html)
        except Exception:
            pass


# ── 功能 6：网格搜索 ─────────────────────────────────────────

def action_grid() -> None:
    from src.utils.logging import setup_logging
    from src.backtest.grid_search import grid_search, format_grid_search_summary

    print("\n—— 网格搜索：自动尝试多组阈值，按【训练期】表现排名，样本外仅检查 ——\n")
    split = ask("样本外起始日期（该日期后的数据只做验证）", "2024-07-01")
    cost = ask_float("往返交易成本", 0.002)

    param_grid = {
        "min_net_buy_ratio": [0.0, 0.01, 0.03, 0.05],
        "min_lhb_turnover_ratio": [0.10, 0.20, 0.30],
    }
    base_filters = dict(PRESETS["1"][1])

    setup_logging(level="WARNING")
    results = grid_search(
        param_grid=param_grid,
        base_filters=base_filters,
        sort_by="net_oo_return_1d",
        oos_split_date=split,
        round_trip_cost=cost,
        min_sample_count=100,
        save_path=str(OUTPUT_DIR / "grid_search_result.csv"),
        plot_dir=str(OUTPUT_DIR / "grid_search_plots"),
    )
    print("\n" + format_grid_search_summary(
        results, param_names=list(param_grid.keys()),
        sort_by="net_oo_return_1d", top_n=8,
    ))
    print("  排名依据 = 训练期隔夜持有标签费后均值（net_oo_return_1d），谨防过拟合。")
    print(f"  完整结果: {OUTPUT_DIR / 'grid_search_result.csv'}")


# ── 功能 7/8/9 ───────────────────────────────────────────────

def action_report() -> None:
    from src.utils.logging import setup_logging
    import matplotlib
    matplotlib.use("Agg")
    from src.backtest.filter_engine import run_lhb_backtest
    from src.reports.make_report import generate_html_report

    setup_logging(level="WARNING")
    print("\n生成报告中（默认严格首板方案）…")
    result = run_lhb_backtest(filters=dict(PRESETS["1"][1]), round_trip_cost=0.002)
    html = generate_html_report(result, title="龙虎榜回测报告")
    print(f"  报告: {html}")
    try:
        os.startfile(str(PROJECT_ROOT / html) if not Path(html).is_absolute() else html)
    except Exception:
        pass


def action_open() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.startfile(str(OUTPUT_DIR))


def action_tests() -> None:
    print("\n运行单元测试（约 10 秒）…\n")
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=str(PROJECT_ROOT))


def action_workbench() -> None:
    from src.technical.server import serve
    serve(PROJECT_ROOT)


# ── 主循环 ───────────────────────────────────────────────────

ACTIONS = {
    "1": action_health, "2": action_update, "3": action_detail,
    "4": action_rebuild, "5": action_backtest, "6": action_grid,
    "7": action_report, "8": action_open, "9": action_tests, "10": action_workbench,
}

ALIASES = {
    "health": "1", "update": "2", "detail": "3", "rebuild": "4",
    "backtest": "5", "grid": "6", "report": "7", "open": "8", "test": "9",
    "workbench": "10",
}


def main() -> None:
    # 命令行直达：python tools/launcher.py backtest
    if len(sys.argv) > 1:
        key = ALIASES.get(sys.argv[1].lower())
        if key and key in ACTIONS:
            outcome = ACTIONS[key]()
            if key == "2" and outcome is False:
                raise SystemExit(1)
            if sys.stdin.isatty():
                input("\n执行完毕，按回车关闭窗口 …")
            return

    while True:
        print(MENU)
        choice = input("请输入数字并回车: ").strip()
        if choice == "0":
            break
        fn = ACTIONS.get(choice)
        if fn is None:
            print("  无效选项，请输入 0-10。")
            continue
        try:
            fn()
        except KeyboardInterrupt:
            print("\n  已中断，返回菜单。")
        except Exception as exc:
            print(f"\n  [出错了] {exc}")
            print("  常见原因：数据未准备好（先跑 2 和 4），或网络不可用。")
        pause()


if __name__ == "__main__":
    main()
