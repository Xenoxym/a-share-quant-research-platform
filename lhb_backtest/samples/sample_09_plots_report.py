# -*- coding: utf-8 -*-
"""示例 09：图表与 HTML 报告。

跑一次回测，然后：
1. 生成单张研究图（收益分布 / 因子散点 / 累计收益 / 月度统计）；
2. 生成一份自包含的 HTML 报告（图表内嵌，可直接发给别人看）。
"""

import _bootstrap  # noqa: F401

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 无窗口环境也能出图

from src.utils.logging import setup_logging
from src.backtest.filter_engine import run_lhb_backtest
from src.reports.plot_results import (
    plot_return_distribution,
    plot_factor_vs_return,
    plot_cumulative_returns,
    plot_monthly_stats,
)
from src.reports.make_report import generate_html_report

OUT_DIR = Path("data/output/sample09_plots")


def main() -> None:
    setup_logging(level="INFO")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    result = run_lhb_backtest(
        filters={
            "exclude_ST": True,
            "include_only_limit_up_event": True,
            "include_only_first_board": True,
            "exclude_consecutive_lhb_day": True,
            "only_single_day_board": True,
            "exclude_untradable_next_day": True,
            "exclude_lhb_reason": "无价格涨跌幅限制",
        },
        round_trip_cost=0.002,
    )
    detail = result.detail
    print(f"回测样本 {len(detail):,} 个事件，开始出图 …")

    # 1) T+1 开盘口径的收益分布
    fig = plot_return_distribution(detail, return_col="entry_open_exit_open_1d",
                                   title="T+1 Open Entry 1D Return")
    fig.savefig(OUT_DIR / "01_return_dist.png", dpi=120, bbox_inches="tight")

    # 2) 因子 vs 收益散点
    fig = plot_factor_vs_return(detail, factor_col="net_buy_ratio",
                                return_col="entry_open_exit_open_1d")
    fig.savefig(OUT_DIR / "02_factor_scatter.png", dpi=120, bbox_inches="tight")

    # 3) 等权累计收益曲线（按事件日均值串联）
    fig = plot_cumulative_returns(detail, return_col="entry_open_exit_open_1d")
    fig.savefig(OUT_DIR / "03_cumulative.png", dpi=120, bbox_inches="tight")

    # 4) 月度样本数与关键指标
    fig = plot_monthly_stats(detail)
    fig.savefig(OUT_DIR / "04_monthly.png", dpi=120, bbox_inches="tight")

    print(f"4 张图已保存到 {OUT_DIR.resolve()}")

    # 5) HTML 报告（自动嵌入图表）
    html_path = generate_html_report(result, title="首板龙虎榜回测报告")
    print(f"HTML 报告: {Path(html_path).resolve()}")
    print("双击 HTML 文件即可在浏览器查看。")


if __name__ == "__main__":
    main()
