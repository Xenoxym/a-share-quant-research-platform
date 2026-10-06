# -*- coding: utf-8 -*-
"""示例 07：分组统计与时段分析。

1. 按分类列分组（上榜原因 / 交易所板块 / 涨停与否），比较各组表现；
2. 按季度切片，找出策略表现最好/最差的时间段（风格漂移检测）。
核心 API：compute_grouped_statistics()、compute_rolling_period_statistics()
"""

import _bootstrap  # noqa: F401

import pandas as pd

from src.utils.io import load_parquet
from src.utils.logging import setup_logging
from src.backtest.filter_engine import apply_filters
from src.backtest.statistics import (
    compute_grouped_statistics,
    compute_rolling_period_statistics,
)

BASE_FILTERS = {
    "exclude_ST": True,
    "only_single_day_board": True,
    "exclude_untradable_next_day": True,
    "exclude_consecutive_lhb_day": True,
    "exclude_lhb_reason": "无价格涨跌幅限制",
}

SHOW_COLS = ["sample_count", "avg_oo_return_1d", "next_day_limit_up_rate", "oo_win_rate_1d"]


def main() -> None:
    setup_logging(level="WARNING")

    df = load_parquet("data/factor/lhb_event_labeled.parquet")
    df = apply_filters(df, BASE_FILTERS)
    print(f"过滤后样本 {len(df):,} 个事件\n")

    # ── 1) 按板块分组（用股票代码后缀推断）──────────────────
    df["板块"] = df["stock_code"].str[-2:].map({"SZ": "深市", "SH": "沪市", "BJ": "北交所"})
    g1 = compute_grouped_statistics(df, group_col="板块", horizons=[1])
    print("====== 按交易所板块 ======")
    print(g1[["板块"] + SHOW_COLS].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ── 2) 按"上榜日是否涨停"分组 ──────────────────────────
    df["上榜日涨停"] = df["event_day_limit_up"].map({1.0: "涨停", 0.0: "未涨停"})
    g2 = compute_grouped_statistics(df, group_col="上榜日涨停", horizons=[1])
    print("\n====== 按上榜日是否涨停 ======")
    print(g2[["上榜日涨停"] + SHOW_COLS].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ── 3) 按上榜原因分组（取样本最多的前 6 类）─────────────
    top_reasons = df["lhb_reason"].value_counts().head(6).index
    g3 = compute_grouped_statistics(df[df["lhb_reason"].isin(top_reasons)], "lhb_reason", horizons=[1])
    g3["lhb_reason"] = g3["lhb_reason"].str.slice(0, 24)
    print("\n====== 按上榜原因（Top6）======")
    print(g3[["lhb_reason"] + SHOW_COLS].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ── 4) 最佳 / 最差季度 ──────────────────────────────────
    periods = compute_rolling_period_statistics(
        df, period_length="Q", metric_col="entry_open_exit_open_1d", top_n=3,
    )
    for key, title in (("best_periods", "表现最好的季度"), ("worst_periods", "表现最差的季度")):
        p = periods[key]
        if not p.empty:
            print(f"\n====== {title}（按 T+1 开盘 1 日收益）======")
            print(p[["period", "sample_count", "avg_oo_return_1d", "next_day_limit_up_rate"]]
                  .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    print("\n提示：若各季度表现差异巨大，说明该策略对市场风格敏感，实盘需配合择时。")


if __name__ == "__main__":
    main()
