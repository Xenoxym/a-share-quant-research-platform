# -*- coding: utf-8 -*-
"""示例 06：因子分层（分位数）分析。

把事件按某个因子的大小切成 N 层，看各层未来收益是否单调递增/递减——
这是判断"因子有没有预测力"的最直观方法，比单一阈值筛选更稳健。
核心 API：compute_quantile_statistics()
"""

import _bootstrap  # noqa: F401

import pandas as pd

from src.utils.io import load_parquet
from src.utils.logging import setup_logging
from src.backtest.filter_engine import apply_filters
from src.backtest.statistics import compute_quantile_statistics

# 想检验的因子（都是小数量纲）
FACTORS = [
    "net_buy_ratio",             # 净买入 / 当日成交额
    "lhb_turnover_ratio",        # 龙虎榜成交 / 当日成交额
    "net_buy_float_mcap_ratio",  # 净买入 / 流通市值
    "entry_open_gap",            # 隔夜跳空（本身也是一个"情绪"因子）
    # "buy1_concentration",      # 席位内买一集中度（需明细覆盖）
]

BASE_FILTERS = {
    "exclude_ST": True,
    "include_only_limit_up_event": True,
    "only_single_day_board": True,
    "exclude_untradable_next_day": True,
    "exclude_consecutive_lhb_day": True,
    "exclude_lhb_reason": "无价格涨跌幅限制",
}

N_QUANTILES = 5
TARGET = "avg_oo_return_1d"    # 隔夜价格标签，尚未模拟退出成交


def main() -> None:
    setup_logging(level="WARNING")

    df = load_parquet("data/factor/lhb_event_labeled.parquet")
    df = apply_filters(df, BASE_FILTERS)
    print(f"过滤后样本 {len(df):,} 个事件\n")

    for factor in FACTORS:
        if factor not in df.columns:
            print(f"[跳过] {factor}: 列不存在")
            continue
        q = compute_quantile_statistics(df, factor_col=factor, n_quantiles=N_QUANTILES, horizons=[1, 5])
        if q.empty:
            print(f"[跳过] {factor}: 有效样本不足")
            continue

        view = q[["quantile", "sample_count", "factor_min", "factor_max", TARGET, "next_day_limit_up_rate"]].copy()
        view.columns = ["分位", "样本数", "因子下界", "因子上界", "T+1开盘至T+2开盘价格收益", "次日连板率"]
        print(f"====== 因子: {factor} （Q1最小 → Q{N_QUANTILES}最大）======")
        print(view.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

        # 简单单调性判断：最高层 vs 最低层
        spread = q[TARGET].iloc[-1] - q[TARGET].iloc[0]
        print(f"  多空价差(Q{N_QUANTILES} - Q1) = {spread:+.4f}"
              f"  {'← 值得关注' if abs(spread) > 0.005 else '（微弱）'}\n")


if __name__ == "__main__":
    main()
