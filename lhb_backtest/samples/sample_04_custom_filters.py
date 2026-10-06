# -*- coding: utf-8 -*-
"""示例 04：多套筛选条件横向对比。

一次性比较几套不同的过滤方案，快速看出哪类事件值得深入。
所有方案共用一份标签数据（只读一次盘），核心 API：
    apply_filters(df, filters)              # 纯过滤
    compute_backtest_statistics(df, ...)    # 纯统计
"""

import _bootstrap  # noqa: F401

import pandas as pd

from src.utils.io import load_parquet
from src.utils.logging import setup_logging
from src.backtest.filter_engine import apply_filters
from src.backtest.statistics import compute_backtest_statistics

# 公共底线：可交易、非ST、单日榜、剔除新股无涨跌幅事件
BASE = {
    "exclude_ST": True,
    "exclude_untradable_next_day": True,
    "only_single_day_board": True,
    "exclude_consecutive_lhb_day": True,
    "exclude_lhb_reason": "无价格涨跌幅限制",
}

# 待对比的方案（在 BASE 之上叠加）
SCHEMES = {
    "全部龙虎榜": {},
    "涨停上榜": {"include_only_limit_up_event": True},
    "首板涨停": {"include_only_limit_up_event": True, "include_only_first_board": True},
    "首板+高榜占比": {
        "include_only_limit_up_event": True,
        "include_only_first_board": True,
        "min_lhb_turnover_ratio": 0.30,
    },
    "首板+强净买": {
        "include_only_limit_up_event": True,
        "include_only_first_board": True,
        "min_net_buy_ratio": 0.05,
    },
}

METRICS = [
    ("sample_count", "样本数", "{:,.0f}"),
    ("avg_return_1d", "1日收益(收盘)", "{:+.4f}"),
    ("avg_oo_return_1d", "T+1开盘至T+2开盘价格收益", "{:+.4f}"),
    ("net_oo_return_1d", "1日费后", "{:+.4f}"),
    ("next_day_limit_up_rate", "次日连板率", "{:.1%}"),
    ("entry_win_rate_1d", "可交易胜率", "{:.1%}"),
]


def main() -> None:
    setup_logging(level="WARNING")   # 安静模式，只看表格

    df = load_parquet("data/factor/lhb_event_labeled.parquet")
    base_df = apply_filters(df, BASE)
    print(f"总事件 {len(df):,} → 公共底线过滤后 {len(base_df):,}\n")

    rows = []
    for name, extra in SCHEMES.items():
        sub = apply_filters(base_df, extra)
        stats = compute_backtest_statistics(sub, horizons=[1, 5], round_trip_cost=0.002)
        row = {"方案": name}
        for col, label, fmt in METRICS:
            v = stats[col].iloc[0] if col in stats.columns else float("nan")
            row[label] = fmt.format(v) if pd.notna(v) else "-"
        rows.append(row)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print("\n提示：注意「收盘口径」与「T+1开盘口径」的差距 —— 那就是买不到的隔夜跳空。")


if __name__ == "__main__":
    main()
