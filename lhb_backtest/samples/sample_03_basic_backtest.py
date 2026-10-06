# -*- coding: utf-8 -*-
"""示例 03：基础回测 —— 最常用的入口。

演示 run_lhb_backtest() 的标准用法：
- 选择日期区间
- 配置事件筛选条件（过滤器）
- 设置往返交易成本
- 读取返回的 summary（汇总指标）和 detail（逐事件明细）

前置：data/factor/lhb_event_labeled.parquet 已生成（见 sample_02）。
"""

import _bootstrap  # noqa: F401

from src.utils.logging import setup_logging
from src.backtest.filter_engine import run_lhb_backtest

# ── 可修改的参数 ─────────────────────────────────────────────
START_DATE = "2022-01-01"      # None 表示不限
END_DATE = "2026-05-26"

FILTERS = {
    # 事件类型
    "include_only_limit_up_event": True,   # 上榜日涨停（精确 high_limit 判定）
    "include_only_first_board": True,      # 只要首板（前一日未涨停）
    "exclude_consecutive_lhb_day": True,   # 连续上榜只取第一天
    "only_single_day_board": True,         # 剔除三日榜（因子量纲一致）
    # 可交易性
    "exclude_ST": True,                    # 历史 ST 精确剔除
    "exclude_untradable_next_day": True,   # 剔除 T+1 一字板/停牌/无K线
    "exclude_lhb_reason": "无价格涨跌幅限制",  # 剔除新股等无涨跌幅事件
    # 数值阈值（全部为小数：0.05 = 5%）
    "min_net_buy_ratio": 0.01,             # 净买入占成交额 ≥ 1%
    "min_lhb_turnover_ratio": 0.10,        # 龙虎榜成交占比 ≥ 10%
    # "min_buy1_concentration": 0.4,       # 买一席位集中度（需席位明细覆盖）
}

ROUND_TRIP_COST = 0.002        # 往返成本 0.2%（佣金+印花税+滑点）
# ────────────────────────────────────────────────────────────


def main() -> None:
    setup_logging(level="INFO")

    result = run_lhb_backtest(
        start_date=START_DATE,
        end_date=END_DATE,
        filters=FILTERS,
        round_trip_cost=ROUND_TRIP_COST,
    )

    s = result.summary.iloc[0]
    print("\n" + "=" * 56)
    print(f"样本数: {int(s['sample_count']):,}   区间: {START_DATE} ~ {END_DATE}")
    print("=" * 56)
    print("--- 研究口径（上榜日收盘入场，不可实际成交）---")
    print(f"  1日均收益 {s['avg_return_1d']:+.4f}   胜率 {s['win_rate_1d']:.1%}")
    print(f"  5日均收益 {s['avg_return_5d']:+.4f}   胜率 {s['win_rate_5d']:.1%}")
    print(f"  次日连板率 {s['next_day_limit_up_rate']:.1%}")
    print("--- 持有期合规的价格标签（尚非模拟成交收益）---")
    print(f"  1日均收益 {s['avg_oo_return_1d']:+.4f}   费后 {s['net_oo_return_1d']:+.4f}")
    print(f"  5日均收益 {s['avg_entry_return_5d']:+.4f}   费后 {s['net_entry_return_5d']:+.4f}")
    print(f"  平均隔夜跳空 {s['avg_entry_open_gap']:+.4f}")
    if "net_oo_alpha_1d" in s.index:
        print(f"--- 超额（vs 基准指数）---")
        print(f"  1日alpha {s['net_oo_alpha_1d']:+.4f}   5日alpha {s.get('avg_entry_alpha_5d', float('nan')):+.4f}")

    # 逐事件明细可继续自行分析
    detail = result.detail
    print(f"\n明细 DataFrame: {detail.shape[0]:,} 行 × {detail.shape[1]} 列")
    print("收益最好的 5 个事件（T+1开盘口径）:")
    top = detail.nlargest(5, "entry_open_exit_open_1d")[
        ["trade_date", "stock_code", "stock_name", "entry_open_exit_open_1d", "lhb_reason"]
    ]
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
