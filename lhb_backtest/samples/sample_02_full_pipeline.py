# -*- coding: utf-8 -*-
"""示例 02：一键跑完整管道（清洗 → 特征 → 标签 → 回测）。

等价于命令行：
    python main.py --step clean
    python main.py --step features
    python main.py --step labels
    python main.py --step backtest

数据更新（fetch）见 sample_01；网格搜索见 sample_05。
本脚本演示如何在代码里逐步调用各阶段函数，方便插入自己的处理逻辑。
"""

import _bootstrap  # noqa: F401

from src.utils.logging import setup_logging
from src.cleaning.clean_kline import clean_daily_kline
from src.cleaning.clean_lhb import clean_lhb_summary, clean_lhb_broker_detail
from src.features.build_lhb_features import build_lhb_event_table
from src.features.compute_lhb_factors import compute_lhb_factors
from src.labels.generate_future_labels import generate_future_labels
from src.backtest.filter_engine import run_lhb_backtest


def main() -> None:
    setup_logging(level="INFO")

    print("\n[1/5] 清洗K线 …")
    clean_daily_kline()

    print("\n[2/5] 清洗龙虎榜汇总 + 席位明细 …")
    clean_lhb_summary()          # 含百分比→小数的量纲转换
    clean_lhb_broker_detail()

    print("\n[3/5] 构建事件宽表 + 计算因子 …")
    build_lhb_event_table()      # 汇总 × 席位透视 × K线，含历史 is_st
    compute_lhb_factors()        # 买卖比、席位集中度、三日榜窗口分离等

    print("\n[4/5] 生成标签（未来收益 / 连板 / T+1可交易口径 / alpha）…")
    generate_future_labels(horizons=[1, 2, 3, 5, 10])

    print("\n[5/5] 用默认过滤器跑一次回测 …")
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
    print(f"\n样本数: {int(result.summary['sample_count'].iloc[0]):,}")
    for col in ("avg_return_1d", "avg_oo_return_1d", "net_oo_return_1d",
                "next_day_limit_up_rate", "win_rate_1d"):
        if col in result.summary.columns:
            print(f"{col:26s} = {result.summary[col].iloc[0]:+.4f}")

    result.to_csv("data/output/backtest_result.csv")
    print("\n完成。明细已保存到 data/output/backtest_result_detail.csv")


if __name__ == "__main__":
    main()
