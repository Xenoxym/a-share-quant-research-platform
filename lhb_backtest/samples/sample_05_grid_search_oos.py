# -*- coding: utf-8 -*-
"""示例 05：网格搜索 + 样本内/样本外（IS/OOS）验证。

在一批阈值组合上批量回测，并用 2024-07-01 之后的数据做样本外验证，
防止选出"只在历史上好看"的过拟合参数。

输出列说明：
- 无前缀列  = 样本内（IS，< oos_split_date）
- ``oos_``  = 样本外（OOS，≥ oos_split_date）——仅验证，不参与排名！
"""

import _bootstrap  # noqa: F401

from src.utils.logging import setup_logging
from src.backtest.grid_search import grid_search, format_grid_search_summary

# 网格：每个 key 是 apply_filters 支持的过滤器，value 是候选阈值列表
# （全部小数量纲；组合数 = 各列表长度之积，注意控制规模）
PARAM_GRID = {
    "min_net_buy_ratio": [0.0, 0.01, 0.03, 0.05],
    "min_lhb_turnover_ratio": [0.10, 0.20, 0.30],
    # "min_buy1_concentration": [0.3, 0.4, 0.5],   # 席位明细覆盖足够后再启用
}

BASE_FILTERS = {
    "exclude_ST": True,
    "include_only_limit_up_event": True,
    "include_only_first_board": True,
    "exclude_consecutive_lhb_day": True,
    "only_single_day_board": True,
    "exclude_untradable_next_day": True,
    "exclude_lhb_reason": "无价格涨跌幅限制",
}

OOS_SPLIT = "2024-07-01"      # 该日期后的数据不参与"挑参数"，只用来验证
SORT_BY = "net_oo_return_1d"   # 按训练期隔夜标签费后均值排名


def main() -> None:
    setup_logging(level="INFO")

    results = grid_search(
        param_grid=PARAM_GRID,
        base_filters=BASE_FILTERS,
        sort_by=SORT_BY,
        oos_split_date=OOS_SPLIT,
        round_trip_cost=0.002,
        min_sample_count=100,          # 样本内少于100个事件的组合不参与排名
        save_path="data/output/sample05_grid_result.csv",
        plot_dir="data/output/sample05_grid_plots",
    )

    print("\n" + format_grid_search_summary(
        results, param_names=list(PARAM_GRID.keys()), sort_by=SORT_BY, top_n=8,
    ))

    print("判读要点：")
    print(" 1. IS 和 OOS 的指标方向应一致，否则是过拟合信号；")
    print(" 2. oos_sample_count 太小的组合不可信；")
    print(" 3. 结果 CSV: data/output/sample05_grid_result.csv")


if __name__ == "__main__":
    main()
