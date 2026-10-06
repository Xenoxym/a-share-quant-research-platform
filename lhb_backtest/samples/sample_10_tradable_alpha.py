# -*- coding: utf-8 -*-
"""示例 10：四种收益口径对比 —— 理解"回测赚钱"和"实盘赚钱"的差距。

同一批事件，分别用四个口径统计 1 日收益：
  A. 收盘研究口径  future_return_1d        （不可成交，含前视偏差）
  B. T+1开盘口径   entry_open_exit_open_1d    （持有期合规，成交另验）
  C. 费后口径      B - 往返成本
  D. 超额口径      entry_oo_alpha_1d = B - 基准指数同窗口收益

结论通常是 A >> B > C，差距主要来自隔夜跳空溢价。
"""

import _bootstrap  # noqa: F401

import pandas as pd

from src.utils.io import load_parquet
from src.utils.logging import setup_logging
from src.backtest.filter_engine import apply_filters

FILTERS = {
    "exclude_ST": True,
    "include_only_limit_up_event": True,
    "include_only_first_board": True,
    "exclude_consecutive_lhb_day": True,
    "only_single_day_board": True,
    "exclude_untradable_next_day": True,
    "exclude_lhb_reason": "无价格涨跌幅限制",
}
COST = 0.002


def describe(series: pd.Series, name: str) -> dict:
    s = series.dropna()
    return {
        "口径": name,
        "样本": f"{len(s):,}",
        "均值": f"{s.mean():+.4f}",
        "中位数": f"{s.median():+.4f}",
        "胜率": f"{(s > 0).mean():.1%}",
        "标准差": f"{s.std():.4f}",
    }


def main() -> None:
    setup_logging(level="WARNING")

    df = load_parquet("data/factor/lhb_event_labeled.parquet")
    df = apply_filters(df, FILTERS)
    print(f"过滤后样本 {len(df):,} 个事件（首板+非ST+可交易）\n")

    rows = [
        describe(df["future_return_1d"], "A 收盘研究口径（不可成交）"),
        describe(df["entry_open_exit_open_1d"], "B T+1开至T+2开价格标签（成交另验）"),
        describe(df["entry_open_exit_open_1d"] - COST, f"C 费后（B - {COST:.1%}）"),
    ]
    if "entry_oo_alpha_1d" in df.columns:
        rows.append(describe(df["entry_oo_alpha_1d"], "D 同入场/退出窗口的价格超额"))

    print(pd.DataFrame(rows).to_string(index=False))

    gap = df["entry_open_gap"].dropna()
    print(f"\n隔夜跳空（T+1开盘 vs 上榜日收盘）: 均值 {gap.mean():+.4f}，"
          f"即 A 与 B 差距的主要来源。")
    print("判读：只有 B/C 是真的能赚到的钱；若 C ≤ 0，说明该筛选条件没有可交易 alpha。")


if __name__ == "__main__":
    main()
