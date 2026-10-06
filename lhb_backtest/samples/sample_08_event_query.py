# -*- coding: utf-8 -*-
"""示例 08：用 SQL（DuckDB）随手查询事件数据。

标签宽表本质是一个 10 万行 × 102 列的"事件数据库"，
query_parquet() 让你直接对 parquet 写 SQL（表名固定为 data）。
适合日常研究中的临时问题：某只股票的历史、某天的榜单、条件筛选……
"""

import _bootstrap  # noqa: F401

import pandas as pd

from src.utils.io import query_parquet
from src.utils.logging import setup_logging

LABELED = "data/factor/lhb_event_labeled.parquet"


def main() -> None:
    setup_logging(level="WARNING")
    pd.set_option("display.width", 160)

    # ── 查询 1：某只股票的全部上榜历史 ──────────────────────
    stock = "002131.SZ"
    q1 = query_parquet(f"""
        SELECT trade_date, stock_name, pct_chg, lhb_reason,
               entry_open_exit_open_1d, next_day_limit_up
        FROM data
        WHERE stock_code = '{stock}'
        ORDER BY trade_date DESC
        LIMIT 10
    """, LABELED)
    print(f"====== {stock} 最近 10 次上榜 ======")
    print(q1.to_string(index=False))

    # ── 查询 2：某一天的龙虎榜全景 ──────────────────────────
    date = "2026-04-30"
    q2 = query_parquet(f"""
        SELECT stock_code, stock_name, pct_chg,
               lhb_net_buy / 1e8 AS 净买入_亿, next_day_limit_up
        FROM data
        WHERE trade_date = '{date}' AND event_day_limit_up = 1
        ORDER BY 净买入_亿 DESC
        LIMIT 10
    """, LABELED)
    print(f"\n====== {date} 涨停上榜、净买入前 10 ======")
    print(q2.to_string(index=False))

    # ── 查询 3：条件统计 —— 大额净买 + 高榜占比事件的连板率（按年）──
    q3 = query_parquet("""
        SELECT substr(trade_date, 1, 4) AS 年份,
               count(*)                              AS 样本数,
               avg(next_day_limit_up)                AS 次日连板率,
               avg(entry_open_exit_open_1d)             AS T1开盘1日收益
        FROM data
        WHERE event_day_limit_up = 1
          AND net_buy_ratio > 0.05
          AND lhb_turnover_ratio > 0.2
          AND next_day_untradable = 0
        GROUP BY 1 ORDER BY 1
    """, LABELED)
    print("\n====== 强资金流涨停事件的逐年表现 ======")
    print(q3.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n提示：把 SQL 换成你自己的问题即可，表名固定为 data。")


if __name__ == "__main__":
    main()
