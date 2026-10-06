# -*- coding: utf-8 -*-
"""示例 01：增量更新数据。

三个数据源分别增量拉取（已存在的日期自动跳过）：
1. 日K线   —— 从本地 SimTradeData Parquet 导出读取（无网络请求，快）
2. 龙虎榜汇总 —— AKShare/EastMoney（一次区间请求）
3. 席位明细  —— 东方财富完整披露归档，支持断点续传；已验收内容直接复用。

用法：
    ..\\.venv\\Scripts\\python.exe samples\\sample_01_update_data.py
"""

import _bootstrap  # noqa: F401  (sys.path 引导)

from src.utils.logging import setup_logging
from src.ingestion.load_kline_simtradedata import load_kline_simtradedata
from src.ingestion.fetch_lhb_summary import fetch_lhb_summary
from src.ingestion.fetch_lhb_detail import fetch_lhb_detail

# ── 参数：想更新的日期区间（增量模式下只会补缺失的交易日）──
START_DATE = "2019-01-01"
END_DATE = None             # None follows the accepted export manifest

# 席位明细是否也更新；首次获取耗时取决于来源和网络
FETCH_BROKER_DETAIL = True


def main() -> None:
    setup_logging(level="INFO")
    from main import load_config, resolve_end_date, _require_acquisition_success
    end_date = resolve_end_date(load_config(), END_DATE)

    print(f"\n[1/3] 更新日K线（SimTradeData 本地导出）{START_DATE} ~ {end_date}")
    kline = load_kline_simtradedata(START_DATE, end_date, incremental=True)
    _require_acquisition_success(kline, "K-line")
    print(f"      K线现有 {len(kline):,} 行")

    print(f"\n[2/3] 更新龙虎榜汇总（AKShare）")
    summary = fetch_lhb_summary(start_date=START_DATE, end_date=end_date, incremental=True)
    _require_acquisition_success(summary, "LHB summary")
    print(f"      汇总现有 {len(summary):,} 条事件")

    if FETCH_BROKER_DETAIL:
        print(f"\n[3/3] 更新席位明细（支持中断后续传）")
        detail = fetch_lhb_detail(start_date=START_DATE, end_date=end_date, incremental=True)
        _require_acquisition_success(detail, "LHB detail")
        print(f"      明细现有 {len(detail):,} 行")
    else:
        print("\n[3/3] 已跳过席位明细（FETCH_BROKER_DETAIL = False）")

    print("\n完成。下一步：运行 sample_02_full_pipeline.py 重建特征与标签。")


if __name__ == "__main__":
    main()
