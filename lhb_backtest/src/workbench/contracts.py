"""Versioned strategy card and deliberately bounded execution assumptions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import date
import math


@dataclass(frozen=True)
class StrategyCard:
    name: str = "机构披露净买额 · 五日研究"
    start_date: str = "2019-01-01"
    end_date: str = "latest"
    initial_cash: float = 1_000_000
    holding_days: int = 5
    max_positions: int = 10
    position_fraction: float = 0.10
    min_inst_ratio: float = 0.0
    min_avg_amount: float = 20_000_000
    max_participation: float = 0.001
    commission_rate: float = 0.0003
    minimum_commission: float = 5.0
    other_fee_rate: float = 0.00002
    slippage_bps: float = 10.0
    dividend_tax_reserve: float = 0.20
    match_caliper: float = 0.60
    bootstrap_samples: int = 500
    seed: int = 20260926

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 100:
            raise ValueError("策略名称需为 1–100 个字符")
        start = date.fromisoformat(self.start_date)
        if start < date(2019, 1, 1):
            raise ValueError("第一版研究起点不得早于 2019-01-01")
        if self.end_date != "latest" and date.fromisoformat(self.end_date) < start:
            raise ValueError("结束日期不得早于起始日期")
        integer_bounds = {"holding_days": (1, 60), "max_positions": (1, 100),
                          "bootstrap_samples": (100, 5000), "seed": (0, 2**32 - 1)}
        for key, (low, high) in integer_bounds.items():
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key} 必须为 {low}–{high} 的整数")
        bounds = {"initial_cash": (1000, 1e10), "position_fraction": (0.001, 1),
                  "min_inst_ratio": (0, 1), "min_avg_amount": (0, 1e12),
                  "max_participation": (0.000001, .05), "commission_rate": (0, .01),
                  "minimum_commission": (0, 1000), "other_fee_rate": (0, .01),
                  "slippage_bps": (0, 500), "dividend_tax_reserve": (0, 1),
                  "match_caliper": (0, 3)}
        for key, (low, high) in bounds.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} 必须为 {low}–{high} 的有限数值")

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("策略卡片必须是对象")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"未知策略参数: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self):
        return asdict(self)


FEATURE_REGISTRY = [
    {"name": "inst_disclosed_net", "label": "机构披露净买额", "unit": "CNY",
     "formula": "sum(buy_amount on institutional buy seats) - sum(sell_amount on institutional sell seats)",
     "known_at": "next_session_09:00_assumed", "source": "clean/lhb_broker_detail.parquet",
     "boundary": "普通单日榜；买卖两榜各自求和；并非完整机构资金流；匿名席位不识别实体"},
    {"name": "inst_ratio", "label": "机构披露净买额 / 当日成交额", "unit": "fraction",
     "formula": "inst_disclosed_net / quote.amount", "known_at": "next_session_09:00_assumed",
     "source": "broker + daily_kline", "boundary": "不得混用三日榜和单日成交额"},
    {"name": "float_value", "label": "流通市值", "unit": "CNY",
     "formula": "valuation.float_value on exact event date", "known_at": "event_close",
     "source": "SimTradeData/valuation", "boundary": "当前发布的历史快照；未恢复供应商历次修订版本"},
    {"name": "momentum_20", "label": "20 日累计价格变动", "unit": "fraction",
     "formula": "product(close / pre_close over 20 consecutive market sessions) - 1",
     "known_at": "event_close", "source": "clean/daily_kline.parquet",
     "boundary": "逐日昨收口径；不是含分红总回报；不足 20 个连续市场交易日保持未知"},
    {"name": "avg_amount_20", "label": "20 日平均成交额", "unit": "CNY/day",
     "formula": "mean(amount over 20 consecutive market sessions)", "known_at": "event_close",
     "source": "clean/daily_kline.parquet", "boundary": "包含事件日，不含下单日"},
    {"name": "avg_volume_20", "label": "20 日平均成交股数", "unit": "shares/day",
     "formula": "mean(volume over 20 consecutive market sessions)", "known_at": "event_close",
     "source": "clean/daily_kline.parquet", "boundary": "用于事前限制订单规模"},
]

ASSUMPTIONS = [
    "范围：2019 年起沪深主板普通单日龙虎榜；历史非 ST；普通已验证限价；买入 100 股整数手。",
    "披露缺少逐条发布时间，假设上一交易日完整榜单在下一交易日 09:00 前可知；不声称精确恢复历史发布时间。",
    "订单股数以信号日价格和已知流动性在盘前确定；按下一交易日开盘价模拟，滑点作为额外现金成本。",
    "开盘触及任一价格限制不成交；当日成交量仅作为事后成交上限，不代表开盘竞价可成交量。",
    "持仓转为 ST 时仅退出：按历史状态和已核对规则生成模型限价（2026-07-06 前 5%，之后 10%），需 OHLC 全部符合；逐笔标记，不改写原始隔离限价。",
    "买单当日有效；未成交部分取消；卖单到期后逐日重试，T+1；同股持仓期间新信号忽略。",
    "退出日为入场日之后 holding_days 个市场交易日的开盘；期末不强制虚构平仓。",
    "机构组按披露净额占比降序排队；基础与对照组按固定种子的事件哈希排队。盘前按已有现金与持仓预留，不预支同日卖出款；资金比例是新增订单预算。",
    "现金分红按除息日计提税后应收权益，未有到账日期故不再投入；税率为可配置准备金假设。",
    "送转股按除权日到账并取整是显式模拟假设；配股不认购；结果保留公司行动明细与质量标记。",
    "佣金含最低收费，其他费用按配置模型计提；卖出印花税按 2023-08-28 分界；费用并非券商实际账单。",
    "停牌沿用上次估值并标记；未知缺价、未解释除权、退市未结算分别标记，不删除亏损持仓。",
    "历史数据已被研究过：年度分段与区间估计是历史稳健性描述，不是未见测试集或盈利证明。",
]
