"""Executable, validated definitions. Defaults are hypotheses, not fitted parameters."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import date
import math


@dataclass(frozen=True)
class Strategy:
    name: str = "价格动量 MOM(252,21) · 月度排名"
    direction: str = "momentum"
    lookback: int = 252
    skip: int = 21
    trend_window: int = 0
    min_score: float | None = None
    top_k: int = 20
    rebalance: str = "monthly"
    allocation: float = .98
    min_avg_amount: float = 20_000_000.
    family: str = "price"
    min_history: int = 0
    size_floor_quantile: float = 0.
    positive_profit: bool = False
    seed: int = 11
    require_fundamentals: bool = False
    share_basis: str = "reported"
    allocation_policy: str = "small"
    state_breadth_threshold: float = .4


@dataclass(frozen=True)
class Execution:
    initial_cash: float = 1_000_000.
    commission_rate: float = .0003
    minimum_commission: float = 5.
    other_fee_rate: float = .00002
    slippage_bps: float = 10.
    max_participation: float = .001
    dividend_tax_reserve: float = .20


@dataclass(frozen=True)
class Experiment:
    start_date: str = "2020-01-01"
    end_date: str = "latest"


@dataclass(frozen=True)
class ResearchSpec:
    strategy: Strategy = Strategy()
    execution: Execution = Execution()
    experiment: Experiment = Experiment()

    def __post_init__(self):
        s, e, x = self.strategy, self.execution, self.experiment
        if not isinstance(s.name, str) or not s.name.strip() or len(s.name) > 100:
            raise ValueError("名称应为 1–100 个字符")
        if s.direction not in {"momentum", "reversal"} or s.rebalance not in {"monthly", "weekly"}:
            raise ValueError("不支持的排序方向或调仓频率")
        if s.family not in {"price", "size", "large_size", "earnings_yield", "book_to_price", "profitability", "low_volatility", "random", "market_trend", "cash", "allocation"}:
            raise ValueError("不支持的基础策略家族")
        if s.allocation_policy not in {"small", "small_half", "defensive", "fixed_mix", "vol_budget", "observable", "hmm2_mix", "hmm2_cash", "hmm3_mix", "hmm2_lag5"}:
            raise ValueError("不支持的配置规则")
        if s.share_basis not in {"reported", "known_bonus"}:
            raise ValueError("股本口径必须为 reported 或 known_bonus")
        if type(s.positive_profit) is not bool or type(s.require_fundamentals) is not bool:
            raise ValueError("财务资格开关必须为布尔值")
        for key, low, high in [("lookback", 2, 756), ("skip", 0, 755), ("trend_window", 0, 756), ("top_k", 1, 100), ("min_history", 0, 756), ("seed", 0, 1000000)]:
            v = getattr(s, key)
            if type(v) is not int or not low <= v <= high:
                raise ValueError(f"{key} 必须为 {low}–{high} 的整数")
        if s.skip >= s.lookback or s.trend_window == 1:
            raise ValueError("skip 必须小于 lookback；均线窗口为 0（禁用）或至少 2")
        def number(v, low, high, key):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high:
                raise ValueError(f"{key} 超出范围或不是有限数")
        number(s.allocation, .01, 1., "allocation")
        number(s.min_avg_amount, 0., 1e12, "min_avg_amount")
        number(s.size_floor_quantile, 0., .9, "size_floor_quantile")
        number(s.state_breadth_threshold, .1, .9, "state_breadth_threshold")
        if s.family == "market_trend" and s.trend_window < 2:
            raise ValueError("市场趋势策略需要至少 2 日均线")
        if s.min_score is not None:
            number(s.min_score, -100., 100., "min_score")
        for key, low, high in [("initial_cash", 1000, 1e10), ("commission_rate", 0, .01),
            ("minimum_commission", 0, 1000), ("other_fee_rate", 0, .01), ("slippage_bps", 0, 500),
            ("max_participation", .000001, .05), ("dividend_tax_reserve", 0, 1)]:
            number(getattr(e, key), low, high, key)
        start = date.fromisoformat(x.start_date)
        if s.family == "allocation" and start < date(2022, 1, 1):
            raise ValueError("状态研究从2022年起评估，之前历史用于训练；不能拿全样本拟合后回填")
        if s.family == "allocation" and (s.min_history != 252 or not s.require_fundamentals or s.share_basis != "known_bonus"
                or s.positive_profit or s.size_floor_quantile or s.trend_window or s.min_score is not None):
            raise ValueError("状态配置的两个分支固定采用252日历史、共同财报资格和送转桥接；不接受隐藏分支过滤")
        if start < date(2019, 1, 1) or (x.end_date != "latest" and date.fromisoformat(x.end_date) < start):
            raise ValueError("研究区间无效；clean 行情起于 2019 年")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or set(payload) - {"strategy", "execution", "experiment"}:
            raise ValueError("只接受 strategy、execution、experiment 三部分")
        result = {}
        for key, kind in [("strategy", Strategy), ("execution", Execution), ("experiment", Experiment)]:
            part = payload.get(key, {})
            if not isinstance(part, dict) or set(part) - {f.name for f in fields(kind)}:
                raise ValueError(f"{key} 含未知字段")
            result[key] = kind(**part)
        return cls(**result)


SCENARIOS = {
    "zero_transaction_cost": "零交易费用与滑点",
    "zero_slippage": "零滑点，保留佣金及交易税费",
    "configured": "配置的交易税费与滑点",
}


def describe(spec):
    s = spec.strategy
    value = {
        "universe": "独立证券主表中的沪深主板股票；逐日检查上市状态、非 ST、有效行情和限价。保留历史退市样本，不依赖龙虎榜。",
        "inputs": "i 表示股票，t 表示市场交易日。C=当日原始收盘价（元/股）；P_ref=当日昨收参考价（元/股）；A=成交额（元）；V=成交量（股）。源文件 daily_kline.parquet。",
        "daily_return": "r[i,t] = ln(C[i,t] / P_ref[i,t])；X[i,t] = exp(Σ(u≤t) r[i,u])。X 是参考价衔接价格序列，不宣称为官方全收益价格。",
        "momentum": f"M[i,t] = exp(Σ(j={s.skip}..{s.lookback - 1}) r[i,t-j]) - 1；窗口 L={s.lookback}, S={s.skip}，含 {s.lookback-s.skip} 个日收益。",
        "score": "score = M，降序" if s.direction == "momentum" else "score = -M，降序（反转假设）",
        "threshold": "不设收益门槛；负分也可入选" if s.min_score is None else f"严格要求 score > {s.min_score}",
        "trend": "不启用均线过滤" if not s.trend_window else f"严格要求 X[t] / SMA_{s.trend_window}(X)[t] - 1 > 0",
        "liquidity": f"ADV20 = mean(A[t-19:t]) ≥ {s.min_avg_amount:g} 元；AV20 = mean(V[t-19:t]) > 0。要求窗口内连续市场交易日和有效价格。",
        "ranking": f"在共同资格及过滤后的股票中，按 score 降序、股票代码升序处理并列，取前 K={s.top_k}。",
        "rebalance": "每月最后一个市场交易日收盘决定目标，下一市场交易日开始执行" if s.rebalance == "monthly" else "每个自然周最后一个市场交易日收盘决定目标，下一市场交易日开始执行",
        "availability": "模型假设当日日线在 15:30 可用于收盘后决策，最早下一市场交易日开盘执行；15:30 不是供应商逐条实测的发布时间。",
        "target": f"入选目标权重 w={s.allocation:g}/{s.top_k}；不足 K 只时剩余资金留现金。目标股数 floor(E[t]×w/C[t]/100)×100。",
        "exit": "调仓时，未入选股票目标为 0；入选股票按新目标股数增减。未完成目标逐日重试，新调仓目标覆盖旧目标。没有固定五日持有期。",
        "execution": "数量目标由前收盘生成；买单只使用开盘前现金，并按前收盘的 110% 预留。当天卖出所得最早下一交易日用于买入。T+1；买整手；不强制期末平仓。",
        "capacity": "买单事前股数不超过前 20 日均量×参与率；成交再受当日总量上限约束。日线模型无法确认开盘队列及实际盘口容量。",
        "cost": "N=成交股数×原始开盘价；F=佣金+其他费+卖出印花税+N×滑点基点/10000。滑点以现金成本记账，不重复改成交价。",
        "cost_parameters": f"成交 N>0 时：佣金=max({spec.execution.minimum_commission:g}, N×{spec.execution.commission_rate:g})；其他费=N×{spec.execution.other_fee_rate:g}；配置滑点=N×{spec.execution.slippage_bps:g}/10000。印花税仅卖出收取：2023-08-28 前 0.001×N，之后 0.0005×N。零交易成本情景将四项全置零。",
        "limitations": "零费用情景仍有涨跌停、停牌、整手、容量及现金约束。分红扣准备金后计应收不复投；送转股按除权日到账是假设；不参与配股。停牌日沿用经公司行动调整的估值，不采纳可能未除权的停牌旧价。市场冲击和机会成本没有独立估计。",
        "parameter_origin": "默认 252/21 来自约十二个月动量跳过约一个月的公开研究惯例，是 A 股日频近似；默认 K=20、98% 配置及费用是未优化的研究设定。修改会产生新实验，不代表论文原样复现。",
        "evaluation": "当前历史已被查看，结果属于回顾性研究；没有自动调参，不标为未见样本外结论。",
        "source": "https://www.aqr.com/Insights/Datasets/Momentum-Indices-Monthly",
    }
    if s.family == "allocation":
        from .regimes import definition
        value.update(definition(spec))
    elif s.family != "price" or s.require_fundamentals:
        from .foundation_signals import definition
        value.update(definition(spec))
    return value
