from dataclasses import asdict, dataclass, fields
from datetime import date
import math
import re


@dataclass(frozen=True)
class MLSpec:
    name: str = "日线价量选股 · 五日相对收益基准"
    train_start: str = "2020-01-01"
    train_end: str = "2022-12-31"
    valid_start: str = "2023-01-01"
    valid_end: str = "2023-12-31"
    test_start: str = "2024-01-01"
    test_end: str = "2026-09-24"
    horizon: int = 5
    frequency: str = "weekly"
    min_history: int = 60
    min_avg_amount: float = 20_000_000.0
    feature_set: str = "price_volume"
    top_k: int = 100
    bins: int = 10
    seed: int = 11
    ridge_alpha: float = 10.0
    tree_iterations: int = 80
    tree_leaves: int = 7
    tree_min_samples: int = 200
    tree_learning_rate: float = 0.05
    tree_l2: float = 10.0
    study: str = "signal"
    source_run_id: str | None = None

    def __post_init__(self):
        if self.study not in {"signal", "complete", "universe", "fixed_blend", "financial_context"}:
            raise ValueError("不支持的ML研究类型")
        if self.study in {"complete", "universe", "fixed_blend", "financial_context"} and (not isinstance(self.source_run_id, str) or not re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{8}", self.source_run_id)):
            raise ValueError("完整研究需要已核验的原基准运行编号")
        if self.study == "signal" and self.source_run_id is not None:
            raise ValueError("信号基准不接受外部模型输入")
        if self.study in {"complete", "universe", "fixed_blend", "financial_context"} and (self.frequency != "weekly" or self.top_k != 100 or self.feature_set != "price_volume"):
            raise ValueError("完整固定矩阵使用周频、Top100和原16个价量特征")
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 100:
            raise ValueError("ML名称需为1至100个字符")
        dates = [getattr(self, key) for key in ["train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"]]
        for value in dates:
            if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                raise ValueError("ML日期必须明确为YYYY-MM-DD")
        if not (dates[0] <= dates[1] < dates[2] <= dates[3] < dates[4] <= dates[5]):
            raise ValueError("训练、验证、测试日期必须依次分离")
        if self.frequency not in {"weekly", "monthly"} or self.feature_set not in {"price_volume", "price_volume_size"}:
            raise ValueError("不支持的采样频率或特征组")
        for key, low, high in [("horizon", 1, 63), ("min_history", 60, 756), ("top_k", 1, 100), ("bins", 2, 20), ("seed", 0, 1000000), ("tree_iterations", 1, 300), ("tree_leaves", 2, 31), ("tree_min_samples", 2, 10000)]:
            if type(getattr(self, key)) is not int or not low <= getattr(self, key) <= high:
                raise ValueError(f"{key}必须为{low}至{high}的整数")
        for key, low, high in [("min_avg_amount", 0, 1e12), ("ridge_alpha", 1e-8, 1e6), ("tree_learning_rate", 0.001, 1), ("tree_l2", 0, 1e6)]:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key}不是允许范围内的有限数")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
            raise ValueError("ML配置含未知字段或类型无效")
        return cls(**value)


FEATURES = {
    "return_1": "当日log(收盘/参考前收)",
    "return_5": "最近5个日对数收益之和",
    "return_20": "最近20个日对数收益之和",
    "return_60": "最近60个日对数收益之和",
    "volatility_5": "最近5日对数收益标准差",
    "volatility_20": "最近20日对数收益标准差",
    "volatility_60": "最近60日对数收益标准差",
    "bias_5": "衔接收益指数/5日均值-1",
    "bias_20": "衔接收益指数/20日均值-1",
    "bias_60": "衔接收益指数/60日均值-1",
    "range_1": "(当日高-低)/参考前收",
    "body_1": "(当日收盘-开盘)/参考前收",
    "amount_ratio_5_20": "5日均成交额/20日均成交额-1",
    "volume_ratio_5_20": "5日均成交量/20日均成交量-1",
    "amount_volatility_20": "20日成交额标准差/20日均成交额",
    "close_location": "(当日收盘-最低)/(最高-最低)；平价日为0.5",
}


def feature_names(spec):
    return list(FEATURES) + (["log_market_cap"] if spec.feature_set == "price_volume_size" else [])


def benchmark_proposal(snapshot_id="31eef7349722ec89cd1a3743"):
    return {
        "kind": "ml",
        "hypothesis": "固定价量表示在后续日期是否提供稳定选股排序；工程跑通不以正收益为条件",
        "expected_observation": "输出训练/验证/测试RankIC、分组收益与Top100相对池收益；同池比较岭回归、受限提升树、小市值、反转和固定随机11",
        "falsification": "后续排序或最高组相对收益缺乏稳定性即不晋级；数据/边界/预测核查失败则该执行失败，不搜测试参数",
        "snapshot_id": snapshot_id,
        "timeout_seconds": 1200,
        "spec": MLSpec().to_dict(),
    }


def completion_proposal(source_run_id="20261003T165732-991ae261", snapshot_id="31eef7349722ec89cd1a3743"):
    spec = MLSpec(name="机器学习完整案例 · 账户与研究对照", study="complete", source_run_id=source_run_id)
    return dict(kind="ml", snapshot_id=snapshot_id, timeout_seconds=7200, spec=spec.to_dict(),
        hypothesis="原价量排序信息能否转化为同规则账户表现；周期、信息组、规模暴露和重训各有何贡献",
        expected_observation="固定19个策略账户各两种费用情景，有限验证期调参全尝试、四组消融、年度重训、规模组内及标签打乱对照全部报告",
        falsification="不因历史冠军而晋级；缺少稳定增量则报告缺少增量；标签、训练时间或账户对账失败则执行失败并保留日志")
