**机器学习V1代码逐行阅读**

完整流程先读 [从零教程](ML_FROM_ZERO.md)。这里逐行保留源码和行号；多行语句的说明共享。只解释新增ML文件与现有文件的本次集成差异。

**src/mlresearch/__init__.py · 全部新增源码**



`L1` — 这个文档字符串说明学习包范围，Python据此识别包；没有训练或交易逻辑。

```python
"""Snapshot-backed, time-separated machine learning signal research."""
```

**src/mlresearch/contracts.py · 全部新增源码**



`L1` — 导入：数据类、字段枚举与字典转换；建立及校验固定配置。

```python
from dataclasses import asdict, dataclass, fields
```

`L2` — 导入：标准日期校验，或UTC运行编号。

```python
from datetime import date
```

`L3` — 导入：检查配置有限数值。

```python
import math
```

`L4` — 空行：仅分隔代码段，无计算行为。

```python

```

`L5` — 空行：仅分隔代码段，无计算行为。

```python

```

`L6` — 定义不可直接改写字段的配置类；dataclass 自动生成初始化方法，frozen 约束同一实例不可随意赋值。

```python
@dataclass(frozen=True)
```

`L7` — 定义不可直接改写字段的配置类；dataclass 自动生成初始化方法，frozen 约束同一实例不可随意赋值。

```python
class MLSpec:
```

`L8` — 配置研究名称，只用于展示；实质定义去重时不靠名称制造新实验。

```python
    name: str = "日线价量选股 · 五日相对收益基准"
```

`L9` — 明确三个阶段的起止日；2020—2022 拟合，2023 中间评价，2024—2026 后续评价。

```python
    train_start: str = "2020-01-01"
```

`L10` — 明确三个阶段的起止日；2020—2022 拟合，2023 中间评价，2024—2026 后续评价。

```python
    train_end: str = "2022-12-31"
```

`L11` — 明确三个阶段的起止日；2020—2022 拟合，2023 中间评价，2024—2026 后续评价。

```python
    valid_start: str = "2023-01-01"
```

`L12` — 明确三个阶段的起止日；2020—2022 拟合，2023 中间评价，2024—2026 后续评价。

```python
    valid_end: str = "2023-12-31"
```

`L13` — 明确三个阶段的起止日；2020—2022 拟合，2023 中间评价，2024—2026 后续评价。

```python
    test_start: str = "2024-01-01"
```

`L14` — 明确三个阶段的起止日；2020—2022 拟合，2023 中间评价，2024—2026 后续评价。

```python
    test_end: str = "2026-09-24"
```

`L15` — 未来标签窗口默认为五个市场交易日，不是五条个股记录。

```python
    horizon: int = 5
```

`L16` — 默认周采样；此配置也允许月采样，但本次没有跑月采样。

```python
    frequency: str = "weekly"
```

`L17` — 要求至少60个连续有效市场日历史。

```python
    min_history: int = 60
```

`L18` — 共同候选20日均成交额门槛为2000万元。

```python
    min_avg_amount: float = 20_000_000.0
```

`L19` — 默认只把价量列送入模型；可选加市值的配置未执行。

```python
    feature_set: str = "price_volume"
```

`L20` — 取评分前100只，并将全池按位置分成10组。

```python
    top_k: int = 100
```

`L21` — 取评分前100只，并将全池按位置分成10组。

```python
    bins: int = 10
```

`L22` — 固定随机种子11；随机对照和抽样核查可重复。

```python
    seed: int = 11
```

`L23` — 岭回归系数平方惩罚系数10，并非调参获胜值。

```python
    ridge_alpha: float = 10.0
```

`L24` — 限制提升树为80轮、最多7叶、每叶至少200样本、学习率0.05、叶子L2惩罚10。

```python
    tree_iterations: int = 80
```

`L25` — 限制提升树为80轮、最多7叶、每叶至少200样本、学习率0.05、叶子L2惩罚10。

```python
    tree_leaves: int = 7
```

`L26` — 限制提升树为80轮、最多7叶、每叶至少200样本、学习率0.05、叶子L2惩罚10。

```python
    tree_min_samples: int = 200
```

`L27` — 限制提升树为80轮、最多7叶、每叶至少200样本、学习率0.05、叶子L2惩罚10。

```python
    tree_learning_rate: float = 0.05
```

`L28` — 限制提升树为80轮、最多7叶、每叶至少200样本、学习率0.05、叶子L2惩罚10。

```python
    tree_l2: float = 10.0
```

`L29` — 空行：仅分隔代码段，无计算行为。

```python

```

`L30` — 初始化后执行合法性检查；非法配置在训练前拒绝。

```python
    def __post_init__(self):
```

`L31` — 名称必须是非空文本且不超过100字符。

```python
        if not isinstance(self.name, str) or not self.name.strip() or len(self.name) > 100:
```

`L32` — 名称必须是非空文本且不超过100字符。

```python
            raise ValueError("ML名称需为1至100个字符")
```

`L33` — 按训练起止、验证起止、测试起止取得六个明确日期。

```python
        dates = [getattr(self, key) for key in ["train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"]]
```

`L34` — 逐一要求标准YYYY-MM-DD字符串，拒绝不可解析或非规范日期。

```python
        for value in dates:
```

`L35` — 逐一要求标准YYYY-MM-DD字符串，拒绝不可解析或非规范日期。

```python
            if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
```

`L36` — 逐一要求标准YYYY-MM-DD字符串，拒绝不可解析或非规范日期。

```python
                raise ValueError("ML日期必须明确为YYYY-MM-DD")
```

`L37` — 要求各阶段起止有序且训练、验证、测试互不相交。

```python
        if not (dates[0] <= dates[1] < dates[2] <= dates[3] < dates[4] <= dates[5]):
```

`L38` — 要求各阶段起止有序且训练、验证、测试互不相交。

```python
            raise ValueError("训练、验证、测试日期必须依次分离")
```

`L39` — 只接受已支持的采样频率和两种特征白名单。

```python
        if self.frequency not in {"weekly", "monthly"} or self.feature_set not in {"price_volume", "price_volume_size"}:
```

`L40` — 只接受已支持的采样频率和两种特征白名单。

```python
            raise ValueError("不支持的采样频率或特征组")
```

`L41` — 逐个检查窗口、样本数、Top数、树规模与种子范围；bool不当整数。

```python
        for key, low, high in [("horizon", 1, 63), ("min_history", 60, 756), ("top_k", 1, 100), ("bins", 2, 20), ("seed", 0, 1000000), ("tree_iterations", 1, 300), ("tree_leaves", 2, 31), ("tree_min_samples", 2, 10000)]:
```

`L42` — 逐个检查窗口、样本数、Top数、树规模与种子范围；bool不当整数。

```python
            if type(getattr(self, key)) is not int or not low <= getattr(self, key) <= high:
```

`L43` — 逐个检查窗口、样本数、Top数、树规模与种子范围；bool不当整数。

```python
                raise ValueError(f"{key}必须为{low}至{high}的整数")
```

`L44` — 检查浮点参数类型、有限性及范围，避免NaN/无穷和过大设置进入执行。

```python
        for key, low, high in [("min_avg_amount", 0, 1e12), ("ridge_alpha", 1e-8, 1e6), ("tree_learning_rate", 0.001, 1), ("tree_l2", 0, 1e6)]:
```

`L45` — 检查浮点参数类型、有限性及范围，避免NaN/无穷和过大设置进入执行。

```python
            value = getattr(self, key)
```

`L46` — 检查浮点参数类型、有限性及范围，避免NaN/无穷和过大设置进入执行。

```python
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
```

`L47` — 检查浮点参数类型、有限性及范围，避免NaN/无穷和过大设置进入执行。

```python
                raise ValueError(f"{key}不是允许范围内的有限数")
```

`L48` — 空行：仅分隔代码段，无计算行为。

```python

```

`L49` — 把配置转换为普通字典，用于JSON记录和冻结。

```python
    def to_dict(self):
```

`L50` — 把配置转换为普通字典，用于JSON记录和冻结。

```python
        return asdict(self)
```

`L51` — 空行：仅分隔代码段，无计算行为。

```python

```

`L52` — 从字典恢复配置；未知字段拒绝，避免拼错参数被悄悄忽略。

```python
    @classmethod
```

`L53` — 从字典恢复配置；未知字段拒绝，避免拼错参数被悄悄忽略。

```python
    def from_dict(cls, value):
```

`L54` — 从字典恢复配置；未知字段拒绝，避免拼错参数被悄悄忽略。

```python
        if not isinstance(value, dict) or value.keys() - {f.name for f in fields(cls)}:
```

`L55` — 从字典恢复配置；未知字段拒绝，避免拼错参数被悄悄忽略。

```python
            raise ValueError("ML配置含未知字段或类型无效")
```

`L56` — 从字典恢复配置；未知字段拒绝，避免拼错参数被悄悄忽略。

```python
        return cls(**value)
```

`L57` — 空行：仅分隔代码段，无计算行为。

```python

```

`L58` — 空行：仅分隔代码段，无计算行为。

```python

```

`L59` — 声明16个允许进入模型的列；字典值是人类说明，不是计算公式。

```python
FEATURES = {
```

`L60` — 记录1/5/20/60日对数收益列的含义；公式实际在dataset.py。

```python
    "return_1": "当日log(收盘/参考前收)",
```

`L61` — 记录1/5/20/60日对数收益列的含义；公式实际在dataset.py。

```python
    "return_5": "最近5个日对数收益之和",
```

`L62` — 记录1/5/20/60日对数收益列的含义；公式实际在dataset.py。

```python
    "return_20": "最近20个日对数收益之和",
```

`L63` — 记录1/5/20/60日对数收益列的含义；公式实际在dataset.py。

```python
    "return_60": "最近60个日对数收益之和",
```

`L64` — 记录5/20/60日波动列的含义，未年化。

```python
    "volatility_5": "最近5日对数收益标准差",
```

`L65` — 记录5/20/60日波动列的含义，未年化。

```python
    "volatility_20": "最近20日对数收益标准差",
```

`L66` — 记录5/20/60日波动列的含义，未年化。

```python
    "volatility_60": "最近60日对数收益标准差",
```

`L67` — 记录5/20/60日均线偏离列的含义。

```python
    "bias_5": "衔接收益指数/5日均值-1",
```

`L68` — 记录5/20/60日均线偏离列的含义。

```python
    "bias_20": "衔接收益指数/20日均值-1",
```

`L69` — 记录5/20/60日均线偏离列的含义。

```python
    "bias_60": "衔接收益指数/60日均值-1",
```

`L70` — 记录当日振幅和实体列的含义。

```python
    "range_1": "(当日高-低)/参考前收",
```

`L71` — 记录当日振幅和实体列的含义。

```python
    "body_1": "(当日收盘-开盘)/参考前收",
```

`L72` — 记录成交额放大、成交量放大和成交额不稳定度。

```python
    "amount_ratio_5_20": "5日均成交额/20日均成交额-1",
```

`L73` — 记录成交额放大、成交量放大和成交额不稳定度。

```python
    "volume_ratio_5_20": "5日均成交量/20日均成交量-1",
```

`L74` — 记录成交额放大、成交量放大和成交额不稳定度。

```python
    "amount_volatility_20": "20日成交额标准差/20日均成交额",
```

`L75` — 记录收盘位置及平价日0.5约定。

```python
    "close_location": "(当日收盘-最低)/(最高-最低)；平价日为0.5",
```

`L76` — 结束特征说明字典。

```python
}
```

`L77` — 空行：仅分隔代码段，无计算行为。

```python

```

`L78` — 空行：仅分隔代码段，无计算行为。

```python

```

`L79` — 返回明确列顺序：16价量列，只有指定price_volume_size时才追加log_market_cap。

```python
def feature_names(spec):
```

`L80` — 返回明确列顺序：16价量列，只有指定price_volume_size时才追加log_market_cap。

```python
    return list(FEATURES) + (["log_market_cap"] if spec.feature_set == "price_volume_size" else [])
```

`L81` — 空行：仅分隔代码段，无计算行为。

```python

```

`L82` — 空行：仅分隔代码段，无计算行为。

```python

```

`L83` — 生成可交给研究任务登记的固定ML设计；本函数不执行模型训练。

```python
def benchmark_proposal(snapshot_id="31eef7349722ec89cd1a3743"):
```

`L84` — 生成可交给研究任务登记的固定ML设计；本函数不执行模型训练。

```python
    return {
```

`L85` — 使用kind=ml分派到学习执行器，不进入账户回测执行器。

```python
        "kind": "ml",
```

`L86` — 事前提出是否具有后续选股排序信息的可否证问题。

```python
        "hypothesis": "固定价量表示在后续日期是否提供稳定选股排序；工程跑通不以正收益为条件",
```

`L87` — 规定观察RankIC、分组和Top100，并保留两个学习模型与三个规则对照。

```python
        "expected_observation": "输出训练/验证/测试RankIC、分组收益与Top100相对池收益；同池比较岭回归、受限提升树、小市值、反转和固定随机11",
```

`L88` — 规定排序缺乏稳定性不晋级；数据核查失败记执行失败，不靠测试集搜索修饰。

```python
        "falsification": "后续排序或最高组相对收益缺乏稳定性即不晋级；数据/边界/预测核查失败则该执行失败，不搜测试参数",
```

`L89` — 指定快照、1200秒执行时限及全部默认配置。

```python
        "snapshot_id": snapshot_id,
```

`L90` — 指定快照、1200秒执行时限及全部默认配置。

```python
        "timeout_seconds": 1200,
```

`L91` — 指定快照、1200秒执行时限及全部默认配置。

```python
        "spec": MLSpec().to_dict(),
```

`L92` — 返回设计字典结束。

```python
    }
```

**src/mlresearch/dataset.py · 全部新增源码**



`L1` — 模块文档字符串：说明代码范围，不执行数据处理或证明策略有效。

```python
"""Causal features and separately named labels, addressed on the market calendar."""
```

`L2` — 导入：向量计算、有限值处理与统计抽样。

```python
import numpy as np
```

`L3` — 导入：行情面板、滚动窗口、分组及Parquet读取。

```python
import pandas as pd
```

`L4` — 空行：仅分隔代码段，无计算行为。

```python

```

`L5` — 导入：复用严格已公告报告拼接和估算股本。

```python
from ..technical.foundation_data import attach_reports
```

`L6` — 导入：复用股日主键检查和市场日历采样。

```python
from ..technical.signals import check_keys, decision_dates
```

`L7` — 导入：ML配置、输入白名单与列说明。

```python
from .contracts import FEATURES
```

`L8` — 空行：仅分隔代码段，无计算行为。

```python

```

`L9` — 空行：仅分隔代码段，无计算行为。

```python

```

`L10` — 对一只股票的完整原始日线计算过去可知特征。

```python
def price_features(frame, calendar, min_history=60):
```

`L11` — 按日期排序，复制并重建行索引，避免改源表或顺序错乱。

```python
    b = frame.sort_values("trade_date").copy().reset_index(drop=True)
```

`L12` — 把交易日映射为市场日历的位置，用来识别连续市场日。

```python
    b["session"] = b.trade_date.map({d: i for i, d in enumerate(calendar)})
```

`L13` — 报价日期不在日历就报错，不猜测有效交易日。

```python
    if b.session.isna().any():
```

`L14` — 报价日期不在日历就报错，不猜测有效交易日。

```python
        raise ValueError("行情日期不在快照日历")
```

`L15` — 参考前收和收盘都必须为正且有限才可计算对数收益。

```python
    valid = b.close.gt(0) & b.pre_close.gt(0) & np.isfinite(b.close) & np.isfinite(b.pre_close)
```

`L16` — 计算ln(close/pre_close)，无效配对保留为缺失。

```python
    r = np.log((b.close / b.pre_close).where(valid))
```

`L17` — 仅用截至当天的收益累计建立衔接指数；缺失临时作0衔接，但连续历史资格另行排除。

```python
    # Cumulative past reference returns, never adjusted with future actions.
```

`L18` — 仅用截至当天的收益累计建立衔接指数；缺失临时作0衔接，但连续历史资格另行排除。

```python
    index = np.exp(r.fillna(0).cumsum())
```

`L19` — 分别滚动求1/5/20/60日对数收益和；只向过去看。

```python
    for window in [1, 5, 20, 60]:
```

`L20` — 分别滚动求1/5/20/60日对数收益和；只向过去看。

```python
        b[f"return_{window}"] = r.rolling(window).sum()
```

`L21` — 依次处理5/20/60日窗口的波动与均线偏离。

```python
    for window in [5, 20, 60]:
```

`L22` — 计算窗口内收益样本标准差，未乘年化因子。

```python
        b[f"volatility_{window}"] = r.rolling(window).std()
```

`L23` — 衔接指数除以自身过去窗口均值减1，表达均线偏离。

```python
        b[f"bias_{window}"] = index / index.rolling(window).mean() - 1
```

`L24` — 当日高低差除参考前收，表达振幅。

```python
    b["range_1"] = (b.high - b.low) / b.pre_close
```

`L25` — 收开差除参考前收，表达实体方向和大小。

```python
    b["body_1"] = (b.close - b.open) / b.pre_close
```

`L26` — 计算20日平均成交额，既用于特征分母也用于候选资格。

```python
    b["avg_amount_20"] = b.amount.rolling(20).mean()
```

`L27` — 5日均成交额比20日均额减1，表达近期放量。

```python
    b["amount_ratio_5_20"] = b.amount.rolling(5).mean() / b.avg_amount_20 - 1
```

`L28` — 5日均成交量比20日均量减1，量和额是两列不同输入。

```python
    b["volume_ratio_5_20"] = b.volume.rolling(5).mean() / b.volume.rolling(20).mean() - 1
```

`L29` — 成交额标准差除平均额，表达相对不稳定度。

```python
    b["amount_volatility_20"] = b.amount.rolling(20).std() / b.avg_amount_20
```

`L30` — 收盘位于高低范围中的相对位置；高低相同取0.5避免除零。

```python
    b["close_location"] = ((b.close - b.low) / (b.high - b.low)).where(b.high.ne(b.low), 0.5)
```

`L31` — 要求最近min_history行恰好覆盖连续市场日，且这些日收益全部有效。

```python
    b["history_valid"] = b.session.sub(b.session.shift(min_history - 1)).eq(min_history - 1) & r.rolling(min_history).count().eq(min_history)
```

`L32` — 把特征中的正负无穷转成NaN，留给训练段拟合的填补器。

```python
    b[list(FEATURES)] = b[list(FEATURES)].replace([np.inf, -np.inf], np.nan)
```

`L33` — 私有辅助列_r留给标签函数，白名单不会把它送入模型。

```python
    b["_r"] = r
```

`L34` — 返回带过去特征的单股表。

```python
    return b
```

`L35` — 空行：仅分隔代码段，无计算行为。

```python

```

`L36` — 空行：仅分隔代码段，无计算行为。

```python

```

`L37` — 另设标签函数并复制输入，避免标签代码混进X计算。

```python
def forward_labels(features, calendar, horizon):
```

`L38` — 另设标签函数并复制输入，避免标签代码混进X计算。

```python
    b = features.copy()
```

`L39` — 用市场日历t+h寻址label_end，越过日历末尾的日期没有完整标签。

```python
    b["label_end"] = b.trade_date.map({d: calendar[i + horizon] for i, d in enumerate(calendar) if i + horizon < len(calendar)})
```

`L40` — 先滚动求h日和，再向前移h行；t位置得到t+1到t+h的收益和。

```python
    future_sum = b._r.rolling(horizon).sum().shift(-horizon)
```

`L41` — 验证股票行跨度等于h个市场日；遇缺报价不能偷换成往后h条记录。

```python
    contiguous = b.session.shift(-horizon).sub(b.session).eq(horizon)
```

`L42` — 对数收益转简单累计收益；不连续时保持未知NaN。

```python
    b["label_return"] = np.expm1(future_sum).where(contiguous)
```

`L43` — 返回带未来标签的表，标签仍不是训练输入。

```python
    return b
```

`L44` — 空行：仅分隔代码段，无计算行为。

```python

```

`L45` — 空行：仅分隔代码段，无计算行为。

```python

```

`L46` — 复制面板，默认所有行excluded；只有明确符合阶段和成熟条件才纳入。

```python
def assign_splits(frame, spec):
```

`L47` — 复制面板，默认所有行excluded；只有明确符合阶段和成熟条件才纳入。

```python
    out = frame.copy()
```

`L48` — 复制面板，默认所有行excluded；只有明确符合阶段和成熟条件才纳入。

```python
    out["split"] = "excluded"
```

`L49` — 逐一给出训练、验证、测试的起止日及下一阶段起点。

```python
    for name, start, end, next_start in [
```

`L50` — 逐一给出训练、验证、测试的起止日及下一阶段起点。

```python
        ("train", spec.train_start, spec.train_end, spec.valid_start),
```

`L51` — 逐一给出训练、验证、测试的起止日及下一阶段起点。

```python
        ("valid", spec.valid_start, spec.valid_end, spec.test_start),
```

`L52` — 逐一给出训练、验证、测试的起止日及下一阶段起点。

```python
        ("test", spec.test_start, spec.test_end, None),
```

`L53` — 逐一给出训练、验证、测试的起止日及下一阶段起点。

```python
    ]:
```

`L54` — 采样日必须落在该阶段内。

```python
        mask = out.trade_date.between(start, end)
```

`L55` — 标签必须有计划结束日，且不晚于本阶段末日。

```python
        # Boundary purge is by the planned market label end, not panel row count.
```

`L56` — 标签必须有计划结束日，且不晚于本阶段末日。

```python
        mature = out.label_end.notna() & out.label_end.le(end)
```

`L57` — 有下一阶段时额外要求标签结束日早于下一阶段，防止跨界。

```python
        if next_start:
```

`L58` — 有下一阶段时额外要求标签结束日早于下一阶段，防止跨界。

```python
            mature &= out.label_end.lt(next_start)
```

`L59` — 只给日期与成熟度均合格的行赋train/valid/test。

```python
        out.loc[mask & mature, "split"] = name
```

`L60` — 返回阶段标记；本函数不读取未来收益大小决定去留。

```python
    return out
```

`L61` — 空行：仅分隔代码段，无计算行为。

```python

```

`L62` — 空行：仅分隔代码段，无计算行为。

```python

```

`L63` — 完整面板入口；先核验行情主键，再从市场日历生成周/月采样计划。

```python
def build_panel(bars, calendar, metadata, filings, spec, progress=lambda m: None):
```

`L64` — 完整面板入口；先核验行情主键，再从市场日历生成周/月采样计划。

```python
    check_keys(bars, "ML行情")
```

`L65` — 完整面板入口；先核验行情主键，再从市场日历生成周/月采样计划。

```python
    schedule = decision_dates(calendar, spec.train_start, spec.test_end, spec.frequency)
```

`L66` — 准备收集每只股票的采样行；按股票分组，滚动窗口不串股。

```python
    frames = []
```

`L67` — 准备收集每只股票的采样行；按股票分组，滚动窗口不串股。

```python
    groups = bars.groupby("stock_code", sort=False, observed=True)
```

`L68` — 逐只股票处理，不在股票之间累积价格或收益。

```python
    for n, (code, raw) in enumerate(groups):
```

`L69` — 先算过去特征。

```python
        features = price_features(raw, calendar, spec.min_history)
```

`L70` — 再独立生成未来标签。

```python
        labeled = forward_labels(features, calendar, spec.horizon)
```

`L71` — 明确暂时保留的身份、资格、标签与16特征列。

```python
        cols = ["stock_code", "trade_date", "close", "is_st", "volume", "history_valid", "avg_amount_20", "label_end", "label_return"] + list(FEATURES)
```

`L72` — 只保留采样日；日线仍用于计算过去窗口和未来标签。

```python
        frames.append(labeled.loc[labeled.trade_date.isin(schedule), cols])
```

`L73` — 每处理约500只报告进度，便于长运行观察；不影响研究定义。

```python
        if n % 500 == 0:
```

`L74` — 每处理约500只报告进度，便于长运行观察；不影响研究定义。

```python
            progress(f"构建过去价量特征与独立标签：{n + 1}/{groups.ngroups}只")
```

`L75` — 把单股采样表拼成所有候选的面板。

```python
    panel = pd.concat(frames, ignore_index=True)
```

`L76` — 合并当时上市/退市日期；many_to_one防止元数据重复造成行情膨胀。

```python
    panel = panel.merge(metadata[["stock_code", "listed_date", "de_listed_date"]], on="stock_code", validate="many_to_one")
```

`L77` — 复用已公告报告拼接函数；严格以公告日期早于采样日为信息边界。

```python
    panel = attach_reports(panel, filings)
```

`L78` — 报告期龄0到240天、估算市值正且有限，才符合共同股本资格。

```python
    cap_valid = panel.q_age_days.between(0, 240) & panel.estimated_market_cap.gt(0) & np.isfinite(panel.estimated_market_cap)
```

`L79` — 只按当时上市/未退市、非ST、量为正、连续历史、成交额与股本过滤资格；未用未来标签。

```python
    panel["eligible"] = (panel.listed_date.le(panel.trade_date) & panel.de_listed_date.gt(panel.trade_date)
```

`L80` — 只按当时上市/未退市、非ST、量为正、连续历史、成交额与股本过滤资格；未用未来标签。

```python
        & panel.is_st.eq(0) & panel.volume.gt(0) & panel.history_valid
```

`L81` — 只按当时上市/未退市、非ST、量为正、连续历史、成交额与股本过滤资格；未用未来标签。

```python
        & panel.avg_amount_20.ge(spec.min_avg_amount) & cap_valid)
```

`L82` — 计算对数市值，仅供小市值对照或明确选择的扩展X。

```python
    panel["log_market_cap"] = np.log(panel.estimated_market_cap.where(cap_valid))
```

`L83` — 按计划标签结束日划分阶段并清理边界。

```python
    panel = assign_splits(panel, spec)
```

`L84` — 保留当天合格且阶段成熟行；未知未来收益不因此从排名世界中删除。

```python
    # Eligibility uses no label; keep unknown outcomes and disclose their coverage.
```

`L85` — 保留当天合格且阶段成熟行；未知未来收益不因此从排名世界中删除。

```python
    panel = panel.loc[panel.eligible & panel.split.ne("excluded")].copy()
```

`L86` — 记录完整日线假设在当天15:30可知；并非供应商真实交付时间认证。

```python
    panel["known_at"] = panel.trade_date + "T15:30:00+08:00"
```

`L87` — 有限未来收益标记label_observed，训练与覆盖评价会使用。

```python
    panel["label_observed"] = np.isfinite(panel.label_return)
```

`L88` — 未来收益减同日有标签池均值，形成学习目标Y；不是把未来均值放入X。

```python
    panel["label_relative"] = panel.label_return - panel.groupby("trade_date").label_return.transform("mean")
```

`L89` — 明确最终保存的元数据/标签与输入列。

```python
    cols = ["stock_code", "trade_date", "known_at", "label_end", "split", "label_observed", "label_return", "label_relative", "log_market_cap", "q_publication_date"] + list(FEATURES)
```

`L90` — 按日期和代码稳定排序，便于模型预测文件保持同一行身份。

```python
    panel = panel[cols].sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
```

`L91` — 再次检查一股一天只能一行。

```python
    check_keys(panel, "ML面板")
```

`L92` — 公告日不能等于或晚于采样日，发现即拒绝。

```python
    if (panel.q_publication_date >= panel.trade_date).any():
```

`L93` — 公告日不能等于或晚于采样日，发现即拒绝。

```python
        raise ValueError("股本资格使用了当日或未来公告")
```

`L94` — 返回完整面板；这里没有订单、持仓或成交。

```python
    return panel
```

**src/mlresearch/runner.py · 全部新增源码**



`L1` — 导入：标准日期校验，或UTC运行编号。

```python
from datetime import datetime, timezone
```

`L2` — 导入：SHA256固定排序或文件身份；不是预测模型。

```python
import hashlib
```

`L3` — 导入：记录并核查已安装包版本。

```python
import importlib.metadata
```

`L4` — 导入：读写可追溯配置、收据与结果。

```python
import json
```

`L5` — 导入：安全明确的文件路径处理。

```python
from pathlib import Path
```

`L6` — 导入：记录Python及系统描述。

```python
import platform
```

`L7` — 导入：计时、恢复事件时间或有期限验收等待。

```python
import time
```

`L8` — 导入：给运行编号加独立后缀避免覆盖。

```python
import uuid
```

`L9` — 空行：仅分隔代码段，无计算行为。

```python

```

`L10` — 导入：保存与加载本系统已拟合Pipeline。

```python
import joblib
```

`L11` — 导入numpy但本文件没有使用np别名；这是冗余导入，不参与训练或评价。

```python
import numpy as np
```

`L12` — 导入：行情面板、滚动窗口、分组及Parquet读取。

```python
import pandas as pd
```

`L13` — 导入：真正的小树提升回归算法。

```python
from sklearn.ensemble import HistGradientBoostingRegressor
```

`L14` — 导入：仅从训练数据学习中位数填补。

```python
from sklearn.impute import SimpleImputer
```

`L15` — 导入：岭回归算法。

```python
from sklearn.linear_model import Ridge
```

`L16` — 导入：将预处理和模型串成fit/predict一致对象。

```python
from sklearn.pipeline import make_pipeline
```

`L17` — 导入：只在训练段学习的标准化。

```python
from sklearn.preprocessing import StandardScaler
```

`L18` — 导入：限制本次数值计算线程数。

```python
from threadpoolctl import threadpool_limits
```

`L19` — 空行：仅分隔代码段，无计算行为。

```python

```

`L20` — 复用digest、records、verify_artifacts和write_json；此处content_id没有实际使用，是冗余导入。

```python
from ..technical.artifacts import content_id, digest, records, verify_artifacts, write_json
```

`L21` — 导入：ML配置、输入白名单与列说明。

```python
from .contracts import MLSpec, FEATURES, feature_names
```

`L22` — 导入：行情转研究面板的入口。

```python
from .dataset import build_panel
```

`L23` — 导入：排序诊断与方法白名单。

```python
from .evaluation import evaluate
```

`L24` — 空行：仅分隔代码段，无计算行为。

```python

```

`L25` — 空行：仅分隔代码段，无计算行为。

```python

```

`L26` — 建立并拟合两种固定模型，输入是完整研究面板与配置。

```python
def fit_models(panel, spec):
```

`L27` — 取明确特征白名单及顺序。

```python
    names = feature_names(spec)
```

`L28` — 只选择train且Y已知的行，验证/测试行不参与拟合。

```python
    train = panel[panel.split.eq("train") & panel.label_observed]
```

`L29` — 训练至少100行，并足以支持至少两个最小叶子，否则拒绝。

```python
    if len(train) < max(100, spec.tree_min_samples * 2):
```

`L30` — 训练至少100行，并足以支持至少两个最小叶子，否则拒绝。

```python
        raise ValueError("ML训练样本不足")
```

`L31` — 任何输入列在训练里全空则拒绝；不让填补器悄悄改特征维度。

```python
    if train[names].notna().sum().eq(0).any():
```

`L32` — 任何输入列在训练里全空则拒绝；不让填补器悄悄改特征维度。

```python
        raise ValueError("训练段存在全空特征，不能悄悄删除输入列")
```

`L33` — X只取白名单列，Y取相对收益；股票代码、日期和未来财富都不在X。

```python
    X, y = train[names], train.label_relative
```

`L34` — 建立中位数填补→标准化→岭回归Pipeline，全部统计只在后续fit训练行时学习。

```python
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=spec.ridge_alpha))
```

`L35` — 建立中位数填补→HistGBDT Pipeline；固定轮数、叶数、最小叶样本、步长和L2，关闭内部早停。

```python
    tree = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(
```

`L36` — 建立中位数填补→HistGBDT Pipeline；固定轮数、叶数、最小叶样本、步长和L2，关闭内部早停。

```python
        max_iter=spec.tree_iterations, max_leaf_nodes=spec.tree_leaves,
```

`L37` — 建立中位数填补→HistGBDT Pipeline；固定轮数、叶数、最小叶样本、步长和L2，关闭内部早停。

```python
        min_samples_leaf=spec.tree_min_samples, learning_rate=spec.tree_learning_rate,
```

`L38` — 建立中位数填补→HistGBDT Pipeline；固定轮数、叶数、最小叶样本、步长和L2，关闭内部早停。

```python
        l2_regularization=spec.tree_l2, early_stopping=False, random_state=spec.seed))
```

`L39` — 给两种模型稳定名称，不依据验证或测试表现选择其中一个。

```python
    models = {"ridge": ridge, "hist_gbdt": tree}
```

`L40` — 最多4个计算线程，逐个fit同一训练X/Y；拟合包括各自预处理。

```python
    with threadpool_limits(limits=4):
```

`L41` — 最多4个计算线程，逐个fit同一训练X/Y；拟合包括各自预处理。

```python
        for model in models.values():
```

`L42` — 最多4个计算线程，逐个fit同一训练X/Y；拟合包括各自预处理。

```python
            model.fit(X, y)
```

`L43` — 返回已经学习完成的两个模型对象。

```python
    return models
```

`L44` — 空行：仅分隔代码段，无计算行为。

```python

```

`L45` — 空行：仅分隔代码段，无计算行为。

```python

```

`L46` — 新建预测表，先保留身份与标签供评价；不是把这些列喂给模型。

```python
def predict_panel(panel, models, spec):
```

`L47` — 新建预测表，先保留身份与标签供评价；不是把这些列喂给模型。

```python
    predicted = panel[["stock_code", "trade_date", "label_end", "split", "label_observed", "label_return", "label_relative"]].copy()
```

`L48` — 对面板所有阶段的白名单X预测，沿用训练预处理；没有fit测试数据。

```python
    with threadpool_limits(limits=4):
```

`L49` — 对面板所有阶段的白名单X预测，沿用训练预处理；没有fit测试数据。

```python
        for name, model in models.items():
```

`L50` — 对面板所有阶段的白名单X预测，沿用训练预处理；没有fit测试数据。

```python
            predicted[name] = model.predict(panel[feature_names(spec)])
```

`L51` — 对照分数为负对数市值，分数越高市值越小。

```python
    predicted["small_cap"] = -panel.log_market_cap
```

`L52` — 反转分数为过去五日收益的负值，跌得越多评分越高。

```python
    predicted["reversal_5"] = -panel.return_5
```

`L53` — 代码加种子取SHA256前13个十六进制位，归一化到[0,1)，每股固定分数。

```python
    scores = {c: int(hashlib.sha256(f"{spec.seed}:{c}".encode()).hexdigest()[:13], 16) / 16 ** 13 for c in panel.stock_code.unique()}
```

`L54` — 将固定分数映射到股票各日期；不是每期重新随机选。

```python
    predicted["fixed_random"] = panel.stock_code.map(scores)
```

`L55` — 返回逐股预测和对照分数，不执行账户交易。

```python
    return predicted
```

`L56` — 空行：仅分隔代码段，无计算行为。

```python

```

`L57` — 空行：仅分隔代码段，无计算行为。

```python

```

`L58` — 冻结配置执行入口；用单调计时器记录本次运行耗时。

```python
def run(cfg, progress=lambda m: None):
```

`L59` — 冻结配置执行入口；用单调计时器记录本次运行耗时。

```python
    started = time.perf_counter()
```

`L60` — 定位共享研究库与快照，读取快照清单。

```python
    root = Path(cfg["root"])
```

`L61` — 定位共享研究库与快照，读取快照清单。

```python
    snapshot = root / "snapshots" / cfg["snapshot_id"]
```

`L62` — 定位共享研究库与快照，读取快照清单。

```python
    sm = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
```

`L63` — 清单哈希必须等于登记值，全部快照文件也必须符合清单。

```python
    if digest(snapshot / "manifest.json") != cfg["snapshot_manifest_hash"]:
```

`L64` — 清单哈希必须等于登记值，全部快照文件也必须符合清单。

```python
        raise ValueError("ML快照清单变化")
```

`L65` — 清单哈希必须等于登记值，全部快照文件也必须符合清单。

```python
    verify_artifacts(snapshot, sm)
```

`L66` — 恢复并验证ML配置。

```python
    spec = MLSpec.from_dict(cfg["spec"])
```

`L67` — 核对当前Python及依赖版本与登记时一致；版本变化不能静默执行。

```python
    environment = {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in cfg["environment"]["packages"]}}
```

`L68` — 核对当前Python及依赖版本与登记时一致；版本变化不能静默执行。

```python
    if environment != cfg["environment"]:
```

`L69` — 核对当前Python及依赖版本与登记时一致；版本变化不能静默执行。

```python
        raise ValueError("ML执行环境与登记版本不同")
```

`L70` — UTC时间与随机后缀生成独立运行目录，写running状态，避免覆盖旧结果。

```python
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
```

`L71` — UTC时间与随机后缀生成独立运行目录，写running状态，避免覆盖旧结果。

```python
    folder = root / "ml_runs" / run_id
```

`L72` — UTC时间与随机后缀生成独立运行目录，写running状态，避免覆盖旧结果。

```python
    folder.mkdir(parents=True)
```

`L73` — UTC时间与随机后缀生成独立运行目录，写running状态，避免覆盖旧结果。

```python
    write_json(folder / "status.json", {"status": "running", "run_id": run_id, "kind": "ml"})
```

`L74` — 开始运行保护；之后任何异常写failed状态并保留目录。

```python
    try:
```

`L75` — 保存本次全部配置。

```python
        write_json(folder / "spec.json", spec.to_dict())
```

`L76` — 保存Python、操作系统描述、包版本和线程上限；不是容器冻结。

```python
        write_json(folder / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
```

`L77` — 保存Python、操作系统描述、包版本和线程上限；不是容器冻结。

```python
            "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "joblib", "threadpoolctl"]}, "thread_limit": 4})
```

`L78` — 读快照市场交易日历。

```python
        calendar = json.loads((snapshot / "calendar.json").read_text(encoding="utf-8"))
```

`L79` — 只读取10列原始行情和截至test_end的记录，保留早于训练起点的预热历史。

```python
        columns = ["stock_code", "trade_date", "open", "high", "low", "close", "pre_close", "volume", "amount", "is_st"]
```

`L80` — 只读取10列原始行情和截至test_end的记录，保留早于训练起点的预热历史。

```python
        bars = pd.read_parquet(snapshot / "bars.parquet", columns=columns, filters=[("trade_date", "<=", spec.test_end)])
```

`L81` — 读取上市元数据与已报告财务/股本表，用于共同候选和对照。

```python
        metadata = pd.read_parquet(snapshot / "metadata.parquet")
```

`L82` — 读取上市元数据与已报告财务/股本表，用于共同候选和对照。

```python
        filings = pd.read_parquet(snapshot / "fundamentals.parquet")
```

`L83` — 调用build_panel完成X、Y、资格、时间阶段。

```python
        panel = build_panel(bars, calendar, metadata, filings, spec, progress)
```

`L84` — 释放大行情表引用，减少内存占用。

```python
        del bars
```

`L85` — 逐阶段收集样本数、日期数与标签成熟信息。

```python
        counts = []
```

`L86` — 逐阶段收集样本数、日期数与标签成熟信息。

```python
        for split in ["train", "valid", "test"]:
```

`L87` — 逐阶段收集样本数、日期数与标签成熟信息。

```python
            part = panel[panel.split.eq(split)]
```

`L88` — 各阶段需有足够已知标签及至少两个采样日，否则拒绝评价。

```python
            if part.empty or part.label_observed.sum() < 100 or part.trade_date.nunique() < 2:
```

`L89` — 各阶段需有足够已知标签及至少两个采样日，否则拒绝评价。

```python
                raise ValueError(f"{split}有效日期或标签不足")
```

`L90` — 保存各阶段候选行、有标签行、首末采样日及最大标签结束日。

```python
            counts.append(dict(split=split, rows=len(part), dates=part.trade_date.nunique(),
```

`L91` — 保存各阶段候选行、有标签行、首末采样日及最大标签结束日。

```python
                observed_labels=int(part.label_observed.sum()), first_date=part.trade_date.min(), last_date=part.trade_date.max(), max_label_end=part.label_end.max()))
```

`L92` — 将面板完整写入Parquet，保留逐股可追溯输入。

```python
        panel.to_parquet(folder / "panel.parquet", index=False)
```

`L93` — dataset.json记录白名单、公式、资格、标签缺失、成熟边界和训练范围；明确回顾性时间隔离。

```python
        write_json(folder / "dataset.json", {"counts": counts, "features": feature_names(spec), "feature_definitions": FEATURES,
```

`L94` — dataset.json记录白名单、公式、资格、标签缺失、成熟边界和训练范围；明确回顾性时间隔离。

```python
            "feature_set": spec.feature_set, "panel_sha256": digest(folder / "panel.parquet"),
```

`L95` — dataset.json记录白名单、公式、资格、标签缺失、成熟边界和训练范围；明确回顾性时间隔离。

```python
            "target": "未来h个市场交易日参考价衔接收益，减同日有标签候选池均值",
```

`L96` — dataset.json记录白名单、公式、资格、标签缺失、成熟边界和训练范围；明确回顾性时间隔离。

```python
            "eligibility": "当日主板上市、非ST、连续历史、成交额和已公告且不过期的股本；不使用未来标签筛候选",
```

`L97` — dataset.json记录白名单、公式、资格、标签缺失、成熟边界和训练范围；明确回顾性时间隔离。

```python
            "missing_labels": "训练需可观测标签；评价先对全部候选排序，再统计已知结果及覆盖率，存在非随机缺失风险",
```

`L98` — dataset.json记录白名单、公式、资格、标签缺失、成熟边界和训练范围；明确回顾性时间隔离。

```python
            "boundary": "label_end不超过各段末日；多日跨界标签排除",
```

`L99` — dataset.json记录白名单、公式、资格、标签缺失、成熟边界和训练范围；明确回顾性时间隔离。

```python
            "training": "只拟合train，valid不拟合不选择模型；固定两种模型均报告，不自动挑赢家",
```

`L100` — dataset.json记录白名单、公式、资格、标签缺失、成熟边界和训练范围；明确回顾性时间隔离。

```python
            "evaluation_scope": "retrospective_time_holdout; not genuinely unseen history"})
```

`L101` — 发出训练进度消息。

```python
        progress(f"训练固定岭回归和受限提升树，{counts[0]['rows']}个候选训练行")
```

`L102` — 真正调用fit_models训练一次；不在每个预测周重训。

```python
        models = fit_models(panel, spec)
```

`L103` — 保存两个完整Pipeline，包含已经学习的中位数、缩放和模型参数。

```python
        for name, model in models.items():
```

`L104` — 保存两个完整Pipeline，包含已经学习的中位数、缩放和模型参数。

```python
            joblib.dump(model, folder / f"{name}.joblib")
```

`L105` — 生成全部候选预测及规则对照分数，并保存逐行文件。

```python
        predictions = predict_panel(panel, models, spec)
```

`L106` — 生成全部候选预测及规则对照分数，并保存逐行文件。

```python
        predictions.to_parquet(folder / "predictions.parquet", index=False)
```

`L107` — 调用统计评价，得到逐日、逐组、汇总和分组均值。

```python
        progress("评价日期内排序、十分组收益与Top100相对池收益")
```

`L108` — 调用统计评价，得到逐日、逐组、汇总和分组均值。

```python
        daily, groups, summaries, quantiles = evaluate(predictions, spec)
```

`L109` — 分别保存日期指标、日期分组、可直接查看的CSV分组均值。

```python
        daily.to_parquet(folder / "date_metrics.parquet", index=False)
```

`L110` — 分别保存日期指标、日期分组、可直接查看的CSV分组均值。

```python
        groups.to_parquet(folder / "quantile_dates.parquet", index=False)
```

`L111` — 分别保存日期指标、日期分组、可直接查看的CSV分组均值。

```python
        quantiles.to_csv(folder / "quantiles.csv", index=False, encoding="utf-8-sig")
```

`L112` — 组织结果JSON，绑定任务/实验、样本数、指标、运行耗时与具体限制。

```python
        result = {"kind": "ml", "run_id": run_id, "name": spec.name, "snapshot_id": cfg["snapshot_id"],
```

`L113` — 组织结果JSON，绑定任务/实验、样本数、指标、运行耗时与具体限制。

```python
            "research_context": {"task_id": cfg["task_id"], "experiment_id": cfg["experiment_id"]},
```

`L114` — 组织结果JSON，绑定任务/实验、样本数、指标、运行耗时与具体限制。

```python
            "counts": counts, "summaries": summaries, "quantiles": records(quantiles), "elapsed_seconds": time.perf_counter() - started,
```

`L115` — 组织结果JSON，绑定任务/实验、样本数、指标、运行耗时与具体限制。

```python
            "limitations": ["全部历史已被查看；这里只对模型拟合按时间隔离", "信号收益不是可成交账户收益，不计算年化或费用反事实", "股本滞后与供应商历史修订/退市覆盖仍受原快照限制", "未知未来标签可能非随机，展示总体及入选覆盖率", "4个采样日期循环块区间为逐项诊断，不校正全部研究尝试或多重比较", "固定随机11是一个种子对照，不是通用市场基准"]}
```

`L116` — 写result.json供CLI、界面和核查读取。

```python
        write_json(folder / "result.json", result)
```

`L117` — 建立自动REPORT的简短说明与表头；它原本没有完整人类教程。

```python
        rows = ["机器学习基准：" + spec.name, "", "训练/验证/测试按日期隔离；固定模型，不选择测试冠军。", "", "|区间|模型|RankIC|Top相对池收益/期|相对小市值/期|", "|---|---|---:|---:|---:|"]
```

`L118` — 将验证、测试与测试年份的全部方法逐行写成指标表；不会只保留赢家。

```python
        for row in summaries:
```

`L119` — 将验证、测试与测试年份的全部方法逐行写成指标表；不会只保留赢家。

```python
            if row["scope"] != "train":
```

`L120` — 将验证、测试与测试年份的全部方法逐行写成指标表；不会只保留赢家。

```python
                rows.append(f"|{row['scope']}|{row['model']}|{row['rank_ic']:.4f}|{row['top_excess']:.4%}|{row['top_excess_vs_size']:.4%}|")
```

`L121` — 追加限制并保存摘要REPORT.md；新教程另写，不改冻结摘要。

```python
        rows += ["", *result["limitations"]]
```

`L122` — 追加限制并保存摘要REPORT.md；新教程另写，不改冻结摘要。

```python
        (folder / "REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
```

`L123` — 将状态更新为completed。

```python
        write_json(folder / "status.json", {"status": "completed", "run_id": run_id, "kind": "ml", "name": spec.name})
```

`L124` — 结果清单绑定快照、配置与代码版本，记录当前所有输出文件哈希。

```python
        write_json(folder / "manifest.json", {"kind": "ml", "run_id": run_id, "spec": spec.to_dict(),
```

`L125` — 结果清单绑定快照、配置与代码版本，记录当前所有输出文件哈希。

```python
            "snapshot_id": cfg["snapshot_id"], "snapshot_manifest_sha256": cfg["snapshot_manifest_hash"],
```

`L126` — 结果清单绑定快照、配置与代码版本，记录当前所有输出文件哈希。

```python
            "code_hash": cfg["engine_hash"], "artifacts": {p.name: digest(p) for p in folder.iterdir() if p.is_file()}})
```

`L127` — 返回结果目录给执行器继续进行audit。

```python
        return folder
```

`L128` — 捕获异常、记失败状态、继续抛出，保留失败尝试。

```python
    except BaseException as exc:
```

`L129` — 捕获异常、记失败状态、继续抛出，保留失败尝试。

```python
        write_json(folder / "status.json", {"status": "failed", "run_id": run_id, "kind": "ml", "error": str(exc)})
```

`L130` — 捕获异常、记失败状态、继续抛出，保留失败尝试。

```python
        raise
```

**src/mlresearch/evaluation.py · 全部新增源码**



`L1` — 模块文档字符串：说明代码范围，不执行数据处理或证明策略有效。

```python
"""Signal diagnostics, never an executable account or annualized return."""
```

`L2` — 导入：向量计算、有限值处理与统计抽样。

```python
import numpy as np
```

`L3` — 导入：行情面板、滚动窗口、分组及Parquet读取。

```python
import pandas as pd
```

`L4` — 空行：仅分隔代码段，无计算行为。

```python

```

`L5` — 固定两个模型加三种规则对照的评分列，所有方法均报告。

```python
SCORES = ["ridge", "hist_gbdt", "small_cap", "reversal_5", "fixed_random"]
```

`L6` — 空行：仅分隔代码段，无计算行为。

```python

```

`L7` — 空行：仅分隔代码段，无计算行为。

```python

```

`L8` — 块重抽样入口；转换为浮点并去掉无效诊断值。

```python
def block_interval(values, seed, block=4, repetitions=400):
```

`L9` — 块重抽样入口；转换为浮点并去掉无效诊断值。

```python
    values = np.asarray(values, dtype=float)
```

`L10` — 块重抽样入口；转换为浮点并去掉无效诊断值。

```python
    values = values[np.isfinite(values)]
```

`L11` — 至少两个完整块才给区间；样本过少返回空区间。

```python
    if len(values) < 2 * block:
```

`L12` — 至少两个完整块才给区间；样本过少返回空区间。

```python
        return [None, None]
```

`L13` — 固定随机种子，准备400次重抽样均值。

```python
    rng = np.random.default_rng(seed)
```

`L14` — 固定随机种子，准备400次重抽样均值。

```python
    draws = []
```

`L15` — 固定随机种子，准备400次重抽样均值。

```python
    for _ in range(repetitions):
```

`L16` — 随机抽足够数量的起点，后面每个起点扩成连续4个采样日期。

```python
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block)))
```

`L17` — 用模运算环接尾部，把最后几个日期之后接到开头。

```python
        indices = (starts[:, None] + np.arange(block)) % len(values)
```

`L18` — 铺平块并裁到原长度，记录这次均值。

```python
        draws.append(float(values[indices.ravel()[:len(values)]].mean()))
```

`L19` — 取均值分布2.5%与97.5%分位数；不是投资收益保证。

```python
    return np.quantile(draws, [0.025, 0.975]).tolist()
```

`L20` — 空行：仅分隔代码段，无计算行为。

```python

```

`L21` — 空行：仅分隔代码段，无计算行为。

```python

```

`L22` — 评价入口，按阶段和采样日分池，分别积累日期指标与十分组。

```python
def evaluate(predictions, spec):
```

`L23` — 评价入口，按阶段和采样日分池，分别积累日期指标与十分组。

```python
    daily, groups = [], []
```

`L24` — 评价入口，按阶段和采样日分池，分别积累日期指标与十分组。

```python
    for (split, day), pool in predictions.groupby(["split", "trade_date"], sort=True):
```

`L25` — 计算同日有标签候选的平均未来收益，NaN不填零。

```python
        actual = pool.label_return
```

`L26` — 计算同日有标签候选的平均未来收益，NaN不填零。

```python
        pool_mean = actual.mean()
```

`L27` — 逐方法按评分降序、代码升序排序；未知结果也先参与排序。

```python
        for model in SCORES:
```

`L28` — 逐方法按评分降序、代码升序排序；未知结果也先参与排序。

```python
            ordered = pool.sort_values([model, "stock_code"], ascending=[False, True]).copy()
```

`L29` — 按排序位置切成spec.bins组，1最高；不是按实际结果分组。

```python
            ordered["quantile"] = np.minimum(np.arange(len(ordered)) * spec.bins // len(ordered) + 1, spec.bins)
```

`L30` — 取已知标签标记，只用于相关和后续统计。

```python
            observed = pool.label_observed
```

`L31` — 分数排名与结果排名的相关即RankIC；常量评分无定义记NaN。

```python
            rank_ic = pool.loc[observed, model].rank().corr(pool.loc[observed, "label_return"].rank()) if pool.loc[observed, model].nunique() > 1 else np.nan
```

`L32` — 评分与结果值的普通相关为IC，常量评分同样不伪造0。

```python
            ic = pool.loc[observed, model].corr(pool.loc[observed, "label_return"]) if pool.loc[observed, model].nunique() > 1 else np.nan
```

`L33` — 先取Top K，再统计哪些入选有标签，不先删未知结果。

```python
            top = ordered.head(spec.top_k)
```

`L34` — 求各预测分组的已知未来收益平均。

```python
            by_bin = ordered.groupby("quantile").label_return.mean()
```

`L35` — 记录日期、池/入选数量与覆盖、Top均值、减池均值的相对收益、第一组减末组价差。

```python
            daily.append(dict(split=split, trade_date=day, model=model, rank_ic=rank_ic, ic=ic,
```

`L36` — 记录日期、池/入选数量与覆盖、Top均值、减池均值的相对收益、第一组减末组价差。

```python
                pool_count=len(pool), label_count=int(observed.sum()), pool_mean=pool_mean,
```

`L37` — 记录日期、池/入选数量与覆盖、Top均值、减池均值的相对收益、第一组减末组价差。

```python
                top_count=len(top), top_label_count=int(top.label_observed.sum()),
```

`L38` — 记录日期、池/入选数量与覆盖、Top均值、减池均值的相对收益、第一组减末组价差。

```python
                top_mean=top.label_return.mean(), top_excess=top.label_return.mean() - pool_mean,
```

`L39` — 记录日期、池/入选数量与覆盖、Top均值、减池均值的相对收益、第一组减末组价差。

```python
                spread=by_bin.get(1, np.nan) - by_bin.get(spec.bins, np.nan)))
```

`L40` — 保存每日期每方法每组数量、已知标签数、平均及相对池收益。

```python
            for quantile, group in ordered.groupby("quantile"):
```

`L41` — 保存每日期每方法每组数量、已知标签数、平均及相对池收益。

```python
                groups.append(dict(split=split, trade_date=day, model=model, quantile=int(quantile),
```

`L42` — 保存每日期每方法每组数量、已知标签数、平均及相对池收益。

```python
                    count=len(group), label_count=int(group.label_observed.sum()),
```

`L43` — 保存每日期每方法每组数量、已知标签数、平均及相对池收益。

```python
                    mean_return=group.label_return.mean(), excess=group.label_return.mean() - pool_mean))
```

`L44` — 把累积记录转成表，准备跨日期汇总。

```python
    daily = pd.DataFrame(daily)
```

`L45` — 把累积记录转成表，准备跨日期汇总。

```python
    groups = pd.DataFrame(groups)
```

`L46` — 把累积记录转成表，准备跨日期汇总。

```python
    summaries = []
```

`L47` — 建立train/valid/test三个整体时期。

```python
    scopes = [(split, split, daily[daily.split.eq(split)]) for split in ["train", "valid", "test"]]
```

`L48` — 另建立test各自然年份，用于观察时间变化。

```python
    scopes += [(f"test_{year}", "test", part) for year, part in daily[daily.split.eq("test")].groupby(daily.trade_date.str[:4])]
```

`L49` — 对每时期与方法逐个汇总，保持所有方法输出。

```python
    for scope, split, subset in scopes:
```

`L50` — 对每时期与方法逐个汇总，保持所有方法输出。

```python
        for model, part in subset.groupby("model"):
```

`L51` — 用日期对齐模型Top与小市值Top的相对收益，形成同日配对差。

```python
            small = subset[subset.model.eq("small_cap")].set_index("trade_date").top_excess
```

`L52` — 用日期对齐模型Top与小市值Top的相对收益，形成同日配对差。

```python
            delta = part.set_index("trade_date").top_excess - small
```

`L53` — MSE取相应阶段或测试年份的逐股预测行，阶段条件与指标一致。

```python
            prediction_rows = predictions[predictions.split.eq(split)]
```

`L54` — MSE取相应阶段或测试年份的逐股预测行，阶段条件与指标一致。

```python
            if scope.startswith("test_"):
```

`L55` — MSE取相应阶段或测试年份的逐股预测行，阶段条件与指标一致。

```python
                prediction_rows = prediction_rows[prediction_rows.trade_date.str.startswith(scope[-4:])]
```

`L56` — 预测误差只在有已知Y的行上算。

```python
            observed = prediction_rows[prediction_rows.label_observed]
```

`L57` — 只有两个输出收益单位的学习模型计算MSE；规则分数不冒充收益预测。

```python
            mse = np.mean((observed[model] - observed.label_relative) ** 2) if model in {"ridge", "hist_gbdt"} else None
```

`L58` — 所有相对收益都预测0时的MSE作为简单误差对照。

```python
            zero_mse = np.mean(observed.label_relative ** 2)
```

`L59` — 日期等权平均RankIC，统计有效日期中正RankIC比例。

```python
            summaries.append(dict(scope=scope, model=model, dates=len(part), rank_ic=part.rank_ic.mean(),
```

`L60` — 日期等权平均RankIC，统计有效日期中正RankIC比例。

```python
                rank_ic_positive_fraction=part.rank_ic.dropna().gt(0).mean(),
```

`L61` — RankIC的4日期块区间。

```python
                rank_ic_interval=block_interval(part.rank_ic, spec.seed),
```

`L62` — 日期等权Top相对收益、组间价差及Top相对收益区间。

```python
                top_excess=part.top_excess.mean(), spread=part.spread.mean(),
```

`L63` — 日期等权Top相对收益、组间价差及Top相对收益区间。

```python
                top_excess_interval=block_interval(part.top_excess, spec.seed),
```

`L64` — 日期等权相对小市值配对差及其块区间。

```python
                top_excess_vs_size=delta.mean(), size_delta_interval=block_interval(delta, spec.seed),
```

`L65` — 保存模型MSE、零预测MSE与相对改善1−MSE/zero；不是分类准确率。

```python
                mse=mse, zero_prediction_mse=zero_mse,
```

`L66` — 保存模型MSE、零预测MSE与相对改善1−MSE/zero；不是分类准确率。

```python
                r2_vs_zero=(1 - mse / zero_mse) if mse is not None and zero_mse > 0 else None,
```

`L67` — 按总候选/入选数量累计标签覆盖，分母不是只剩有标签者。

```python
                label_coverage=float(part.label_count.sum() / part.pool_count.sum()),
```

`L68` — 按总候选/入选数量累计标签覆盖，分母不是只剩有标签者。

```python
                top_label_coverage=float(part.top_label_count.sum() / part.top_count.sum())))
```

`L69` — 跨日期平均每阶段每方法每分组的均值；这里未输出分年十分组。

```python
    quantiles = groups.groupby(["split", "model", "quantile"])[["mean_return", "excess"]].mean().reset_index()
```

`L70` — 返回四套统计产物；没有复合净值和年化。

```python
    return daily, groups, summaries, quantiles
```

**src/mlresearch/audit.py · 全部新增源码**



`L1` — 模块文档字符串：说明代码范围，不执行数据处理或证明策略有效。

```python
"""Deterministic artifact/split/prediction checks, not independent investment review."""
```

`L2` — 导入：读写可追溯配置、收据与结果。

```python
import json
```

`L3` — 导入：安全明确的文件路径处理。

```python
from pathlib import Path
```

`L4` — 空行：仅分隔代码段，无计算行为。

```python

```

`L5` — 导入：保存与加载本系统已拟合Pipeline。

```python
import joblib
```

`L6` — 导入：向量计算、有限值处理与统计抽样。

```python
import numpy as np
```

`L7` — 导入：行情面板、滚动窗口、分组及Parquet读取。

```python
import pandas as pd
```

`L8` — 导入：限制本次数值计算线程数。

```python
from threadpoolctl import threadpool_limits
```

`L9` — 空行：仅分隔代码段，无计算行为。

```python

```

`L10` — 导入：复用哈希、完整性核验、JSON写入及数据序列化辅助函数。

```python
from ..technical.artifacts import digest, verify_artifacts
```

`L11` — 导入：ML配置、输入白名单与列说明。

```python
from .contracts import MLSpec, feature_names
```

`L12` — 导入：排序诊断与方法白名单。

```python
from .evaluation import SCORES
```

`L13` — 空行：仅分隔代码段，无计算行为。

```python

```

`L14` — 空行：仅分隔代码段，无计算行为。

```python

```

`L15` — 核查入口，读取结果清单并验证所有输出哈希；不重新训练模型。

```python
def audit(folder, root):
```

`L16` — 核查入口，读取结果清单并验证所有输出哈希；不重新训练模型。

```python
    folder, root = Path(folder), Path(root)
```

`L17` — 核查入口，读取结果清单并验证所有输出哈希；不重新训练模型。

```python
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
```

`L18` — 核查入口，读取结果清单并验证所有输出哈希；不重新训练模型。

```python
    verify_artifacts(folder, manifest)
```

`L19` — 定位本次快照，核对其清单哈希和行情等文件完整性。

```python
    snapshot = root / "snapshots" / manifest["snapshot_id"]
```

`L20` — 定位本次快照，核对其清单哈希和行情等文件完整性。

```python
    if digest(snapshot / "manifest.json") != manifest["snapshot_manifest_sha256"]:
```

`L21` — 定位本次快照，核对其清单哈希和行情等文件完整性。

```python
        raise ValueError("ML快照清单身份不一致")
```

`L22` — 定位本次快照，核对其清单哈希和行情等文件完整性。

```python
    verify_artifacts(snapshot, json.loads((snapshot / "manifest.json").read_text(encoding="utf-8")))
```

`L23` — 恢复配置，读取面板与保存预测。

```python
    spec = MLSpec.from_dict(manifest["spec"])
```

`L24` — 恢复配置，读取面板与保存预测。

```python
    panel = pd.read_parquet(folder / "panel.parquet")
```

`L25` — 恢复配置，读取面板与保存预测。

```python
    predicted = pd.read_parquet(folder / "predictions.parquet")
```

`L26` — 预测行身份、顺序与阶段必须和面板逐行一致。

```python
    if not panel[["stock_code", "trade_date", "split"]].equals(predicted[["stock_code", "trade_date", "split"]]):
```

`L27` — 预测行身份、顺序与阶段必须和面板逐行一致。

```python
        raise ValueError("预测与面板身份不一致")
```

`L28` — 拒绝重复股日主键，所有方法分数必须有限。

```python
    if panel.duplicated(["stock_code", "trade_date"]).any() or not np.isfinite(predicted[SCORES]).all().all():
```

`L29` — 拒绝重复股日主键，所有方法分数必须有限。

```python
        raise ValueError("ML主键或预测无效")
```

`L30` — 标签可观测标记必须精确等于标签是否有限。

```python
    if not panel.label_observed.equals(np.isfinite(panel.label_return)):
```

`L31` — 标签可观测标记必须精确等于标签是否有限。

```python
        raise ValueError("标签可观测标记不一致")
```

`L32` — 四个标签字段在面板与预测文件中必须相同。

```python
    for key in ["label_end", "label_observed", "label_return", "label_relative"]:
```

`L33` — 四个标签字段在面板与预测文件中必须相同。

```python
        if not panel[key].equals(predicted[key]):
```

`L34` — 四个标签字段在面板与预测文件中必须相同。

```python
            raise ValueError("预测文件的标签与面板不同")
```

`L35` — 各阶段非空，采样日和标签结束日均在边界内，标签结束必须晚于采样日。

```python
    for split in ["train", "valid", "test"]:
```

`L36` — 各阶段非空，采样日和标签结束日均在边界内，标签结束必须晚于采样日。

```python
        p = panel[panel.split.eq(split)]
```

`L37` — 各阶段非空，采样日和标签结束日均在边界内，标签结束必须晚于采样日。

```python
        if p.empty or not p.trade_date.between(getattr(spec, split + "_start"), getattr(spec, split + "_end")).all() or not p.label_end.le(getattr(spec, split + "_end")).all() or not p.label_end.gt(p.trade_date).all():
```

`L38` — 各阶段非空，采样日和标签结束日均在边界内，标签结束必须晚于采样日。

```python
            raise ValueError("日期切分或标签成熟边界无效")
```

`L39` — 资格所用公告必须严格早于采样日。

```python
    if not panel.q_publication_date.lt(panel.trade_date).all():
```

`L40` — 资格所用公告必须严格早于采样日。

```python
        raise ValueError("资格股本公告边界无效")
```

`L41` — 定位本次训练行及明确输入白名单。

```python
    train = panel[panel.split.eq("train") & panel.label_observed]
```

`L42` — 定位本次训练行及明确输入白名单。

```python
    names = feature_names(spec)
```

`L43` — 固定种子抽500行复核模型，前100行另复核原始未来标签。

```python
    sample = panel.sample(n=min(500, len(panel)), random_state=spec.seed)
```

`L44` — 固定种子抽500行复核模型，前100行另复核原始未来标签。

```python
    label_sample = sample.head(100)
```

`L45` — 只读取标签抽样所涉股票的原始收盘与参考前收，并按股日索引。

```python
    raw = pd.read_parquet(snapshot / "bars.parquet", columns=["stock_code", "trade_date", "close", "pre_close"],
```

`L46` — 只读取标签抽样所涉股票的原始收盘与参考前收，并按股日索引。

```python
        filters=[("stock_code", "in", label_sample.stock_code.unique().tolist())]).set_index(["stock_code", "trade_date"])
```

`L47` — 读市场日历并建立日期位置，初始化最大标签误差。

```python
    calendar = json.loads((snapshot / "calendar.json").read_text(encoding="utf-8"))
```

`L48` — 读市场日历并建立日期位置，初始化最大标签误差。

```python
    positions = {day: i for i, day in enumerate(calendar)}
```

`L49` — 读市场日历并建立日期位置，初始化最大标签误差。

```python
    label_error = 0.0
```

`L50` — 逐抽样行，取采样日之后的h个市场日期。

```python
    for row in label_sample.itertuples():
```

`L51` — 逐抽样行，取采样日之后的h个市场日期。

```python
        i = positions[row.trade_date]
```

`L52` — 逐抽样行，取采样日之后的h个市场日期。

```python
        dates = calendar[i + 1:i + spec.horizon + 1]
```

`L53` — h日齐全且最后日期等于保存label_end才继续。

```python
        if len(dates) != spec.horizon or dates[-1] != row.label_end:
```

`L54` — h日齐全且最后日期等于保存label_end才继续。

```python
            raise ValueError("标签末日不是市场日历目标")
```

`L55` — 按计划市场日期重索引原始报价，缺日必须表现为缺失。

```python
        quotes = raw.reindex(pd.MultiIndex.from_tuples([(row.stock_code, d) for d in dates]))
```

`L56` — 直接连乘未来close/pre_close重算收益；任一无效报价则未知。

```python
        ratio = quotes.close / quotes.pre_close
```

`L57` — 直接连乘未来close/pre_close重算收益；任一无效报价则未知。

```python
        expected = float(np.prod(ratio) - 1) if ratio.notna().all() and np.isfinite(ratio).all() and ratio.gt(0).all() else np.nan
```

`L58` — 和保存标签在1e−10误差内相符，NaN对NaN允许。

```python
        if not np.allclose(expected, row.label_return, atol=1e-10, equal_nan=True):
```

`L59` — 和保存标签在1e−10误差内相符，NaN对NaN允许。

```python
            raise ValueError("市场日历寻址的未来收益复算不同")
```

`L60` — 跟踪已知标签最大绝对复算误差。

```python
        if np.isfinite(expected):
```

`L61` — 跟踪已知标签最大绝对复算误差。

```python
            label_error = max(label_error, abs(expected - row.label_return))
```

`L62` — 开始逐个学习模型的保存预测核查。

```python
    errors = {}
```

`L63` — 开始逐个学习模型的保存预测核查。

```python
    for name in ["ridge", "hist_gbdt"]:
```

`L64` — 只加载本系统自己产生且前面哈希验证过的joblib，不加载任意用户模型。

```python
        # Only our own hash-verified local model is loaded, never user uploads.
```

`L65` — 只加载本系统自己产生且前面哈希验证过的joblib，不加载任意用户模型。

```python
        model = joblib.load(folder / f"{name}.joblib")
```

`L66` — 填补器保存的中位数必须等于本次train有标签行的列中位数。

```python
        if not np.allclose(model.steps[0][1].statistics_, train[names].median().to_numpy(), rtol=1e-12, atol=1e-12):
```

`L67` — 填补器保存的中位数必须等于本次train有标签行的列中位数。

```python
            raise ValueError("预处理填补统计并非训练段")
```

`L68` — 使用保存Pipeline对抽样X重新predict，限制4线程。

```python
        with threadpool_limits(limits=4):
```

`L69` — 使用保存Pipeline对抽样X重新predict，限制4线程。

```python
            scores = model.predict(sample[names])
```

`L70` — 比较预测表同一行分数，记录误差，超过1e−10则失败。

```python
        error = float(np.max(np.abs(scores - predicted.loc[sample.index, name].to_numpy())))
```

`L71` — 比较预测表同一行分数，记录误差，超过1e−10则失败。

```python
        errors[name] = error
```

`L72` — 比较预测表同一行分数，记录误差，超过1e−10则失败。

```python
        if error > 1e-10:
```

`L73` — 比较预测表同一行分数，记录误差，超过1e−10则失败。

```python
            raise ValueError("保存模型无法复现预测抽样")
```

`L74` — 读取日期指标，初始化RankIC最大误差。

```python
    metrics = pd.read_parquet(folder / "date_metrics.parquet")
```

`L75` — 读取日期指标，初始化RankIC最大误差。

```python
    metric_error = 0.0
```

`L76` — 逐采样日逐方法复算核心排序诊断，不是只抽几个好看的日期。

```python
    for (_, day), p in predicted.groupby(["split", "trade_date"]):
```

`L77` — 逐采样日逐方法复算核心排序诊断，不是只抽几个好看的日期。

```python
        for name in SCORES:
```

`L78` — 对已知标签取评分/结果排名，直接用numpy相关重算RankIC；常量保持未知。

```python
            observed = p[p.label_observed]
```

`L79` — 对已知标签取评分/结果排名，直接用numpy相关重算RankIC；常量保持未知。

```python
            a, b = observed[name].rank().to_numpy(), observed.label_return.rank().to_numpy()
```

`L80` — 对已知标签取评分/结果排名，直接用numpy相关重算RankIC；常量保持未知。

```python
            expected_ic = np.corrcoef(a, b)[0, 1] if np.std(a) > 0 and np.std(b) > 0 else np.nan
```

`L81` — 每方法每日期必须恰好有一条指标记录。

```python
            row = metrics[metrics.trade_date.eq(day) & metrics.model.eq(name)]
```

`L82` — 每方法每日期必须恰好有一条指标记录。

```python
            if len(row) != 1:
```

`L83` — 每方法每日期必须恰好有一条指标记录。

```python
                raise ValueError("日期指标记录缺失或重复")
```

`L84` — 重新按分数及代码取TopK，计算其已知平均减整个池平均。

```python
            actual_ic = row.iloc[0].rank_ic
```

`L85` — 重新按分数及代码取TopK，计算其已知平均减整个池平均。

```python
            top = p.sort_values([name, "stock_code"], ascending=[False, True]).head(spec.top_k)
```

`L86` — 重新按分数及代码取TopK，计算其已知平均减整个池平均。

```python
            excess = top.label_return.mean() - p.label_return.mean()
```

`L87` — 保存的RankIC与Top相对收益必须匹配，容差1e−12。

```python
            if not np.allclose([expected_ic, excess], [actual_ic, row.iloc[0].top_excess], atol=1e-12, equal_nan=True):
```

`L88` — 保存的RankIC与Top相对收益必须匹配，容差1e−12。

```python
                raise ValueError("排序或Top收益指标数值不一致")
```

`L89` — 记录有效日期RankIC最大绝对误差。

```python
            if np.isfinite(expected_ic):
```

`L90` — 记录有效日期RankIC最大绝对误差。

```python
                metric_error = max(metric_error, abs(float(expected_ic - actual_ic)))
```

`L91` — 读取汇总，对整体/分年正确筛选日期指标。

```python
    result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
```

`L92` — 读取汇总，对整体/分年正确筛选日期指标。

```python
    for summary in result["summaries"]:
```

`L93` — 读取汇总，对整体/分年正确筛选日期指标。

```python
        part = metrics[metrics.model.eq(summary["model"])]
```

`L94` — 读取汇总，对整体/分年正确筛选日期指标。

```python
        part = part[part.split.eq(summary["scope"])] if summary["scope"] in {"train", "valid", "test"} else part[part.split.eq("test") & part.trade_date.str.startswith(summary["scope"][-4:])]
```

`L95` — 汇总RankIC与Top均值必须匹配逐日期平均。

```python
        if not np.allclose([part.rank_ic.mean(), part.top_excess.mean()], np.asarray([summary["rank_ic"], summary["top_excess"]], dtype=float), atol=1e-12, equal_nan=True):
```

`L96` — 汇总RankIC与Top均值必须匹配逐日期平均。

```python
            raise ValueError("汇总指标不一致")
```

`L97` — 返回verified、样本数和误差，明确范围只是身份/边界/抽样/排序复算，不是账户或独立投资审查。

```python
    return {"kind": "ml", "run_id": folder.name, "verified": True, "rows": len(panel),
```

`L98` — 返回verified、样本数和误差，明确范围只是身份/边界/抽样/排序复算，不是账户或独立投资审查。

```python
        "prediction_sample_max_error": errors, "rank_ic_max_error": metric_error,
```

`L99` — 返回verified、样本数和误差，明确范围只是身份/边界/抽样/排序复算，不是账户或独立投资审查。

```python
        "label_sample_max_error": label_error,
```

`L100` — 返回verified、样本数和误差，明确范围只是身份/边界/抽样/排序复算，不是账户或独立投资审查。

```python
        "verification_scope": "文件与版本身份、时间切分、训练填补统计、市场日历标签抽样、保存模型预测抽样和排序指标复算；不是账户核查或独立研究复核"}
```

**src/researchops/ml_experiments.py · 全部新增源码**



`L1` — 模块文档字符串：说明代码范围，不执行数据处理或证明策略有效。

```python
"""ML jobs use the existing lease, budget, frozen executor and recovery protocol."""
```

`L2` — 导入：读写可追溯配置、收据与结果。

```python
import json
```

`L3` — 导入：记录并核查已安装包版本。

```python
import importlib.metadata
```

`L4` — 导入：记录Python及系统描述。

```python
import platform
```

`L5` — 这里导入Path但主模块没有使用它；执行器字符串中的Path由字符串内另行导入，是一个冗余导入。

```python
from pathlib import Path
```

`L6` — 导入：严格编号格式与测试页面元数据提取。

```python
import re
```

`L7` — 导入：冻结源码复制或受约束的自有QA目录清理。

```python
import shutil
```

`L8` — 导入：计时、恢复事件时间或有期限验收等待。

```python
import time
```

`L9` — 空行：仅分隔代码段，无计算行为。

```python

```

`L10` — 导入：登记必填文本校验。

```python
from .store import text
```

`L11` — 导入：复用哈希、完整性核验、JSON写入及数据序列化辅助函数。

```python
from ..technical.artifacts import content_id, digest, verify_artifacts, write_json
```

`L12` — 导入：ML配置、白名单与固定设计。

```python
from ..mlresearch.contracts import MLSpec
```

`L13` — 空行：仅分隔代码段，无计算行为。

```python

```

`L14` — 定义登记时写入冻结目录的独立执行器字符串；这段字符串以后作为Python程序运行。

```python
EXECUTOR = '''from pathlib import Path
```

`L15` — 执行器导入依赖，把冻结code目录放在模块搜索最前，再导入冻结runner/audit。

```python
import json, sys, traceback
```

`L16` — 执行器导入依赖，把冻结code目录放在模块搜索最前，再导入冻结runner/audit。

```python
code=Path(__file__).resolve().parent
```

`L17` — 执行器导入依赖，把冻结code目录放在模块搜索最前，再导入冻结runner/audit。

```python
sys.path.insert(0,str(code))
```

`L18` — 执行器导入依赖，把冻结code目录放在模块搜索最前，再导入冻结runner/audit。

```python
from src.technical.artifacts import write_json, digest, verify_artifacts
```

`L19` — 执行器导入依赖，把冻结code目录放在模块搜索最前，再导入冻结runner/audit。

```python
from src.mlresearch.runner import run
```

`L20` — 执行器导入依赖，把冻结code目录放在模块搜索最前，再导入冻结runner/audit。

```python
from src.mlresearch.audit import audit
```

`L21` — 定位工作目录，读取登记input，并检查冻结代码文件哈希。

```python
job=code.parent
```

`L22` — 定位工作目录，读取登记input，并检查冻结代码文件哈希。

```python
try:
```

`L23` — 定位工作目录，读取登记input，并检查冻结代码文件哈希。

```python
    cfg=json.loads((job/'input.json').read_text(encoding='utf-8'))
```

`L24` — 定位工作目录，读取登记input，并检查冻结代码文件哈希。

```python
    verify_artifacts(code,json.loads((job/'code_manifest.json').read_text(encoding='utf-8')))
```

`L25` — 执行器调用runner.run跑特征、训练、预测及统计，进度写日志。

```python
    folder=run(cfg,progress=lambda m:print(m,flush=True))
```

`L26` — 结果运行audit，保存核查报告和完成收据；收据关联run_id与audit哈希。

```python
    checked=audit(folder,cfg['root'])
```

`L27` — 结果运行audit，保存核查报告和完成收据；收据关联run_id与audit哈希。

```python
    write_json(job/'ml_audit.json',checked)
```

`L28` — 结果运行audit，保存核查报告和完成收据；收据关联run_id与audit哈希。

```python
    write_json(job/'receipt.json',{'status':'completed','kind':'ml','run_id':folder.name,'audit_hash':digest(job/'ml_audit.json')})
```

`L29` — 失败也写收据、输出堆栈并以非0退出，保留失败事实。

```python
except BaseException as exc:
```

`L30` — 失败也写收据、输出堆栈并以非0退出，保留失败事实。

```python
    write_json(job/'receipt.json',{'status':'failed','kind':'ml','error':str(exc)})
```

`L31` — 失败也写收据、输出堆栈并以非0退出，保留失败事实。

```python
    traceback.print_exc()
```

`L32` — 失败也写收据、输出堆栈并以非0退出，保留失败事实。

```python
    sys.exit(1)
```

`L33` — 结束执行器字符串；登记函数本身仍在主进程中。

```python
'''
```

`L34` — 空行：仅分隔代码段，无计算行为。

```python

```

`L35` — 空行：仅分隔代码段，无计算行为。

```python

```

`L36` — 登记入口，严格验证proposal字段、kind及必填研究假设；不接收账户费用情景。

```python
def register(research, session, proposal):
```

`L37` — 登记入口，严格验证proposal字段、kind及必填研究假设；不接收账户费用情景。

```python
    required = {"kind", "hypothesis", "expected_observation", "falsification", "snapshot_id", "spec"}
```

`L38` — 登记入口，严格验证proposal字段、kind及必填研究假设；不接收账户费用情景。

```python
    if not required <= proposal.keys() or proposal.keys() - (required | {"timeout_seconds"}) or proposal["kind"] != "ml":
```

`L39` — 登记入口，严格验证proposal字段、kind及必填研究假设；不接收账户费用情景。

```python
        raise ValueError("ML登记字段无效；不接受账户成本情景")
```

`L40` — 用已有租约机制确认当前session确实持有任务。

```python
    with research.store.connection() as db:
```

`L41` — 用已有租约机制确认当前session确实持有任务。

```python
        research.store.owned(db, session)
```

`L42` — 快照编号必须24位十六进制，禁止任意路径输入。

```python
    sid = proposal["snapshot_id"]
```

`L43` — 快照编号必须24位十六进制，禁止任意路径输入。

```python
    if not isinstance(sid, str) or not re.fullmatch(r"[0-9a-f]{24}", sid):
```

`L44` — 快照编号必须24位十六进制，禁止任意路径输入。

```python
        raise ValueError("ML快照编号无效")
```

`L45` — 读取指定快照并核对全部清单文件。

```python
    snapshot = research.root / "snapshots" / sid
```

`L46` — 读取指定快照并核对全部清单文件。

```python
    sm = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
```

`L47` — 读取指定快照并核对全部清单文件。

```python
    verify_artifacts(snapshot, sm)
```

`L48` — 验证配置、日期覆盖和股本数据存在，防止空数据或范围外研究。

```python
    spec = MLSpec.from_dict(proposal["spec"])
```

`L49` — 验证配置、日期覆盖和股本数据存在，防止空数据或范围外研究。

```python
    if spec.train_start < sm["start"] or spec.test_end > sm["end"] or "fundamentals" not in sm["counts"]:
```

`L50` — 验证配置、日期覆盖和股本数据存在，防止空数据或范围外研究。

```python
        raise ValueError("ML基准需要覆盖全部日期、含历史股本的快照")
```

`L51` — 执行时限默认1200秒，只接受30—7200秒。

```python
    timeout = proposal.get("timeout_seconds", 1200)
```

`L52` — 执行时限默认1200秒，只接受30—7200秒。

```python
    if type(timeout) is not int or not 30 <= timeout <= 7200:
```

`L53` — 执行时限默认1200秒，只接受30—7200秒。

```python
        raise ValueError("ML时限必须为30至7200秒")
```

`L54` — 冻结全部ML模块与四个已复用技术文件；不复制整份大行情。

```python
    sources = sorted((research.project / "src/mlresearch").glob("*.py"))
```

`L55` — 冻结全部ML模块与四个已复用技术文件；不复制整份大行情。

```python
    sources += [research.project / "src/technical" / name for name in ["__init__.py", "artifacts.py", "foundation_data.py", "signals.py"]]
```

`L56` — 把每个源码相对路径及哈希组成登记代码清单。

```python
    inventory = {path.relative_to(research.project).as_posix(): digest(path) for path in sources}
```

`L57` — runner必须存在，不能登记一个无执行器的案例。

```python
    if not (research.project / "src/mlresearch/runner.py").exists():
```

`L58` — runner必须存在，不能登记一个无执行器的案例。

```python
        raise ValueError("当前项目缺少ML执行代码")
```

`L59` — 假设、预期、否证条件必须是有效文本。

```python
    p = {key: text(proposal[key], key) for key in ["hypothesis", "expected_observation", "falsification"]}
```

`L60` — 登记完整定义、代码/快照/执行器身份、回顾性范围、Worker及Python/依赖版本。

```python
    p.update(kind="ml", spec=spec.to_dict(), snapshot_id=sid, snapshot_manifest_hash=digest(snapshot / "manifest.json"),
```

`L61` — 登记完整定义、代码/快照/执行器身份、回顾性范围、Worker及Python/依赖版本。

```python
        timeout_seconds=timeout, engine_inventory=inventory, engine_hash=content_id(inventory),
```

`L62` — 登记完整定义、代码/快照/执行器身份、回顾性范围、Worker及Python/依赖版本。

```python
        executor_hash=content_id(EXECUTOR), evaluation_scope="retrospective_time_holdout", submitted_by=session["worker"])
```

`L63` — 登记完整定义、代码/快照/执行器身份、回顾性范围、Worker及Python/依赖版本。

```python
    p["environment"] = {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "joblib", "threadpoolctl"]}}
```

`L64` — 建立去重定义，去掉名称和描述等无实质变化项；不以改名制造新试验。

```python
    canonical = json.loads(json.dumps(p))
```

`L65` — 建立去重定义，去掉名称和描述等无实质变化项；不以改名制造新试验。

```python
    canonical["spec"].pop("name")
```

`L66` — 建立去重定义，去掉名称和描述等无实质变化项；不以改名制造新试验。

```python
    for key in ["submitted_by", "hypothesis", "expected_observation", "falsification", "timeout_seconds"]:
```

`L67` — 建立去重定义，去掉名称和描述等无实质变化项；不以改名制造新试验。

```python
        canonical.pop(key)
```

`L68` — 已有Store核验预算与所有权，并登记或返回同实质实验。

```python
    experiment = research.store.register(session, p, content_id(canonical))
```

`L69` — 若冻结已ready则复用，不再次跑或覆盖。

```python
    job = research.root / "worker_jobs" / experiment["id"]
```

`L70` — 若冻结已ready则复用，不再次跑或覆盖。

```python
    if (job / "ready.json").exists():
```

`L71` — 若冻结已ready则复用，不再次跑或覆盖。

```python
        return experiment
```

`L72` — 存在未完成冻结目录也拒绝覆盖，失败尝试需保留。

```python
    if job.exists():
```

`L73` — 存在未完成冻结目录也拒绝覆盖，失败尝试需保留。

```python
        raise ValueError("ML冻结尝试已存在，不能覆盖")
```

`L74` — 采用登记返回的proposal，创建独立code目录。

```python
    p = experiment["proposal"]
```

`L75` — 采用登记返回的proposal，创建独立code目录。

```python
    code = job / "code"
```

`L76` — 采用登记返回的proposal，创建独立code目录。

```python
    code.mkdir(parents=True)
```

`L77` — 开始冻结保护，写入src包初始化文件。

```python
    try:
```

`L78` — 开始冻结保护，写入src包初始化文件。

```python
        (code / "src").mkdir()
```

`L79` — 开始冻结保护，写入src包初始化文件。

```python
        (code / "src/__init__.py").write_text("", encoding="utf-8")
```

`L80` — 逐源文件复制到对应路径并马上核哈希，防止复制过程中开发文件改变。

```python
        for name, sha in p["engine_inventory"].items():
```

`L81` — 逐源文件复制到对应路径并马上核哈希，防止复制过程中开发文件改变。

```python
            target = code / name
```

`L82` — 逐源文件复制到对应路径并马上核哈希，防止复制过程中开发文件改变。

```python
            target.parent.mkdir(parents=True, exist_ok=True)
```

`L83` — 逐源文件复制到对应路径并马上核哈希，防止复制过程中开发文件改变。

```python
            shutil.copy2(research.project / name, target)
```

`L84` — 逐源文件复制到对应路径并马上核哈希，防止复制过程中开发文件改变。

```python
            if digest(target) != sha:
```

`L85` — 逐源文件复制到对应路径并马上核哈希，防止复制过程中开发文件改变。

```python
                raise ValueError("ML代码在冻结期间变化")
```

`L86` — 把上述执行器写入冻结execute.py。

```python
        (code / "execute.py").write_text(EXECUTOR, encoding="utf-8")
```

`L87` — 记录冻结执行目录每个文件哈希。

```python
        write_json(job / "code_manifest.json", {"artifacts": {v.relative_to(code).as_posix(): digest(v) for v in code.rglob("*") if v.is_file()}})
```

`L88` — 冻结输入，明确共享root、项目、任务和实验归属。

```python
        write_json(job / "input.json", dict(p, project=str(research.project), root=str(research.root), task_id=session["task_id"], experiment_id=experiment["id"]))
```

`L89` — ready.json绑定input和code_manifest哈希，之后变动会拒绝执行。

```python
        write_json(job / "ready.json", {"input_hash": digest(job / "input.json"), "code_manifest_hash": digest(job / "code_manifest.json")})
```

`L90` — 冻结失败也在数据库记failed与事件并重新抛出，不擦除尝试。

```python
    except Exception as exc:
```

`L91` — 冻结失败也在数据库记failed与事件并重新抛出，不擦除尝试。

```python
        with research.store.connection(True) as db:
```

`L92` — 冻结失败也在数据库记failed与事件并重新抛出，不擦除尝试。

```python
            db.execute("UPDATE experiments SET status='failed',error=?,updated=? WHERE id=?", (str(exc), time.time(), experiment["id"]))
```

`L93` — 冻结失败也在数据库记failed与事件并重新抛出，不擦除尝试。

```python
            research.store.event(db, session["task_id"], session["worker"], "freeze_failed", {"experiment_id": experiment["id"], "error": str(exc)})
```

`L94` — 冻结失败也在数据库记failed与事件并重新抛出，不擦除尝试。

```python
        raise
```

`L95` — 返回登记实验；这里还没有训练，run才执行。

```python
    return experiment
```

`L96` — 空行：仅分隔代码段，无计算行为。

```python

```

`L97` — 空行：仅分隔代码段，无计算行为。

```python

```

`L98` — 核验已完成实验的冻结输入和代码清单未变。

```python
def verify_completed(research, experiment):
```

`L99` — 核验已完成实验的冻结输入和代码清单未变。

```python
    job = research.root / "worker_jobs" / experiment["id"]
```

`L100` — 核验已完成实验的冻结输入和代码清单未变。

```python
    ready = json.loads((job / "ready.json").read_text(encoding="utf-8"))
```

`L101` — 核验已完成实验的冻结输入和代码清单未变。

```python
    if digest(job / "input.json") != ready["input_hash"] or digest(job / "code_manifest.json") != ready["code_manifest_hash"]:
```

`L102` — 实际源码、input与登记proposal、任务及实验归属均须一致。

```python
        raise ValueError("ML冻结输入变化")
```

`L103` — 实际源码、input与登记proposal、任务及实验归属均须一致。

```python
    verify_artifacts(job / "code", json.loads((job / "code_manifest.json").read_text(encoding="utf-8")))
```

`L104` — 实际源码、input与登记proposal、任务及实验归属均须一致。

```python
    p = experiment["proposal"]
```

`L105` — 实际源码、input与登记proposal、任务及实验归属均须一致。

```python
    cfg = json.loads((job / "input.json").read_text(encoding="utf-8"))
```

`L106` — 实际源码、input与登记proposal、任务及实验归属均须一致。

```python
    if any(cfg.get(k) != v for k, v in p.items()) or cfg["experiment_id"] != experiment["id"] or cfg["task_id"] != experiment["task_id"]:
```

`L107` — 运行清单、全部输出、模型代码版本、配置、快照及快照清单身份须符合登记。

```python
        raise ValueError("ML冻结输入与登记不同")
```

`L108` — 运行清单、全部输出、模型代码版本、配置、快照及快照清单身份须符合登记。

```python
    folder = research.root / "ml_runs" / experiment["run_id"]
```

`L109` — 运行清单、全部输出、模型代码版本、配置、快照及快照清单身份须符合登记。

```python
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
```

`L110` — 运行清单、全部输出、模型代码版本、配置、快照及快照清单身份须符合登记。

```python
    verify_artifacts(folder, manifest)
```

`L111` — 运行清单、全部输出、模型代码版本、配置、快照及快照清单身份须符合登记。

```python
    if manifest["code_hash"] != p["engine_hash"] or manifest["spec"] != p["spec"] or manifest["snapshot_id"] != p["snapshot_id"] or manifest["snapshot_manifest_sha256"] != p["snapshot_manifest_hash"]:
```

`L112` — 再核验原快照身份和文件，不能换源数据后沿用老结果。

```python
        raise ValueError("ML登记版本不一致")
```

`L113` — 再核验原快照身份和文件，不能换源数据后沿用老结果。

```python
    snapshot = research.root / "snapshots" / p["snapshot_id"]
```

`L114` — 再核验原快照身份和文件，不能换源数据后沿用老结果。

```python
    if digest(snapshot / "manifest.json") != p["snapshot_manifest_hash"]:
```

`L115` — 再核验原快照身份和文件，不能换源数据后沿用老结果。

```python
        raise ValueError("ML快照清单变化")
```

`L116` — 再核验原快照身份和文件，不能换源数据后沿用老结果。

```python
    verify_artifacts(snapshot, json.loads((snapshot / "manifest.json").read_text(encoding="utf-8")))
```

`L117` — 保存audit文件哈希须与登记完成记录一致；完整性复核，不重新做投资研究。

```python
    if digest(job / "ml_audit.json") != experiment["result"]["audit_hash"]:
```

`L118` — 保存audit文件哈希须与登记完成记录一致；完整性复核，不重新做投资研究。

```python
        raise ValueError("ML核查证据变化")
```

`L119` — 空行：仅分隔代码段，无计算行为。

```python

```

`L120` — 空行：仅分隔代码段，无计算行为。

```python

```

`L121` — 根据子进程收据恢复实验状态，定位工作目录。

```python
def recover(research, experiment, receipt):
```

`L122` — 根据子进程收据恢复实验状态，定位工作目录。

```python
    eid = experiment["id"]
```

`L123` — 根据子进程收据恢复实验状态，定位工作目录。

```python
    job = research.root / "worker_jobs" / eid
```

`L124` — 根据子进程收据恢复实验状态，定位工作目录。

```python
    try:
```

`L125` — run_id格式必须有效，避免通过路径定位任意目录。

```python
        rid = receipt["run_id"]
```

`L126` — run_id格式必须有效，避免通过路径定位任意目录。

```python
        if not isinstance(rid, str) or not re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{8}", rid):
```

`L127` — run_id格式必须有效，避免通过路径定位任意目录。

```python
            raise ValueError("ML运行编号无效")
```

`L128` — 恢复前验证冻结input、代码清单和各源码文件。

```python
        ready = json.loads((job / "ready.json").read_text(encoding="utf-8"))
```

`L129` — 恢复前验证冻结input、代码清单和各源码文件。

```python
        if digest(job / "input.json") != ready["input_hash"] or digest(job / "code_manifest.json") != ready["code_manifest_hash"]:
```

`L130` — 恢复前验证冻结input、代码清单和各源码文件。

```python
            raise ValueError("ML冻结输入已变化")
```

`L131` — 恢复前验证冻结input、代码清单和各源码文件。

```python
        verify_artifacts(job / "code", json.loads((job / "code_manifest.json").read_text(encoding="utf-8")))
```

`L132` — 定位run并核验输出、模型代码、配置、快照和登记身份一致。

```python
        folder = research.root / "ml_runs" / rid
```

`L133` — 定位run并核验输出、模型代码、配置、快照和登记身份一致。

```python
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
```

`L134` — 定位run并核验输出、模型代码、配置、快照和登记身份一致。

```python
        verify_artifacts(folder, manifest)
```

`L135` — 定位run并核验输出、模型代码、配置、快照和登记身份一致。

```python
        p = experiment["proposal"]
```

`L136` — 定位run并核验输出、模型代码、配置、快照和登记身份一致。

```python
        if manifest["code_hash"] != p["engine_hash"] or manifest["spec"] != p["spec"] or manifest["snapshot_id"] != p["snapshot_id"] or manifest["snapshot_manifest_sha256"] != p["snapshot_manifest_hash"]:
```

`L137` — 定位run并核验输出、模型代码、配置、快照和登记身份一致。

```python
            raise ValueError("ML运行与登记版本不一致")
```

`L138` — 恢复前再验证引用的数据快照未改变。

```python
        snapshot = research.root / "snapshots" / p["snapshot_id"]
```

`L139` — 恢复前再验证引用的数据快照未改变。

```python
        if digest(snapshot / "manifest.json") != p["snapshot_manifest_hash"]:
```

`L140` — 恢复前再验证引用的数据快照未改变。

```python
            raise ValueError("ML快照清单变化")
```

`L141` — 恢复前再验证引用的数据快照未改变。

```python
        verify_artifacts(snapshot, json.loads((snapshot / "manifest.json").read_text(encoding="utf-8")))
```

`L142` — audit必须与收据哈希一致。

```python
        if digest(job / "ml_audit.json") != receipt["audit_hash"]:
```

`L143` — audit必须与收据哈希一致。

```python
            raise ValueError("ML核查证据变化")
```

`L144` — 读核查与结果，要求verified且run/task/experiment归属完全一致。

```python
        checked = json.loads((job / "ml_audit.json").read_text(encoding="utf-8"))
```

`L145` — 读核查与结果，要求verified且run/task/experiment归属完全一致。

```python
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
```

`L146` — 读核查与结果，要求verified且run/task/experiment归属完全一致。

```python
        if not checked.get("verified") or checked["run_id"] != rid or result["research_context"] != {"task_id": experiment["task_id"], "experiment_id": eid}:
```

`L147` — 读核查与结果，要求verified且run/task/experiment归属完全一致。

```python
            raise ValueError("ML结果归属或核查无效")
```

`L148` — Store完成登记，记录kind=ml、指标、样本、核查范围和manifest/audit身份。

```python
        return research.store.finish(eid, "completed", run_id=rid, result={"kind": "ml", "verified": True,
```

`L149` — Store完成登记，记录kind=ml、指标、样本、核查范围和manifest/audit身份。

```python
            "name": result["name"], "summaries": result["summaries"], "counts": result["counts"],
```

`L150` — Store完成登记，记录kind=ml、指标、样本、核查范围和manifest/audit身份。

```python
            "verification_scope": checked["verification_scope"], "manifest_hash": digest(folder / "manifest.json"),
```

`L151` — Store完成登记，记录kind=ml、指标、样本、核查范围和manifest/audit身份。

```python
            "audit_hash": receipt["audit_hash"], "audit": checked})
```

`L152` — 恢复核验失败则实验记failed，异常继续抛出。

```python
    except Exception as exc:
```

`L153` — 恢复核验失败则实验记failed，异常继续抛出。

```python
        research.store.finish(eid, "failed", error=str(exc))
```

`L154` — 恢复核验失败则实验记failed，异常继续抛出。

```python
        raise
```

**tools/ml_research.py · 全部新增源码**



`L1` — 模块文档字符串：说明代码范围，不执行数据处理或证明策略有效。

```python
"""Inspect registered ML results; execution stays on research.py register/run."""
```

`L2` — 导入：命令行参数解析。

```python
import argparse
```

`L3` — 导入：读写可追溯配置、收据与结果。

```python
import json
```

`L4` — 导入：安全明确的文件路径处理。

```python
from pathlib import Path
```

`L5` — 导入：严格编号格式与测试页面元数据提取。

```python
import re
```

`L6` — 导入：设置导入目录、UTF-8输出与失败退出码。

```python
import sys
```

`L7` — 空行：仅分隔代码段，无计算行为。

```python

```

`L8` — 定位lhb_backtest项目并加入Python模块搜索路径，使CLI可导入src。

```python
PROJECT = Path(__file__).resolve().parents[1]
```

`L9` — 定位lhb_backtest项目并加入Python模块搜索路径，使CLI可导入src。

```python
sys.path.insert(0, str(PROJECT))
```

`L10` — 导入：ML配置、白名单与固定设计。

```python
from src.mlresearch.contracts import benchmark_proposal
```

`L11` — 导入：复用哈希、完整性核验、JSON写入及数据序列化辅助函数。

```python
from src.technical.artifacts import write_json
```

`L12` — 空行：仅分隔代码段，无计算行为。

```python

```

`L13` — 空行：仅分隔代码段，无计算行为。

```python

```

`L14` — 建立只读查询/设计CLI，root默认当前项目技术研究库；worktree可显式共享root。

```python
def main():
```

`L15` — 建立只读查询/设计CLI，root默认当前项目技术研究库；worktree可显式共享root。

```python
    parser = argparse.ArgumentParser(description="机器学习基准设计与结果")
```

`L16` — 建立只读查询/设计CLI，root默认当前项目技术研究库；worktree可显式共享root。

```python
    parser.add_argument("--root", type=Path, default=PROJECT / "data/technical")
```

`L17` — 要求子命令，不接受无意义的空调用。

```python
    commands = parser.add_subparsers(dest="command", required=True)
```

`L18` — plan可选快照并把proposal写入指定路径。

```python
    plan = commands.add_parser("plan")
```

`L19` — plan可选快照并把proposal写入指定路径。

```python
    plan.add_argument("--snapshot", default="31eef7349722ec89cd1a3743")
```

`L20` — plan可选快照并把proposal写入指定路径。

```python
    plan.add_argument("--output", type=Path)
```

`L21` — list列出已执行学习结果；不是启动训练。

```python
    commands.add_parser("list")
```

`L22` — show与verify都要明确run_id。

```python
    for name in ["show", "verify"]:
```

`L23` — show与verify都要明确run_id。

```python
        commands.add_parser(name).add_argument("run_id")
```

`L24` — 解析命令行参数。

```python
    args = parser.parse_args()
```

`L25` — plan产生固定基准proposal，尚未登记或训练。

```python
    if args.command == "plan":
```

`L26` — plan产生固定基准proposal，尚未登记或训练。

```python
        value = benchmark_proposal(args.snapshot)
```

`L27` — 如请求输出文件，已有路径拒绝覆盖，保护旧协议。

```python
        if args.output:
```

`L28` — 如请求输出文件，已有路径拒绝覆盖，保护旧协议。

```python
            if args.output.exists():
```

`L29` — 如请求输出文件，已有路径拒绝覆盖，保护旧协议。

```python
                raise ValueError("设计文件已存在，避免覆盖旧研究协议")
```

`L30` — 如请求输出文件，已有路径拒绝覆盖，保护旧协议。

```python
            write_json(args.output, value)
```

`L31` — list读取每个ML运行status并按目录逆序展示，失败也可能出现在列表。

```python
    elif args.command == "list":
```

`L32` — list读取每个ML运行status并按目录逆序展示，失败也可能出现在列表。

```python
        value = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((args.root / "ml_runs").glob("*/status.json"), reverse=True)]
```

`L33` — show/verify要求run_id格式合法，禁止路径穿越。

```python
    else:
```

`L34` — show/verify要求run_id格式合法，禁止路径穿越。

```python
        if not re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{8}", args.run_id):
```

`L35` — show/verify要求run_id格式合法，禁止路径穿越。

```python
            raise ValueError("ML运行编号无效")
```

`L36` — 定位该运行目录。

```python
        folder = args.root / "ml_runs" / args.run_id
```

`L37` — show只读取汇总JSON。

```python
        if args.command == "show":
```

`L38` — show只读取汇总JSON。

```python
            value = json.loads((folder / "result.json").read_text(encoding="utf-8"))
```

`L39` — verify显式调用数值核查器；与新训练不同。

```python
        else:
```

`L40` — verify显式调用数值核查器；与新训练不同。

```python
            from src.mlresearch.audit import audit
```

`L41` — verify显式调用数值核查器；与新训练不同。

```python
            value = audit(folder, args.root)
```

`L42` — 输出稳定JSON，拒绝非标准NaN值。

```python
    print(json.dumps({"ok": True, "data": value}, ensure_ascii=False, allow_nan=False))
```

`L43` — 空行：仅分隔代码段，无计算行为。

```python

```

`L44` — 空行：仅分隔代码段，无计算行为。

```python

```

`L45` — 直接运行脚本时统一UTF-8并调用main。

```python
if __name__ == "__main__":
```

`L46` — 直接运行脚本时统一UTF-8并调用main。

```python
    sys.stdout.reconfigure(encoding="utf-8")
```

`L47` — 直接运行脚本时统一UTF-8并调用main。

```python
    try:
```

`L48` — 直接运行脚本时统一UTF-8并调用main。

```python
        main()
```

`L49` — 失败输出ok:false和错误，非0退出供自动流程识别。

```python
    except Exception as exc:
```

`L50` — 失败输出ok:false和错误，非0退出供自动流程识别。

```python
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
```

`L51` — 失败输出ok:false和错误，非0退出供自动流程识别。

```python
        sys.exit(1)
```

**tools/verify_ml_ui.py · 全部新增源码**



`L1` — 模块文档字符串：说明代码范围，不执行数据处理或证明策略有效。

```python
"""Desktop/narrow-screen ML UI acceptance in an owned headless browser."""
```

`L2` — 导入：命令行参数解析。

```python
import argparse
```

`L3` — 导入：读写可追溯配置、收据与结果。

```python
import json
```

`L4` — 导入：安全明确的文件路径处理。

```python
from pathlib import Path
```

`L5` — 导入：冻结源码复制或受约束的自有QA目录清理。

```python
import shutil
```

`L6` — 导入：选择本次本地调试端口。

```python
import socket
```

`L7` — 导入：启动/结束本次自有验收浏览器。

```python
import subprocess
```

`L8` — 导入：设置导入目录、UTF-8输出与失败退出码。

```python
import sys
```

`L9` — 导入：运行本地服务或后台执行，不是多模型研究证据。

```python
import threading
```

`L10` — 导入：计时、恢复事件时间或有期限验收等待。

```python
import time
```

`L11` — 空行：仅分隔代码段，无计算行为。

```python

```

`L12` — 导入：访问本地浏览器调试端点。

```python
import requests
```

`L13` — 空行：仅分隔代码段，无计算行为。

```python

```

`L14` — 定位项目并导入已有本地服务、JSON写入和浏览器控制辅助类。

```python
PROJECT = Path(__file__).resolve().parents[1]
```

`L15` — 定位项目并导入已有本地服务、JSON写入和浏览器控制辅助类。

```python
sys.path.insert(0, str(PROJECT))
```

`L16` — 定位项目并导入已有本地服务、JSON写入和浏览器控制辅助类。

```python
from src.technical.server import make_server
```

`L17` — 定位项目并导入已有本地服务、JSON写入和浏览器控制辅助类。

```python
from src.technical.artifacts import write_json
```

`L18` — 定位项目并导入已有本地服务、JSON写入和浏览器控制辅助类。

```python
from verify_workbench_ui import Browser
```

`L19` — 空行：仅分隔代码段，无计算行为。

```python

```

`L20` — 空行：仅分隔代码段，无计算行为。

```python

```

`L21` — 解析待查看run、任务、全新输出目录、Chrome/Edge选择；工具不训练模型。

```python
def main():
```

`L22` — 解析待查看run、任务、全新输出目录、Chrome/Edge选择；工具不训练模型。

```python
    parser = argparse.ArgumentParser()
```

`L23` — 解析待查看run、任务、全新输出目录、Chrome/Edge选择；工具不训练模型。

```python
    parser.add_argument("run_id")
```

`L24` — 解析待查看run、任务、全新输出目录、Chrome/Edge选择；工具不训练模型。

```python
    parser.add_argument("--task-id", required=True)
```

`L25` — 解析待查看run、任务、全新输出目录、Chrome/Edge选择；工具不训练模型。

```python
    parser.add_argument("--output", type=Path, required=True)
```

`L26` — 解析待查看run、任务、全新输出目录、Chrome/Edge选择；工具不训练模型。

```python
    parser.add_argument("--browser", choices=["chrome", "edge"], default="chrome")
```

`L27` — 解析待查看run、任务、全新输出目录、Chrome/Edge选择；工具不训练模型。

```python
    args = parser.parse_args()
```

`L28` — 创建验收输出并寻找可用浏览器；不可用就明确失败。

```python
    args.output.mkdir(parents=True, exist_ok=True)
```

`L29` — 创建验收输出并寻找可用浏览器；不可用就明确失败。

```python
    candidates = [Path("C:/Program Files/Google/Chrome/Application/chrome.exe")] if args.browser == "chrome" else [Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")]
```

`L30` — 创建验收输出并寻找可用浏览器；不可用就明确失败。

```python
    chrome = next((p for p in candidates if p.exists()), None)
```

`L31` — 创建验收输出并寻找可用浏览器；不可用就明确失败。

```python
    if chrome is None:
```

`L32` — 创建验收输出并寻找可用浏览器；不可用就明确失败。

```python
        raise RuntimeError("本机没有可用Chromium进行UI验收")
```

`L33` — 启动本次自有本地服务，动态端口，不依赖用户既有服务。

```python
    server = make_server(PROJECT, port=0)
```

`L34` — 启动本次自有本地服务，动态端口，不依赖用户既有服务。

```python
    thread = threading.Thread(target=server.serve_forever, daemon=True)
```

`L35` — 启动本次自有本地服务，动态端口，不依赖用户既有服务。

```python
    thread.start()
```

`L36` — 寻找浏览器调试空闲本地端口。

```python
    with socket.socket() as sock:
```

`L37` — 寻找浏览器调试空闲本地端口。

```python
        sock.bind(("127.0.0.1", 0))
```

`L38` — 寻找浏览器调试空闲本地端口。

```python
        debug_port = sock.getsockname()[1]
```

`L39` — 配置本地URL与独立QA浏览器profile，禁止复用旧profile；打开本次日志。

```python
    app_url = f"http://127.0.0.1:{server.server_port}"
```

`L40` — 配置本地URL与独立QA浏览器profile，禁止复用旧profile；打开本次日志。

```python
    profile = args.output.resolve() / "browser_profile"
```

`L41` — 配置本地URL与独立QA浏览器profile，禁止复用旧profile；打开本次日志。

```python
    if profile.exists():
```

`L42` — 配置本地URL与独立QA浏览器profile，禁止复用旧profile；打开本次日志。

```python
        raise ValueError("QA浏览器目录已存在，请使用新的验收输出目录")
```

`L43` — 配置本地URL与独立QA浏览器profile，禁止复用旧profile；打开本次日志。

```python
    browser_log = (args.output / "browser.log").open("w", encoding="utf-8")
```

`L44` — 无可见窗口启动自有无界面Chromium；仅QA进程用no-sandbox与本地调试，独立profile，不涉及用户日常浏览器。

```python
    proc = subprocess.Popen([str(chrome), "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--remote-debugging-address=127.0.0.1",
```

`L45` — 无可见窗口启动自有无界面Chromium；仅QA进程用no-sandbox与本地调试，独立profile，不涉及用户日常浏览器。

```python
        "--no-sandbox", "--disable-extensions", "--disable-background-networking", "--disable-background-mode", "--enable-logging=stderr",
```

`L46` — 无可见窗口启动自有无界面Chromium；仅QA进程用no-sandbox与本地调试，独立profile，不涉及用户日常浏览器。

```python
        f"--remote-debugging-port={debug_port}", f"--remote-allow-origins=http://127.0.0.1:{debug_port}",
```

`L47` — 无可见窗口启动自有无界面Chromium；仅QA进程用no-sandbox与本地调试，独立profile，不涉及用户日常浏览器。

```python
        f"--user-data-dir={profile}", app_url], stdout=browser_log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
```

`L48` — 准备浏览器对象、通过项和异常处理。

```python
    browser = None
```

`L49` — 准备浏览器对象、通过项和异常处理。

```python
    checks = []
```

`L50` — 准备浏览器对象、通过项和异常处理。

```python
    try:
```

`L51` — 最多20秒等待本次本地页面可调试；网络错误重试且短暂停顿。

```python
        deadline = time.monotonic() + 20
```

`L52` — 最多20秒等待本次本地页面可调试；网络错误重试且短暂停顿。

```python
        while time.monotonic() < deadline:
```

`L53` — 最多20秒等待本次本地页面可调试；网络错误重试且短暂停顿。

```python
            try:
```

`L54` — 最多20秒等待本次本地页面可调试；网络错误重试且短暂停顿。

```python
                pages = requests.get(f"http://127.0.0.1:{debug_port}/json", timeout=1).json()
```

`L55` — 最多20秒等待本次本地页面可调试；网络错误重试且短暂停顿。

```python
                if any(p.get("url", "").startswith(app_url) for p in pages):
```

`L56` — 最多20秒等待本次本地页面可调试；网络错误重试且短暂停顿。

```python
                    break
```

`L57` — 最多20秒等待本次本地页面可调试；网络错误重试且短暂停顿。

```python
            except requests.RequestException:
```

`L58` — 最多20秒等待本次本地页面可调试；网络错误重试且短暂停顿。

```python
                pass
```

`L59` — 最多20秒等待本次本地页面可调试；网络错误重试且短暂停顿。

```python
            time.sleep(.2)
```

`L60` — 连接浏览器，设1440×1100桌面视口，等待脚本与导航存在。

```python
        browser = Browser(debug_port, app_url)
```

`L61` — 连接浏览器，设1440×1100桌面视口，等待脚本与导航存在。

```python
        browser.call("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1100, "deviceScaleFactor": 1, "mobile": False})
```

`L62` — 连接浏览器，设1440×1100桌面视口，等待脚本与导航存在。

```python
        browser.until("!!window.mlWorkbench && !!document.querySelector('[data-view=ml]')")
```

`L63` — 点击ML页面，检查固定协议确实写了未来五个交易日。

```python
        browser.evaluate("document.querySelector('[data-view=ml]').click()")
```

`L64` — 点击ML页面，检查固定协议确实写了未来五个交易日。

```python
        browser.until("!!document.getElementById('ml-run')")
```

`L65` — 点击ML页面，检查固定协议确实写了未来五个交易日。

```python
        assert browser.evaluate("document.getElementById('ml-content').textContent.includes('未来五个交易日')")
```

`L66` — 打开既有运行，等待五行模型/对照，截图并检查默认测试段。

```python
        browser.evaluate(f"window.mlWorkbench.open({json.dumps(args.run_id)})")
```

`L67` — 打开既有运行，等待五行模型/对照，截图并检查默认测试段。

```python
        browser.until("document.querySelectorAll('#ml-scores tbody tr').length===5")
```

`L68` — 打开既有运行，等待五行模型/对照，截图并检查默认测试段。

```python
        browser.screenshot(args.output / "ml-desktop.png")
```

`L69` — 打开既有运行，等待五行模型/对照，截图并检查默认测试段。

```python
        assert browser.evaluate("document.querySelector('#ml-scope').value==='test'")
```

`L70` — 打开既有运行，等待五行模型/对照，截图并检查默认测试段。

```python
        checks.append("ML navigation, fixed protocol and five model/control test rows")
```

`L71` — 切2024指标，确认十分组仍明确标注全部测试期。

```python
        browser.evaluate("document.querySelector('#ml-scope').value='test_2024';document.querySelector('#ml-scope').dispatchEvent(new Event('change'))")
```

`L72` — 切2024指标，确认十分组仍明确标注全部测试期。

```python
        assert browser.evaluate("document.querySelector('#ml-quantiles').textContent.includes('全部测试期')")
```

`L73` — 切2024指标，确认十分组仍明确标注全部测试期。

```python
        checks.append("year selector and whole-test quantile label")
```

`L74` — 打开研究任务并点击ML结果，确认任务与诊断能互相跳转。

```python
        browser.evaluate(f"document.querySelector('[data-view=tasks]').click();window.taskWorkbench.open({json.dumps(args.task_id)})")
```

`L75` — 打开研究任务并点击ML结果，确认任务与诊断能互相跳转。

```python
        browser.until("!!document.querySelector('[data-task-ml]')")
```

`L76` — 打开研究任务并点击ML结果，确认任务与诊断能互相跳转。

```python
        assert browser.evaluate("document.getElementById('task-detail').textContent.includes('机器学习信号分析')")
```

`L77` — 打开研究任务并点击ML结果，确认任务与诊断能互相跳转。

```python
        browser.evaluate("document.querySelector('[data-task-ml]').click()")
```

`L78` — 打开研究任务并点击ML结果，确认任务与诊断能互相跳转。

```python
        browser.until("document.querySelectorAll('#ml-scores tbody tr').length===5")
```

`L79` — 打开研究任务并点击ML结果，确认任务与诊断能互相跳转。

```python
        checks.append("research-task result opens ML diagnostics")
```

`L80` — 改390×844窄屏，截图并检查整页不产生横向溢出；表格可内部滚动。

```python
        browser.call("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True})
```

`L81` — 改390×844窄屏，截图并检查整页不产生横向溢出；表格可内部滚动。

```python
        browser.evaluate("window.scrollTo(0,0)")
```

`L82` — 改390×844窄屏，截图并检查整页不产生横向溢出；表格可内部滚动。

```python
        browser.screenshot(args.output / "ml-mobile.png")
```

`L83` — 改390×844窄屏，截图并检查整页不产生横向溢出；表格可内部滚动。

```python
        assert browser.evaluate("document.documentElement.scrollWidth<=window.innerWidth+2")
```

`L84` — 改390×844窄屏，截图并检查整页不产生横向溢出；表格可内部滚动。

```python
        checks.append("narrow-screen document fits viewport; wide tables scroll internally")
```

`L85` — 要求没有JS运行错误，保存通过项JSON并输出结果。

```python
        assert not browser.errors, browser.errors
```

`L86` — 要求没有JS运行错误，保存通过项JSON并输出结果。

```python
        write_json(args.output / "ui_checks.json", {"ok": True, "checks": checks, "runtime_errors": browser.errors, "run_id": args.run_id})
```

`L87` — 要求没有JS运行错误，保存通过项JSON并输出结果。

```python
        print(json.dumps({"ok": True, "checks": checks}, ensure_ascii=False))
```

`L88` — 失败保存错误与已通过检查，再抛出，不伪造验收成功。

```python
    except Exception as exc:
```

`L89` — 失败保存错误与已通过检查，再抛出，不伪造验收成功。

```python
        write_json(args.output / "ui_failure.json", {"ok": False, "error": str(exc), "completed_checks": checks})
```

`L90` — 失败保存错误与已通过检查，再抛出，不伪造验收成功。

```python
        raise
```

`L91` — 结束时关闭调试连接，只停止本次浏览器PID进程树，并关闭自有服务器和日志。

```python
    finally:
```

`L92` — 结束时关闭调试连接，只停止本次浏览器PID进程树，并关闭自有服务器和日志。

```python
        if browser:
```

`L93` — 结束时关闭调试连接，只停止本次浏览器PID进程树，并关闭自有服务器和日志。

```python
            browser.ws.close()
```

`L94` — 结束时关闭调试连接，只停止本次浏览器PID进程树，并关闭自有服务器和日志。

```python
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
```

`L95` — 结束时关闭调试连接，只停止本次浏览器PID进程树，并关闭自有服务器和日志。

```python
        proc.wait(timeout=10)
```

`L96` — 结束时关闭调试连接，只停止本次浏览器PID进程树，并关闭自有服务器和日志。

```python
        server.shutdown()
```

`L97` — 结束时关闭调试连接，只停止本次浏览器PID进程树，并关闭自有服务器和日志。

```python
        server.server_close()
```

`L98` — 结束时关闭调试连接，只停止本次浏览器PID进程树，并关闭自有服务器和日志。

```python
        thread.join(timeout=5)
```

`L99` — 结束时关闭调试连接，只停止本次浏览器PID进程树，并关闭自有服务器和日志。

```python
        browser_log.close()
```

`L100` — 清理前验证绝对profile只能是本次输出目录下browser_profile，越界就拒绝。

```python
        if profile.exists():
```

`L101` — 清理前验证绝对profile只能是本次输出目录下browser_profile，越界就拒绝。

```python
            resolved = profile.resolve()
```

`L102` — 清理前验证绝对profile只能是本次输出目录下browser_profile，越界就拒绝。

```python
            if resolved.parent != args.output.resolve() or resolved.name != "browser_profile":
```

`L103` — 清理前验证绝对profile只能是本次输出目录下browser_profile，越界就拒绝。

```python
                raise RuntimeError("QA目录越界，停止清理")
```

`L104` — 仅清理已经验证的自有QA目录；Windows锁定时有限重试。

```python
            for retry in range(10):
```

`L105` — 仅清理已经验证的自有QA目录；Windows锁定时有限重试。

```python
                try:
```

`L106` — 仅清理已经验证的自有QA目录；Windows锁定时有限重试。

```python
                    shutil.rmtree(resolved)
```

`L107` — 仅清理已经验证的自有QA目录；Windows锁定时有限重试。

```python
                    break
```

`L108` — 仅清理已经验证的自有QA目录；Windows锁定时有限重试。

```python
                except PermissionError:
```

`L109` — 仅清理已经验证的自有QA目录；Windows锁定时有限重试。

```python
                    time.sleep(.2)
```

`L110` — 文件仍锁定就保留目录并写说明，不扩展删除范围。

```python
            else:
```

`L111` — 文件仍锁定就保留目录并写说明，不扩展删除范围。

```python
                write_json(args.output / "cleanup_note.json", {"owned_profile": str(resolved), "reason": "Windows仍锁定QA文件，保留目录而不影响已完成的UI检查"})
```

`L112` — 空行：仅分隔代码段，无计算行为。

```python

```

`L113` — 空行：仅分隔代码段，无计算行为。

```python

```

`L114` — 作为脚本运行时进入main。

```python
if __name__ == "__main__":
```

`L115` — 作为脚本运行时进入main。

```python
    main()
```

**tests/test_mlresearch.py · 全部新增源码**



`L1` — 模块文档字符串：说明代码范围，不执行数据处理或证明策略有效。

```python
"""Leakage boundaries, unknown outcomes, and the real frozen ML executor."""
```

`L2` — replace没有被后面的测试使用；此行是冗余导入，没有测试或学习行为。

```python
from dataclasses import replace
```

`L3` — 导入：读写可追溯配置、收据与结果。

```python
import json
```

`L4` — 导入：安全明确的文件路径处理。

```python
from pathlib import Path
```

`L5` — 导入：冻结源码复制或受约束的自有QA目录清理。

```python
import shutil
```

`L6` — 导入：运行本地服务或后台执行，不是多模型研究证据。

```python
import threading
```

`L7` — 导入：严格编号格式与测试页面元数据提取。

```python
import re
```

`L8` — 导入：识别API拒绝的HTTP异常。

```python
from urllib.error import HTTPError
```

`L9` — 导入：测试专用本地HTTP请求。

```python
from urllib.request import Request, urlopen
```

`L10` — 空行：仅分隔代码段，无计算行为。

```python

```

`L11` — 导入：向量计算、有限值处理与统计抽样。

```python
import numpy as np
```

`L12` — 导入：行情面板、滚动窗口、分组及Parquet读取。

```python
import pandas as pd
```

`L13` — 导入：测试断言、临时目录夹具与预期错误。

```python
import pytest
```

`L14` — 空行：仅分隔代码段，无计算行为。

```python

```

`L15` — 导入：ML配置、白名单与固定设计。

```python
from src.mlresearch.contracts import MLSpec, FEATURES, benchmark_proposal
```

`L16` — 导入：过去特征、未来标签及时间切分函数。

```python
from src.mlresearch.dataset import price_features, forward_labels, assign_splits
```

`L17` — 导入：统计评价和完整方法列表。

```python
from src.mlresearch.evaluation import evaluate, SCORES
```

`L18` — 导入：固定模型训练入口。

```python
from src.mlresearch.runner import fit_models
```

`L19` — 导入：复用任务、租约、预算、登记、运行和提交流程。

```python
from src.researchops.service import Research
```

`L20` — 导入：复用哈希、完整性核验、JSON写入及数据序列化辅助函数。

```python
from src.technical.artifacts import digest, write_json
```

`L21` — 空行：仅分隔代码段，无计算行为。

```python

```

`L22` — 定位真实项目源码，用来复制到测试专属临时项目；不改真实研究数据。

```python
PROJECT = Path(__file__).resolve().parents[1]
```

`L23` — 空行：仅分隔代码段，无计算行为。

```python

```

`L24` — 空行：仅分隔代码段，无计算行为。

```python

```

`L25` — 创建确定性的合成收益与价格，不是模拟真实金融规律证明策略。

```python
def quotes(calendar, code="000001.SZ", phase=0):
```

`L26` — 创建确定性的合成收益与价格，不是模拟真实金融规律证明策略。

```python
    t = np.arange(len(calendar))
```

`L27` — 创建确定性的合成收益与价格，不是模拟真实金融规律证明策略。

```python
    r = 0.0003 + 0.004 * np.sin(t / 6 + phase)
```

`L28` — 创建确定性的合成收益与价格，不是模拟真实金融规律证明策略。

```python
    close = 20 * np.exp(np.cumsum(r))
```

`L29` — 生成开高低收、参考前收、量额、ST等测试列，供边界测试。

```python
    return pd.DataFrame(dict(stock_code=code, trade_date=calendar, close=close,
```

`L30` — 生成开高低收、参考前收、量额、ST等测试列，供边界测试。

```python
        pre_close=close / np.exp(r), open=close * .998, high=close * 1.01,
```

`L31` — 生成开高低收、参考前收、量额、ST等测试列，供边界测试。

```python
        low=close * .99, volume=np.full(len(t), 2000000), amount=close * 2000000,
```

`L32` — 生成开高低收、参考前收、量额、ST等测试列，供边界测试。

```python
        is_st=np.zeros(len(t))))
```

`L33` — 空行：仅分隔代码段，无计算行为。

```python

```

`L34` — 空行：仅分隔代码段，无计算行为。

```python

```

`L35` — 准备150个工作日与合成行情。

```python
def test_features_do_not_change_when_future_is_appended_or_changed():
```

`L36` — 准备150个工作日与合成行情。

```python
    cal = pd.bdate_range("2020-01-01", periods=150).strftime("%Y-%m-%d").tolist()
```

`L37` — 准备150个工作日与合成行情。

```python
    raw = quotes(cal)
```

`L38` — 分别算前100日和完整150日特征。

```python
    prefix = price_features(raw.iloc[:100], cal)
```

`L39` — 分别算前100日和完整150日特征。

```python
    full = price_features(raw, cal)
```

`L40` — 把第100行之后价格/前收/额乘10，再算特征，故意制造未来扰动。

```python
    changed = raw.copy()
```

`L41` — 把第100行之后价格/前收/额乘10，再算特征，故意制造未来扰动。

```python
    changed.loc[100:, ["close", "pre_close", "amount"]] *= 10
```

`L42` — 把第100行之后价格/前收/额乘10，再算特征，故意制造未来扰动。

```python
    future_changed = price_features(changed, cal)
```

`L43` — 前100行的所有16个特征必须完全不变，检查未来不影响过去。

```python
    for comparison in [full.iloc[:100], future_changed.iloc[:100]]:
```

`L44` — 前100行的所有16个特征必须完全不变，检查未来不影响过去。

```python
        pd.testing.assert_frame_equal(prefix[list(FEATURES)], comparison[list(FEATURES)])
```

`L45` — 空行：仅分隔代码段，无计算行为。

```python

```

`L46` — 空行：仅分隔代码段，无计算行为。

```python

```

`L47` — 准备80日行情并构造五日标签。

```python
def test_forward_label_uses_market_sessions_and_gap_is_unknown():
```

`L48` — 准备80日行情并构造五日标签。

```python
    cal = pd.bdate_range("2020-01-01", periods=80).strftime("%Y-%m-%d").tolist()
```

`L49` — 准备80日行情并构造五日标签。

```python
    raw = quotes(cal)
```

`L50` — 准备80日行情并构造五日标签。

```python
    labeled = forward_labels(price_features(raw, cal), cal, 5)
```

`L51` — 第60行答案须等于61到65日收益连乘减1，结束日须为第65个市场日期。

```python
    assert labeled.loc[60, "label_return"] == pytest.approx(np.prod(raw.loc[61:65, "close"] / raw.loc[61:65, "pre_close"]) - 1)
```

`L52` — 第60行答案须等于61到65日收益连乘减1，结束日须为第65个市场日期。

```python
    assert labeled.loc[60, "label_end"] == cal[65]
```

`L53` — 删掉未来第62行报价后标签必须未知；尾部五日没有成熟答案也必须未知。

```python
    missing = forward_labels(price_features(raw.drop(index=62), cal), cal, 5)
```

`L54` — 删掉未来第62行报价后标签必须未知；尾部五日没有成熟答案也必须未知。

```python
    assert np.isnan(missing.loc[missing.trade_date.eq(cal[60]), "label_return"].iloc[0])
```

`L55` — 删掉未来第62行报价后标签必须未知；尾部五日没有成熟答案也必须未知。

```python
    assert labeled.tail(5).label_return.isna().all()
```

`L56` — 空行：仅分隔代码段，无计算行为。

```python

```

`L57` — 空行：仅分隔代码段，无计算行为。

```python

```

`L58` — 建立三个阶段及四个刻意跨界/不跨界样本，直接检查成熟日规则。

```python
def test_label_boundary_purge_not_arbitrary_panel_rows():
```

`L59` — 建立三个阶段及四个刻意跨界/不跨界样本，直接检查成熟日规则。

```python
    spec = MLSpec(train_start="2020-01-01", train_end="2020-12-31", valid_start="2021-01-01", valid_end="2021-12-31", test_start="2022-01-01", test_end="2022-12-31")
```

`L60` — 建立三个阶段及四个刻意跨界/不跨界样本，直接检查成熟日规则。

```python
    frame = pd.DataFrame({"trade_date": ["2020-12-25", "2020-12-30", "2021-12-30", "2022-12-30"],
```

`L61` — 建立三个阶段及四个刻意跨界/不跨界样本，直接检查成熟日规则。

```python
        "label_end": ["2020-12-31", "2021-01-05", "2022-01-06", "2023-01-06"]})
```

`L62` — 只有第一个样本能归训练，其余都excluded；不是任意删面板行。

```python
    assert assign_splits(frame, spec).split.tolist() == ["train", "excluded", "excluded", "excluded"]
```

`L63` — 空行：仅分隔代码段，无计算行为。

```python

```

`L64` — 空行：仅分隔代码段，无计算行为。

```python

```

`L65` — 构造120训练+20测试行，给可学习Y，准备检测训练隔离。

```python
def test_preprocessing_and_fit_ignore_validation_and_future_label_columns():
```

`L66` — 构造120训练+20测试行，给可学习Y，准备检测训练隔离。

```python
    rng = np.random.default_rng(11)
```

`L67` — 构造120训练+20测试行，给可学习Y，准备检测训练隔离。

```python
    frame = pd.DataFrame(rng.normal(size=(140, len(FEATURES))), columns=list(FEATURES))
```

`L68` — 构造120训练+20测试行，给可学习Y，准备检测训练隔离。

```python
    frame["split"] = ["train"] * 120 + ["test"] * 20
```

`L69` — 构造120训练+20测试行，给可学习Y，准备检测训练隔离。

```python
    frame["label_observed"] = True
```

`L70` — 构造120训练+20测试行，给可学习Y，准备检测训练隔离。

```python
    frame["label_relative"] = frame.return_5 * .01
```

`L71` — 加一列未来财富诱饵，并制造输入缺失，检查白名单和填补。

```python
    frame["future_wealth"] = rng.normal(size=len(frame)) * 1000
```

`L72` — 加一列未来财富诱饵，并制造输入缺失，检查白名单和填补。

```python
    frame.loc[:5, "return_1"] = np.nan
```

`L73` — 用小树设置完成第一次合成训练。

```python
    spec = MLSpec(tree_iterations=3, tree_min_samples=2)
```

`L74` — 用小树设置完成第一次合成训练。

```python
    models = fit_models(frame, spec)
```

`L75` — 把测试X、测试Y和诱饵未来列大幅改动，再训练；训练输入应保持一致。

```python
    changed = frame.copy()
```

`L76` — 把测试X、测试Y和诱饵未来列大幅改动，再训练；训练输入应保持一致。

```python
    changed.loc[120:, list(FEATURES)] = 10000
```

`L77` — 把测试X、测试Y和诱饵未来列大幅改动，再训练；训练输入应保持一致。

```python
    changed.loc[120:, "label_relative"] = -999
```

`L78` — 把测试X、测试Y和诱饵未来列大幅改动，再训练；训练输入应保持一致。

```python
    changed["future_wealth"] = -frame.future_wealth
```

`L79` — 把测试X、测试Y和诱饵未来列大幅改动，再训练；训练输入应保持一致。

```python
    again = fit_models(changed, spec)
```

`L80` — 填补维度正确，中位数来自前120训练行，两次训练的训练预测必须相同。

```python
    for name in models:
```

`L81` — 填补维度正确，中位数来自前120训练行，两次训练的训练预测必须相同。

```python
        assert len(models[name].steps[0][1].statistics_) == len(FEATURES)
```

`L82` — 填补维度正确，中位数来自前120训练行，两次训练的训练预测必须相同。

```python
        assert np.allclose(models[name].steps[0][1].statistics_, frame.iloc[:120][list(FEATURES)].median())
```

`L83` — 填补维度正确，中位数来自前120训练行，两次训练的训练预测必须相同。

```python
        assert np.allclose(models[name].predict(frame.iloc[:120][list(FEATURES)]), again[name].predict(frame.iloc[:120][list(FEATURES)]))
```

`L84` — 空行：仅分隔代码段，无计算行为。

```python

```

`L85` — 空行：仅分隔代码段，无计算行为。

```python

```

`L86` — 构造10只同日候选和已知结果，准备未知结果选择测试。

```python
def test_unknown_outcome_stays_in_top_selection():
```

`L87` — 构造10只同日候选和已知结果，准备未知结果选择测试。

```python
    frame = pd.DataFrame({"split": ["test"] * 10, "trade_date": ["2024-01-05"] * 10,
```

`L88` — 构造10只同日候选和已知结果，准备未知结果选择测试。

```python
        "stock_code": [str(i) for i in range(10)], "label_return": np.arange(10) / 100,
```

`L89` — 构造10只同日候选和已知结果，准备未知结果选择测试。

```python
        "label_relative": (np.arange(10) - 4.5) / 100, "label_observed": [True] * 10})
```

`L90` — 所有方法用同一分数，最高分股票未来标签改未知。

```python
    for model in SCORES:
```

`L91` — 所有方法用同一分数，最高分股票未来标签改未知。

```python
        frame[model] = np.arange(10)
```

`L92` — 所有方法用同一分数，最高分股票未来标签改未知。

```python
    frame.loc[9, ["label_return", "label_relative"]] = np.nan
```

`L93` — 所有方法用同一分数，最高分股票未来标签改未知。

```python
    frame.loc[9, "label_observed"] = False
```

`L94` — Top2仍选2只，其中只1只有标签，覆盖50%；未知结果没有被提前删掉。

```python
    daily, _, summary, _ = evaluate(frame, MLSpec(top_k=2, bins=2))
```

`L95` — Top2仍选2只，其中只1只有标签，覆盖50%；未知结果没有被提前删掉。

```python
    assert daily.top_count.eq(2).all()
```

`L96` — Top2仍选2只，其中只1只有标签，覆盖50%；未知结果没有被提前删掉。

```python
    assert daily.top_label_count.eq(1).all()
```

`L97` — Top2仍选2只，其中只1只有标签，覆盖50%；未知结果没有被提前删掉。

```python
    assert all(s["top_label_coverage"] == .5 for s in summary)
```

`L98` — 空行：仅分隔代码段，无计算行为。

```python

```

`L99` — 空行：仅分隔代码段，无计算行为。

```python

```

`L100` — pytest夹具为每测试创建专属临时project/store/snapshot，复制源码并建数据目录。

```python
@pytest.fixture
```

`L101` — pytest夹具为每测试创建专属临时project/store/snapshot，复制源码并建数据目录。

```python
def ml_research(tmp_path):
```

`L102` — pytest夹具为每测试创建专属临时project/store/snapshot，复制源码并建数据目录。

```python
    project = tmp_path / "project"
```

`L103` — pytest夹具为每测试创建专属临时project/store/snapshot，复制源码并建数据目录。

```python
    for package in ["technical", "mlresearch"]:
```

`L104` — pytest夹具为每测试创建专属临时project/store/snapshot，复制源码并建数据目录。

```python
        shutil.copytree(PROJECT / "src" / package, project / "src" / package, ignore=shutil.ignore_patterns("__pycache__"))
```

`L105` — pytest夹具为每测试创建专属临时project/store/snapshot，复制源码并建数据目录。

```python
    root = tmp_path / "store"
```

`L106` — pytest夹具为每测试创建专属临时project/store/snapshot，复制源码并建数据目录。

```python
    sid = "a" * 24
```

`L107` — pytest夹具为每测试创建专属临时project/store/snapshot，复制源码并建数据目录。

```python
    snapshot = root / "snapshots" / sid
```

`L108` — pytest夹具为每测试创建专属临时project/store/snapshot，复制源码并建数据目录。

```python
    snapshot.mkdir(parents=True)
```

`L109` — 造8只股票的合成日线及上市元数据，完全独立于实际快照。

```python
    cal = pd.bdate_range("2019-01-01", "2022-06-30").strftime("%Y-%m-%d").tolist()
```

`L110` — 造8只股票的合成日线及上市元数据，完全独立于实际快照。

```python
    codes = [f"00000{i}.SZ" for i in range(1, 9)]
```

`L111` — 造8只股票的合成日线及上市元数据，完全独立于实际快照。

```python
    pd.concat([quotes(cal, code, phase=i / 2) for i, code in enumerate(codes)]).to_parquet(snapshot / "bars.parquet", index=False)
```

`L112` — 造8只股票的合成日线及上市元数据，完全独立于实际快照。

```python
    pd.DataFrame(dict(stock_code=codes, listed_date="2010-01-01", de_listed_date="9999-12-31")).to_parquet(snapshot / "metadata.parquet", index=False)
```

`L113` — 逐股票逐季度造已公告股本报告供共同资格逻辑跑通。

```python
    reports = []
```

`L114` — 逐股票逐季度造已公告股本报告供共同资格逻辑跑通。

```python
    for code in codes:
```

`L115` — 逐股票逐季度造已公告股本报告供共同资格逻辑跑通。

```python
        for period in pd.date_range("2018-12-31", "2022-03-31", freq="QE"):
```

`L116` — 逐股票逐季度造已公告股本报告供共同资格逻辑跑通。

```python
            reports.append(dict(stock_code=code, report_date=period.strftime("%Y-%m-%d"), publication_date=(period + pd.Timedelta(days=30)).strftime("%Y-%m-%d"), total_shares=1e8, np_parent_company_owners=1e7, total_shareholder_equity=1e8, total_assets=2e8))
```

`L117` — 逐股票逐季度造已公告股本报告供共同资格逻辑跑通。

```python
    pd.DataFrame(reports).to_parquet(snapshot / "fundamentals.parquet", index=False)
```

`L118` — 保存测试日历和各文件哈希清单。

```python
    write_json(snapshot / "calendar.json", cal)
```

`L119` — 保存测试日历和各文件哈希清单。

```python
    write_json(snapshot / "manifest.json", {"snapshot_id": sid, "start": cal[0], "end": cal[-1], "counts": {"bars": len(cal) * len(codes), "fundamentals": len(reports)},
```

`L120` — 保存测试日历和各文件哈希清单。

```python
        "artifacts": {p.name: digest(p) for p in snapshot.iterdir()}})
```

`L121` — 在临时研究库建立任务、领取、生成MLproposal。

```python
    research = Research(project, root)
```

`L122` — 在临时研究库建立任务、领取、生成MLproposal。

```python
    task = research.create({"title": "ML冻结验收", "question": "时间隔离", "rationale": "工程测试", "success_criteria": ["数值核查通过"], "stop_criteria": ["一次"], "max_experiments": 2})
```

`L123` — 在临时研究库建立任务、领取、生成MLproposal。

```python
    session = research.store.claim(task["id"], "builder")
```

`L124` — 在临时研究库建立任务、领取、生成MLproposal。

```python
    proposal = benchmark_proposal(sid)
```

`L125` — 缩短日期和树规模，返回测试上下文；这些参数不是实际投资基准参数。

```python
    proposal["spec"] = MLSpec(train_end="2020-12-31", valid_start="2021-01-01", valid_end="2021-12-31", test_start="2022-01-01", test_end="2022-06-30", min_avg_amount=0, tree_iterations=3, tree_min_samples=2, bins=2, top_k=2).to_dict()
```

`L126` — 缩短日期和树规模，返回测试上下文；这些参数不是实际投资基准参数。

```python
    return research, session, proposal
```

`L127` — 空行：仅分隔代码段，无计算行为。

```python

```

`L128` — 空行：仅分隔代码段，无计算行为。

```python

```

`L129` — 登记后仅改描述再次登记须返回原实验，验证实质去重。

```python
def test_native_frozen_ml_executor_and_typed_submission(ml_research):
```

`L130` — 登记后仅改描述再次登记须返回原实验，验证实质去重。

```python
    research, session, proposal = ml_research
```

`L131` — 登记后仅改描述再次登记须返回原实验，验证实质去重。

```python
    experiment = research.register(session, proposal)
```

`L132` — 登记后仅改描述再次登记须返回原实验，验证实质去重。

```python
    assert research.register(session, {**proposal, "hypothesis": "描述不同不应重跑"})["id"] == experiment["id"]
```

`L133` — 故意让临时开发runner报错，冻结执行仍须成功并数值核查通过，证明执行用登记源码。

```python
    (research.project / "src/mlresearch/runner.py").write_text('raise RuntimeError("changed development code")')
```

`L134` — 故意让临时开发runner报错，冻结执行仍须成功并数值核查通过，证明执行用登记源码。

```python
    result = research.execute(session, experiment["id"])
```

`L135` — 故意让临时开发runner报错，冻结执行仍须成功并数值核查通过，证明执行用登记源码。

```python
    assert result["status"] == "completed", result
```

`L136` — 故意让临时开发runner报错，冻结执行仍须成功并数值核查通过，证明执行用登记源码。

```python
    assert result["result"]["kind"] == "ml"
```

`L137` — 故意让临时开发runner报错，冻结执行仍须成功并数值核查通过，证明执行用登记源码。

```python
    assert result["result"]["audit"]["prediction_sample_max_error"]["ridge"] < 1e-10
```

`L138` — 启动测试专用服务与本地URL，准备API验证。

```python
    from src.technical.server import make_server
```

`L139` — 启动测试专用服务与本地URL，准备API验证。

```python
    server = make_server(research.project, port=0, root=research.root)
```

`L140` — 启动测试专用服务与本地URL，准备API验证。

```python
    thread = threading.Thread(target=server.serve_forever, daemon=True)
```

`L141` — 启动测试专用服务与本地URL，准备API验证。

```python
    thread.start()
```

`L142` — 启动测试专用服务与本地URL，准备API验证。

```python
    url = f"http://127.0.0.1:{server.server_port}"
```

`L143` — 打开页面，检查ML导航及服务启动元数据。

```python
    try:
```

`L144` — 打开页面，检查ML导航及服务启动元数据。

```python
        with urlopen(url) as response:
```

`L145` — 打开页面，检查ML导航及服务启动元数据。

```python
            page = response.read().decode()
```

`L146` — 打开页面，检查ML导航及服务启动元数据。

```python
        assert 'data-view="ml"' in page
```

`L147` — 打开页面，检查ML导航及服务启动元数据。

```python
        boot = json.loads(re.search(r"window.TECHNICAL=(.*?);</script>", page).group(1))
```

`L148` — 固定plan、结果JSON和REPORT下载接口都须可读。

```python
        with urlopen(url + "/api/ml/plan") as response:
```

`L149` — 固定plan、结果JSON和REPORT下载接口都须可读。

```python
            assert json.load(response)["kind"] == "ml"
```

`L150` — 固定plan、结果JSON和REPORT下载接口都须可读。

```python
        with urlopen(url + "/api/ml/" + result["run_id"]) as response:
```

`L151` — 固定plan、结果JSON和REPORT下载接口都须可读。

```python
            assert json.load(response)["summaries"]
```

`L152` — 固定plan、结果JSON和REPORT下载接口都须可读。

```python
        with urlopen(url + "/api/ml/" + result["run_id"] + "/download?file=REPORT.md") as response:
```

`L153` — 固定plan、结果JSON和REPORT下载接口都须可读。

```python
            assert response.read()
```

`L154` — 路径穿越下载必须HTTP失败。

```python
        with pytest.raises(HTTPError):
```

`L155` — 路径穿越下载必须HTTP失败。

```python
            urlopen(url + "/api/ml/" + result["run_id"] + "/download?file=../../input.json")
```

`L156` — 无研究令牌的启动请求必须403拒绝。

```python
        with pytest.raises(HTTPError) as unauthorized:
```

`L157` — 无研究令牌的启动请求必须403拒绝。

```python
            urlopen(Request(url + "/api/ml/benchmark", data=b"{}"))
```

`L158` — 无研究令牌的启动请求必须403拒绝。

```python
        assert unauthorized.value.code == 403
```

`L159` — 即便有令牌，固定基准入口的隐藏参数也必须400拒绝。

```python
        with pytest.raises(HTTPError) as override:
```

`L160` — 即便有令牌，固定基准入口的隐藏参数也必须400拒绝。

```python
            urlopen(Request(url + "/api/ml/benchmark", data=b'{"hidden":true}', headers={"X-Research-Token": boot["token"]}))
```

`L161` — 即便有令牌，固定基准入口的隐藏参数也必须400拒绝。

```python
        assert override.value.code == 400
```

`L162` — 关闭自有测试服务器和线程。

```python
    finally:
```

`L163` — 关闭自有测试服务器和线程。

```python
        server.shutdown()
```

`L164` — 关闭自有测试服务器和线程。

```python
        server.server_close()
```

`L165` — 关闭自有测试服务器和线程。

```python
        thread.join(timeout=5)
```

`L166` — 提交应明确registered_ml_analysis而非账户类型，另角色标识可完成临时测试复核。

```python
    submitted = research.submit(session, {"summary": "ML验收完成", "findings": ["核查通过"], "limitations": ["合成"], "next_steps": ["另一角色复核"]})
```

`L167` — 提交应明确registered_ml_analysis而非账户类型，另角色标识可完成临时测试复核。

```python
    assert submitted["submission"]["evidence_kind"] == "registered_ml_analysis"
```

`L168` — 提交应明确registered_ml_analysis而非账户类型，另角色标识可完成临时测试复核。

```python
    reviewed = research.review(session["task_id"], "reviewer", "stop", "工程验收，非投资结论")
```

`L169` — 提交应明确registered_ml_analysis而非账户类型，另角色标识可完成临时测试复核。

```python
    assert reviewed["status"] == "completed"
```

`L170` — 子任务以ML运行作baseline，context必须识别其kind。

```python
    child = research.create({"title": "ML接续", "question": "接力", "rationale": "已存在基准", "success_criteria": ["可读"], "stop_criteria": ["一次"], "baseline_runs": [result["run_id"]]})
```

`L171` — 子任务以ML运行作baseline，context必须识别其kind。

```python
    assert research.context(child["id"])["baselines"][0]["kind"] == "ml"
```

`L172` — 空行：仅分隔代码段，无计算行为。

```python

```

`L173` — 空行：仅分隔代码段，无计算行为。

```python

```

`L174` — 在临时研究库登记后，读取冻结输入准备故意篡改。

```python
def test_tampered_ml_input_fails_before_launch(ml_research):
```

`L175` — 在临时研究库登记后，读取冻结输入准备故意篡改。

```python
    research, session, proposal = ml_research
```

`L176` — 在临时研究库登记后，读取冻结输入准备故意篡改。

```python
    experiment = research.register(session, proposal)
```

`L177` — 在临时研究库登记后，读取冻结输入准备故意篡改。

```python
    path = research.root / "worker_jobs" / experiment["id"] / "input.json"
```

`L178` — 在临时研究库登记后，读取冻结输入准备故意篡改。

```python
    cfg = json.loads(path.read_text(encoding="utf-8"))
```

`L179` — 把冻结日期改非法后执行必须在启动前拒绝，且实验留failed记录。

```python
    cfg["spec"]["test_start"] = "2020-01-01"
```

`L180` — 把冻结日期改非法后执行必须在启动前拒绝，且实验留failed记录。

```python
    write_json(path, cfg)
```

`L181` — 把冻结日期改非法后执行必须在启动前拒绝，且实验留failed记录。

```python
    with pytest.raises(ValueError, match="冻结输入"):
```

`L182` — 把冻结日期改非法后执行必须在启动前拒绝，且实验留failed记录。

```python
        research.execute(session, experiment["id"])
```

`L183` — 把冻结日期改非法后执行必须在启动前拒绝，且实验留failed记录。

```python
    assert research.store.experiment(experiment["id"])["status"] == "failed"
```

`L184` — 空行：仅分隔代码段，无计算行为。

```python

```

`L185` — 空行：仅分隔代码段，无计算行为。

```python

```

`L186` — MLproposal不接受账户费用情景，时间阶段乱序也必须拒绝。

```python
def test_ml_rejects_cost_override_and_invalid_date_order(ml_research):
```

`L187` — MLproposal不接受账户费用情景，时间阶段乱序也必须拒绝。

```python
    research, session, proposal = ml_research
```

`L188` — MLproposal不接受账户费用情景，时间阶段乱序也必须拒绝。

```python
    with pytest.raises(ValueError):
```

`L189` — MLproposal不接受账户费用情景，时间阶段乱序也必须拒绝。

```python
        research.register(session, {**proposal, "scenarios": ["configured"]})
```

`L190` — MLproposal不接受账户费用情景，时间阶段乱序也必须拒绝。

```python
    with pytest.raises(ValueError):
```

`L191` — MLproposal不接受账户费用情景，时间阶段乱序也必须拒绝。

```python
        MLSpec(test_start="2020-01-01")
```

**src/technical/assets/ml.js · 全部新增源码**



`L1` — 立即执行闭包，向window暴露mlWorkbench；浏览器只展示/调用服务，不训练。

```javascript
window.mlWorkbench=(()=>{
```

`L2` — 严格模式减少隐式变量等JS错误。

```javascript
  'use strict';
```

`L3` — 保存API、页面切换和运行通知回调，以及当前选中run_id。

```javascript
  let api,activate,started,selected=null;
```

`L4` — DOM查找和HTML转义；外部结果文本不能直接当标签插入。

```javascript
  const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
```

`L5` — 格式化百分比和小数，缺失显示破折号。

```javascript
  const pct=v=>v==null?'—':(100*v).toFixed(3)+'%',num=v=>v==null?'—':Number(v).toFixed(4);
```

`L6` — 内部方法名映射成人类名称；受限提升树是HistGBDT配置的简写。

```javascript
  const names={ridge:'岭回归',hist_gbdt:'受限提升树',small_cap:'小市值',reversal_5:'五日反转',fixed_random:'固定随机排序'};
```

`L7` — 为训练/验证/测试显示信息范围标签。

```javascript
  const scopes={train:'训练 · 同样本',valid:'验证 · 后续时间',test:'测试 · 后续时间'};
```

`L8` — 通用可横向滚动表格，列名转义；单元格传入已经处理的展示内容。

```javascript
  const table=(cols,rows)=>`<div class="table-wrap"><table><thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
```

`L9` — 把失败显示到状态文本，不掩盖错误。

```javascript
  function fail(e){$('ml-message').textContent=e.message||String(e);}
```

`L10` — 加载历史ML状态和固定设计，两次独立GET同时请求。

```javascript
  async function load(){
```

`L11` — 加载历史ML状态和固定设计，两次独立GET同时请求。

```javascript
    const [runs,plan]=await Promise.all([api('/api/ml'),api('/api/ml/plan')]);
```

`L12` — 绘制页面说明：比较后续排序并保存设计/模型/预测。

```javascript
    $('ml-content').innerHTML=`<div class="section-intro"><span class="eyebrow">MACHINE LEARNING RESEARCH</span><h2>先确认模型学到了选股信息</h2><p>比较固定模型在后续时间的股票排序与分组收益。每次保留设计、模型和逐股预测。</p></div>
```

`L13` — 绘制固定日期、目标、输入、对照、研究限制、运行/刷新按钮和配置JSON。

```javascript
      <div class="panel"><h3>日线价量基准案例</h3><p>每周采样，预测未来五个交易日的相对收益。训练 2020—2022，验证 2023，测试 2024—2026-09-24。</p><p>岭回归和受限提升树，比较同池小市值、五日反转与固定随机排序。模型仅用价量输入；股本用于共同候选资格和小市值对照。</p><p class="note">这些历史已经被查看。此处验证模型的时间迁移，不代表真正未见历史或可成交账户收益。</p><div class="toolbar"><button id="ml-run" class="primary">登记并运行基准案例</button><button id="ml-refresh">刷新结果</button></div><details><summary>固定设计与模型设置</summary><pre>${esc(JSON.stringify(plan.spec,null,2))}</pre></details><p id="ml-message" role="status"></p></div>
```

`L14` — 绘制全部运行记录，包括失败状态和错误，准备详情容器。

```javascript
      <div class="panel"><h3>研究记录</h3>${runs.length?runs.map(r=>`<p><button class="text-button" data-ml-run="${esc(r.run_id)}">${esc(r.name||r.run_id)} ↗</button> · ${esc({completed:'已完成',running:'运行中',failed:'失败'}[r.status]||r.status)} ${r.error?esc(r.error):''}</p>`).join(''):'<p>尚无机器学习案例。</p>'}</div><div id="ml-result"></div>`;
```

`L15` — 刷新按钮重新读已有结果，不重新训练。

```javascript
    $('ml-refresh').onclick=()=>load().catch(fail);
```

`L16` — 用户点击运行后禁用按钮，POST空配置到固定执行接口；返回任务并通知主工作台轮询。

```javascript
    $('ml-run').onclick=async()=>{try{$('ml-run').disabled=true;const r=await api('/api/ml/benchmark',{});$('ml-message').textContent='已启动，研究任务 '+r.task_id;started();}catch(e){$('ml-run').disabled=false;fail(e);}};
```

`L17` — 给历史记录按钮绑定详情加载。

```javascript
    document.querySelectorAll('[data-ml-run]').forEach(b=>b.onclick=()=>detail(b.dataset.mlRun).catch(fail));
```

`L18` — 有已选择且完成的运行则恢复其详情，再结束load。

```javascript
    if(selected&&runs.some(r=>r.run_id===selected&&r.status==='completed'))await detail(selected);
```

`L19` — 有已选择且完成的运行则恢复其详情，再结束load。

```javascript
  }
```

`L20` — detail取得指定run的汇总，并枚举可选评价时期。

```javascript
  async function detail(id){
```

`L21` — detail取得指定run的汇总，并枚举可选评价时期。

```javascript
    selected=id;const r=await api('/api/ml/'+id),available=[...new Set(r.summaries.map(s=>s.scope))];
```

`L22` — 构造指标、十分组、覆盖、限制、下载和运行身份区域，解释信号收益不是账户净值。

```javascript
    $('ml-result').innerHTML=`<div class="panel"><h2>${esc(r.name)}</h2><div class="toolbar"><label>评价时期<select id="ml-scope">${available.map(s=>`<option value="${esc(s)}" ${s==='test'?'selected':''}>${esc(scopes[s]||s.replace('test_','测试 '))}</option>`).join('')}</select></label></div><div id="ml-scores"></div><h3>预测十分组 · 最高评分在第 1 组</h3><div id="ml-quantiles"></div><p class="note">Top相对收益是每个采样日入选股票的五日平均收益减候选池平均收益，再按日期取平均；没有复合成年化或账户净值。</p><details><summary>候选样本与标签覆盖</summary>${table(['区间','候选行','采样日期','有标签行'],r.counts.map(c=>[esc(c.split),esc(c.rows),esc(c.dates),esc(c.observed_labels)]))}<p>资格先按当时信息确定；缺失未来标签仍保留在排名中。小市值口径是已公告股本估算。</p></details><details><summary>研究限制</summary><ul>${r.limitations.map(v=>`<li>${esc(v)}</li>`).join('')}</ul></details><div class="toolbar"><a href="/api/ml/${esc(id)}/download?file=REPORT.md">下载报告</a><a href="/api/ml/${esc(id)}/download?file=predictions.parquet">下载逐股预测</a><a href="/api/ml/${esc(id)}/download?file=dataset.json">特征和标签定义</a></div><p class="note">运行 ${esc(id)} · 快照 ${esc(r.snapshot_id)}</p></div>`;
```

`L23` — 定义时期切换后的渲染函数，筛选相应scope指标。

```javascript
    const render=()=>{const scope=$('ml-scope').value,rows=r.summaries.filter(s=>s.scope===scope);
```

`L24` — 绘制5种方法的RankIC、区间、Top、相对小市值及标签覆盖。

```javascript
      $('ml-scores').innerHTML=table(['模型/对照','RankIC','95%块区间','Top相对池/期','相对小市值/期','入选标签覆盖'],rows.map(s=>[esc(names[s.model]||s.model),num(s.rank_ic),s.rank_ic_interval.map(num).join(' 至 '),pct(s.top_excess),pct(s.top_excess_vs_size),pct(s.top_label_coverage)]));
```

`L25` — 测试年份映射回test十分组，因为当前分组JSON只有整段结果。

```javascript
      const split=scope.startsWith('test')?'test':scope;
```

`L26` — 绘制模型和对照的10组相对池收益，最高组为1。

```javascript
      $('ml-quantiles').innerHTML=table(['模型',...Array.from({length:10},(_,i)=>'第'+(i+1)+'组')],Object.keys(names).map(m=>[esc(names[m]),...Array.from({length:10},(_,i)=>pct(r.quantiles.find(q=>q.split===split&&q.model===m&&q.quantile===i+1)?.excess))]));
```

`L27` — 切测试年份时补充分组表展示全部测试期的明确说明。

```javascript
      if(scope.startsWith('test_'))$('ml-quantiles').innerHTML+='<p class="note">十分组表展示全部测试期；上方指标展示所选年份。</p>';
```

`L28` — 绑定时期选择事件，首次立即绘制，再结束detail。

```javascript
    };$('ml-scope').onchange=render;render();
```

`L29` — 绑定时期选择事件，首次立即绘制，再结束detail。

```javascript
  }
```

`L30` — 供任务页面展示kind=ml实验、固定假设、核查范围、结果跳转或失败原因。

```javascript
  function experiment(e){return `<article class="task-experiment"><b>${esc(e.proposal.spec.name)}</b><p>${esc(e.proposal.hypothesis)}</p><p>${esc(e.status)} · 机器学习信号分析</p>${e.result?`<p class="note">${esc(e.result.verification_scope)}</p><button class="text-button" data-task-ml="${esc(e.run_id)}">查看模型基准 ↗</button>`:`<p>${esc(e.error||'固定配置与代码已登记')}</p>`}<details><summary>学习设计</summary><pre>${esc(JSON.stringify(e.proposal.spec,null,2))}</pre></details></article>`;}
```

`L31` — 从研究任务打开ML结果，记住run并切换页面。

```javascript
  async function open(id){selected=id;activate();}
```

`L32` — 暴露初始化、加载、打开和实验卡片方法；回调由原app传入。

```javascript
  return {init(a,p,s){api=a;activate=p;started=s;},load,open,experiment};
```

`L33` — 执行闭包并保存接口到window。

```javascript
})();
```

**src/researchops/service.py · 本次集成新增行**

复用已有研究协议：定位ML运行目录，catalog报告能力，baseline/context识别ML；register/recover/submit/review按kind走ML冻结与核查，账户路径继续走原核查。

`L58` — 复用已有研究协议：定位ML运行目录，catalog报告能力，baseline/context识别ML；register/recover/submit/review按kind走ML冻结与核查，账户路径继续走原核查。 当前差异位于：class Research:。

```python
        account = self.root / "runs" / run_id
```

`L59` — 复用已有研究协议：定位ML运行目录，catalog报告能力，baseline/context识别ML；register/recover/submit/review按kind走ML冻结与核查，账户路径继续走原核查。 当前差异位于：class Research:。

```python
        ml = self.root / "ml_runs" / run_id
```

`L60` — 复用已有研究协议：定位ML运行目录，catalog报告能力，baseline/context识别ML；register/recover/submit/review按kind走ML冻结与核查，账户路径继续走原核查。 当前差异位于：class Research:。

```python
        return ml if not account.exists() and ml.exists() else account
```

`L106` — 登记预先固定设计并冻结输入/代码，随后执行；不是在页面调测试冠军。

```python
                "registered_ml_analysis",
```

`L126` — 检测kind=ml，把该分支交给学习处理；账户仍走现有分支。

```python
                if result.get("kind") == "ml":
```

`L127` — 让接续任务定位既有ML运行，区分它与资金账户结果。

```python
                    baselines.append({"kind": "ml", "run_id": rid, "name": result["name"], "summaries": result["summaries"], "counts": result["counts"]})
```

`L128` — 复用已有研究协议：定位ML运行目录，catalog报告能力，baseline/context识别ML；register/recover/submit/review按kind走ML冻结与核查，账户路径继续走原核查。 当前差异位于：class Research:。

```python
                    continue
```

`L167` — 检测kind=ml，把该分支交给学习处理；账户仍走现有分支。

```python
        if isinstance(proposal, dict) and proposal.get("kind") == "ml":
```

`L168` — 登记预先固定设计并冻结输入/代码，随后执行；不是在页面调测试冠军。

```python
            from .ml_experiments import register
```

`L169` — 沿用研究任务领取凭据与所有权；运行不得绕过登记与任务预算。

```python
            return register(self, session, proposal)
```

`L385` — 检测kind=ml，把该分支交给学习处理；账户仍走现有分支。

```python
        if ex["proposal"].get("kind") == "ml":
```

`L386` — 复用已有研究协议：定位ML运行目录，catalog报告能力，baseline/context识别ML；register/recover/submit/review按kind走ML冻结与核查，账户路径继续走原核查。 当前差异位于：class Research:。

```python
            from .ml_experiments import recover
```

`L387` — 复用已有研究协议：定位ML运行目录，catalog报告能力，baseline/context识别ML；register/recover/submit/review按kind走ML冻结与核查，账户路径继续走原核查。 当前差异位于：class Research:。

```python
            return recover(self, ex, receipt)
```

`L506` — 检测kind=ml，把该分支交给学习处理；账户仍走现有分支。

```python
                if ex["proposal"].get("kind") == "ml":
```

`L507` — 在提交/复核阶段核验登记ML输出和冻结身份；不使用账户核查器验证ML。

```python
                    from .ml_experiments import verify_completed
```

`L508` — 在提交/复核阶段核验登记ML输出和冻结身份；不使用账户核查器验证ML。

```python
                    verify_completed(self, ex)
```

`L509` — 复用已有研究协议：定位ML运行目录，catalog报告能力，baseline/context识别ML；register/recover/submit/review按kind走ML冻结与核查，账户路径继续走原核查。 当前差异位于：class Research:。

```python
                else:
```

`L510` — 复用已有研究协议：定位ML运行目录，catalog报告能力，baseline/context识别ML；register/recover/submit/review按kind走ML冻结与核查，账户路径继续走原核查。 当前差异位于：class Research:。

```python
                    verify(p)
```

`L520` — 检测kind=ml，把该分支交给学习处理；账户仍走现有分支。

```python
                if ex["proposal"].get("kind") == "ml":
```

`L521` — 在提交/复核阶段核验登记ML输出和冻结身份；不使用账户核查器验证ML。

```python
                    from .ml_experiments import verify_completed
```

`L522` — 在提交/复核阶段核验登记ML输出和冻结身份；不使用账户核查器验证ML。

```python
                    verify_completed(self, ex)
```

`L523` — 复用已有研究协议：定位ML运行目录，catalog报告能力，baseline/context识别ML；register/recover/submit/review按kind走ML冻结与核查，账户路径继续走原核查。 当前差异位于：class Research:。

```python
                else:
```

`L524` — 复用已有研究协议：定位ML运行目录，catalog报告能力，baseline/context识别ML；register/recover/submit/review按kind走ML冻结与核查，账户路径继续走原核查。 当前差异位于：class Research:。

```python
                    verify(p)
```

**src/researchops/store.py · 本次集成新增行**

提交时根据完成实验kind标记registered_ml_analysis或mixed_registered_experiments，避免把模型统计说成账户核查。

`L427` — 提交时根据完成实验kind标记registered_ml_analysis或mixed_registered_experiments，避免把模型统计说成账户核查。 当前差异位于：class Store:。

```python
            if completed:
```

`L428` — 提交时根据完成实验kind标记registered_ml_analysis或mixed_registered_experiments，避免把模型统计说成账户核查。 当前差异位于：class Store:。

```python
                kinds = {json.loads(r["result"]).get("kind", "account") for r in completed}
```

`L429` — 检测kind=ml，把该分支交给学习处理；账户仍走现有分支。

```python
                if kinds == {"ml"}:
```

`L430` — 登记预先固定设计并冻结输入/代码，随后执行；不是在页面调测试冠军。

```python
                    report["evidence_kind"] = "registered_ml_analysis"
```

`L431` — 检测kind=ml，把该分支交给学习处理；账户仍走现有分支。

```python
                elif "ml" in kinds:
```

`L432` — 登记预先固定设计并冻结输入/代码，随后执行；不是在页面调测试冠军。

```python
                    report["evidence_kind"] = "mixed_registered_experiments"
```

**src/technical/server.py · 本次集成新增行**

增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。

`L39` — 核对下载/代码版本身份，防止结果篡改或服务混用新旧源码。

```python
    from .artifacts import digest
```

`L40` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
    def ml_code():
```

`L41` — 核对下载/代码版本身份，防止结果篡改或服务混用新旧源码。

```python
        return {p.name: digest(p) for p in sorted((project / "src/mlresearch").glob("*.py"))}
```

`L42` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
    boot_ml_code = ml_code()
```

`L53` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
    def ml_folder(run_id):
```

`L54` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
        if not RUN_ID.fullmatch(run_id):
```

`L55` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
            raise ValueError("ML运行编号无效")
```

`L56` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
        p = root / "ml_runs" / run_id
```

`L57` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
        if not (p / "manifest.json").exists():
```

`L58` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
            raise ValueError("ML实验未完成或不存在")
```

`L59` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
        return p
```

`L60` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python

```

`L106` — 增加明确ML接口或下载路径；GET读取既有记录，POST固定入口才会启动任务。

```python
            if path == "/api/ml/plan":
```

`L107` — 登记预先固定设计并冻结输入/代码，随后执行；不是在页面调测试冠军。

```python
                from ..mlresearch.contracts import benchmark_proposal
```

`L108` — 登记预先固定设计并冻结输入/代码，随后执行；不是在页面调测试冠军。

```python
                return self.send(benchmark_proposal())
```

`L109` — 增加明确ML接口或下载路径；GET读取既有记录，POST固定入口才会启动任务。

```python
            if path == "/api/ml":
```

`L110` — 更新本地运行状态/进度，供原工作台轮询及完成后展示。

```python
                return self.send([json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "ml_runs").glob("*/status.json"), reverse=True)])
```

`L111` — 增加明确ML接口或下载路径；GET读取既有记录，POST固定入口才会启动任务。

```python
            mm = re.fullmatch(r"/api/ml/([^/]+)(?:/(download))?", path)
```

`L112` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
            if mm:
```

`L113` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                p = ml_folder(mm[1])
```

`L114` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                if mm[2]:
```

`L115` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                    name = query.get("file", "")
```

`L116` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                    manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
```

`L117` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                    if name not in manifest["artifacts"] or "/" in name or "\\" in name:
```

`L118` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                        raise ValueError("未归档的ML文件")
```

`L119` — 核对下载/代码版本身份，防止结果篡改或服务混用新旧源码。

```python
                    from .artifacts import digest
```

`L120` — 核对下载/代码版本身份，防止结果篡改或服务混用新旧源码。

```python
                    if digest(p / name) != manifest["artifacts"][name]:
```

`L121` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                        raise ValueError("ML文件完整性检查失败")
```

`L122` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                    return self.send((p / name).read_bytes(), "application/octet-stream")
```

`L123` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                return self.send(json.loads((p / "result.json").read_text(encoding="utf-8")))
```

`L273` — 增加明确ML接口或下载路径；GET读取既有记录，POST固定入口才会启动任务。

```python
                if not review_match and not campaign_action and self.path not in ("/api/research/campaigns", "/api/research/tasks", "/api/run", "/api/definition", "/api/batch-plan", "/api/batch", "/api/foundations", "/api/regimes", "/api/ml/benchmark"):
```

`L279` — 增加明确ML接口或下载路径；GET读取既有记录，POST固定入口才会启动任务。

```python
                if self.path == "/api/ml/benchmark":
```

`L280` — 固定工作台入口要求空payload，拒绝未显示的参数覆盖。

```python
                    if payload != {}:
```

`L281` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                        raise ValueError("基准入口不接受隐藏参数，修改设计请通过研究CLI登记")
```

`L282` — 核对下载/代码版本身份，防止结果篡改或服务混用新旧源码。

```python
                    if code_inventory() != boot_code or ml_code() != boot_ml_code:
```

`L283` — 更新本地运行状态/进度，供原工作台轮询及完成后展示。

```python
                        return self.send({"error": "代码已更新，请重启本地服务"}, status=409)
```

`L284` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                    with guard:
```

`L285` — 更新本地运行状态/进度，供原工作台轮询及完成后展示。

```python
                        if job["status"] == "running":
```

`L286` — 更新本地运行状态/进度，供原工作台轮询及完成后展示。

```python
                            return self.send({"error": "已有实验运行"}, status=409)
```

`L287` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                        task = research.create({"title": "日线机器学习固定基准", "question": "价量模型是否提供后续排序信息？", "rationale": "用户从工作台启动固定基准",
```

`L288` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                            "success_criteria": ["冻结执行并核查全部模型和对照"], "stop_criteria": ["一次登记执行，不搜索测试参数"],
```

`L289` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                            "max_experiments": 1, "scope": "ML信号分析；不修改成交与会计"})
```

`L290` — 沿用研究任务领取凭据与所有权；运行不得绕过登记与任务预算。

```python
                        session = research.store.claim(task["id"], "ml-workbench")
```

`L291` — 沿用研究任务领取凭据与所有权；运行不得绕过登记与任务预算。

```python
                        write_json_path = root / "sessions" / ("ml-ui-" + task["id"] + ".json")
```

`L292` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                        from .artifacts import write_json
```

`L293` — 沿用研究任务领取凭据与所有权；运行不得绕过登记与任务预算。

```python
                        write_json(write_json_path, session)
```

`L294` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                        job.clear()
```

`L295` — 更新本地运行状态/进度，供原工作台轮询及完成后展示。

```python
                        job.update(status="running", task_id=task["id"], messages=["登记机器学习基准"])
```

`L296` — 服务器后台线程执行登记流程，让页面可以轮询进度；不是独立研究复核。

```python
                    def work_ml():
```

`L297` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                        try:
```

`L298` — 登记预先固定设计并冻结输入/代码，随后执行；不是在页面调测试冠军。

```python
                            from ..mlresearch.contracts import benchmark_proposal
```

`L299` — 沿用研究任务领取凭据与所有权；运行不得绕过登记与任务预算。

```python
                            ex = research.register(session, benchmark_proposal())
```

`L300` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                            with guard:
```

`L301` — 更新本地运行状态/进度，供原工作台轮询及完成后展示。

```python
                                job["messages"].append("执行冻结训练、后续预测与排序核查")
```

`L302` — 沿用研究任务领取凭据与所有权；运行不得绕过登记与任务预算。

```python
                            outcome = research.execute(session, ex["id"])
```

`L303` — 运行已登记学习实验并检查完成状态，保留返回run_id。

```python
                            if outcome["status"] != "completed":
```

`L304` — 运行已登记学习实验并检查完成状态，保留返回run_id。

```python
                                raise ValueError(outcome.get("error", "ML执行失败"))
```

`L305` — 沿用研究任务领取凭据与所有权；运行不得绕过登记与任务预算。

```python
                            research.submit(session, {"summary": "固定ML基准执行完成，结果待独立复核", "findings": ["已保存模型、逐股预测及全部对照；不根据测试期挑选冠军"],
```

`L306` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                                "limitations": ["回顾性时间隔离，信号收益不是账户收益"], "next_steps": ["复核时间边界、标签覆盖与分时期排序"]})
```

`L307` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                            with guard:
```

`L308` — 运行已登记学习实验并检查完成状态，保留返回run_id。

```python
                                job.update(status="completed", ml_run_id=outcome["run_id"])
```

`L309` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                        except Exception as exc:
```

`L310` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                            try:
```

`L311` — 沿用研究任务领取凭据与所有权；运行不得绕过登记与任务预算。

```python
                                research.store.checkpoint(session, "ML基准失败，保留尝试与日志：" + str(exc), True)
```

`L312` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                            except ValueError:
```

`L313` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                                pass
```

`L314` — 增加ML查询/下载与固定基准POST；哈希白名单保护下载；有效请求经任务、租约、登记、冻结执行、提交；源码变化要求重启，任务进行中拒绝重复启动。 当前差异位于：def make_server(project, port=8765, root=None):。

```python
                            with guard:
```

`L315` — 失败留错误和任务交接记录；不把异常当成有效结果。

```python
                                job.update(status="failed", error=str(exc))
```

`L316` — 服务器后台线程执行登记流程，让页面可以轮询进度；不是独立研究复核。

```python
                    threading.Thread(target=work_ml, daemon=False).start()
```

`L317` — 更新本地运行状态/进度，供原工作台轮询及完成后展示。

```python
                    return self.send({"status": "running", "task_id": task["id"]}, status=202)
```

**src/technical/web.py · 本次集成新增行**

按依赖顺序将ml.js加入页面脚本，在app初始化调用它之前先定义接口。

`L10` — 按依赖顺序将ml.js加入页面脚本，在app初始化调用它之前先定义接口。 当前差异位于：def render_page(boot):。

```python
    script = "\n".join((assets / name).read_text(encoding="utf-8") for name in ["chart.js", "foundations.js", "regimes.js", "tasks.js", "controller.js", "ml.js", "app.js"])
```

**src/technical/assets/app.js · 本次集成新增行**

主页面切换加入ml，运行轮询完成时打开ml_run_id，并给ML组件注入API、激活页面和启动通知。

`L17` — 主页面切换加入ml，运行轮询完成时打开ml_run_id，并给ML组件注入API、激活页面和启动通知。 当前差异位于：。

```javascript
  function page(name){document.querySelectorAll('[data-page]').forEach(e=>e.hidden=e.dataset.page!==name);document.querySelectorAll('[data-view]').forEach(e=>e.classList.toggle('active',e.dataset.view===name));$('page-title').textContent={results:'结果与诊断',campaigns:'自动研究总控',tasks:'研究任务与交接',create:'定义策略',lab:'参数对照',foundations:'基础策略研究',regimes:'状态与组合研究',ml:'机器学习研究',decisions:'逐日决策',archive:'实验档案',legacy:'旧模板审计'}[name];if(name==='ml')window.mlWorkbench.load().catch(error);if(name==='tasks')window.taskWorkbench.load().catch(error);if(name==='campaigns')window.controllerWorkbench.load().catch(error);if(name==='archive')archive().catch(error);if(name==='legacy')legacy().catch(error);if(name==='lab')batchHistory().catch(error);if(name==='regimes')window.regimeWorkbench.load().catch(error);if(name==='foundations')window.foundationWorkbench.load().catch(error);}
```

`L68` — 失败留错误和任务交接记录；不把异常当成有效结果。

```javascript
  async function poll(){clearTimeout(pollTimer);try{const j=await api('/api/job'),running=j.status==='running';if(running)watchingJob=true;$('job').hidden=!running;$('run').disabled=running;$('batch-run').disabled=running;$('job-log').textContent=(j.messages||[]).slice(-6).join('\n');if(j.status==='completed'&&watchingJob){watchingJob=false;if(j.ml_run_id){await window.mlWorkbench.open(j.ml_run_id);}else if(j.regime_id){page('regimes');}else if(j.suite_id){page('foundations');await window.foundationWorkbench.load();}else if(j.batch_id){await loadBatch(j.batch_id);page('lab');}else if(j.run_id){await load(j.run_id);page('results');}}if(j.status==='failed'&&watchingJob){watchingJob=false;error(j.error);}pollTimer=setTimeout(poll,running?1200:5000);}catch(e){error(e);pollTimer=setTimeout(poll,5000);}}
```

`L73` — 主页面切换加入ml，运行轮询完成时打开ml_run_id，并给ML组件注入API、激活页面和启动通知。 当前差异位于：。

```javascript
  window.mlWorkbench.init(api,()=>page('ml'),()=>{watchingJob=true;poll();});
```

**src/technical/assets/index.html · 本次集成新增行**

增加机器学习导航按钮和对应页面容器，容器实际内容由ml.js加载。

`L3` — 更新本地运行状态/进度，供原工作台轮询及完成后展示。

```text
<body><aside><div class="brand"><span class="brand-mark">研</span>A 股研究工作台<small>RESEARCH / EVIDENCE / ITERATION</small></div><nav aria-label="主导航"><button data-view="results" class="active"><span>01</span> 结果与诊断</button><button data-view="campaigns"><span>◎</span> 自动研究总控</button><button data-view="tasks"><span>↗</span> 研究任务与交接</button><button data-view="foundations"><span>02</span> 基础策略研究</button><button data-view="ml"><span>ML</span> 机器学习研究</button><button data-view="regimes"><span>↗</span> 状态与组合研究</button><button data-view="lab"><span>03</span> 参数对照</button><button data-view="create"><span>04</span> 定义策略</button><button data-view="decisions"><span>05</span> 逐日决策</button><button data-view="archive"><span>06</span> 实验档案</button><button data-view="legacy"><span>↳</span> 旧模板审计</button></nav><p class="side-note"><b>让每一个结论可追溯</b><br>假设 → 实验 → 诊断 → 下一轮<br><span class="status-dot"></span> 本地研究 · V7</p></aside>
```

`L9` — 增加机器学习导航按钮和对应页面容器，容器实际内容由ml.js加载。 当前差异位于：。

```text
<section data-page="ml" hidden><div id="ml-content"></div></section>
```

**src/technical/assets/tasks.js · 本次集成新增行**

任务卡片遇ML调用ML展示，绑定ML结果按钮；baseline链接区分账户与ML结果。

`L22` — 任务卡片遇ML调用ML展示，绑定ML结果按钮；baseline链接区分账户与ML结果。 当前差异位于：window.taskWorkbench=(()=>{。

```javascript
  function bindRuns(){ $('task-detail').querySelectorAll('[data-task-ml]').forEach(b=>b.onclick=()=>window.mlWorkbench.open(b.dataset.taskMl).catch(fail)); $('task-detail').querySelectorAll('[data-task-run]').forEach(b=>b.onclick=()=>openRun(b.dataset.taskRun).catch(fail)); }
```

`L33` — 让接续任务定位既有ML运行，区分它与资金账户结果。

```javascript
      ${s.baseline_runs.length?`<details><summary>已有基线 · ${s.baseline_runs.length} 项</summary>${s.baseline_runs.map(id=>`<button class="text-button" ${c.baselines?.find(b=>b.run_id===id)?.kind==='ml'?'data-task-ml':'data-task-run'}="${esc(id)}">${esc(c.baselines?.find(b=>b.run_id===id)?.name||id)} ↗</button>`).join('<br>')}</details>`:''}
```

`L34` — 核对下载/代码版本身份，防止结果篡改或服务混用新旧源码。

```javascript
      <h3>登记的实验与证据</h3>${ex.length?ex.map(e=>{if(e.proposal.kind==='ml')return window.mlWorkbench.experiment(e);const m=e.result?.metrics||[],net=m.find(v=>v.scenario==='configured'),gross=m.find(v=>v.scenario==='zero_transaction_cost');return `<article class="task-experiment"><div class="panel-heading"><b>${esc(e.proposal.spec?.strategy?.name||e.id)}</b><span class="badge">${labels[e.status]}</span></div><p>${esc(e.proposal.hypothesis)}</p>${e.result?`<div class="task-metrics"><span>含成本收益<b>${pct(net?.total_return)}</b></span><span>零成本收益<b>${pct(gross?.total_return)}</b></span><span>含成本最大回撤<b>${pct(net?.max_drawdown)}</b></span></div><p class="note">${esc(e.result.data_start)} → ${esc(e.result.data_end)} · 通过文件核验与账户对账，不代表策略已有效。</p>${runButton(e.run_id)}`:`<p class="note">${esc(e.error||'结果完成后可下钻到账户、区间和逐笔证据。')}</p>`}<details><summary>预期、否证条件与冻结版本</summary><p>预期：${esc(e.proposal.expected_observation)}</p><p>否证：${esc(e.proposal.falsification)}</p><p class="note">${esc(e.id)}<br>快照 ${esc(e.proposal.snapshot_id)}<br>代码 ${esc(e.proposal.engine_hash)}<br>时限 ${esc(e.proposal.timeout_seconds)} 秒</p><pre>${esc(JSON.stringify(e.proposal.spec,null,2))}</pre></details></article>`;}).join(''):'<p class="empty muted">先明确假设与否证条件，再登记实验。此处不会按收益自动追逐最优参数。</p>'}
```

**docs/WORKER_GUIDE.md · 本次集成新增行**

增加原生kind=ml流程与证据类型的指引，链接操作指南。

`L87` — 完成后提交待其他角色复核，保留学习统计证据类型。

```text
登记执行器支持 `ResearchSpec` 账户回测，以及 `kind=ml` 的固定模型信号分析，后者的设计、登记、结果和核查见[机器学习研究指南](ML_RESEARCH_GUIDE.md)。ML结果标为 `registered_ml_analysis`，不冒充账户核查。其他纯特征统计等可通过`attach`导入分析包，允许零新增账户实验的负面成果进入submit/review；导入只是事后完整性归档，不等同通用分析执行、数值重现或事前登记。详情见总控指南。不要为了完成任务把重复回测伪装成新研究。源码副本隔离不等于环境容器：Python与第三方包版本被记录，但没有冻结操作系统或限制内存。当前主进程负责超时终止；若主进程本身被强杀，需检查子进程与收据后恢复。数据库和大数据证据不会随Git自动备份。
```
