# 成熟开源项目接入：锁定版本后的实际兼容性

本阶段只读取官方发行元数据、标签及许可文本，并与本机版本比较；没有安装第三方环境、调用模型或运行账户。它回答哪些项目适合先做隔离验证，不证明它们已接入。

## 当前环境

Windows x64、Python3.12.2；NumPy2.4.6、pandas3.0.3、scikit-learn1.9.0、pyarrow24.0.0。冻结旧实验所用环境保持。

|项目／固定发行|Windows/Python3.12安装包|已安装依赖的冲突|未装基础依赖数|下一步|
|---|---|---|---:|---|
|pyqlib 0.9.7|有匹配wheel|声明约束未检出冲突；不是运行通过|12|优先隔离验证数据/特征接口|
|vnpy 4.5.0|有匹配wheel|声明约束未检出冲突；不是运行通过|8|与Qlib并列小尺寸比较；alpha含额外依赖|
|nautilus_trader 1.231.0|有匹配wheel|pyarrow>=25.0.0，本机24.0.0；tqdm<5.0.0,>=4.68.4，本机4.67.3；tzdata>=2026.3，本机2026.2|5|保留跨引擎路线，独立环境验证A股语义|
|rqalpha 6.4.1|没有匹配wheel；不等于不可安装|pandas<3.0.0,>=1.0.5，本机3.0.3|8|隔离环境及许可/固定源码确认后再接入|

表格按实际PEP508条件及本机wheel平台标签检查；vnpy匹配的是py3-none-any通用wheel，不能只按文件名有无cp312判断可用。未装依赖数仅基础依赖，不是全部递归依赖的下载体积或安装成本。没有运行完整依赖解析器，也没有做运行时ABI/API验证。

## 具体影响

Qlib0.9.7有Windows CPython3.12 wheel，基础依赖没有检出已安装版本约束冲突，但缺12项基础依赖。它的RL额外依赖要求NumPy低于2，而本机NumPy是2.4.6；因此RL路线必须独立环境，不能用基础依赖检查冒称RL可用。[官方发行](https://pypi.org/project/pyqlib/0.9.7/)

vnpy4.5.0基础依赖包括GUI和TA-Lib；alpha额外依赖还包含LightGBM、Polars及Torch。先研究其alpha数据/特征接口，不能因为仓库存在alpha目录就假设接入成本只有一行import。[官方发行](https://pypi.org/project/vnpy/4.5.0/)

NautilusTrader1.231.0稳定发行匹配本机Python3.12；此前观察开发分支README的Python要求不能替代这个发行结果。当前latest文档讲2.x，必须与固定1.x发行分开。稳定发行要求pyarrow≥25、tqdm≥4.68.4及tzdata≥2026.3，本机三者不满足。本次没有验证完整A股T+1、涨跌停和公司行动语义。[官方发行](https://pypi.org/project/nautilus_trader/1.231.0/) · [当前安装文档](https://nautilustrader.io/docs/latest/getting_started/installation/)

RQAlpha6.4.1要求pandas<3，本机3.0.3不满足；发行没有匹配wheel，只看到源码发行。尝试6.4.1与v6.4.1两个官方标签均404，未确认固定发行源码提交和该发行许可；不能把之前阅读的主分支许可代作固定发行许可。[官方发行](https://pypi.org/project/rqalpha/6.4.1/)

## 来源固定与许可事实

Qlib v0.9.7固定到da920b7f954f48ab1bb64117c976710de198373e；vnpy4.5.0到c6e231caf32b7fc97e6459817fff66458cf7e7c4；NautilusTrader v1.231.0标签解析到27a8e54e7ac3c57d6cbf8891f0283dfbaee97317。三个固定提交的许可文本分别标示MIT、MIT和LGPL v3。RQAlpha该发行固定源码及许可尚未确认，暂缓采用；本项目自己的许可证仍未决定。许可标识是源码读取事实，不替代具体发布方式的许可审阅。

[Qlib固定源码](https://github.com/microsoft/qlib/tree/da920b7f954f48ab1bb64117c976710de198373e) · [vnpy固定源码](https://github.com/vnpy/vnpy/tree/c6e231caf32b7fc97e6459817fff66458cf7e7c4) · [Nautilus固定源码](https://github.com/nautechsystems/nautilus_trader/tree/27a8e54e7ac3c57d6cbf8891f0283dfbaee97317)

## 如何继续，怎样证明接入真的有用

下一步选Qlib和vnpy.alpha的小型隔离验证：使用12只股票、120个交易日和8个预先固定基础运算，同输入比较数值、时点规则、耗时及内存。最多2个数值进程起步；安装方案先固定依赖、许可、下载/磁盘预算，保持当前生产环境与旧作业不变。通过后才做固定小账户语义对照，再决定采用哪一套主干。

保留本项目历史可知数据、龙虎榜及板形态、独立账户核查与研究记账；外部项目优先承担它已成熟实现的部分。并不删除旧证据，也不因为外部框架成熟就假设本数据不会泄漏或存在盈利策略。完整重任务仍需要通过现有内存门槛。

## 本阶段限制和实际路径

读取四个PyPI发行 → 按实际平台/依赖比较 → 固定三个源码提交和许可 → 一个过大的GitHub commit响应被预算截断并保留失败，后用小标签接口 → RQ两个候选标签404明确保留 → 生成兼容性矩阵。总元数据请求14次，低于16次；初始部分响应字节数未留存，不能声称12MiB累计流量计量被完整验证。0新增市场拟合、0账户、0安装、0外部费用。

这仍是元数据兼容性研究；运行速度、实际内存节省、A股规则兼容及收益改善尚未得到证实。

[上一阶段开源架构审查](ML_OPEN_SOURCE_REUSE_REVIEW_20261010.html) · 实际兼容矩阵（证据保留本机）

[当前总览](PROJECT_MAP.html) · [最大蓝图](ML_SYSTEM_BLUEPRINT_20261005.html) · [完整工作包计划](CODEX_PROJECT_DELIVERY_PLAN_20261005.html) · [详细建设记录](BUILD_PROGRESS_20261006.md)


显示修订：旧冻结HTML单字渲染错误保留；本修订按行渲染，不改变研究指标。
