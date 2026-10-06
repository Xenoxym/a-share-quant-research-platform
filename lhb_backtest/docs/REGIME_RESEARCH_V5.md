# V5：回撤图与市场状态配置研究

交付报告：研究判断与全部结果（本机证据，未随公开源码提供）。

本地工作台新增“状态与组合研究”。比较12个预设账户的实际成交资金路径，逐年展示只用过去数据拟合的2/3状态HMM，提供概率、目标配置、收益、回撤、年度表现、联合增量区间以及进入原始交易案例的入口。研究复核与3个事后敏感性实验通过单独JSON接入，不修改首轮冻结结果。

区间回撤SVG按容器实际宽度绘制，独立高度190像素，轴上限0%。ResizeObserver处理缩放与页签显示；悬停和键盘读数沿用同一日期映射。区间计算保留前一交易日权益作为起始高点。

运行预先固定的12个对照：

```powershell
..\.venv\Scripts\python.exe -m pip install -e ".[research]"
..\.venv\Scripts\python.exe tools/technical_research.py regimes
```

`research`可选依赖新增hmmlearn 0.3.3。每次运行冻结源码、输入、模型参数及依赖版本；模型逐年只用过去最多756日拟合，交易概率由前向过滤计算。`state_breadth_threshold`可用于独立检验简单趋势/广度规则；首轮固定0.4，修改是新的实验。配置策略要求2022年起评估，两个分支共用252日历史、财报资格和已发生送转桥接，参数校验拒绝含糊的分支过滤。

核心文件：

- `src/technical/regimes.py`：市场特征、年度HMM、逐日过滤与分支目标合并。
- `src/technical/regime_lab.py`：冻结协议、配对账户、年度与联合增量统计。
- `src/technical/assets/regimes.js`：对照、状态概率、配置与研究判断界面。
- `tests/test_regimes.py`：修改未来数据不影响历史、Bayes过滤、年度训练截止、重叠持仓合并、双分支持仓上限与空仓退出等边界测试。
- `tools/verify_regimes_ui.py`：使用专用Edge调试配置验证桌面/手机、悬停回撤与API一致、曲线开关、模型和逐日决策联动。
- `research/regime_allocation_20260927/audit_and_analyze.py`：独立复算模型概率、账本与配置证据。

研究表述边界：滚动拟合避免未来模型参数进入当日交易，但无法消除研究者已查看历史带来的选择偏差。高波动状态概率不是未来下跌概率。固定组合、半仓与频率对照必须保留，不能因动态模型名称复杂就优先采用。
