# A-Share Quant Research Platform

**项目当前入口：** [架构、全部工作包、现在进度和下一步](lhb_backtest/docs/PROJECT_MAP.md)（[本机网页](lhb_backtest/docs/PROJECT_MAP.html)）。

面向 A 股的本机量化研究平台：数据快照、特征和模型研究、组合账户回测、证据核查，以及有预算的研究任务与接力。

项目正在从早期龙虎榜事件回测扩展为完整研究系统。当前主要产品是 `lhb_backtest/src/technical` 工作台、`src/mlresearch` 机器学习层和 `src/researchops` 任务层。长期蓝图中的自动 Alpha 搜索、更多模型与数据治理能力仍需逐项建设。

- [安装、入口与当前能力](lhb_backtest/README.md)
- [长期项目纲领](lhb_backtest/docs/PROJECT_CHARTER.md)
- [研究雷达](lhb_backtest/docs/RESEARCH_RADAR.md)
- [系统蓝图](lhb_backtest/docs/ML_SYSTEM_BLUEPRINT_20261005.md)
- [Worker 入场与证据规则](lhb_backtest/docs/WORKER_GUIDE.md)
- [公开交付与验证边界](lhb_backtest/docs/PUBLIC_DELIVERY.md)

应用目录保留 `lhb_backtest`，Python 导入保留 `src.*` / `compat.*`，以兼容已有实验；产品和发行包名称已扩展为 `a-share-quant-research-platform`。原始行情、任务数据库、模型和私有研究结果不随公开源码交付。公开许可证尚待项目所有者选择；当前没有 MIT 或其他开源许可证授权。

- [项目结构与总体蓝图审查](lhb_backtest/docs/LOCAL_STRUCTURE_REVIEW_20261006.md)
- [当前建设与接续进度](lhb_backtest/docs/BUILD_PROGRESS_20261006.md)
