从模型和数据的基本概念读起，请看[从零讲解：项目机器学习V1做了什么](ML_FROM_ZERO.md)，或打开[浏览器阅读版](ML_FROM_ZERO.html)；逐行实现见[代码阅读器](ML_CODE_WALKTHROUGH.html)。本页负责命令与操作流程。

机器学习研究层从已登记快照构建输入，用原有任务租约、预算、register/run、冻结执行和收据恢复。入口为工作台「机器学习研究」、tools/ml_research.py 和 tools/research.py。原成交和会计规则保持原实现。

默认基准先检验选股信息：每周最后交易日采样，预测未来五个市场交易日参考价衔接收益，减同日有标签候选池平均收益。固定训练 2020—2022、验证 2023、测试 2024—2026-09-24。训练跨越阶段末日的标签排除；验证不参与拟合、早停或模型选择，两种固定模型全部报告。所有这些历史已经被查看，这是对模型拟合的时间隔离。

16 个价量输入在 src/mlresearch/contracts.py 明确列出；未来标签、股票代码、日期和财报列不会因出现在表中而自动成为 X。岭回归采用训练段中位数填补和标准化，提升树采用训练段中位数填补；提升树关闭自动随机验证和早停。需要增加特征时更新白名单、公式及前缀稳定性测试。

共同候选池是当时主板已上市、非 ST、成交量为正、连续至少 60 个市场交易日历史、20 日均成交额至少 2000 万元，而且存在严格早于当日已公告、不超过 240 天报告期龄的有效股本。股本资格让小市值对照可在同池进行；默认模型不把市值作为输入。该约束意味着基准不是全部 A 股。可将 feature_set 改为 price_volume_size，作为另行登记的信息组实验。

对照为小市值、五日反转、按固定种子和股票代码哈希的固定随机排序，以及整个候选池等权平均收益。随机对照是一个固定种子，不代表所有策略统一的收益率基准。

首先对全部当时合格候选预测、排名和分组，再计算已知标签的结果与覆盖率。不会删除未知未来收益后再挑 Top100。训练需要可观测标签；其缺失可能非随机，无法凭这个处理消除退市或供应商缺口偏差。阶段末尾不成熟的标签会排除，也不会填成零。

结果保存到 data/technical/ml_runs/<run-id>：

| 文件 | 含义 |
|---|---|
| spec.json / environment.json | 固定配置、包版本与线程上限 |
| panel.parquet / dataset.json | 特征、独立标签、可知时间、分段、公式与覆盖 |
| ridge.joblib / hist_gbdt.joblib | 包含预处理的已拟合模型；仅加载本系统自己生成且核验哈希的文件 |
| predictions.parquet | 全部候选逐股预测、对照分数与标签 |
| date_metrics.parquet | 每个采样日期和模型的 IC/RankIC、Top 相对池收益、分组价差和标签覆盖 |
| quantile_dates.parquet / quantiles.csv | 十分组日期记录与分段均值 |
| result.json / REPORT.md | 训练、验证、测试及测试年份汇总 |
| manifest.json | 结果文件哈希、快照与代码身份 |

Top 相对池收益是每期入选股票有标签结果的平均收益减同日池平均收益，再按日期平均。分组价差是预测最高组减最低组。这些是学习诊断，不复合成年化或伪称账户收益。四个采样日期的循环块区间是逐项诊断，不校正全部研究尝试和多重比较。

登记方式与旧任务兼容。从 lhb_backtest 运行：

```powershell
..\.venv\Scripts\python.exe tools/ml_research.py plan --output research/你的任务/proposal.json
..\.venv\Scripts\python.exe tools/research.py register --session data/technical/sessions/你的凭据.json --file research/你的任务/proposal.json
..\.venv\Scripts\python.exe tools/research.py run exp-登记返回编号 --session data/technical/sessions/你的凭据.json
..\.venv\Scripts\python.exe tools/ml_research.py list
..\.venv\Scripts\python.exe tools/ml_research.py show 实际运行编号
..\.venv\Scripts\python.exe tools/ml_research.py verify 实际运行编号
```

运行前仍需 create/claim 任务，固定假设、预期、否证条件、日期、快照和预算。plan 生成 kind=ml 的原生 proposal。不要在同一任务外调用 runner 绕开预算。新 worktree 在 research.py 和 ml_research.py 中显式指定 --root 原研究库绝对路径；源行情和快照只读，ML 结果写入该研究库。

工作台「登记并运行基准案例」建立一个一次执行预算的任务，领取凭据仅保存在服务端，完成后提交待复核。固定入口不接受隐藏参数覆盖；自定义实验经 CLI 登记。源码更新后重启本地工作台。

原生 ML 结果由核查器检查文件身份、市场日历标签抽样、分段边界、训练填补统计、保存模型预测抽样，以及逐日排序/Top 指标复算。实验提交类型为 registered_ml_analysis，区别于 audited_experiment 账户核查和分析完成后的 attach 导入。数值核查不证明原供应商数据正确、训练程序具有外部认证，或策略未来有效；独立研究复核另行进行。模型序列化并不冻结操作系统，环境和版本必须同时保留。

完整日线研究现已补齐：原模型账户、周期对齐目标、四组特征消融、加市值、十规模组内对照、打乱标签、年度扩展重训及有限验证期参数选择。入口和已核查结果见[真实账户的人话报告](ML_ACCOUNT_RESULTS.html)，调用关系见[新增代码阅读](ML_COMPLETION_CODE.html)。原V1教程和原运行保持历史版本；逐行V1阅读器对应最初版本。

```powershell
..\.venv\Scripts\python.exe tools/ml_research.py plan --complete --source-run 20261003T165732-991ae261 --output research/你的任务/proposal.json
# 先create/claim，再按上方register/run执行。相同设计已有结果时复用，不重跑。
```

完整矩阵的spec.study为complete，仍由kind=ml登记和冻结执行；study_plan把19个策略、最多30次新拟合和两种账户费用情景一并冻结。实际首轮运行20261003T195515-13624598，登记实验exp-9eff17e74d89：20次新拟合、复用2个原模型、38个账户情景，所有尝试保留。父ML清单链接并核验每个原生账户的清单及预测来源；子账户仍可通过verify_technical.py审计现金/股数/开盘价格/费用。

在完整运行目录中，open_labels.parquet记录下一开盘到下一次周调仓开盘的参考价衔接目标。末端尚不成熟的周保留在推断和交易中；只从训练及学习评价排除。fits.json保存训练截止及最大标签末日、特征列和参数，assignments.json将每段预测绑定保存模型，tuning.json保留6个验证期候选。验证期选择只按日期平均RankIC，年度重训只使用截至上年末成熟标签。已看历史不会因年度重训变成真正未见数据。

account_index.json链接19个data/technical/runs账户，保存两种情景的指标与年度结果。执行器通过已有runner的prepared接口提供冻结评分、选中目标和独立信号说明；账户Strategy中的lookback不参与ML选股。全部候选在父predictions.parquet可查，子decisions.parquet只记录入选目标。零费用仍独立重算现金和股数。完整ML提交类型仍为registered_ml_analysis，其verification_scope和accounts字段明确包含已审计子账户；不能把旧V1信号运行说成已审计账户。

原案例的正五日排序没有转化为正账户收益。看到账户差异后追加的时点诊断是事后只读分析，单独归档，非新事前假设。在端点恰好匹配的117周中，原树选中股票入场前隔夜的相对涨幅，解释了相当一部分闭盘学习目标与开盘持有结果的差距。不要通过事后把买价改回已用于评分的当日收盘价制造收益。

当前仍未覆盖分钟/L2输入、实时推断服务、实盘连接、长期无人值守训练或任意超参搜索。现有的是固定有限研究矩阵、年度批量扩展重训和6候选验证期选择。69项相关回归及桌面/窄屏工作台验收通过；数值核查是实现者自检，独立研究复核尚待另一角色进行。

## 2026-10-06 源码整合与日历契约修复

主源码现整合此前冻结实验中的 `fixed_blend` 与 `financial_context` 扩展，包括登记时的来源约束、预算、冻结和核查器。原失败实验仍保留，不通过改写旧文件变成成功。

市场代理特征只使用当前行情快照实际覆盖的交易日，截止于研究结束日。更长的原市场日历继续用于标签和其他原有逻辑；需要的60交易日窗口缺失、内部缺行情或决策日期越界都会拒绝，不能填零。财务审计器独立构建并检查同一覆盖契约。

本轮验证属于工程：合成数据回归与既有日历元数据核对，不是重新训练真实模型或完成新的账户实验。下一轮真实财务研究须用修复后的代码重新登记，保留原失败预算记录，不覆盖旧结论。固定评分组合与财务扩展都需要有效且核查通过的来源运行，不能在空公开工作区引用原作者本机 ID 直接执行。

方法依据：[scikit-learn 防止预处理泄漏的说明](https://scikit-learn.org/stable/common_pitfalls.html)、[HistGradientBoostingRegressor 官方参数说明](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)。本地实际包版本记录在每次 environment.json。
