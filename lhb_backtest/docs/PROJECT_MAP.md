# 项目总览：当前架构、全计划和进度

更新（UTC）：2026-10-13T22:13:07.460393+00:00 · 当前工作包：五项缺口接续：真实132列准备完成；矩阵→训练→账户与学习证据恢复已跑通

我们建设的是：把已有市场信息变成可检验的投资方法，再用真实资金账户的收益和回撤判断它是否值得继续。当前软件模块已具备研究闭环的基本接线；真实账户计算与独立接受的进度分别见下文。

## 从这里看整个计划

- [五项缺口接续施工：全量准备与自动接线](FOUNDATION_CONTINUATION_20261013.md)（[网页](FOUNDATION_CONTINUATION_20261013.html)）
- [五项缺口首批施工：实际改动、结果和未完成](FOUNDATION_CONSTRUCTION_20261013.md)（[网页](FOUNDATION_CONSTRUCTION_20261013.html)）
- [内存与自动接线：实际改动](EXECUTION_EFFICIENCY_20261012.md)（[网页](EXECUTION_EFFICIENCY_20261012.html)）
- [接下来怎样用Codex：一个主Chat、持续规划、按需协作](CODEX_PROJECT_WORKFLOW.md)（[网页](CODEX_PROJECT_WORKFLOW.html)）
- [执行系统重构：做法、实测、清理和缺口](EXECUTION_REFACTOR_RESULTS_20261012.md)（[网页](EXECUTION_REFACTOR_RESULTS_20261012.html)）
- [工程研究与施工方案：来源、架构、验收](EXECUTION_REFACTOR_PLAN_20261012.md)（[网页](EXECUTION_REFACTOR_PLAN_20261012.html)）
- [24小时成本复盘：时间、Token与返工怎样减少](ML_TIME_TOKEN_POSTMORTEM_20261011.md)（[网页](ML_TIME_TOKEN_POSTMORTEM_20261011.html)）
- [64列、额外单列选股、程序错误与24小时实际工作](ML_SCREEN_MATURITY_DISCOVERY_20261011.md)（[网页](ML_SCREEN_MATURITY_DISCOVERY_20261011.html)）
- [64组合特征算出来后，还缺哪些收益验证](ML_COMBINED_FEATURE_PROGRESS_20261011.md)（[网页](ML_COMBINED_FEATURE_PROGRESS_20261011.html)）
- [旧评分进入新账户：16项实际存档验证与持仓规则例子](ML_V2_ACCOUNT_ARCHIVE_20261011.md)（[网页](ML_V2_ACCOUNT_ARCHIVE_20261011.html)）
- [旧模型评分怎样复用、还缺什么](ML_SCORE_REUSE_20261011_04.md)（[网页](ML_SCORE_REUSE_20261011_04.html)）
- [首次完整68项小MLP的真实亏损账户报告](ML_MLP_ACCOUNT_RESULTS_20261011_02.md)（[网页](ML_MLP_ACCOUNT_RESULTS_20261011_02.html)）
- [小型神经网络的真实工程检查（尚无行情收益）](ML_MLP_ENGINEERING_20261011_02.md)（[网页](ML_MLP_ENGINEERING_20261011_02.html)）
- [完整68项数据准备容量实测](ML_FULL68_PREPARATION_CAPACITY_20261011.md)（[网页](ML_FULL68_PREPARATION_CAPACITY_20261011.html)）
- [排名模型真实账户结果](ML_RANK_ACCOUNT_RESULTS_20261011.md)（[网页](ML_RANK_ACCOUNT_RESULTS_20261011.html)）
- [这次接续的已完成与未完成，用人话说明](ML_RESEARCH_RESUME_STATUS_20261011.md)（[网页](ML_RESEARCH_RESUME_STATUS_20261011.html)）
- [排名实验：准备成功但训练尚未启动](ML_RANK_PRETRAIN_RESOURCE_FAILURE_20261011.md)（[网页](ML_RANK_PRETRAIN_RESOURCE_FAILURE_20261011.html)）
- [成熟六公式接入现有特征协议：51项实际集成](ML_MATURE_FEATURE_ADAPTER_RESULTS_20261011.md)（[网页](ML_MATURE_FEATURE_ADAPTER_RESULTS_20261011.html)）
- [26项小准备等价：已验证的范围与全量限制](ML_DATA_PREPARATION_EQUIVALENCE_20261011.md)（[网页](ML_DATA_PREPARATION_EQUIVALENCE_20261011.html)）
- [成熟六公式接口：72组真实数值及未来前缀通过](ML_OPEN_SOURCE_EXPRESSION_API_RESULTS_20261011.md)（[网页](ML_OPEN_SOURCE_EXPRESSION_API_RESULTS_20261011.html)）
- [隔离安装真实通过：验收检查故障与明确接续](ML_OPEN_SOURCE_EXPRESSION_INSTALL_AND_CHECK_FAILURE_20261011.md)（[网页](ML_OPEN_SOURCE_EXPRESSION_INSTALL_AND_CHECK_FAILURE_20261011.html)）
- [安装前失败：错误来自自己的wheel布局检查](ML_OPEN_SOURCE_WHEEL_LAYOUT_FAILURE_20261010.md)（[网页](ML_OPEN_SOURCE_WHEEL_LAYOUT_FAILURE_20261010.html)）
- [真实Qlib接口首次试点：下载前失败及原因](ML_OPEN_SOURCE_EXPRESSION_FIRST_TRIAL_20261010_02.md)（[网页](ML_OPEN_SOURCE_EXPRESSION_FIRST_TRIAL_20261010_02.html)）
- [固定Qlib八内核：真实数值与时点兼容结果](ML_OPEN_SOURCE_KERNEL_RESULTS_20261010.md)（[网页](ML_OPEN_SOURCE_KERNEL_RESULTS_20261010.html)）
- [首次Qlib实测：导入成功、取数错误及明确接续](ML_OPEN_SOURCE_KERNEL_FIRST_TRIAL_20261010.md)（[网页](ML_OPEN_SOURCE_KERNEL_FIRST_TRIAL_20261010.html)）
- [真实半批训练目标账户结果与接续](ML_TARGET_PARTIAL_ACCOUNT_RESULTS_20261010_02.md)（[网页](ML_TARGET_PARTIAL_ACCOUNT_RESULTS_20261010_02.html)）
- [固定开源发行与当前环境：实际兼容性](ML_OPEN_SOURCE_COMPATIBILITY_20261010_02.md)（[网页](ML_OPEN_SOURCE_COMPATIBILITY_20261010_02.html)）
- [复核原意见已存档、目录现象与接续限制](ML_REVIEW_RECORDS_AND_RESUME_20261010.md)（[网页](ML_REVIEW_RECORDS_AND_RESUME_20261010.html)）
- [成熟开源项目复用审查：哪些在重复、接下来怎样采用](ML_OPEN_SOURCE_REUSE_REVIEW_20261010.md)（[网页](ML_OPEN_SOURCE_REUSE_REVIEW_20261010.html)）
- [最初要求逐项对照：已完成、部分完成、尚未开始](ML_PROJECT_REQUIREMENTS_AUDIT_20261010.md)（[网页](ML_PROJECT_REQUIREMENTS_AUDIT_20261010.html)）
- [68项训练为何停止及实际资源需求](ML_68_FEATURE_RESOURCE_STOP_20261010.md)（[网页](ML_68_FEATURE_RESOURCE_STOP_20261010.html)）
- [完整最大系统蓝图](ML_SYSTEM_BLUEPRINT_20261005.md)（[网页](ML_SYSTEM_BLUEPRINT_20261005.html)）
- [完整工作包及交付计划](CODEX_PROJECT_DELIVERY_PLAN_20261005.md)（[网页](CODEX_PROJECT_DELIVERY_PLAN_20261005.html)）
- [逐阶段详细建设记录](BUILD_PROGRESS_20261006.md)
- [长期目标与持续探索准则](PROJECT_CHARTER.md)（[网页](PROJECT_CHARTER.html)）
- [本批实验的人话说明](ML_E19_REAL_MATRIX_20261008.md)（[网页](ML_E19_REAL_MATRIX_20261008.html)）
- [这次账户核查与研究路径](ML_E19_ACCOUNT_RECHECK_20261009.md)（[网页](ML_E19_ACCOUNT_RECHECK_20261009.html)）
- [本批全部账户结果](ML_E19_ACCOUNT_COMPARISON_20261009.md)（[网页](ML_E19_ACCOUNT_COMPARISON_20261009.html)）
- [模型保存分数学到了什么](ML_E19_SCORE_DIAGNOSIS_20261009.md)（[网页](ML_E19_SCORE_DIAGNOSIS_20261009.html)）
- [序列接口的人话说明](ALPHA_SEQUENCE_WINDOWS.md)（[网页](ALPHA_SEQUENCE_WINDOWS.html)）
- [训练目标与模型存档：人话说明](ML_TARGETS_AND_SAVED_MODELS_20261009.md)（[网页](ML_TARGETS_AND_SAVED_MODELS_20261009.html)）
- [公式组合研究的人话说明](ML_FORMULA_RESEARCH_20261009.md)（[网页](ML_FORMULA_RESEARCH_20261009.html)）
- [首次真实批次为何失败](ML_FORMULA_CAPACITY_20261009.md)（[网页](ML_FORMULA_CAPACITY_20261009.html)）
- [共享名单存储改造](ML_FORMULA_SHARED_STORAGE_20261009.md)（[网页](ML_FORMULA_SHARED_STORAGE_20261009.html)）

## 现在进行到哪里

完整132施工之后实际7fit、10账户作业/20情景；MLPplanned不算fit，费用$0、Token未知。

新增一次完整132列准备成功：439424成熟训练／105251预测样本，115.52秒，采样进程树RSS峰值4.73GiB；0行情fit／账户。v3固定矩阵→学习→账户真实合成小样本链及中断恢复8例通过。真实HGB学习证据189文件／有效载荷1,017,274,361字节备份19.41秒，隔离复制核对5.89秒；同盘副本、不启动恢复worker。

受影响范围回归58项通过／229.30秒，一次限定复核完成；本批公开CI待收尾。上一批源码34613aa双平台完整CI通过。五项没有全部完成；动态公式→矩阵、序列／图／自适应试点、未来封存、同条件小市值及收益稳健性仍缺。

另一路已记录工作（不计为本轮产物）：本轮数据获取：4股三年小时11,576根、2股近期5分钟576根，另9行因子；232项相关回归通过，四份完成收据均有来源价量差异且默认禁止研究读取。原MLP仍未启动、0新模型/账户、费用$0。

新有限阶段保留原两次准备失败；实际内存改善后一次完整准备成功。增加现有计划v3的固定矩阵学习适配步骤与选定学习证据本机备份／离线核对；只做一次限定独立工程复核，修正清单元数据也须计入字节帽。未新增市场模型或账户收益。

## 最简架构：六步就能理解

![当前最简功能架构](PROJECT_ARCHITECTURE_SIMPLE.svg)

## 略详细架构：每一步里有什么

![当前有用模块](PROJECT_ARCHITECTURE_DETAIL.svg)

图中只画已经实现的有用功能。工程验收表示程序与约束经过检查，不表示有市场优势；模块可用也不表示所有大规模数据路径都已认证。小型神经网络已接入并完成一次完整市场账户；图模型、遗传/RL/LLM自动公式搜索仍在后面的计划表。

## 一周例子

示例：周五收盘后，以当时可知的信息给每只合格股票打分；按预先固定的规则选股和分配资金；下一个交易日开盘尝试调仓，此后逐日计算账户。新学习协议默认预测两次计划调仓开盘之间的收益，不是某天的绝对价格，也不是直接训练最大回撤。固定随机对照中的11是种子，不是11只股票。E19已固定共同主板名单、周调仓、100只目标持仓及四个对照；具体实验说明见本批报告。

## 接下来要做什么

1. 收尾本批相关回归、精确发布和完整CI；不重复训练旧模型证明接线。
2. 复用表达式组装／分区合并函数，补公式产物→矩阵写出，先跑完整小样本。
3. 补严格同名单小市值对照，共同日期／成熟标签／成本／账户规则分别核验，以收益和回撤判断增量。
4. 选择一个序列模型或自适应公式有限试点，固定未来验证方案；已观察2026不称未见样本。
5. 扩展完整与异盘恢复前明确范围和授权位置；盘中数据另一路的来源价量差异仍须第二来源核对。

## 全部工作包：原计划与补充包

原编号表示设计和依赖，不是按编号顺序执行；E15—E18先补接线，E12—E14后补有限执行能力，详见建设记录。没有用完成包数给项目虚构完成百分比。

| 工作包 | 人话说明 | 当前状态 | 已做到什么 / 还缺什么 | 依赖 |
|---|---|---|---|---|
| [R0](LOCAL_STRUCTURE_REVIEW_20261006.md) | 保留旧项目和已有改动 | 已完成首轮 | 旧脏Git与私人证据保留；发布用独立导出仓库。 | 无 |
| [R1](PUBLIC_DELIVERY.md) | 决定哪些文件能公开 | 已完成首轮 | 精确清单及忽略规则，行情/数据库/私密会话留本机。 | R0 |
| [R2](PUBLIC_DELIVERY.md) | 依赖、安装和打包 | 部分完成 | 当前wheel已核查；高级模型依赖随对应工作包接入。 | R0 |
| [R3](PUBLIC_DELIVERY.md) | 跨系统的软件检查 | 已完成首轮 | Windows/Ubuntu CI通过；不证明真实策略有效。 | R1,R2 |
| [R4](WORKER_GUIDE.md) | 新人说明与协作规则 | 部分完成 | 入场与长期准则已保存；许可证尚未决定。 | R1 |
| [R5](PUBLIC_DELIVERY.md) | 发布到GitHub | 已完成首轮 | 公开仓库已建立并多次核查发布，后续持续维护。 | R3,R4 |
| [E00](BUILD_PROGRESS_20261006.md) | 修复财务市场日历问题 | 已完成工程验收 | 执行器与核查器日期范围、60日回看一致；0策略fit。 | 已有task-28ab0d6b9fe2 |
| [E01](ALPHA_CAUSAL_CONTRACTS.md) | 什么时候能知道什么 | 已完成工程验收 | 明确决策、信息可用、成交、标签结束时间，拒绝未来信息。 | E00,R0 |
| [E02](ALPHA_EVENT_SNAPSHOTS.md) | 龙虎榜原报告快照 | 已完成工程验收 | 原报告、席位、重复及窗口可追溯；披露时间仍是保守代理。 | E01 |
| [E03](ALPHA_EVENT_PANEL.md) | 日期×股票的统一名单 | 已完成工程验收 | 行情/事件对齐，缺数据≠没事件；原工程真实切片27日。 | E01,E02 |
| [E04](ALPHA_DAILY_BOARDS.md) | 连板与涨跌停形态 | 已完成工程验收 | 29项定义；日线不能辨别天地板的盘中顺序。 | E03 |
| [E05](ALPHA_LHB_FEATURES.md) | 龙虎榜与集中度特征 | 已完成工程验收 | 34项报告、150项决策投影；部分全缺失，数量不代表有效Alpha。 | E03 |
| [E06](ALPHA_FINANCIAL_MARKET.md) | 财务和市场环境 | 已完成工程验收 | 14项财务、12项市场；保留财务修订及市场覆盖限制。 | E01,E03 |
| [E07](ALPHA_FEATURE_REGISTRY.md) | 特征字典与组装 | 已完成工程验收 | 255项登记定义，工程矩阵221列；包含缺失列，并非221个盈利信号。 | E04,E05,E06 |
| [E08](ALPHA_EXPRESSION_CONTRACTS.md) | 安全组合公式语法 | 已完成工程验收 | 受限表达式组合基础信息，拒绝任意代码执行。 | E01,E07 |
| [E09](ALPHA_NUMERICAL_OPERATORS.md) | 组合公式的计算算子 | 已完成工程验收 | 排序/滞后/差分/滚动按真实交易日历计算。 | E08 |
| [E10](ALPHA_SHARED_CACHE.md) | 共享表达式与缓存 | 已完成工程验收 | 复用中间计算、识别损坏；原小批实测未证明缓存更快。 | E09 |
| [E11](ALPHA_BATCH_REGISTRATION.md) | 先登记再运行 | 已完成工程验收 | 冻结候选、数据、代码、预算；失败和取消留档。 | E07,R3 |
| [E12](ALPHA_BATCH_EXECUTION.md) | 有限计算与中断恢复 | 已完成工程验收 | 监督进程、有限资源、取消与恢复；采样软限制。 | E10,E11 |
| [E13](ALPHA_SCREENING.md) | 快速筛掉差候选 | 已完成工程验收 | 用预测/代理收益诊断筛选，尚不是资金账户回测。 | E07,E11 |
| [E14](ALPHA_CAPACITY_RESULTS_20261007.md) | 实测机器能跑多少 | 已完成工程验收 | 160个公式意图只有8类工作形状；不是160个有效策略。 | E12,E13 |
| [E15](ALPHA_MODEL_INTERFACES.md) | 六类表格模型 | 已完成工程验收 | Ridge/ElasticNet/Huber/RF/ExtraTrees/HGB共同接口。 | E07,R2 |
| [E16](ALPHA_CAUSAL_LEARNING.md) | 训练、目标与预测协议 | 已完成工程验收 | 时间划分/成熟标签/预测名单约束；保存预测与元数据。 | E15,E11 |
| [E17](ALPHA_SCORE_TO_ACCOUNT.md) | 分数变成持仓名单 | 已完成工程验收 | 按当时资格/分数/持仓数/权重生成计划，不自动挖特征。 | E16,E01 |
| [E18](ALPHA_ACCOUNT_AUDIT.md) | 独立检查名单与账本 | 已完成工程验收 | 重建选择并复算现金/持仓/权益/收益，不重新训练。 | E17,E02 |
| [E19](ML_E19_ACCOUNT_COMPARISON_20261009.md) | 首次新流水线真实闭环 | 12模型/32账户证据已重检报告；原父任务未正式结案 | 原12failed保留；后续独立任务完成账户重检。task-32a60b66115c仍active、无submission/review；证据状态与任务结案分开。 | E14,E18 |
| [E20](ALPHA_SEQUENCE_WINDOWS.md) | 序列与事件数据集 | 日历窗口已正式接入，事件序列/真实试点待做 | E20私有30检查，E20B正式32检查；明确每日特征、可知时间/缺失掩码和有限批。没有序列模型。 | E07,E16 |
| [E21](ML_SYSTEM_BLUEPRINT_20261005.md) | 神经网络模型 | 未开始 | MLP→GRU/TCN→Transformer，逐步比较并实测成本。 | E20,R2,E19 |
| [E22](ML_SYSTEM_BLUEPRINT_20261005.md) | 席位关系图 | 未开始 | 当时关系统计→图模型，隔离未来新增关系。 | E02,E20,E19 |
| [E23](ML_FORMULA_SHARED_STORAGE_20261009.md) | 遗传与符号搜索公式 | 有限公式库的八类训练筛选已完成；自适应搜索未开始 | 256有符号公式/128模板/8类完整训练筛选；原G14作业及原归档超时保留，独立恢复已接受。12表现+8探索只是模型输入候选，0新fit/account；GP/RL/LLM自适应搜索未实现。 | E09,E12,E19 |
| [E24](ML_SYSTEM_BLUEPRINT_20261005.md) | 强化学习生成公式 | 未开始 | 固定奖励和反馈范围，防止评价区间泄漏。 | E23 |
| [E25](ML_SYSTEM_BLUEPRINT_20261005.md) | LLM提出受限公式 | 未开始 | 提出假设与表达式，按调用/研究预算核查。 | E23 |
| [E26](ML_SYSTEM_BLUEPRINT_20261005.md) | 组合权重与风险研究 | 部分完成 | 旧系统已有组合/风险实验，新统一流水线研究待E19。 | E18,E19 |
| [E27](ML_SYSTEM_BLUEPRINT_20261005.md) | 工作台展示研究过程 | 部分完成 | 已有任务/账户/结果工作台；新候选前沿尚待接入。 | E11,E19 |
| [E28](ML_SYSTEM_BLUEPRINT_20261005.md) | 记录真正未来的预测 | 未开始 | 可载入实际拟合模型和训练trace的存档基础已由E21C接入；真正未来数据推断与前瞻预测封存仍未开始。旧E19没有保存拟合对象，不可补造；已看历史不算未见验证。 | E18,E19 |
| [E29](FOUNDATION_CONTINUATION_20261013.md) | 备份和灾后恢复 | 元数据库及选定学习证据恢复通过；全项目／异盘仍缺 | 新增真实HGB189文件约0.947GiB有效载荷备份19.41秒，隔离核对5.89秒；包含选定输入／源码／模型／预测，不加载模型、不激活恢复任务。其他账户／完整行情／异盘和自动时程未做。 | R3,E12 |
| [E12P](ALPHA_WINDOWS_CI_REPAIR.md) | 补充：Windows进程身份 | 已完成工程验收 | 进程身份与启动恢复修复，保留失败迭代。 | E12 |
| [E13B1](ALPHA_BINDING_INDEX.md) | 补充：复用特征绑定索引 | 已完成工程验收 | 复用绑定解析；非同口径实测不宣称精确加速倍数。 | E13 |
| [E13D](ALPHA_SCREEN_DURABLE.md) | 补充：持久初筛任务 | 已完成工程验收 | 正式3作业：2完成、1预期输出上限失败；0fit/账户。 | E12、E13 |
| [E16D](ALPHA_LEARNING_DURABLE.md) | 补充：持久训练任务 | 已完成工程验收 | 合成拟合3次：2完成、1预期不收敛；0真实拟合/账户。 | E12、E16 |
| [E19A](ML_E19_ACCOUNT_COMPARISON_20261009.md) | 补充：金额舍入与原账户重检 | 工程任务已完成，原档案保留 | 24模型场景修正核查+8原对照通过；16同名单比较发布；0新fit/模拟。 | E19 |
| [E19D](ML_E19_SCORE_DIAGNOSIS_20261009.md) | 保存分数的失效诊断 | 完成并已发布 | 保存分数全12诊断完成，9个头部标签超额为负；不改任何账户结果。 | E19,E19A |
| [E20B](ALPHA_SEQUENCE_WINDOWS.md) | 正式序列接口与日线接入 | 已发布；两平台CI通过 | sequence正式32检查；69e20f3 Windows1272/Ubuntu1269+3skip，真实试点和神经模型未做。 | E20,E19A |
| [E21T](ML_TARGETS_AND_SAVED_MODELS_20261009.md) | 补充：成熟目标与私有模型存档 | 已完成私有接口验收 | 实际任务标签E21。四种目标、六合成fit；原35+补3覆盖38条件；原12市场模型权重没有补回。 | E19D |
| [E21A](ML_TARGETS_AND_SAVED_MODELS_20261009.md) | 补充：三个测试构造修复 | 已完成 | 三节点实际补验通过，0fit；启动前XML读取失败保留。 | E21T |
| [E21B](ML_TARGETS_AND_SAVED_MODELS_20261009.md) | 补充：正式目标与模型存档模块 | 已完成并公开 | 正式独立API迁入，当时本地31通过0fit；精确12文件发布，Windows1310/Ubuntu1307+3skip CI成功。实际模型/训练trace存档由E21C接入；目标变换仍是独立接口，正式worker集成待做。 | E21T,E21A |
| [E23S](ML_FORMULA_SHARED_STORAGE_20261009.md) | 补充：共享名单存储与独立核查 | 已完成并公开 | 144生产文件；原29加补13验收42条件，v1/v2各2合成公式共4，0真实公式/fit/account。首次失败保留。 公开提交a21ecc7；Windows1331/Ubuntu1328+3skip CI成功。 | E23,E12 |
| [E23P](ML_FORMULA_SHARED_STORAGE_20261009.md) | 补充：真实32公式容量试跑 | 已完成并公开 | 1正式批次32全完成；7实际依赖/11PV输入列/32原料库。213MB，旧13完整语义一致；473s/1.43GB；0screen/fit/account。 | E23S |
| [E23F](ML_FORMULA_SCREEN_THROUGHPUT_20261009.md) | 补充：八家族训练筛选 | 部分阶段已归档：1/15例 | 复用P32完成32通道，native443s/outer1171.5s接近20min单例上限；全8家族尚未完成，余14案例未启动；先改造重复排序。 归档789.625s、两次原冻结核查；task-d878a2473869完成的是吞吐里程碑，整批仍待接续。 | E23P,E12 |
| [E23C](ML_EXACT_RANK_REUSE_20261009.md) | 补充：精确共同名单排序复用 | 已完成并公开 | 75软件通过；1次真实379842×32/496对全部与原算法/原表精确相同，rank992→32、pair124.25→7.891s；145源码，0fit/account。两名结果接受并归档，未宣称whole-screen提速。 公开提交457a90f；Windows1353/Ubuntu1350+3skip CI成功。 | E23F,E23S |
| [E23G](ML_EIGHT_FAMILY_TRAINING_SCREEN_20261009.md) | 补充：接续七类224公式与筛选 | 已完成本地归档；公开发布待核查 | 224新公式/224新筛选＋首32复用＝256有符号公式/128模板/8类；20候选并非成功策略。双RESULT接受，原7200秒归档超时保留；独立恢复实际完成14次核查，未重置原阶段；0新模型/账户。 | E23C,E23F |
| [E21C](ML_ACTUAL_TRAINING_ARTIFACTS_20261009.md) | 补充：真实训练作业存档与零重训重现 | 实际工程归档、公开发布与新双平台CI完成 | 39项合成条件实际通过，原失败5fit保留；独立短目录恢复5库级+3隔离=8新fit，本机共13合成fit；其中存档容量失败仍计数。实际模型/X/Y/权重/train分数保存与0refit重载接入；145→146源。0新真实模型/账户。 精确15文件发布，新提交Windows1392/Ubuntu1389+3skip通过，打包与模板通过；这些CI中的合成拟合另记，未计入本机13。 | E21B,E23G |
| [E23GR](ML_EIGHT_FAMILY_TRAINING_SCREEN_20261009.md) | 补充：最终裁决超时后的独立恢复 | 已完成；原超时保留 | 原7200秒案例失败，145源/30旧数据库行及完整目录保全。独立7800阶段/5400案例实际14核查，0新公式/fit/account；通过review.next_task_id准确衔接E21C。 | E23G |
| [E23H0](ML_FORECAST_FEATURE_PROTOCOL_20261009.md) | 补充：原生后续期公式执行协议 | 已归档、已公开；实际双平台CI成功 | 51条件实际通过，4隔离批8合成公式。后续X、预热与固定训练名单原生协议接入，v1/v2兼容，146→147源；0真实公式/模型/账户。 实际CI：Windows1443，Ubuntu1440＋3跳过，wheel/template均成功。 | E21C |
| [E23H](ML_FORMULA_CAPACITY_BLOCK_20261009.html) | 补充：固定20候选的真实后续特征 | 原部分阶段已关闭；20项目标由接续补齐 | 旧4项复用、原失败与7声明永久保留；接续E23HR新增16项，共20完整输出。旧7200时钟未重置。 | E23G,E23H0 |
| [E19E](ML_COMBINED_FEATURE_ACCOUNT_RESULTS_20261010.md) | 补充：组合特征的真实模型与账户对照 | 7模型/14真实账户已双复核正式结案 | 16/28/36项与同日置乱标签对照完整完成；累计8拟合意图/16原生登记，含1未知拟合调用失败和1配对账户失败，均保留。正常提升树改善失败基准但仍落后小市值。 | E21C,E23HR |
| [E23I](ML_SYSTEM_BLUEPRINT_20261005.md) | 补充：公式直接选股与条件/平分对照 | 独立研究路线，未开始 | 不伪装成已训练模型；明确零分、平分、资格和现金规则，另登记真实账户。 | E23H,E18 |
| [E23HR](ML_FEATURE_COMPLETION_20261010.html) | 补充：资源恢复后补齐固定特征 | 20特征齐备并正式归档 | 6新任务补齐16项、复用4项；累计9登记/8成功/1旧失败/23声明。两名独立RESULT接受，0新模型/账户。关闭耗时下界1211秒超过600预算，原时钟保留。 | E23H,E23H0 |
| [R23D](ML_68_FEATURE_RESOURCE_STOP_20261010.md) | 补充：直接信息与组合共同输入的真实账户检验 | 68项组装完成；首项训练因可承诺空间不足停止并正式结案；发布与新CI完成 | 68项=原16价量+20组合+32已有板型/龙虎榜/财务/市场信息，933284共同样本；组装峰2.81GiB，首项训练准备峰7.10GiB、执行峰6.74GiB。1失败意图，0成功拟合/账户，实际fit调用未知；非策略有效性结果，未自动重试，未改OS或终止其他应用。 | E19E,E21C,E23HR |
| [R23C](ML_FOUR_MODEL_ACCOUNT_RESULTS_20261010.md) | 四种模型在同36项信息上的资金结果比较 | 真实RESULT已双复核结案、发布与新提交CI均完成 | 极端随机树、随机森林、ElasticNet、Huber各拟合一次；同379842成熟训练/550711预测、相同36输入/周Top100/账户，两费用各真实执行；全部失败历史与零费用对照保留。 | R23D,E15,E16 |
| [R24T](ML_TARGET_PARTIAL_ACCOUNT_RESULTS_20261010_02.md) | 成熟训练目标与组合映射的工程和账户检验 | 真实部分结案；仅1/2模型及2/4路径 | 父task completed；centered配置费CAGR1.08%/DD26.39%，被半仓小市值支配。旧Errno22及全部失败保留；rank未跑。 | R23C,E15,E16 |
| [R24M](ML_SINGLE_PREPARATION_ENGINEERING_20261010.md) | 单次准备数据与旧模型作业兼容 | 实测、双复核、6文件升级、公开与新CI完成 | 单次prepare与旧v1/v2冻结兼容已通过60测试：22微型真实fit+1假失败，0市场fit/0账户；68首次准备峰值仍未证明解决。 | R23C,R24T |
| [R24N](ML_NATIVE_TRAINING_TARGET_ENGINEERING_20261010.md) | 明确原始收益与训练目标的原生接口 | 实际微型验证、双复核、本机升级、公开与新提交双系统CI完成 | 46项测试、10合成真实fit/0市场/0账户；原始/变换/实际Y及单位、存档/名单审计、旧冻结和旧库核查完成。 | R24M,R24T |
| [R24P](ML_DATA_PREPARATION_EQUIVALENCE_20261011.md) | 首次prepare容量与批量吞吐工程 | 26项小准备等价通过、部分结案；全68尚未验证 | 1实际pytest/0fit账户；候选不改变X/Y/名单/权重，生产学习源码未升级。全量另R24PF task-1d0120711bce。 | R24N,R24T |
| [OSS01](ML_OPEN_SOURCE_REUSE_REVIEW_20261010.md) | 现成量化框架比较与复用 | 来源/选定源码对照已完成；外部集成未开始 | 十项候选、23条主来源；通用模块需比较后采用。0新增训练/账户，未测兼容或迁移性能。 | 已有E01/E07/E16/E18；后续独立有限兼容试点 |
| [R25R](ML_RANK_PRETRAIN_RESOURCE_FAILURE_20261011.md) | 只接续未执行的排名目标 | 训练前失败已双RESULT归档 | 379842/550711输入准备，native训练前断言失败；exp-b89345f8998d取消且1意图保留；实际0fit/predict/账户。 | R24T部分结案 |
| [OSS02](ML_OPEN_SOURCE_COMPATIBILITY_20261010_02.md) | 固定外部量化发行与本机兼容性 | 已完成元数据分析及真实双RESULT | 固定4发行/依赖/平台/许可已检查；未做完整安装和后端兼容。canonical OSS03子任务已真实创建。 | OSS01 |
| [OSS03](ML_OPEN_SOURCE_KERNEL_FIRST_TRIAL_20261010.md) | Qlib八基础内核小尺寸兼容性 | 已按失败范围真实结案 | 首次扩展导入成功，但代码后缀取数0行，0比较；双RESULT失败分析通过，不是兼容成功。 | OSS02 |
| [OSS04](ML_OPEN_SOURCE_KERNEL_RESULTS_20261010.md) | 修正证券代码契约后的内核验证 | 小数据120组通过及双RESULT结案 | 1440行/8kernel×15cases；A独立核120/B20，0fit/账户/生产install，完整后端未测。 | OSS03 |
| [OSS05](ML_OPEN_SOURCE_EXPRESSION_FIRST_TRIAL_20261010_02.md) | 隔离Qlib真实表达式与本地数据提供器 | 失败诊断双RESULT及正式结案 | 下载前唯一准备失败；独立诊断仅项目说明文件的发现路径差异。报告02纠正68准入，原失败/意见保留。 | OSS04 |
| [OSS06](ML_OPEN_SOURCE_WHEEL_LAYOUT_FAILURE_20261010.md) | 修正版本发现口径后的实际Qlib接口 | 安装前失败，双RESULT结案 | 143基础版本通过，下载12wheel共3769172字节；误拒dill合法.data/scripts。0环境安装/API/fit账户；不是Qlib不兼容。 | OSS05 |
| [OSS07](ML_OPEN_SOURCE_EXPRESSION_INSTALL_AND_CHECK_FAILURE_20261011.md) | 复用缓存与标准pip脚本映射的接口验证 | 安装通过、检查失败；双RESULT及正式结案 | 13缓存wheel标准pip安装及666成员核验；C.registered在重设配置后报错，0数组/比较/fit账户。后续不用重新安装。 | OSS06 |
| [OSS08](ML_OPEN_SOURCE_EXPRESSION_API_RESULTS_20261011.md) | 复用已装环境的真实公式验收 | 72组实际通过、双RESULT及正式结案 | A独立72原始组；B独立72原始及72未来组；前60逐位同。0fit账户/新安装下载，完整后端未验收。 | OSS07 |
| [OSS09](ML_MATURE_FEATURE_ADAPTER_RESULTS_20261011.md) | 成熟特征薄适配现有协议 | 真实51项集成双RESULT结案 | task completed；25stub+24回归+2实际Qlib六公式，0fit/predict/账户。Win嵌套证据复制失败保留，同字节平铺包完成归档；OSS10另排。 | OSS08 |
| [PUB10](ML_REVIEW_RECORDS_AND_RESUME_20261010.md) | 经核查报告及公开导航交付 | 实际公开及新提交双平台CI、双RESULT完成 | 32公开清单/30文件改变，3114548；Windows1520、Linux1517+3跳过，wheel/template通过。 | OSS08 |
| [PUB11](ML_RESEARCH_RESUME_STATUS_20261011.md) | 下轮实质结果的集中公开 | 公开、双平台CI、正式验收完成 | 0703d507；18清单/16变动；Windows1545、Linux1542+3跳过，0市场fit/account；PUB12已排队。 | PUB10 |
| [R24PF](ML_FULL68_PREPARATION_CAPACITY_20261011.md) | 完整68项候选准备容量 | 完整准备及独立结果正式验收完成 | task-1d0120711bce；一次933284记录/68项准备，233秒/6.04GiB；0市场训练或回测。完整内容保持一致，后续生产升级独立。 | R24P |
| [OSS10](ML_MATURE_FEATURE_ADAPTER_RESULTS_20261011.md) | 成熟公式的信息重复与账户增量设计 | 真实排队，未执行 | task-57448daddabc；不重跑144小公式接口、不默认全Alpha158/360或完整后端。优先rank及68容量。 | OSS09 |
| [R25RR](ML_RANK_ACCOUNT_RESULTS_20261011.md) | 排名实验资源记录与安全接续 | 1模型2账户正式完成，收益未晋级 | task-7204d3bd57ca；含费−1.30%/每日回撤−25.14%，零费11.95%/−22.14%；真实RESULT/66.2秒关闭完成。 | R25R |
| [R26](ML_FULL68_ACCOUNT_RESULTS_20261011.html) | 完整信息的同模型账户消融 | 已完成真实账户验收 | 真实2模型/4费用路径已正式双RESULT03接受并133.110秒完整关闭。中心化68净年化1.4645%、DD−27.1413%，未达策略晋级；旧回测不重跑。 | R25RR/R24PF/R24PR |
| [PUB12](ML_FULL68_PREPARATION_CAPACITY_20261011.md) | 真实账户和容量报告公开交付 | 已完成 | 745ddd9精确报告交付，实际新head双平台CI成功及双RESULT02正式关闭；原提交前EOF失败、收尾代码审查REVISE保持。 | PUB11 |
| [R24PU](ML_FULL68_PREPARATION_CAPACITY_20261011.md) | 完整68项数据准备代码升级 | 原失败正式归档，后续R24PR补齐 | task-aa9e5d395877正式结案为87通过4目录失败，不重标原通过；生产精确候选继续保留，补4项归独立R24PR。 | R24PF |
| [R24PR](ML_FULL68_PRODUCTION_UPGRADE_20261011_02.md) | 短目录补齐4项未完成回归 | 正式完成，91项合计通过 | task-dd5d010a8df1；仅补4项全通过，复用原87项。实际双复核及正式结案完成，生产代码已验收；0新市场模型/账户。 | R24PU |
| [PUB13](ML_RESEARCH_PROGRESS_20261011.html) | 公开已经验收的数据准备源码 | 已完成实际代码交付 | 精确7文件/5变动497833a，accepted f607准备代码公开；Windows1545/Linux1542+3及wheel/template成功，真实双RESULT正式关闭，后续PUB14。 | PUB12,R24PR |
| [R27M](ML_MLP_ENGINEERING_20261011_02.html) | 非树模型：隔离神经网络适配与小测试 | 已完成隔离工程验收 | 一次14passed、5合成fit/4predict；真实双RESULT02和11.828秒完整关闭已完成。0行情模型/账户，本隔离阶段未晋级；后续R27P已经完成接入。 | R24PR |
| [PUB14](PUBLIC_DELIVERY.md) | 公开下一批已经验收的结果 | 推送完成；真实CI失败已保留 | 18候选实际16文件发布；Ubuntu与Windows均因旧genericMLP测试树参数失败，未接受该版本CI，后续PUB14F只修复一文件。 | PUB13,后续实际验收 |
| [R27P](ML_MLP_NATIVE_INTEGRATION_20261011.html) | 神经网络接入登记、核查与模型存档 | 已完成原生生产接入验收 | 3处必要源码精确接入，13新检查一次通过17.81秒/完整27.203秒；1微型39成熟行拟合、40预测，真实双RESULT和正式关闭。0行情模型/账户，旧14项不重复。 | R27M实际验收,R26正式关闭 |
| [R28](ML_MLP_ACCOUNT_RESULTS_20261011_02.html) | 首次完整信息神经网络账户检验 | 已完成真实账户核查 | 1个68项MLP、2条真实费用账户；含费年化−22.22%、DD−61.67%；两独立RESULT与正式归档完成，不晋级。 | R26实际完成,R27P真实接入验收 |
| [PUB14F](BUILD_PROGRESS_20261006.md) | 修复旧测试对MLP误传参数 | 已完成修复及真实CI验收 | 原失败保留；单测试修复2c14bb5、Windows1559/Linux1556+3跳过、打包通过，最终两复核及正式归档完成。 | PUB14真实失败/R27M已知信号参数 |
| [R29](ML_SCORE_REUSE_20261011_04.html) | 旧模型分数复用，减少重复计算 | 原失败已保留，后续R29R恢复验收 | 原38项34reportedpass/4fail永久保留。R29R只修23受影响检查，15合法未变检查复用；不擦掉原失败。 | R28、PUB14F正式结束 |
| [PUB15](ML_MLP_ACCOUNT_RESULTS_20261011_02.html) | 公开MLP真实结果和最新项目总览 | 已发布，真实双平台CI已通过 | 26ae92f：精确10份文档、9项实际变动；Windows1559通过、Ubuntu1556通过+3skip，打包与模板均成功。没有未验收SRC或原数据。 | R28、PUB14F正式结束 |
| [R29R](ML_SCORE_REUSE_20261011_04.html) | 修正复用接口的测试样本 | 已完成工程验收；没有新账户收益 | 23actualPASS＋15valid复用；550711保存评分/18900选择。0fit/predict/load/account。首次归档180秒失败保留，后续单次有限元数据恢复成功；不代表v2原生账户或持仓保留已验证。 | R29真实失败正式结案 |
| [R30](ML_SCORE_REUSE_20261011_04.html) | 旧评分进入完整账户存档的反例检查 | 原失败正式结案，原存档保留 | 临时current路径消失导致8dots无最终收据；归档第一次超过180秒也保留。双复核后的进程监督、明确依赖范围恢复实际完成，0新数值。 | R29R实际恢复closure02/end02 |
| [PUB16](ML_V2_ACCOUNT_ARCHIVE_20261011.md) | 公开经过验收的评分复用代码和说明 | 已完成公开验收 | 精确20文件，公开提交aaa3014；实际CI38140350972，Windows1613/Ubuntu1610+3跳过；0新市场研究。 | PUB15、R29R实际恢复结束 |
| [R30R](ML_V2_ACCOUNT_ARCHIVE_20261011.html) | 稳定测试目录并复用原合成存档 | 已完成实际16项工程验收及正式归档 | task-7854c9cdab68；16唯一PASS、48进度；0fit/predict/load/nativeRun/simulate、一原存档读取。RESULT01未执行关闭的两个缺口修正后，RESULT02双接受，closure01/end01/Ownedwrapperend01真实完成。 | R30失败关闭 |
| [R31](ML_ACTUAL_HOLDING_ACCOUNT_RESULTS_20261011_02.md) | 根据实际持股保留股票 | 工程与正式账户已验收 | 44测试，1登记/2真实费用账户；净年化5.48%、DD27.41%，同评分原规则1.46%/27.14%；仍逊于适当对照，不晋级。 | R29R,R30R,R26 |
| [PUB17](ML_ACTUAL_HOLDING_ACCOUNT_RESULTS_20261011_02.md) | 公开实际持仓与真实回测结果 | 已发布；旧CI失败已如实结案 | fcef6dc已公开R31持仓规则和真实回测；两平台同一项测试的预期拒绝理由过时。原失败保留，后续PUB17F只修这一项断言。 | R31,PUB16 |
| [R32](BUILD_PROGRESS_20261006.md) | 完整信息适配与64个跨信息公式 | 150小样本检查及模块接入完成；真实时间类型兼容待修 | 真实原数据揭示ns/us存储单位误判，工程测试此前未覆盖。旧完整150检查事实保留，不等于真实容量通过；原数据和旧账户结果没有被修改。 | R31 |
| [R33](BUILD_PROGRESS_20261006.md) | 同模型比较组合信息的价值 | 全部64列预测与132列模型对照未完成 | 应优先原68 vs68+ALL64；单列Top10不是必经条件。旧预测草稿只接少数列且部分未完成；132列需调整旧128列预算，123193488单元超过现1亿容量上限，须分块或明确升级容量协议。 | 预测期固定全64接线、容量工程与有限模型预算（尚未完成） |
| [PUB17F](BUILD_PROGRESS_20261006.md) | 修复单项测试并核对真实CI | 实际发布与双平台CI已验收 | f8b62d5只改一个测试断言；Windows1652通过，Ubuntu1649通过+3跳过，安装包/模板均通过；不改变生产守卫或回测结果。 | PUB17真实失败结案 |
| [R32N](BUILD_PROGRESS_20261006.md) | 在真实数据上计算并筛选64个组合特征 | 真实失败已如实结案；原尝试不重试 | 运行34.859秒，在完整933284原样本时钟校验失败；0注册/批次/筛选/模型/账户。最高RSS约2.37GiB、最低可承诺约6.75GiB，不是内存不足。所有失败永久保留。 | R32工程、PUB17F实际通过及独立源码复核 |
| [R32T](BUILD_PROGRESS_20261006.md) | 修复相同时刻的纳秒/微秒误判 | 96项实际工程检查和双复核通过；已接入 | 27精度+3原生输出+21准备+13组装+32时间/目标契约检查，全部通过；一次合成worker、0行情模型/账户。同刻跨存储单位通过、1ns实差拒绝，包括ISO字符串。原数据不改。结果整理目录检查失败保留，复用真实测试后完成独立元数据归档。 | R32N真实失败诊断 |
| [R32NR](BUILD_PROGRESS_20261006.md) | 修复后独立真实64组合与训练筛选 | 失败已如实结案，旧尝试不重跑 | 63.297秒为外层监督运行总耗时；12.734秒独立只读诊断确认原15:30被准备脚本误写15:00。失败保留；双复核负面结案，0登记/模型/账户。 | R32T实际验收与独立SOURCE |
| [R32NC](BUILD_PROGRESS_20261006.md) | 纠正原决策时间后执行64项组合 | 真实64项部分结果已结案；筛选未运行 | 两名实际复核接受后，40.297秒保存部分关闭，完整进程exit0。唯一64项批次保留；原2100.094秒到期未撤销。后继R32S已排队，不重算64。 | R32NR negative RESULT/close |
| [PUB18](BUILD_PROGRESS_20261006.md) | 公开组合信息与时间精度工程 | 提交前失败已如实结案；独立恢复任务排队 | v5实际两审接受并关闭，完整53.562秒、95次只读Git调用单列；15项暂存保持，0新提交推送/CI。后继PUB18R将分别验证候选原字节与Git规范化字节；旧失败保留。 | R32与R32T已验收工程 |
| [R32S](BUILD_PROGRESS_20261006.md) | 复用已完成64项，避免重复完整核查 | 真实工程接入完成 | 新元数据阶段下，302原件双复核111.582/142.670秒准时接受；单次收尾61.297/180成功，精确1源码1测试晋级，真实子任务已创建。原3600超时保留；20测试和64特征未重跑。 | R32NC部分结果正式结案 |
| [PUB18R](BUILD_PROGRESS_20261006.md) | 恢复15项已暂存文件的公开发布 | 真实发布、双平台CI与独立RESULT已验收并关闭 | 精确15项，未扩大公开数据。HEAD96919d9…、CI38175271350；Windows1717/Ubuntu1714通过3跳过。关闭Owned19.953秒、Job0；旧失败与自动审批拒绝及用户明确授权均保留。下一发布任务PUB19仅规划排队。 | PUB18真实负面结案 |
| [R32SC](ML_SCREEN_MATURITY_DISCOVERY_20261011.md) | 筛选已有64个组合特征 | 实际完成并归档；确认比较区间缺陷 | 154周包含1个未成熟周；56项选中未知Y被淘汰，8项缺输入持现金而通过。347原件与7大输入、双独立RESULT04接受；原名单保存但不支持模型准入。0新模型/账户。 | R32S真实完成与SOURCE双独立复核 |
| [R32SM](CONSTRUCTION_20261012.md) | 让所有候选用同一组成熟周比较 | 工程已完成 | 共同日期仅由固定时钟决定，保留原输入/选股；87项相关测试与两名独立复核通过。真实辅助诊断单次输出预算失败，不自动重跑。 | R32SC真实缺陷诊断与完成归档 |
| [E33X](EXECUTION_REFACTOR_RESULTS_20261012.md) | 统一执行入口、恢复与精简留存 | 已完成本轮；GitHub双平台CI通过 | 统一有限计划、原实验恢复、审计复用；首轮165目录及收尾缓存合计释放约5.48GB，1307助手可恢复归档。实际调用/反例通过，源代码e23f5d6的Windows/Ubuntu完整测试和打包通过。132列科学任务未执行。 | 现有Research、SQLite、worker和资源池 |
| [R32FALL](CONSTRUCTION_20261012.md) | 全部组合特征进入预测期 | 工程已完成，小样本贯通 | 保留原全部64候选，不依赖旧≤8名单；复用原worker及完整键域合并，189项回归及最终51项检查通过。 | R32SM、原NC64 |
| [R32FPALL](CONSTRUCTION_20261012.md) | 真实64列预测与132列输入 | 已完成 | exp-494b295f54cc首次核查通过：64/0，每项550711预测行，训练重算0。完整132列933284样本已保存；原68逐值/缺失/时钟不变；公式计算保存135秒，执行核查阶段1400秒。 | R32FALL |
| [R32CAP](CONSTRUCTION_20261012.md) | 完整模型容量与Windows启动检查 | 工程已完成 | task-16dfd7427637；原始矩阵显式有限上限两亿格，默认3200万不变；commit不足启动前拒绝。108模型加14资源测试、一次独立工程复核。 | R32FALL |
| [R32PAIR](CONSTRUCTION_20261012.md) | 132列同条件真实模型与账户 | 已完成，负结果保留 | 复用旧68；新增一次centered RF和两条真实费用情景，含费−3.10%/32.46%，未改善；不按单列盈利门槛删特征。 | R32FPALL、R32CAP、原68中心化基线 |
| [R32DIVERSITY](CONSTRUCTION_20261012.md) | 同132输入的两种互补模型 | 已完成，未达重要收益改善 | task-f80028761bc9；实际2fit、2账户/4情景：ridge−12.13%/DD46.02%；HGB40+3.09%/29.61%，同897日且另算指标。 | R32PAIR、已保存132矩阵 |
| [R33TIME](CONSTRUCTION_20261012.md) | 2026近期训练与旧训练匹配账户 | 已完成，改善但仍亏损 | task-3a12d6d9106e；实际1fit、2账户四情景；2026同169日含费旧训练−13.20%/DD27.24%，近期−4.69%/24.96%；不是年化，期末7695元估值未解决，已见样本。 | R32DIVERSITY、已保存132矩阵与完整旧HGB预测 |
| [R33ATOM](CONSTRUCTION_20261012.md) | 修复真实JSON临时文件与输出监控不一致 | 已发布，双平台完整CI通过 | d21cfb8；CI38323501697 Windows1811/Ubuntu1808通过+3跳过。原相关147通过+2权限跳过；旧完成账户不重算。 | 已完成数值阶段；0c95e51 CI |
| [R33REP](CONSTRUCTION_20261012.md) | 同近期模型的68列消融与两种目标表示 | 已完成，目标变换未改善账户 | 3fit/3账户6情景，439424成熟/105251预测/169日严格一致；68−8.63%/26.93%，132zscore−7.29%/28.68%，132rank−7.88%/28.09%，原132centered−4.69%/24.96%。 | R33TIME、保存132矩阵、R33ATOM本机147回归 |
| [R34REUSE](CONSTRUCTION_20261012.md) | 只改临时名字时复用原模型预测 | 已完成并发布，双平台完整CI通过 | 3164cfc；19反例/五模块96通过，CI Windows1830/Ubuntu1827+3跳过。精确兼容旧预测，2026buffer−1.65%/DD22.86%，0fit。 | R33REP,R33ATOM,既有holding_policy |
| [R34FULL](CONSTRUCTION_20261012.md) | 固定缓冲在较长已见时期的验证 | 已完成，收益改善而回撤略升 | 0fit/predict；同897日固定150缓冲含费年化3.09%→5.22%，DD29.61%→29.92%；不是跨时期统一降风险或小市值胜出。 | R34REUSE,已完成原HGB账户 |
| [R35MLP](CONSTRUCTION_20261012.md) | 与树互补的固定小型神经网络 | 原实验planned，尚未训练；本轮优先执行效率改造 | exp-e880b2716ef8原输入／预算保留，0fit/0predict。新单worker策略不再以12GiB全额要求阻断；内存下降/OOM仍保留失败。 | R34REUSE,R34FULL,已保存132矩阵 |
| [R36MEM](EXECUTION_EFFICIENCY_20261012.md) | 内存上限与启动条件分开 | 已交付；213项相关回归和双平台完整CI通过 | 单任务无其他预约时可低于RSS上限试运行；启动最低余量+最多256MiB，运行检查RSS/物理/commit，失败不重试。 | 既有资源池、R32CAP与用户新授权 |
| [R36PIPE](EXECUTION_EFFICIENCY_20261012.md) | 已完成学习自动生成账户提案 | 已交付；213项相关回归和双平台完整CI通过 | v2/v3固定模板绑定真实预测／收据；plan-v2派生账户，登记中断不重建。未新算历史账户。 | 原身份复用、统一执行入口 |
| [R37READ](FOUNDATION_CONTINUATION_20261013.md) | 训练只读需要的列和日期 | 真实132列完整准备通过；其他入口分块仍缺 | 原9项小样本等价／反例及双平台CI保留；本次439424／105251行准备115.52秒、RSS峰值4.73GiB，0fit。旧2次资源停止保留，不报告未测新旧全量节省比例。 | R36MEM,R36PIPE |
| [R37CACHE](FOUNDATION_CONSTRUCTION_20261013.md) | 无关修改不重做数值审计 | 限定工程复核和35相关测试通过 | 相关import闭包及载入函数／属性／常量绑定；源包快照只检查相关文件，未知状态不复用。 本批完整Windows1893／Ubuntu1890＋3跳过CI通过。 | R36MEM,R36PIPE |
| [R37MAT](FOUNDATION_CONTINUATION_20261013.md) | 保存矩阵自动绑定学习模板 | 矩阵模板及v3自动学习步骤已接入 | 五文件替换保留研究选择；真实合成小fit→账户两费用情景、中断登记恢复通过。尚缺动态公式→矩阵写出。 | R36MEM,R36PIPE |

## 报告接续规则

以后每份阶段结果和最终回复都附本总览、完整蓝图、完整工作包计划、详细建设记录；重要策略成果仍以经核查的收益/回撤改善为准，不能仅以RankIC或工程通过作为停止理由。

进度依据：本机任务库、CHECKPOINT及阶段收据。本页是当前摘要，不覆盖历史冻结结果。旧报告的‘下一步’指写作当时，当前下一步以本页和任务库为准。更新方式：修改PROJECT_MAP.json后运行python tools/render_project_map.py；阶段报告仍须关联原始证据。
