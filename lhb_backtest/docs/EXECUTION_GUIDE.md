# 实验怎样执行、怎样接着跑

2026-10-12开始，日常实验默认使用 `tools/execution.py`。旧总控仍兼容，只有需要它的多Agent研究流程时才调用。本页描述实际实现，不代表自动生成任意研究任务、永远运行或自动等待额度已完成。

## 一句话理解
Agent先写“做哪些实验”的计划；Python负责按顺序登记和运行，保存进度，下次接着同一个实验。训练、公式计算和账户回测仍由原来的worker完成，不另造一套引擎。

## 最小例子
在已有任务中复用一个完成的实验，计划如下（编号必须换成真实编号）：

```json
{
  "schema": "execution-plan-v1",
  "id": "my-first-plan",
  "task_id": "task-真实任务",
  "max_starts": 0,
  "timeout_seconds": 3600,
  "steps": [
    {"id": "existing-account", "needs": [], "experiment_id": "exp-真实实验"}
  ]
}
```

需要新实验时，把该步骤的experiment_id换成proposal对象，即原先交给research.py register的完整提案。步骤可写needs引用前面步骤。每份计划最多40步；同任务底层实验预算仍独立生效。

从lhb_backtest目录运行：

```powershell
..\.venv\Scripts\python.exe -B tools/research.py claim task-真实任务 --worker researcher --session data/technical/sessions/one.json
..\.venv\Scripts\python.exe -B tools/execution.py run --file plan.json --session data/technical/sessions/one.json
..\.venv\Scripts\python.exe -B tools/execution.py status my-first-plan
```

重复run就是恢复，不是重新做整份实验。用--steps 1限制一次访问一个步骤。任务已交接/凭据过期先重新领取，再用原计划。已完成的其他任务实验可作为引用；其未完成实验不能越权运行。新worktree显式传--root指向共享研究库。

## 中断和失败具体怎样处理
- 登记完成但还没写下步骤进度：数据库里的登记请求会指回原实验。即使工作区源码变了，也使用已冻结版本。
- 原worker仍运行：尝试收取原结果；无法确认完成就返回blocked和日志目录，绝不另起一份。
- 失败或取消：保留记录，后续依赖不运行。没有隐藏的自动重试。
- 调整同一计划的预算或内容：拒绝。真实修正应写新假设和新计划，已有任务的总实验预算及失败记录仍保留。
- 计划截止时间从首次运行保存，续跑不重置；新worker的登记超时时间必须容纳于剩余窗口。现有worker的监督与收集仍受原协议约束，检查/收取可以超过启动窗口；这不是整个会话的硬实时截止器。
- 状态保存在同一research.sqlite3，不需要复制目录、新数据库或一个AI来批准状态转换。
- Session凭据在领取事务内写入并flush/fsync；写入失败数据库回滚，不留下不可用领取。机器突然断电的文件/数据库双介质提交并非分布式原子事务。

## 检查还保留什么
首次完成时做现有独立数值审计。审计缓存检查实验定义、冻结代码、输入、输出、当前Python审计依赖和包版本的内容绑定。改变任一部分就不能直接使用旧缓存。账户还绑定原生账户目录和冻结数据快照。
这省去重复完整审计与数据载入（不同协议检查范围不同，公式批次本来不重算表达式）；大文件仍要读字节校验，所以不是零耗时，也不表示数据供应商正确。
可用alpha_execution.verify_completed(..., reaudit=True)显式重做数值核查。历史alpha记录第一次进入新缓存需要核查一次，不修改旧收据。

## Agent什么时候介入
| 工作 | 默认 |
|---|---|
| 登记、运行、恢复、文件校验、格式、测试 | Python，无LLM调用 |
| 新问题、来源研究、特征/模型实现、异常解释 | 当前研究Agent |
| 关键执行/缓存/时间逻辑修改 | 针对改动的一次独立Agent工程复核 |
| 账户规则修改、重要收益发现 | 两名独立Agent针对性复核 |
| 文档、小路径修改、垃圾清理 | 自动检查；不另开双复核与裁决 |

桌面聊天Token没有读取接口；执行器的llm_calls=0只描述这个Python入口，不能据此说当前聊天没消耗Token。

## 清理与留存
使用tools/clean_execution_artifacts.ps1先preview再apply。候选仅为明确命名的测试临时目录、构建/工具缓存、历史HTML预览profile及wheel输出；任何链接目录都拒绝。有程序仍使用对应目录就跳过。清单保存在本机research目录。
原数据、快照、worker_jobs、runs、ml_runs、虚拟环境、发布Git、失败证据和旧阶段脚本不属于自动删除候选。旧脚本退出默认执行入口；其冻结引用未全部查清之前不作永久删除。
已通过的测试临时产物可删，正式运行证据不能当“缓存”清理。

## 当前能力的边界
v1计划支持预先写好的提案和实验引用；v2只新增一种已完成学习→固定账户模板的派生步骤（见下）。没有任意结果占位符、自动搜索、任意Shell和新的并行调度。数值步骤串行执行，已有两槽资源池继续生效。需要动态训练→预测→账户提案时，通过专用适配器构建下一份计划，不能声称通用DAG已实现。

相关：[施工方案和外部来源](EXECUTION_REFACTOR_PLAN_20261012.md) · [底层API](WORKER_GUIDE.md) · [旧总控](CONTROLLER_GUIDE.md)


## 先看一个任务的短摘要

从lhb_backtest运行：

```powershell
..\.venv\Scripts\python.exe tools/execution.py brief task-25edb495e891
```

这是只读入口：返回任务目标与标准的有限摘要、实验状态计数、最近3项实验／错误、证据数量和登记预算。truncated_fields列出省略字段；完整规格、结果和失败按需展开。失败也占已登记数量；unregistered_slots不是实际fit余量，也不是启动授权。命令不领取任务、不初始化不存在的库、不启动计算、不核查全量产物或调用模型。结果后怎样重新规划见[Codex项目施工方式](CODEX_PROJECT_WORKFLOW.html)。


## 大输入先登记，再单独启动

准备完整训练输入可能临时占用数GiB内存。用下列命令只登记实验，成功后程序立即退出；从返回的实验编号写执行计划，然后另起进程执行原来的run命令。这样准备数据的内存不会继续留在启动程序里。

```powershell
..\.venv\Scripts\python.exe -B tools/execution.py register --file proposal.json --session session.json
```

这个命令直接使用原Research.register校验、冻结和预算计数，不拟合、不预测、不启动worker。成功只输出编号、类型、状态和实际登记数量，不输出会话凭据或提案全文；相同合法登记请求沿用同一个实验。JSON须是对象，重复键及NaN等非有限值会被拒绝。登记数量不是实际拟合次数，两者应分别报告。

2026-10-12，相关四项小样本测试通过，一次限定独立静态工程复核未发现阻断；真实132列浅层提升树已通过这个入口登记和执行。当前仍由主Chat根据结果构建下一份提案，没有新增自动研究主管或审批链。


2026-10-12另修复JSON写入与输出监控的约定：短临时文件使用明确保留名字，监控兼容新旧原子写入；正式文件丢失、权限和链接错误仍拒绝。真实writer反例及执行计划/账户登记/批次执行相关回归147通过、2链接权限跳过；一次限定独立静态工程复核无阻断。只作用于新冻结代码，旧完成结果不重算。完整结果及新提交CI状态见[施工报告](CONSTRUCTION_20261012.html)。

2026-10-13，保存预测复用仅额外兼容已核实的JSON临时名迁移（空前后缀→`.j-`/`.tmp`）；原两侧源码身份与其余整模块语义保持检查。96项相关反例/小账户通过；不重训或改旧收据。研究结果见[施工报告](CONSTRUCTION_20261012.html)。


## 单任务内存尝试启动

RSS上限是运行停止线，不是预估占用。若没有其他数值预约，可在“原最低余量＋至多256MiB启动空间”满足时尝试，不再要求先空出整个RSS上限。并行仍要求完整保守预约。运行中约0.2秒检查RSS、物理与Windows可承诺余量，越线停止保留失败，不自动重跑；瞬时OOM仍可能发生。原冻结资源字段和预算不改。观测记录新增memory_admission与minimum_available_commit_bytes。Windows读数为进程可承诺余量，可能小于全系统余量。

## 学习结果自动接到账户模板

`account-proposal --learning <完成学习ID> --template <公共v2/v3账户提案JSON> --output <项目内文件>` 只验证已完成学习证据和输入、生成提案，不登记、拟合或预测。输出已存在且相同则复用，不同则拒绝覆盖。完整账户覆盖和快照仍在后续登记验证。

有限计划的`schema=execution-plan-v2`允许下面的步骤；learn须是前面的已完成学习步骤，且列入needs：

```python
import json
from pathlib import Path

template = json.loads(Path("research/model_capacity_20261012/buffer150-account-proposal.json").read_text(encoding="utf-8"))
account_step = {"id": "account", "needs": ["learn"],
                "account_from_learning": {"learning_step": "learn", "template": template}}
```

把该步骤放入v2有限计划，前面的learn步骤仍须声明真实提案或已登记实验、任务与原预算。程序自动绑定真实预测及学习收据，保留模板其他声明；中断恢复先查持久登记操作，不再次生成或登记。没有更改v1计划，也不能用此功能跳过有限预算或失败。

[本次改动及验证边界](EXECUTION_EFFICIENCY_20261012.html)


## 保存的矩阵自动绑定学习模板

```powershell
..\.venv\Scripts\python.exe tools/execution.py learning-proposal --matrix research/my-matrix --template research/my-learning-template.json --output research/my-learning.json
```

模板是公共alpha_learn提案。矩阵目录含assembly_values.parquet、assembly_missing.parquet、definition.json、registry.json、receipt.json。程序只替换这五项输入，保留标签／决策文件、特征选择、日期、模型、目标及全部预算。它核对内容／收据／元数据，不解码全部特征值、不登记、不训练。下一步仍用原register命令进行完整键、成熟标签和因果检查；这不是自动生成研究方案或完整特征DAG。输出相同可复用，不同拒绝覆盖。

内建数值审计现按实际本地import闭包绑定当前源实现，无关UI／计划源码变化不重做数值审计。源包启动时记录内容快照，相关源文件运行中改变需用新进程；属性、函数默认值及声明状态有确定性绑定，无法证明的状态不读写缓存。未知核查器／动态导入仍保守退回全src。原输入、冻结代码、产物与账户快照的核查保持。[本轮实际验证和缺口](FOUNDATION_CONSTRUCTION_20261013.html)。

## 已存矩阵直接进入有限计划（v3）

`execution-plan-v3`保留v1/v2全部已有步骤，新增固定适配步骤：

```json
{"id":"learn","needs":[],"learning_from_matrix":{"matrix_folder":"research/my-matrix","template":{}}}
```

这里的空template只是说明位置，实际必须填完整公共alpha_learn提案，不可直接运行这个片段。学习之后可接原account_from_learning步骤，并在needs中引用learn。矩阵目录使用原五文件规范，模板保留决策／标签、日期、特征、模型和预算。预算检查先于构建；中断恢复先查原登记，不重新构建已登记提案。v1/v2计划定义与预算保持原状；同一持久计划不可改模板或目录。

## 学习证据备份与离线核对

`backup-learning --learning <实验编号> --output research/backups/<目录> --max-bytes <整数>`复制一份已完成v2/v3学习的输入、冻结源码、模型／预测／日志和SQLite快照。现成SQLite在线备份生成独立文件，不修改原库。字节帽包含最终清单和完成标记；拒绝覆盖，失败副本保留且没有完成标记。

`verify-backup --folder <副本目录>`只验证副本，不初始化研究库、不读取原绝对路径、不反序列化模型、不启动worker。它输出有效载荷bytes和完整目录directory_bytes。只有选定学习实验的文件被复制；整个市场、其他实验和账户快照没有一起备份。元数据库仍有旧绝对路径及历史记录，不支持直接激活迁移后的库。默认同盘本机副本不能代替异盘灾备，资料不得公开上传。

本批实际结果见[接续施工报告](FOUNDATION_CONTINUATION_20261013.html)。


## 已完成的公式分区写成学习矩阵

```powershell
..\.venv\Scripts\python.exe tools/execution.py expression-matrix --development exp-开发批次编号 --forecast exp-预测批次编号 --output research/new-matrix --max-rows 1000000 --max-cells 150000000 --max-bytes 2147483648
```

编号须替换为真实已完成的v4开发批次和同一producer的v6预测批次。这只是参数形式示例，行数、单元格、字节上限由实际研究预算决定。入口读取原prepared base与全部预先声明的候选输出，保留完整股票日期键、时钟和缺失原因；不读取标签，不筛选单项收益，不重算公式、训练或登记。

输出是原有学习入口认识的五文件矩阵。随后把这个目录交给learning-proposal或v3计划的learning_from_matrix，再接account_from_learning。已存在的目标目录拒绝覆盖；成功后直接复用原矩阵继续计划。写入失败的部分目录保留，没有有效receipt，不自动重试。预算包括Windows换行后的实际receipt字节。

这补了“已完成公式→矩阵文件”的程序接口；公式生成、计算、矩阵保存还没有合并成一份持久计划自动派生。当前保存与合并仍在内存中，只有完整小样本通过，不能把这组默认上限当作大规模内存保证。
