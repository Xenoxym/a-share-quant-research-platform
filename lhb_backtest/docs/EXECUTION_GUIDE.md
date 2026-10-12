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
计划首版是预先写好的提案和实验引用；没有结果占位符、自动生成下一份提案、自动搜索、任意Shell和新的并行调度。数值步骤串行执行，已有两槽资源池继续生效。需要动态训练→预测→账户提案时，通过专用适配器构建下一份计划，不能声称通用DAG已实现。

相关：[施工方案和外部来源](EXECUTION_REFACTOR_PLAN_20261012.md) · [底层API](WORKER_GUIDE.md) · [旧总控](CONTROLLER_GUIDE.md)
