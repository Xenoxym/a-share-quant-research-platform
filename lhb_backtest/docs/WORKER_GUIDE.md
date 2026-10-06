# Worker 入场与接力协议（V7）

自动多Agent调度见[总控指南](CONTROLLER_GUIDE.md)：现已支持有预算的研究、双复核、综合裁决及下一轮。本页保留底层任务协议。总控已经为研究者领取任务时，不要再claim；只读复核角色不领取或修改任务。

你可以是任何能够访问本地目录并调用命令的 Agent。平台不要求继承前一个 Chat 的对话。任务数据库与证据才是共享状态。

## 先知道项目处于哪里

- 主线：主板基础策略与状态配置；旧龙虎榜模板不属于当前默认研究。
- 最近结论：V5 研究报告（本机证据，未随公开源码提供）。原型仍保留；简单状态门槛敏感；高波动就防御的 HMM 映射会错过反弹。
- 数据边界：[数据约定](DATA_CONTRACT.md)、[基础策略说明](FOUNDATION_RESEARCH_V4.md)。季度估算股本、历史修订、退市覆盖与分红应收不复投问题仍在。独立账本核查不是供应商准确性证明。
- 当前主要实现：`src/technical` 回测；`src/researchops` 任务与冻结执行；`tools/research.py` 通用 JSON 接口。自由编写研究特征可以扩展引擎分支；用数据实证而非文字直觉作为晋级依据。

从 `lhb_backtest` 工作目录运行，以下所有命令的标准输出均为 `{ok, data}` 或 `{ok:false,error}`，错误退出码为1。日志写入每个执行目录。

```powershell
..\.venv\Scripts\python.exe tools/research.py catalog
..\.venv\Scripts\python.exe tools/research.py list
..\.venv\Scripts\python.exe tools/research.py context task-v5-next-1
```

`catalog` 给出冻结数据、已导入知识、已复核任务结论、工具能力与限制。成功与失败的结论都进入共同记忆，并携带复核决定与证据类型。`context` 给出问题、基线运行ID、范围、预算、所有实验、事件、交接、结论及后续任务。首轮后续任务通过 `init` 幂等导入；不要为重复导入改写已有任务。

`catalog/list/context/show/export`以只读连接访问既有研究库，不初始化目录或修改表结构。总控复核者以冻结的context.json为本轮依据；实时查询失败时应说明，不能据此捏造实时状态。

## 长期目标与主动探索

完成入场查询后，读取[项目长期纲领](PROJECT_CHARTER.md)、[研究雷达](RESEARCH_RADAR.md)及当前任务交接。纲领第一节连接已保存的系统蓝图和代码交付规划；它们分别承担长期方向、系统设计、阶段实施，不代表设计能力已经完成。具体任务范围和预算仍按当前context执行，用户后续指令优先。

初始16个特征、提升树、小市值对照和周调仓不是永久研究边界。选择新研究主题时，主动探索近期实验与用户点名清单之外的方向，留下来源、阅读深度、竞争解释、最小否证试验及取舍理由；普通包内实施或紧急修复可说明后延。三个独立研究阶段完成或整批研究结束时检查研究覆盖和投入。

阶段记录使用[记录模板](RESEARCH_STAGE_TEMPLATE.md)，更新雷达需关联真实任务和证据。向用户宣布重要策略成果应有核查后的账户收益/回撤改善，预测指标单独用于诊断；报告末尾简述自上次汇报以来的研究路径。上述是工作准则，自动搜索及覆盖调度不因此成为已实现功能。

## 完成一轮研究

1. 领取任务；使用本 Worker 独有的 session 文件。租约默认30分钟；分析超过此时间需要主动续约。运行实验时平台每15秒自动续约。

```powershell
..\.venv\Scripts\python.exe tools/research.py claim task-v5-next-1 --worker researcher-a --session data/technical/sessions/researcher-a-01.json
..\.venv\Scripts\python.exe tools/research.py heartbeat --session data/technical/sessions/researcher-a-01.json
```

2. 写 `proposal.json`，登记后返回实验ID。策略定义可从基线 `data/technical/runs/<run-id>/spec.json` 复制并明确修改。快照ID从 `catalog` 选择，日期不得超出覆盖。默认同时重跑零交易成本与配置成本。

```json
{
  "hypothesis": "可证伪的具体假设，注明论文依据、历史诊断或待检验直觉",
  "expected_observation": "相对哪个对照，哪些区间与指标应如何改变",
  "falsification": "出现什么结果就不再继续这条假设",
  "snapshot_id": "用实际24位快照ID替换",
  "spec": {"strategy": {}, "execution": {}, "experiment": {"start_date":"2022-01-01", "end_date":"2026-09-24"}},
  "timeout_seconds": 1200
}
```

上例是字段说明，不能将空策略当作完整研究定义；请复制实际基线。配置策略还需满足固定分支资格。统一费用和数据并不意味着不同股票池自动构成纯净消融。

```powershell
..\.venv\Scripts\python.exe tools/research.py register --session data/technical/sessions/researcher-a-01.json --file proposal.json
..\.venv\Scripts\python.exe tools/research.py run exp-实际编号 --session data/technical/sessions/researcher-a-01.json
```

代码、策略、核查器、快照清单在登记时冻结。更改当前源码不影响已登记实验。每个任务同一时间只执行一个实验，不同任务使用不同执行目录。不要自行绕开预算：失败与冻结失败也消耗名额。相同实质定义（忽略名称与描述）重复登记返回原实验，不重新启动已执行的实验。

3. 分析账户与典型案例。新结果自动进入现有“结果与诊断”及“研究任务”页面，使用原有订单、费用、区间与逐股归因。并列展示时确认时间、初始资金、数据与执行口径一致。零成本不是费用加回。

4. 留存进度或交接。主动交接立即作废旧租约；租约过期后另一个 Worker 可领取，旧 session 无法修改任务。

```powershell
..\.venv\Scripts\python.exe tools/research.py handoff --session data/technical/sessions/researcher-a-01.json --note "已完成哪些实验；证据在哪里；尚未回答什么；下一步建议"
```

5. 所有登记实验结束后，当前持有者用 `submit --session ... --file report.json` 提交研究结论。字段为 `summary`（文本）、`findings`、`limitations`、`next_steps`（非空文本列表）。全部失败／取消的任务仍能提交失败总结，系统强制标记 execution_failure_only；不能把它当作策略有效性证据。未启动的错误登记可用 cancel <实验ID> --session ... --reason "依据" 取消，仍计入预算。

6. 另一角色检查证据后执行 `review <task> --reviewer reviewer-b --decision continue|defer|stop|revise --rationale "依据"`。`continue` 必须提供 `--next-task task.json`，原子创建父子关联的新任务；`revise` 退回同一任务，保留原尝试与预算。暂缓任务出现新依据后可用 reopen <任务ID> --actor ... --reason "新依据" 重新排队。不同角色名称只是追溯标签，不等同独立模型认证。

新任务JSON：`title`、`question`、`rationale`、`success_criteria`与`stop_criteria`文本列表；可附`scope`、`baseline_runs`、`tags`及`max_experiments`（默认4）。GUI也可创建任务。`export <task> --output ...json`可导出交接包；实际运行和数据仍通过其中ID在共享研究库定位。

## 中断与恢复

`recover <experiment-id>`检查执行收据，核验完整文件、策略、模型代码与独立账本证据，收取完成结果。父 Worker 退出但子进程完成时，可以由下一 Worker 恢复。无收据时显示“需检查PID和日志”，不会猜测子进程已退出并重复执行。执行时限到达会停止本次启动的进程树并保留失败记录；这是操作时限，不是研究有效性门槛。

核心状态是本机 `data/technical/research.sqlite3`（WAL）。执行目录为 `worker_jobs/<experiment-id>`；资金路径仍在 `runs/<run-id>`，快照仍在`snapshots/<snapshot-id>`。不要在有写入时直接复制单独的sqlite文件做备份，应停用写入后连同库与证据备份，或使用SQLite备份API。

## 新 Chat 与隔离分支

对新 Chat 可以直接说：“阅读根目录AGENTS.md和Worker指南，读取任务X的context，领取任务，在范围和预算内完成一轮研究并提交证据。”

新 worktree 的命令需增加全局选项 `--root <LOCAL_REPOSITORY>/lhb_backtest/data/technical`。`--project`默认当前CLI所在项目，代码从该项目冻结，输出进入共享研究库。环境Python和大数据不会随Git复制；共用环境时记录依赖，源码变化通过冻结副本隔离。需要引擎/会计规则变化时另建工程任务并运行回归。

## 当前交付边界

提供模型无关的任务协议、跨进程交接、确定性实验执行及通用展示。V7总控已能自动调用本机Codex完成有限研究循环；没有无限常驻搜索、API金额预算或真正未见验证集管理。总控仍需经过本协议，不能用连续聊天绕过证据要求。

登记执行器支持 `ResearchSpec` 账户回测，以及 `kind=ml` 的固定模型信号分析和完整日线研究矩阵，设计、登记、结果和核查见[机器学习研究指南](ML_RESEARCH_GUIDE.md)。ML结果标为 `registered_ml_analysis`；spec.study=complete时其清单链接已审计原生账户，核查范围与accounts字段明确记录，原V1信号分析仍不属于账户证据。其他纯特征统计等可通过`attach`导入分析包，允许零新增账户实验的负面成果进入submit/review；导入只是事后完整性归档，不等同通用分析执行、数值重现或事前登记。详情见总控指南。不要为了完成任务把重复回测伪装成新研究。源码副本隔离不等于环境容器：Python与第三方包版本被记录，但没有冻结操作系统或限制内存。当前主进程负责超时终止；若主进程本身被强杀，需检查子进程与收据后恢复。数据库和大数据证据不会随Git自动备份。
