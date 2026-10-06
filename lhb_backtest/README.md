# A-Share Quant Research Platform

这个项目用来研究 A 股策略，并检查模型预测能否转化为可核查的账户收益和回撤。它包含本机浏览器工作台、命令行、数据快照、机器学习研究和有预算的研究任务管理。

当前默认入口是技术与机器学习研究。旧龙虎榜事件模块、`main.py` 和 `samples/` 保留作历史接口，不能把旧“机构五日”案例当作默认策略。

## 安装与最小检查

建议 Python 3.12。本地验证平台是 Windows / Python 3.12；发行元数据允许 Python 3.10 及以上，其余版本和平台需要 CI 进一步确认。

在仓库根目录创建环境，然后进入应用目录。命令不下载行情，也不调用 Codex。

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".\lhb_backtest[research,dev]"
cd lhb_backtest
..\.venv\Scripts\python.exe -X utf8 -m pytest -q --basetemp ../.qa_ci
..\.venv\Scripts\python.exe -X utf8 tools/technical_research.py template
```

Linux/macOS 对应使用 `../.venv/bin/python`；上述系统尚需实际验证。`research` 包含 scikit-learn 和 hmmlearn；`dev` 包含测试和静态检查工具。数据服务 token 放在环境变量，配置模板内保持为空。

发行包名为 `a-share-quant-research-platform`，内部导入仍为 `src.*` / `compat.*`。wheel 含模块和工作台静态资源；下面命令需要源码工作区中的 `tools/`、配置和文档。

## 打开工作台

```powershell
..\.venv\Scripts\python.exe -X utf8 tools/technical_research.py serve --port 8765
```

服务只监听本机。空工作区可以查看入口、定义策略和任务，不能复现原作者已有结果。先配置并准备有使用权的数据；公开源码不包含行情、模型、任务库或冻结研究证据。

当前技术账户主要支持主板及其交易约束，不声称已经支持全 A 股所有板块、分钟或实时 L2，更没有实盘下单接口。

## 当前有哪些能力

| 层 | 已有实现 | 边界 |
| --- | --- | --- |
| `src/technical` | 策略定义、持仓与费用账户、逐日决策、结果工作台、独立账本核查 | 受数据覆盖和执行假设限制；核查不证明供应商正确 |
| `src/mlresearch` | 特征/标签契约、岭回归与受限提升树、时间划分、原生账户矩阵、固定评分组合、财务/市场上下文扩展 | 新恢复扩展的工程测试不代表新真实研究已成功 |
| `src/researchops` | 任务租约、预算、登记冻结、执行收据、证据导入、交接与复核 | 分析导入是事后文件完整性核查 |
| 有界总控 | 本机 Codex CLI 研究、两名新会话只读复核、综合裁决及有限接续 | 需要显式启动和登录；有次数/轮数/时限，不是无限自主系统 |
| 旧事件层 | 龙虎榜数据清洗、事件因子和案例工具 | 历史研究入口，数据需自行合法取得 |

类型安全 Alpha DSL、大规模自动发现、多模型表示接口、强制多重比较和真正封存的未见日期，属于后续建设目标。请从[长期纲领](docs/PROJECT_CHARTER.md)、[系统蓝图](docs/ML_SYSTEM_BLUEPRINT_20261005.md)与[研究雷达](docs/RESEARCH_RADAR.md)查看完整范围。

## 数据与第一轮研究

1. 阅读[数据约定](docs/DATA_CONTRACT.md)。在 `config/config.yaml` 配置自己的行情导出目录；默认的相邻 SimTradeData 路径只是原工作区约定，不是随源码安装的数据服务。
2. 首次打开工作台会建立空的本机研究库；然后运行 `tools/research.py catalog` 和 `list`，查看实际快照、能力与历史任务。只读命令不会自行初始化库。空候选不包含原本的已登记快照；不要把教程里的 ID 当成自己的输入。
3. 按 [Worker 指南](docs/WORKER_GUIDE.md)建立任务，领取独有 session，写明确的假设、否证条件、日期和有限预算，再 `register/run`。长分析续租，结束提交证据或交接。
4. ML 的输入、目标、训练、账户和核查见[机器学习研究指南](docs/ML_RESEARCH_GUIDE.md)。预测排序分数用于诊断，策略比较最终看收益、回撤、成本、风险暴露与不确定性。
5. 新工作区若共用原研究库，需要显式 `--root`。源行情及快照只读；冻结代码和历史结果不覆盖。换模型或换会话不会把已看历史变成未见样本。

直接调用旧工具可能发起行情网络请求；启动总控可能消耗 Codex 使用额度。安装和软件测试不自动触发这些研究动作。

## 文档与交付

- [当前目录、公开范围与验证边界](docs/PUBLIC_DELIVERY.md)
- [研究总控](docs/CONTROLLER_GUIDE.md)
- [从零讲解模型输入与流程](docs/ML_FROM_ZERO.md)：初版历史教程，后续账户目标见研究指南
- [账户结果说明](docs/ML_ACCOUNT_RESULTS.md)：历史结果，不是新版本收益承诺
- [工程建设与交付规划](docs/CODEX_PROJECT_DELIVERY_PLAN_20261005.md)

公开文档保留历史日期，部分结果引用的是本机证据，候选源码不包含这些结果文件。许可证尚未决定；项目当前不提供 MIT 或其他开源许可证授权。数据使用和再分发许可需另行确认。
