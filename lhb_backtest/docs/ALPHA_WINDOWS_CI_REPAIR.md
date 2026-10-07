# E12P：Windows 公开检查发现的进程身份问题

这是工程兼容性修复，未新增真实策略训练或账户回测，也没有修订旧收益结论。

## 触发与修复

公开提交 75bb69a 的 GitHub 检查中，Linux 通过；Windows 出现 22 个失败与 35 个错误，共同原因是 worker 身份检查拒绝子进程。云端使用直接 Python，本机通常使用虚拟环境 Python。原实现将“允许隐藏控制台宿主”错误地与“存在虚拟环境启动器”绑在一起。

现在直接 Windows Python 也可以登记一个隐藏控制台宿主。检查仍要求它是直接子进程、路径为系统 System32/conhost.exe、参数为已观察到的 0x4，并保留 PID 与创建时间。实际 Python 子解释器角色仍仅允许虚拟环境启动器分支，且可执行文件与完整作业参数一致。Linux 不允许这个控制台角色。多余子进程、重复角色、错误路径或参数仍拒绝。

这不是任意子进程白名单。以后出现另一种合法系统宿主形态时也需先保留失败证据、核查身份，再明确扩展协议。

## 验证与记录

新增 16 个软件检查，包括直接 Python/虚拟环境的角色矩阵、九种错误身份，以及在本机真正启动两种解释器的隐藏窗口检查。这些进程仅等待一秒，不运行策略或市场数据。

本机结果不替代云端结果；发布修复后仍须分别查看 Windows 与 Linux 的 GitHub 检查。原 E12 冻结源码、失败云端运行和阶段完成证据保留，修复以单独的 E12P 工程任务登记。

微软的 [进程创建标志说明](https://learn.microsoft.com/en-us/windows/win32/procthread/process-creation-flags)描述 CREATE_NO_WINDOW 的用途；[Windows Console 说明](https://devblogs.microsoft.com/commandline/windows-command-line-inside-the-windows-console/)区分应用进程与控制台宿主。这里具体允许的路径、参数与父子形态依据本项目观察，文档并不保证所有 Windows 环境都会呈现相同的树。

完整监督、资源预留与收取边界见 [Alpha 批次执行说明](ALPHA_BATCH_EXECUTION.md)。修复完成后接续候选筛选模块 E13。
