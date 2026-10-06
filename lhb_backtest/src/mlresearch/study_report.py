"""Explain the actual experiment in ordinary language, including all cases."""


def render_report(r):
    pct = lambda x: "—" if x is None else f"{100*x:.2f}%"
    accounts = {a["case"]: a for a in r["accounts"]}
    configured = {k: next(p for p in a["portfolios"] if p["metrics"]["scenario"] == "configured") for k,a in accounts.items()}
    zero = {k: next(p for p in a["portfolios"] if p["metrics"]["scenario"] == "zero_transaction_cost") for k,a in accounts.items()}
    lines = ["# 这次机器学习到底做完了什么", "",
        f"运行 `{r['run_id']}`；登记实验 `{r['research_context']['experiment_id']}`；任务 `{r['research_context']['task_id']}`。",
        "", "这次已经真正跑了资金账户。以前只问‘模型选中的股票后来平均涨多少’，现在还计算每天剩多少钱、持有哪些股票、哪些买卖成功，以及账户最终赚亏多少。",
        "", f"固定矩阵共 {len(accounts)} 个策略，每个分别跑配置费用和零交易费用，共 {len(accounts)*2} 个账户情景；新增 {r['new_fits']} 次拟合，复用原来两个模型，没有重新训练原案例。所有列出的结果均保留。",
        "", "## 从周五到下周：实际发生的事", "",
        "1. 周内每天只给已有持仓记账、检查未完成委托；每周最后交易日收盘后做一次选股。",
        "2. 找出当时合格的主板股票：非ST、60个连续交易日、20日均成交额至少2000万、存在有效已公告股本。这是共同名单，不是先挑小市值。",
        "3. 每只股票给模型一行16个X：当日K线、过去5/20/60日收益与波动、均线偏离、量额变化等汇总。加市值实验才额外输入第17列。",
        "4. 各方法独立排序选前100只，目标总投入98%，每只目标0.98%。下个交易日开盘开始调整持仓；不是每天清仓再买，也不是每只股强制持有恰好五天。",
        "5. 老持仓不再入选就尝试卖出，仍入选就按目标调整。涨跌停、停牌、整手、容量和现金规则来自项目已有账户引擎，未完成目标每日重试直到被下个周目标替换。当天卖出所得保守地下一交易日才可用于新买入。",
        "6. 每天记录现金、股票市值、分红应收及总权益。2026年9月24日以持仓估值结束，不额外强制清仓。",
        "", "## 模型学的Y、训练与参数选择", "",
        "原模型仍预测五个交易日后的相对收盘衔接涨幅。新模型改学‘下次开盘买入，到下一次周调仓开盘’的衔接涨幅，再减当时共同名单的平均结果。遇到节假日，期限随真实调仓日历变化。两者都不是未来股价，也都没有直接学习账户利润。",
        "", "岭回归拟合平方误差加L2正则；提升树逐轮拟合损失的梯度。默认固定模型只用2020—2022成熟标签。验证期是2023，账户测试是2024—2026-09-24。原参数模型与新目标模型各自都展示，不根据测试结果选冠军。",
        "", "参数对照在2023验证期比较三种岭回归alpha及三种提升树配置，按逐日期平均RankIC选择，然后用截至2023年底的成熟标签重训。另一个独立对照在2024、2025、2026年开始前，每次用2020年至上年末成熟标签扩展重训；参数保持默认。",
        "", "预测不要求知道未来。末端尚没有完整未来收益的股票仍会评分和交易；训练及学习评价按标签成熟日期截断。模型、填补器、标准化器、输入列、训练截止日和全部调参尝试都已保存。所有历史已被查看，这里的‘测试’指拟合时间隔离。",
        "", "## 真实账户结果", "",
        "年化是账户总权益按252交易日换算；最大回撤是历史峰值至后续低点的跌幅。夏普使用零无风险利率，只作诊断。零费用列是重新计算现金和股数的独立账户。",
        "", "| 策略 | 配置费用年化 | 累计收益 | 最大回撤 | 零费用年化 | 夏普(零利率) |",
        "|---|---:|---:|---:|---:|---:|"]
    for key,a in accounts.items():
        p=configured[key]; m=p["metrics"]; z=zero[key]["metrics"]
        lines.append(f"| {a['name']} | {pct(m['annualized_return'])} | {pct(m['total_return'])} | {pct(m['max_drawdown'])} | {pct(z['annualized_return'])} | {p['sharpe_zero_rate']:.2f} |")
    lines += ["", "## 每年表现与对照", "", "2026仅包含截至9月24日的累计收益；以下没有把部分年度收益称为完整年度收益。",
        "", "| 策略 | 2024 | 2025 | 2026截至9月24日 |", "|---|---:|---:|---:|"]
    for key in ["small_cap", "fixed_random", "ridge_legacy", "tree_legacy", "ridge_open", "tree_open", "tree_size", "ridge_rolling", "tree_rolling", "ridge_tuned", "tree_tuned"]:
        years={a["year"]:a["return_value"] for a in configured[key]["annual"]}
        lines.append(f"| {accounts[key]['name']} | {pct(years.get('2024'))} | {pct(years.get('2025'))} | {pct(years.get('2026'))} |")
    lines += ["", "## 为什么需要消融与规模检验", "",
        "四个删除组分别移除收益/均线偏离、波动率、量额、当日K线形状，再用完全相同流程重训。‘删掉后变差’是这个配置可能依赖该组信息的迹象；一次历史消融不能证明这一组独立创造Alpha。所有删除实验都有实际账户。",
        "", "价量加市值检查增加第17列的影响。按市值将完整名单分为十组，每组各取10只的模型，与每组各取10只的固定随机对照比较，减少粗略规模分布差异。另保存剔除规模、波动、近期收益及流动性线性关联后，预测与残差收益的排序相关；没有行业中性化，也不是因果结论。",
        "", "| 条件检验 | 规模组内RankIC | 残差RankIC | 入选中最小规模十分组占比 |", "|---|---:|---:|---:|"]
    exposure={a["model"]:a for a in r["exposures"]}; conditional={a["model"]:a for a in r["conditional"]}
    for key in ["tree_open", "tree_size", "small_cap", "tree_shuffled", "tree_size_balanced", "random_size_balanced"]:
        c=conditional[key]; e=exposure[key]
        lines.append(f"| {accounts[key]['name']} | {c['within_size_rank_ic']:.4f} | {c['residual_rank_ic']:.4f} | {pct(e['small_decile_fraction'])} |")
    lines += ["", "## 对小市值的优势有多确定", "",
        "下面按相同日历周配对实际账户周收益，以连续4周循环块重抽样400次。区间是平均每周差值的逐项95%区间，单位为百分点，不是年化收益区间；没有校正整个项目的多重尝试。跨零意味着这份样本无法清晰区分该项平均优势。",
        "", "| 策略 | 配置费用平均周收益差 | 95%块区间 |", "|---|---:|---:|"]
    for key in ["ridge_legacy", "tree_legacy", "ridge_open", "tree_open", "tree_size", "ridge_rolling", "tree_rolling", "ridge_tuned", "tree_tuned"]:
        p=configured[key]; lo,hi=p["weekly_delta_interval"]
        lines.append(f"| {accounts[key]['name']} | {pct(p['mean_weekly_return_delta_vs_size'])} | {pct(lo)} 至 {pct(hi)} |")
    lines += ["", "## 参数选择账本", "", "| 模型 | 参数 | 2023验证RankIC | 选择 |", "|---|---|---:|---|"]
    for a in r["tuning"]:
        lines.append(f"| {a['kind']} | `{a['parameters']}` | {a['valid_mean_rank_ic']:.5f} | {'是' if a.get('selected') else '否'} |")
    lines += ["", "## 到哪里取出结果", "",
        "父运行目录在 `data/technical/ml_runs/"+r["run_id"]+"/`。`panel.parquet` 是实际X与原标签，`open_labels.parquet` 是新标签，`predictions.parquet` 是全候选逐股评分。`fits.json`、`assignments.json`、`models/`、`tuning.json` 记录每次拟合和每个年份使用哪个模型。",
        "", "`account_index.json` 链接19个真实账户，位于 `data/technical/runs/<run-id>/`。每个都有 `decisions.parquet`、`schedule.json` 和两种情景的 `equity`、`positions`、`orders`、`fills`、`ledger`、`targets`、`warnings` 文件；工作台可打开账户明细。`exposures.parquet` 与 `conditional_metrics.parquet` 是逐周暴露及条件检验。",
        "", "## 完成边界", "",
        "本次完成当前日线数据的离线研究闭环：输入→成熟标签→拟合→验证期参数对照→预测→选股→实际账户→对照/消融/年度重训→数值核查与报告。实盘服务、分钟数据、L2订单簿及新的外部数据属于另外的工程范围。数值审计不是独立研究复核。", ""]
    lines += ["- "+v for v in r["limitations"]]
    return "\n".join(lines)+"\n"
