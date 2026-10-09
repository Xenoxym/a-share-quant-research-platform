from pathlib import Path
import argparse
import html
import json


def esc(value):
    return html.escape(str(value), quote=True)


def diagram(simple):
    suffix = "simple" if simple else "detail"
    if simple:
        groups = [
            ("1  取数据", "行情 · 公告 · 龙虎榜"),
            ("2  整理信息", "过去可知的特征与公式"),
            ("3  给股票打分", "模型学习 / 公式计算"),
            ("4  形成持仓", "资格 · 排名 · 权重"),
            ("5  计算账户", "现金 · 持股 · 费用 · 权益"),
            ("6  判断结果", "收益 · 回撤 · 独立核查"),
        ]
        width, height = 760, 730
        nodes = []
        for i, (title, sub) in enumerate(groups):
            y = 38 + i * 112
            nodes.append(f'<rect x="120" y="{y}" width="520" height="82" rx="14" fill="#ffffff" stroke="#a7c7df"/><text x="380" y="{y+32}" class="title" text-anchor="middle">{esc(title)}</text><text x="380" y="{y+61}" class="sub" text-anchor="middle">{esc(sub)}</text>')
            if i < len(groups)-1:
                nodes.append(f'<path d="M380 {y+83} V{y+106}" stroke="#49779b" stroke-width="2" marker-end="url(#arrow-{suffix})"/>')
        label = "当前功能最简图"
    else:
        bands = [
            ("数据来源与冻结", ["日线行情、交易日历、历史状态与公司行动", "龙虎榜原报告 / 席位；财报与公告股本", "保存快照、覆盖范围和信息可用时间"]),
            ("信息加工与候选", ["价量16项；板型29项；龙虎榜、财务、市场", "特征字典、统一组装；日历序列及缺失掩码", "受限组合公式、计算算子、共享缓存、快速初筛"]),
            ("学习与预测", ["六类表格模型；按时间划分训练和预测", "只使用已成熟标签；预处理仅学习训练数据", "目标变换与模型存取独立接口；正式worker接入待做"]),
            ("组合与资金账户", ["当时资格、分数排序、持仓数量、权重与调仓", "原资金引擎：订单、现金、持股、权益事件", "零费与配置费分开运行；收益 / 回撤 / 仓位 / 换手"]),
            ("独立核查与展示", ["重新生成选择名单，逐日检查现金与资产算数", "不认证供应商历史，也不重新拟合模型分数", "原工作台展示任务、账户、订单与诊断结果"]),
        ]
        width, height = 980, 970
        nodes = []
        for i, (title, lines) in enumerate(bands):
            y=36+i*178
            nodes.append(f'<rect x="26" y="{y}" width="650" height="146" rx="14" fill="#ffffff" stroke="#a7c7df"/><text x="50" y="{y+33}" class="title">{esc(title)}</text>')
            for j, line in enumerate(lines):
                nodes.append(f'<text x="50" y="{y+65+j*29}" class="sub">{esc(line)}</text>')
            if i < len(bands)-1:
                nodes.append(f'<path d="M350 {y+147} V{y+172}" stroke="#49779b" stroke-width="2" marker-end="url(#arrow-{suffix})"/>')
        nodes.append('<rect x="700" y="36" width="254" height="858" rx="14" fill="#eaf3f9" stroke="#a7c7df"/>')
        support=["贯穿全流程的支持", "任务与共享进度", "事前登记 / 固定预算", "保存数据和代码版本", "有限进程与资源监督", "失败 / 取消全部保留", "原结果收取与恢复", "本机研究证据留档", "两名只读复核", "选择性GitHub发布"]
        for i, line in enumerate(support):
            cls="title" if i==0 else "sub"
            nodes.append(f'<text x="724" y="{76+i*76}" class="{cls}">{esc(line)}</text>')
        label="当前有用模块的功能架构"
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="label-{suffix}"><title id="label-{suffix}">{label}</title><style>.title{{font:600 23px system-ui,Arial,sans-serif;fill:#173b57}}.sub{{font:19px system-ui,Arial,sans-serif;fill:#405567}}</style><defs><marker id="arrow-{suffix}" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="none" stroke="#49779b"/></marker></defs><rect width="100%" height="100%" fill="#f4f8fb"/>{"".join(nodes)}</svg>'


def render(app):
    docs=app/"docs"
    data=json.loads((docs/"PROJECT_MAP.json").read_text(encoding="utf-8"))
    assert data["schema"]=="human-project-map-v1"
    ids=[x["id"] for x in data["packages"]]
    assert len(ids)==len(set(ids)) and set(ids)>={"E"+str(i).zfill(2) for i in range(30)}|{"R"+str(i) for i in range(6)}
    for row in data["packages"]:
        assert (docs/row["guide"]).is_file(),row["guide"]
    for link in data["navigation"]:
        for kind in ("md","html"):
            if kind in link:assert (docs/link[kind]).is_file(),link[kind]
    simple,detail=diagram(True),diagram(False)
    (docs/"PROJECT_ARCHITECTURE_SIMPLE.svg").write_text(simple+"\n",encoding="utf-8")
    (docs/"PROJECT_ARCHITECTURE_DETAIL.svg").write_text(detail+"\n",encoding="utf-8")
    c=data["phase_counts"]
    intro="我们建设的是：把已有市场信息变成可检验的投资方法，再用真实资金账户的收益和回撤判断它是否值得继续。当前软件模块已具备研究闭环的基本接线；真实账户计算与独立接受的进度分别见下文。"
    boundary="图中只画已经实现的有用功能。工程验收表示程序与约束经过检查，不表示有市场优势；模块可用也不表示所有大规模数据路径都已认证。神经网络、图模型、遗传/RL/LLM自动公式搜索仍在后面的计划表。"
    recent=data.get("phase_summary") or "本轮完成E19正式工程3作业：2完成、1按计划启动前取消。共登记6个合成账户情景，实际执行4个；模型拟合0次。两个完成作业从原结果恢复，没有重训。合成分数仅用于接线检查，不能视为策略收益。此前实际拟合输出到旧账户桥接的兼容缺陷已修复，旧验收表述也已更正；没有据此认定旧真实收益数字错误。"
    nav_md="\n".join(f'- [{x["title"]}]({x["md"]})'+(f'（[网页]({x["html"]})）' if "html" in x else "") for x in data["navigation"])
    table=["| 工作包 | 人话说明 | 当前状态 | 已做到什么 / 还缺什么 | 依赖 |","|---|---|---|---|---|"]
    cards=[]
    for row in data["packages"]:
        table.append(f'| [{row["id"]}]({row["guide"]}) | {row["title"]} | {row["status"]} | {row["detail"]} | {row["dependencies"]} |')
        cells=[row["id"],row["title"],row["status"],row["detail"],row["dependencies"]]
        labels=["编号","工作内容","当前状态","完成与缺口","依赖"]
        tds=[]
        for i,(value,label) in enumerate(zip(cells,labels)):
            text=f'<a href="{esc(row["guide"])}">{esc(value)}</a>' if i==0 else esc(value)
            tds.append(f'<td data-label="{label}">{text}</td>')
        cards.append(f'<tr id="package-{esc(row["id"])}">{"".join(tds)}</tr>')
    footer="进度依据：本机任务库、CHECKPOINT及阶段收据。本页是当前摘要，不覆盖历史冻结结果。旧报告的‘下一步’指写作当时，当前下一步以本页和任务库为准。更新方式：修改PROJECT_MAP.json后运行python tools/render_project_map.py；阶段报告仍须关联原始证据。"
    counters=f'本E19阶段：工程{c.get("engineering_jobs",3)}作业（{c.get("engineering_completed",2)}完成/{c.get("engineering_cancelled",1)}取消）；模型登记{c.get("registered_fit_intents",0)}项，有收据确认拟合{c["real_fits"]}次、模型核查完成{c.get("audited_learning_completed",0)}项；真实账户已计算{c["real_accounts"]}情景，其中当前核查接受{c.get("qualified_account_scenarios",0)}情景。上限{c["fit_limit"]}次真实拟合/{c["real_account_limit"]}个真实账户情景，外部费用${c["paid_cost_usd"]}。'
    md=f'# 项目总览：当前架构、全计划和进度\n\n更新（UTC）：{data["updated_at_utc"]} · 当前工作包：{data["current_package"]}\n\n{intro}\n\n## 从这里看整个计划\n\n{nav_md}\n\n## 现在进行到哪里\n\n{counters}\n\n{data["current_result"]}\n\n{recent}\n\n## 最简架构：六步就能理解\n\n![当前最简功能架构](PROJECT_ARCHITECTURE_SIMPLE.svg)\n\n## 略详细架构：每一步里有什么\n\n![当前有用模块](PROJECT_ARCHITECTURE_DETAIL.svg)\n\n{boundary}\n\n## 一周例子\n\n示例：周五收盘后，以当时可知的信息给每只合格股票打分；按预先固定的规则选股和分配资金；下一个交易日开盘尝试调仓，此后逐日计算账户。新学习协议默认预测两次计划调仓开盘之间的收益，不是某天的绝对价格，也不是直接训练最大回撤。固定随机对照中的11是种子，不是11只股票。E19已固定共同主板名单、周调仓、100只目标持仓及四个对照；具体实验说明见本批报告。\n\n## 接下来四步\n\n'+"\n".join(f'{i}. {x}' for i,x in enumerate(data['near_term'],1))+f'\n\n## 全部工作包：原计划与补充包\n\n原编号表示设计和依赖，不是按编号顺序执行；E15—E18先补接线，E12—E14后补有限执行能力，详见建设记录。没有用完成包数给项目虚构完成百分比。\n\n'+"\n".join(table)+f'\n\n## 报告接续规则\n\n以后每份阶段结果和最终回复都附本总览、完整蓝图、完整工作包计划、详细建设记录；重要策略成果仍以经核查的收益/回撤改善为准，不能仅以RankIC或工程通过作为停止理由。\n\n{footer}\n'
    (docs/"PROJECT_MAP.md").write_text(md,encoding="utf-8")
    nav="".join(f'<a href="{esc(x.get("html",x["md"]))}">{esc(x["title"])}</a>' for x in data["navigation"])
    steps="".join(f'<li>{esc(x)}</li>' for x in data["near_term"])
    css='body{margin:0;background:#f4f7fa;color:#233c4e;font:16px/1.75 system-ui,Arial,sans-serif}main{max-width:1120px;margin:auto;padding:40px 28px 72px}h1{font-size:34px;line-height:1.35}h2{margin-top:44px;font-size:24px}a{color:#126294;text-decoration-thickness:1px;text-underline-offset:3px}.muted{color:#586c7b}.card{background:white;border:1px solid #d7e3eb;border-radius:16px;padding:22px;margin:18px 0}nav{display:flex;flex-wrap:wrap;gap:10px}nav a{background:#e7f2fa;padding:9px 14px;border-radius:8px}.now{border-left:5px solid #1476a8}.diagram{max-width:780px;margin:16px auto}.diagram svg{display:block;width:100%;height:auto;border-radius:14px}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{padding:13px 11px;text-align:left;vertical-align:top;border-bottom:1px solid #dce6ed}th{background:#eaf2f7}td:first-child{white-space:nowrap;font-weight:600}td:nth-child(3){min-width:96px}td:nth-child(4){width:42%}.note{background:#edf4f8;padding:16px;border-radius:10px}li{margin:10px 0}footer{margin-top:40px;font-size:14px;color:#586c7b}@media(max-width:680px){main{padding:22px 16px 48px}h1{font-size:27px}h2{font-size:22px}.card{padding:16px}nav a{display:block;width:100%;box-sizing:border-box}thead{display:none}table,tbody,tr,td{display:block}tr{border:1px solid #d7e3eb;border-radius:12px;margin:12px 0;padding:10px}td,td:nth-child(4){width:auto;min-width:0;padding:5px 4px;border:0}td:before{content:attr(data-label) "：";font-weight:600;color:#536c7d}td:first-child{white-space:normal}.diagram{margin:8px -4px}}'
    doc=f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>项目总览：架构与全计划</title><style>{css}</style></head><body><main><p class="muted">固定入口 · 更新UTC {esc(data["updated_at_utc"])}</p><h1>我们的项目由什么组成，<br>现在做到哪里？</h1><p>{intro}</p><nav>{nav}</nav><section id="current" class="card now"><h2 style="margin-top:0">当前：{data["current_package"]} 真实闭环研究进行中</h2><p>{esc(counters)}</p><p>{esc(data["current_result"])}</p><details><summary>这一轮具体做了什么</summary><p>{esc(recent)}</p><p>详细原始证据保留在本机阶段目录，不随公开文档上传。</p></details></section><h2 id="simple">最简架构：六步就能理解</h2><div class="diagram">{simple}</div><p><a href="PROJECT_ARCHITECTURE_SIMPLE.svg">打开大图</a></p><h2 id="detail">略详细架构：每一步里有什么</h2><div class="diagram">{detail}</div><p><a href="PROJECT_ARCHITECTURE_DETAIL.svg">打开大图</a></p><p class="note">{esc(boundary)}</p><h2>一周例子</h2><p>示例：周五收盘后，用当时可知的信息给股票打分，按事前规则选股和分配资金；下个交易日开盘尝试调仓，随后逐日算账户。</p><p>默认学习目标是两次计划调仓开盘之间的收益，不是第五天绝对价格，也没有直接训练最大回撤。“固定随机11”的11是种子，不是股票数量。E19的日期、共同主板名单、100只目标持仓和四个对照已固定；最新实验状态见本页摘要。</p><h2 id="next">接下来四步</h2><ol>{steps}</ol><h2 id="packages">全计划：R0—R5、E00—E29及补充包</h2><p>编号表达依赖，不是执行先后；实际路径见建设记录。工程验收≠真实收益验证，未按包数量虚构完成百分比。</p><table><thead><tr><th>编号</th><th>工作内容</th><th>当前状态</th><th>完成与缺口</th><th>依赖</th></tr></thead><tbody>{"".join(cards)}</tbody></table><h2>以后怎样报告</h2><p>每份阶段结果和最终回复都附本总览、完整蓝图、完整工作包计划、详细建设记录。重要策略成果以核查后的账户收益/回撤改善为依据；工程通过与RankIC变化单独标注。</p><nav>{nav}</nav><footer>{esc(footer)}</footer></main></body></html>'
    (docs/"PROJECT_MAP.html").write_text(doc+"\n",encoding="utf-8")
    print(json.dumps({"rendered_files":4,"packages":len(ids),"current":data["current_package"]},ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="Render the human project map from its explicit status JSON.")
    parser.add_argument("--app",type=Path,default=Path(__file__).resolve().parents[1])
    render(parser.parse_args().app.resolve())
