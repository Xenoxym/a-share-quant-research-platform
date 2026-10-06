(() => {
  'use strict';
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=v=>v==null?'—':(v*100).toFixed(2)+'%';
  const money=v=>v==null?'—':(v/10000).toFixed(2)+' 万';
  const source=url=>/^https:\/\//.test(url||'')?`<a href="${esc(url)}" target="_blank" rel="noopener">原始研究 ↗</a>`:'预登记对照';
  let api,openRun,startPoll,current;
  const table=(heads,rows)=>`<div class="table-wrap"><table><thead><tr>${heads.map(x=>`<th>${x}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  function story(spec,d){
    const s=spec.strategy;
    return `<div class="strategy-story"><p class="strategy-idea">${esc(d.idea||'基础策略研究')}</p><div class="logic-flow"><div><small>01 · 数据</small><b>${esc(d.raw_data)}</b></div><i>→</i><div><small>02 · 特征</small><b>${esc(d.feature_description)}</b></div><i>→</i><div><small>03 · 交易</small><b>${s.family==='cash'?'持有现金':(s.rebalance==='weekly'?'周末':'月末')+'决定目标，下一交易日开始执行'}</b><span>${esc(d.ranking)}</span></div></div><p class="note">${esc(d.parameter_origin)}</p><details><summary>公式、资格、原始研究与执行边界</summary><p class="formula">${esc(d.formula)}</p><p>${source(d.source)}</p>${['filters','availability','execution','capacity','cost_parameters','limitations'].map(k=>`<p>${esc(d[k]||'')}</p>`).join('')}</details></div>`;
  }
  const verdict=row=>current?.research_reviews?.map(r=>r.decisions?.[row.key]?.label).find(Boolean)||row.evidence.verdict;
  function detail(row){
    const e=row.evidence;
    const bands=(e.joint_active_bands||[]).map(b=>`${b.block_sessions} 日块：${pct(b.lower)} ～ ${pct(b.upper)}`).join('；');
    return `<article class="panel"><div class="panel-heading"><h3>${esc(row.name)}</h3><button data-foundation-run="${row.run_id}">净值、区间与原始案例 ↗</button></div><p><span class="badge">${esc(verdict(row))}</span> ${source(row.definition.source)}</p><p>${esc(row.definition.idea)}</p><p class="note">相对收益统计标签：${esc(e.verdict)}。防御、容量及数据问题需要研究者另行判断。</p><p class="note">${esc(row.definition.formula)}</p><div class="metrics"><div class="metric"><small>全期含费收益</small><strong>${pct(row.metrics.total_return)}</strong></div><div class="metric"><small>零交易成本收益</small><strong>${pct(row.zero_cost.total_return)}</strong></div><div class="metric"><small>最大回撤</small><strong>${pct(row.metrics.max_drawdown)}</strong></div><div class="metric"><small>平均股票仓位</small><strong>${pct(row.metrics.average_exposure)}</strong></div></div><p><b>判断依据</b>：${e.reasons.map(esc).join('；')}。</p><p title="平均日收益差乘252，不是复合年化收益之差；时间块重采样保留日期依赖，同一批候选联合校正。">平均日增量年化：${pct(e.mean_annual_active_return)}。${esc(bands)}</p><p><b>下一步要回答</b>：${esc(e.next_question)}</p><details><summary>逐年表现与需先检查的区间</summary>${table(['年份','策略含费','随机参考','相对财富收益','区间回撤'],e.annual.map(y=>[y.year,pct(y.return),pct(y.reference_return),pct(y.relative_return),pct(y.max_drawdown)]))}${table(['较差相对区间','策略收益','参考收益','相对财富收益'],e.worst_relative_windows.map(w=>[w.start+' — '+w.end,pct(w.strategy_return),pct(w.reference_return),pct(w.relative_return)]))}<p>入选样本估算市值中位数 ${e.selected_median_cap==null?"—":(e.selected_median_cap/1e8).toFixed(2)} 亿元；股本报表距决策日中位数 ${e.selected_median_share_report_age??'—'} 天。</p>${table(['主要亏损股票','累计净损益'],e.worst_stocks.map(s=>[esc(s.stock_code),money(s.net_pnl)]))}</details></article>`;
  }
  async function show(id){
    current=await api('/api/foundations/'+id);
    const b=current;
    document.getElementById('foundation-results').innerHTML=`<div class="panel"><div class="eyebrow">FOUNDATION STUDY / ${esc(id)}</div><h2>不同基础机制的本地回测</h2><p>${esc(b.protocol.notice)}</p>${(b.research_reviews||[]).map(r=>`<p class="notice">${esc(r.summary)}</p>`).join('')}<p class="notice">${esc(b.protocol.data_policy)}</p><p>${esc(b.evidence.reference)}</p>${table(['方向','全期含费','零交易成本','最大回撤','后续区间','研究优先级'],[...b.rows.filter(r=>r.role==='candidate'),...b.rows.filter(r=>r.role!=='candidate')].map(r=>[`<button data-foundation-detail="${esc(r.key)}" class="text-button">${esc(r.name)}</button>`,pct(r.metrics.total_return),pct(r.zero_cost.total_return),pct(r.metrics.max_drawdown),pct(r.evidence.later.return),esc(verdict(r))]))}<p class="note">${esc(b.evidence.inference)}</p></div><div id="foundation-detail">${detail(b.rows.find(r=>r.key==='small')||b.rows[0])}</div>${(b.research_reviews||[]).map(r=>`<div class="panel"><h3>研究者复核与追加实验</h3><p>${esc(r.summary)}</p><details><summary>核查依据、典型失败与数据问题</summary>${(r.findings||[]).map(v=>`<p>${esc(typeof v==='string'?v:JSON.stringify(v))}</p>`).join('')}</details>${(r.followups||[]).map(f=>`<p>${esc(f.question)} ${f.run_id?`<button data-foundation-run="${esc(f.run_id)}">打开复核 ↗</button>`:''} ${esc(f.conclusion||'')}</p>`).join('')}<details><summary>已排序的下一轮研究问题与停止条件</summary>${(r.next_queue||[]).map(q=>`<p><b>${q.priority} · ${esc(q.task)}</b>：${esc(q.reason)}<br>停止条件：${esc(q.stop_rule)}</p>`).join('')}</details></div>`).join('')}`;
    bind();
  }
  function bind(){
    document.querySelectorAll('[data-foundation-run]').forEach(b=>b.onclick=()=>openRun(b.dataset.foundationRun));
    document.querySelectorAll('[data-foundation-detail]').forEach(b=>b.onclick=()=>{document.getElementById('foundation-detail').innerHTML=detail(current.rows.find(r=>r.key===b.dataset.foundationDetail));bind();document.getElementById('foundation-detail').scrollIntoView({behavior:'smooth'});});
  }
  async function load(){
    const root=document.getElementById('foundation-content');
    if(!root.dataset.ready){
      const plan=await api('/api/foundations/plan');
      root.innerHTML=`<div class="section-intro"><div class="eyebrow">DISCOVER / REPLICATE / QUESTION</div><h2>从不同收益机制出发</h2><p>文献给出起点，账户和证据决定下一步。每个方向同时保留假设、来源、失败和疑点。</p></div><div class="panel"><details><summary>查看已登记的 ${plan.trials.length} 个账户定义及公开来源</summary><p>${esc(plan.cost_policy)}</p><p>${esc(plan.selection_policy)}</p>${table(['方向','思想','指标定义','依据'],plan.trials.map(t=>[esc(t.name),esc(t.definition.idea),esc(t.definition.formula),source(t.definition.source)]))}</details><div class="toolbar"><button id="foundation-run" class="primary">按完整协议重跑基础策略研究</button><button id="foundation-refresh">刷新研究记录</button></div><p id="foundation-state" role="status"></p></div><div id="foundation-results"></div>`;
      root.dataset.ready='1';
      document.getElementById('foundation-run').onclick=async()=>{try{await api('/api/foundations',{});startPoll();document.getElementById('foundation-state').textContent='已提交，运行期间可以查看已有证据。';}catch(e){document.getElementById('foundation-state').textContent=e.message;}};
      document.getElementById('foundation-refresh').onclick=()=>load();
    }
    const history=await api('/api/foundations');
    const complete=history.find(r=>r.status==='completed');
    document.getElementById('foundation-state').textContent=history[0]?.status==='running'?'新研究正在运行；下面保留最近完成的结果。':history[0]?.status==='failed'?'最近运行失败：'+history[0].error:complete?'研究已完成；下方结论用于安排后续实验，不是实盘推荐。':'尚未运行此协议。';
    if(complete)await show(complete.suite_id);
  }
  window.foundationWorkbench={story,load,show,init:function(a,o,p){api=a;openRun=o;startPoll=p;}};
})();
