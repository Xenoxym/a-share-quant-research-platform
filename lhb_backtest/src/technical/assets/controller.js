/* The local journal is authoritative; no model credentials enter the browser. */
window.controllerWorkbench=(()=>{
  'use strict';
  let api,showTask,selected,timer,version=0;
  const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const states={queued:'待启动',starting:'启动中',running:'运行中',completed:'已收束',needs_attention:'需要处理',budget_exhausted:'预算到达',stopped:'已停止',failed:'执行失败',reserved:'等待执行'};
  const roles={researcher:'研究者',method_reviewer:'方法复核',data_reviewer:'数据复核',synthesis:'综合裁决'};
  const decisions={accept:'接受结论',revise:'退回补证',defer:'暂缓',continue:'新假设进入下一轮',stop:'收束本设计'};
  const list=a=>`<ul>${(a||[]).map(v=>`<li>${esc(v)}</li>`).join('')}</ul>`;
  const taskLink=id=>`<button class="text-button" data-controller-task="${esc(id)}">${esc(id)} ↗</button>`;
  function fail(e){$('controller-message').textContent=e.message||e;}
  function bindTasks(){document.querySelectorAll('[data-controller-task]').forEach(b=>b.onclick=()=>showTask(b.dataset.controllerTask));}
  async function detail(id){
    selected=id;const current=++version,c=await api('/api/research/campaigns/'+id);if(current!==version)return;
    const s=c.state,cfg=c.config,jobs=c.jobs,usage=jobs.reduce((a,j)=>a+(j.state.usage?.input_tokens||0)+(j.state.usage?.output_tokens||0),0);
    const rounds=[...new Set(jobs.map(j=>j.round))].sort((a,b)=>b-a);
    $('controller-detail').innerHTML=`<div class="controller-heading"><div><span class="eyebrow">${esc(c.id)}</span><h2>${esc(cfg.title)}</h2></div><span class="badge task-status-${s.status}">${states[s.status]}</span></div>
      <p class="controller-stage">${esc(s.stage)}</p><div class="controller-meters"><div><b>${jobs.length} / ${cfg.max_calls}</b><span>Agent 调用 · 失败保留</span></div><div><b>${Math.min(s.round,cfg.max_rounds)} / ${cfg.max_rounds}</b><span>轮次上限</span></div><div><b>${jobs.filter(j=>j.state.status==='running').length} / ${cfg.max_parallel}</b><span>正在执行 / 并发上限</span></div><div><b>${usage.toLocaleString()}</b><span title="来自已结束CLI会话的输入与输出token合计；不等同实时消耗或金额上限。">已报告 token ⓘ</span></div></div>
      <div class="toolbar">${s.status==='queued'?'<button id="controller-run" class="primary">启动这轮研究</button>':''}${['queued','starting','running'].includes(s.status)?'<button id="controller-stop">停止总控</button>':''}<span class="note">单次 ${cfg.job_timeout_seconds/60} 分钟 · 总时限 ${cfg.wall_seconds/60} 分钟</span></div>
      ${s.error?`<p class="controller-blocker">${esc(s.error)}</p>`:''}
      ${[...(s.history||[])].reverse().map(h=>`<article class="controller-verdict"><span class="eyebrow">第 ${h.round} 轮 · 总控结论</span><p>${esc(h.summary)}</p>${h.decisions.map(d=>`<div><b>${decisions[d.decision]}</b> ${taskLink(d.task_id)}<details><summary>查看完整裁决依据</summary><p class="controller-rationale">${esc(d.rationale)}</p></details>${d.next_task?`<p>下一项：${esc(d.next_task.question)}</p>`:''}</div>`).join('')}</article>`).join('')}
      ${rounds.map(round=>`<h3>第 ${round} 轮 · 独立执行记录</h3><div class="controller-agents">${jobs.filter(j=>j.round===round).map(j=>{const v=j.state,r=v.result;return `<article class="controller-agent ${v.status}"><div class="panel-heading"><b>${roles[j.role]}</b><span class="badge">${states[v.status]}</span></div><p>${j.task_id==='all'?'汇总本轮全部研究':taskLink(j.task_id)}</p><p class="note">${esc(j.id)}${v.thread_id?`<br>独立会话 ${esc(v.thread_id)}`:''}</p>${r?`<p>${esc(r.summary)}</p>${r.verdict?`<b>${decisions[r.verdict]}</b>`:''}${r.blockers?.length?`<div class="controller-blocker">${list(r.blockers)}</div>`:''}<details><summary>查看核对依据与限制</summary>${list(r.findings)}${list(r.limitations)}${list(r.next_steps)}${r.decisions?`<pre>${esc(JSON.stringify(r.decisions,null,2))}</pre>`:''}</details>`:`<p>${esc(v.error||v.last_message||'正在读取资料与检查证据；完成后由总控收取结构化结果。')}</p>`}<details><summary>执行与文件追溯</summary><p class="note">${esc(v.folder||'尚未创建执行目录')}<br>此目录保存 prompt、context、final、events 和 stderr。租约凭据不通过页面提供。</p><p class="note">${v.result_hash?'结果 SHA256 '+esc(v.result_hash):'尚无已验收的结果文件'}</p></details></article>`;}).join('')}</div>`).join('')}
      ${s.pending.length?`<p>待处理任务：${s.pending.map(taskLink).join(' ')}</p>`:''}
      <details><summary>运行边界与会话在哪里</summary><p>使用本机已登录的 Codex CLI，继承本机模型与思考设置。每名研究者、复核者和综合裁决者各开新会话；相同模型仍可能有共同盲点。</p><p>这些后台会话不保证进入 Codex 桌面侧栏。本页提供独立的派发、过程、结论和文件追溯。调用数与时限是硬限制，token 仅在会话结束后报告，尚非金额预算。</p><p>研究者使用独立目录、共享证据库；不是环境容器。任意分析脚本的搜索次数仍依赖研究协议约束。总控退出异常时先检查进程和日志，不会自动重复执行。</p></details>`;
    if($('controller-run'))$('controller-run').onclick=async()=>{try{await api(`/api/research/campaigns/${id}/run`,{});await detail(id);}catch(e){fail(e);}};
    if($('controller-stop'))$('controller-stop').onclick=async()=>{try{await api(`/api/research/campaigns/${id}/stop`,{});await detail(id);}catch(e){fail(e);}};
    bindTasks();clearTimeout(timer);
    if(['queued','starting','running'].includes(s.status))timer=setTimeout(()=>{if(!document.querySelector('[data-page="campaigns"]').hidden)detail(id).catch(fail);},5000);
  }
  async function createForm(){
    const tasks=await api('/api/research/tasks'),available=tasks.filter(t=>['queued','review'].includes(t.status));
    $('controller-create').innerHTML=`<form class="panel" id="campaign-form"><h3>给总控一组问题和明确的边界</h3><p>可并行研究不同问题。已提交成果直接进入双复核；待领取任务先交给研究者。</p><label>本轮名称<input name="title" required maxlength="200" value="新一轮证据研究"></label><div class="controller-task-options">${available.map(t=>`<label><input type="checkbox" name="task" value="${esc(t.id)}"><span><b>${esc(t.spec.title)}</b><small>${t.status==='review'?'直接复核成果':'研究后提交复核'}</small></span></label>`).join('')||'<p>先到“研究任务”创建一个有成功与停止标准的问题。</p>'}</div><div class="fields"><label>最多同时执行<input name="max_parallel" type="number" min="1" max="3" value="2" required></label><label>轮数上限<input name="max_rounds" type="number" min="1" max="5" value="2" required></label><label>Agent 调用总上限<input name="max_calls" type="number" min="3" max="40" value="12" required></label><label>单次时限（分钟）<input name="timeout" type="number" min="1" max="60" value="15" required></label><label>总时限（分钟）<input name="wall" type="number" min="1" max="720" value="60" required></label></div><p class="note">一项新研究通常至少需4次调用：研究者、两名复核者、综合裁决。预算不足以完成整轮时，系统保留任务并停止派发。</p><button class="primary">建立总控</button><button type="button" id="campaign-cancel">取消</button></form>`;
    $('campaign-cancel').onclick=()=>{$('controller-create').innerHTML='';};
    $('campaign-form').onsubmit=async e=>{e.preventDefault();const f=new FormData(e.target);try{const c=await api('/api/research/campaigns',{title:f.get('title'),task_ids:f.getAll('task'),max_parallel:Number(f.get('max_parallel')),max_rounds:Number(f.get('max_rounds')),max_calls:Number(f.get('max_calls')),job_timeout_seconds:Number(f.get('timeout'))*60,wall_seconds:Number(f.get('wall'))*60});selected=c.id;await load();}catch(err){fail(err);}};
  }
  async function load(){
    clearTimeout(timer);const rows=await api('/api/research/campaigns');
    $('controller-content').innerHTML=`<div class="section-intro"><span class="eyebrow">RESEARCH CONTROL ROOM</span><h2>让研究有人接续，也有人质疑</h2><p>总控在预算内组织独立研究、并行复核与综合裁决。每一步留下证据；接受负面结论，也保留下一步值得回答的问题。</p></div><div class="controller-flow"><span><b>01</b> 研究问题与边界</span><i>→</i><span><b>02</b> 研究者执行</span><i>→</i><span><b>03</b> 方法 / 数据双复核</span><i>→</i><span><b>04</b> 裁决与下一轮</span></div><div class="toolbar"><button class="primary" id="controller-new">组织一轮研究</button><button id="controller-refresh">刷新</button><span class="note">本地 Codex · 有界自动循环</span></div><p id="controller-message" role="status"></p><div id="controller-create"></div><div class="controller-select">${rows.map(c=>`<button data-campaign="${esc(c.id)}">${esc(c.config.title)} <span class="badge">${states[c.state.status]}</span></button>`).join('')}</div><div id="controller-detail" class="panel">尚无总控。先创建研究任务，再选择本轮范围。</div>`;
    $('controller-new').onclick=()=>createForm().catch(fail);$('controller-refresh').onclick=()=>load().catch(fail);
    document.querySelectorAll('[data-campaign]').forEach(b=>b.onclick=()=>detail(b.dataset.campaign).catch(fail));
    if(rows.length)await detail(rows.some(c=>c.id===selected)?selected:rows[0].id);
  }
  return {init:(request,openTask)=>{api=request;showTask=openTask;},load};
})();
