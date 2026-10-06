window.mlWorkbench=(()=>{
  'use strict';
  let api,activate,started,openAccount,selected=null;
  const $=id=>document.getElementById(id),esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct=v=>v==null?'—':(100*v).toFixed(3)+'%',num=v=>v==null?'—':Number(v).toFixed(4);
  const names={ridge:'岭回归',hist_gbdt:'受限提升树',small_cap:'小市值',reversal_5:'五日反转',fixed_random:'固定随机排序'};
  const scopes={train:'训练 · 同样本',valid:'验证 · 后续时间',test:'测试 · 后续时间'};
  const table=(cols,rows)=>`<div class="table-wrap"><table><thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
  function fail(e){$('ml-message').textContent=e.message||String(e);}
  async function load(){
    const [runs,plan,complete]=await Promise.all([api('/api/ml'),api('/api/ml/plan'),api('/api/ml/complete-plan')]);
    $('ml-content').innerHTML=`<div class="section-intro"><span class="eyebrow">MACHINE LEARNING RESEARCH</span><h2>先确认模型学到了选股信息</h2><p>比较固定模型在后续时间的股票排序与分组收益。每次保留设计、模型和逐股预测。</p></div>
      <div class="panel"><h3>日线价量基准案例</h3><p>每周采样，预测未来五个交易日的相对收益。训练 2020—2022，验证 2023，测试 2024—2026-09-24。</p><p>岭回归和受限提升树，比较同池小市值、五日反转与固定随机排序。模型仅用价量输入；股本用于共同候选资格和小市值对照。</p><p class="note">这些历史已经被查看。此处验证模型的时间迁移，不代表真正未见历史或可成交账户收益。</p><div class="toolbar"><button id="ml-run" class="primary">登记并运行基准案例</button><button id="ml-refresh">刷新结果</button></div><details><summary>固定设计与模型设置</summary><pre>${esc(JSON.stringify(plan.spec,null,2))}</pre></details><p id="ml-message" role="status"></p></div>
      <div class="panel"><h3>从预测到实际资金账户</h3><p>固定19种策略，各跑配置费用与零交易费用。包含原模型、周期对齐、四组特征删除、加入市值、规模组内选股、标签打乱、逐年重训和验证期选参数。</p><p>每周收盘选前100只，下个交易日开盘开始调仓；记录现金、持仓、委托、成交及净值。</p><button id="ml-complete" ${complete.available?'':'disabled'}>登记并运行完整研究</button><p class="note">${complete.available?'复用已保存原模型；有限矩阵全部展示，测试期不用于参数选择。':'先通过CLI指定一个已核验原基准；当前固定原案例尚不存在。'}</p></div>
      <div class="panel"><h3>研究记录</h3>${runs.length?runs.map(r=>`<p><button class="text-button" data-ml-run="${esc(r.run_id)}">${esc(r.name||r.run_id)} ↗</button> · ${esc({completed:'已完成',running:'运行中',failed:'失败'}[r.status]||r.status)} ${r.error?esc(r.error):''}</p>`).join(''):'<p>尚无机器学习案例。</p>'}</div><div id="ml-result"></div>`;
    $('ml-refresh').onclick=()=>load().catch(fail);
    $('ml-run').onclick=async()=>{try{$('ml-run').disabled=true;const r=await api('/api/ml/benchmark',{});$('ml-message').textContent='已启动，研究任务 '+r.task_id;started();}catch(e){$('ml-run').disabled=false;fail(e);}};
    $('ml-complete').onclick=async()=>{try{$('ml-complete').disabled=true;const r=await api('/api/ml/complete',{});$('ml-message').textContent='已启动完整研究，任务 '+r.task_id;started();}catch(e){$('ml-complete').disabled=false;fail(e);}};
    document.querySelectorAll('[data-ml-run]').forEach(b=>b.onclick=()=>detail(b.dataset.mlRun).catch(fail));
    if(selected&&runs.some(r=>r.run_id===selected&&r.status==='completed'))await detail(selected);
  }
  async function detail(id){
    selected=id;const r=await api('/api/ml/'+id),available=[...new Set(r.summaries.map(s=>s.scope))];
    if(r.study==='universe')return universeDetail(r);
    if(r.study==='complete')return studyDetail(r);
    $('ml-result').innerHTML=`<div class="panel"><h2>${esc(r.name)}</h2><div class="toolbar"><label>评价时期<select id="ml-scope">${available.map(s=>`<option value="${esc(s)}" ${s==='test'?'selected':''}>${esc(scopes[s]||s.replace('test_','测试 '))}</option>`).join('')}</select></label></div><div id="ml-scores"></div><h3>预测十分组 · 最高评分在第 1 组</h3><div id="ml-quantiles"></div><p class="note">Top相对收益是每个采样日入选股票的五日平均收益减候选池平均收益，再按日期取平均；没有复合成年化或账户净值。</p><details><summary>候选样本与标签覆盖</summary>${table(['区间','候选行','采样日期','有标签行'],r.counts.map(c=>[esc(c.split),esc(c.rows),esc(c.dates),esc(c.observed_labels)]))}<p>资格先按当时信息确定；缺失未来标签仍保留在排名中。小市值口径是已公告股本估算。</p></details><details><summary>研究限制</summary><ul>${r.limitations.map(v=>`<li>${esc(v)}</li>`).join('')}</ul></details><div class="toolbar"><a href="/api/ml/${esc(id)}/download?file=REPORT.md">下载报告</a><a href="/api/ml/${esc(id)}/download?file=predictions.parquet">下载逐股预测</a><a href="/api/ml/${esc(id)}/download?file=dataset.json">特征和标签定义</a></div><p class="note">运行 ${esc(id)} · 快照 ${esc(r.snapshot_id)}</p></div>`;
    const render=()=>{const scope=$('ml-scope').value,rows=r.summaries.filter(s=>s.scope===scope);
      $('ml-scores').innerHTML=table(['模型/对照','RankIC','95%块区间','Top相对池/期','相对小市值/期','入选标签覆盖'],rows.map(s=>[esc(names[s.model]||s.model),num(s.rank_ic),s.rank_ic_interval.map(num).join(' 至 '),pct(s.top_excess),pct(s.top_excess_vs_size),pct(s.top_label_coverage)]));
      const split=scope.startsWith('test')?'test':scope;
      $('ml-quantiles').innerHTML=table(['模型',...Array.from({length:10},(_,i)=>'第'+(i+1)+'组')],Object.keys(names).map(m=>[esc(names[m]),...Array.from({length:10},(_,i)=>pct(r.quantiles.find(q=>q.split===split&&q.model===m&&q.quantile===i+1)?.excess))]));
      if(scope.startsWith('test_'))$('ml-quantiles').innerHTML+='<p class="note">十分组表展示全部测试期；上方指标展示所选年份。</p>';
    };$('ml-scope').onchange=render;render();
  }
  function studyDetail(r){
    const chosen=a=>a.portfolios.find(p=>p.metrics.scenario===$('ml-cost').value);
    $('ml-result').innerHTML=`<div class="panel"><h2>模型已经进入真实账户</h2><p>初始100万元，每周选股，目标总投入98%、前100只等权。测试2024—2026-09-24，末日保留持仓估值。</p><p>${r.new_fits}次新拟合，复用2个原模型；${r.accounts.length}种策略、${r.accounts.length*2}个账户情景全部保留。</p><label>费用情景 <select id="ml-cost"><option value="configured">配置费用与滑点</option><option value="zero_transaction_cost">零交易费用与滑点</option></select></label><div id="ml-accounts"></div><p class="note">模型评分先从共同完整名单选股，随后由项目账户引擎处理买卖。年化与回撤来自真实资金账户；RankIC仍是学习诊断。</p><h3>验证期选择参数的全部尝试</h3>${table(['模型','参数','2023验证RankIC','选中'],r.tuning.map(t=>[esc(t.kind),esc(JSON.stringify(t.parameters)),num(t.valid_mean_rank_ic),t.selected?'是':'否']))}<p class="note">只按2023验证日期平均RankIC选参数，随后用截至2023的成熟标签拟合；测试不参与选择。年度重训使用截至上年末的成熟标签，保持默认参数。</p><details><summary>数据覆盖与研究限制</summary>${table(['区间','候选行','周数','开盘目标已知'],r.counts.map(c=>[esc(c.split),c.rows,c.dates,c.observed_labels]))}<ul>${r.limitations.map(v=>`<li>${esc(v)}</li>`).join('')}</ul></details><div class="toolbar"><a href="/api/ml/${esc(r.run_id)}/download?file=REPORT.md">完整人话报告</a><a href="/api/ml/${esc(r.run_id)}/download?file=account_index.json">全部账户结果</a><a href="/api/ml/${esc(r.run_id)}/download?file=fits.json">模型拟合账本</a><a href="/api/ml/${esc(r.run_id)}/download?file=predictions.parquet">全候选预测</a></div></div>`;
    const render=()=>{
      $('ml-accounts').innerHTML=table(['策略','年化','累计收益','最大回撤','2024','2025','2026截至9/24','学习RankIC','账户'],r.accounts.map(a=>{
        const p=chosen(a),m=p.metrics,y=Object.fromEntries(p.annual.map(v=>[v.year,v.return_value])),s=r.summaries.find(v=>v.model===a.case&&v.scope==='test');
        return [esc(a.case==='fixed_random'?'固定随机顺序（seed=11，Top100）':a.name),pct(m.annualized_return),pct(m.total_return),pct(m.max_drawdown),pct(y['2024']),pct(y['2025']),pct(y['2026']),num(s?.rank_ic),`<button class="text-button" data-ml-account="${esc(a.run_id)}">查看持仓与买卖 ↗</button>`];
      }));
      document.querySelectorAll('[data-ml-account]').forEach(b=>b.onclick=()=>openAccount(b.dataset.mlAccount).catch(fail));
    };$('ml-cost').onchange=render;render();
  }
  async function universeDetail(r){
    const context=await api('/api/research/tasks/'+r.research_context.task_id),attempt=context.task.experiments.find(e=>e.id===r.research_context.experiment_id);
    const statusNote=attempt?.status==='failed'?'<p class="note">原登记执行在最后核查接口报错，状态为失败。已生成结果另经修正核查器事后复核；证据导入与原失败记录在研究任务中，不能称成功登记执行。</p>':'';
    const pools={qualified:'原主板资格',no_financial:'主板取消股本资格',mainboard_available:'非ST主板有价有量',local_A_available:'全部本地有价有量A股（含ST/新股）'};
    const models={ridge:'岭回归',tree:'提升树',fixed_random:'固定随机顺序（seed=11）',weekly_random:'每周重抽随机100（seed=11）'};
    const label=id=>{const [pool,kind]=id.split('__');return (pools[pool]||pool)+' · '+(models[kind]||kind);};
    $('ml-result').innerHTML=`<div class="panel"><h2>股票范围对照：${r.new_fits}次固定参数拟合</h2>${statusNote}<p>学习目标为下一周开盘到再下一周开盘；训练2020—2022，验证2023，历史测试2024—2026-09-24。每个范围重新训练价量模型，不搜索参数。</p><p class="note">本地覆盖主板、创业板、科创板，没有北交所。其他板块只检查学习指标；${r.accounts.length}个主板策略、${r.accounts.length*2}个账户情景使用原主板会计。不能称完整全A股账户。</p>${table(['范围','历史测试平均每周候选','历史测试周数','短历史行','ST行'],r.counts.filter(c=>c.split==='test').map(c=>[esc(pools[c.pool]),num(c.mean_candidates),c.dates,c.short_history_rows,c.st_rows]))}<label>评价时期 <select id="ml-universe-scope"><option value="test">历史测试</option><option value="valid">验证</option><option value="train">训练</option></select></label><div id="ml-universe-scores"></div><p class="note">Top100相对收益以各自池均值为参照，池不同不能直接当成同基准增量。下面另报共同原主板名单的结果。</p><h3>只在共同原主板名单上比较</h3>${table(['训练范围及模型','RankIC','Top100相对共同池/周'],r.common_summaries.map(s=>[esc(label(s.model)),num(s.rank_ic),pct(s.top_excess)]))}<h3>主板账户</h3><label>费用 <select id="ml-universe-cost"><option value="configured">配置费用</option><option value="zero_transaction_cost">零交易费用</option></select></label><div id="ml-universe-accounts"></div><details><summary>范围与研究限制</summary><ul>${r.limitations.map(s=>`<li>${esc(s)}</li>`).join('')}</ul></details><a href="/api/ml/${esc(r.run_id)}/download?file=result.json">下载全部结果</a> · <a href="/api/ml/${esc(r.run_id)}/download?file=study_plan.json">查看固定研究设计</a></div>`;
    const renderScores=()=>{$('ml-universe-scores').innerHTML=table(['范围及模型','RankIC','95%块区间','Top100相对各自池/周','入选标签覆盖'],r.summaries.filter(s=>s.scope===$('ml-universe-scope').value).map(s=>[esc(label(s.model)),num(s.rank_ic),s.rank_ic_interval.map(num).join(' 至 '),pct(s.top_excess),pct(s.top_label_coverage)]));};
    const renderAccounts=()=>{$('ml-universe-accounts').innerHTML=table(['范围及模型','年化','最大回撤','平均股票仓位','账户'],r.accounts.map(a=>{const m=a.portfolios.find(p=>p.metrics.scenario===$('ml-universe-cost').value).metrics;return [esc(label(a.case)),pct(m.annualized_return),pct(m.max_drawdown),pct(m.average_exposure),`<button class="text-button" data-ml-account="${esc(a.run_id)}">查看持仓与买卖 ↗</button>`];}));document.querySelectorAll('[data-ml-account]').forEach(b=>b.onclick=()=>openAccount(b.dataset.mlAccount).catch(fail));};
    $('ml-universe-scope').onchange=renderScores;$('ml-universe-cost').onchange=renderAccounts;renderScores();renderAccounts();
  }
  function experiment(e){return `<article class="task-experiment"><b>${esc(e.proposal.spec.name)}</b><p>${esc(e.proposal.hypothesis)}</p><p>${esc(e.status)} · ${e.proposal.spec.study==='complete'?'机器学习账户及完整研究':e.proposal.spec.study==='universe'?'机器学习股票范围对照':'机器学习信号分析'}</p>${e.result?`<p class="note">${esc(e.result.verification_scope)}</p><button class="text-button" data-task-ml="${esc(e.run_id)}">查看模型基准 ↗</button>`:`<p>${esc(e.error||'固定配置与代码已登记')}</p>`}<details><summary>学习设计</summary><pre>${esc(JSON.stringify(e.proposal.spec,null,2))}</pre></details></article>`;}
  async function open(id){selected=id;activate();}
  return {init(a,p,s,o){api=a;activate=p;started=s;openAccount=o;},load,open,experiment};
})();
