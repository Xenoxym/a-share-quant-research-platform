(() => {
  'use strict';
  const boot = window.WORKBENCH;
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct = v => v == null || !Number.isFinite(Number(v)) ? '—' : (Number(v)*100).toFixed(2)+'%';
  const num = v => v == null ? '—' : Number(v).toLocaleString('zh-CN', {maximumFractionDigits:2});
  const money = v => v == null ? '—' : Math.abs(v)>=1e8 ? (v/1e8).toFixed(2)+' 亿' : Math.abs(v)>=1e4 ? (v/1e4).toFixed(2)+' 万' : num(v);
  const colors = ['#176957','#6f879f','#b28c48','#5473b7','#b85261'];
  const reasonNames = {eligible:'可研究',unsupported_board:'第一版范围之外',not_ordinary_single_day:'非普通单日榜',missing_seats:'席位不完整',missing_quote:'缺少行情',st_or_unknown:'ST 或状态未知',unknown_price_limits:'限价制度未验证',missing_valuation:'缺少历史估值',missing_20_sessions:'不足连续 20 日行情',invalid_flow:'机构金额未知',below_liquidity_threshold:'低于流动性门槛',already_held:'已有持仓',position_limit:'持仓数已满',cash_or_lot_size:'资金或整手限制',prior_liquidity:'事前流动性限制',open_at_upper_limit:'开盘处于涨停价',open_at_lower_limit:'开盘处于跌停价',halt:'停牌',zero_volume:'无成交量',daily_liquidity:'当日容量限制',cash_or_gap:'资金或跳空限制',beyond_observation_window:'观察期尚未覆盖',model_fill:'按模型成交',partial_capacity_or_cash:'容量或资金下部分成交',unexplained_price_reference_change:'昨收变动缺少公司行动解释',missing_corporate_action_file:'缺少公司行动来源文件',corporate_timing_assumption:'送转到账或配股处理假设',missing_mark:'缺价，沿用估值',halt_mark:'停牌，沿用估值',delisted_unsettled:'退市权益尚未结算',observed_price_label:'固定日期价格标签',missing_endpoint_quote:'标签端点缺行情',missing_action_source:'标签缺公司行动来源'};
  Object.assign(reasonNames,{dated_st_exit_model:'采用有日期版本的 ST 退出模型',unresolved_exit_regime:'退出限价制度仍未确定'});
  const reason = v => reasonNames[v] || v || '—';
  let result = boot.result || null, page = 1, pageTotal = 0, eventRows = [], requestSequence = 0;
  let eventSequence = 0;
  function message(text) { $('message').textContent=text; $('message').hidden=!text; }
  async function api(path, options={}) {
    const response = await fetch(path,{...options,headers:{'Content-Type':'application/json','X-Workbench-Token':boot.token || '',...(options.headers||{})}});
    const data = await response.json();
    if(!response.ok) throw new Error(data.error || '请求失败');
    return data;
  }
  function view(name) {
    document.querySelectorAll('[data-view]').forEach(e=>e.hidden=e.dataset.view!==name);
    document.querySelectorAll('[data-page]').forEach(e=>e.classList.toggle('active',e.dataset.page===name));
    $('page-title').textContent={overview:'研究概览',create:'创建研究',events:'事件与交易',archive:'实验档案',contract:'数据与假设'}[name];
    if(name==='archive') loadArchive().catch(e=>message(e.message));
    if(name==='events') loadEvents().catch(e=>message(e.message));
    if(name==='contract') renderContract();
  }
  function table(columns, rows) {
    if(!rows.length) return '<p class="muted">当前没有记录。</p>';
    return '<table><thead><tr>'+columns.map(c=>'<th>'+esc(c[0])+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+columns.map(c=>'<td>'+(c[2] ? c[2](row[c[1]],row) : esc(row[c[1]] ?? '—'))+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
  }
  function explainComparisons() {
    const groups = [
      ['完整机构信号组合','全部合格上榜事件中，机构披露净买占比超过门槛的事件；按占比从高到低买入。'],
      ['全部基础事件组合','全部合格上榜事件，包含机构信号事件；按固定种子顺序买入。它仍依赖龙虎榜，不是全市场价量基线。'],
      ['配对机构组合','机构信号中能找到同日相似对照的子集；仍按机构占比排序。用于观察匹配后样本发生的变化。'],
      ['配对对照组合','同日机构占比未超过门槛、流通市值、20 日动量和成交额相近的上榜事件；按固定种子排序。'],
      ['机构组合：费用与滑点压力','沿用完整机构信号，将佣金、最低佣金、其他费用与滑点加倍，印花税不变；重新模拟资金路径。']
    ];
    return '<ol class="assumptions">'+groups.map(([name,description])=>'<li><strong>'+esc(name)+'：</strong>'+esc(description)+'</li>').join('')+'</ol><p class="muted">这是一个固定事件假设的组合对照与压力测试。样本和排序存在差异，不是五种独立策略，也不是严格逐项消融；配对差异不能解释为龙虎榜信息带来的因果收益。</p>';
  }
  $('template-comparison').innerHTML=explainComparisons();
  const eventLink=(value)=>'<button class="event-link" data-event="'+esc(value)+'">'+esc(value)+'</button>';
  function link(file) { return boot.offline ? file : '/api/runs/'+encodeURIComponent(result.run_id)+'/download?file='+encodeURIComponent(file); }
  function lineChart(id, series, valueKey, formatter=pct) {
    const element=$(id); if(!element || !series.length) return;
    const width=980,height=260,left=65,right=20,top=15,bottom=35;
    const values=series.flatMap(s=>s.rows.map(r=>r[valueKey])).filter(v=>Number.isFinite(v));
    if(!values.length) { element.textContent='暂无可画数据';return; }
    let low=Math.min(...values),high=Math.max(...values);if(low===high){low-=.01;high+=.01;}
    const pad=(high-low)*.08;low-=pad;high+=pad;
    const rows=series[0].rows,n=rows.length;
    const x=i=>left+i/Math.max(1,n-1)*(width-left-right),y=v=>top+(high-v)/(high-low)*(height-top-bottom);
    let svg='<svg class="chart" viewBox="0 0 '+width+' '+height+'" role="img" aria-label="'+esc(valueKey==='nav'?'组合净值曲线':'组合回撤曲线')+'">';
    for(let i=0;i<5;i++){const v=low+(high-low)*i/4;svg+='<line class="grid" x1="'+left+'" x2="'+(width-right)+'" y1="'+y(v)+'" y2="'+y(v)+'"/><text x="'+(left-8)+'" y="'+(y(v)+4)+'" text-anchor="end">'+esc(formatter(v))+'</text>';}
    for(const i of [...new Set([0,Math.floor(n/3),Math.floor(2*n/3),n-1])]) svg+='<text x="'+x(i)+'" y="'+(height-7)+'" text-anchor="'+(i===0?'start':i===n-1?'end':'middle')+'">'+esc(rows[i].date)+'</text>';
    series.forEach((s,j)=>{const path=s.rows.map((r,i)=>(i?'L':'M')+x(i).toFixed(2)+' '+y(r[valueKey]).toFixed(2)).join(' ');svg+='<path d="'+path+'" fill="none" stroke="'+colors[j]+'" stroke-width="1.8"/>';});
    svg+='</svg><div class="chart-tip">移动查看日期与数值；点击日期查看机构组合当天持仓与订单。</div>';
    element.innerHTML=svg;
    const chart=element.querySelector('svg'),tip=element.querySelector('.chart-tip');
    const indexAt=e=>{const rect=chart.getBoundingClientRect();return Math.max(0,Math.min(n-1,Math.round(((e.clientX-rect.left)/rect.width*width-left)/(width-left-right)*(n-1))));};
    chart.addEventListener('pointermove',e=>{const i=indexAt(e);tip.textContent=rows[i].date+' · '+series.map(s=>s.name+' '+formatter(s.rows[i][valueKey])).join(' / ');});
    chart.addEventListener('click',e=>loadDay(rows[indexAt(e)].date).catch(error=>message(error.message)));
  }
  function metric(label,value,note,negative=false){return '<div class="metric"><span>'+esc(label)+'</span><span class="value '+(negative?'negative':'')+'">'+esc(value)+'</span><small>'+esc(note)+'</small></div>';}
  function renderResult() {
    if(!result) return;
    $('empty').hidden=true;$('overview-result').hidden=false;
    const p=result.portfolios[0],m=p.metrics,r=result.research,ci=r.paired_interval;
    const warningCount=Object.values(m.warnings).reduce((a,b)=>a+b,0);
    const issueKeys=['unexplained_price_reference_change','missing_corporate_action_file','missing_mark','delisted_unsettled','unresolved_exit_regime'];
    const severe=issueKeys.some(k=>m.warnings[k]>0);
    const funnel=[['全部龙虎榜事件',r.funnel.all_events],['基础候选事件',r.funnel.base_eligible],['机构信号',r.funnel.institutional_signals],['同日配对',r.funnel.matched_pairs],['完整机构组合买入',m.buy_fills]];
    const portfolioTable=table([['组合','label'],['累计收益','total_return',pct],['最大回撤','max_drawdown',pct],['平均资金暴露','average_exposure',pct],['买入笔数','buy_fills',num],['已结束交易','closed_trades',num],['期末持仓','open_positions',num],['累计费用','cost',money],['数据疑点记录','data_issues',num]],result.portfolios.map(p=>({...p.metrics,data_issues:issueKeys.reduce((sum,k)=>sum+(p.metrics.warnings[k]||0),0)})));
    $('overview-result').innerHTML='<div class="section-top"><div><h2>'+esc(result.name)+'</h2><p class="muted">'+esc(result.data_start)+' — '+esc(result.data_end)+' · 数据快照 '+esc(result.snapshot_id.slice(0,12))+'</p></div><span class="badge '+(severe?'warning':'')+'">'+(severe?'存在数据估值疑点':'历史日频模拟')+'</span></div>'+
      '<details class="panel" open><summary>五组比较说明 · 固定事件模板，尚无独立价量基线</summary>'+explainComparisons()+'</details>'+
      '<div class="notice">日线成交属于模型假设；价格标签与资金账户分开计算。'+(severe?'机构组合存在未解释的估值或数据问题，请先查看下方质量记录，不能将本次收益视为已核实表现。':'历史区间已经用于研究，分段结果不代表全新的样本外验证。')+(m.unresolved_market_value>0?' 期末未解决估值 '+money(m.unresolved_market_value)+' 元；将这些权益估值设为零的压力情景累计收益为 '+pct(m.return_zero_unresolved)+'。这不代表实际退市结算价格。':'')+'</div>'+
      '<div class="metrics">'+metric('机构组合累计收益',pct(m.total_return),'含费用与模型公司行动',m.total_return<0)+metric('最大回撤',pct(m.max_drawdown),'以每日收盘权益计算',true)+metric('实际模拟买入',num(m.buy_fills),'来自 '+num(r.funnel.institutional_signals)+' 个机构信号')+metric('期末持仓 / 应收分红',num(m.open_positions)+' / '+money(m.receivable),'保留未平仓与未到账权益')+'</div>'+
      '<div class="panel"><div class="section-top"><div><h2>资金约束下的组合轨迹</h2><p class="muted">基础组合和配对组合使用相同资金与交易模型；信号集与排序方式有差异。</p></div></div><div class="legend">'+result.portfolios.slice(0,4).map((p,i)=>'<span><i style="background:'+colors[i]+'"></i>'+esc(p.metrics.label)+'</span>').join('')+'</div><div id="nav-chart"></div><div id="day-detail"></div></div>'+
      '<div class="panel"><h2>机构组合回撤</h2><div id="drawdown-chart"></div></div>'+
      '<div class="panel"><h2>同一套规则下的比较</h2><div class="table-wrap">'+portfolioTable+'</div><p class="muted">压力情景将佣金、最低佣金、其他费用和滑点加倍，卖出印花税保持原日历规则。期末持仓按可得价格估值；未平仓收益不计入已结束交易胜率。</p></div>'+
      '<div class="two-cols"><div class="panel"><h2>样本与交易漏斗</h2>'+funnel.map(([k,v])=>'<div class="funnel-row"><span>'+esc(k)+'</span><div class="funnel-bar"><span style="width:'+Math.max(1,v/r.funnel.all_events*100)+'%"></span></div><strong>'+num(v)+'</strong></div>').join('')+'<p class="muted">配对用于增量比较；完整机构组合保留未匹配信号。未成交也留在订单记录。</p></div><div class="panel"><h2>配对后，价格标签相差多少</h2><div class="metrics" style="grid-template-columns:1fr"><div class="metric"><span>机构组 − 对照组</span><strong class="value">'+pct(ci.mean)+'</strong><small>95% 时间块区间：'+pct(ci.low)+' 至 '+pct(ci.high)+'</small></div></div><p class="muted">'+num(ci.pairs)+' 对有效标签 / '+num(ci.event_days)+' 个事件日。'+(ci.low==null?'有效事件日不足 30 天，不报告区间。':'')+'价格标签未施加成交约束和费用；该差异不等于可实现超额收益。</p></div></div>'+
      '<div class="panel"><h2>收益分布与年度稳定性</h2><div class="table-wrap">'+table([['样本','cohort'],['事件数','events',num],['有效标签','observed',num],['均值','mean',pct],['中位数','median',pct],['5% 分位','p05',pct],['95% 分位','p95',pct]],r.distribution)+'</div><h3 style="margin-top:24px">年度配对差异 · 已剔除跨年标签</h3><div class="table-wrap">'+table([['年份','year'],['有效配对','valid_pairs',num],['机构组均值','treated_mean',pct],['对照组均值','control_mean',pct],['差异','difference',pct]],r.yearly)+'</div><details style="margin-top:18px"><summary>查看配对平衡及研究方法</summary><p class="muted">'+esc(r.interpretation)+'</p>'+table([['特征','feature'],['阶段','stage'],['标准化均值差','standardized_difference',num]],r.balance)+'</details></div>'+
      '<div class="panel"><h2>未成交与质量记录</h2><div class="two-cols"><div><h3>机构组合买入未成交原因</h3>'+table([['原因','reason'],['记录数','count',num]],Object.entries(m.rejection_reasons).map(([k,v])=>({reason:reason(k),count:v})))+'</div><div><h3>持仓质量与模型假设 · '+num(warningCount)+' 条</h3>'+table([['情况','reason'],['记录数','count',num]],Object.entries(m.warnings).map(([k,v])=>({reason:reason(k),count:v})))+'<p class="muted">同一持仓跨日未解决会产生多条记录；账户对账通过不代表来源完整或成交确定。</p></div></div><div class="download-row"><a class="button" href="'+link('institutional/orders.csv')+'">订单 CSV</a><a class="button" href="'+link('institutional/ledger.csv')+'">账户流水 CSV</a><a class="button" href="'+link('institutional/positions.csv')+'">每日持仓 CSV</a><a class="button" href="'+link('institutional/warnings.csv')+'">质量明细 CSV</a><a class="button" href="'+link('manifest.json')+'">实验清单</a></div></div>';
    lineChart('nav-chart',result.portfolios.slice(0,4).map(p=>({name:p.metrics.label,rows:p.equity})),'nav',v=>Number(v).toFixed(3));
    lineChart('drawdown-chart',[{name:'机构组合回撤',rows:p.equity}],'drawdown');
    renderContract();
  }
  async function loadDay(day){
    if(boot.offline){$('day-detail').innerHTML='<p class="notice">'+esc(day)+'：逐日持仓和订单见本报告底部 CSV；工作台支持在线联动查询。</p>';return;}
    const data=await api('/api/runs/'+result.run_id+'/day?date='+encodeURIComponent(day));
    $('day-detail').innerHTML='<h3>'+esc(day)+' · 机构组合持仓与订单</h3><div class="table-wrap">'+table([['来源事件','event_id',eventLink],['持股数','shares',num],['估值','mark',num],['持仓市值','market_value',money],['状态','stale_reason',v=>esc(reason(v))]],data.positions)+'</div><div class="table-wrap">'+table([['来源事件','event_id',eventLink],['方向','side'],['请求股数','requested_shares',num],['成交股数','filled_shares',num],['原因','reason',v=>esc(reason(v))]],data.orders)+'</div>';
  }
  function renderContract(){
    const assumptions=result?.assumptions || boot.assumptions || [],features=result?.features || boot.features || [];
    $('contract-content').innerHTML='<h2>数据、时间与执行约定</h2><p>每次实验保存使用的数据快照、策略卡片、工作台源码、环境版本和文件校验值。</p><ol class="assumptions">'+assumptions.map(a=>'<li>'+esc(a)+'</li>').join('')+'</ol><h3>本次使用的特征</h3><div class="table-wrap">'+table([['名称','label'],['单位','unit'],['可用时间','known_at'],['边界','boundary']],features)+'</div>'+(result?'<h3 style="margin-top:25px">追溯标识</h3><p class="mono">实验 '+esc(result.run_id)+'<br>数据 '+esc(result.snapshot_id)+'<br>代码 '+esc(result.code_hash)+'</p>':'')+'<p class="muted">股票简称仅用于展示，可能是供应商当前名称。历史 ST 资格由当日状态确定。估值历史采用当前已发布版本，不假设已恢复历次修订。</p>';
  }
  async function loadEvents(){
    if(!result){$('events-table').innerHTML='<p>请先完成或打开一个实验。</p>';return;}
    const sequence=++requestSequence,query=$('event-query').value.trim(),kind=$('event-kind').value;
    let data;
    if(boot.offline){let rows=result.offline_events || [];if(query)rows=rows.filter(r=>[r.stock_code,r.stock_name,r.trade_date].some(v=>String(v).includes(query)));if(kind==='matched')rows=rows.filter(r=>r.matched_treatment);if(kind==='control')rows=rows.filter(r=>r.matched_control);if(kind==='rejected')rows=[];data={rows:rows.slice((page-1)*50,page*50),total:rows.length};}
    else data=await api('/api/runs/'+result.run_id+'/events?'+new URLSearchParams({page,query,kind}));
    if(sequence!==requestSequence)return;
    eventRows=data.rows;pageTotal=data.total;
    $('event-scope').textContent=boot.offline?'离线预览：最近 100 个机构信号，前 12 个内置完整追溯。全量浏览请打开工作台。':'全部样本可搜索；机构信号包括未成交、未匹配及观察期未结束的事件。';
    $('events-download').href=link('events.csv');
    $('events-table').innerHTML=table([['事件','event_id',eventLink],['简称（展示）','stock_name'],['机构披露净额','inst_disclosed_net',money],['占成交额','inst_ratio',pct],['流通市值','float_value',money],['20 日动量','momentum_20',pct],['价格标签','label_return',pct],['资格','eligibility_reason',v=>esc(reason(v))]],eventRows);
    $('event-page').textContent='第 '+page+' 页 / 共 '+num(pageTotal)+' 个事件';$('previous-events').disabled=page===1;$('next-events').disabled=page*50>=pageTotal;
  }
  async function loadEvent(eid){
    view('events');const sequence=++eventSequence;
    $('event-detail').innerHTML='<div class="panel">正在读取事件证据…</div>';
    let data;
    if(boot.offline){data=boot.evidence[eid];if(!data){$('event-detail').innerHTML='<div class="panel">这个事件未内置离线明细。请在工作台打开该实验；完整特征仍可在事件 CSV 中查看。</div>';return;}}
    else data=await api('/api/runs/'+result.run_id+'/event?event_id='+encodeURIComponent(eid));
    if(sequence!==eventSequence)return;
    const e=data.event;
    $('event-detail').innerHTML='<div class="panel"><h2>'+esc(e.stock_code)+' · '+esc(e.trade_date)+'</h2><p class="muted">'+esc(e.summary_selected_reason)+'</p><div class="detail-grid"><div><span>信息可用时间假设：</span>'+esc(e.known_at_assumed)+'</div><div><span>机构披露净买额：</span>'+money(e.inst_disclosed_net)+'</div><div><span>计划入场：</span>'+esc(e.entry_date || '超出日历')+'</div><div><span>价格标签：</span>'+pct(e.label_return)+' · '+esc(reason(e.label_status))+'</div></div><h3 style="margin-top:20px">事件前后价格 · 原始未复权</h3><div id="event-chart"></div><h3>所选报告的原始席位</h3><div class="table-wrap">'+table([['方向','direction'],['排名','rank'],['席位','broker_name'],['买入金额','buy_amount',money],['卖出金额','sell_amount',money],['来源报告','report_id']],data.brokers)+'</div><h3 style="margin-top:24px">该事件在各组合中的订单</h3><div class="table-wrap">'+table([['组合','portfolio'],['日期','date'],['方向','side'],['请求','requested_shares',num],['成交','filled_shares',num],['价格','price',num],['费用','total',num],['原因','reason',v=>esc(reason(v))]],data.orders)+'</div><h3 style="margin-top:24px">配对及公司行动</h3><div class="table-wrap">'+table([['机构事件','treated_event_id',eventLink],['对照事件','control_event_id',eventLink],['距离','distance',num],['价格标签差','difference',pct]],data.pairs)+table([['日期','trade_date'],['每股分红','dividend',num],['每股送转','allotted_ps',num],['每股配股','rationed_ps',num]],data.actions)+'</div><p class="mono" style="margin-top:20px">'+esc(JSON.stringify(data.source_reference))+'</p></div>';
    candleChart(data.bars,e.trade_date);
    $('event-detail').scrollIntoView({behavior:'smooth',block:'start'});
  }
  function candleChart(bars,eventDate){
    if(!bars.length){$('event-chart').textContent='当前范围无对应行情。';return;}
    const w=980,h=240,l=60,b=30,t=15,lo=Math.min(...bars.map(r=>r.low)),hi=Math.max(...bars.map(r=>r.high));
    const x=i=>l+(i+.5)*(w-l-10)/bars.length,y=v=>t+(hi-v)/Math.max(.01,hi-lo)*(h-t-b),bw=Math.min(12,(w-l-10)/bars.length*.6);
    let s='<svg class="chart" viewBox="0 0 '+w+' '+h+'" role="img" aria-label="事件前后日K线">';
    for(let i=0;i<4;i++){let v=lo+(hi-lo)*i/3;s+='<line class="grid" x1="'+l+'" x2="970" y1="'+y(v)+'" y2="'+y(v)+'"/><text x="50" y="'+(y(v)+4)+'" text-anchor="end">'+v.toFixed(2)+'</text>';}
    bars.forEach((r,i)=>{const c=r.close>=r.open?'#b85261':'#218574';if(r.trade_date===eventDate)s+='<rect x="'+(x(i)-bw)+'" y="0" width="'+(bw*2)+'" height="210" fill="#e6edf5"/>';s+='<g><title>'+esc(r.trade_date+' 开 '+r.open+' 高 '+r.high+' 低 '+r.low+' 收 '+r.close)+'</title><line x1="'+x(i)+'" x2="'+x(i)+'" y1="'+y(r.high)+'" y2="'+y(r.low)+'" stroke="'+c+'"/><rect x="'+(x(i)-bw/2)+'" y="'+y(Math.max(r.open,r.close))+'" width="'+bw+'" height="'+Math.max(1,Math.abs(y(r.open)-y(r.close)))+'" fill="'+c+'"/></g>';});
    s+='<text x="60" y="233">'+esc(bars[0].trade_date)+'</text><text x="970" y="233" text-anchor="end">'+esc(bars[bars.length-1].trade_date)+'</text></svg><p class="muted">阴影为事件日；悬停蜡烛查看 OHLC。除权可能造成未复权价格跳变。</p>';$('event-chart').innerHTML=s;
  }
  async function loadArchive(){
    if(boot.offline){$('archive-list').innerHTML='<div class="panel">当前是单次实验的离线报告。实验档案和重新运行功能位于本地工作台。</div>';return;}
    const runs=await api('/api/runs');
    $('archive-list').innerHTML=runs.length?runs.map(r=>'<div class="panel archive-card"><div><span class="badge '+(r.status==='failed'?'warning':'')+'">'+esc({completed:'已完成',failed:'失败',running:'运行中',cancelled:'已中断'}[r.status]||r.status)+'</span><h3 style="margin:12px 0 3px">'+esc(r.name||r.run_id)+'</h3><p class="mono">'+esc(r.run_id)+'</p><p class="muted">'+(r.metrics?'收益 '+pct(r.metrics.total_return)+' / 回撤 '+pct(r.metrics.max_drawdown)+' / 持有 '+r.card.holding_days+' 日 / 滑点 '+r.card.slippage_bps+' 基点':'')+esc(r.error||'')+'</p></div>'+(r.status==='completed'?'<button data-run="'+esc(r.run_id)+'">打开实验</button>':'')+'</div>').join(''):'<div class="panel">尚无实验记录。</div>';
  }
  async function openRun(id){result=await api('/api/runs/'+encodeURIComponent(id)+'/result');page=1;$('event-detail').innerHTML='';renderResult();view('overview');message('');}
  const scales={position_fraction:100,min_inst_ratio:100,max_participation:100,commission_rate:10000,other_fee_rate:10000,dividend_tax_reserve:100,min_avg_amount:.0001};
  function fillForm(){const card=result?.card||boot.defaults;if(!card)return;for(const [key,value]of Object.entries(card)){const field=$('strategy-form').elements.namedItem(key);if(field)field.value=typeof value==='number'?Number((value*(scales[key]||1)).toPrecision(12)):value;}}
  function newStudy(){if(boot.offline){message('离线报告不能启动计算。请双击“研究工作台.bat”创建实验。');return;}fillForm();view('create');}
  async function pollJob(){
    const job=await api('/api/job');$('job-log').textContent=(job.messages||[]).slice(-7).join('\n');
    if(job.status==='running'){setTimeout(()=>pollJob().catch(e=>message(e.message)),1500);return;}
    $('run-button').disabled=false;$('job').hidden=true;
    if(job.status==='completed')await openRun(job.run_id);else if(job.error)message(job.error);
  }
  $('strategy-form').addEventListener('submit',async e=>{e.preventDefault();try{const card={};for(const [key,value]of new FormData(e.target)){card[key]=['name','start_date','end_date'].includes(key)?value:Number(value)/(scales[key]||1);}await api('/api/run',{method:'POST',body:JSON.stringify(card)});$('run-button').disabled=true;$('job').hidden=false;message('');view('overview');await pollJob();}catch(error){message(error.message);$('run-button').disabled=false;}});
  document.addEventListener('click',e=>{const nav=e.target.closest('[data-page]'),ev=e.target.closest('[data-event]'),run=e.target.closest('[data-run]');if(nav){if(nav.dataset.page==='create')newStudy();else view(nav.dataset.page);}if(ev)loadEvent(ev.dataset.event).catch(error=>message(error.message));if(run)openRun(run.dataset.run).catch(error=>message(error.message));});
  $('new-study').onclick=newStudy;$('empty-create').onclick=newStudy;
  $('event-search').onclick=()=>{page=1;loadEvents().catch(e=>message(e.message));};$('event-query').addEventListener('keydown',e=>{if(e.key==='Enter')$('event-search').click();});
  $('event-kind').onchange=()=>{page=1;loadEvents().catch(e=>message(e.message));};$('previous-events').onclick=()=>{page--;loadEvents().catch(e=>message(e.message));};$('next-events').onclick=()=>{page++;loadEvents().catch(e=>message(e.message));};
  $('connection').textContent=boot.offline?'离线报告 · 可分享':'本地工作台 · 127.0.0.1';
  if(result)renderResult();else if(boot.latest)openRun(boot.latest).catch(e=>message(e.message));
  if(!boot.offline)api('/api/job').then(j=>{if(j.status==='running'){$('job').hidden=false;$('run-button').disabled=true;pollJob().catch(e=>message(e.message));}}).catch(e=>message(e.message));
})();
