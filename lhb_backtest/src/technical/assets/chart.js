// Common-date inspection, keyboard support, independent curve controls and range brush.
window.researchChart = function(id, series, {format=String,onDate=null,onRange=null,height=null,yMax=null,yMin=null}={}) {
  const host=document.getElementById(id);if(!host)return;host._chartObserver?.disconnect();
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const visible=new Set(series.filter(s=>s.visible!==false).map(s=>s.name));
  const days=[...new Set(series.flatMap(s=>s.rows.map(r=>r.date)))].sort();
  if(!days.length){host.textContent='暂无数据';return;}
  const maps=series.map(s=>new Map(s.rows.map(r=>[r.date,r.value])));let focus=days.length-1;
  function draw(){
    const active=series.filter(s=>visible.has(s.name)),values=active.flatMap(s=>s.rows.map(r=>r.value)).filter(Number.isFinite);
    let lo=values.length?Math.min(...values):0,hi=values.length?Math.max(...values):1;
    if(hi===lo){hi+=.01;lo-=.01;}const pad=(hi-lo)*.08;lo-=pad;hi+=pad;
    if(yMax!==null){hi=yMax;if(lo>=hi)lo=hi-.01;}
    if(yMin!==null){lo=yMin;if(hi<=lo)hi=lo+.01;}
    const w=Math.max(280,host.clientWidth||1100),h=height||Math.max(230,Math.min(420,w*.3)),left=w<500?62:78,right=w-20,top=18,bottom=h-40;
    const x=i=>left+i/Math.max(1,days.length-1)*(right-left),y=v=>bottom-(v-lo)/(hi-lo)*(bottom-top),indices=new Map(days.map((d,i)=>[d,i]));
    host.innerHTML=`<div class="chart-tools">${series.map(s=>`<button class="curve-toggle ${visible.has(s.name)?'on':''}" aria-pressed="${visible.has(s.name)}" data-curve="${esc(s.name)}"><i style="background:${s.color}"></i>${esc(s.name)}</button>`).join('')}<small>悬停看数值${onRange?' · 拖动选区间':''}${onDate?' · 单击查当日':''}</small></div><div class="plot"><svg class="chart" viewBox="0 0 ${w} ${h}" height="${h}" tabindex="0" role="img" aria-label="可交互时间序列，左右键查看日期，回车检查当日"><g class="grid">${Array.from({length:5},(_,i)=>{const v=lo+(hi-lo)*i/4;return `<line x1="${left}" x2="${right}" y1="${y(v)}" y2="${y(v)}"/><text x="${left-12}" y="${y(v)+4}" text-anchor="end">${esc(format(v))}</text>`;}).join('')}</g>${active.map(s=>{let connected=false;return `<path class="series-line" d="${s.rows.map(r=>{if(!Number.isFinite(r.value)){connected=false;return '';}const part=(connected?'L':'M')+x(indices.get(r.date)).toFixed(2)+' '+y(r.value).toFixed(2);connected=true;return part;}).join(' ')}" stroke="${s.color}" ${s.dash?'stroke-dasharray="6 4"':''}/>`;}).join('')}${[0,Math.floor((days.length-1)/2),days.length-1].map(i=>`<text x="${x(i)}" y="${h-12}" text-anchor="${i===0?'start':i===days.length-1?'end':'middle'}">${days[i]}</text>`).join('')}<rect class="brush" y="${top}" height="${bottom-top}" width="0"/><line class="crosshair" x1="0" x2="0" y1="${top}" y2="${bottom}" visibility="hidden"/></svg><div class="chart-tooltip" hidden></div></div>`;
    host.querySelectorAll('[data-curve]').forEach(b=>b.onclick=()=>{visible.has(b.dataset.curve)?visible.delete(b.dataset.curve):visible.add(b.dataset.curve);draw();});
    const svg=host.querySelector('svg'),tip=host.querySelector('.chart-tooltip'),cross=host.querySelector('.crosshair'),brush=host.querySelector('.brush');let down=null;
    const position=e=>Math.max(0,Math.min(days.length-1,Math.round((((e.clientX-svg.getBoundingClientRect().left)/svg.getBoundingClientRect().width)*w-left)/(right-left)*(days.length-1))));
    function inspect(i){focus=i;cross.setAttribute('x1',x(i));cross.setAttribute('x2',x(i));cross.setAttribute('visibility','visible');tip.hidden=false;tip.innerHTML=`<b>${days[i]}</b>`+series.map((s,j)=>visible.has(s.name)?`<div><i style="background:${s.color}"></i>${esc(s.name)} <strong>${Number.isFinite(maps[j].get(days[i]))?esc(format(maps[j].get(days[i]))):'无数据'}</strong></div>`:'').join('');tip.style.left=Math.min(72,Math.max(1,x(i)/w*100-12))+'%';}
    svg.onpointermove=e=>{const i=position(e);inspect(i);if(down!==null){brush.setAttribute('x',x(Math.min(i,down)));brush.setAttribute('width',Math.abs(x(i)-x(down)));}};
    svg.onpointerleave=()=>{if(down===null){tip.hidden=true;cross.setAttribute('visibility','hidden');}};
    svg.onpointerdown=e=>{down=position(e);svg.setPointerCapture(e.pointerId);};
    svg.onpointerup=e=>{const end=position(e),start=down;down=null;brush.setAttribute('width','0');if(start===null)return;if(onRange&&Math.abs(end-start)>2)onRange(days[Math.min(start,end)],days[Math.max(start,end)]);else if(onDate)onDate(days[end]);};
    svg.onpointercancel=()=>{down=null;brush.setAttribute('width','0');};
    svg.onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();inspect(Math.max(0,Math.min(days.length-1,focus+(e.key==='ArrowRight'?1:-1))));}if(e.key==='Enter'&&onDate)onDate(days[focus]);};
  }draw();
  let lastWidth=host.clientWidth;if(typeof ResizeObserver!=='undefined'){host._chartObserver=new ResizeObserver(()=>{if(!host.isConnected){host._chartObserver.disconnect();return;}if(Math.abs(host.clientWidth-lastWidth)>1){lastWidth=host.clientWidth;draw();}});host._chartObserver.observe(host);}
};
