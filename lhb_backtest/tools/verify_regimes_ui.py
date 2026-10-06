"""V5 acceptance via a dedicated headless Edge profile at localhost:9225."""
import json
from pathlib import Path
from urllib.request import urlopen
from verify_workbench_ui import Browser


def verify():
    out=Path('data/technical/validation/v5');out.mkdir(parents=True,exist_ok=True)
    checks=[];b=Browser(9225,'http://127.0.0.1:8765');b.errors.clear()
    try:
        b.call('Emulation.setDeviceMetricsOverride',{'width':2192,'height':1100,'deviceScaleFactor':1,'mobile':False})
        b.call('Page.navigate',{'url':'http://127.0.0.1:8765/'})
        b.until("document.querySelector('#drawdown-chart svg')!==null")
        geometry=b.evaluate("(()=>{const h=document.querySelector('#drawdown-chart'),s=h.querySelector('svg'),p=h.querySelector('path');return {host:h.clientWidth,svg:s.getBoundingClientRect().width,height:s.getBoundingClientRect().height,viewbox:s.getAttribute('viewBox'),path:p.getBBox().width}})()")
        assert geometry['height']==190 and geometry['path']>geometry['host']*.9,geometry
        b.evaluate("document.getElementById('range-start').value='2024-01-15';document.getElementById('range-end').value='2024-02-07';document.getElementById('apply-range').click()")
        b.until("document.querySelector('#period-content .note')?.textContent.includes('2024-01-15')")
        b.evaluate("document.querySelector('#drawdown-chart').scrollIntoView({block:'center'})")
        points=b.evaluate("(()=>{const s=document.querySelector('#drawdown-chart svg'),r=s.getBoundingClientRect(),p=s.querySelector('path');const a=[...p.getAttribute('d').matchAll(/[ML]([0-9.]+) ([0-9.]+)/g)].map(v=>[+v[1],+v[2]]);const lowest=a.reduce((p,q)=>q[1]>p[1]?q:p);s.dispatchEvent(new PointerEvent('pointermove',{clientX:r.x+lowest[0],clientY:r.y+lowest[1]}));return {tip:document.querySelector('#drawdown-chart .chart-tooltip').textContent,labels:[...s.querySelectorAll('.grid text')].map(t=>t.textContent)}})()")
        assert points['labels'][-1]=='0.00%',points
        # Compare the graph's worst point with the server's independent interval calculation.
        latest=json.loads(Path('data/technical/latest.json').read_text())['run_id']
        with urlopen(f'http://127.0.0.1:8765/api/runs/{latest}/period?start=2024-01-15&end=2024-02-07') as r:period=json.load(r)
        assert f"{period['stats']['max_drawdown']*100:.2f}%" in points['tip'],(points,period.keys())
        b.screenshot(out/'drawdown-fixed-desktop.png')
        checks.append(dict(check='wide drawdown fills width; fixed height, zero ceiling, hover equals interval account drawdown',geometry=geometry,tooltip=points['tip']))
        b.evaluate("document.querySelector('[data-view=regimes]').click();window.scrollTo(0,0)")
        b.until("document.querySelectorAll('[data-regime-detail]').length===12")
        b.screenshot(out/'regimes-desktop.png')
        b.evaluate("document.querySelector('[data-regime-detail=hmm3_mix]').click()")
        b.until("document.querySelector('#regime-detail h3')?.textContent.includes('三状态')")
        assert b.evaluate("document.querySelector('#regime-allocation-chart svg')!==null && document.querySelector('#regime-state-chart svg')!==null")
        assert b.evaluate("document.querySelectorAll('#regime-nav-chart .series-line').length===3 && document.querySelectorAll('#regime-state-chart .series-line').length===3")
        b.evaluate("document.querySelector('#regime-nav-chart [data-curve]').click()")
        assert b.evaluate("document.querySelectorAll('#regime-nav-chart .series-line').length===2")
        b.evaluate("document.querySelector('#regime-detail').scrollIntoView({block:'start'})")
        b.screenshot(out/'regime-evidence.png')
        checks.append('12 executed accounts; change selected strategy, toggle curves, see allocation probabilities and annual evidence')
        b.evaluate("document.querySelector('#regime-detail [data-regime-run]').click()")
        b.until("!document.querySelector('[data-page=results]').hidden && document.querySelector('.result-heading h2')?.textContent.includes('三状态')")
        b.evaluate("document.querySelector('[data-view=decisions]').click();document.querySelector('#load-decisions').click()")
        b.until("document.querySelector('#decision-content')?.textContent.includes('模型训练截至')")
        assert b.evaluate("document.querySelector('#decision-content').textContent.includes('两个分支入选股票的并集')")
        checks.append('allocation strategy drilldown explains component weight, target allocation and model training cutoff')
        b.evaluate("document.querySelector('[data-view=create]').click();document.querySelector('[name=family]').value='allocation';document.querySelector('[name=family]').dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('[name=allocation_policy]').value='hmm2_mix';document.querySelector('[name=allocation_policy]').dispatchEvent(new Event('input',{bubbles:true}))")
        b.until("document.querySelector('#definition-preview')?.textContent.includes('HMM两状态软切换')")
        assert b.evaluate("document.querySelector('[name=start_date]').value>='2022-01-01'")
        checks.append('create form exposes allocation rule and prevents pre-training evaluation start')
        b.call('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':1,'mobile':True})
        b.evaluate("document.querySelector('[data-view=results]').click();document.querySelector('#drawdown-chart').scrollIntoView({block:'center'})")
        b.until("document.querySelector('#drawdown-chart svg')?.getBoundingClientRect().width<400")
        assert b.evaluate("document.documentElement.scrollWidth<=window.innerWidth+2")
        b.screenshot(out/'drawdown-fixed-mobile.png')
        b.evaluate("document.querySelector('[data-view=regimes]').click();window.scrollTo(0,0)")
        b.until("document.querySelectorAll('[data-regime-detail]').length===12")
        assert b.evaluate("document.documentElement.scrollWidth<=window.innerWidth+2")
        b.screenshot(out/'regimes-mobile.png')
        assert not b.errors,b.errors
        checks.append('mobile layout fits viewport; charts resize on navigation and viewport changes; no JavaScript errors')
        payload=dict(status='pass',checks=checks,javascript_errors=b.errors)
        (out/'browser_checks.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        return payload
    finally:b.ws.close()


if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=True))
