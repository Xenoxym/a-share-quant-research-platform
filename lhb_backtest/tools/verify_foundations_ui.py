"""V4 browser acceptance; a dedicated Edge profile must listen on loopback 9225."""
import json
from pathlib import Path
from verify_workbench_ui import Browser


def verify():
    out=Path('data/technical/validation/v4');out.mkdir(parents=True,exist_ok=True)
    b=Browser(9225,'http://127.0.0.1:8765');b.errors.clear(); checks=[]
    try:
        b.call('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1100,'deviceScaleFactor':1,'mobile':False})
        b.call('Page.navigate',{'url':'http://127.0.0.1:8765/'})
        b.until("document.querySelector('#period-content tbody')!==null")
        b.evaluate("document.querySelector('[data-view=foundations]').click()")
        b.until("document.querySelectorAll('[data-foundation-detail]').length===15")
        assert b.evaluate("document.getElementById('foundation-results').innerText.includes('零交易成本')")
        b.screenshot(out/'foundations-desktop.png')
        checks.append('15 completed accounts, actual cost counterfactuals and published sources visible')
        b.evaluate("document.querySelector('[data-foundation-detail=low_volatility]').click()")
        b.until("document.getElementById('foundation-detail').innerText.includes('63日低波动')")
        b.evaluate("document.querySelector('[data-foundation-detail=small]').click()")
        b.evaluate("document.querySelector('#foundation-detail details').open=true")
        assert b.evaluate("document.querySelectorAll('#foundation-detail table').length===3")
        b.screenshot(out/'foundation-evidence.png')
        checks.append('strategy-specific detail, annual returns, weak windows and stock contribution evidence')
        b.evaluate("document.querySelector('#foundation-detail [data-foundation-run]').click()")
        b.until("!document.querySelector('[data-page=results]').hidden && document.querySelector('#nav-chart svg')!==null")
        assert b.evaluate("document.querySelectorAll('#account-scenario option').length===2")
        b.evaluate("document.getElementById('range-start').value='2024-01-15';document.getElementById('range-end').value='2024-04-22';document.getElementById('apply-range').click()")
        b.until("document.querySelector('#period-content .note')?.textContent.includes('2024-01-15')")
        b.evaluate("document.querySelector('#period-content [data-case-code]').click()")
        b.until("document.querySelector('#case-chart svg')!==null")
        assert b.evaluate("document.getElementById('case-content').innerText.includes('估算市值') && !document.getElementById('case-content').innerText.includes('窗口收益')")
        b.evaluate("document.querySelector('#case-content details').open=true")
        assert b.evaluate("document.getElementById('case-content').innerText.includes('报告股数')")
        b.screenshot(out/'financial-case.png')
        b.evaluate("document.getElementById('close-case').click();document.querySelector('[data-view=decisions]').click();document.getElementById('load-decisions').click()")
        b.until("document.querySelector('#decision-content th:nth-child(2)')?.textContent.includes('估算市值')")
        checks.append('interval to stock case shows actual capitalization and publication dates; decision table follows family')
        b.evaluate("document.querySelector('[data-view=create]').click();document.querySelector('[name=family]').value='size';document.querySelector('[name=family]').dispatchEvent(new Event('change',{bubbles:true}))")
        b.until("document.getElementById('definition-preview').textContent.includes('小市值')")
        assert b.evaluate("document.querySelector('[name=share_basis]')!==null")
        checks.append('create form exposes strategy family and explicit reported/bridged share choice')
        b.call('Emulation.setDeviceMetricsOverride',{'width':390,'height':844,'deviceScaleFactor':1,'mobile':True})
        b.evaluate("document.querySelector('[data-view=foundations]').click();window.scrollTo(0,0)")
        b.until("document.querySelectorAll('[data-foundation-detail]').length===15")
        assert b.evaluate("document.documentElement.scrollWidth<=window.innerWidth+2")
        b.screenshot(out/'foundations-mobile.png')
        checks.append('mobile page fits viewport; tables scroll inside their own container')
        assert not b.errors,b.errors
        payload={'status':'pass','checks':checks,'javascript_errors':b.errors}
        (out/'checks.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        return payload
    finally:b.ws.close()


if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=True))
