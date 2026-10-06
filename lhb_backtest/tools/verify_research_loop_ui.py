"""End-to-end acceptance against a dedicated headless Edge debugging profile."""
import json
import argparse
from pathlib import Path
from verify_workbench_ui import Browser


def verify(port=9224, url="http://127.0.0.1:8765", output=Path("data/technical/validation/v3"), submit=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    b = Browser(port, url)
    checks = []
    try:
        b.call("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1100, "deviceScaleFactor": 1, "mobile": False})
        b.call("Page.navigate", {"url": url})
        b.until("document.querySelectorAll('#period-content tbody tr').length > 0")
        assert b.evaluate("document.getElementById('error').hidden")
        assert b.evaluate("document.querySelectorAll('#nav-chart .series-line').length === 2")
        b.screenshot(output / "overview.png")
        checks.append("overview and reconciled stock attribution; two default curves")
        b.evaluate("document.getElementById('nav-chart').scrollIntoView({block:'center'})")
        box = b.evaluate("(()=>{const r=document.querySelector('#nav-chart svg').getBoundingClientRect();return {x:r.x+r.width*.6,y:r.y+r.height*.4}})()")
        b.call("Input.dispatchMouseEvent", {"type": "mouseMoved", **box})
        b.until("!document.querySelector('#nav-chart .chart-tooltip').hidden")
        assert b.evaluate("document.querySelector('#nav-chart .chart-tooltip').textContent.includes('沪深300')")
        b.screenshot(output / "chart-hover.png")
        b.evaluate("document.querySelectorAll('#nav-chart [data-curve]')[1].click()")
        assert b.evaluate("document.querySelectorAll('#nav-chart .series-line').length === 1")
        checks.append("crosshair shows date and values; independent curve toggle")
        box = b.evaluate("(()=>{const r=document.querySelector('#nav-chart svg').getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}})()")
        start_x, end_x = box["x"]+box["width"]*.35, box["x"]+box["width"]*.7
        y = box["y"]+box["height"]*.5
        b.call("Input.dispatchMouseEvent", {"type": "mousePressed", "x": start_x, "y": y, "button": "left", "clickCount": 1})
        b.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": end_x, "y": y, "button": "left", "buttons": 1})
        b.call("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": end_x, "y": y, "button": "left", "clickCount": 1})
        b.until("document.getElementById('range-start').value > '2020-01-02' && document.getElementById('range-end').value < '2026-09-24'")
        b.until("document.querySelector('#period-content .note').textContent.includes(document.getElementById('range-start').value)")
        checks.append("actual mouse drag changes interval and refreshes account attribution")
        b.evaluate("document.querySelectorAll('.month')[12].click()")
        b.until("document.querySelector('#period-content .note').textContent.includes(document.getElementById('range-start').value)")
        assert b.evaluate("document.getElementById('range-start').value !== window.TECHNICAL.result?.data_start")
        checks.append("monthly heatmap updates interval account and contribution ranking")
        b.evaluate("document.querySelector('#period-content [data-case-code]').click()")
        b.until("document.querySelectorAll('#case-chart svg').length === 1")
        assert b.evaluate("document.getElementById('case-dialog').open && document.getElementById('case-content').textContent.includes('源码')")
        b.screenshot(output / "case.png")
        b.evaluate("document.getElementById('close-case').click()")
        checks.append("stock contribution opens raw quotes, decisions, orders, holdings and provenance")
        b.evaluate("document.getElementById('reset-range').click();document.querySelector('[data-help=costs]').focus()")
        assert b.evaluate("!document.getElementById('help-popover').hidden && document.getElementById('help-popover').textContent.includes('模拟滑点')")
        checks.append("keyboard-accessible cost definition includes slippage")
        b.evaluate("document.querySelector('[data-view=create]').click()")
        b.until("document.querySelector('#definition-preview math') !== null")
        b.evaluate("document.querySelector('[name=direction]').value='reversal';document.querySelector('[name=lookback]').value=20;document.querySelector('[name=skip]').value=0;document.querySelector('[name=trend_window]').value=126;document.querySelector('[name=direction]').dispatchEvent(new Event('input',{bubbles:true}))")
        b.until("document.getElementById('definition-preview').textContent.includes('最近跌得较多') && document.getElementById('definition-preview').textContent.includes('SMA126')")
        b.screenshot(output / "strategy.png")
        checks.append("plain-language strategy, data-to-action flow and typeset formula update together")
        b.evaluate("document.querySelector('[data-view=lab]').click();document.getElementById('batch-preview').click()")
        b.until("document.querySelectorAll('#batch-plan tbody tr').length === 24")
        b.until("document.getElementById('batch-results').textContent.includes('本轮研究判断')")
        assert b.evaluate("document.getElementById('batch-results').textContent.includes('本轮未找到值得晋级的方案')")
        b.evaluate("document.getElementById('batch-results').scrollIntoView({block:'start'})")
        b.screenshot(output / "laboratory.png")
        checks.append("24 declared hypotheses, cost settings and time split visible before execution")
        checks.append("actual completed batch links two post-screen experiments and an evidence-based review")
        b.evaluate("document.querySelector('[data-view=decisions]').click();document.getElementById('load-decisions').click()")
        b.until("document.querySelectorAll('#decision-content tbody tr').length > 0")
        b.evaluate("document.getElementById('load-account').click()")
        b.until("document.querySelectorAll('#account-content tbody tr').length > 0")
        checks.append("decision and daily account endpoints remain usable")
        b.evaluate("document.getElementById('account-scenario').value='zero_transaction_cost';document.getElementById('load-account').click()")
        b.until("document.querySelector('#account-content [data-case-scenario=zero_transaction_cost]') !== null")
        b.evaluate("document.querySelector('#account-content [data-case-code]').click()")
        b.until("document.querySelector('#case-chart svg') !== null")
        assert b.evaluate("document.getElementById('case-title').textContent.includes('零交易成本')")
        b.evaluate("document.getElementById('close-case').click()")
        checks.append("case drill-down preserves the selected cost scenario")
        b.call("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True})
        b.evaluate("document.querySelector('[data-view=results]').click();window.scrollTo(0,0)")
        assert b.evaluate("document.documentElement.scrollWidth <= window.innerWidth+2")
        b.screenshot(output / "mobile.png")
        checks.append("mobile result layout fits viewport")
        if submit:
            b.evaluate("document.querySelector('[data-view=lab]').click();document.getElementById('batch-run').click()")
            b.until("!document.getElementById('job').hidden")
            checks.append("actual 24-trial batch submitted through the UI")
        assert not b.errors, b.errors
        payload = {"status": "pass", "checks": checks, "javascript_errors": b.errors}
        (output / "checks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    finally:
        b.ws.close()


def main():
    parser = argparse.ArgumentParser(description="V3 browser acceptance; requires the completed delivery batch")
    parser.add_argument('--port', type=int, default=9224)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--output', type=Path, default=Path('data/technical/validation/v3'))
    parser.add_argument('--submit', action='store_true', help='Submit another real 24-trial batch after UI checks')
    args = parser.parse_args()
    print(json.dumps(verify(args.port, args.url, args.output, args.submit), ensure_ascii=True))


if __name__ == "__main__":
    main()
