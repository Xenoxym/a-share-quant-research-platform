"""Browser acceptance via a dedicated local Chromium debugging session.

Requires websocket-client for QA only; the actual workbench uses stdlib HTTP.
Start a dedicated headless browser/profile with remote debugging on loopback,
then invoke this tool. Never attaches to a user's normal browser profile.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import time

import requests
import websocket


class Browser:
    def __init__(self, port, app_url):
        origin = f"http://127.0.0.1:{port}"
        pages = requests.get(origin + "/json", timeout=10).json()
        page = next(p for p in pages if p.get("type") == "page" and p.get("url", "").startswith(app_url))
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], origin=origin, timeout=20)
        self.sequence = 0
        self.errors = []
        self.call("Runtime.enable")
        self.call("Page.enable")

    def call(self, method, params=None):
        self.sequence += 1
        self.ws.send(json.dumps({"id": self.sequence, "method": method, "params": params or {}}))
        while True:
            reply = json.loads(self.ws.recv())
            if reply.get("method") == "Runtime.exceptionThrown":
                self.errors.append(reply["params"])
            if reply.get("id") == self.sequence:
                if "error" in reply:
                    raise AssertionError(reply["error"])
                return reply.get("result", {})

    def evaluate(self, expression):
        reply = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if "exceptionDetails" in reply:
            raise AssertionError(reply["exceptionDetails"])
        return reply.get("result", {}).get("value")

    def until(self, expression, timeout=30):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            result = self.evaluate(expression)
            if result:
                return result
            time.sleep(.2)
        raise AssertionError(f"UI timeout: {expression}")

    def screenshot(self, path):
        path.write_bytes(base64.b64decode(self.call("Page.captureScreenshot", {"format": "png"})["data"]))


def verify(port, app_url, output, submit=False):
    output.mkdir(parents=True, exist_ok=True)
    browser = Browser(port, app_url)
    checks = []
    try:
        browser.call("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1100, "deviceScaleFactor": 1, "mobile": False})
        browser.call("Page.navigate", {"url": app_url})
        browser.until("document.querySelectorAll('#nav-chart svg').length===1")
        browser.screenshot(output / "desktop.png")
        assert browser.evaluate("document.getElementById('message').hidden")
        checks.append("desktop result and chart render without error")
        browser.evaluate("document.querySelector('[data-page=events]').click()")
        browser.until("document.querySelectorAll('#events-table [data-event]').length>0")
        eid = browser.evaluate("document.querySelector('#events-table [data-event]').dataset.event")
        browser.evaluate("document.querySelector('#events-table [data-event]').click()")
        browser.until("document.querySelectorAll('#event-chart svg').length===1")
        browser.evaluate("document.getElementById('event-detail').scrollIntoView({behavior:'instant',block:'start'})")
        browser.until("Math.abs(document.getElementById('event-detail').getBoundingClientRect().top)<10")
        browser.screenshot(output / "event-detail.png")
        assert browser.evaluate("document.getElementById('event-detail').innerText.includes('所选报告的原始席位')")
        checks.append(f"event {eid}: candles, seats, pairs, orders and source references")
        browser.evaluate("document.getElementById('event-query').value='DOES_NOT_EXIST'; document.getElementById('event-search').click()")
        browser.until("document.getElementById('event-page').textContent.includes('共 0 个事件')")
        checks.append("literal event search and empty result")
        browser.evaluate("document.querySelector('[data-page=overview]').click(); document.getElementById('nav-chart').scrollIntoView()")
        browser.evaluate("(()=>{let e=document.querySelector('#nav-chart svg'),r=e.getBoundingClientRect();e.dispatchEvent(new MouseEvent('click',{bubbles:true,clientX:r.left+r.width*.5,clientY:r.top+50}));})()")
        browser.until("document.getElementById('day-detail').innerText.includes('机构组合持仓与订单')")
        checks.append("equity chart date -> daily positions and orders")
        browser.evaluate("document.querySelector('[data-page=archive]').click()")
        browser.until("document.querySelectorAll('[data-run]').length>0")
        checks.append("experiment history loads")
        browser.evaluate("document.querySelector('[data-page=create]').click(); window.scrollTo(0,0)")
        browser.screenshot(output / "strategy-card.png")
        assert browser.evaluate("Number(document.querySelector('[name=position_fraction]').value)===10")
        checks.append("strategy card displays human units")
        browser.call("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True})
        browser.evaluate("window.scrollTo(0,0)")
        browser.screenshot(output / "mobile-card.png")
        assert browser.evaluate("document.documentElement.scrollWidth<=window.innerWidth+2")
        checks.append("mobile strategy card fits viewport")
        if submit:
            browser.call("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1100, "deviceScaleFactor": 1, "mobile": False})
            browser.evaluate("document.querySelector('[name=name]').value='界面验收 · 短区间';document.querySelector('[name=start_date]').value='2026-09-01';document.querySelector('[name=end_date]').value='2026-09-24';document.getElementById('strategy-form').requestSubmit()")
            browser.until("!document.getElementById('job').hidden", timeout=10)
            checks.append("browser submission starts background computation")
            # Return a pending state; callers can poll without blocking user communication.
            return {"status": "submitted", "checks": checks, "javascript_errors": browser.errors}
        assert not browser.errors, browser.errors
        return {"status": "pass", "checks": checks, "javascript_errors": browser.errors}
    finally:
        browser.ws.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-port", type=int, default=9224)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=Path("data/workbench/validation"))
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    result = verify(args.debug_port, args.url, args.output, args.submit)
    (args.output / "ui_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
