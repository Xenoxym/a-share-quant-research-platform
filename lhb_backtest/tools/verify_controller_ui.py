"""V7 acceptance using a dedicated headless browser. Never starts paid agents."""

import json
from pathlib import Path

from verify_workbench_ui import Browser


def main():
    out = Path("data/technical/validation/controller-v7")
    out.mkdir(parents=True, exist_ok=True)
    b = Browser(9227, "http://127.0.0.1:8765")
    checks = []
    try:
        b.errors.clear()
        b.call("Page.navigate", {"url": "http://127.0.0.1:8765/"})
        b.until("document.querySelector('#drawdown-chart svg')!==null")
        b.call(
            "Emulation.setDeviceMetricsOverride",
            dict(width=1600, height=1100, deviceScaleFactor=1, mobile=False),
        )
        b.evaluate("document.querySelector('[data-view=campaigns]').click();window.scrollTo(0,0)")
        b.until("document.querySelectorAll('.controller-agent').length>=4")
        assert b.evaluate(
            "document.getElementById('controller-detail').textContent.includes('方法复核')"
        )
        assert b.evaluate(
            "document.getElementById('controller-detail').textContent.includes('数据复核')"
        )
        assert b.evaluate(
            "document.getElementById('controller-detail').textContent.includes('独立会话')"
        )
        assert not b.evaluate("document.getElementById('controller-message').textContent")
        b.screenshot(out / "desktop.png")
        checks.append(
            "Live campaign, budgets, independent sessions, failed attempts and review evidence rendered"
        )
        b.evaluate("document.querySelector('.controller-agents').scrollIntoView({block:'start'})")
        b.screenshot(out / "reviews.png")
        b.evaluate("document.getElementById('controller-new').click()")
        b.until("document.getElementById('campaign-form')!==null")
        assert b.evaluate(
            "document.querySelectorAll('#campaign-form input[type=number]').length===5"
        )
        b.evaluate("document.getElementById('campaign-cancel').click()")
        checks.append(
            "Multi-task and bounded budget form can be opened/cancelled without dispatching agents"
        )
        b.evaluate("document.querySelector('[data-controller-task]').click()")
        b.until(
            "!document.querySelector('[data-page=tasks]').hidden && document.getElementById('task-detail')?.textContent.includes('分析证据包')"
        )
        assert b.evaluate(
            "document.getElementById('task-detail').textContent.includes('非预登记账户实验')"
        )
        b.screenshot(out / "analysis-evidence.png")
        checks.append(
            "Campaign drilldown opens correct task with imported analysis and evidence boundary"
        )
        b.evaluate("document.querySelector('[data-view=campaigns]').click()")
        b.until("document.querySelector('.controller-agent')!==null")
        b.call(
            "Emulation.setDeviceMetricsOverride",
            dict(width=390, height=844, deviceScaleFactor=1, mobile=True),
        )
        b.evaluate("window.scrollTo(0,0)")
        assert b.evaluate("document.documentElement.scrollWidth <= innerWidth+2")
        b.screenshot(out / "mobile.png")
        checks.append("Mobile controller fits viewport")
        assert not b.errors, b.errors
        payload = dict(status="pass", checks=checks, javascript_errors=b.errors)
        (out / "browser_checks.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, ensure_ascii=True))
    finally:
        b.ws.close()


if __name__ == "__main__":
    main()
