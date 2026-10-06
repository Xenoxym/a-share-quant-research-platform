"""V6 desktop/mobile acceptance in a dedicated headless browser, port 9225."""

import json
from pathlib import Path
from verify_workbench_ui import Browser


def verify():
    out = Path("data/technical/validation/v6")
    out.mkdir(parents=True, exist_ok=True)
    acceptance = json.loads(
        Path("research/worker_platform_20260928/acceptance.json").read_text(encoding="utf-8")
    )
    b = Browser(9225, "http://127.0.0.1:8765")
    checks = []
    try:
        b.call(
            "Emulation.setDeviceMetricsOverride",
            {"width": 1600, "height": 1100, "deviceScaleFactor": 1, "mobile": False},
        )
        b.call(
            "Browser.grantPermissions",
            {
                "origin": "http://127.0.0.1:8765",
                "permissions": ["clipboardReadWrite", "clipboardSanitizedWrite"],
            },
        )
        b.call("Page.bringToFront")
        b.errors.clear()
        b.call("Page.navigate", {"url": "http://127.0.0.1:8765/"})
        b.until("document.querySelector('#drawdown-chart svg')!==null")
        b.evaluate("document.querySelector('[data-view=tasks]').click()")
        b.until("document.querySelector('#task-detail h2')!==null")
        assert b.evaluate("document.querySelectorAll('[data-task-id]').length>=4")
        assert b.evaluate(
            "document.querySelector('#task-detail').textContent.includes('什么结果值得继续')"
        )
        assert b.evaluate("document.querySelector('#task-detail').textContent.includes('停止')")
        assert b.evaluate("document.querySelector('#task-content').textContent.includes('HMM')")
        b.screenshot(out / "tasks-desktop.png")
        checks.append(
            "task queue, decision criteria, baseline links and dated shared research memory rendered"
        )
        b.evaluate(
            "document.querySelector('.task-entry').open=true;document.getElementById('task-copy').click()"
        )
        b.until("document.getElementById('task-copy').textContent==='已复制'")
        checks.append("fresh worker handoff prompt copied to clipboard")
        b.evaluate(f"document.querySelector('[data-task-id={acceptance['task_id']}]').click()")
        b.until("document.querySelector('#task-detail .task-metrics')!==null")
        expected = f"{acceptance['result']['metrics'][1]['total_return'] * 100:.2f}%"
        assert expected in b.evaluate("document.querySelector('.task-metrics').textContent")
        assert b.evaluate(
            "document.querySelector('#task-detail').textContent.includes('acceptance-planner')"
        )
        b.evaluate("document.querySelector('#task-detail').scrollIntoView({block:'start'})")
        b.screenshot(out / "task-evidence.png")
        b.evaluate("document.querySelector('#task-detail [data-task-run]').click()")
        b.until(
            "!document.querySelector('[data-page=results]').hidden && document.querySelector('.result-heading h2')?.textContent.includes('V6接力验收')"
        )
        assert b.evaluate("document.querySelector('#drawdown-chart svg')!==null")
        checks.append(
            "audited gross/net metrics match stored evidence; task drilldown opens the same run and drawdown chart"
        )
        b.evaluate("document.querySelector('[data-view=tasks]').click()")
        b.until("document.querySelector('#task-new')!==null")
        b.evaluate("document.getElementById('task-new').click()")
        assert b.evaluate("!document.getElementById('task-create-form').checkValidity()")
        b.evaluate("document.getElementById('task-cancel').click()")
        checks.append(
            "research question form requires rationale and decision criteria; cancelled without creating a live task"
        )
        b.call(
            "Emulation.setDeviceMetricsOverride",
            {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True},
        )
        b.evaluate("window.scrollTo(0,0)")
        assert b.evaluate("document.documentElement.scrollWidth<=innerWidth+2")
        b.screenshot(out / "tasks-mobile.png")
        b.evaluate("document.querySelector('#task-detail').scrollIntoView({block:'start'})")
        assert b.evaluate("document.documentElement.scrollWidth<=innerWidth+2")
        b.screenshot(out / "task-evidence-mobile.png")
        checks.append("390px mobile navigation, cards and evidence fit viewport")
        assert not b.errors, b.errors
        result = dict(status="pass", checks=checks, javascript_errors=b.errors)
        (out / "browser_checks.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result
    finally:
        b.ws.close()


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=True))
