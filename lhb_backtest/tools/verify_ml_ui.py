"""Desktop/narrow-screen ML UI acceptance in an owned headless browser."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time

import requests

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.technical.server import make_server
from src.technical.artifacts import write_json
from verify_workbench_ui import Browser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--browser", choices=["chrome", "edge"], default="chrome")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = [Path("C:/Program Files/Google/Chrome/Application/chrome.exe")] if args.browser == "chrome" else [Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")]
    chrome = next((p for p in candidates if p.exists()), None)
    if chrome is None:
        raise RuntimeError("本机没有可用Chromium进行UI验收")
    server = make_server(PROJECT, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        debug_port = sock.getsockname()[1]
    app_url = f"http://127.0.0.1:{server.server_port}"
    profile = args.output.resolve() / "browser_profile"
    if profile.exists():
        raise ValueError("QA浏览器目录已存在，请使用新的验收输出目录")
    browser_log = (args.output / "browser.log").open("w", encoding="utf-8")
    proc = subprocess.Popen([str(chrome), "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check", "--remote-debugging-address=127.0.0.1",
        "--no-sandbox", "--disable-extensions", "--disable-background-networking", "--disable-background-mode", "--enable-logging=stderr",
        f"--remote-debugging-port={debug_port}", f"--remote-allow-origins=http://127.0.0.1:{debug_port}",
        f"--user-data-dir={profile}", app_url], stdout=browser_log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    browser = None
    checks = []
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                pages = requests.get(f"http://127.0.0.1:{debug_port}/json", timeout=1).json()
                if any(p.get("url", "").startswith(app_url) for p in pages):
                    break
            except requests.RequestException:
                pass
            time.sleep(.2)
        browser = Browser(debug_port, app_url)
        browser.call("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1100, "deviceScaleFactor": 1, "mobile": False})
        browser.until("!!window.mlWorkbench && !!document.querySelector('[data-view=ml]')")
        browser.evaluate("document.querySelector('[data-view=ml]').click()")
        browser.until("!!document.getElementById('ml-run')")
        assert browser.evaluate("document.getElementById('ml-content').textContent.includes('未来五个交易日')")
        browser.evaluate(f"window.mlWorkbench.open({json.dumps(args.run_id)})")
        browser.until("document.querySelectorAll('#ml-scores tbody tr').length===5")
        browser.screenshot(args.output / "ml-desktop.png")
        assert browser.evaluate("document.querySelector('#ml-scope').value==='test'")
        checks.append("ML navigation, fixed protocol and five model/control test rows")
        browser.evaluate("document.querySelector('#ml-scope').value='test_2024';document.querySelector('#ml-scope').dispatchEvent(new Event('change'))")
        assert browser.evaluate("document.querySelector('#ml-quantiles').textContent.includes('全部测试期')")
        checks.append("year selector and whole-test quantile label")
        browser.evaluate(f"document.querySelector('[data-view=tasks]').click();window.taskWorkbench.open({json.dumps(args.task_id)})")
        browser.until("!!document.querySelector('[data-task-ml]')")
        assert browser.evaluate("document.getElementById('task-detail').textContent.includes('机器学习信号分析')")
        browser.evaluate("document.querySelector('[data-task-ml]').click()")
        browser.until("document.querySelectorAll('#ml-scores tbody tr').length===5")
        checks.append("research-task result opens ML diagnostics")
        browser.call("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True})
        browser.evaluate("window.scrollTo(0,0)")
        browser.screenshot(args.output / "ml-mobile.png")
        assert browser.evaluate("document.documentElement.scrollWidth<=window.innerWidth+2")
        checks.append("narrow-screen document fits viewport; wide tables scroll internally")
        assert not browser.errors, browser.errors
        write_json(args.output / "ui_checks.json", {"ok": True, "checks": checks, "runtime_errors": browser.errors, "run_id": args.run_id})
        print(json.dumps({"ok": True, "checks": checks}, ensure_ascii=False))
    except Exception as exc:
        write_json(args.output / "ui_failure.json", {"ok": False, "error": str(exc), "completed_checks": checks})
        raise
    finally:
        if browser:
            browser.ws.close()
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        proc.wait(timeout=10)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        browser_log.close()
        if profile.exists():
            resolved = profile.resolve()
            if resolved.parent != args.output.resolve() or resolved.name != "browser_profile":
                raise RuntimeError("QA目录越界，停止清理")
            for retry in range(10):
                try:
                    shutil.rmtree(resolved)
                    break
                except PermissionError:
                    time.sleep(.2)
            else:
                write_json(args.output / "cleanup_note.json", {"owned_profile": str(resolved), "reason": "Windows仍锁定QA文件，保留目录而不影响已完成的UI检查"})


if __name__ == "__main__":
    main()
