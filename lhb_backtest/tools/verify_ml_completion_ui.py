"""Read-only acceptance of the completed study, costs and account navigation."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import requests

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.technical.server import make_server
from src.technical.artifacts import write_json
from verify_workbench_ui import Browser


def main():
    p=argparse.ArgumentParser();p.add_argument("run_id");p.add_argument("--task-id",required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    edge=Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
    server=make_server(PROJECT,port=0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0));port=sock.getsockname()[1]
    url=f"http://127.0.0.1:{server.server_port}"
    profile=args.output.resolve()/"browser_profile"
    log=(args.output/"browser.log").open("w",encoding="utf-8")
    proc=subprocess.Popen([str(edge),"--headless=new","--disable-gpu","--no-first-run","--no-default-browser-check","--no-sandbox",
        "--disable-extensions","--disable-background-networking",f"--remote-debugging-port={port}",f"--remote-allow-origins=http://127.0.0.1:{port}",f"--user-data-dir={profile}",url],
        stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    browser=None;checks=[]
    try:
        for _ in range(100):
            try:
                if requests.get(f"http://127.0.0.1:{port}/json",timeout=1).ok:break
            except requests.RequestException:pass
            time.sleep(.2)
        browser=Browser(port,url)
        browser.call("Emulation.setDeviceMetricsOverride",{"width":1440,"height":1250,"deviceScaleFactor":1,"mobile":False})
        browser.until("!!window.mlWorkbench")
        browser.evaluate(f"window.mlWorkbench.open({json.dumps(args.run_id)})")
        browser.until("document.querySelectorAll('#ml-accounts tbody tr').length===19")
        checks.append("19 real-account rows and complete-study navigation")
        assert browser.evaluate("document.getElementById('ml-result').textContent.includes('38个账户情景')")
        browser.evaluate("document.getElementById('ml-result').scrollIntoView()")
        browser.screenshot(args.output/"accounts-desktop.png")
        before=browser.evaluate("document.querySelector('#ml-accounts tbody tr').textContent")
        browser.evaluate("document.getElementById('ml-cost').value='zero_transaction_cost';document.getElementById('ml-cost').dispatchEvent(new Event('change'))")
        after=browser.evaluate("document.querySelector('#ml-accounts tbody tr').textContent")
        assert before!=after
        checks.append("cost selector changes actual recomputed account metrics")
        browser.evaluate("document.querySelector('[data-ml-account]').click()")
        browser.until("document.querySelector('[data-page=results]').hidden===false")
        assert browser.evaluate("document.querySelector('[data-page=results]').textContent.includes('冻结评分')")
        checks.append("account opens weekly external-score story and cash/holdings diagnostics")
        browser.screenshot(args.output/"account-detail.png")
        browser.evaluate(f"window.mlWorkbench.open({json.dumps(args.run_id)})")
        browser.until("document.querySelectorAll('#ml-accounts tbody tr').length===19")
        browser.call("Emulation.setDeviceMetricsOverride",{"width":390,"height":844,"deviceScaleFactor":1,"mobile":True})
        browser.evaluate("document.getElementById('ml-result').scrollIntoView()")
        browser.screenshot(args.output/"accounts-mobile.png")
        assert browser.evaluate("document.documentElement.scrollWidth<=window.innerWidth+2")
        checks.append("mobile layout fits viewport with scrolling comparison table")
        assert not browser.errors,browser.errors
        write_json(args.output/"ui_checks.json",dict(ok=True,checks=checks,runtime_errors=browser.errors))
        print(json.dumps(dict(ok=True,checks=checks),ensure_ascii=False))
    except Exception as exc:
        write_json(args.output/"ui_failure.json",dict(ok=False,error=str(exc),checks=checks));raise
    finally:
        if browser:browser.ws.close()
        subprocess.run(["taskkill","/PID",str(proc.pid),"/T","/F"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
        proc.wait(timeout=10);server.shutdown();server.server_close();thread.join(timeout=5);log.close()
        # Owned profile retained for troubleshooting; never touch a normal browser.


if __name__=="__main__":main()
