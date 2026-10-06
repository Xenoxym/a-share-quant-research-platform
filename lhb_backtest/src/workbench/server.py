"""Loopback-only UI. No arbitrary paths, shell commands or strategy-code execution."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import socket
import threading
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen
import webbrowser

import pandas as pd

from .artifacts import content_id, jsonable, records
from .contracts import ASSUMPTIONS, FEATURE_REGISTRY, StrategyCard
from .report import EVENT_COLUMNS, PORTFOLIOS, event_evidence, render_page
from .runner import code_inventory, run_research

RUN_PATTERN = re.compile(r"^\d{8}T\d{6}-[0-9a-f]{8}$")
EVENT_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}\.(?:SH|SZ|BJ)$")


def make_server(project, root=None, port=8765):
    project = Path(project).resolve()
    root = Path(root or project / "data/workbench").resolve()
    token = secrets.token_urlsafe(32)
    workspace_id = content_id([str(project), str(root)])
    boot_code = code_inventory()
    guard = threading.Lock()
    job = {"status": "idle", "messages": []}

    def folder_for(run_id):
        if not RUN_PATTERN.fullmatch(run_id):
            raise ValueError("无效实验编号")
        folder = root / "runs" / run_id
        if not (folder / "manifest.json").exists():
            raise ValueError("实验尚未完成或不存在")
        return folder

    def snapshot_for(folder):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        key = manifest["snapshot_id"]
        if not re.fullmatch(r"[a-f0-9]{24}", key):
            raise ValueError("无效快照编号")
        return root / "snapshots" / key

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def valid_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def send(self, data, content_type="application/json; charset=utf-8", status=200, filename=None):
            body = json.dumps(jsonable(data), ensure_ascii=False, allow_nan=False).encode("utf-8") if content_type.startswith("application/json") else data.encode("utf-8") if isinstance(data, str) else data
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_GET(self):
            if not self.valid_host():
                return self.send({"error": "仅允许本机访问"}, status=403)
            try:
                self.get_route()
            except (ValueError, KeyError, FileNotFoundError) as exc:
                self.send({"error": str(exc)}, status=400)
            except Exception as exc:
                self.send({"error": f"读取失败: {exc}"}, status=500)

        def get_route(self):
            parsed = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            if parsed.path == "/":
                latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))["run_id"] if (root / "latest.json").exists() else None
                return self.send(render_page({"offline": False, "token": token, "latest": latest,
                                 "defaults": StrategyCard().to_dict(), "assumptions": ASSUMPTIONS,
                                 "features": FEATURE_REGISTRY}), "text/html; charset=utf-8")
            if parsed.path == "/favicon.ico":
                return self.send(b"", "image/x-icon", status=204)
            if parsed.path == "/api/job":
                with guard:
                    return self.send(dict(job))
            if parsed.path == "/api/health":
                return self.send({"app": "lhb-workbench", "workspace_id": workspace_id})
            if parsed.path == "/api/runs":
                runs = []
                for folder in sorted((root / "runs").glob("*"), reverse=True):
                    if not RUN_PATTERN.fullmatch(folder.name):
                        continue
                    status_path = folder / "status.json"
                    if not status_path.exists():
                        continue
                    state = json.loads(status_path.read_text(encoding="utf-8"))
                    row = {k: state.get(k) for k in ["run_id", "status", "error", "started_at"]}
                    if (folder / "manifest.json").exists():
                        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                        row.update(card=manifest["card"], name=manifest["card"]["name"],
                                   metrics=json.loads((folder / "institutional/metrics.json").read_text(encoding="utf-8")))
                    runs.append(row)
                return self.send(runs)
            parts = parsed.path.strip("/").split("/")
            if len(parts) != 4 or parts[:2] != ["api", "runs"]:
                return self.send({"error": "未找到页面"}, status=404)
            folder = folder_for(parts[2])
            endpoint = parts[3]
            if endpoint == "result":
                return self.send(json.loads((folder / "result.json").read_text(encoding="utf-8")))
            if endpoint == "events":
                frame = pd.read_parquet(folder / "events.parquet", columns=EVENT_COLUMNS)
                kind = query.get("kind", "signal")
                if kind == "signal":
                    frame = frame.loc[frame.signal]
                elif kind == "matched":
                    frame = frame.loc[frame.matched_treatment]
                elif kind == "control":
                    frame = frame.loc[frame.matched_control]
                elif kind == "rejected":
                    frame = frame.loc[~frame.eligibility_reason.eq("eligible")]
                elif kind != "all":
                    raise ValueError("未知事件分类")
                search = query.get("query", "")[:100]
                if search:
                    mask = frame.stock_code.str.contains(search, regex=False, na=False) | frame.stock_name.str.contains(search, regex=False, na=False) | frame.trade_date.str.contains(search, regex=False, na=False)
                    frame = frame.loc[mask]
                page = max(1, min(100000, int(query.get("page", "1"))))
                frame = frame.sort_values(["trade_date", "stock_code"], ascending=[False, True])
                return self.send({"total": len(frame), "rows": records(frame.iloc[(page - 1)*50:page*50])})
            if endpoint == "event":
                event_id = query.get("event_id", "")
                if not EVENT_PATTERN.fullmatch(event_id):
                    raise ValueError("无效事件编号")
                return self.send(event_evidence(folder, snapshot_for(folder), event_id))
            if endpoint == "day":
                day = query.get("date", "")
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
                    raise ValueError("无效日期")
                return self.send({name: records(pd.read_parquet(folder / "institutional" / f"{name}.parquet", filters=[("date", "=", day)])) for name in ["positions", "orders"]})
            if endpoint == "download":
                file = query.get("file", "")
                allowed = {"events.csv", "pairs.csv", "manifest.json", "card.json", "report.html", "result.json", "feature_registry.json", "assumptions.json"}
                allowed |= {f"{p}/{f}.csv" for p in PORTFOLIOS for f in ["orders", "trades", "equity", "ledger", "positions", "warnings"]}
                if file not in allowed:
                    raise ValueError("该文件不属于可下载的研究结果")
                path = folder / file
                content_type = "application/octet-stream"
                return self.send(path.read_bytes(), content_type, filename=file.replace("/", "_"))
            self.send({"error": "未找到接口"}, status=404)

        def do_POST(self):
            if not self.valid_host() or self.headers.get("X-Workbench-Token") != token:
                return self.send({"error": "请求来源校验失败，请刷新本地工作台"}, status=403)
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}:
                return self.send({"error": "不允许跨站请求"}, status=403)
            if self.path != "/api/run":
                return self.send({"error": "未找到接口"}, status=404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("请求大小无效")
                card = StrategyCard.from_dict(json.loads(self.rfile.read(length)))
                if code_inventory() != boot_code:
                    return self.send({"error": "工作台代码已更新，请关闭并重新启动服务后运行研究"}, status=409)
                with guard:
                    if job["status"] == "running":
                        return self.send({"error": "已有研究正在运行"}, status=409)
                    job.clear()
                    job.update(status="running", messages=["准备研究输入…"])

                def progress(message):
                    with guard:
                        job["messages"].append(message)

                def work():
                    try:
                        folder = run_research(project, card, root=root, progress=progress)
                        with guard:
                            job.update(status="completed", run_id=folder.name)
                    except Exception as exc:
                        with guard:
                            job.update(status="failed", error=str(exc))

                # Non-daemon: a normal server shutdown lets the immutable run finish safely.
                threading.Thread(target=work, name="research-run", daemon=False).start()
                self.send({"status": "running"}, status=202)
            except (ValueError, TypeError) as exc:
                self.send({"error": str(exc)}, status=400)

    class LocalServer(ThreadingHTTPServer):
        # SO_REUSEADDR on Windows can let a second live server bind the same port.
        allow_reuse_address = False

        def server_bind(self):
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()

    return LocalServer(("127.0.0.1", port), Handler)


def serve(project, root=None, port=8765, open_browser=True):
    project = Path(project).resolve()
    root = Path(root or project / "data/workbench").resolve()
    try:
        server = make_server(project, root, port)
    except OSError:
        # Re-opening the launcher should reuse this workspace's existing server.
        url = f"http://127.0.0.1:{port}"
        try:
            with urlopen(url + "/api/health", timeout=2) as response:
                health = json.loads(response.read(4096))
            if health.get("app") != "lhb-workbench" or health.get("workspace_id") != content_id([str(project), str(root)]):
                raise ValueError("端口由其他服务占用")
        except Exception as exc:
            raise RuntimeError(f"端口 {port} 不可用，请使用 --port 指定另一个端口") from exc
        print(f"研究工作台已运行: {url}", flush=True)
        if open_browser:
            webbrowser.open(url)
        return
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"研究工作台: {url}  （Ctrl+C 关闭服务）", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("服务已停止接收请求。正在计算的研究会完成保存。", flush=True)
    finally:
        server.server_close()
