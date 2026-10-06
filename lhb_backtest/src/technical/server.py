"""Loopback-only technical research UI; legacy reports are read-only artifacts."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen
import webbrowser

import pandas as pd

from .artifacts import content_id, jsonable, records
from .contracts import ResearchSpec, SCENARIOS, describe
from .runner import code_inventory, run
from .web import render_page
from .diagnostics import analysis_for, execution_for, inspect_case, inspect_period
from .laboratory import plan, run_batch

RUN_ID = re.compile(r"^\d{8}T\d{6}-[0-9a-f]{8}$")


def make_server(project, port=8765, root=None):
    from ..researchops.service import Research
    from ..researchops.controller import Campaigns
    project = Path(project).resolve()
    root = Path(root or project / "data/technical").resolve()
    research = Research(project, root)
    campaigns = Campaigns(research)
    token, guard = secrets.token_urlsafe(32), threading.Lock()
    boot_code, job = code_inventory(), {"status": "idle", "messages": []}
    from .artifacts import digest
    def ml_code():
        return {p.name: digest(p) for p in sorted((project / "src/mlresearch").glob("*.py"))}
    boot_ml_code = ml_code()
    workspace = content_id([str(project), str(root)])

    def folder(run_id):
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("实验编号无效")
        p = root / "runs" / run_id
        if not (p / "manifest.json").exists():
            raise ValueError("实验未完成或不存在")
        return p

    def ml_folder(run_id):
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("ML运行编号无效")
        p = root / "ml_runs" / run_id
        if not (p / "manifest.json").exists():
            raise ValueError("ML实验未完成或不存在")
        return p

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, value, mime="application/json; charset=utf-8", status=200):
            body = json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False).encode() if mime.startswith("application/json") else value.encode() if isinstance(value, str) else value
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def valid_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def do_GET(self):
            if not self.valid_host():
                return self.send({"error": "仅允许本机访问"}, status=403)
            try:
                self.route_get()
            except (ValueError, KeyError, FileNotFoundError) as exc:
                self.send({"error": str(exc)}, status=400)
            except Exception as exc:
                self.send({"error": str(exc)}, status=500)

        def route_get(self):
            parsed = urlparse(self.path)
            path = parsed.path
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            if path == "/":
                latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))["run_id"] if (root / "latest.json").exists() else None
                return self.send(render_page({"token": token, "defaults": ResearchSpec().to_dict(), "latest": latest}), "text/html; charset=utf-8")
            if path == "/favicon.ico":
                return self.send(b"", "image/x-icon", 204)
            if path == "/api/health":
                return self.send({"app": "technical-research", "workspace_id": workspace})
            if path == "/api/job":
                with guard:
                    return self.send(dict(job))
            if path == "/api/ml/plan":
                from ..mlresearch.contracts import benchmark_proposal
                return self.send(benchmark_proposal())
            if path == "/api/ml/complete-plan":
                from ..mlresearch.contracts import completion_proposal
                p = completion_proposal()
                return self.send({"available": (root / "ml_runs" / p["spec"]["source_run_id"] / "manifest.json").exists(), "proposal": p})
            if path == "/api/ml":
                return self.send([json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "ml_runs").glob("*/status.json"), reverse=True)])
            mm = re.fullmatch(r"/api/ml/([^/]+)(?:/(download))?", path)
            if mm:
                p = ml_folder(mm[1])
                if mm[2]:
                    name = query.get("file", "")
                    manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
                    if name not in manifest["artifacts"] or "/" in name or "\\" in name:
                        raise ValueError("未归档的ML文件")
                    from .artifacts import digest
                    if digest(p / name) != manifest["artifacts"][name]:
                        raise ValueError("ML文件完整性检查失败")
                    return self.send((p / name).read_bytes(), "application/octet-stream")
                return self.send(json.loads((p / "result.json").read_text(encoding="utf-8")))
            if path == "/api/research/catalog":
                return self.send(research.catalog())
            if path == "/api/research/tasks":
                return self.send(research.store.list())
            if path == "/api/research/campaigns":
                return self.send(campaigns.list())
            campaign_match = re.fullmatch(r"/api/research/campaigns/(campaign-[a-f0-9]{12})", path)
            if campaign_match:
                return self.send(campaigns.get(campaign_match[1]))
            task_match = re.fullmatch(r"/api/research/tasks/(task-[a-z0-9-]+)", path)
            if task_match:
                return self.send(research.context(task_match[1]))
            if path == "/api/foundations/plan":
                from .foundation_lab import plan as foundation_plan
                return self.send(foundation_plan())
            if path == "/api/regimes/plan":
                from .regime_lab import plan as regime_plan
                return self.send(regime_plan())
            if path == "/api/regimes":
                return self.send([json.loads((p / "status.json").read_text(encoding="utf-8"))
                    for p in sorted((root / "regimes").glob("*"), reverse=True) if (p / "status.json").exists()])
            rm = re.fullmatch(r"/api/regimes/([^/]+)", path)
            if rm:
                if not RUN_ID.fullmatch(rm[1]):
                    raise ValueError("研究批次编号无效")
                value = json.loads((root / "regimes" / rm[1] / "result.json").read_text(encoding="utf-8"))
                value["research_reviews"] = [r for p in sorted((project / "research").glob("*/regime_review.json"))
                    if (r := json.loads(p.read_text(encoding="utf-8"))).get("suite_id") == rm[1]]
                return self.send(value)
            if path == "/api/foundations":
                return self.send([json.loads((p / "status.json").read_text(encoding="utf-8"))
                    for p in sorted((root / "foundations").glob("*"), reverse=True) if (p / "status.json").exists()])
            fm = re.fullmatch(r"/api/foundations/([^/]+)", path)
            if fm:
                if not RUN_ID.fullmatch(fm[1]):
                    raise ValueError("研究批次编号无效")
                foundation_folder = root / "foundations" / fm[1]
                value = json.loads((foundation_folder / "result.json").read_text(encoding="utf-8"))
                reviews = []
                for p in sorted((project / "research").glob("*/foundation_review.json")):
                    review = json.loads(p.read_text(encoding="utf-8"))
                    if review.get("suite_id") == fm[1]:
                        reviews.append(review)
                value["research_reviews"] = reviews
                return self.send(value)
            if path == "/api/runs":
                rows = []
                for p in sorted((root / "runs").glob("*"), reverse=True):
                    if not RUN_ID.fullmatch(p.name) or not (p / "status.json").exists():
                        continue
                    state = json.loads((p / "status.json").read_text(encoding="utf-8"))
                    if (p / "spec.json").exists():
                        state["name"] = json.loads((p / "spec.json").read_text(encoding="utf-8"))["strategy"]["name"]
                    rows.append(state)
                return self.send(rows)
            if path == "/api/batches":
                return self.send([json.loads((p / "status.json").read_text(encoding="utf-8"))
                    for p in sorted((root / "batches").glob("*"), reverse=True) if (p / "status.json").exists()])
            batch_match = re.fullmatch(r"/api/batches/([^/]+)", path)
            if batch_match:
                if not RUN_ID.fullmatch(batch_match[1]):
                    raise ValueError("批次编号无效")
                value = json.loads((root / "batches" / batch_match[1] / "result.json").read_text(encoding="utf-8"))
                # Reviews are separately dated research artifacts; never rewrite the
                # frozen protocol or claim later hypotheses were known beforehand.
                reviews = []
                for followup in sorted((project / "research").glob("*/followup_results.json")):
                    review = json.loads(followup.read_text(encoding="utf-8"))
                    if review.get("protocol", {}).get("parent_batch") == batch_match[1]:
                        note = followup.with_name("review.json")
                        if note.exists():
                            review["review"] = json.loads(note.read_text(encoding="utf-8"))
                        reviews.append(review)
                value["followups"] = reviews
                return self.send(value)
            if path == "/api/legacy":
                audit = project / "research/research_reset_20260926/cost_counterfactuals/result.json"
                old = project / "data/workbench/latest.json"
                return self.send({"rows": json.loads(audit.read_text(encoding="utf-8"))["rows"] if audit.exists() else [],
                    "latest": json.loads(old.read_text(encoding="utf-8"))["run_id"] if old.exists() else None})
            match = re.fullmatch(r"/legacy/([^/]+)/(.+)", path)
            if match:
                run_id, name = match.groups()
                if not RUN_ID.fullmatch(run_id):
                    raise ValueError("旧实验编号无效")
                p = project / "data/workbench/runs" / run_id
                manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
                if name not in manifest["artifacts"] or name.startswith("code/"):
                    raise ValueError("未归档的旧文件")
                mime = "text/html; charset=utf-8" if name == "report.html" else "application/octet-stream"
                return self.send((p / name).read_bytes(), mime)
            match = re.fullmatch(r"/api/runs/([^/]+)(?:/(decisions|day|download|analysis|period|case))?", path)
            if not match:
                return self.send({"error": "不存在的接口"}, status=404)
            p = folder(match[1])
            if not match[2]:
                data = json.loads((p / "result.json").read_text(encoding="utf-8"))
                # Presentation upgrades do not rewrite immutable historical evidence.
                if "analysis" not in data:
                    data["analysis"] = analysis_for(p)
                elif "execution" not in data["analysis"]:
                    data["analysis"]["execution"] = execution_for(p)
                return self.send(data)
            if match[2] == "analysis":
                return self.send(analysis_for(p))
            if match[2] == "period":
                scenario = query.get("scenario", "configured")
                if scenario not in SCENARIOS:
                    raise ValueError("情景无效")
                start, end = query.get("start", ""), query.get("end", "")
                if not all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) for d in [start, end]) or start > end:
                    raise ValueError("区间无效")
                return self.send(inspect_period(p, start, end, scenario))
            if match[2] == "case":
                code = query.get("code", "")
                if not re.fullmatch(r"\d{6}\.(?:SH|SZ)", code):
                    raise ValueError("股票代码无效")
                scenario = query.get("scenario", "configured")
                if scenario not in SCENARIOS:
                    raise ValueError("情景无效")
                return self.send(inspect_case(p, query.get("date", ""), code, scenario))
            if match[2] == "download":
                name = query.get("file", "")
                manifest = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
                if name not in manifest["artifacts"] and name != "manifest.json":
                    raise ValueError("文件未登记")
                return self.send((p / name).read_bytes(), "application/octet-stream")
            day = query.get("date", "")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
                raise ValueError("日期无效")
            if match[2] == "decisions":
                frame = pd.read_parquet(p / "decisions.parquet", filters=[("trade_date", "=", day)])
                targets = pd.read_parquet(p / "configured/targets.parquet", filters=[("signal_date", "=", day)])
                if not targets.empty:
                    frame = frame.merge(targets[["stock_code", "target_shares", "equity_used"]], on="stock_code", how="left", validate="one_to_one")
                frame = frame.sort_values(["selected", "rank", "stock_code"], ascending=[False, True, True])
                return self.send({"total": len(frame), "selected": int(frame.selected.sum()), "rows": records(frame.head(100))})
            scenario = query.get("scenario", "configured")
            if scenario not in SCENARIOS:
                raise ValueError("情景无效")
            return self.send({name: records(pd.read_parquet(p / scenario / f"{name}.parquet", filters=[("date", "=", day)])) for name in ["equity", "orders", "positions"]})

        def do_POST(self):
            expected = f"http://{self.headers.get('Host')}"
            if not self.valid_host() or self.headers.get("X-Research-Token") != token or self.headers.get("Origin") not in (None, expected):
                return self.send({"error": "本地请求校验失败"}, status=403)
            try:
                review_match = re.fullmatch(r"/api/research/tasks/(task-[a-z0-9-]+)/review", self.path)
                campaign_action = re.fullmatch(r"/api/research/campaigns/(campaign-[a-f0-9]{12})/(run|stop)", self.path)
                if not review_match and not campaign_action and self.path not in ("/api/research/campaigns", "/api/research/tasks", "/api/run", "/api/definition", "/api/batch-plan", "/api/batch", "/api/foundations", "/api/regimes", "/api/ml/benchmark", "/api/ml/complete"):
                    return self.send({"error": "不存在的接口"}, status=404)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("请求大小无效")
                payload = json.loads(self.rfile.read(length))
                if self.path in ("/api/ml/benchmark", "/api/ml/complete"):
                    complete = self.path == "/api/ml/complete"
                    if payload != {}:
                        raise ValueError("基准入口不接受隐藏参数，修改设计请通过研究CLI登记")
                    if code_inventory() != boot_code or ml_code() != boot_ml_code:
                        return self.send({"error": "代码已更新，请重启本地服务"}, status=409)
                    with guard:
                        if job["status"] == "running":
                            return self.send({"error": "已有实验运行"}, status=409)
                        from ..mlresearch.contracts import completion_proposal
                        proposal = completion_proposal() if complete else None
                        task = research.create({"title": "机器学习账户与固定研究矩阵" if complete else "日线机器学习固定基准", "question": "模型在真实账户与固定研究对照中表现如何？" if complete else "价量模型是否提供后续排序信息？", "rationale": "用户从工作台启动固定基准",
                            "success_criteria": ["冻结执行并核查全部模型和对照"], "stop_criteria": ["一次登记执行，不搜索测试参数"],
                            "max_experiments": 1, "scope": "固定19账户、两种费用、消融/重训/验证期选择；不修改成交与会计" if complete else "ML信号分析；不修改成交与会计",
                            "baseline_runs": [proposal["spec"]["source_run_id"]] if complete else []})
                        session = research.store.claim(task["id"], "ml-workbench")
                        write_json_path = root / "sessions" / ("ml-ui-" + task["id"] + ".json")
                        from .artifacts import write_json
                        write_json(write_json_path, session)
                        job.clear()
                        job.update(status="running", task_id=task["id"], messages=["登记机器学习基准"])
                    def work_ml():
                        try:
                            from ..mlresearch.contracts import benchmark_proposal
                            ex = research.register(session, proposal if complete else benchmark_proposal())
                            with guard:
                                job["messages"].append("执行冻结训练、后续预测与排序核查")
                            outcome = research.execute(session, ex["id"])
                            if outcome["status"] != "completed":
                                raise ValueError(outcome.get("error", "ML执行失败"))
                            research.submit(session, {"summary": "固定ML基准执行完成，结果待独立复核", "findings": ["已保存模型、逐股预测及全部对照；不根据测试期挑选冠军"],
                                "limitations": ["回顾性时间隔离，数值核查不是独立研究复核" if complete else "回顾性时间隔离，信号收益不是账户收益"], "next_steps": ["复核时间边界、标签覆盖与分时期排序及账户"]})
                            with guard:
                                job.update(status="completed", ml_run_id=outcome["run_id"])
                        except Exception as exc:
                            try:
                                research.store.checkpoint(session, "ML基准失败，保留尝试与日志：" + str(exc), True)
                            except ValueError:
                                pass
                            with guard:
                                job.update(status="failed", error=str(exc))
                    threading.Thread(target=work_ml, daemon=False).start()
                    return self.send({"status": "running", "task_id": task["id"]}, status=202)
                if self.path == "/api/research/tasks":
                    return self.send(research.create(payload))
                if self.path == "/api/research/campaigns":
                    return self.send(campaigns.create(payload))
                if campaign_action:
                    if payload != {}:
                        raise ValueError("总控运行操作不接受隐藏配置")
                    cid, action = campaign_action.groups()
                    if action == "stop":
                        return self.send(campaigns.stop(cid))
                    with guard:
                        campaigns.begin_launch(cid)
                        launch = campaigns.root / cid
                        launch.mkdir(parents=True, exist_ok=True)
                        try:
                            with (launch / "controller.stdout.log").open("a", encoding="utf-8") as out, (launch / "controller.stderr.log").open("a", encoding="utf-8") as err:
                                process = subprocess.Popen([sys.executable, "-X", "utf8", str(project / "tools/controller.py"),
                                    "--project", str(project), "--root", str(root), "run", cid], cwd=project,
                                    stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                            campaigns.update(cid, launcher_pid=process.pid)
                        except Exception as exc:
                            campaigns.update(cid, status="needs_attention", error=str(exc), stage="总控进程未成功启动")
                            raise
                    return self.send(campaigns.get(cid))
                if review_match:
                    if not isinstance(payload, dict) or set(payload)-{'reviewer','decision','rationale','next_task'}:
                        raise ValueError('复核请求字段无效')
                    return self.send(research.review(review_match[1],payload.get('reviewer','user'),
                        payload.get('decision'),payload.get('rationale'),payload.get('next_task')))
                is_foundation = self.path == "/api/foundations"
                is_regime = self.path == "/api/regimes"
                if (is_foundation or is_regime) and payload != {}:
                    raise ValueError("基础策略复现使用完整预登记协议；本入口不接受隐藏参数覆盖")
                is_batch = self.path in ("/api/batch-plan", "/api/batch")
                if is_batch and (not isinstance(payload, dict) or set(payload) - {"spec", "split_date", "parent_run_id"}):
                    raise ValueError("批量请求仅接受 spec、split_date、parent_run_id")
                spec = ResearchSpec.from_dict(payload.get("spec", {}) if is_batch else payload)
                if is_batch:
                    protocol = plan(spec, payload.get("split_date", ""))
                    if self.path == "/api/batch-plan":
                        return self.send(protocol)
                if self.path == "/api/definition":
                    return self.send({"spec": spec.to_dict(), "definition": describe(spec)})
                if code_inventory() != boot_code:
                    return self.send({"error": "代码已更新，请重启本地服务"}, status=409)
                with guard:
                    if job["status"] == "running":
                        return self.send({"error": "已有实验运行"}, status=409)
                    job.clear()
                    job.update(status="running", messages=["准备独立技术研究输入"])
                def progress(message):
                    with guard:
                        job["messages"].append(message)
                def work():
                    try:
                        if is_regime:
                            from .regime_lab import run_suite
                            p = run_suite(project, root=root, progress=progress)
                        elif is_foundation:
                            from .foundation_lab import run_suite
                            p = run_suite(project, root=root, progress=progress)
                        else:
                            p = run_batch(project, spec, payload["split_date"], root=root, progress=progress,
                                          parent_run_id=payload.get("parent_run_id")) if is_batch else run(project, spec, root=root, progress=progress)
                        with guard:
                            job.update(status="completed", **({"regime_id": p.name} if is_regime else {"suite_id": p.name} if is_foundation else {"batch_id": p.name} if is_batch else {"run_id": p.name}))
                    except Exception as exc:
                        with guard:
                            job.update(status="failed", error=str(exc))
                threading.Thread(target=work, daemon=False).start()
                return self.send({"status": "running"}, status=202)
            except (ValueError, TypeError) as exc:
                self.send({"error": str(exc)}, status=400)

    class Server(ThreadingHTTPServer):
        allow_reuse_address = False
        def server_bind(self):
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            super().server_bind()
    return Server(("127.0.0.1", port), Handler)


def serve(project, port=8765, open_browser=True):
    url = f"http://127.0.0.1:{port}"
    try:
        server = make_server(project, port)
    except OSError:
        with urlopen(url + "/api/health", timeout=3) as response:
            health = json.load(response)
        workspace = content_id([str(Path(project).resolve()), str((Path(project) / "data/technical").resolve())])
        if health.get("app") != "technical-research" or health.get("workspace_id") != workspace:
            raise RuntimeError("此端口由旧工作台或其他服务使用，请先关闭该服务")
        if open_browser:
            webbrowser.open(url)
        return
    print(f"技术策略研究: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
