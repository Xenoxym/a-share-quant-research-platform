"""Real CLI handoff acceptance, not two independently reasoning AI workers.

Each command is a fresh process. One real historical experiment uses an existing
snapshot and two independently simulated cost accounts. Never edits old results.
"""

import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.technical.artifacts import write_json


def main():
    proof = PROJECT / "research/worker_platform_20260928"
    proof.mkdir(exist_ok=True)
    work = PROJECT / "data/technical/validation/v6"
    work.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    checks = []

    def cli(*args, ok=True):
        result = subprocess.run(
            [sys.executable, str(PROJECT / "tools/research.py"), *map(str, args)],
            cwd=PROJECT,
            capture_output=True,
            encoding="utf-8",
            timeout=1200,
        )
        value = json.loads(result.stdout)
        assert value["ok"] == ok, (args, result.stdout, result.stderr)
        assert (result.returncode == 0) == ok
        return value.get("data", value)

    task = dict(
        title="V6 工程验收：跨进程研究接力",
        question="冻结的实验能否由新 Worker 接续，并进入统一结果页？",
        rationale="验证任务协议与执行隔离，不评估新策略优劣；角色名称由同一施工 Agent 设置。",
        success_criteria=[
            "新进程读取交接并执行原登记定义",
            "同一真实数据快照上的两个成本账户通过独立核查",
            "旧凭据和重复启动被拒绝",
        ],
        stop_criteria=["代码、区间或账户核查不一致则停止发布"],
        max_experiments=1,
        tags=["工程验收", "非策略发现"],
    )
    write_json(work / "task.json", task)
    t = cli("create", "--file", work / "task.json")
    tid = t["id"]
    a, b = work / f"session-a-{stamp}.json", work / f"session-b-{stamp}.json"
    cli("claim", tid, "--worker", "acceptance-planner", "--session", a)
    parent = PROJECT / "data/technical/runs/20260928T033859-574fc6dd"
    spec = json.loads((parent / "spec.json").read_text(encoding="utf-8"))
    spec["strategy"].update(name="V6接力验收 · 小市值季度股本口径", family="size")
    spec["experiment"].update(start_date="2024-01-02", end_date="2024-03-15")
    manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    proposal = dict(
        hypothesis="登记后交接不改变代码、快照及回测定义",
        expected_observation="2024年一季度指定区间完成两个费用情景的独立账本核查",
        falsification="执行代码或参数身份不同、区间超界、任一账户核查失败均否定本轮工程验收",
        snapshot_id=manifest["snapshot_id"],
        spec=spec,
    )
    write_json(work / "proposal.json", proposal)
    ex = cli("register", "--session", a, "--file", work / "proposal.json")
    duplicate = cli("register", "--session", a, "--file", work / "proposal.json")
    assert duplicate["id"] == ex["id"]
    cli(
        "handoff",
        "--session",
        a,
        "--note",
        "已固定真实数据快照和指定区间。继续运行已登记实验，勿重复登记；这是流程验收，不是策略发现。",
    )
    cli("heartbeat", "--session", a, ok=False)
    cli("claim", tid, "--worker", "acceptance-executor", "--session", b)
    context = cli("context", tid)
    assert context["task"]["checkpoint"]["worker"] == "acceptance-planner"
    assert len(context["task"]["experiments"]) == 1
    result = cli("run", ex["id"], "--session", b)
    assert result["status"] == "completed", result
    assert result["result"]["verified"] and result["result"]["data_end"] == "2024-03-15"
    cli("run", ex["id"], "--session", b, ok=False)
    checks.extend(
        [
            "fresh-process context survives handoff; old claim rejected",
            "duplicate registration is idempotent; second execution rejected",
            "real snapshot, frozen code, bounded dates, both cost accounts independently audited",
        ]
    )
    report = dict(
        summary="跨进程接力验收通过；未开展新的有效性筛选。",
        findings=[
            f"登记 {ex['id']} 由新CLI进程执行为 {result['run_id']}，原输入与代码哈希一致。",
            "真实行情账户的逐日现金、股数、价格、费用与权益通过独立核查。",
        ],
        limitations=[
            "同一施工Agent通过两个自报Worker角色验收，不是独立AI间的研究结论验证。",
            "当前历史已经查看；本实验仅验证平台接力，不比较新策略优劣。",
        ],
        next_steps=["由新研究Worker领取V5后续任务，阅读原有基线证据后提出实验。"],
    )
    write_json(work / "submission.json", report)
    cli("submit", "--session", b, "--file", work / "submission.json")
    final = cli(
        "review",
        tid,
        "--reviewer",
        "acceptance-checker",
        "--decision",
        "stop",
        "--rationale",
        "完成工程验收。由同一施工Agent执行角色分离测试，不能解释为独立研究者复核。",
    )
    assert final["status"] == "completed"
    context = cli("export", tid, "--output", proof / "handoff_context.json")
    assert "token_hash" not in json.dumps(context) and '"token"' not in json.dumps(context)
    evidence = dict(
        status="pass",
        task_id=tid,
        experiment_id=ex["id"],
        run_id=result["run_id"],
        checks=checks,
        scope="real independent CLI processes, simulated worker roles operated by one construction agent",
        result=result["result"],
    )
    write_json(proof / "acceptance.json", evidence)
    print(json.dumps(evidence, ensure_ascii=True))


if __name__ == "__main__":
    main()
