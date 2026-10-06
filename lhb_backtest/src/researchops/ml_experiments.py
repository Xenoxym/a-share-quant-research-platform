"""ML jobs use the existing lease, budget, frozen executor and recovery protocol."""
import json
import importlib.metadata
import platform
from pathlib import Path
import re
import shutil
import time

from .store import text
from ..technical.artifacts import content_id, digest, verify_artifacts, write_json
from ..mlresearch.contracts import MLSpec

EXECUTOR = '''from pathlib import Path
import json, sys, traceback
code=Path(__file__).resolve().parent
sys.path.insert(0,str(code))
from src.technical.artifacts import write_json, digest, verify_artifacts
from src.mlresearch.runner import run
from src.mlresearch.audit import audit
job=code.parent
try:
    cfg=json.loads((job/'input.json').read_text(encoding='utf-8'))
    verify_artifacts(code,json.loads((job/'code_manifest.json').read_text(encoding='utf-8')))
    folder=run(cfg,progress=lambda m:print(m,flush=True))
    checked=audit(folder,cfg['root'])
    write_json(job/'ml_audit.json',checked)
    write_json(job/'receipt.json',{'status':'completed','kind':'ml','run_id':folder.name,'audit_hash':digest(job/'ml_audit.json')})
except BaseException as exc:
    write_json(job/'receipt.json',{'status':'failed','kind':'ml','error':str(exc)})
    traceback.print_exc()
    sys.exit(1)
'''


def register(research, session, proposal):
    required = {"kind", "hypothesis", "expected_observation", "falsification", "snapshot_id", "spec"}
    if not required <= proposal.keys() or proposal.keys() - (required | {"timeout_seconds"}) or proposal["kind"] != "ml":
        raise ValueError("ML登记字段无效；不接受账户成本情景")
    with research.store.connection() as db:
        research.store.owned(db, session)
    sid = proposal["snapshot_id"]
    if not isinstance(sid, str) or not re.fullmatch(r"[0-9a-f]{24}", sid):
        raise ValueError("ML快照编号无效")
    snapshot = research.root / "snapshots" / sid
    sm = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    verify_artifacts(snapshot, sm)
    spec = MLSpec.from_dict(proposal["spec"])
    if spec.train_start < sm["start"] or spec.test_end > sm["end"] or "fundamentals" not in sm["counts"]:
        raise ValueError("ML基准需要覆盖全部日期、含历史股本的快照")
    timeout = proposal.get("timeout_seconds", 1200)
    if type(timeout) is not int or not 30 <= timeout <= 7200:
        raise ValueError("ML时限必须为30至7200秒")
    sources = sorted((research.project / "src/mlresearch").glob("*.py"))
    sources += [research.project / "src/technical" / name for name in ["__init__.py", "artifacts.py", "foundation_data.py", "signals.py"]]
    extra = {}
    if spec.study == "financial_context":
        from ..mlresearch.financial_study import financial_plan, financial_source_identity
        source = research.root / "ml_runs" / spec.source_run_id
        source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        verify_artifacts(source, source_manifest)
        if source_manifest["snapshot_id"] != sid or source_manifest["spec"].get("study") != "complete":
            raise ValueError("固定组合必须复用同一快照的完整研究")
        original = MLSpec.from_dict(source_manifest["spec"])
        for key, value in original.to_dict().items():
            if key not in {"name", "study", "source_run_id"} and getattr(spec, key) != value:
                raise ValueError("固定组合不得修改源日期、资格或模型定义")
        sources += [p for p in (research.project / "src/technical").rglob("*") if p.suffix in {".py", ".js", ".html", ".css"}]
        sources += [research.project / "tools/verify_technical.py", research.project / "tools/research.py"]
        sources += [research.project / "src/researchops" / n for n in ["ml_experiments.py", "service.py", "store.py", "__init__.py"]]
        sources = sorted(set(sources))
        extra = dict(source_manifest_hash=digest(source / "manifest.json"), study_plan=financial_plan(spec), source_identity=financial_source_identity(source, research.root))
    if spec.study == "fixed_blend":
        from ..mlresearch.fixed_blend import blend_plan, source_identity
        source = research.root / "ml_runs" / spec.source_run_id
        source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        verify_artifacts(source, source_manifest)
        if source_manifest["snapshot_id"] != sid or source_manifest["spec"].get("study") != "complete":
            raise ValueError("固定组合必须复用同一快照的完整研究")
        original = MLSpec.from_dict(source_manifest["spec"])
        for key, value in original.to_dict().items():
            if key not in {"name", "study", "source_run_id"} and getattr(spec, key) != value:
                raise ValueError("固定组合不得修改源日期、资格或模型定义")
        sources += [p for p in (research.project / "src/technical").rglob("*") if p.suffix in {".py", ".js", ".html", ".css"}]
        sources += [research.project / "tools/verify_technical.py", research.project / "tools/research.py"]
        sources += [research.project / "src/researchops" / n for n in ["ml_experiments.py", "service.py", "store.py", "__init__.py"]]
        sources = sorted(set(sources))
        extra = dict(source_manifest_hash=digest(source / "manifest.json"), study_plan=blend_plan(spec), source_identity=source_identity(source, research.root))
    if spec.study == "universe":
        from ..mlresearch.universe import universe_plan
        source = research.root / "ml_runs" / spec.source_run_id
        source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        verify_artifacts(source, source_manifest)
        if source_manifest["spec"].get("study") != "complete" or sm.get("universe_policy") != "local-A-price-learning-v1;not-full-market;no-BSE":
            raise ValueError("股票范围研究需要已核验完整案例和本地A股学习快照")
        original = MLSpec.from_dict(source_manifest["spec"])
        for key, value in original.to_dict().items():
            if key not in {"name", "study", "source_run_id"} and getattr(spec, key) != value:
                raise ValueError("股票范围对照固定原日期及模型参数")
        if sm.get("account_snapshot_id") != source_manifest["snapshot_id"]:
            raise ValueError("账户主板快照与原完整案例不同")
        sources += [p for p in (research.project / "src/technical").rglob("*") if p.suffix in {".py", ".js", ".html", ".css"}]
        sources += [research.project / "tools/verify_technical.py"]
        sources = sorted(set(sources))
        extra = {"source_manifest_hash": digest(source / "manifest.json"), "study_plan": universe_plan(spec)}
    if spec.study == "complete":
        from ..mlresearch.study import study_plan
        source = research.root / "ml_runs" / spec.source_run_id
        source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        verify_artifacts(source, source_manifest)
        if source_manifest["snapshot_id"] != sid or source_manifest["spec"].get("study", "signal") != "signal":
            raise ValueError("复用输入必须是相同快照的原信号基准")
        original = MLSpec.from_dict(source_manifest["spec"])
        # A different date, pool or original model setup requires a different design.
        for key, value in original.to_dict().items():
            if key not in {"name", "study", "source_run_id"} and getattr(spec, key) != value:
                raise ValueError("完整研究必须沿用原基准的日期、候选资格及模型参数")
        sources += [p for p in (research.project / "src/technical").rglob("*") if p.suffix in {".py", ".js", ".html", ".css"}]
        sources += [research.project / "tools/verify_technical.py"]
        sources = sorted(set(sources))
        extra = {"source_manifest_hash": digest(source / "manifest.json"), "study_plan": study_plan(spec)}
    inventory = {path.relative_to(research.project).as_posix(): digest(path) for path in sources}
    if not (research.project / "src/mlresearch/runner.py").exists():
        raise ValueError("当前项目缺少ML执行代码")
    p = {key: text(proposal[key], key) for key in ["hypothesis", "expected_observation", "falsification"]}
    p.update(kind="ml", spec=spec.to_dict(), snapshot_id=sid, snapshot_manifest_hash=digest(snapshot / "manifest.json"),
        timeout_seconds=timeout, engine_inventory=inventory, engine_hash=content_id(inventory),
        executor_hash=content_id(EXECUTOR), evaluation_scope="retrospective_time_holdout", submitted_by=session["worker"])
    p.update(extra)
    p["environment"] = {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "joblib", "threadpoolctl"]}}
    if spec.study in {"complete", "universe", "fixed_blend", "financial_context"}:
        p["environment"]["packages"].update({name: importlib.metadata.version(name) for name in ["duckdb", "pyyaml"]})
    canonical = json.loads(json.dumps(p))
    canonical["spec"].pop("name")
    for key in ["submitted_by", "hypothesis", "expected_observation", "falsification", "timeout_seconds"]:
        canonical.pop(key)
    experiment = research.store.register(session, p, content_id(canonical))
    job = research.root / "worker_jobs" / experiment["id"]
    if (job / "ready.json").exists():
        return experiment
    if job.exists():
        raise ValueError("ML冻结尝试已存在，不能覆盖")
    p = experiment["proposal"]
    code = job / "code"
    code.mkdir(parents=True)
    try:
        (code / "src").mkdir()
        (code / "src/__init__.py").write_text("", encoding="utf-8")
        for name, sha in p["engine_inventory"].items():
            target = code / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(research.project / name, target)
            if digest(target) != sha:
                raise ValueError("ML代码在冻结期间变化")
        (code / "execute.py").write_text(EXECUTOR, encoding="utf-8")
        write_json(job / "code_manifest.json", {"artifacts": {v.relative_to(code).as_posix(): digest(v) for v in code.rglob("*") if v.is_file()}})
        write_json(job / "input.json", dict(p, project=str(research.project), root=str(research.root), task_id=session["task_id"], experiment_id=experiment["id"]))
        write_json(job / "ready.json", {"input_hash": digest(job / "input.json"), "code_manifest_hash": digest(job / "code_manifest.json")})
    except Exception as exc:
        with research.store.connection(True) as db:
            db.execute("UPDATE experiments SET status='failed',error=?,updated=? WHERE id=?", (str(exc), time.time(), experiment["id"]))
            research.store.event(db, session["task_id"], session["worker"], "freeze_failed", {"experiment_id": experiment["id"], "error": str(exc)})
        raise
    return experiment


def verify_completed(research, experiment):
    job = research.root / "worker_jobs" / experiment["id"]
    ready = json.loads((job / "ready.json").read_text(encoding="utf-8"))
    if digest(job / "input.json") != ready["input_hash"] or digest(job / "code_manifest.json") != ready["code_manifest_hash"]:
        raise ValueError("ML冻结输入变化")
    verify_artifacts(job / "code", json.loads((job / "code_manifest.json").read_text(encoding="utf-8")))
    p = experiment["proposal"]
    cfg = json.loads((job / "input.json").read_text(encoding="utf-8"))
    if any(cfg.get(k) != v for k, v in p.items()) or cfg["experiment_id"] != experiment["id"] or cfg["task_id"] != experiment["task_id"]:
        raise ValueError("ML冻结输入与登记不同")
    folder = research.root / "ml_runs" / experiment["run_id"]
    if digest(folder / "manifest.json") != experiment["result"]["manifest_hash"]:
        raise ValueError("ML结果清单与已核验收据不同")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    verify_artifacts(folder, manifest)
    if manifest["code_hash"] != p["engine_hash"] or manifest["spec"] != p["spec"] or manifest["snapshot_id"] != p["snapshot_id"] or manifest["snapshot_manifest_sha256"] != p["snapshot_manifest_hash"]:
        raise ValueError("ML登记版本不一致")
    snapshot = research.root / "snapshots" / p["snapshot_id"]
    if digest(snapshot / "manifest.json") != p["snapshot_manifest_hash"]:
        raise ValueError("ML快照清单变化")
    verify_artifacts(snapshot, json.loads((snapshot / "manifest.json").read_text(encoding="utf-8")))
    if digest(job / "ml_audit.json") != experiment["result"]["audit_hash"]:
        raise ValueError("ML核查证据变化")
    if p["spec"].get("study") in {"complete", "universe", "fixed_blend", "financial_context"}:
        from ..mlresearch.study_audit import verify_links
        verify_links(folder, research.root)


def recover(research, experiment, receipt):
    eid = experiment["id"]
    job = research.root / "worker_jobs" / eid
    try:
        rid = receipt["run_id"]
        if not isinstance(rid, str) or not re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{8}", rid):
            raise ValueError("ML运行编号无效")
        ready = json.loads((job / "ready.json").read_text(encoding="utf-8"))
        if digest(job / "input.json") != ready["input_hash"] or digest(job / "code_manifest.json") != ready["code_manifest_hash"]:
            raise ValueError("ML冻结输入已变化")
        verify_artifacts(job / "code", json.loads((job / "code_manifest.json").read_text(encoding="utf-8")))
        folder = research.root / "ml_runs" / rid
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        verify_artifacts(folder, manifest)
        p = experiment["proposal"]
        if manifest["code_hash"] != p["engine_hash"] or manifest["spec"] != p["spec"] or manifest["snapshot_id"] != p["snapshot_id"] or manifest["snapshot_manifest_sha256"] != p["snapshot_manifest_hash"]:
            raise ValueError("ML运行与登记版本不一致")
        snapshot = research.root / "snapshots" / p["snapshot_id"]
        if digest(snapshot / "manifest.json") != p["snapshot_manifest_hash"]:
            raise ValueError("ML快照清单变化")
        verify_artifacts(snapshot, json.loads((snapshot / "manifest.json").read_text(encoding="utf-8")))
        if digest(job / "ml_audit.json") != receipt["audit_hash"]:
            raise ValueError("ML核查证据变化")
        checked = json.loads((job / "ml_audit.json").read_text(encoding="utf-8"))
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        if not checked.get("verified") or checked["run_id"] != rid or result["research_context"] != {"task_id": experiment["task_id"], "experiment_id": eid}:
            raise ValueError("ML结果归属或核查无效")
        return research.store.finish(eid, "completed", run_id=rid, result={"kind": "ml", "verified": True,
            "name": result["name"], "summaries": result["summaries"], "counts": result["counts"],
            **({"accounts": result["accounts"], "study_plan": result["study_plan"]} if p["spec"].get("study") in {"complete", "universe", "fixed_blend", "financial_context"} else {}),
            "verification_scope": checked["verification_scope"], "manifest_hash": digest(folder / "manifest.json"),
            "audit_hash": receipt["audit_hash"], "audit": checked})
    except Exception as exc:
        research.store.finish(eid, "failed", error=str(exc))
        raise
