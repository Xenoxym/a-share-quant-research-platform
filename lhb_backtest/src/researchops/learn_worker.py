"""Frozen single-fit learner; recovery audits never refit."""
import argparse
import json
import os
from pathlib import Path
import threading
import traceback

import psutil
import pyarrow as pa

from .alpha_worker import verify_frozen_job, _watchdog
from .alpha_execution import cancel_reason, THREAD_ENV
from .alpha_experiments import verify_inputs
from .learn_experiments import inspect_inputs, planned_attempts, same_json
from .resources import WorkerResources
from ..alpharesearch.learning import LearningRunner
from ..technical.artifacts import digest, jsonable, write_json

FILES = {"predictions.parquet", "learning_receipt.json", "model_metadata.json", "attempts.json"}


def verify_job(job):
    return verify_frozen_job(job, "alpha_learn")


def _used(folder):
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())


def compute(job):
    job = Path(job).resolve()
    if Path(__file__).resolve().parents[2] != (job / "code").resolve():
        raise ValueError("Learning must run registered frozen modules")
    cfg = verify_job(job); resources = WorkerResources.from_dict(cfg["resources"])
    pa.set_cpu_count(resources.library_threads); pa.set_io_thread_count(1)
    if any(os.environ.get(key) != str(resources.library_threads) for key in THREAD_ENV):
        raise ValueError("Learning numerical thread policy differs")
    if cancel_reason(job) is not None:
        raise RuntimeError("Cancellation requested: " + cancel_reason(job))
    spec, model, _, info = inspect_inputs(cfg["project"], cfg, include_data=True)
    data = info.pop("data"); info.pop("prepared")
    if not same_json(info, cfg["inspection"]):
        raise ValueError("Frozen learning input scope changed")
    plan = planned_attempts(spec, model)
    if not same_json(json.loads((job / "candidate_manifest.json").read_text(encoding="utf-8")), plan):
        raise ValueError("Frozen learning fit intent changed")
    folder = job / "learn_result"; folder.mkdir(exist_ok=False)
    attempts = dict(plan, attempts=[dict(plan["attempts"][0], status="running")], ledger=[])
    write_json(folder / "attempts.json", attempts)
    runner = LearningRunner(max_calls=1, max_fit_intents=1)
    try:
        result = runner.run(spec, model, *data)
    except Exception as exc:
        attempts.update(attempts=[dict(plan["attempts"][0], status="failed")],
            ledger=jsonable(runner.ledger), error=str(exc))
        write_json(folder / "attempts.json", attempts)
        raise
    estimate = int(result.predictions.memory_usage(index=False, deep=True).sum())
    if _used(folder) + estimate > resources.max_output_bytes:
        raise ValueError("Learning output byte budget exceeded before prediction write")
    result.predictions.to_parquet(folder / "predictions.parquet", index=False)
    for name, value in (("learning_receipt", result.receipt), ("model_metadata", result.receipt["fitted_model"])):
        projected = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")) * 2
        if _used(folder) + projected > resources.max_output_bytes:
            raise ValueError("Learning output byte budget exceeded before JSON write")
        write_json(folder / (name + ".json"), value)
    attempts.update(attempts=[dict(plan["attempts"][0], status="succeeded")], ledger=jsonable(runner.ledger))
    write_json(folder / "attempts.json", attempts)
    verify_job(job)
    verify_inputs(cfg["project"], cfg["inputs"], cfg["spec"]["budget"]["max_input_bytes"])
    manifest = dict(schema="registered-alpha-learn-manifest-v1", experiment_id=job.name,
        task_id=cfg["task_id"], input_hash=digest(job / "input.json"), engine_hash=cfg["engine_hash"],
        candidate_manifest_hash=digest(job / "candidate_manifest.json"),
        artifacts={p.name: digest(p) for p in folder.iterdir() if p.is_file()})
    write_json(folder / "manifest.json", manifest)
    if _used(folder) > resources.max_output_bytes:
        raise ValueError("Actual learning output byte budget exceeded; retain artifacts")
    return folder


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("job", type=Path)
    job = parser.parse_args().job.resolve(); done = threading.Event(); watchdog = None
    try:
        cfg = verify_job(job)
        startup = dict(schema="alpha-worker-startup-v1", experiment_id=job.name,
            input_hash=digest(job / "input.json"), pid=os.getpid(), created=psutil.Process().create_time())
        temp = job / "worker_started.tmp"; write_json(temp, startup); temp.replace(job / "worker_started.json")
        watchdog = threading.Thread(target=_watchdog, args=(job, cfg, done), daemon=True); watchdog.start()
        folder = compute(job); done.set(); watchdog.join()
        write_json(job / "receipt.json", dict(status="completed", kind="alpha_learn",
            experiment_id=job.name, result_manifest_hash=digest(folder / "manifest.json")))
    except BaseException as exc:
        done.set()
        if watchdog is not None:
            watchdog.join()
        if not (job / "receipt.json").exists():
            write_json(job / "receipt.json", dict(status="failed", kind="alpha_learn",
                experiment_id=job.name, error=str(exc) or type(exc).__name__))
        traceback.print_exc(); raise SystemExit(1)


if __name__ == "__main__":
    main()
