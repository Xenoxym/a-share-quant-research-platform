"""Frozen finite screening worker. No arbitrary model selection, fits or accounts."""
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
from .screen_experiments import inspect_inputs, planned_attempts, same_json
from .resources import WorkerResources
from ..alpharesearch.screening import screen
from ..technical.artifacts import digest, jsonable, write_json

TABLES = ("daily", "selections", "windows", "trials", "redundancy")


def verify_job(job):
    return verify_frozen_job(job, "alpha_screen")


def compute(job):
    job = Path(job).resolve()
    if Path(__file__).resolve().parents[2] != (job / "code").resolve():
        raise ValueError("Screening must run registered frozen modules")
    cfg = verify_job(job); resources = WorkerResources.from_dict(cfg["resources"])
    pa.set_cpu_count(resources.library_threads); pa.set_io_thread_count(1)
    if any(os.environ.get(key) != str(resources.library_threads) for key in THREAD_ENV):
        raise ValueError("Screen numerical thread policy differs")
    if cancel_reason(job) is not None:
        raise RuntimeError("Cancellation requested: " + cancel_reason(job))
    spec, _, info = inspect_inputs(cfg["project"], cfg, include_data=True)
    if not same_json(json.loads((job / "candidate_manifest.json").read_text(encoding="utf-8")), planned_attempts(spec)):
        raise ValueError("Frozen screen intent plan changed")
    folder = job / "screen_result"; folder.mkdir(exist_ok=False)
    attempts = planned_attempts(spec)
    attempts["attempts"] = [dict(c, status="running") for c in spec.to_dict()["channels"]]
    write_json(folder / "attempts.json", attempts)
    result = screen(spec, *info.pop("data"))
    for name in TABLES:
        document = dict(columns=list(getattr(result, name).columns),
            rows=jsonable(getattr(result, name).to_dict("records")))
        projected = len(json.dumps(document, ensure_ascii=False).encode("utf-8")) * 2
        used = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
        if used + projected > resources.max_output_bytes:
            raise ValueError("Screen output byte budget exceeded before table write")
        write_json(folder / (name + ".json"), document)
    write_json(folder / "diagnostic_receipt.json", result.receipt)
    attempts["attempts"] = [dict(c, status="evaluated") for c in spec.to_dict()["channels"]]
    write_json(folder / "attempts.json", attempts)
    verify_job(job)
    from .alpha_experiments import verify_inputs
    verify_inputs(cfg["project"], cfg["inputs"], cfg["spec"]["budget"]["max_input_bytes"])
    manifest = dict(schema="registered-alpha-screen-manifest-v1", experiment_id=job.name,
        task_id=cfg["task_id"], input_hash=digest(job / "input.json"), engine_hash=cfg["engine_hash"],
        candidate_manifest_hash=digest(job / "candidate_manifest.json"),
        artifacts={p.name: digest(p) for p in folder.iterdir() if p.is_file()})
    write_json(folder / "manifest.json", manifest)
    if sum(p.stat().st_size for p in folder.iterdir()) > resources.max_output_bytes:
        raise ValueError("Actual final screen output byte budget exceeded; retain artifacts")
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
        write_json(job / "receipt.json", dict(status="completed", kind="alpha_screen",
            experiment_id=job.name, result_manifest_hash=digest(folder / "manifest.json")))
    except BaseException as exc:
        done.set()
        if watchdog is not None:
            watchdog.join()
        if not (job / "receipt.json").exists():
            write_json(job / "receipt.json", dict(status="failed", kind="alpha_screen",
                experiment_id=job.name, error=str(exc) or type(exc).__name__))
        traceback.print_exc(); raise SystemExit(1)


if __name__ == "__main__":
    main()
