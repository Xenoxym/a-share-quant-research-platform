"""Reuse all declared training formulas in a separate prepared forecast partition.

Proposal construction does no formula evaluation. Registration authenticates the
completed producer through the existing default audit. Workers use frozen source
references; this is not a blind holdout or vendor-history certification.
"""
from copy import deepcopy

from ..technical.artifacts import digest

SOURCE_FILES = {
    "input_role": ("producer_input", "input.json"),
    "audit_role": ("producer_audit", "alpha_audit.json"),
    "manifest_role": ("producer_manifest", "alpha_result/manifest.json"),
}
NUMERIC_CORE = ("src/alpharesearch/operators.py", "src/alpharesearch/dsl.py",
                "src/alpharesearch/contracts.py", "src/alpharesearch/features/base.py",
                "src/alpharesearch/prepared_assembly.py", "src/alpharesearch/registry.py")


def _producer(research, experiment_id):
    actual = research.store.experiment(experiment_id)
    if (actual["status"] != "completed" or actual["proposal"].get("kind") != "alpha_batch"
            or actual["proposal"]["spec"]["schema"] != "alpha-feature-batch-v4"
            or actual.get("result", {}).get("verified") is not True):
        raise ValueError("Actual completed prepared development producer required")
    return actual


def prepare_registered_forecast(research, experiment_id, *, name, budget=None,
                                resources=None, timeout_seconds=1200):
    """Build one proposal from all original candidates; no Screen or label inputs.

    This only reads metadata. Full producer verification occurs at registration,
    using the existing audit cache, before a numerical experiment is registered.
    """
    actual = _producer(research, experiment_id)
    old = actual["proposal"]
    doc = deepcopy(old["spec"])
    doc.update(schema="alpha-feature-batch-v6", name=name,
               execution_partition="forecast_features",
               forecast_warmup_start=doc["ranges"]["holdout_start"])
    if budget is not None:
        doc["budget"] = deepcopy(budget)
    source = dict(producer_experiment_id=actual["id"], producer_task_id=actual["task_id"],
                  producer_fingerprint=actual["fingerprint"])
    inputs = deepcopy(old["inputs"])
    job = research.root / "worker_jobs" / actual["id"]
    for field, (role, filename) in SOURCE_FILES.items():
        file = job / filename
        source[field] = role
        inputs.append(dict(role=role, path=file.relative_to(research.project).as_posix(),
                           sha256=digest(file), bytes=file.stat().st_size))
    doc["forecast_source"] = source
    return dict(kind="alpha_batch", spec=doc, inputs=inputs,
                resources=deepcopy(resources if resources is not None else old["resources"]),
                timeout_seconds=timeout_seconds,
                hypothesis="All original declared prepared formulas, forecast dates only",
                expected_observation="Exact forecast membership and missing reasons; zero training recomputation",
                falsification="Changed formula/source or missing keys; preserve failure without hidden selection")


def verify_registered_forecast_producer(research, proposal):
    """One actual DB check plus the standard cached audit, before registration."""
    source = proposal["spec"]["forecast_source"]
    actual = _producer(research, source["producer_experiment_id"])
    if (source["producer_task_id"] != actual["task_id"]
            or source["producer_fingerprint"] != actual["fingerprint"]):
        raise ValueError("Forecast producer DB identity differs")
    from ..researchops.alpha_execution import verify_completed
    audit = verify_completed(research, actual)
    if audit.get("failed_candidates") != 0:
        raise ValueError("All producer candidates must complete")
    records = {r["role"]: r for r in proposal["inputs"]}
    job = research.root / "worker_jobs" / actual["id"]
    for field, (_, filename) in SOURCE_FILES.items():
        record = records.get(source[field])
        file = job / filename
        if (record is None or record["sha256"] != digest(file)
                or record["bytes"] != file.stat().st_size):
            raise ValueError("Forecast reference differs from actual completed producer")
    # Same AST alone cannot establish same numerical semantics after operator changes.
    for filename in NUMERIC_CORE:
        if digest(research.project / filename) != actual["proposal"]["engine_inventory"][filename]:
            raise ValueError("Numerical operator implementation differs from producer")
    return actual
