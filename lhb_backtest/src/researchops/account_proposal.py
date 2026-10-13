"""Build a scored-account proposal from an audited saved model; never refit."""
import copy
from pathlib import Path

from .account_experiments import protocol, ROLES
from .alpha_experiments import verify_inputs
from .completed_learning_identity import verify_completed_learning_identity
from ..technical.artifacts import digest

PUBLIC_FIELDS={"kind","hypothesis","expected_observation","falsification","spec","inputs","resources","timeout_seconds"}
REQUIRED_FIELDS=PUBLIC_FIELDS-{"resources","timeout_seconds"}


def validate_account_template(template):
    if not isinstance(template,dict) or not REQUIRED_FIELDS <= set(template) or set(template)-PUBLIC_FIELDS:
        raise ValueError("Public account proposal template required")
    if template["kind"]!="alpha_account" or (template["spec"].get("schema") if isinstance(template["spec"],dict) else None) not in {"registered-alpha-account-v2","registered-alpha-account-v3"}:
        raise ValueError("Saved-score account template v2/v3 required")
    protocol(template["spec"])
    if not isinstance(template["inputs"],list) or len(template["inputs"])!=len(ROLES) or any(not isinstance(x,dict) for x in template["inputs"]) or {x.get("role") for x in template["inputs"]}!=ROLES:
        raise ValueError("Exactly four distinct account input roles required")


def account_from_learning(research, learning_experiment_id, template):
    """Bind the actual producer's bytes; references, fees and budgets stay explicit.

    This is proposal preparation, not account registration or account validation.
    Registration subsequently checks full portfolio coverage and snapshot contents.
    """
    validate_account_template(template)
    verify_completed_learning_identity(research,learning_experiment_id)
    proposal=copy.deepcopy(template)
    folder=research.root/"worker_jobs"/learning_experiment_id/"learn_result"
    for record in proposal["inputs"]:
        name={"predictions":"predictions.parquet","learning_receipt":"learning_receipt.json"}.get(record["role"])
        if name is None:continue
        path=folder/name
        # Preserve the existing project-relative data/research input contract.
        record.update(path=path.resolve().relative_to(research.project.resolve()).as_posix(),sha256=digest(path),bytes=path.stat().st_size)
    proposal["spec"]["score_source"]={"kind":"registered_learning","experiment_id":learning_experiment_id,
        "verification_mode":"reuse_completed_identity_v1"}
    name="account_"+learning_experiment_id.replace("-","_")
    proposal["spec"]["portfolio"]["name"]=name
    proposal["spec"]["account"]["strategy"]["name"]=name
    verify_inputs(research.project,proposal["inputs"],proposal["spec"]["budget"]["max_input_bytes"])
    protocol(proposal["spec"])
    return proposal
