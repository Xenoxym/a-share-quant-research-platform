"""Bind a saved matrix to a fixed learning template without reading feature values.

Full keys, clocks, labels and numerical checks remain in ordinary registration.
This adapter neither computes features nor chooses features using outcomes.
"""
import copy
import json
import pyarrow.parquet as pq

from .alpha_experiments import _path, verify_inputs
from .learn_experiments import protocol
from .screen_experiments import ROLES, same_json
from ..alpharesearch.registry import FeatureRegistry
from ..technical.artifacts import content_id, digest

MATRIX_FILES={"assembly_values":"assembly_values.parquet","assembly_missing":"assembly_missing.parquet",
    "assembly_definition":"definition.json","registry":"registry.json","assembly_receipt":"receipt.json"}
PUBLIC_FIELDS={"kind","hypothesis","expected_observation","falsification","spec","inputs","resources","timeout_seconds"}


def learning_from_matrix(research,matrix_folder,template):
    """Replace only the five matrix references; preserve fixed research choices."""
    if (not isinstance(template,dict) or not PUBLIC_FIELDS-{"resources","timeout_seconds"}<=set(template)
            or set(template)-PUBLIC_FIELDS or template["kind"]!="alpha_learn"):
        raise ValueError("Public learning proposal template required")
    spec,_,doc=protocol(template["spec"])
    inputs=template["inputs"]
    if not isinstance(inputs,list) or len(inputs)!=len(ROLES) or any(not isinstance(x,dict) for x in inputs) or {x.get("role") for x in inputs}!=ROLES:
        raise ValueError("Exactly seven distinct learning input roles required")
    if not isinstance(matrix_folder,str):raise ValueError("Portable project-relative matrix folder required")
    paths={role:_path(research.project,matrix_folder+"/"+name) for role,name in MATRIX_FILES.items()}
    size=sum(p.stat().st_size for p in paths.values())+sum(x.get("bytes",0) for x in inputs if x["role"] not in MATRIX_FILES)
    if size>doc["budget"]["max_input_bytes"]:raise ValueError("Frozen input byte budget exceeded")
    proposal=copy.deepcopy(template)
    for record in proposal["inputs"]:
        if record["role"] in paths:
            path=paths[record["role"]]
            record.update(path=path.relative_to(research.project).as_posix(),sha256=digest(path),bytes=path.stat().st_size)
    verify_inputs(research.project,proposal["inputs"],doc["budget"]["max_input_bytes"])
    registry=FeatureRegistry.from_dict(json.loads(paths["registry"].read_text(encoding="utf-8-sig")))
    definition=json.loads(paths["assembly_definition"].read_text(encoding="utf-8-sig"))
    units=definition["units"];bindings=definition["source_bindings"]
    if definition["registry_version"]!=registry.version_id or not isinstance(bindings,list) or not bindings:
        raise ValueError("Matrix registry or source bindings differ")
    definitions=registry.resolve_bound_definitions(bindings,spec.to_dict()["features"])
    if any(name not in units or units[name]!=item.unit for name,item in definitions.items()):
        raise ValueError("Selected feature was not supplied by matrix")
    footer=pq.ParquetFile(paths["assembly_values"])
    rows=footer.metadata.num_rows
    if rows>spec.to_dict()["max_input_rows"]:raise ValueError("Declared input row cap exceeded")
    expected=dict(schema="screen-assembly-reference-v1",artifacts={x["role"]:x["sha256"] for x in proposal["inputs"] if x["role"] in {"assembly_values","assembly_missing","assembly_definition"}},
        registry_version=registry.version_id,assembly_version=definition["version_id"],source_bindings=bindings,
        rows=rows,upstream_execution_certified=False)
    expected["reference_id"]=content_id(expected)
    if not same_json(json.loads(paths["assembly_receipt"].read_text(encoding="utf-8-sig")),expected):
        raise ValueError("Matrix reference differs from supplied files/declarations")
    # Close the small metadata-read window with the existing byte verifier.
    verify_inputs(research.project,proposal["inputs"],doc["budget"]["max_input_bytes"])
    return proposal
