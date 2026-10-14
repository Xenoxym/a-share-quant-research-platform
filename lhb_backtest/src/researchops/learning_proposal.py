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


def validate_learning_template(template):
    """Validate fixed public choices without reading source files."""
    if (not isinstance(template,dict) or not PUBLIC_FIELDS-{"resources","timeout_seconds"}<=set(template)
            or set(template)-PUBLIC_FIELDS or template["kind"]!="alpha_learn"):
        raise ValueError("Public learning proposal template required")
    spec,_,doc=protocol(template["spec"])
    inputs=template["inputs"]
    if not isinstance(inputs,list) or len(inputs)!=len(ROLES) or any(not isinstance(x,dict) for x in inputs) or {x.get("role") for x in inputs}!=ROLES:
        raise ValueError("Exactly seven distinct learning input roles required")
    return spec,doc


def learning_from_matrix(research,matrix_folder,template):
    """Replace only the five matrix references; preserve fixed research choices."""
    spec,doc=validate_learning_template(template)
    inputs=template["inputs"]
    if not isinstance(matrix_folder,str) or not matrix_folder.strip():raise ValueError("Portable project-relative matrix folder required")
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



def learning_from_expression_batches(research,development_id,forecast_id,matrix_folder,template,*,max_rows,max_matrix_cells,max_output_bytes):
    """Materialize once, or reuse a complete source-bound matrix before registration.

    Partial directories never get retried or overwritten. A completed matrix
    is checked against both real frozen producers, not a success flag or mtime.
    """
    from .alpha_execution import verify_completed
    from .matrix_materialization import _output,_budget,matrix_from_registered_expressions
    validate_learning_template(template)
    _budget(max_rows=max_rows,max_matrix_cells=max_matrix_cells,max_output_bytes=max_output_bytes)
    folder=_output(research.project,matrix_folder,allow_existing=True)
    dev=research.store.experiment(development_id);future=research.store.experiment(forecast_id)
    if any(x['status']!='completed' or x['proposal'].get('kind')!='alpha_batch' for x in (dev,future)):
        raise ValueError('Actual completed expression producers required')
    a=dev['proposal']['spec'];b=future['proposal']['spec']
    if (a['schema']!='alpha-feature-batch-v4' or b['schema']!='alpha-feature-batch-v6'
            or len(a['blocks'])!=1 or a['blocks']!=b['blocks'] or a['candidates']!=b['candidates']):
        raise ValueError('Exact prepared development/forecast declarations required')
    audits=[verify_completed(research,x) for x in (dev,future)]
    reused=folder.exists()
    if not reused:
        matrix_from_registered_expressions(research,development_id,forecast_id,matrix_folder,
            max_rows=max_rows,max_matrix_cells=max_matrix_cells,max_output_bytes=max_output_bytes)
    if not folder.is_dir() or {p.name for p in folder.iterdir()}!=set(MATRIX_FILES.values()):
        raise ValueError('Complete original five-file matrix required; partial output retained')
    if sum(p.stat().st_size for p in folder.iterdir())>max_output_bytes:
        raise ValueError('Existing matrix output byte budget exceeded')
    proposal=learning_from_matrix(research,matrix_folder,template)
    meta=json.loads((folder/'definition.json').read_text(encoding='utf-8-sig'))
    provenance=meta.get('source_provenance',{})
    if provenance.get('schema')!='prepared-expression-merge-source-v1':raise ValueError('Original expression merge provenance required')
    for key,ex,audit in zip(('development','forecast'),(dev,future),audits):
        source=provenance.get(key,{}).get('source_provenance',{})
        expected=dict(experiment_id=ex['id'],task_id=ex['task_id'],batch_fingerprint=ex['fingerprint'],output_manifest_sha256=audit['manifest_hash'])
        if any(source.get(k)!=v for k,v in expected.items()):raise ValueError('Saved matrix expression producer differs')
    record=next(iter(a['blocks'].values()))
    paths=verify_inputs(research.project,dev['proposal']['inputs'],a['budget']['max_input_bytes'])
    original=json.loads(paths[record['definition']].read_text(encoding='utf-8-sig'));base_units=original.pop('units')
    expected_units=dict(base_units,**{'expr.'+c['candidate_id']:'ratio' for c in a['candidates']})
    original_hashes={k:digest(paths[record[k]]) for k in ('values','missing','definition','source_receipt')}
    original_hashes.update(registry=digest(paths['registry']),membership=digest(paths['membership']))
    prepared=json.loads(paths[record['receipt']].read_text(encoding='utf-8-sig'))
    rows=pq.ParquetFile(paths['membership']).metadata.num_rows
    if (meta['units']!=expected_units or provenance.get('original_base_metadata')!=original
            or provenance.get('original_prepared')!=prepared or prepared.get('artifact_hashes')!=original_hashes
            or rows>max_rows or rows*len(expected_units)>max_matrix_cells
            or pq.ParquetFile(folder/'assembly_values.parquet').metadata.num_rows!=rows):
        raise ValueError('Saved matrix base, membership or finite budget differs')
    if reused:
        from .matrix_materialization import verify_expression_matrix_content
        verify_expression_matrix_content(research,development_id,forecast_id,matrix_folder,
            max_rows=max_rows,max_matrix_cells=max_matrix_cells)
        # Close comparisons against the same frozen input bytes used above.
        verify_inputs(research.project,proposal['inputs'],proposal['spec']['budget']['max_input_bytes'])
    if research.store.experiment(development_id)!=dev or research.store.experiment(forecast_id)!=future:
        raise ValueError('Expression producer changed during matrix reuse')
    return proposal
