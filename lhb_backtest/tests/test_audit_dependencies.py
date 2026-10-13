"""Real frozen feature audits prove unrelated code edits do not rerun numbers."""
import importlib,json,sys,uuid
from contextlib import contextmanager
from types import ModuleType
import pytest
from src.researchops.audit_cache import numerical_audit,_binding
from src.researchops.audit_dependencies import audit_dependency_files
from src.researchops.screen_output_audit import _FrozenFinder
from tests.test_alpha_batch_registration import setup as base_setup
from src.researchops.resources import WorkerResources

def setup(*args,**kwargs):
    result=base_setup(*args,**kwargs)
    result[2]["resources"]=WorkerResources(min_available_memory_bytes=1<<28).to_dict()
    return result

@contextmanager
def source_auditor(ops):
    prefix='_dependency_test_'+uuid.uuid4().hex
    package=ModuleType(prefix);package.__path__=[str(ops.project)];sys.modules[prefix]=package
    finder=_FrozenFinder(prefix,ops.project);sys.meta_path.insert(0,finder)
    try:yield importlib.import_module(prefix+'.src.researchops.alpha_output_audit')
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name==prefix or name.startswith(prefix+'.'):del sys.modules[name]

def test_real_audit_reuses_after_unrelated_ui_and_plan_edits_then_forced_audit_passes(tmp_path):
    ops,session,proposal,_,_=setup(tmp_path);ex=ops.register(session,proposal);ex=ops.execute(session,ex['id'])
    with source_auditor(ops) as module:
        calls=[];original=module.verify_registered
        def counted(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
        module.verify_registered=counted
        files,policy=audit_dependency_files(ops.project,'alpha_batch',module.audit_output)
        assert policy['policy']=='builtin_import_closure_v1'
        assert not any(p.name=='workflow.py' or 'workbench' in p.parts for p in files)
        assert numerical_audit(ops,ex,module.audit_output)['verified'];assert len(calls)==1
        ui=ops.project/'src/workbench/unrelated_display.py';ui.parent.mkdir(exist_ok=True);ui.write_text('TITLE="new UI"\n',encoding='utf-8')
        planner=ops.project/'src/researchops/workflow.py';planner.write_text(planner.read_text(encoding='utf-8')+'\n# unrelated plan display\n',encoding='utf-8')
        assert numerical_audit(ops,ex,module.audit_output)['verified'];assert len(calls)==1
        assert numerical_audit(ops,ex,module.audit_output,force=True)['verified'];assert len(calls)==2
        resource=ops.project/'src/researchops/resources.py';resource.write_text(resource.read_text(encoding='utf-8')+'\n# related validation source\n',encoding='utf-8')
        with pytest.raises(ValueError,match="fresh process"):
            numerical_audit(ops,ex,module.audit_output)
        assert len(calls)==2
    # A fresh namespace loads the actual changed source; related edits miss.
    with source_auditor(ops) as fresh:
        original=fresh.verify_registered
        def counted_fresh(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
        fresh.verify_registered=counted_fresh
        assert numerical_audit(ops,ex,fresh.audit_output)["verified"];assert len(calls)==3

def test_related_operator_or_frozen_bytes_change_binding(tmp_path):
    ops,session,proposal,_,_=setup(tmp_path);ex=ops.register(session,proposal);ex=ops.execute(session,ex['id'])
    with source_auditor(ops) as module:
        before=_binding(ops,ex,auditor=module.audit_output)
        operator=ops.project/'src/alpharesearch/operators.py';operator.write_text(operator.read_text(encoding='utf-8')+'\n# changed operator declaration\n',encoding='utf-8')
        with pytest.raises(ValueError,match="fresh process"):_binding(ops,ex,auditor=module.audit_output)
    with source_auditor(ops) as module:
        assert _binding(ops,ex,auditor=module.audit_output)!=before
        before=_binding(ops,ex,auditor=module.audit_output)
        frozen=ops.root/'worker_jobs'/ex['id']/'code/src/alpharesearch/operators.py';frozen.write_text(frozen.read_text(encoding='utf-8')+'\n# changed frozen bytes\n',encoding='utf-8')
        assert _binding(ops,ex,auditor=module.audit_output)!=before

@pytest.mark.parametrize('fault',['unknown_dynamic','wildcard','unresolved_local','stale_loaded_function'])
def test_unprovable_import_scope_reverts_to_whole_source(tmp_path,fault):
    ops,_,_,_,_=setup(tmp_path)
    with source_auditor(ops) as module:
        path=ops.project/'src/researchops/resources.py';source=path.read_text(encoding='utf-8')
        if fault=='unknown_dynamic':source+='\nimport importlib\ndef future_plugin(name):return importlib.import_module(name)\n'
        if fault=='wildcard':source+='\nfrom .store import *\n'
        if fault=='unresolved_local':source+='\nfrom .nonexistent_future_backend import function\n'
        if fault=='stale_loaded_function':source=source.replace('Sampled worker RSS limit exceeded','changed RSS error',1)
        path.write_text(source,encoding='utf-8')
        files,policy=audit_dependency_files(ops.project,'alpha_batch',module.audit_output)
        assert policy['policy']=='whole_src_fallback_v1'
        assert len(files)==len(list((ops.project/'src').rglob('*.py')))

def test_unknown_auditor_keeps_former_full_scope(tmp_path):
    ops,_,_,_,_=setup(tmp_path)
    files,policy=audit_dependency_files(ops.project,'alpha_batch',lambda *a:None)
    assert policy['policy']=='whole_src_fallback_v1'
    assert ops.project/'src/researchops/workflow.py' in files


def test_runtime_code_hash_is_stable_and_detects_changed_constants():
    from src.researchops.audit_dependencies import runtime_code_hash
    namespace={}
    exec(compile("def function(x):\n    return x + 17\n", "synthetic.py", "exec"),namespace)
    function=namespace["function"];before=runtime_code_hash(function.__code__)
    for _ in range(100):assert function(2)==19
    assert runtime_code_hash(function.__code__)==before
    changed=function.__code__.replace(co_consts=tuple(23 if x==17 else x for x in function.__code__.co_consts))
    assert runtime_code_hash(changed)!=before


def test_callback_replacement_during_audit_is_rejected(tmp_path):
    ops,session,proposal,_,_=setup(tmp_path)
    ex=ops.execute(session,ops.register(session,proposal)["id"])
    namespace={}
    exec("def auditor(research,experiment):\n    auditor.__code__=replacement.__code__\n    return {'verified':True}\ndef replacement(research,experiment):\n    return {'verified':True}\n",namespace)
    with pytest.raises(ValueError,match="changed during numerical validation"):
        numerical_audit(ops,ex,namespace["auditor"])


@pytest.mark.parametrize("mutation",["function","property"])
def test_loaded_callable_changes_cannot_hide_in_fallback(tmp_path,mutation):
    ops,session,proposal,_,_=setup(tmp_path);ex=ops.execute(session,ops.register(session,proposal)["id"])
    with source_auditor(ops) as module:
        assert numerical_audit(ops,ex,module.audit_output)["verified"]
        root=module.__name__.split(".src.")[0]
        if mutation=="function":
            target=importlib.import_module(root+".src.researchops.resources").WorkerResources.to_dict
            target.__code__=target.__code__.replace(co_consts=target.__code__.co_consts+("stale-marker",))
        else:
            target=importlib.import_module(root+".src.alpharesearch.registry").FeatureRegistry.version_id.fget
            target.__code__=(lambda self:"changed_registry").__code__
        with pytest.raises(ValueError,match="fresh process"):
            numerical_audit(ops,ex,module.audit_output)


def test_declared_constant_change_invalidates_runtime_identity(tmp_path):
    ops,_,_,_,_=setup(tmp_path)
    with source_auditor(ops) as module:
        before=audit_dependency_files(ops.project,"alpha_batch",module.audit_output)[1]
        root=module.__name__.split(".src.")[0]
        registry=importlib.import_module(root+".src.alpharesearch.registry")
        registry.DOMAINS=set()
        after=audit_dependency_files(ops.project,"alpha_batch",module.audit_output)[1]
        assert before!=after and after["reusable"]
        registry.DOMAINS=object()
        after=audit_dependency_files(ops.project,"alpha_batch",module.audit_output)[1]
        assert not after["reusable"] and any(x.endswith("DOMAINS") for x in after["unprovable_runtime_state"])


def test_fresh_namespace_preserves_cache_key_after_unrelated_source_edit(tmp_path):
    ops,session,proposal,_,_=setup(tmp_path);ex=ops.execute(session,ops.register(session,proposal)["id"])
    with source_auditor(ops) as module:before=_binding(ops,ex,auditor=module.audit_output)
    ui=ops.project/"src/workbench/unrelated_display.py";ui.parent.mkdir(exist_ok=True)
    ui.write_text("TITLE='changed UI'\n",encoding="utf-8")
    with source_auditor(ops) as module:
        assert _binding(ops,ex,auditor=module.audit_output)==before


def test_statically_bound_but_not_globally_loaded_source_change_is_rejected(tmp_path):
    ops,_,_,_,_=setup(tmp_path)
    with source_auditor(ops) as module:
        files,policy=audit_dependency_files(ops.project,"alpha_batch",module.audit_output)
        loaded=set(policy["runtime_implementations"])
        path=next(p for p in files if ".".join(p.relative_to(ops.project).with_suffix("").parts) not in loaded and p.name!="__init__.py")
        path.write_text(path.read_text(encoding="utf-8")+"\n# static dependency changed\n",encoding="utf-8")
        assert audit_dependency_files(ops.project,"alpha_batch",module.audit_output)[1]["stale_loaded_source"]


def test_fresh_alias_cannot_mask_a_stale_loaded_copy(tmp_path):
    ops,_,_,_,_=setup(tmp_path)
    with source_auditor(ops) as old:
        root=old.__name__.split(".src.")[0]
        target=importlib.import_module(root+".src.researchops.resources").WorkerResources.to_dict
        target.__code__=target.__code__.replace(co_consts=target.__code__.co_consts+("old-mutated-copy",))
        with source_auditor(ops):
            policy=audit_dependency_files(ops.project,"alpha_batch",old.audit_output)[1]
            assert policy["stale_loaded_source"] and not policy["reusable"]


def test_new_namespace_uses_its_own_package_snapshot(tmp_path):
    ops,_,_,_,_=setup(tmp_path);path=ops.project/"src/__init__.py"
    path.write_text(path.read_text(encoding="utf-8")+"\n# current package\n",encoding="utf-8")
    with source_auditor(ops) as module:
        path=ops.project/"src/researchops/resources.py"
        path.write_text(path.read_text(encoding="utf-8")+"\nfrom .store import *\n",encoding="utf-8")
    with source_auditor(ops) as module:
        policy=audit_dependency_files(ops.project,"alpha_batch",module.audit_output)[1]
        assert policy["policy"]=="whole_src_fallback_v1" and not policy["stale_loaded_source"] and policy["reusable"]
