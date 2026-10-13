"""Conservative local import closure for the four built-in numerical auditors.

Unknown callbacks/imports retain the former whole-src policy. Frozen namespaces
are already covered by the complete job/code binding in audit_cache.
"""
import ast
import importlib.util
import inspect
from pathlib import Path
import hashlib
import json
from types import CodeType
import sys
from types import ModuleType
from enum import Enum
import re

ROOTS={"alpha_batch":"src.researchops.alpha_output_audit",
       "alpha_screen":"src.researchops.screen_output_audit",
       "alpha_learn":"src.researchops.learn_output_audit",
       "alpha_account":"src.researchops.account_output_audit"}


def runtime_code_hash(code):
    """Hash immutable code fields, independent of marshal's interning state."""
    def constant(value):
        if isinstance(value,CodeType):return ["code",payload(value)]
        if isinstance(value,tuple):return ["tuple",[constant(x) for x in value]]
        if isinstance(value,frozenset):
            return ["frozenset",sorted([constant(x) for x in value],key=lambda x:json.dumps(x,sort_keys=True))]
        if value is None or value is Ellipsis or isinstance(value,(str,bytes,int,float,complex)):
            return [type(value).__name__,repr(value)]
        raise ValueError("Unsupported runtime code constant")
    def payload(item):
        names=("co_argcount","co_posonlyargcount","co_kwonlyargcount","co_nlocals","co_stacksize",
            "co_flags","co_code","co_names","co_varnames","co_filename","co_name","co_qualname",
            "co_firstlineno","co_linetable","co_exceptiontable","co_freevars","co_cellvars")
        fields={name:constant(getattr(item,name)) for name in names if hasattr(item,name)}
        fields["co_consts"]=[constant(x) for x in item.co_consts]
        return fields
    return hashlib.sha256(json.dumps(payload(code),sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()


def _source_namespace(name):
    if name=="src" or name.endswith(".src"):return name
    return name.split(".src.",1)[0]+".src" if ".src." in name else "src"


def _runtime_binding(project,files,nodes):
    """Bind known runtime functions/properties/constants in both scope policies."""
    identities={};unknown=[];stale=False;unproven=False
    def state(value):
        if value is None or value is Ellipsis or isinstance(value,(str,bytes,bool,int,float,complex)):
            return [type(value).__name__,repr(value)]
        if isinstance(value,(list,tuple,set,frozenset)):
            items=[state(x) for x in value]
            if isinstance(value,(set,frozenset)):items.sort(key=lambda x:json.dumps(x,sort_keys=True))
            return [type(value).__name__,items]
        if isinstance(value,dict):
            items=[[state(k),state(v)] for k,v in value.items()]
            return ["dict",sorted(items,key=lambda x:json.dumps(x[0],sort_keys=True))]
        if isinstance(value,re.Pattern):return ["regex",state(value.pattern),value.flags]
        if isinstance(value,Enum):return ["enum",type(value).__qualname__,value.name,state(value.value)]
        if inspect.isfunction(value):return ["function",runtime_code_hash(value.__code__)]
        if inspect.isclass(value):return ["class",value.__qualname__]
        raise TypeError("Unprovable declared runtime state")
    instances=[(name,item) for name,value in nodes.items() for item in (value if isinstance(value,list) else [value])]
    for canonical,loaded in instances:
        source=Path(loaded.__file__).resolve()
        if source not in files:continue
        namespace=_source_namespace(loaded.__name__)
        package=sys.modules.get(namespace);snapshot=getattr(package,"_SOURCE_SHA256",None)
        if snapshot is None:unproven=True
        elif snapshot.get(source.relative_to(project/"src").as_posix())!=hashlib.sha256(source.read_bytes()).hexdigest():stale=True
        try:compiled=compile(source.read_text(encoding="utf-8-sig"),str(loaded.__file__),"exec",dont_inherit=True,optimize=sys.flags.optimize)
        except (SyntaxError,UnicodeError):
            stale=True;compiled=compile("",str(loaded.__file__),"exec")
        code_objects={}
        def gather(code):
            code_objects[(getattr(code,"co_qualname",code.co_name),code.co_firstlineno)]=code
            for value in code.co_consts:
                if isinstance(value,CodeType):gather(value)
        gather(compiled)
        functions={};constants={}
        def function(key,value):
            nonlocal stale
            original=inspect.unwrap(value);code=original.__code__
            declared=key.removesuffix(".fget").removesuffix(".fset").removesuffix(".fdel")
            if any(qualname==declared for qualname,_ in code_objects) and code.co_filename!=str(loaded.__file__):stale=True
            if code.co_filename==str(loaded.__file__) and code_objects.get((getattr(code,"co_qualname",code.co_name),code.co_firstlineno))!=code:stale=True
            entry={"code":runtime_code_hash(value.__code__)}
            try:entry.update(defaults=state(value.__defaults__),kwdefaults=state(value.__kwdefaults__))
            except (TypeError,RecursionError):unknown.append(canonical+"."+key+".defaults")
            functions[key]=entry
        def constant(key,value):
            try:constants[key]=state(value)
            except (TypeError,RecursionError):unknown.append(canonical+"."+key)
        for key,value in vars(loaded).items():
            # Import-time snapshot is projected by file above, never a whole-tree key.
            if key.isupper() and key!="_SOURCE_SHA256":constant(key,value)
            if inspect.isfunction(value):function(key,value)
            elif inspect.isclass(value) and value.__module__==loaded.__name__:
                for key2,method in vars(value).items():
                    if key2.isupper():constant(key+"."+key2,method)
                    if isinstance(method,property):
                        for accessor in ("fget","fset","fdel"):
                            fn=getattr(method,accessor)
                            if fn is not None:function(key+"."+key2+"."+accessor,fn)
                    else:
                        if isinstance(method,(classmethod,staticmethod)):method=method.__func__
                        if inspect.isfunction(method):function(key+"."+key2,method)
        identities.setdefault(canonical,[]).append({"functions":functions,"constants":constants})
    for entries in identities.values():entries.sort(key=lambda x:json.dumps(x,sort_keys=True))
    return {"runtime_implementations":identities,"stale_loaded_source":stale,
        "source_origin_unproven":unproven,"unprovable_runtime_state":sorted(unknown),
        "reusable":not(stale or unproven or unknown)}


def audit_dependency_files(project,kind,auditor):
    project=project.resolve();files=sorted(p for p in (project/"src").rglob("*.py") if "__pycache__" not in p.parts)
    def fallback(reason,*,force_stale=False):
        nodes={}
        for loaded in list(sys.modules.values()):
            filename=getattr(loaded,"__file__",None)
            if filename and Path(filename).resolve() in files:
                canonical=".".join(Path(filename).resolve().relative_to(project).with_suffix("").parts)
                nodes.setdefault(canonical,[]).append(loaded)
        bound=_runtime_binding(project,files,nodes)
        # Unknown external callbacks keep the former whole-source policy; their
        # own code is independently bound by audit_cache.
        if not nodes:bound["reusable"]=True
        if force_stale:bound.update(stale_loaded_source=True,reusable=False)
        return files,{"policy":"whole_src_fallback_v1","reason":reason,**bound}
    expected=ROOTS.get(kind);name=getattr(auditor,"__module__","") or ""
    runtime=sys.modules.get(name)
    if (expected is None or not (name==expected or name.endswith("."+expected))
            or runtime is None or getattr(runtime,"audit_output",None) is not auditor
            or getattr(auditor,"__qualname__",None)!="audit_output"):
        return fallback("unknown auditor")
    root=project.joinpath(*expected.split('.')).with_suffix('.py')
    if not getattr(runtime,"__file__",None) or root!=Path(runtime.__file__).resolve() or Path(auditor.__code__.co_filename).resolve()!=root:
        return fallback("auditor outside registered source root")
    modules={'.'.join(p.relative_to(project).with_suffix('').parts):p for p in files}
    for key,path in list(modules.items()):
        if key.endswith('.__init__'):modules[key[:-9]]=path
    pending=[expected];seen=set();selected=set();runtime_nodes={expected:runtime}
    while pending:
        module=pending.pop()
        if module in seen:continue
        if module not in modules:return fallback("unresolved local import")
        seen.add(module);path=modules[module];selected.add(path)
        if not path.resolve().is_relative_to(project) or any(p.is_symlink() or (hasattr(p,"is_junction") and p.is_junction()) for p in (path,*path.parents) if p!=project and project in p.parents):raise ValueError("Linked or escaped audit source refused")
        try:tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        except (SyntaxError,UnicodeError):return fallback("unparseable current source")
        package=module if path.name=='__init__.py' else module.rsplit('.',1)[0]
        frozen_suffix='.'+module
        guarded_frozen_import=(module in ROOTS.values() and any(
            isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_FrozenFinder'
            and len(n.args)==2 and ast.dump(n.args[0])==ast.dump(ast.Name(id='prefix',ctx=ast.Load()))
            and ast.dump(n.args[1])==ast.dump(ast.BinOp(left=ast.Name(id='job',ctx=ast.Load()),op=ast.Div(),right=ast.Constant(value='code')))
            for n in ast.walk(tree)))
        import_aliases=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module=='importlib':
                import_aliases.update(a.asname or a.name for a in node.names if a.name=='import_module')
        for node in ast.walk(tree):
            targets=[]
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith("src/") and node.value.endswith(".py"):
                declared=project/node.value
                if declared not in files:return fallback("missing declared source dependency")
                targets=['.'.join(Path(node.value).with_suffix('').parts)]
            if isinstance(node,ast.Import):targets=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom):
                try:base=importlib.util.resolve_name('.'*node.level+(node.module or ''),package) if node.level else node.module or ''
                except (ImportError,ValueError):return fallback("unresolved relative import")
                targets=[base]+[base+'.'+x.name for x in node.names]
                if base.startswith('src.') and base not in modules:return fallback("missing local module")
                if any(x.name=='*' for x in node.names):return fallback("wildcard import")
            elif isinstance(node,ast.Call):
                dynamic=(isinstance(node.func,ast.Attribute) and node.func.attr in {'import_module','__import__'}) or (isinstance(node.func,ast.Name) and node.func.id in import_aliases|{'__import__'})
                if dynamic:
                    argument=node.args[0] if node.args else None
                    approved=(guarded_frozen_import and isinstance(node.func,ast.Attribute)
                        and isinstance(node.func.value,ast.Name) and node.func.value.id=='importlib' and node.func.attr=='import_module'
                        and isinstance(argument,ast.BinOp) and isinstance(argument.op,ast.Add)
                        and isinstance(argument.left,ast.Name) and argument.left.id=='prefix'
                        and isinstance(argument.right,ast.Constant) and argument.right.value==frozen_suffix)
                    if not approved:return fallback("unknown dynamic import")
            for target in targets:
                if target in modules:pending.append(target)
                elif target.startswith('src.') and isinstance(node,ast.Import):return fallback("missing local module")
                for i in range(1,len(target.split('.'))):
                    parent='.'.join(target.split('.')[:i])
                    if parent in modules:pending.append(parent)
        # Union loaded local imports with source imports. An edited source file
        # must not hide a still-loaded old helper before the next fresh process.
        loaded=runtime_nodes.get(module)
        if loaded is not None:
            for value in vars(loaded).values():
                ref=value if isinstance(value,ModuleType) else sys.modules.get(getattr(value,'__module__',None)) if inspect.isfunction(value) or inspect.isclass(value) else None
                filename=getattr(ref,'__file__',None)
                if not filename:continue
                path_ref=Path(filename).resolve()
                if path_ref in files:
                    canonical='.'.join(path_ref.relative_to(project).with_suffix('').parts)
                    if canonical.endswith('.__init__'):canonical=canonical[:-9]
                    runtime_nodes[canonical]=ref;pending.append(canonical)
    runtime_binding=_runtime_binding(project,files,runtime_nodes)
    package=sys.modules.get(_source_namespace(runtime.__name__))
    snapshot=getattr(package,"_SOURCE_SHA256",None)
    if snapshot is None:runtime_binding.update(source_origin_unproven=True,reusable=False)
    elif any(snapshot.get(path.relative_to(project/"src").as_posix())!=hashlib.sha256(path.read_bytes()).hexdigest() for path in selected):
        runtime_binding.update(stale_loaded_source=True,reusable=False)
    if runtime_binding["stale_loaded_source"]:return fallback("loaded implementation differs from current file",force_stale=True)
    return sorted(selected),{"policy":"builtin_import_closure_v1","root":expected,
        "current_source_files":len(selected),**runtime_binding}
