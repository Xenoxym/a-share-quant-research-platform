"""Bounded structural DAG reuse and content-bound, non-pickle numerical cache."""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
import zipfile

import numpy as np
import pandas as pd

from .dsl import Expr,ExpressionLimits,compile_expression,verify_compiled
from .operators import ExpressionEvaluator,_Array,PRESENT,REASONS
from .registry import implementation_hashes
from ..technical.artifacts import content_id,digest


class CachedExpressionEvaluator(ExpressionEvaluator):
    """Exact structural subtrees share results; algebraic equivalence is not inferred.

    An evaluator pins each prepared leaf when first read. Callers must treat bound
    blocks as immutable during a batch. Actual prepared values/reasons are hashed,
    so a false/reused declared materialization ID cannot select a prior result.
    """
    def __init__(self,*args,cache_root,memory_cache_bytes=64_000_000,
                 disk_cache_bytes=512_000_000,prepared_leaf_bytes=128_000_000,**kwargs):
        for value in (memory_cache_bytes,disk_cache_bytes,prepared_leaf_bytes):
            if type(value) is not int or value<1:raise ValueError('Finite positive cache budgets required')
        requested=Path(cache_root)
        if requested.is_symlink():raise ValueError('Cache root cannot be a symbolic link')
        self.cache_root=requested.resolve();self.cache_root.mkdir(parents=True,exist_ok=True)
        self.memory_budget=memory_cache_bytes;self.disk_budget=disk_cache_bytes;self.leaf_budget=prepared_leaf_bytes
        self._memory=OrderedDict();self._memory_bytes=0;self._prepared_bytes=0;self._input_ids={}
        self.stats={k:0 for k in ('memory_hits','disk_hits','computed_nodes','disk_writes','write_budget_skips','write_busy_skips','memory_evictions')}
        self.engine={'schema':'alpha-dag-cache-v1','implementation':implementation_hashes(('src/alpharesearch/cache.py','src/alpharesearch/operators.py','src/alpharesearch/dsl.py','src/alpharesearch/contracts.py','src/alpharesearch/features/base.py','src/alpharesearch/registry.py')),
            'numpy':np.__version__,'pandas':pd.__version__,'python':platform.python_version(),'platform':platform.platform(),'byteorder':sys.byteorder}
        super().__init__(*args,**kwargs)

    def _leaf(self,key,definition_id):
        token=(key,definition_id)
        if token in self._leaf_cache:return self._leaf_cache[token]
        domain=self.registry.resolve(key,definition_id).domain
        estimated=(len(self.days) if domain=='market_day' else int(np.prod(self.shape)))*9
        if self._prepared_bytes+estimated>self.leaf_budget:raise ValueError('Prepared-leaf memory budget exceeded before allocation')
        arr=super()._leaf(key,definition_id);arr.values[np.isnan(arr.values)]=np.nan
        arr.values.setflags(write=False);arr.reasons.setflags(write=False)
        self._prepared_bytes+=arr.values.nbytes+arr.reasons.nbytes
        binding=self.leaves[key]
        sha=hashlib.sha256();sha.update(np.ascontiguousarray(arr.values).tobytes());sha.update(np.ascontiguousarray(arr.reasons).tobytes())
        keys=['trade_date','stock_code'] if domain=='stock_day' else ['trade_date']
        clock_frame=binding.block.values[keys+['observed_end','known_at']].set_index(keys).reindex(self.index if domain=='stock_day' else self.days)
        for name in ('observed_end','known_at'):
            clock=pd.to_datetime(clock_frame[name],utc=True).dt.as_unit('ns')
            sha.update(clock.array.asi8.tobytes())
        self._input_ids[token]={'key':key,'definition_id':definition_id,'materialization_id':binding.materialization_id,
            'source_id':binding.block.metadata['source_id'],'version_id':binding.block.metadata['version_id'],'vintage':binding.block.metadata['vintage'],
            'prepared_shape':list(arr.values.shape),'prepared_domain':arr.domain,'prepared_content_sha256':sha.hexdigest()}
        return arr

    def _request(self,node):
        references=set()
        def refs(n):
            if n['op']=='ref':references.add((n['parameters']['key'],n['parameters']['definition_id']))
            for child in n['args']:refs(child)
        refs(node)
        for key,definition_id in sorted(references):self._leaf(key,definition_id)
        limits=ExpressionLimits(**self._active_limits)
        compiled=compile_expression(Expr.from_dict(node,max_nodes=limits.max_nodes,max_depth=limits.max_depth),self.registry,require_alpha=False,limits=limits)
        shape={'scalar':(1,1),'market_day':(len(self.days),1),'stock_day':self.shape}[compiled.domain]
        request={'schema':'alpha-node-cache-v1','ast':node,'context_id':self.context_id,'engine':self.engine,
            'inputs':[self._input_ids[t] for t in sorted(references)],'domain':compiled.domain,'shape':list(shape)}
        return content_id(request),request

    def _folder(self,key):
        folder=self.cache_root/key
        if folder.is_symlink() or not folder.resolve().is_relative_to(self.cache_root):raise ValueError('Cache entry path is not confined to its root')
        return folder

    @staticmethod
    def _validate_array(arr,request):
        shape=tuple(request['shape'])
        if arr.domain!=request['domain'] or arr.values.shape!=shape or arr.reasons.shape!=shape:raise ValueError('Cached array shape/domain changed')
        if arr.values.dtype!=np.dtype('float64') or arr.reasons.dtype!=np.dtype('uint8'):raise ValueError('Cached array dtype changed')
        if (arr.reasons>=len(REASONS)).any() or np.isinf(arr.values).any() or not np.array_equal(np.isfinite(arr.values),arr.reasons==PRESENT):raise ValueError('Cached values and reasons disagree')
        arr.values.setflags(write=False);arr.reasons.setflags(write=False)
        return arr

    def _read_disk(self,key,request):
        folder=self._folder(key)
        if not folder.exists():return None
        manifest_path=folder/'manifest.json';file=folder/'arrays.npz'
        if not manifest_path.is_file() or not file.is_file() or manifest_path.is_symlink() or file.is_symlink():raise ValueError('Incomplete/linked cache entry')
        if manifest_path.stat().st_size>200_000:raise ValueError('Cache manifest too large')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if set(manifest)!={'request','cache_id','artifact_sha256'} or manifest['request']!=json.loads(json.dumps(request)) or manifest['cache_id']!=key:raise ValueError('Cache request identity changed')
        bound=int(np.prod(request['shape']))*9+20_000
        if file.stat().st_size>bound or digest(file)!=manifest['artifact_sha256']:raise ValueError('Cache artifact content or size changed')
        with zipfile.ZipFile(file) as archive:
            entries=archive.infolist()
            if {e.filename for e in entries}!={'values.npy','reasons.npy'} or len(entries)!=2 or any(e.compress_type!=zipfile.ZIP_STORED for e in entries) or sum(e.file_size for e in entries)>bound:raise ValueError('Unexpected cached array container')
        with np.load(file,allow_pickle=False) as loaded:
            arr=_Array(loaded['values'],loaded['reasons'],request['domain'])
        return self._validate_array(arr,request)

    @contextmanager
    def _write_lock(self):
        lock=self.cache_root/'.cache-write.lock'
        try:fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        except FileExistsError:yield False;return
        try:
            os.write(fd,str(os.getpid()).encode('ascii'));os.close(fd);yield True
        finally:
            if lock.is_symlink() or not lock.resolve().is_relative_to(self.cache_root):raise ValueError('Cache lock path changed')
            lock.unlink(missing_ok=True)

    def _write_disk(self,key,request,arr):
        with self._write_lock() as owned:
            if not owned:self.stats['write_busy_skips']+=1;return
            existing=self._read_disk(key,request)
            if existing is not None:return
            used=0
            for item in self.cache_root.rglob('*'):
                if item.name=='.cache-write.lock':continue
                if item.is_symlink() or not item.resolve().is_relative_to(self.cache_root):raise ValueError('Cache budget scan found linked or escaped entry')
                if item.is_file():used+=item.stat().st_size
            estimated=arr.values.nbytes+arr.reasons.nbytes+len(json.dumps(request).encode('utf-8'))+4096
            if used+estimated>self.disk_budget:self.stats['write_budget_skips']+=1;return
            with TemporaryDirectory(prefix='.building-',dir=self.cache_root) as temp:
                folder=Path(temp);assert folder.resolve().is_relative_to(self.cache_root)
                np.savez(folder/'arrays.npz',values=arr.values,reasons=arr.reasons)
                manifest={'request':request,'cache_id':key,'artifact_sha256':digest(folder/'arrays.npz')}
                (folder/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True),encoding='utf-8')
                actual=sum(p.stat().st_size for p in folder.iterdir())
                if used+actual>self.disk_budget:self.stats['write_budget_skips']+=1;return
                target=self._folder(key);assert not target.exists();os.replace(folder,target)
            self.stats['disk_writes']+=1

    def _remember(self,key,arr):
        size=arr.values.nbytes+arr.reasons.nbytes
        if size>self.memory_budget:return
        while self._memory and self._memory_bytes+size>self.memory_budget:
            _,removed=self._memory.popitem(last=False);self._memory_bytes-=removed.values.nbytes+removed.reasons.nbytes;self.stats['memory_evictions']+=1
        self._memory[key]=arr;self._memory_bytes+=size

    def _visit(self,node):
        key,request=self._request(node)
        if key in self._memory:
            self._memory.move_to_end(key);self.stats['memory_hits']+=1;return self._memory[key]
        arr=self._read_disk(key,request)
        if arr is not None:self.stats['disk_hits']+=1
        else:
            arr=self._validate_array(super()._visit(node),request);self.stats['computed_nodes']+=1;self._write_disk(key,request,arr)
        self._remember(key,arr);return arr

    def evaluate(self,compiled):
        self._active_limits=json.loads(compiled.limits_json)
        result=super().evaluate(compiled)
        inputs=[self._input_ids[(self.registry.resolve(d.key,d.definition_id).key,d.definition_id)] for d in self.registry.definitions if d.definition_id in compiled.dependencies]
        result.metadata['version_id']=content_id({'base_version':result.metadata['version_id'],'engine':self.engine,'prepared_inputs':inputs})
        result.metadata['prepared_inputs']=inputs
        result.metadata['cache']={'engine':self.engine,'stats':self.stats.copy(),'memory_bytes':self._memory_bytes,'memory_budget':self.memory_budget,
            'prepared_leaf_bytes':self._prepared_bytes,'prepared_leaf_budget':self.leaf_budget,'disk_budget':self.disk_budget,
            'integrity':'local cache content/contract checks; not adversarial authenticity or historical source certification'}
        return result

    def evaluate_many(self,compiled_expressions,*,max_expressions=64):
        if type(max_expressions) is not int or max_expressions<1:raise ValueError('Finite expression-count budget required')
        batch=[]
        for expression in compiled_expressions:
            if len(batch)>=max_expressions:raise ValueError('Expression-count budget exceeded before batch execution')
            batch.append(verify_compiled(expression.to_dict(),self.registry))
        references=set()
        for compiled in batch:
            for definition_id in compiled.dependencies:
                definition=next(d for d in self.registry.definitions if d.definition_id==definition_id);references.add((definition.key,definition_id))
        # Snapshot needed leaves before the first result. No fitting or selection.
        for key,definition_id in sorted(references):self._leaf(key,definition_id)
        for compiled in batch:yield compiled,self.evaluate(compiled)
