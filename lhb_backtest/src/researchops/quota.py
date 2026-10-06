"""Read plan quota over the public local app-server RPC; never read credentials."""
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

from .controller import CodexTransport


def read_limits(executable=None,timeout=25):
    executable=executable or CodexTransport().executable
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    proc=subprocess.Popen([str(executable),'--no-daemon','app-server','--listen','stdio://'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',creationflags=flags)
    messages=queue.Queue()
    def receive():
        for line in proc.stdout:
            try:messages.put(json.loads(line))
            except json.JSONDecodeError:pass
        messages.put(None)
    def drain():
        for _ in proc.stderr:pass
    reader=threading.Thread(target=receive,daemon=True);reader.start()
    threading.Thread(target=drain,daemon=True).start()
    def send(value):proc.stdin.write(json.dumps(value)+'\n');proc.stdin.flush()
    def response(request_id):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            try:
                message=messages.get(timeout=max(.01,end-time.monotonic()))
            except queue.Empty as exc:
                raise TimeoutError('额度接口超时') from exc
            if message is None:raise RuntimeError('额度读取app-server已退出')
            if message.get('id')==request_id:
                if 'error' in message:raise RuntimeError('额度接口错误: '+str(message['error'].get('message','unknown')))
                return message['result']
        raise TimeoutError('额度接口超时')
    try:
        send(dict(id=1,method='initialize',params=dict(clientInfo=dict(name='a_share_research_quota',title='Local research quota observer',version='1.0'))))
        response(1);send(dict(method='initialized',params={}))
        send(dict(id=2,method='account/rateLimits/read',params=dict(excludeResetCreditDetails=True)))
        return response(2)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=5)
            except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
        for pipe in [proc.stdin, proc.stdout, proc.stderr]:
            if pipe:pipe.close()


def quota_status(payload,now=None):
    """Require backend permission; percentages/reset times cannot prove recovery."""
    now=time.time() if now is None else now
    buckets=payload.get('rateLimitsByLimitId') or {}
    # Observe the generic Codex bucket; unrelated model-specific meters don't
    # prevent a job that inherits the user's actual selected model.
    primary=buckets.get('codex') or payload.get('rateLimits')
    permission=payload.get('ordinaryUsageAllowed')
    if not isinstance(primary,dict):return dict(known=False,blocked=True,reason='no_codex_bucket',reset_at=None,windows=[],ordinary_usage_allowed=permission if type(permission) is bool else None)
    windows=[];blocking=[]
    for name in ['primary','secondary']:
        w=primary.get(name)
        if isinstance(w,dict) and type(w.get('usedPercent')) in (int,float) and math.isfinite(w['usedPercent']):
            windows.append(dict(name=name,used_percent=w['usedPercent'],window_minutes=w.get('windowDurationMins'),reset_at=w.get('resetsAt')))
            if w['usedPercent']>=100: blocking.append(w)
    reset=[w.get('resetsAt') for w in blocking if type(w.get('resetsAt')) in (int,float) and math.isfinite(w['resetsAt']) and w['resetsAt']>now]
    reached=primary.get('rateLimitReachedType')
    blocked=bool(blocking) or reached is not None or permission is not True
    reason=reached or ('window_exhausted' if blocking else 'backend_denied' if permission is False else 'permission_unavailable' if permission is not True else None)
    return dict(known=bool(windows) and type(permission) is bool,blocked=blocked,reason=reason,
        reset_at=max(reset) if reset and len(reset)==len(blocking) else None,windows=windows,limit_id=primary.get('limitId','codex'),ordinary_usage_allowed=permission if type(permission) is bool else None)
