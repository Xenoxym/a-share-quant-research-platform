"""Exact helper identities: direct Windows interpreter, venv launcher, and refusals."""
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from src.researchops import alpha_execution as execution


def mocked(monkeypatch, *, platform="nt", launcher=False, children=None, parent_created=10):
    command = ["/venv/python.exe" if launcher else "/base/python.exe", "-B", "-m", "declared.worker", "job"]
    monkeypatch.setattr(execution, "os", SimpleNamespace(name=platform, path=os.path,
                                                       environ={"SystemRoot": "/windows"}))
    monkeypatch.setattr(execution, "sys", SimpleNamespace(executable=command[0], _base_executable="/base/python.exe"))
    proc = SimpleNamespace(pid=100, poll=lambda: None)
    parent = SimpleNamespace(create_time=lambda: parent_created, children=lambda **kwargs: children or [])
    monkeypatch.setattr(execution.psutil, "Process", lambda pid: parent)
    return proc, command


def child(role="console", *, pid=101, ppid=100, executable=None, args=None):
    path = "/windows/System32/conhost.exe" if role == "console" else "/base/python.exe"
    tail = ["0x4"] if role == "console" else ["-B", "-m", "declared.worker", "job"]
    return SimpleNamespace(pid=pid, ppid=lambda: ppid, exe=lambda: executable or path,
                           cmdline=lambda: [executable or path] + (tail if args is None else args),
                           create_time=lambda: 11)


@pytest.mark.parametrize("launcher,roles", [
    (False, []), (False, ["console"]), (True, ["interpreter"]),
    (True, ["interpreter", "console"]), (True, ["console", "interpreter"]),
])
def test_exact_direct_and_venv_identity_roles(monkeypatch, launcher, roles):
    children = [child(r, pid=101+i) for i, r in enumerate(roles)]
    proc, command = mocked(monkeypatch, launcher=launcher, children=children)
    chain = execution._interpreter_chain(proc, 10, command)
    expected = [{"pid": c.pid, "created": 11,
                 "role": "console_host" if r == "console" else "interpreter"}
                for c, r in zip(children, roles)]
    assert chain == sorted(expected, key=lambda c: c["role"])


@pytest.mark.parametrize("fault", [
    "foreign_parent", "wrong_console_path", "wrong_console_args", "wrong_python_args",
    "python_child_without_venv", "console_on_linux", "duplicate_console",
    "too_many_children", "reused_parent_pid",
])
def test_refuses_child_identity_relaxation(monkeypatch, fault):
    launcher, platform, created = True, "nt", 10
    children = [child("interpreter")]
    if fault == "foreign_parent": children = [child(ppid=999)]
    if fault == "wrong_console_path": children = [child(executable="/tmp/conhost.exe")]
    if fault == "wrong_console_args": children = [child(args=["0x8"])]
    if fault == "wrong_python_args": children = [child("interpreter", args=["-c", "different"])]
    if fault == "python_child_without_venv": launcher = False
    if fault == "console_on_linux": platform = "posix"; launcher = False; children = [child()]
    if fault == "duplicate_console": children = [child(pid=101), child(pid=102)]
    if fault == "too_many_children": children = [child(pid=101+i) for i in range(3)]
    if fault == "reused_parent_pid": created = 9
    proc, command = mocked(monkeypatch, launcher=launcher, platform=platform,
                           parent_created=created, children=children)
    with pytest.raises(ValueError):
        execution._interpreter_chain(proc, 10, command)


@pytest.mark.skipif(os.name != "nt", reason="Windows exact hidden-console process integration")
@pytest.mark.parametrize("direct", [False, True])
def test_actual_windows_interpreters_account_for_known_helpers(monkeypatch, direct):
    executable = sys._base_executable if direct else sys.executable
    if direct:
        monkeypatch.setattr(execution, "sys", SimpleNamespace(executable=executable, _base_executable=executable))
    command = [executable, "-c", "import time; time.sleep(1)"]
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        created = psutil.Process(proc.pid).create_time()
        chain = execution._interpreter_chain(proc, created, command)
        launch = dict(pid=proc.pid, created=created, interpreter_chain=chain)
        assert execution._sample(launch) > 0
        if direct:
            assert all(c["role"] == "console_host" for c in chain)
        deadline = time.monotonic()+5
        while proc.poll() is None and time.monotonic() < deadline:
            execution._sample(launch)
            time.sleep(.02)
        assert proc.wait(timeout=3) == 0
        while execution._alive(launch) and time.monotonic() < deadline:
            time.sleep(.02)
        assert not execution._alive(launch)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3)


def test_self_watchdog_freezes_helpers_and_counts_their_memory(monkeypatch):
    from src.researchops import alpha_worker as worker

    monkeypatch.setattr(worker, "os", SimpleNamespace(name="nt", environ={"SystemRoot": "/windows"}))
    helper = child(); helper.memory_info = lambda: SimpleNamespace(rss=20)
    children = [helper]
    current = SimpleNamespace(pid=100, children=lambda **kwargs: children,
                              memory_info=lambda: SimpleNamespace(rss=100))
    rss, accepted = worker._watchdog_memory(current, None)
    assert rss == 120 and accepted == (dict(pid=101, created=11, role="console_host"),)
    assert worker._watchdog_memory(current, accepted) == (120, accepted)
    children.clear()
    assert worker._watchdog_memory(current, accepted) == (100, accepted)
    changed = child(pid=102); changed.memory_info = helper.memory_info
    children.append(changed)
    with pytest.raises(ValueError, match="identity changed"):
        worker._watchdog_memory(current, accepted)
    with pytest.raises(ValueError, match="after first sample"):
        worker._watchdog_memory(current, ())


@pytest.mark.parametrize("fault", ["python", "foreign_parent", "wrong_path", "wrong_args",
                                  "multiple", "linux", "invalid_created"])
def test_self_watchdog_refuses_any_unregistered_numeric_or_wrong_helper(monkeypatch, fault):
    from src.researchops import alpha_worker as worker

    monkeypatch.setattr(worker, "os", SimpleNamespace(name="posix" if fault=="linux" else "nt",
                                                    environ={"SystemRoot": "/windows"}))
    helper = child(); helper.memory_info = lambda: SimpleNamespace(rss=20)
    if fault=="python": helper = child("interpreter")
    if fault=="foreign_parent": helper = child(ppid=999)
    if fault=="wrong_path": helper = child(executable="/tmp/conhost.exe")
    if fault=="wrong_args": helper = child(args=["0x8"])
    if fault=="invalid_created": helper.create_time = lambda: float("nan")
    children = [helper, child(pid=102)] if fault=="multiple" else [helper]
    current = SimpleNamespace(pid=100, children=lambda **kwargs: children,
                              memory_info=lambda: SimpleNamespace(rss=100))
    with pytest.raises(ValueError):
        worker._watchdog_memory(current, None)


@pytest.mark.skipif(os.name != "nt", reason="Windows actual direct interpreter feature worker")
def test_actual_direct_python_frozen_worker_completes_without_watchdog_false_failure(tmp_path, monkeypatch):
    import sysconfig
    from tests.test_alpha_batch_registration import setup

    ops, session, proposal, _, _ = setup(tmp_path)
    experiment = ops.register(session, proposal)
    real_popen = subprocess.Popen
    base = sys._base_executable
    purelib = sysconfig.get_path("purelib")
    def launch(command, **kwargs):
        assert command[0] == base
        # Local base Python uses the already-installed venv dependencies.
        # CI's direct Python has its installed site packages naturally.
        env = dict(kwargs["env"])
        env["PYTHONPATH"] = os.pathsep.join([env["PYTHONPATH"], purelib])
        return real_popen(command, **dict(kwargs, env=env))
    monkeypatch.setattr(execution, "sys", SimpleNamespace(executable=base, _base_executable=base))
    monkeypatch.setattr(execution, "subprocess", SimpleNamespace(Popen=launch,
                        CREATE_NO_WINDOW=subprocess.CREATE_NO_WINDOW, STDOUT=subprocess.STDOUT))
    result = ops.execute(session, experiment["id"])
    job = ops.root/"worker_jobs"/experiment["id"]
    assert result["status"] == "completed", (result.get("error"), (job/"execution.log").read_text())
    assert not list((ops.root/".alpha_slots").glob("slot*.json"))
    assert ops.recover(experiment["id"])["status"] == "completed"
