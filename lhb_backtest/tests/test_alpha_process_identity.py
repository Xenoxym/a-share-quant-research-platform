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
