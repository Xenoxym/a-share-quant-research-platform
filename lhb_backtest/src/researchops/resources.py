"""Finite local reservations; OS locks survive a parent crash without stale guards."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

import psutil

from ..technical.artifacts import write_json


class GuardBusy(RuntimeError):
    """Transient contention; it is not a numerical experiment failure."""


def _available_commit_bytes():
    """Windows process-available commit headroom; other platforms keep RSS policy."""
    if os.name != "nt":
        return None
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    try:
        query = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
        query.argtypes = [ctypes.POINTER(MemoryStatusEx)]
        query.restype = ctypes.c_int
        ok = query(ctypes.byref(status))
    except (OSError, AttributeError) as exc:
        raise RuntimeError("Windows commit headroom unavailable; batch remains planned") from exc
    if not ok or status.ullTotalPageFile == 0 or status.ullAvailPageFile > status.ullTotalPageFile:
        raise RuntimeError("Windows commit headroom query failed; batch remains planned")
    # ullAvailPageFile is available commitment, not unused pagefile disk space.
    return int(status.ullAvailPageFile)


@dataclass(frozen=True)
class WorkerResources:
    library_threads: int = 2
    max_process_rss_bytes: int = 2_147_483_648
    min_available_memory_bytes: int = 1_073_741_824
    max_output_bytes: int = 1_073_741_824
    output_scope: str = "development_only"

    def to_dict(self):
        return dict(schema="alpha-worker-resources-v1", slots_per_research_root=2,
                    library_threads=self.library_threads,
                    max_process_rss_bytes=self.max_process_rss_bytes,
                    min_available_memory_bytes=self.min_available_memory_bytes,
                    max_output_bytes=self.max_output_bytes, memory_policy="sampled_soft_stop",
                    output_scope=self.output_scope)

    @classmethod
    def from_dict(cls, value):
        expected = cls().to_dict()
        if not isinstance(value, dict) or set(value)!=set(expected):
            raise ValueError("Strict worker resource fields required")
        for key in ("schema", "slots_per_research_root", "memory_policy"):
            if type(value[key]) is not type(expected[key]) or value[key]!=expected[key]:
                raise ValueError("Unsupported worker resource policy")
        if type(value["output_scope"]) is not str or value["output_scope"] not in {"development_only", "scored_account_result"}:
            raise ValueError("Unsupported worker output scope")
        bounds = {"library_threads":(1, 2), "max_process_rss_bytes":(134_217_728, 17_179_869_184),
                  "min_available_memory_bytes":(268_435_456, 17_179_869_184),
                  "max_output_bytes":(1_048_576, 107_374_182_400)}
        for key,(low,high) in bounds.items():
            if type(value[key]) is not int or not low<=value[key]<=high:
                raise ValueError("Worker resource limit is not a bounded integer: "+key)
        return cls(output_scope=value["output_scope"], **{key:value[key] for key in bounds})


def process_alive(pid, created):
    """Creation identity avoids mistaking a reused PID for our worker."""
    if type(pid) is not int or pid<1 or type(created) not in (int,float) or not math.isfinite(created) or created<=0:
        raise ValueError("Exact PID/creation identity required")
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.create_time()==created and process.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    # AccessDenied is deliberately not proof that a process died.


def _read(path):
    if path.is_symlink() or (hasattr(path,"is_junction") and path.is_junction()) or path.stat().st_size>16_384:
        raise ValueError("Invalid resource ownership file")
    value=json.loads(path.read_text(encoding="utf-8"))
    required={"schema","experiment_id","token","parent_pid","parent_created","child_pid",
              "child_created","child_chain","launching","max_process_rss_bytes"}
    if not isinstance(value,dict) or set(value)!=required or value["schema"]!="alpha-resource-reservation-v1":
        raise ValueError("Strict reservation record required")
    if (not isinstance(value["experiment_id"],str) or not re.fullmatch(r"exp-[0-9a-f]{12}",value["experiment_id"])
            or not isinstance(value["token"],str) or not re.fullmatch(r"[0-9a-f]{32}",value["token"])
            or type(value["launching"]) is not bool
            or type(value["max_process_rss_bytes"]) is not int or not 134_217_728<=value["max_process_rss_bytes"]<=17_179_869_184):
        raise ValueError("Malformed reservation identity/resources")
    chain=value["child_chain"]
    if not isinstance(chain,list) or len(chain)>2:raise ValueError("Bounded interpreter/console helper identities required")
    for child in chain:
        if not isinstance(child,dict) or set(child)!={"pid","created","role"} or child["role"] not in {"interpreter","console_host"}:
            raise ValueError("Strict interpreter/helper identity required")
        process_alive(child["pid"],child["created"])
    if len({c["role"] for c in chain})!=len(chain):raise ValueError("Duplicate helper role")
    # Validation without treating a dead parent as a dead child.
    for prefix in ("parent","child"):
        pid,created=value[prefix+"_pid"],value[prefix+"_created"]
        if prefix=="child" and pid is None and created is None:
            continue
        if type(pid) is not int or pid<1 or type(created) not in (int,float) or not math.isfinite(created) or created<=0:
            raise ValueError("Malformed reservation process identity")
    return value


def _pool(root):
    root=Path(root).resolve();pool=root/".alpha_slots"
    if pool.is_symlink() or (hasattr(pool,"is_junction") and pool.is_junction()):
        raise ValueError("Resource pool must not be linked")
    pool.mkdir(parents=True,exist_ok=True)
    if not pool.resolve().is_relative_to(root):
        raise ValueError("Resource pool escaped research root")
    return pool


@contextmanager
def _guard(pool):
    """Kernel file lock; never unlink the permanent lock inode or guess a PID."""
    path=pool/"guard.lock"
    if path.is_symlink() or not path.resolve().is_relative_to(pool):
        raise ValueError("Resource guard path escaped pool")
    with path.open("a+b") as f:
        if os.fstat(f.fileno()).st_size==0:
            f.write(b"0");f.flush()
        deadline=time.monotonic()+2;locked=False
        while not locked:
            try:
                f.seek(0)
                if os.name=="nt":
                    import msvcrt
                    msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                locked=True
            except OSError:
                if time.monotonic()>deadline:
                    raise GuardBusy("Resource guard busy; no state changed")
                time.sleep(.05)
        try:
            yield
        finally:
            f.seek(0)
            if os.name=="nt":
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(f.fileno(),fcntl.LOCK_UN)


@dataclass
class Reservation:
    pool: Path
    path: Path
    token: str
    record: dict
    retain_on_exit: bool = False
    admission: dict | None = None

    def bind(self, proc):
        created=psutil.Process(proc.pid).create_time()
        with _guard(self.pool):
            current=_read(self.path)
            if current["token"]!=self.token:
                raise ValueError("Resource reservation ownership changed")
            self.record.update(child_pid=proc.pid,child_created=created,launching=False)
            write_json(self.path,self.record)
        return created

    def bind_chain(self, chain):
        with _guard(self.pool):
            current=_read(self.path)
            if current["token"]!=self.token or current["child_pid"] is None:
                raise ValueError("Interpreter chain owner not bound")
            self.record["child_chain"]=chain
            write_json(self.path,self.record)

    def launching(self):
        with _guard(self.pool):
            if _read(self.path)["token"]!=self.token:
                raise ValueError("Resource reservation ownership changed")
            self.record["launching"]=True;write_json(self.path,self.record)

    def abort_unbound_launch(self, proc=None):
        if proc is not None and proc.poll() is None:
            raise ValueError("Cannot clear an active unbound process")
        with _guard(self.pool):
            current=_read(self.path)
            if current["token"]!=self.token or current.get("child_pid") is not None:
                raise ValueError("Unbound launch ownership changed")
            self.record["launching"]=False;write_json(self.path,self.record)

    def release(self):
        with _guard(self.pool):
            # A terminal recovery collector may have released/reassigned this
            # slot already. A proven dead exact child makes that idempotent.
            dead=(self.record["child_pid"] is not None and
                  not process_alive(self.record["child_pid"],self.record["child_created"]) and
                  not any(process_alive(c["pid"],c["created"]) for c in self.record["child_chain"]))
            if not self.path.exists():
                if dead:return
                raise ValueError("Unbound resource reservation disappeared")
            current=_read(self.path)
            if current["token"]!=self.token:
                if dead:return
                raise ValueError("Cannot release foreign resource ownership")
            if not self.path.resolve().is_relative_to(self.pool):
                raise ValueError("Reservation escaped pool")
            pid=current["child_pid"]
            if current["launching"] and pid is None:
                raise ValueError("Unbound launch identity unresolved; reservation retained")
            if (pid is not None and process_alive(pid,current["child_created"])) or any(
                    process_alive(c["pid"],c["created"]) for c in current["child_chain"]):
                raise ValueError("Worker still alive; resource reservation retained")
            self.path.unlink()


@contextmanager
def reserve(root, experiment_id, resources):
    if not isinstance(resources,WorkerResources) or not isinstance(experiment_id,str) or not re.fullmatch(r"exp-[0-9a-f]{12}",experiment_id):
        raise ValueError("Validated worker resources/experiment identity required")
    pool=_pool(root)
    with _guard(pool):
        paths=[pool/("slot"+str(i)+".json") for i in range(2)]
        used=[_read(path) for path in paths if path.exists()]
        capacity=sum(record["max_process_rss_bytes"] for record in used);owned_rss=0
        for record in used:
            pid=record["child_pid"]
            try:
                if pid is not None and process_alive(pid,record["child_created"]):
                    owned_rss+=psutil.Process(pid).memory_info().rss
                for child in record["child_chain"]:
                    if process_alive(child["pid"],child["created"]):owned_rss+=psutil.Process(child["pid"]).memory_info().rss
            except psutil.NoSuchProcess:
                pass
        available=psutil.virtual_memory().available
        free=next((p for p in paths if not p.exists()),None)
        if free is None:
            raise RuntimeError("Both numerical resource slots are reserved; batch remains planned")
        commit_available = _available_commit_bytes()
        required_physical = capacity + resources.max_process_rss_bytes + resources.min_available_memory_bytes - owned_rss
        required_commit = capacity + resources.max_process_rss_bytes + resources.min_available_memory_bytes
        covered = available >= required_physical and (commit_available is None or commit_available >= required_commit)
        mode = "fully_reserved"
        if not covered:
            # A ceiling is not an allocation forecast. Allow only one monitored
            # trial, keeping the declared runtime reserve plus a bootstrap margin.
            # Other workers still require their full conservative reservations.
            if used:
                raise RuntimeError(
                    "Insufficient headroom for parallel reservations "
                    f"(existing reservations={capacity/(1 << 30):.3f} GiB); batch remains planned")
            startup_floor = resources.min_available_memory_bytes + min(1 << 28, resources.max_process_rss_bytes)
            if available < startup_floor:
                raise RuntimeError("Insufficient unreserved available-memory startup floor; batch remains planned")
            if commit_available is not None and commit_available < startup_floor:
                raise RuntimeError(
                    f"Insufficient Windows commit headroom: available={commit_available/(1 << 30):.3f} GiB, "
                    f"required startup floor={startup_floor/(1 << 30):.3f} GiB; batch remains planned")
            mode = "monitored_trial"
        admission = dict(schema="memory-admission-v2", mode=mode,
            available_physical_bytes=available, available_commit_bytes=commit_available,
            declared_rss_ceiling_bytes=resources.max_process_rss_bytes,
            full_reservation_physical_bytes=required_physical,
            full_reservation_commit_bytes=required_commit,
            runtime_reserve_bytes=resources.min_available_memory_bytes)
        token=uuid.uuid4().hex
        record=dict(schema="alpha-resource-reservation-v1",experiment_id=experiment_id,token=token,
                    parent_pid=os.getpid(),parent_created=psutil.Process().create_time(),
                    child_pid=None,child_created=None,child_chain=[],launching=False,
                    max_process_rss_bytes=resources.max_process_rss_bytes)
        write_json(free,record)
    reservation=Reservation(pool,free,token,record,admission=admission)
    try:
        yield reservation
    finally:
        if not reservation.retain_on_exit:
            reservation.release()


def check_terminal_reservation(root, experiment_id, launch):
    """Authenticate live reservation BEFORE committing an experiment terminal state."""
    pool=_pool(root)
    with _guard(pool):
        records=[_read(p) for p in (pool/"slot0.json",pool/"slot1.json") if p.exists()]
        owners=[r for r in records if r["experiment_id"]==experiment_id]
        if len(owners)!=1:raise ValueError("Exact terminal reservation missing or ambiguous; preserve job")
        record=owners[0]
        if (record["child_pid"]!=launch["pid"] or record["child_created"]!=launch["created"]
                or record["child_chain"]!=launch["interpreter_chain"]
                or record["max_process_rss_bytes"]!=launch["resources"]["max_process_rss_bytes"]):
            raise ValueError("Terminal launch differs from resource owner")
        return not (process_alive(record["child_pid"],record["child_created"]) or any(
            process_alive(c["pid"],c["created"]) for c in record["child_chain"]))


def release_terminal_reservation(root, experiment_id, launch):
    """A verified terminal collector releases only an exited exact child."""
    pool=_pool(root)
    with _guard(pool):
        for path in (pool/"slot0.json",pool/"slot1.json"):
            if not path.exists():continue
            record=_read(path)
            if record["experiment_id"]!=experiment_id:continue
            if (record["child_pid"]!=launch["pid"] or record["child_created"]!=launch["created"]
                    or record["child_chain"]!=launch.get("interpreter_chain",[])
                    or ("resources" in launch and record["max_process_rss_bytes"]!=launch["resources"]["max_process_rss_bytes"])):
                raise ValueError("Terminal launch differs from resource owner")
            if process_alive(launch["pid"],launch["created"]) or any(
                    process_alive(c["pid"],c["created"]) for c in record["child_chain"]):return False
            path.unlink()
    return True


def hidden_console_record(child, parent_pid, *, windows, system_root):
    """Validate the one observed hidden console helper, never a numeric child."""
    if not windows or not isinstance(system_root, str) or not system_root:
        raise ValueError("Hidden console helper requires Windows system root")
    expected = os.path.normcase(os.path.join(system_root, "System32", "conhost.exe"))
    if (child.ppid() != parent_pid or os.path.normcase(child.exe()) != expected
            or child.cmdline()[1:] != ["0x4"]):
        raise ValueError("Worker child is not the exact hidden console host")
    created = child.create_time()
    if (type(child.pid) is not int or child.pid < 1 or type(created) not in (int, float)
            or not math.isfinite(created) or created <= 0):
        raise ValueError("Hidden console identity requires exact PID and creation time")
    return dict(pid=child.pid, created=created, role="console_host")


def check_memory_limits(resources, rss, available_physical, available_commit):
    """Sampled stop conditions; no claim of an OS-hard allocation guarantee."""
    if rss > resources.max_process_rss_bytes:
        raise MemoryError("Sampled worker RSS limit exceeded")
    if available_physical < resources.min_available_memory_bytes:
        raise MemoryError("Available-memory reserve crossed")
    if available_commit is not None and available_commit < resources.min_available_memory_bytes:
        raise MemoryError("Windows available-commit reserve crossed")
