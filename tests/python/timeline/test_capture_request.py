"""Tests for TVM_TIMELINE_CAPTURE_REQUEST directed capture + negative-ts fix +
directed immediate-flush (trace published at ExitCapture, not only at exit).

Most tests spawn a subprocess that runs the model and exits; the trace is written
either at process exit (full mode / destructor fallback) or immediately after the
target request (directed mode — ExitCapture publish). The parent reads the JSON
trace and asserts on its content. The flush tests use a sync-file handshake so the
parent can inspect the file WHILE the subprocess is still alive."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

TVM_ROOT = Path(__file__).resolve().parents[3]

PARALLEL_RUNNER = str(Path(__file__).resolve().parent / "_parallel_runner.py")

RUNNER = r"""
import os, sys
import numpy as np
import tvm
from tvm import relax
from tvm.relax import BlockBuilder

exec_mode = os.environ.get("TEST_EXEC_MODE", "compiled")
n_runs = int(os.environ.get("TEST_N_RUNS", "5"))

N_OPS = 4
M, D = 32, 64
rng = np.random.default_rng(0)
bb = BlockBuilder()
x = relax.Var("x", relax.TensorStructInfo((M, D), "float32"))
ws = [(rng.standard_normal((D, D)) * 0.1).astype("float32") for _ in range(N_OPS)]
with bb.function("main", [x]):
    with bb.dataflow():
        cur = x
        for i in range(N_OPS):
            cur = bb.emit(relax.op.matmul(cur, relax.const(ws[i])))
            cur = bb.emit(relax.op.nn.relu(cur))
        gv = bb.emit_output(cur)
    bb.emit_func_output(gv)
mod = bb.get()
mod = relax.transform.LegalizeOps()(mod)
if exec_mode == "compiled":
    from tvm.relax.timeline import instrument_timeline
    mod = instrument_timeline(mod)
ex = relax.build(mod, target=tvm.target.Target("llvm"), exec_mode=exec_mode)
dev = tvm.cpu()
vm = relax.VirtualMachine(ex, dev)
xt = tvm.runtime.empty((M, D), "float32", dev)
xt.copyfrom(rng.standard_normal((M, D)).astype("float32"))
for _ in range(n_runs):
    vm["main"](xt)
"""


def _subprocess_env(extra=None):
    env = dict(os.environ)
    # FC-3: 子进程不得继承父进程的 timeline 变量——外层 TVM_TIMELINE_ENABLE=1
    # 会让 test_env_off_no_file 等用例稳定失败（砚砚 fresh-context review FC-3）。
    # 先清空，再 apply 各用例的 extra 参数。
    for k in ("TVM_TIMELINE_ENABLE", "TVM_TIMELINE_CAPTURE_REQUEST",
              "TVM_TIMELINE_OUTPUT_FILE", "TVM_TIMELINE_MAX_EVENTS"):
        env.pop(k, None)
    env["PYTHONPATH"] = ":".join([
        str(TVM_ROOT / "python"),
        str(TVM_ROOT / "3rdparty" / "tvm-ffi" / "python"),
        str(TVM_ROOT / "3rdparty" / "tvm-ffi" / "build"),
    ])
    env["TVM_LIBRARY_PATH"] = str(TVM_ROOT / "build")
    if extra:
        env.update(extra)
    return env


def _run_trace(env_extra, expect_file=True, runner_script=None):
    """Run model in subprocess; return parsed trace dict or None if no file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        trace_path = f.name
    if os.path.exists(trace_path):
        os.unlink(trace_path)
    env = _subprocess_env(env_extra)
    env["TVM_TIMELINE_OUTPUT_FILE"] = trace_path
    cmd = ([sys.executable, runner_script] if runner_script
           else [sys.executable, "-c", RUNNER])
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"subprocess exit {result.returncode}\n{result.stderr[-800:]}"
    if not os.path.exists(trace_path):
        if expect_file:
            return None
        return None
    with open(trace_path) as f:
        data = json.load(f)
    os.unlink(trace_path)
    return data


def test_directed_capture_req3_of5():
    """N=3, 5 runs: exactly 1 inference event with request_index=3."""
    trace = _run_trace({
        "TVM_TIMELINE_ENABLE": "1",
        "TVM_TIMELINE_CAPTURE_REQUEST": "3",
        "TEST_N_RUNS": "5",
        "TEST_EXEC_MODE": "compiled",
    })
    assert trace is not None, "no trace file"
    events = trace["traceEvents"]
    inf = [e for e in events if e.get("cat") == "inference"]
    assert len(inf) == 1, f"expected 1 inference, got {len(inf)}"
    assert inf[0]["args"]["request_index"] == 3
    # FR-1: strict count — RUNNER is N_OPS=4 (4 matmul + 4 relu = 8 kernels per
    # request); target=3 means only req 3's kernels are recorded. >0 would pass
    # even if the gate leaked and recorded all 5 requests' kernels (5*8=40).
    kernels = [e for e in events if e.get("cat") == "kernel"]
    assert len(kernels) == 8, (
        f"expected exactly 8 kernels (req 3 only: 4 matmul + 4 relu), got {len(kernels)}")
    for e in events:
        assert e["ts"] >= 0, f"negative ts: {e}"


def test_full_mode_all_requests():
    """No capture_request: full mode, every request gets inference boundary."""
    trace = _run_trace({
        "TVM_TIMELINE_ENABLE": "1",
        "TEST_N_RUNS": "5",
        "TEST_EXEC_MODE": "compiled",
    })
    assert trace is not None
    inf = [e for e in trace["traceEvents"] if e.get("cat") == "inference"]
    assert len(inf) == 5, f"expected 5 inference in full mode, got {len(inf)}"


def test_ts_nonneg_full():
    """All events ts >= 0 (negative-timestamp fix)."""
    trace = _run_trace({
        "TVM_TIMELINE_ENABLE": "1",
        "TEST_N_RUNS": "3",
        "TEST_EXEC_MODE": "compiled",
    })
    assert trace is not None
    for e in trace["traceEvents"]:
        assert e["ts"] >= 0, f"negative ts: {e}"


def test_env_off_no_file():
    """TVM_TIMELINE_ENABLE unset: no trace file."""
    trace = _run_trace({}, expect_file=False)
    assert trace is None, "trace file written when timeline disabled"


def test_invalid_n_full_mode():
    """N=3abc: invalid, falls back to full mode (3 inference events)."""
    trace = _run_trace({
        "TVM_TIMELINE_ENABLE": "1",
        "TVM_TIMELINE_CAPTURE_REQUEST": "3abc",
        "TEST_N_RUNS": "3",
        "TEST_EXEC_MODE": "compiled",
    })
    assert trace is not None
    inf = [e for e in trace["traceEvents"] if e.get("cat") == "inference"]
    assert len(inf) == 3, f"invalid N should fall back; got {len(inf)}"


def test_huge_n_falls_back_to_full():
    """FR-3: N=99999999999 (> INT_MAX) — on LLP64 (Windows), strtol saturates at
    LONG_MAX==INT_MAX and sets errno=ERANGE. Without errno check this would be
    silently accepted; with the fix it falls back to full mode."""
    trace = _run_trace({
        "TVM_TIMELINE_ENABLE": "1",
        "TVM_TIMELINE_CAPTURE_REQUEST": "99999999999",
        "TEST_N_RUNS": "3",
        "TEST_EXEC_MODE": "compiled",
    })
    assert trace is not None
    inf = [e for e in trace["traceEvents"] if e.get("cat") == "inference"]
    assert len(inf) == 3, (
        f"huge N should fall back to full mode (errno=ERANGE), got {len(inf)} inference")


def test_bytecode_mode_works():
    """bytecode exec_mode also produces inference events."""
    trace = _run_trace({
        "TVM_TIMELINE_ENABLE": "1",
        "TVM_TIMELINE_CAPTURE_REQUEST": "2",
        "TEST_N_RUNS": "3",
        "TEST_EXEC_MODE": "bytecode",
    })
    assert trace is not None
    inf = [e for e in trace["traceEvents"] if e.get("cat") == "inference"]
    assert len(inf) == 1
    assert inf[0]["args"]["request_index"] == 2


def test_target_not_reached_no_file():
    """N=10, only 3 runs: target not reached, no trace file + stderr warning."""
    # FC-6: 用唯一 tempfile 路径，运行前确认不存在，结束后断言仍不存在——
    # 固定 /tmp 路径会被残留文件或并行测试掩盖（砚砚 FC-6）。
    tmpdir = tempfile.mkdtemp(prefix="tl_target_unreached_")
    trace_path = os.path.join(tmpdir, "trace.json")
    assert not os.path.exists(trace_path), "precondition: trace must not exist"
    result = subprocess.run(
        [sys.executable, "-c", RUNNER],
        env=_subprocess_env({
            "TVM_TIMELINE_ENABLE": "1",
            "TVM_TIMELINE_CAPTURE_REQUEST": "10",
            "TEST_N_RUNS": "3",
            "TVM_TIMELINE_OUTPUT_FILE": trace_path,
        }),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0
    assert "target request was not reached" in result.stderr
    assert not os.path.exists(trace_path), (
        f"trace file written despite target not reached: {trace_path}")
    os.rmdir(tmpdir)


RUNNER_CONCURRENT = r"""
import os, sys, threading
import numpy as np
import tvm
from tvm import relax
from tvm.relax import BlockBuilder
from tvm.relax.timeline import instrument_timeline

# FR-2 deterministic overlap (砚砚 formal review): a Barrier only guarantees
# simultaneous start — the OS scheduler could run B to completion before A
# reaches its target request, so the old process-global `capture_active_` impl
# would *also* pass (B's request ran while the flag was still false). To kill
# that bug deterministically, B's request must land *inside* A's capture window.
#
# A's compiled model calls this identity DPS hook mid-inference. Because
# EnterCapture runs at `main` entry (vm.cc _LookupFunction lambda), by the time
# the hook fires on A's target request A is already inside the capture window.
# A blocks here until B finishes its single non-target request, so B's request
# is guaranteed to overlap the window. Under the old global-state bug, B's
# kernels would be recorded (the gate sees capture_active_=true set by A) -> leak
# to 16. Under thread_local isolation, B's gate stays false -> no leak (stays 8).
#
# threading.local separates roles: B's `main` shares the compiled exe and thus
# also calls this hook, but B's call is a no-op (only A synchronizes).
tls = threading.local()
b_can_start = threading.Event()
b_done = threading.Event()
sync_errors = []


@tvm.register_global_func("test.timeline_sync_hook")
def sync_hook(out, x):
    x.copyto(out)  # identity: keeps output correct and wires the call into the
                   # dataflow so it cannot be dead-code eliminated.
    if getattr(tls, "role", None) != "A":
        return  # B's main hits the same hook (shared exe); no-op for B.
    if getattr(tls, "armed", False):
        tls.armed = False  # idempotent: a repeat would be a 2nd no-op, not a 2nd wait
        b_can_start.set()  # release B: run your non-target request now
        if not b_done.wait(timeout=30):
            sync_errors.append("B did not finish within 30s")


N_OPS, M, D = 4, 32, 64
rng = np.random.default_rng(0)
bb = BlockBuilder()
x = relax.Var("x", relax.TensorStructInfo((M, D), "float32"))
ws = [(rng.standard_normal((D, D)) * 0.1).astype("float32") for _ in range(N_OPS)]
with bb.function("main", [x]):
    with bb.dataflow():
        cur = x
        for i in range(N_OPS):
            cur = bb.emit(relax.op.matmul(cur, relax.const(ws[i])))
            cur = bb.emit(relax.op.nn.relu(cur))
        cur = bb.emit(relax.op.call_dps_packed(
            "test.timeline_sync_hook", cur,
            relax.TensorStructInfo((M, D), "float32")))
        gv = bb.emit_output(cur)
    bb.emit_func_output(gv)
mod = bb.get()
mod = relax.transform.LegalizeOps()(mod)
mod = instrument_timeline(mod)
ex = relax.build(mod, target=tvm.target.Target("llvm"), exec_mode="compiled")
dev = tvm.cpu()


def make_vm():
    vm = relax.VirtualMachine(ex, dev)
    xt = tvm.runtime.empty((M, D), "float32", dev)
    xt.copyfrom(rng.standard_normal((M, D)).astype("float32"))
    return vm, xt


# Each thread owns a VirtualMachine; sharing immutable `ex` is safe. The per-VM
# request counter (vm.cc timeline_request_counter_) is instance-local, so A's
# counter and B's counter are independent. capture_request=2 means A's 2nd main
# call is the target (capture on), while B's only main call is req 1 != 2 (B
# never enters capture -> any recording of B is purely a gate leak).
vm_a, xt_a = make_vm()
vm_b, xt_b = make_vm()

errors = []


def worker_a():
    try:
        tls.role = "A"
        tls.armed = False
        vm_a["main"](xt_a)  # req 1 (not target): capture off, hook no-op
        tls.armed = True     # arm: the next hook (inside req 2's capture) will sync
        vm_a["main"](xt_a)  # req 2 == target: EnterCapture -> matmuls -> hook(blocks on B)
    except Exception as e:
        errors.append(("A", repr(e)))


def worker_b():
    try:
        tls.role = "B"
        if not b_can_start.wait(timeout=30):
            errors.append(("B", "A never signaled capture entry"))
            return
        # B's only request, deterministically inside A's capture window (A is
        # blocked in the hook waiting on b_done). B counter=1 != target=2, so B
        # does not EnterCapture — under correct thread_local isolation B's
        # tl_capture_active is false and nothing of B's is recorded.
        vm_b["main"](xt_b)
        b_done.set()
    except Exception as e:
        errors.append(("B", repr(e)))


ta = threading.Thread(target=worker_a)
tb = threading.Thread(target=worker_b)
ta.start(); tb.start()
ta.join(timeout=60); tb.join(timeout=60)
if errors or sync_errors:
    sys.stderr.write("concurrent runner errors: " + str(errors + sync_errors) + "\n")
    sys.exit(1)
"""


def test_concurrent_vms_no_leak():
    """Two VMs on two threads: thread B's non-target events must not leak via
    thread A's active capture (thread_local isolation, P1 fix). FR-2: the
    runner uses a deterministic in-model sync hook (see RUNNER_CONCURRENT) so
    B's request provably lands inside A's capture window — a process-global
    `capture_active_` would record B's kernels (16 total) and fail here."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        trace_path = f.name
    if os.path.exists(trace_path):
        os.unlink(trace_path)
    env = _subprocess_env({
        "TVM_TIMELINE_ENABLE": "1",
        "TVM_TIMELINE_CAPTURE_REQUEST": "2",
        "TVM_TIMELINE_OUTPUT_FILE": trace_path,
    })
    result = subprocess.run(
        [sys.executable, "-c", RUNNER_CONCURRENT],
        env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"concurrent runner failed:\n{result.stderr[-800:]}"
    assert os.path.exists(trace_path), "no trace file"
    with open(trace_path) as f:
        trace = json.load(f)
    os.unlink(trace_path)
    events = trace["traceEvents"]
    inf = [e for e in events if e.get("cat") == "inference"]
    assert len(inf) == 1, f"expected 1 inference (thread A req 2), got {len(inf)}"
    assert inf[0]["args"]["request_index"] == 2
    kernels = [e for e in events if e.get("cat") == "kernel"]
    # FR-2: A's req 2 = 4 matmul + 4 relu = 8 kernels. B contributes 0 (B's
    # thread_local tl_capture_active is false). The runner's sync hook makes
    # B's request overlap A's window *deterministically*, so the old
    # process-global capture_active_ impl would record B's 8 kernels too
    # (total 16) and this assert would fail — that is the regression guard.
    assert len(kernels) == 8, (
        f"expected 8 kernels (thread A only), got {len(kernels)} — possible cross-thread leak")


def test_parallel_compiled_capture():
    """FC-2: compiled + parallel_region — capture target req, assert parallel_region
    events present and only from the target request (gate covers R-events, not just K)."""
    trace = _run_trace({
        "TVM_TIMELINE_ENABLE": "1",
        "TVM_TIMELINE_CAPTURE_REQUEST": "1",
        "TEST_N_RUNS": "3",
        "TEST_EXEC_MODE": "compiled",
    }, runner_script=PARALLEL_RUNNER)
    assert trace is not None, "no trace file"
    events = trace["traceEvents"]
    inf = [e for e in events if e.get("cat") == "inference"]
    assert len(inf) == 1 and inf[0]["args"]["request_index"] == 1
    pr = [e for e in events if e.get("cat") == "parallel_region"]
    # FR-1: strict — _parallel_runner.padd is a single T.parallel loop, compiled
    # to exactly 1 parallel_region per request; target=1 means only req 1 recorded.
    assert len(pr) == 1, (
        f"expected exactly 1 parallel_region (req 1 only), got {len(pr)}")
    kernels = [e for e in events if e.get("cat") == "kernel"]
    assert len(kernels) == 1, (
        f"expected exactly 1 kernel (req 1 only, single padd call), got {len(kernels)}")
    for e in events:
        assert e["ts"] >= 0, f"negative ts: {e}"


def test_parallel_bytecode_capture():
    """FC-2: bytecode + parallel_region — same isolation as compiled mode."""
    trace = _run_trace({
        "TVM_TIMELINE_ENABLE": "1",
        "TVM_TIMELINE_CAPTURE_REQUEST": "1",
        "TEST_N_RUNS": "3",
        "TEST_EXEC_MODE": "bytecode",
    }, runner_script=PARALLEL_RUNNER)
    assert trace is not None, "no trace file"
    events = trace["traceEvents"]
    inf = [e for e in events if e.get("cat") == "inference"]
    assert len(inf) == 1 and inf[0]["args"]["request_index"] == 1
    pr = [e for e in events if e.get("cat") == "parallel_region"]
    # FR-1: strict — same model as compiled, bytecode emits identical event
    # shape (1 parallel_region + 1 kernel for the single padd call). Verified
    # empirically: bytecode still executes the compiled padd PrimFunc, so
    # T.parallel -> 1 parallel_region and TVMBackendTimelineBegin/End -> 1 kernel.
    assert len(pr) == 1, (
        f"expected exactly 1 parallel_region (req 1 only), got {len(pr)}")
    # FR-1 (砚砚 formal review bytecode kernel gap): the compiled sibling test
    # already asserts kernel==1; bytecode must match or a gate that leaked the
    # kernel track while filtering parallel_region (or vice versa) would pass
    # undetected.
    kernels = [e for e in events if e.get("cat") == "kernel"]
    assert len(kernels) == 1, (
        f"expected exactly 1 kernel (req 1 only, single padd call), got {len(kernels)}")
    for e in events:
        assert e["ts"] >= 0, f"negative ts: {e}"


# ---------------------------------------------------------------------------
# Directed immediate-flush tests: TVM_TIMELINE_CAPTURE_REQUEST publishes the trace
# at ExitCapture (the instant the target request completes), NOT only at static
# teardown. Motivation: Tomcat / JNI / FFM hosts never reach static destruction, so
# the trace must already be on disk by the time the Nth inference returns.
# Design (converged with 砚砚 review):
#   - dirty_ set on every event append; cleared on a successful WriteTraceLocked.
#   - Each directed ExitCapture publishes a cumulative snapshot (multiple VMs each
#     hitting target N all land in the file).
#   - WriteTraceLocked copies events_ before rebasing/sorting (no in-place mutation
#     → a second flush stays on one coordinate system).
#   - Atomic publish: <path>.partial → close+verify → platform atomic replace
#     (std::rename on POSIX, MoveFileExA on Windows — no <filesystem>, GCC 7.1 safe).
#     External readers never see a half-written JSON. ofstream::close flushes the
#     user buffer only (NOT fsync) — visibility to concurrent readers, not power-loss
#     durability.
# ---------------------------------------------------------------------------

RUNNER_FLUSH = r"""
import os, sys, time
import numpy as np
import tvm
from tvm import relax
from tvm.relax import BlockBuilder
from tvm.relax.timeline import instrument_timeline

n_runs = int(os.environ.get("TEST_N_RUNS", "2"))

N_OPS, M, D = 4, 32, 64
rng = np.random.default_rng(0)
bb = BlockBuilder()
x = relax.Var("x", relax.TensorStructInfo((M, D), "float32"))
ws = [(rng.standard_normal((D, D)) * 0.1).astype("float32") for _ in range(N_OPS)]
with bb.function("main", [x]):
    with bb.dataflow():
        cur = x
        for i in range(N_OPS):
            cur = bb.emit(relax.op.matmul(cur, relax.const(ws[i])))
            cur = bb.emit(relax.op.nn.relu(cur))
        gv = bb.emit_output(cur)
    bb.emit_func_output(gv)
mod = bb.get()
mod = relax.transform.LegalizeOps()(mod)
mod = instrument_timeline(mod)
ex = relax.build(mod, target=tvm.target.Target("llvm"), exec_mode="compiled")
dev = tvm.cpu()
vm = relax.VirtualMachine(ex, dev)
xt = tvm.runtime.empty((M, D), "float32", dev)
xt.copyfrom(rng.standard_normal((M, D)).astype("float32"))
for _ in range(n_runs):
    vm["main"](xt)
# The directed flush for target N fired inside the last ExitCapture. Signal the
# parent via a sync file (so it can inspect the trace WHILE we are still alive),
# then wait for a proceed file before exiting normally (so the static destructor
# runs and the parent can compare before/after).
with open(os.environ["TL_SYNC_FILE"], "w") as _f:
    _f.write("done")
deadline = time.time() + 120
while not os.path.exists(os.environ["TL_PROCEED_FILE"]):
    if time.time() > deadline:
        break
    time.sleep(0.05)
"""


def _spawn_flush_runner(env_extra, n_runs=2):
    tmpdir = tempfile.mkdtemp(prefix="tl_flush_")
    trace_path = os.path.join(tmpdir, "trace.json")
    sync_file = os.path.join(tmpdir, "sync")
    proceed_file = os.path.join(tmpdir, "proceed")
    env = _subprocess_env(dict(
        env_extra, TEST_N_RUNS=str(n_runs), TVM_TIMELINE_OUTPUT_FILE=trace_path,
        TL_SYNC_FILE=sync_file, TL_PROCEED_FILE=proceed_file))
    proc = subprocess.Popen(
        [sys.executable, "-c", RUNNER_FLUSH], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc, trace_path, sync_file, proceed_file, tmpdir


def _wait_sync_file(sync_file, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(sync_file):
            return True
        time.sleep(0.1)
    return False


def test_directed_flush_before_exit():
    """Core: the trace JSON exists WHILE the process is still alive — the directed
    flush fires at ExitCapture, not at static teardown. This is the fix for long-
    running hosts (Tomcat/JNI) that never exit."""
    proc, trace_path, sync_file, proceed_file, tmpdir = _spawn_flush_runner(
        {"TVM_TIMELINE_ENABLE": "1", "TVM_TIMELINE_CAPTURE_REQUEST": "2"}, n_runs=2)
    try:
        if not _wait_sync_file(sync_file, timeout=120):
            proc.terminate()
            _, err = proc.communicate(timeout=10)
            pytest.fail(
                f"never saw sync file (exited={proc.poll() is not None})\n{err[:1200]}")
        assert proc.poll() is None, "subprocess exited before we could inspect the file"
        assert os.path.exists(trace_path), (
            "trace not published before exit — directed flush did not fire at ExitCapture")
        with open(trace_path) as f:
            trace = json.load(f)
        inf = [e for e in trace["traceEvents"] if e.get("cat") == "inference"]
        assert len(inf) == 1 and inf[0]["args"]["request_index"] == 2
        kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
        assert len(kernels) == 8, f"expected 8 kernels (N_OPS=4), got {len(kernels)}"
        for e in trace["traceEvents"]:
            assert e["ts"] >= 0, f"negative ts: {e}"
    finally:
        if proc.poll() is None:
            try:
                open(proceed_file, "w").close()
            except OSError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_directed_no_double_write():
    """After a successful directed flush, dirty_ is cleared so the destructor does
    NOT re-write. Verified by identical file bytes before/after normal exit and
    exactly one '[TIMELINE] wrote' log line."""
    proc, trace_path, sync_file, proceed_file, tmpdir = _spawn_flush_runner(
        {"TVM_TIMELINE_ENABLE": "1", "TVM_TIMELINE_CAPTURE_REQUEST": "2"}, n_runs=2)
    try:
        assert _wait_sync_file(sync_file, timeout=120), "never saw sync file"
        with open(trace_path, "rb") as f:
            before = f.read()
        try:
            open(proceed_file, "w").close()
        except OSError:
            pass
        proc.wait(timeout=30)
        stderr = proc.stderr.read()
        with open(trace_path, "rb") as f:
            after = f.read()
        assert before == after, "destructor re-wrote the file after a successful flush"
        wrote_count = stderr.count("[TIMELINE] wrote")
        assert wrote_count == 1, (
            f"expected exactly 1 'wrote' log, got {wrote_count}\nstderr:\n{stderr[:1200]}")
    finally:
        if proc.poll() is None:
            proc.kill()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_flush_fail_destructor_retry():
    """Output path unwritable: the directed flush fails (cannot open .partial) →
    dirty_ stays true → the destructor retries at teardown. Exactly 2 'cannot open'
    logs: one from ExitCapture's directed publish, one from the destructor fallback.
    Uses a regular file as the parent directory of the output path (ENOTDIR), per
    砚砚 review. ==2 (not >=2) so an unexpected third write is caught."""
    tmpdir = tempfile.mkdtemp(prefix="tl_flushfail_")
    blocker = os.path.join(tmpdir, "blocker")
    with open(blocker, "w") as f:
        f.write("x")  # regular file — parent-dir open yields ENOTDIR
    trace_path = os.path.join(blocker, "trace.json")
    result = subprocess.run(
        [sys.executable, "-c", RUNNER],
        env=_subprocess_env({
            "TVM_TIMELINE_ENABLE": "1", "TVM_TIMELINE_CAPTURE_REQUEST": "2",
            "TEST_N_RUNS": "2", "TVM_TIMELINE_OUTPUT_FILE": trace_path,
        }),
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"runner failed:\n{result.stderr[:1200]}"
    cannot = result.stderr.count("cannot open")
    assert cannot == 2, (
        f"expected exactly 2 'cannot open' (ExitCapture flush + destructor retry), got "
        f"{cannot}\nstderr:\n{result.stderr[:1200]}")
    assert not os.path.exists(trace_path)
    shutil.rmtree(tmpdir, ignore_errors=True)


RUNNER_MULTI_VM = r"""
import numpy as np
import tvm
from tvm import relax
from tvm.relax import BlockBuilder
from tvm.relax.timeline import instrument_timeline

N_OPS, M, D = 2, 32, 64
rng = np.random.default_rng(0)
bb = BlockBuilder()
x = relax.Var("x", relax.TensorStructInfo((M, D), "float32"))
ws = [(rng.standard_normal((D, D)) * 0.1).astype("float32") for _ in range(N_OPS)]
with bb.function("main", [x]):
    with bb.dataflow():
        cur = x
        for i in range(N_OPS):
            cur = bb.emit(relax.op.matmul(cur, relax.const(ws[i])))
            cur = bb.emit(relax.op.nn.relu(cur))
        gv = bb.emit_output(cur)
    bb.emit_func_output(gv)
mod = bb.get()
mod = relax.transform.LegalizeOps()(mod)
mod = instrument_timeline(mod)
ex = relax.build(mod, target=tvm.target.Target("llvm"), exec_mode="compiled")
dev = tvm.cpu()


def make_vm():
    vm = relax.VirtualMachine(ex, dev)
    xt = tvm.runtime.empty((M, D), "float32", dev)
    xt.copyfrom(rng.standard_normal((M, D)).astype("float32"))
    return vm, xt


# Two VM instances; each has its own per-VM request counter (vm.cc member), so
# CAPTURE_REQUEST=1 targets each VM's first main call independently. VM-A's
# ExitCapture publishes A's events; VM-B's ExitCapture publishes the cumulative
# A+B snapshot. The final file must contain BOTH VMs' events. A one-shot
# directed_flush_done_ flag would set done=true at VM-A's flush and skip the
# destructor — if VM-B's ExitCapture ever didn't fire before exit, B's appended
# events would be lost. dirty_ (set on every append) guarantees the destructor
# always publishes unwritten data. This test also guards the snapshot-rebase fix:
# mutating events_ in place would corrupt timestamps on the second flush (two
# coordinate systems coexisting).
vm_a, xt_a = make_vm()
vm_b, xt_b = make_vm()
vm_a["main"](xt_a)
vm_b["main"](xt_b)
"""


def test_multi_vm_directed_flush():
    """Two VMs each hit target (CAPTURE_REQUEST=1): VM-A's ExitCapture publishes A's
    events, VM-B's ExitCapture publishes the cumulative A+B snapshot. The file must
    contain BOTH VMs' events and all timestamps must be non-negative. This guards the
    snapshot-rebase fix (in-place mutation would corrupt ts on the 2nd flush) and the
    cumulative-snapshot contract (each flush writes the full events_ so far). NB: it
    does NOT directly exercise the destructor-publishes-tail path — for that, the
    full-mode tests + code inspection of dirty_ (set on every Record/ExitCapture push,
    cleared only on successful WriteTraceLocked) are the guarantee."""
    tmpdir = tempfile.mkdtemp(prefix="tl_multi_")
    trace_path = os.path.join(tmpdir, "trace.json")
    result = subprocess.run(
        [sys.executable, "-c", RUNNER_MULTI_VM],
        env=_subprocess_env({
            "TVM_TIMELINE_ENABLE": "1", "TVM_TIMELINE_CAPTURE_REQUEST": "1",
            "TVM_TIMELINE_OUTPUT_FILE": trace_path,
        }),
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"multi-vm runner failed:\n{result.stderr[:1200]}"
    assert os.path.exists(trace_path), "no trace file"
    with open(trace_path) as f:
        trace = json.load(f)
    events = trace["traceEvents"]
    inf = [e for e in events if e.get("cat") == "inference"]
    assert len(inf) == 2, f"expected 2 inference (one per VM), got {len(inf)}"
    kernels = [e for e in events if e.get("cat") == "kernel"]
    # N_OPS=2: 2 matmul + 2 relu = 4 kernels/VM x 2 VMs = 8
    assert len(kernels) == 8, (
        f"expected 8 kernels (2 VMs x 4), got {len(kernels)} — second flush may have "
        f"lost VM-A's events")
    for e in events:
        assert e["ts"] >= 0, f"negative ts (snapshot rebase corruption): {e}"
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
