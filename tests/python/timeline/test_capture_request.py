"""Tests for TVM_TIMELINE_CAPTURE_REQUEST directed capture + negative-ts fix.

Each test spawns a subprocess that runs the model and exits (the trace is
written at process exit by TimelineRecorder's static destructor). The parent
test reads the JSON trace and asserts on its content.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

TVM_ROOT = Path(__file__).resolve().parents[3]

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
    env["PYTHONPATH"] = ":".join([
        str(TVM_ROOT / "python"),
        str(TVM_ROOT / "3rdparty" / "tvm-ffi" / "python"),
        str(TVM_ROOT / "3rdparty" / "tvm-ffi" / "build"),
    ])
    env["TVM_LIBRARY_PATH"] = str(TVM_ROOT / "build")
    if extra:
        env.update(extra)
    return env


def _run_trace(env_extra, expect_file=True):
    """Run model in subprocess; return parsed trace dict or None if no file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        trace_path = f.name
    if os.path.exists(trace_path):
        os.unlink(trace_path)
    env = _subprocess_env(env_extra)
    env["TVM_TIMELINE_OUTPUT_FILE"] = trace_path
    result = subprocess.run(
        [sys.executable, "-c", RUNNER],
        env=env, capture_output=True, text=True, timeout=120,
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
    kernels = [e for e in events if e.get("cat") == "kernel"]
    assert len(kernels) > 0, "no kernel events in compiled mode"
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
    result = subprocess.run(
        [sys.executable, "-c", RUNNER],
        env=_subprocess_env({
            "TVM_TIMELINE_ENABLE": "1",
            "TVM_TIMELINE_CAPTURE_REQUEST": "10",
            "TEST_N_RUNS": "3",
            "TVM_TIMELINE_OUTPUT_FILE": "/tmp/tl_target_unreached.json",
        }),
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0
    assert "target request was not reached" in result.stderr
    if os.path.exists("/tmp/tl_target_unreached.json"):
        os.unlink("/tmp/tl_target_unreached.json")


RUNNER_CONCURRENT = r"""
import os, sys, threading
import numpy as np
import tvm
from tvm import relax
from tvm.relax import BlockBuilder
from tvm.relax.timeline import instrument_timeline

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

# Two threads share the process (and thus TimelineRecorder::Global()), but each
# has its own thread_local capture state. Thread A reaches target=2 and captures;
# Thread B runs once (target=2 not reached) — B's events must NOT leak into the
# trace just because A's capture is active.
errors = []
def worker(n_runs, label):
    try:
        for _ in range(n_runs):
            vm["main"](xt)
    except Exception as e:
        errors.append((label, str(e)))

ta = threading.Thread(target=worker, args=(2, "A"))  # reaches target=2
tb = threading.Thread(target=worker, args=(1, "B"))  # does not reach target
ta.start(); tb.start()
ta.join(); tb.join()
if errors:
    sys.exit(1)
"""


def test_concurrent_vms_no_leak():
    """Two VMs on two threads: thread B's non-target events must not leak via
    thread A's active capture (thread_local isolation, P1 fix)."""
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
    # Thread A's req 2: 4 matmul + 4 relu = 8 kernels. Thread B contributes 0
    # (its tl_capture_active is false). If P1 were unfixed (global state), B's
    # events would leak in and kernel count would be 16.
    assert len(kernels) == 8, (
        f"expected 8 kernels (thread A only), got {len(kernels)} — possible cross-thread leak")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
