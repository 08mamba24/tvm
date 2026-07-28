"""Parallel-region runner for capture_request isolation tests (FC-2).

Spawned as a subprocess by test_capture_request.py. Uses a TVMScript module with
an explicit ``T.parallel`` loop so the compiled PrimFunc drives
``TVMBackendParallelLaunch`` and emits ``parallel_region`` timeline events —
something the default Relax ``LegalizeOps`` schedule does NOT produce for plain
matmul/relu. This lets the capture_request gate be exercised on the
parallel_region event path (not just kernel events).

Reads TEST_EXEC_MODE / TEST_N_RUNS from env; the timeline env knobs are set by
the parent test via ``_subprocess_env``.
"""
import os

import numpy as np
import tvm
from tvm import relax
from tvm.relax.timeline import instrument_timeline
from tvm.script import ir as I, relax as R, tir as T

N = 256


@I.ir_module
class ParallelMod:
    @T.prim_func
    def padd(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (N,))
        B = T.match_buffer(b, (N,))
        # Explicit parallel loop -> compiled to TVMBackendParallelLaunch, which
        # is the source of parallel_region timeline events.
        for i in T.parallel(N):
            B[i] = A[i] * 2.0 + 1.0

    @R.function
    def main(x: R.Tensor((N,), "float32")) -> R.Tensor((N,), "float32"):
        cls = ParallelMod
        with R.dataflow():
            gv = R.call_tir(cls.padd, (x,), R.Tensor((N,), "float32"))
            R.output(gv)
        return gv


mode = os.environ.get("TEST_EXEC_MODE", "compiled")
n_runs = int(os.environ.get("TEST_N_RUNS", "3"))

mod = instrument_timeline(ParallelMod)
ex = relax.build(mod, target=tvm.target.Target("llvm"), exec_mode=mode)
dev = tvm.cpu()
vm = relax.VirtualMachine(ex, dev)
xt = tvm.runtime.empty((N,), "float32", dev)
xt.copyfrom(np.ones((N,), "float32"))
for _ in range(n_runs):
    vm["main"](xt)
