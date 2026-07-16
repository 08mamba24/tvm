# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# [fork extension] compile-side pass for the TVM_TIMELINE_ENABLE kernel track.
"""Timeline kernel-track instrumentation: wrap every TIR PrimFunc with
TVMBackendTimelineBegin(name) / TVMBackendTimelineEnd() call_externs.

Pairs with the fork's TVM_TIMELINE_ENABLE runtime (c_backend_api.h /
thread_pool.cc). Because the probes live INSIDE the kernel body, coverage is
exec_mode independent (bytecode AND compiled) and includes kernels with no
parallel loop — the blind spot of the TVMBackendParallelLaunch-only track.

Usage in any compile script, right before relax.build:

.. code-block:: python

    from tvm.relax.timeline import instrument_timeline

    mod = instrument_timeline(mod)
    ex = relax.build(mod, target, exec_mode=...)

Then run with TVM_TIMELINE_ENABLE=1 to get kernel-track events in the
Chrome Trace output. Diagnostic builds only; don't ship one (the probes
defeat pattern-matched rewrites such as reshape's zero-copy view).
"""
from tvm import tir


def instrument_timeline(mod):
    """Return mod with every PrimFunc wrapped in TimelineBegin/End probes.

    The probes go INSIDE the root block body: several relax.build passes
    (e.g. RewriteDataflowReshape's HasReshapePattern ICHECK) require the
    PrimFunc body to stay a root BlockRealize, so wrapping outside breaks
    the build.

    Parameters
    ----------
    mod : tvm.IRModule
        Module whose PrimFuncs to instrument (typically post-LegalizeOps).

    Returns
    -------
    mod : tvm.IRModule
        The same module, PrimFuncs updated in place.
    """
    updates = {}
    for gv, func in mod.functions.items():
        if not isinstance(func, tir.PrimFunc):
            continue
        begin = tir.Evaluate(
            tir.call_extern("int32", "TVMBackendTimelineBegin", tir.StringImm(gv.name_hint))
        )
        end = tir.Evaluate(tir.call_extern("int32", "TVMBackendTimelineEnd"))
        body = func.body
        if isinstance(body, tir.BlockRealize):
            blk = body.block
            new_blk = tir.Block(
                blk.iter_vars,
                blk.reads,
                blk.writes,
                blk.name_hint,
                tir.stmt_seq(begin, blk.body, end),
                blk.init,
                blk.alloc_buffers,
                blk.match_buffers,
                blk.annotations,
            )
            updates[gv] = func.with_body(tir.BlockRealize(body.iter_values, body.predicate, new_blk))
        else:
            updates[gv] = func.with_body(tir.stmt_seq(begin, body, end))
    for gv, f in updates.items():
        mod[gv] = f
    return mod
