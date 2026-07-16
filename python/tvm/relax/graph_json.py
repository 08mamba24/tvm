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
# [fork extension] operator-structure JSON export for Netron visualization.
"""Emit a TVM-0.9 graph_executor style graph.json from a relax module.

The relax VM has no graph.json (bytecode replaced the graph), so this walks a
relax function's call_tir DAG and emits the JSON schema the 0.9 loader used
(tvm-v0.9 graph_executor.h:257-384): nodes(op/name/inputs, tvm_op attrs
func_name/num_inputs/num_outputs/flatten_data), arg_nodes, heads,
node_row_ptr, attrs{dltype:list_str, shape:list_shape, storage_id:list_int}.
Netron opens this format directly (same NNVM lineage as MXNet symbol json) —
each node is one TIR kernel, mirroring what timeline/profiler events time.

One addition on top of the 0.9 schema: each tvm_op node carries a display-only
``out_type`` attr (e.g. ``float32[100, 512]``) because Netron's tvm.js renders
node attrs but ignores the graph-level dltype/shape arrays. Strip the key if
the JSON ever needs to feed a real 0.9 graph_executor.

Call it on the SAME mod that goes into relax.build (post-LegalizeOps, so the
kernel names exist), which guarantees graph and .so come from one IR:

.. code-block:: python

    from tvm.relax.graph_json import emit_graph_json

    mod = relax.transform.LegalizeOps()(mod)
    emit_graph_json(mod, "model_graph.json")
    ex = relax.build(mod, target, exec_mode=...)
"""
import json

import tvm

from . import expr as rx


def emit_graph_json(mod, out_path, entry="main"):
    """Walk mod[entry]'s call_tir DAG and write 0.9 graph_executor JSON.

    Parameters
    ----------
    mod : tvm.IRModule
        Relax module after LegalizeOps (call_tir bindings present).
    out_path : str
        Where to write the Netron-openable JSON.
    entry : str
        Relax entry function name to walk.

    Returns
    -------
    graph : dict
        The graph structure that was written.
    """
    main = mod[entry]

    nodes, arg_nodes, dltype, shapes = [], [], [], []
    node_row_ptr = [0]
    entry_of = {}  # relax.Var -> (node_id, output_index)
    tuple_vals = {}  # relax.Var bound to a relax.Tuple -> [entry refs]
    name_count = {}
    stats = {"const": 0, "skipped_args": 0, "unhandled": []}

    def uniq(name):
        k = name_count.get(name, 0)
        name_count[name] = k + 1
        return name if k == 0 else f"{name}_{k}"

    def sinfo_fields(si):
        return [int(d) for d in si.shape.values], str(si.dtype)

    def type_str(out_sinfos):
        # Netron's tvm.js ignores the graph-level dltype/shape arrays but
        # renders every node attrs entry, so carry a display-only copy here.
        parts = []
        for si in out_sinfos:
            sh, dt = sinfo_fields(si)
            parts.append(f"{dt}[{', '.join(map(str, sh))}]")
        return ", ".join(parts)

    def add_node(op, name, inputs, out_sinfos, attrs=None):
        nid = len(nodes)
        node = {"op": op, "name": name, "inputs": inputs}
        if attrs is not None:
            node["attrs"] = attrs
        nodes.append(node)
        for si in out_sinfos:
            sh, dt = sinfo_fields(si)
            shapes.append(sh)
            dltype.append(dt)
        node_row_ptr.append(node_row_ptr[-1] + len(out_sinfos))
        return nid

    def ref(expr):
        """expr -> [node, out_idx, 0] input ref, or None for non-tensor args."""
        if isinstance(expr, rx.Var):
            if expr in entry_of:
                n, i = entry_of[expr]
                return [n, i, 0]
            return None
        if isinstance(expr, rx.Constant):
            nid = add_node("null", f"p{stats['const']}", [], [expr.struct_info])
            arg_nodes.append(nid)
            stats["const"] += 1
            entry_of[expr] = (nid, 0)
            return [nid, 0, 0]
        stats["skipped_args"] += 1  # ShapeExpr / PrimValue etc.
        return None

    for p in main.params:
        nid = add_node("null", p.name_hint, [], [p.struct_info])
        arg_nodes.append(nid)
        entry_of[p] = (nid, 0)

    call_tir_op = tvm.ir.Op.get("relax.call_tir")
    tuple_sinfo = tvm.relax.TupleStructInfo

    for blk in main.body.blocks:
        for bd in blk.bindings:
            if not isinstance(bd, rx.VarBinding):
                stats["unhandled"].append(type(bd).__name__)
                continue
            v = bd.value
            if isinstance(v, rx.Call) and v.op == call_tir_op:
                fname = v.args[0].name_hint
                in_refs = [r for r in (ref(x) for x in v.args[1].fields) if r]
                si = bd.var.struct_info
                outs = list(si.fields) if isinstance(si, tuple_sinfo) else [si]
                nid = add_node(
                    "tvm_op",
                    uniq(fname),
                    in_refs,
                    outs,
                    attrs={
                        "func_name": fname,
                        "num_inputs": str(len(in_refs)),
                        "num_outputs": str(len(outs)),
                        "flatten_data": "0",
                        "out_type": type_str(outs),
                    },
                )
                entry_of[bd.var] = (nid, 0)
            elif isinstance(v, rx.Call):  # relax op that survived legalization
                opname = v.op.name if isinstance(v.op, tvm.ir.Op) else str(v.op)
                in_refs = [r for r in (ref(x) for x in v.args) if r]
                si = bd.var.struct_info
                outs = list(si.fields) if isinstance(si, tuple_sinfo) else [si]
                nid = add_node(
                    "tvm_op",
                    uniq(opname.replace(".", "_")),
                    in_refs,
                    outs,
                    attrs={
                        "func_name": opname,
                        "num_inputs": str(len(in_refs)),
                        "num_outputs": str(len(outs)),
                        "flatten_data": "0",
                        "out_type": type_str(outs),
                    },
                )
                entry_of[bd.var] = (nid, 0)
            elif isinstance(v, rx.TupleGetItem):
                n, _ = entry_of[v.tuple_value]
                entry_of[bd.var] = (n, v.index)
            elif isinstance(v, rx.Var):
                if v in entry_of:
                    entry_of[bd.var] = entry_of[v]
                elif v in tuple_vals:
                    tuple_vals[bd.var] = tuple_vals[v]
            elif isinstance(v, rx.Tuple):
                tuple_vals[bd.var] = [r for r in (ref(f) for f in v.fields) if r]
            else:
                stats["unhandled"].append(type(v).__name__)

    ret = main.body.body
    if isinstance(ret, rx.Tuple):
        heads = [r for r in (ref(f) for f in ret.fields) if r]
    elif isinstance(ret, rx.Var) and ret in tuple_vals:
        heads = tuple_vals[ret]
    else:
        n, i = entry_of[ret]
        heads = [[n, i, 0]]

    graph = {
        "nodes": nodes,
        "arg_nodes": arg_nodes,
        "heads": heads,
        "attrs": {
            "dltype": ["list_str", dltype],
            "shape": ["list_shape", shapes],
            "storage_id": ["list_int", list(range(node_row_ptr[-1]))],
        },
        "node_row_ptr": node_row_ptr,
    }
    with open(out_path, "w") as f:
        json.dump(graph, f, indent=1)

    n_op = sum(1 for n in nodes if n["op"] == "tvm_op")
    print(
        f"[graph_json] wrote {out_path}: {len(nodes)} nodes "
        f"({n_op} tvm_op, {len(arg_nodes)} null), heads={heads}"
    )
    if stats["skipped_args"] or stats["unhandled"]:
        print(
            f"[graph_json] skipped non-tensor args={stats['skipped_args']}, "
            f"unhandled bindings={stats['unhandled']}"
        )
    return graph
