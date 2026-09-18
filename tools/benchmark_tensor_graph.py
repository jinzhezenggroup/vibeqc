"""Opt-in complete TensorIR endpoint benchmark for ordinary vs graph replay.

Run only inside an allocated CUDA job. This is a launch-sensitive tensor region,
not a full SCF/CC solver. No performance profile is promoted by running it.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np
from vibeqc.profiles import find_nvcc
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    input_tensor,
    multiply,
    reduce_sum,
)
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_plan import plan_cuda


def benchmark(args):
    nvcc = find_nvcc()
    if nvcc is None:
        raise RuntimeError("set VIBEQC_NVCC to a CUDA compiler")
    compiler = CudaCompilerAdapter(nvcc, cuda_target_info(args.arch))
    axis = Index("i", IndexSpace("graph_axis", "batch", args.size))
    spec = TensorSpec((axis,), role="input")
    x, scale, bias = (input_tensor(name, spec) for name in ("x", "scale", "bias"))
    value = x
    for _ in range(args.depth):
        value = add(multiply(value, scale), bias)
    program = Program({"result": value, "sum": reduce_sum(value, (0,))})
    plan = plan_cuda(program, compiler.target)
    artifact = compile_cuda(plan, compiler, args.cache)
    with (
        PreparedCuda(plan, artifact) as ordinary,
        PreparedCuda(plan, artifact, execution_mode="cuda-graph") as graph,
    ):
        base = np.linspace(-0.25, 0.75, args.size)
        feeds = {
            "x": base.copy(),
            "scale": np.full(args.size, 0.99),
            "bias": np.full(args.size, 0.001),
        }
        warmup = graph.execute(feeds)
        capture = graph.execute(feeds)
        if capture.metrics["graph_mode"] != "captured":
            raise RuntimeError(f"graph qualification fell back: {graph.graph_status}")
        for _ in range(5):
            ordinary.execute(feeds)
            graph.execute(feeds)
        trials = []
        max_error = 0.0
        for trial in range(args.trials):
            records = {name: [] for name in ("ordinary", "graph")}
            for iteration in range(args.repeats):
                if args.changed_inputs:
                    # Updates numeric inputs, not scientific topology or shape.
                    feeds["x"][:] = base + 1e-4 * iteration
                order = (("ordinary", ordinary), ("graph", graph))
                if (trial + iteration) % 2:
                    order = order[::-1]
                results = {}
                for name, prepared in order:
                    results[name] = prepared.execute(feeds, diagnostics=True)
                    records[name].append(results[name].metrics)
                reference = feeds["x"].copy()
                for _ in range(args.depth):
                    reference = reference * feeds["scale"] + feeds["bias"]
                for result in results.values():
                    max_error = max(
                        max_error,
                        float(np.max(np.abs(result.outputs["result"] - reference))),
                    )
                    np.testing.assert_allclose(
                        result.outputs["result"], reference, rtol=2e-12, atol=2e-13
                    )
                    np.testing.assert_allclose(
                        result.outputs["sum"], reference.sum(), rtol=2e-12, atol=2e-12
                    )
                assert results["graph"].metrics["graph_captures"] == 1
            trials.append(
                {
                    name: {
                        field: statistics.median(row[field] for row in rows)
                        for field in ("endpoint_ms", "device_ms", "graph_submission_ms")
                    }
                    for name, rows in records.items()
                }
            )
        ordinary_ms = statistics.median(t["ordinary"]["endpoint_ms"] for t in trials)
        graph_ms = statistics.median(t["graph"]["endpoint_ms"] for t in trials)
        setup_ms = (
            capture.metrics["graph_capture_ms"]
            + capture.metrics["graph_instantiate_ms"]
        )
        result = {
            "schema": "vibeqc.tensor.graph-benchmark.v1",
            "scope": "complete host-staged TensorIR endpoint; not an iterative electronic-structure endpoint",
            "size": args.size,
            "depth": args.depth,
            "trials": args.trials,
            "repeats_per_trial": args.repeats,
            "changed_numeric_inputs": args.changed_inputs,
            "plan": plan.identity,
            "artifact": artifact.metadata["key"],
            "capture_contract": graph.capture_contract.identity,
            "device": graph.device,
            "compiler": str(compiler.nvcc),
            "ordinary_endpoint_ms": ordinary_ms,
            "graph_endpoint_ms": graph_ms,
            "endpoint_speedup": ordinary_ms / graph_ms,
            "capture_ms": capture.metrics["graph_capture_ms"],
            "instantiate_ms": capture.metrics["graph_instantiate_ms"],
            "cold_warmup_endpoint_ms": warmup.metrics["endpoint_ms"],
            "capture_endpoint_ms": capture.metrics["endpoint_ms"],
            "setup_cost_scope": "capture + instantiate CPU time; excludes initial warmup and first-launch upload",
            "event_time_scope": "transfers + device region including host submission gaps; not kernel-only time",
            "amortization_replays": setup_ms / (ordinary_ms - graph_ms)
            if ordinary_ms > graph_ms
            else None,
            "graph_nodes": capture.metrics["graph_node_count"],
            "graph_retained_device_delta_bytes": capture.metrics[
                "graph_retained_device_bytes"
            ],
            "graph_memory_scope": "conservative device free-memory delta around capture/instantiate; driver-owned host graph storage unmeasured",
            "mandatory_completion_fences_per_endpoint": 1,
            "submission_scope": "device region only; excludes input/output staging",
            "last_ordinary_submission_ms": results["ordinary"].metrics[
                "graph_submission_ms"
            ],
            "last_graph_submission_ms": results["graph"].metrics["graph_submission_ms"],
            "max_absolute_error": max_error,
            "trial_medians": trials,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch", default=os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120")
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--changed-inputs", action="store_true")
    args = parser.parse_args()
    if any(getattr(args, k) < 1 for k in ("size", "depth", "repeats", "trials")):
        parser.error("size/depth/repeats/trials must be positive")
    benchmark(args)


if __name__ == "__main__":
    main()
