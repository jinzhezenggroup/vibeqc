"""Qualify explicit producer layouts on complete host-staged TensorIR endpoints.

Run only within a finite allocated GPU job. Raw paired timings, all numerical
checks, static work/byte counts and compiled identities remain in local evidence;
this script never publishes artifacts, starts background work or changes defaults.
"""

from __future__ import annotations

import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.provenance import atomic_json
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    einsum,
    input_tensor,
    multiply,
)
from vibeqc_compiler.tensor.cuda_execute import tensor_source_identity
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
from vibeqc_compiler.tensor.cuda_tune import tune_cuda


def fixture(i, b, k, j, seed):
    """F[bij] = sum_k X[ibk]^2 Y[bkj]; no molecular-method claim."""
    dimensions = {"i": i, "b": b, "k": k, "j": j}
    spaces = {
        name: IndexSpace(name, "batch", size) for name, size in dimensions.items()
    }
    x, y = (
        input_tensor(
            name,
            TensorSpec(
                tuple(Index(axis, spaces[axis]) for axis in labels), role="input"
            ),
        )
        for name, labels in (("x", "ibk"), ("y", "bkj"))
    )
    program = Program({"out": einsum("ibk,bkj->bij", multiply(x, x), y)})
    rng = np.random.default_rng(seed)
    feeds = {
        "x": rng.uniform(-0.5, 0.5, x.spec.shape),
        "y": rng.uniform(-0.5, 0.5, y.spec.shape),
    }
    return program, feeds


def run(args):
    target = cuda_target_info(args.architecture)
    compiler = CudaCompilerAdapter(args.nvcc, target)
    program, feeds = fixture(*args.shape, args.seed)
    baseline = plan_cuda(program, target, max_bytes=args.max_bytes)
    candidate = plan_cuda(
        program, target, max_bytes=args.max_bytes, schedule=TensorSchedule(layouts=True)
    )
    fixtures = [
        feeds,
        {name: np.asfortranarray(value * 0.7) for name, value in feeds.items()},
    ]
    selection = tune_cuda(
        baseline,
        compiler,
        fixtures,
        args.output / "cache",
        schedules=[TensorSchedule(layouts=True)],
        repeats=args.repeats,
        maximum_seconds=args.maximum_seconds,
    )
    atomic_json(args.output / "evidence.json", selection.evidence)
    row = selection.evidence["candidates"][0]
    root = Path(__file__).resolve().parents[1]
    summary = {
        "schema": "vibeqc.tensor.layout-qualification.v1",
        "base_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        ),
        "tensor_source_identity": tensor_source_identity(),
        "equation": "out[bij] = sum_k x[ibk]^2 * y[bkj]",
        "logical_hash": program.logical_hash,
        "shape_ibkj": args.shape,
        "seed": args.seed,
        "device": selection.evidence["device"],
        "identity": selection.evidence["identity"],
        "baseline": {
            "plan": baseline.identity,
            "layout": baseline.layout_identity,
            "arena_bytes": baseline.arena_bytes,
            "panel_bytes": baseline.panel_bytes,
            "peak_bytes": baseline.peak_bytes,
            "layout_planning": baseline.layout_decision.to_payload(),
        },
        "candidate": {
            "plan": candidate.identity,
            "layout": candidate.layout_identity,
            "arena_bytes": candidate.arena_bytes,
            "panel_bytes": candidate.panel_bytes,
            "peak_bytes": candidate.peak_bytes,
            "layout_planning": candidate.layout_decision.to_payload(),
        },
        "baseline_startup": selection.evidence["baseline_startup"],
        "status": row["status"],
        **{
            name: row[name]
            for name in (
                "gates",
                "shared_gates",
                "samples",
                "profiles",
                "max_absolute_error",
                "reason",
            )
            if name in row
        },
        "artifact_keys": {
            "baseline": selection.evidence["baseline_artifact"]["key"],
            "candidate": row.get("artifact", {}).get("key"),
        },
        "selected_plan": selection.plan.identity,
        "scope": "complete warmed TensorIR endpoint, including validation/staging/transfers/result allocation; not molecular CCSD/DFT speedup",
    }
    atomic_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "status",
                    "baseline",
                    "candidate",
                    "gates",
                    "max_absolute_error",
                )
                if key in summary
            },
            indent=2,
        )
    )
    return 0 if row["status"] == "accepted" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--shape",
        nargs=4,
        type=int,
        default=[33, 7, 65, 31],
        metavar=("I", "B", "K", "J"),
    )
    parser.add_argument("--seed", type=int, default=509)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--maximum-seconds", type=float, default=300)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024**2)
    args = parser.parse_args()
    if any(size < 1 for size in args.shape):
        parser.error("benchmark dimensions must be positive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
