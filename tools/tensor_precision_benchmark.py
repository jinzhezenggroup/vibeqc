"""Qualify TensorIR precision schedules on complete host-staged CUDA endpoints.

This benchmark compares strict FP64, qualified full FP32, and qualified FP32
compute with FP64 reduction accumulation for one matrix contraction. It records
all endpoint samples, numerical error, resource accounting, cast traffic and
plan identities. It never promotes a schedule or changes runtime defaults.
"""

from __future__ import annotations

import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
import statistics
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.provenance import atomic_json
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    PrecisionDirective,
    Program,
    TensorSpec,
    einsum,
    input_tensor,
    lower_precision,
)
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_plan import plan_cuda
from vibeqc_compiler.tensor.cuda_search import estimate_schedule


def fixture(size: int, seed: int) -> tuple[Program, dict[str, np.ndarray]]:
    space = IndexSpace(f"precision_bench_{size}", "batch", size)
    left = input_tensor(
        "left",
        TensorSpec(
            (Index("i", space), Index("k", space)),
            dtype="float64",
            role="parameter",
        ),
    )
    right = input_tensor(
        "right",
        TensorSpec(
            (Index("k2", space), Index("j", space)),
            dtype="float64",
            role="parameter",
        ),
    )
    contraction = einsum("ik,kj->ij", left, right)
    program = Program({"out": contraction})
    rng = np.random.default_rng(seed)
    feeds = {
        "left": rng.normal(size=(size, size)).astype(np.float64),
        "right": rng.normal(size=(size, size)).astype(np.float64),
    }
    return program, feeds


def variants(program: Program, qualification: str) -> dict[str, Program]:
    contraction = program.outputs["out"]
    name = program.debug_names[contraction]
    return {
        "fp64": program,
        "fp32": lower_precision(
            program,
            {
                name: PrecisionDirective(
                    "float32",
                    "float32",
                    "float32",
                    qualification=f"{qualification}/fp32",
                )
            },
        ),
        "fp32_compute_fp64_accum": lower_precision(
            program,
            {
                name: PrecisionDirective(
                    "float32",
                    "float32",
                    "float64",
                    qualification=f"{qualification}/fp32-compute-fp64-accum",
                )
            },
        ),
    }


def run(args: argparse.Namespace) -> dict:
    target = cuda_target_info(args.architecture)
    compiler = CudaCompilerAdapter(args.nvcc, target)
    program, feeds = fixture(args.size, args.seed)
    candidates = variants(
        program, f"issue528/{args.architecture}/{args.size}/seed-{args.seed}"
    )
    oracle = feeds["left"] @ feeds["right"]
    plans = {
        name: plan_cuda(candidate, target, max_bytes=args.max_bytes)
        for name, candidate in candidates.items()
    }
    artifacts = {
        name: compile_cuda(plan, compiler, args.output / "cache")
        for name, plan in plans.items()
    }

    samples = {name: [] for name in candidates}
    outputs: dict[str, np.ndarray] = {}
    names = tuple(candidates)
    with ExitStack() as stack:
        prepared = {
            name: stack.enter_context(PreparedCuda(plans[name], artifacts[name]))
            for name in names
        }
        for _ in range(args.warmups):
            for name in names:
                prepared[name].execute(feeds)
        for repeat in range(args.repeats):
            offset = repeat % len(names)
            order = names[offset:] + names[:offset]
            for name in order:
                result = prepared[name].execute(feeds)
                samples[name].append(result.metrics["endpoint_ms"])
                outputs[name] = result.outputs["out"]

    rows = {}
    for name in names:
        plan = plans[name]
        error = np.abs(outputs[name] - oracle)
        scale = np.maximum(np.abs(oracle), 1.0e-30)
        step = next(step for step in plan.steps if step.node.op == "einsum")
        estimate = estimate_schedule(plan)
        rows[name] = {
            "median_endpoint_ms": statistics.median(samples[name]),
            "samples_ms": samples[name],
            "max_abs_error": float(error.max()),
            "max_rel_error": float((error / scale).max()),
            "gemm": step.gemm,
            "peak_bytes": plan.peak_bytes,
            "device_bytes": plan.device_bytes,
            "host_bytes": plan.host_bytes,
            "cast_read_bytes": plan.precision_schedule.cast_read_bytes,
            "cast_write_bytes": plan.precision_schedule.cast_write_bytes,
            "fp64_accumulation_terms": estimate["estimated_fp64_accumulation_terms"],
            "precision_schedule_identity": plan.precision_schedule.identity,
            "plan_identity": plan.identity,
        }
    return {
        "schema": "vibeqc.tensor.precision-qualification.v1",
        "architecture": args.architecture,
        "size": args.size,
        "seed": args.seed,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "measurement": (
            "complete PreparedCuda.execute endpoint; variants interleaved with "
            "rotating first position"
        ),
        "oracle": "NumPy float64 matrix product",
        "promotion": "none; evidence only",
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=528)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.size < 1 or args.warmups < 0 or args.repeats < 3:
        raise SystemExit("size must be positive, warmups nonnegative, repeats >= 3")
    args.output.mkdir(parents=True, exist_ok=True)
    result = run(args)
    atomic_json(args.output / "precision.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
