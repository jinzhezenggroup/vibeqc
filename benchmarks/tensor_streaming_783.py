"""Real-device qualification for the #783 TensorIR streaming schedule."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import Program, execute
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda

from benchmarks._support import raw_output_path, write_result
from tools.vibeqc_cc.triples_tiles import (
    TileSpec,
    build_runtime_tile_triples_program,
    runtime_tile_controls,
    runtime_tile_static_feeds,
)


def _arrays(nocc: int, nvir: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t2 = rng.normal(size=(nocc, nocc, nvir, nvir))
    t2 = (t2 + t2.transpose(1, 0, 3, 2)) / 2
    return {
        "ovvv": rng.normal(size=(nocc, nvir, nvir, nvir)),
        "ovoo": rng.normal(size=(nocc, nvir, nocc, nocc)),
        "ovov": rng.normal(size=(nocc, nvir, nocc, nvir)),
        "fov": rng.normal(size=(nocc, nvir)),
        "t1": rng.normal(size=(nocc, nvir)),
        "t2": t2,
        "eps_o": np.linspace(-1.0, -0.5, nocc),
        "eps_v": np.linspace(0.5, 1.5, nvir),
    }


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _variant(
    label: str,
    program: Program,
    feeds: dict[str, np.ndarray],
    expected: float,
    compiler: CudaCompilerAdapter,
    schedule: TensorSchedule,
    repeats: int,
    max_bytes: int,
) -> dict:
    plan = plan_cuda(
        program,
        compiler.target,
        schedule=schedule,
        max_bytes=max_bytes,
    )
    with tempfile.TemporaryDirectory(prefix=f"vibeqc-783-{label}-") as directory:
        start = time.perf_counter()
        artifact = compile_cuda(plan, compiler, Path(directory))
        compile_seconds = time.perf_counter() - start
        with PreparedCuda(plan, artifact) as prepared:
            prepared.execute(feeds)
            samples = []
            actual = None
            for _ in range(repeats):
                start = time.perf_counter()
                actual = float(prepared.execute(feeds).outputs["triples_energy"])
                samples.append(time.perf_counter() - start)
    assert actual is not None
    return {
        "label": label,
        "schedule": plan.to_payload()["schedule"],
        "plan_identity": plan.identity,
        "arena_bytes": plan.arena_bytes,
        "peak_bytes": plan.peak_bytes,
        "provider_bytes": plan.provider_bytes,
        "virtual_steps": sum(step.virtual for step in plan.steps),
        "gemm_steps": sum(step.gemm != "none" for step in plan.steps),
        "compile_seconds": compile_seconds,
        "run_seconds": samples,
        "median_run_seconds": statistics.median(samples),
        "energy": actual,
        "absolute_error": abs(actual - expected),
        "relative_error": abs(actual - expected) / max(abs(expected), 1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--nocc", type=int, required=True)
    parser.add_argument("--nvir", type=int, required=True)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--seed", type=int, default=783)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-bytes-mib", type=int, default=2048)
    parser.add_argument("--reference-max-bytes-mib", type=int, default=4096)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()

    if args.nocc < 1 or args.nvir < 1 or args.repeats < 1:
        raise ValueError("nocc, nvir, and repeats must be positive")
    triples = args.nvir * (args.nvir + 1) * (args.nvir + 2) // 6
    capacity = triples if args.capacity is None else args.capacity
    if capacity < triples:
        raise ValueError(
            "qualification capacity must contain the full triangular domain"
        )

    arrays = _arrays(args.nocc, args.nvir, args.seed)
    program = build_runtime_tile_triples_program(
        args.nocc,
        args.nvir,
        capacity=capacity,
    )
    feeds = {
        **runtime_tile_static_feeds(arrays),
        **runtime_tile_controls(TileSpec(0, args.nvir, args.nvir), capacity),
    }
    expected = float(
        execute(
            program,
            feeds,
            max_bytes=args.reference_max_bytes_mib << 20,
        ).outputs["triples_energy"]
    )
    compiler = CudaCompilerAdapter(
        args.nvcc,
        cuda_target_info(args.architecture),
    )
    max_bytes = args.max_bytes_mib << 20
    variants = [
        _variant(
            "baseline",
            program,
            feeds,
            expected,
            compiler,
            TensorSchedule(),
            args.repeats,
            max_bytes,
        ),
        _variant(
            "streaming_frontier",
            program,
            feeds,
            expected,
            compiler,
            TensorSchedule(views=True, stream_reductions=True),
            args.repeats,
            max_bytes,
        ),
    ]
    result = {
        "schema": "vibeqc.tensor.streaming-reduction-qualification/1",
        "issue": 783,
        "git_head": _command_output(["git", "rev-parse", "HEAD"]),
        "git_status": _command_output(["git", "status", "--porcelain"]),
        "device": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "nvcc": _command_output([str(args.nvcc), "--version"]),
        "target": compiler.target.to_payload(),
        "shape": {
            "nocc": args.nocc,
            "nvir": args.nvir,
            "capacity": capacity,
            "triangular_triples": triples,
        },
        "seed": args.seed,
        "repeats": args.repeats,
        "reference_energy": expected,
        "graph_nodes": len(program.live_nodes),
        "variants": variants,
        "conclusion": (
            "qualification-only: retain the baseline by default unless the "
            "streaming schedule wins an endpoint performance gate"
        ),
    }
    write_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
