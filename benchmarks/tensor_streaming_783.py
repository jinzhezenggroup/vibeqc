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
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import Program, execute
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_plan import (
    TensorSchedule,
    estimated_cuda_launches,
    plan_cuda,
)

from benchmarks._support import raw_output_path, write_result
from tools.vibeqc_cc.triples_tiles import (
    TileSpec,
    build_runtime_tile_triples_program,
    runtime_tile_controls,
    runtime_tile_static_feeds,
    tile_triples_energy,
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
    changed_feeds: dict[str, np.ndarray],
    expected: float,
    changed_expected: float,
    compiler: CudaCompilerAdapter,
    schedule: TensorSchedule,
    repeats: int,
    max_bytes: int,
) -> dict:
    plan_start = time.perf_counter()
    plan = plan_cuda(
        program,
        compiler.target,
        schedule=schedule,
        max_bytes=max_bytes,
    )
    plan_seconds = time.perf_counter() - plan_start
    with tempfile.TemporaryDirectory(prefix=f"vibeqc-783-{label}-") as directory:
        start = time.perf_counter()
        artifact = compile_cuda(plan, compiler, Path(directory))
        compile_seconds = time.perf_counter() - start
        with PreparedCuda(plan, artifact) as prepared:
            start = time.perf_counter()
            prepared.execute(feeds)
            cold_run_seconds = time.perf_counter() - start
            samples = []
            device_samples = []
            actual = None
            for _ in range(repeats):
                start = time.perf_counter()
                execution = prepared.execute(feeds)
                actual = float(execution.outputs["triples_energy"])
                samples.append(time.perf_counter() - start)
                device_samples.append(execution.metrics["device_ms"])
            changed = float(prepared.execute(changed_feeds).outputs["triples_energy"])
            invalid = dict(changed_feeds)
            invalid["a_map"] = changed_feeds["a_map"].copy()
            invalid["a_map"][0] = feeds["eps_v"].size
            try:
                prepared.execute(invalid)
            except RuntimeError as error:
                invalid_diagnostic = str(error)
            else:
                raise AssertionError("out-of-bounds runtime map was accepted")
            replay = float(prepared.execute(feeds).outputs["triples_energy"])
    assert actual is not None
    if not np.isclose(changed, changed_expected, atol=1e-10, rtol=1e-10):
        raise AssertionError("changed runtime tile disagrees with CPU tile oracle")
    if not np.isclose(replay, expected, atol=1e-10, rtol=1e-10):
        raise AssertionError("replay after runtime map failure disagrees with oracle")
    return {
        "label": label,
        "schedule": plan.to_payload()["schedule"],
        "plan_identity": plan.identity,
        "artifact_key": artifact.metadata["key"],
        "artifact_binary_sha256": artifact.metadata["binary_sha256"],
        "arena_bytes": plan.arena_bytes,
        "peak_bytes": plan.peak_bytes,
        "provider_bytes": plan.provider_bytes,
        "virtual_steps": sum(step.virtual for step in plan.steps),
        "gemm_steps": sum(step.gemm != "none" for step in plan.steps),
        "estimated_launches": estimated_cuda_launches(plan),
        "estimated_flops": plan.estimated_flops,
        "semantic_traffic": plan.semantic_traffic,
        "plan_seconds": plan_seconds,
        "compile_seconds": compile_seconds,
        "cold_run_seconds": cold_run_seconds,
        "run_seconds": samples,
        "device_ms": device_samples,
        "median_run_seconds": statistics.median(samples),
        "median_device_ms": statistics.median(device_samples),
        "energy": actual,
        "changed_tile_energy": changed,
        "changed_tile_absolute_error": abs(changed - changed_expected),
        "replayed_energy_after_invalid_map": replay,
        "invalid_map_diagnostic": invalid_diagnostic,
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
    changed_tile = TileSpec(0, max(1, args.nvir // 2), args.nvir)
    changed_feeds = {
        **runtime_tile_static_feeds(arrays),
        **runtime_tile_controls(changed_tile, capacity),
    }
    expected = float(
        execute(
            program,
            feeds,
            max_bytes=args.reference_max_bytes_mib << 20,
        ).outputs["triples_energy"]
    )
    independent_energy = tile_triples_energy(
        TileSpec(0, args.nvir, args.nvir),
        args.nocc,
        *(
            arrays[name]
            for name in ("ovvv", "ovoo", "ovov", "fov", "t1", "t2", "eps_o", "eps_v")
        ),
    )
    changed_independent_energy = tile_triples_energy(
        changed_tile,
        args.nocc,
        *(
            arrays[name]
            for name in ("ovvv", "ovoo", "ovov", "fov", "t1", "t2", "eps_o", "eps_v")
        ),
    )
    if not np.isclose(expected, independent_energy, atol=1e-10, rtol=1e-10):
        raise AssertionError(
            "TensorIR reference differs from the independent CPU tile oracle"
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
            changed_feeds,
            independent_energy,
            changed_independent_energy,
            compiler,
            TensorSchedule(),
            args.repeats,
            max_bytes,
        ),
        _variant(
            "streaming_frontier",
            program,
            feeds,
            changed_feeds,
            independent_energy,
            changed_independent_energy,
            compiler,
            TensorSchedule(views=True, stream_reductions=True),
            args.repeats,
            max_bytes,
        ),
        _variant(
            "streamed_generated_reduction",
            program,
            feeds,
            changed_feeds,
            independent_energy,
            changed_independent_energy,
            compiler,
            TensorSchedule(
                views=True,
                stream_reductions=True,
                streamed_gemm_reduction=True,
            ),
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
        "independent_tile_energy": independent_energy,
        "changed_tile_independent_energy": changed_independent_energy,
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
