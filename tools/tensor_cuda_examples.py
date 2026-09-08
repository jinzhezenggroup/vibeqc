"""Compile, validate or tune fixed TensorIR endpoints using CG01 evidence.

GPU modes must run inside the caller's finite device allocation. No external
quantum-chemistry program participates in generated tensor execution.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vibeqc.profiles import find_nvcc

from tools.vibeqc_codegen.cuda_adapter import CudaCompilerAdapter
from tools.vibeqc_codegen.cuda_target import cuda_target_info
from tools.vibeqc_tensor.cuda_execute import (
    PreparedCuda,
    compile_cuda,
    tensor_source_identity,
)
from tools.vibeqc_tensor.cuda_fixtures import cc_fixtures
from tools.vibeqc_tensor.cuda_plan import Reservations, TensorSchedule, plan_cuda
from tools.vibeqc_tensor.cuda_tune import tune_cuda
from tools.vibeqc_tensor.interpreter import execute
from tools.vibeqc_validation.schema import (
    GATES,
    block_error,
    canonical_hash,
    new_evidence,
    outcome,
    validate_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    """Retain full tuning ledgers and one shared-schema record per input scale."""
    nvcc = args.nvcc or find_nvcc()
    if nvcc is None:
        raise ValueError("provide --nvcc or VIBEQC_NVCC")
    compiler = CudaCompilerAdapter(
        nvcc, cuda_target_info(args.architecture), args.compile_timeout
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for nocc, nvir in args.shape:
        for case in cc_fixtures(nocc, nvir, seed=args.seed):
            label = f"{case.name}-o{nocc}-v{nvir}"
            amplitude_bytes = 8 * nocc * nocc * nvir * nvir
            reservations = Reservations(
                t=amplitude_bytes, r=amplitude_bytes, diis=4 * amplitude_bytes
            )
            baseline = plan_cuda(
                case.program,
                compiler.target,
                max_bytes=args.max_bytes,
                reservations=reservations,
            )
            selected_plan, artifact = (
                baseline,
                compile_cuda(baseline, compiler, args.cache),
            )
            fixtures = [
                {name: value * scale for name, value in case.inputs.items()}
                for scale in args.scale
            ]
            tuning = None
            if args.mode == "tune":
                tuning = tune_cuda(
                    baseline,
                    compiler,
                    fixtures,
                    args.cache,
                    repeats=args.repeats,
                    maximum_seconds=args.tune_seconds,
                    device=args.device,
                )
                selected_plan, artifact = tuning.plan, tuning.artifact
                (args.output / f"{label}-tuning.json").write_text(
                    json.dumps(tuning.evidence, indent=2, sort_keys=True) + "\n"
                )
            for index, (scale, feeds) in enumerate(
                zip(args.scale, fixtures, strict=True)
            ):
                inputs_hash = canonical_hash(
                    {
                        "equation": case.program.logical_hash,
                        "seed": case.seed,
                        "shape": [nocc, nvir],
                        "scale": scale,
                        "arrays": {
                            name: canonical_hash(value.tolist())
                            for name, value in feeds.items()
                        },
                    }
                )
                record = new_evidence(
                    tier="cuda-compile"
                    if args.mode == "compile"
                    else "gpu-numerical"
                    if args.mode == "numerical"
                    else "endpoint",
                    subject=f"TensorIR CUDA/{label}/scale{scale:g}",
                    inputs_hash=inputs_hash,
                )
                record.update(
                    revision=revision,
                    backend_selected="cuda",
                    toolchain={
                        **artifact.metadata["identity"]["toolchain"],
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                    },
                    settings={
                        "device": "cuda",
                        "fast_compile": False,
                        "dirty": dirty,
                        "seed": case.seed,
                        "nocc": nocc,
                        "nvir": nvir,
                        "scale": scale,
                        "scope": "complete fixed TensorIR endpoint; no CC solver or molecular method",
                        "reference": "independent CPU TensorIR interpreter",
                        "comparison_kind": "kernel",
                        "fixed_state_hash": inputs_hash,
                        "workload_semantics": "unchanged-geometry denotes fixed supplied tensors; this endpoint has no geometry",
                        "promotion_limits": {
                            "peak_bytes": args.max_bytes,
                            "compile_seconds": args.compile_timeout,
                        },
                        "plan": selected_plan.to_payload(),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    },
                )
                record["hashes"].update(
                    equation=case.program.logical_hash,
                    ir=canonical_hash(case.program.to_payload()),
                    source=tensor_source_identity(),
                    schedule=selected_plan.identity,
                )
                record["compilation"] = {
                    "seconds": artifact.metadata["compile_seconds"],
                    "reason": None,
                    "resources": artifact.metadata["resources"],
                    "binary_sha256": artifact.metadata["binary_sha256"],
                }
                for stage in ("representation", "source", "compilation"):
                    record["stages"][stage] = outcome("pass")
                if args.mode != "compile":
                    reference = execute(case.program, feeds).outputs
                    with PreparedCuda(
                        selected_plan, artifact, device=args.device
                    ) as prepared:
                        result = prepared.execute(feeds)
                        profile = prepared.execute(feeds, profile=True)
                        record.update(device=prepared.device, hardware=outcome("pass"))
                        record["settings"]["graph_status"] = prepared.graph_status
                    for name, value in reference.items():
                        record["block_errors"][name] = block_error(
                            result.outputs[name], value, **GATES["integral_fp64"]
                        )
                    # Force minimum packing/recomputation independently of
                    # whether tuning chose either strategy for performance.
                    constrained = plan_cuda(
                        case.program,
                        compiler.target,
                        schedule=TensorSchedule(
                            views=True,
                            fuse=True,
                            recompute=True,
                            direct_gemm=False,
                            tile_m=7,
                            tile_n=29,
                            tile_k=7,
                        ),
                        reservations=reservations,
                        library_bytes=0,
                        max_bytes=args.max_bytes,
                    )
                    constrained = plan_cuda(
                        case.program,
                        compiler.target,
                        schedule=constrained.schedule,
                        reservations=reservations,
                        library_bytes=0,
                        max_bytes=constrained.peak_bytes,
                    )
                    with PreparedCuda(
                        constrained,
                        compile_cuda(constrained, compiler, args.cache),
                        device=args.device,
                    ) as prepared:
                        constrained_result = prepared.execute(feeds)
                    for name, value in reference.items():
                        record["block_errors"]["constrained_" + name] = block_error(
                            constrained_result.outputs[name],
                            value,
                            **GATES["integral_fp64"],
                        )
                    record["settings"].update(
                        endpoint_metrics=result.metrics,
                        profile_metrics=profile.metrics,
                        constrained_plan=constrained.to_payload(),
                        constrained_metrics=constrained_result.metrics,
                    )
                    record["memory"] = {
                        "allocated_bytes": selected_plan.allocation_bytes
                        + profile.metrics["provider_retained_bytes"],
                        "peak_bytes": selected_plan.peak_bytes,
                        "reason": None,
                        "scope": "native allocation, checked provider-retained allowance, and conservative host staging/validation/current-output capacity; excludes general CUDA context/module/stack overhead and caller-owned arrays",
                        "provider_retained_bytes": profile.metrics[
                            "provider_retained_bytes"
                        ],
                        "provider_allowance_bytes": selected_plan.provider_bytes,
                        "observed_device_delta": profile.metrics[
                            "observed_device_delta"
                        ],
                        "additional_runtime_and_allocator_delta": max(
                            0,
                            profile.metrics["observed_device_delta"]
                            - selected_plan.allocation_bytes
                            - profile.metrics["provider_retained_bytes"],
                        ),
                    }
                    passed = all(e["passed"] for e in record["block_errors"].values())
                    record["stages"]["numerical"] = outcome(
                        "pass" if passed else "fail",
                        None if passed else "TensorIR numerical gate failed",
                    )
                    record["stages"]["endpoint"] = outcome(
                        "pass" if passed else "fail",
                        None if passed else "whole program parity failed",
                    )
                if tuning:
                    winner = next(
                        (
                            c
                            for c in tuning.evidence["candidates"]
                            if c.get("plan_identity") == selected_plan.identity
                            and c["status"] == "accepted"
                        ),
                        None,
                    )
                    if winner:
                        record["timings"] = winner["samples"][index]
                        record["performance"] = outcome("pass")
                        record["stages"]["production"] = outcome(
                            "pass",
                            scope="this exact TensorIR plan and measured fixture bucket",
                        )
                    else:
                        record["performance"] = outcome(
                            "not-run",
                            "no candidate passed all endpoint non-regression gates; retain baseline",
                        )
                        record["stages"]["production"] = outcome(
                            "not-run", "no optimized plan selected"
                        )
                    record["attachments"].append(
                        {
                            "kind": "tensor-tuning",
                            "path": f"{label}-tuning.json",
                            "schema_version": 1,
                        }
                    )
                validate_evidence(record)
                records.append(record)
                print(
                    f"{label} scale={scale:g}: {record['stages']['numerical']['status']}; optimization={record['performance']['status']}",
                    flush=True,
                )
            (args.output / f"{label}-equation.json").write_text(case.program.dumps())
            # Write incrementally so a later failed case does not erase earlier
            # measured records; the process still returns failure to the caller.
            (args.output / "evidence.json").write_text(
                json.dumps(records, indent=2, sort_keys=True) + "\n"
            )
    return records


def _shape(value):
    try:
        result = tuple(int(n) for n in value.split(","))
        if len(result) != 2 or min(result) < 1:
            raise ValueError
        return result
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be positive nocc,nvir") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("compile", "numerical", "tune"), default="numerical"
    )
    parser.add_argument("--nvcc", type=Path)
    parser.add_argument(
        "--architecture",
        default="sm_120",
        help="explicit target; preparation rejects a mismatched allocated GPU",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="visible CUDA ordinal; visibility is never overridden",
    )
    parser.add_argument(
        "--shape",
        type=_shape,
        action="append",
        help="repeat nocc,nvir shape buckets; default 3,7 and 5,17 and 8,31",
    )
    parser.add_argument("--scale", type=float, nargs="+", default=(0.001, 1.0, 100.0))
    parser.add_argument("--seed", type=int, default=146)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--tune-seconds", type=float, default=300)
    parser.add_argument("--compile-timeout", type=float, default=120)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--cache", type=Path, default=Path("build/tensor-cuda-cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.shape = args.shape or ((3, 7), (5, 17), (8, 31))
    if any(not np.isfinite(s) or s == 0 for s in args.scale):
        parser.error("scales must be finite and nonzero")
    records = run(args)
    return int(any(r["stages"]["numerical"]["status"] == "fail" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
