"""Archive CPU/compile/native-GPU XC diagnostics and explicit local timings.

GPU tiers must run under a finite Slurm GPU allocation on this workstation.
The endpoint tier times bounded feature upload -> XC -> output consumption; it
is not an SCF endpoint and cannot promote public DFT or a production schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

import numpy as np
from vibeqc.profiles import atomic_json, canonical_hash

from tools.vibeqc_codegen.cuda_adapter import CudaCompilerAdapter
from tools.vibeqc_codegen.cuda_target import cuda_target_info
from tools.vibeqc_validation.schema import (
    block_error,
    new_evidence,
    outcome,
    validate_evidence,
)
from tools.vibeqc_xc import build_program, functional
from tools.vibeqc_xc.capabilities import query_capability
from tools.vibeqc_xc.cuda import CudaXC, compile_cuda
from tools.vibeqc_xc.cuda_emit import XCSchedule, emit_cuda
from tools.vibeqc_xc.fixtures import load_fixture
from tools.vibeqc_xc.spec import CATALOG


def run(args, name, spin, variant, device):
    """Record each candidate's independent numerical and resource evidence."""
    start = perf_counter()
    program = build_program(functional(name, spin=spin), order=args.order)
    build_seconds = perf_counter() - start
    schedule = XCSchedule(variant, args.threads, args.group_size)
    source, contract, _ = emit_cuda(program, schedule)
    folder = args.output / f"{name}-{spin}-{variant}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "xc.cu").write_text(source)
    atomic_json(folder / "contract.json", contract)
    record = new_evidence(
        tier=args.tier,
        subject=f"XC/{name}/{spin}/{variant}",
        inputs_hash=program.expression_hash,
    )
    record.update(
        {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "identity": contract["identity"],
            "device": device,
            "backend_selected": "cpu" if args.tier == "cpu" else "cuda",
            "build_seconds": build_seconds,
        }
    )
    record["dirty"] = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    record["hashes"] = {
        "equation": program.spec.identity,
        "ir": program.expression_hash,
        "source": hashlib.sha256(source.encode()).hexdigest(),
        "schedule": contract["identity"],
    }
    record["settings"] = {
        "order": args.order,
        "variant": variant,
        "threads": args.threads,
        "tile_points": args.tile_points,
        "points": args.points,
        "repeats": args.repeats,
        "fast_compile": False,
        "device": "cpu" if args.tier == "cpu" else "cuda",
        "scope": "bounded feature-array consumer; no SCF/force endpoint",
    }
    record["stages"]["representation"] = outcome("pass")
    record["stages"]["source"] = outcome("pass")
    artifact = None
    if args.tier != "cpu":
        compiler = CudaCompilerAdapter(args.nvcc, cuda_target_info(args.target))
        start = perf_counter()
        artifact = compile_cuda(program, compiler, args.cache, schedule=schedule)
        record["prepare_compile_wall_seconds"] = perf_counter() - start
        record["binary_sha256"] = artifact.runtime.metadata["binary_sha256"]
        record["toolchain"] = artifact.runtime.metadata["identity"]
        record["compilation"] = {
            "seconds": artifact.runtime.metadata["compile_seconds"],
            "reason": None,
        }
        record["resources"] = artifact.runtime.metadata["resources"]
        record["stages"]["compilation"] = outcome("pass")
        record["spill_free"] = all(
            r["spill_load_bytes"] == r["spill_store_bytes"] == r["stack_bytes"] == 0
            for r in record["resources"]
        )
        (folder / "compiler.log").write_text(
            (artifact.runtime.library.parent / "compiler.log").read_text()
        )
    if args.tier in ("gpu-numerical", "endpoint"):
        record["hardware"] = outcome(
            "pass", probe="nvidia-smi and native architecture check"
        )
        start = perf_counter()
        gpu = CudaXC(
            program,
            artifact,
            tile_points=args.tile_points,
            budget_bytes=args.budget_bytes,
        )
        record["prepare_seconds"] = perf_counter() - start
        evaluate = gpu.evaluate
    else:
        gpu = None
        evaluate = program.evaluate
    try:
        if args.tier != "cuda-compile":
            diagnostics = {}
            for domain in ("typical", "boundary"):
                meta, x, expected, raw = load_fixture(name, spin=spin, domain=domain)
                expected, raw = (
                    expected[: len(program.outputs)],
                    raw[: len(program.outputs)],
                )
                actual = np.concatenate(
                    [
                        evaluate(x[:, i : i + args.tile_points])
                        for i in range(0, x.shape[1], args.tile_points)
                    ],
                    axis=1,
                )
                tolerance = meta[f"{domain}_tolerance"]
                record["block_errors"][domain] = block_error(
                    actual, expected, **tolerance
                )
                # This diagnostic is deliberately separate from the oracle gate:
                # raw Libxc exchange has documented polarized cancellation errors.
                diagnostics[domain] = {
                    "oracle": meta["oracle"],
                    "raw_libxc_error": block_error(actual, raw, **tolerance),
                    "reference_metadata": meta,
                }
            record["reference_diagnostics"] = diagnostics
            passed = all(v["passed"] for v in record["block_errors"].values())
            record["stages"]["numerical"] = outcome(
                "pass" if passed else "fail",
                None if passed else "independent XC gate failed",
            )
        if args.tier == "endpoint":
            _, tile, _, _ = load_fixture(name, spin=spin)
            features = np.tile(
                tile, (1, (args.points + tile.shape[1] - 1) // tile.shape[1])
            )[:, : args.points]
            # Consuming every derivative in a weighted checksum makes transfer
            # and host output handling visible. It is not a molecular energy.
            weights = np.linspace(0.5, 1.5, len(program.outputs))[:, None]
            input_hash = canonical_hash(features.tolist())

            def workflow(callback, values):
                result = np.zeros(len(program.outputs))
                for begin in range(0, values.shape[1], args.tile_points):
                    result += np.sum(
                        callback(values[:, begin : begin + args.tile_points]) * weights,
                        axis=1,
                    )
                return result

            target = workflow(program.evaluate, features)
            check = block_error(
                workflow(evaluate, features), target, atol=1e-9, rtol=1e-10
            )
            record["block_errors"]["consumer"] = check
            for repeat in range(args.repeats):
                order = (
                    ("baseline", "candidate")
                    if repeat % 2 == 0
                    else ("candidate", "baseline")
                )
                for selection in order:
                    start = perf_counter()
                    result = workflow(
                        program.evaluate if selection == "baseline" else evaluate,
                        features,
                    )
                    elapsed = perf_counter() - start
                    error = block_error(result, target, atol=1e-9, rtol=1e-10)
                    record["timings"].append(
                        {
                            "selection": selection,
                            "seconds": elapsed,
                            "workload": "unchanged-geometry",
                            "inputs_hash": input_hash,
                            "repeat": repeat,
                            "diagnostics": {
                                "checksum_error": error,
                                "scope": "fixed features",
                            },
                        }
                    )
            # Changed inputs and small partial tiles exercise plan reuse. SCF,
            # density construction, quadrature and forces remain downstream.
            changed = features.copy()
            changed *= 1.01
            record["block_errors"]["changed_features"] = block_error(
                workflow(evaluate, changed),
                workflow(program.evaluate, changed),
                atol=1e-9,
                rtol=1e-10,
            )
            baseline = [
                r["seconds"] for r in record["timings"] if r["selection"] == "baseline"
            ]
            candidate = [
                r["seconds"] for r in record["timings"] if r["selection"] == "candidate"
            ]
            record["feature_consumer_speedup"] = float(
                np.median(baseline) / np.median(candidate)
            )
            passed = all(v["passed"] for v in record["block_errors"].values()) and all(
                r["diagnostics"]["checksum_error"]["passed"] for r in record["timings"]
            )
            record["stages"]["endpoint"] = outcome(
                "pass" if passed else "fail",
                None if passed else "feature consumer mismatch",
                scope="fixed-feature upload/evaluation/download/consumption only",
            )
            record["performance"] = outcome(
                "not-run",
                "no downstream SCF endpoint; reported ratios are feature-consumer observations only",
            )
        if gpu:
            metrics = gpu.metrics()
            record["metrics"] = metrics
            record["memory"] = {
                "allocated_bytes": gpu.plan.allocation_bytes,
                "peak_bytes": None,
                "reason": "numeric capacity bounded; allocator/runtime high-water mark not independently profiled",
                "device_capacity": gpu.plan.device_bytes,
                "host_capacity": gpu.plan.host_bytes,
                "observed_prepare_device_delta": metrics["prepare_device_delta"],
                "observed_device_delta": metrics["observed_device_delta"],
            }
    finally:
        if gpu:
            gpu.close()
    validate_evidence(record)
    record["capability"] = query_capability(
        program, schedule=schedule, artifact=artifact, evidence=record
    )
    atomic_json(folder / "report.json", record)
    print(record["subject"], record["stages"]["numerical"]["status"], flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        choices=("cpu", "cuda-compile", "gpu-numerical", "endpoint"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/xc161-cuda-cache"))
    parser.add_argument(
        "--nvcc", type=Path, default=Path("/group/software/cuda-12.9.1/bin/nvcc")
    )
    parser.add_argument("--target", default="sm_120")
    parser.add_argument(
        "--functionals", nargs="+", choices=tuple(CATALOG), default=list(CATALOG)
    )
    parser.add_argument(
        "--spins",
        nargs="+",
        choices=("polarized", "unpolarized"),
        default=["polarized", "unpolarized"],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("baseline", "fused", "split"),
        default=["baseline", "fused", "split"],
    )
    parser.add_argument("--order", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--tile-points", type=int, default=256)
    parser.add_argument("--budget-bytes", type=int, default=64 << 20)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.points < 1 or args.repeats < 1:
        parser.error("points and repeats must be positive")
    device = None
    if args.tier in ("gpu-numerical", "endpoint"):
        device = {
            "nvidia_smi": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
            "host": platform.node(),
        }
    records = [
        run(args, name, spin, variant, device)
        for name in args.functionals
        for spin in args.spins
        for variant in args.variants
    ]
    summary = {
        "tier": args.tier,
        "records": [
            {
                "subject": r["subject"],
                "identity": r["identity"],
                "numerical": r["stages"]["numerical"],
                "endpoint": r["stages"]["endpoint"],
                "spill_free": r.get("spill_free"),
                "speedup": r.get("feature_consumer_speedup"),
            }
            for r in records
        ],
    }
    atomic_json(args.output / "summary.json", summary)
    if any(
        r["stages"][s]["status"] == "fail"
        for r in records
        for s in ("numerical", "endpoint")
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
