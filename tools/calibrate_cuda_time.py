"""Collect real-device CUDA timing calibration or refit retained CPU-readable data.

Run collection through srun on this machine. This tool owns measurement/build
work; the compiler estimator remains GPU-free and never invokes this runner.
Training cases alone select the rates, common saturation threshold, and error
band. Held-out cases are scored only after those parameters have been frozen.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_ROOT))

from generativeqc_compiler.common.cuda_cost_model import (
    StaticCudaCost,
    static_cuda_cost,
)
from generativeqc_compiler.common.cuda_resources import parse_resources
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.cuda_time_estimator import (
    CudaTimingCalibration,
    estimate_cuda_time,
)
from generativeqc_compiler.common.gpu_profitability import GpuProfitability

# Freeze accuracy gates before collecting data. These qualify only this probe
# workload, not chemistry endpoints or other instructions/cache/launch regimes.
GATES = {
    "median_relative_error": 0.20,
    "p95_relative_error": 0.35,
    "max_relative_error": 0.50,
}


def _json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=600, **kwargs
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"command failed: {command!r}\n{exc.stderr}") from exc


def collect(
    nvcc: Path,
    architecture: str,
    samples: int,
    destination: Path,
    *,
    refined: bool = False,
    training_only: bool = False,
) -> dict[str, Any]:
    """Compile with ccache and measure only within the caller's Slurm allocation.

    CUDA_VISIBLE_DEVICES is inherited unchanged by every child. Native ordinal 0
    selects the allocated visible device. Build logs/binaries stay in build/;
    the compact measurement retains source, compiler, binary and PTXAS identity.
    """
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("collection requires a Slurm GPU allocation; run through srun")
    ccache = shutil.which("ccache")
    if ccache is None:
        raise ValueError("ccache is required for CUDA calibration builds")
    target = cuda_target_info(architecture)
    build = _ROOT / "build/cuda-timing-calibration"
    build.mkdir(parents=True, exist_ok=True)
    source = _ROOT / "tools/cuda_timing_probe.cu"
    obj, binary = build / "probe.o", build / "probe"
    cache_version = _run([ccache, "--version"]).stdout
    cache_before = _run([ccache, "--show-stats"]).stdout
    compiler_version = _run([str(nvcc), "--version"]).stdout
    flags = [
        "-O3",
        "-std=c++17",
        f"-arch={target.architecture}",
        "-lineinfo",
        "-Xptxas=-v",
    ]
    command = [
        ccache,
        str(nvcc),
        *flags,
        "-c",
        str(source.relative_to(_ROOT)),
        "-o",
        str(obj.relative_to(_ROOT)),
    ]
    compiled = _run(
        command, cwd=_ROOT, env={**os.environ, "CCACHE_BASEDIR": str(_ROOT)}
    )
    (build / "compile.log").write_text(compiled.stderr, encoding="utf-8")
    resources = parse_resources(compiled.stderr)
    if len(resources) != 4 or any(
        row.spill_load_bytes or row.spill_store_bytes for row in resources
    ):
        raise ValueError(
            "calibration requires four complete, spill-free PTXAS kernel records"
        )
    _run(
        [str(nvcc), f"-arch={target.architecture}", str(obj), "-o", str(binary)],
        cwd=_ROOT,
    )
    query = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,pstate,clocks.sm,clocks.mem,power.limit,temperature.gpu",
        "--format=csv",
    ]
    before = _run(query).stdout
    probe_args = [str(binary), str(samples)]
    if refined:
        probe_args.append("--refined")
        if training_only:
            probe_args.append("--training-only")
    result = _run(probe_args, cwd=_ROOT)
    (build / "progress.txt").write_text(result.stderr, encoding="utf-8")
    measurement = json.loads(result.stdout)
    if measurement["architecture"] != target.architecture:
        raise ValueError("allocated GPU architecture differs from the compiled target")
    measurement["provenance"] = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "source_revision": _run(["git", "rev-parse", "HEAD"], cwd=_ROOT).stdout.strip(),
        "source_files": {
            str(path.relative_to(_ROOT)): _hash(path)
            for path in (
                source,
                Path(__file__).resolve(),
                _ROOT / "tools/cuda_timing_refined.py",
                _ROOT / "python/generativeqc_compiler/common/cuda_time_estimator.py",
            )
        },
        "binary_sha256": _hash(binary),
        "compiler_version": compiler_version,
        "compile_command": command,
        "ptxas_resources": [asdict(row) for row in resources],
        "ccache_version": cache_version,
        "ccache_before": cache_before,
        "ccache_after": _run([ccache, "--show-stats"]).stdout,
        "device_before": before,
        "device_after": _run(query).stdout,
        "timing_scope": "warm serial same-stream kernel batch: host submission through final event synchronization; excludes allocation, transfers, warmup and numerical validation",
        "count_convention": "FP64 FMA=2 ops; fma seed/initialization/reduction=16 ops/thread; semantic global read/write bytes, all repetitions",
        "sample_policy": "fixed seed 1787 shuffles cases; three warmups/case; adaptive batch targets 15 ms with 2..256 launches, empty launch batches fixed; no sample removed",
    }
    _json(destination / "measurement.json", measurement)
    return measurement


def cost_for_case(row: dict[str, Any], measurement: dict[str, Any]) -> StaticCudaCost:
    """Reconstruct total semantic work with measured kernel resource evidence."""
    launches = row["launch_count"]
    return static_cuda_cost(
        GpuProfitability(
            semantic_traffic_bytes=row["bytes_per_launch"] * launches,
            arithmetic_operation_count=row["operations_per_launch"] * launches,
            launch_count=launches,
            compiled_registers_per_thread=row["registers_per_thread"],
            shared_bytes=row["shared_bytes"],
            spill_store_bytes=0,
            spill_load_bytes=0,
        ),
        cuda_target_info(measurement["architecture"]),
        row["block_threads"],
        grid_blocks=row["grid_blocks"],
        sm_count=measurement["sm_count"],
    )


def _quantile(values: list[float], q: float) -> float:
    """Linear interpolation, including singleton groups, for reproducible reports."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def _errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row["relative_error"] for row in rows]
    result = {
        "count": len(rows),
        "median_relative_error": statistics.median(errors),
        "p95_relative_error": _quantile(errors, 0.95),
        "max_relative_error": max(errors),
        "engineering_band_coverage": sum(row["in_engineering_band"] for row in rows)
        / len(rows),
    }
    result["passes_accuracy_gates"] = all(
        result[name] <= limit for name, limit in GATES.items()
    )
    return result


def fit(
    measurement: dict[str, Any], measurement_sha256: str
) -> tuple[CudaTimingCalibration, dict[str, Any]]:
    """Fit the existing linear-occupancy model using training rows only.

    Empty launches estimate incremental submission/execution latency from median
    batch wall time per launch. For each saturation threshold on a fixed 0.001
    grid, geometric-mean achieved rates minimize log error within compute/memory
    families. Equal-weight family loss selects a single common threshold.
    """
    if measurement.get("schema") != "generativeqc.cuda-timing-probe.v1":
        raise ValueError("unsupported timing measurement schema")
    cases = measurement["cases"]
    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("duplicate case identity")
    for row in cases:
        if row["split"] not in {"train", "holdout"}:
            raise ValueError("every measurement needs an explicit train/holdout split")
        if (
            not math.isfinite(row["max_absolute_error"])
            or not 0 <= row["max_absolute_error"] <= 2e-12
            or row["local_bytes"] != 0
        ):
            raise ValueError("probe numerical/resource acceptance failed")
        if type(row["launch_count"]) is not int or row["launch_count"] < 1:
            raise ValueError("measurement launch_count must be a positive integer")
        if not row["wall_seconds"] or any(
            not math.isfinite(x) or x <= 0 for x in row["wall_seconds"]
        ):
            raise ValueError("measurement times must be finite and positive")
    train = [row for row in cases if row["split"] == "train"]
    launch_samples = [
        statistics.median(row["wall_seconds"]) / row["launch_count"]
        for row in train
        if row["family"] == "launch"
    ]
    if not launch_samples:
        raise ValueError("missing launch calibration samples")
    launch_seconds = statistics.median(launch_samples)
    families = {
        family: [row for row in train if row["family"] == family]
        for family in ("fma", "copy")
    }
    if not all(families.values()):
        raise ValueError("compute and memory training samples are both required")
    inputs = {}
    for family, rows in families.items():
        field = "operations_per_launch" if family == "fma" else "bytes_per_launch"
        inputs[family] = []
        for row in rows:
            occupancy = cost_for_case(row, measurement).device_occupancy_upper_bound
            body = (
                statistics.median(row["wall_seconds"]) / row["launch_count"]
                - launch_seconds
            )
            if occupancy is None or occupancy <= 0 or body <= 0:
                raise ValueError(
                    "training body must exceed launch latency with known positive occupancy"
                )
            inputs[family].append((occupancy, row[field] / body))
        if len({occupancy for occupancy, _ in inputs[family]}) < 2:
            raise ValueError(
                "calibration requires multiple distinct occupancies per family"
            )
    best = None
    for tick in range(1, 1001):
        threshold = tick / 1000
        loss = 0.0
        rates = {}
        for family, pairs in inputs.items():
            logs = [
                math.log(rate / min(1.0, occupancy / threshold))
                for occupancy, rate in pairs
            ]
            mean = statistics.mean(logs)
            rates[family] = math.exp(mean)
            loss += statistics.mean((value - mean) ** 2 for value in logs)
        candidate = (loss, threshold, rates["fma"], rates["copy"])
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    loss, threshold, compute_rate, memory_rate = best
    options = {
        "device": measurement["device"],
        "architecture": measurement["architecture"],
        "sm_count": measurement["sm_count"],
        "workload": "FP64 independent FMA chains and streaming copy; FMA=2 ops; streaming arrays >=4x L2 each; warm serial same-stream CUDA batches",
        "provenance": f"measurement.json sha256:{measurement_sha256}; tools/calibrate_cuda_time.py; CUDA runtime {measurement['runtime_version']}",
        "effective_compute_ops_per_second": compute_rate,
        "effective_memory_bytes_per_second": memory_rate,
        "launch_seconds": launch_seconds,
        "saturation_occupancy": threshold,
    }
    provisional = CudaTimingCalibration(**options, uncertainty_fraction=0.0)
    deviations = []
    for row in train:
        estimate = estimate_cuda_time(cost_for_case(row, measurement), provisional)
        if estimate.estimated_seconds is None or estimate.estimated_seconds <= 0:
            raise ValueError("training prediction unexpectedly unknown or zero")
        deviations.append(
            abs(statistics.median(row["wall_seconds"]) / estimate.estimated_seconds - 1)
        )
    # A rounded training envelope is disclosed engineering tolerance, never a
    # fitted held-out confidence claim. Do not silently clip a failed envelope.
    uncertainty = max(0.10, math.ceil(max(deviations) / 0.05) / 20)
    if uncertainty > 1.0:
        raise ValueError("training residual exceeds representable engineering band")
    calibration = CudaTimingCalibration(**options, uncertainty_fraction=uncertainty)
    scored = []
    for row in cases:
        prediction = estimate_cuda_time(cost_for_case(row, measurement), calibration)
        measured = statistics.median(row["wall_seconds"])
        if prediction.estimated_seconds is None:
            raise ValueError("qualification prediction unexpectedly unknown")
        scored.append(
            {
                "id": row["id"],
                "split": row["split"],
                "family": row["family"],
                "measured_seconds": measured,
                "predicted_seconds": prediction.estimated_seconds,
                "relative_error": abs(prediction.estimated_seconds / measured - 1),
                "signed_relative_error": prediction.estimated_seconds / measured - 1,
                "in_engineering_band": prediction.lower_seconds
                <= measured
                <= prediction.upper_seconds,
                "sample_relative_range": (
                    max(row["wall_seconds"]) - min(row["wall_seconds"])
                )
                / measured,
                "operations": prediction.cost.arithmetic_operation_count,
                "traffic_bytes": prediction.cost.semantic_traffic_bytes,
                "launch_count": prediction.cost.launch_count,
                "grid_blocks": row["grid_blocks"],
                "parallelism_fraction": prediction.parallelism_fraction,
            }
        )
    heldout = [row for row in scored if row["split"] == "holdout"]
    if not heldout:
        raise ValueError("qualification requires held-out measurements")
    family_errors = {
        family: _errors([row for row in heldout if row["family"] == family])
        for family in sorted({row["family"] for row in heldout})
    }
    summary = _errors(heldout)
    return calibration, {
        "schema": "generativeqc.cuda-timing-qualification.v1",
        "measurement_sha256": measurement_sha256,
        "fit": {
            "objective": "sum of within-family mean squared log body-time error",
            "loss": loss,
            "threshold_grid_step": 0.001,
            "uncertainty_policy": "max training relative-to-prediction residual rounded up to 0.05; minimum 0.10",
        },
        "gates": GATES,
        "training": _errors([row for row in scored if row["split"] == "train"]),
        "holdout": summary,
        "holdout_by_family": family_errors,
        "qualified": summary["passes_accuracy_gates"]
        and all(group["passes_accuracy_gates"] for group in family_errors.values()),
        "scope": "probe workload only; no chemistry endpoint speedup or production promotion claim",
        "cases": scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--measurement", type=Path, help="refit retained measurements without CUDA"
    )
    parser.add_argument("--nvcc", type=Path)
    parser.add_argument("--arch", help="explicit CUDA architecture for collection")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--suite", choices=("legacy", "refined"), default="legacy")
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="collect refined training before inspecting fresh holdouts",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.measurement:
        raw = args.measurement.read_bytes()
        if args.measurement.suffix == ".gz":
            raw = gzip.decompress(raw)
        measurement = json.loads(raw)
        source = args.output / "measurement.json"
        if source.resolve() != args.measurement.resolve():
            source.write_bytes(raw)
    else:
        if args.nvcc is None or args.arch is None:
            parser.error("collection requires --nvcc and --arch")
        if args.training_only and args.suite != "refined":
            parser.error("--training-only requires --suite refined")
        measurement = collect(
            args.nvcc.resolve(),
            args.arch,
            args.samples,
            args.output,
            refined=args.suite == "refined",
            training_only=args.training_only,
        )
        source = args.output / "measurement.json"
    if measurement.get("schema") == "generativeqc.cuda-timing-probe.v2":
        from tools.cuda_timing_refined import fit_refined, score_refined

        profiles = fit_refined(measurement, _hash(source))
        qualification = score_refined(measurement, profiles, _hash(source))
        calibration_payload = {
            name: profile.to_payload() for name, profile in profiles.items()
        }
        for name, payload in calibration_payload.items():
            _json(args.output / f"calibration-{name}.json", payload)
        if qualification["holdout"] is not None:
            # The matched v1 comparator uses its original training recipe:
            # pure compute/copy and long launch batches. Both models predict
            # every same fresh holdout; no case is removed from the comparison.
            baseline_input = {
                **measurement,
                "schema": "generativeqc.cuda-timing-probe.v1",
                "cases": [
                    row
                    for row in measurement["cases"]
                    if row["split"] == "holdout"
                    or (
                        row["family"] != "mixed"
                        and (row["family"] != "launch" or row["launch_count"] >= 256)
                    )
                ],
            }
            baseline, comparison = fit(baseline_input, _hash(source))
            qualification["baseline"] = comparison
            _json(args.output / "calibration-baseline.json", baseline.to_payload())
    else:
        calibration, qualification = fit(measurement, _hash(source))
        calibration_payload = calibration.to_payload()
        _json(args.output / "calibration.json", calibration_payload)
    _json(args.output / "qualification.json", qualification)
    print(
        json.dumps(
            {
                "calibration": calibration_payload,
                "holdout": qualification["holdout"],
                "holdout_by_family": qualification["holdout_by_family"],
                "qualified": qualification["qualified"],
            },
            indent=2,
        )
    )
    if qualification["holdout"] is not None and not qualification["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
