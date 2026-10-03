"""Constrained family calibration for the refined CUDA timing model.

All parameters come from training cases. Throughput curves are monotone;
resource-crossover penalties stay in [0, 1]. The unseen validation grids and
intensities are scored only after the family profiles have been frozen.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import replace
from typing import Any

from generativeqc_compiler.common.cuda_time_estimator import (
    CudaTimingCalibration,
    estimate_cuda_time,
)

REFINED_GATES = {
    "median_relative_error": 0.05,
    "p95_relative_error": 0.12,
    "max_relative_error": 0.20,
}
FAMILY_PROFILE = {
    "fma": "arithmetic",
    "copy": "copy",
    "mixed": "streaming",
    "launch": "copy",
}


def _isotonic(values: list[float]) -> list[float]:
    """Equal-weight pool-adjacent-violators projection, preserving every sample."""
    pools: list[tuple[float, int]] = []
    for value in values:
        pools.append((value, 1))
        while (
            len(pools) > 1 and pools[-2][0] / pools[-2][1] > pools[-1][0] / pools[-1][1]
        ):
            right, nright = pools.pop()
            left, nleft = pools.pop()
            pools.append((left + right, nleft + nright))
    return [total / count for total, count in pools for _ in range(count)]


def _geomean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def _launch_fit(rows: list[dict[str, Any]]) -> tuple[float, float]:
    """Robust positive slope plus nonnegative fixed batch cost (Theil-Sen)."""
    points = [
        (row["launch_count"], statistics.median(row["wall_seconds"])) for row in rows
    ]
    slopes = [
        (y2 - y1) / (x2 - x1)
        for i, (x1, y1) in enumerate(points)
        for x2, y2 in points[i + 1 :]
        if x1 != x2
    ]
    if not slopes:
        raise ValueError("launch calibration needs multiple batch sizes")
    slope = statistics.median(slopes)
    if slope <= 0:
        raise ValueError("launch calibration requires a positive incremental cost")
    return slope, max(0.0, statistics.median(y - slope * x for x, y in points))


def _stream_fit(
    groups: dict[float, list[tuple[float, float, float]]],
    memory_rate: float,
    copy_fractions: list[float],
) -> tuple[float, list[float], list[float]]:
    """Fit a compute rate and bounded memory/crossover response at measured knots.

    Each group contains normalized ops, bytes and body time for several training
    intensities. A fixed rate grid fits the global compute limit; each occupancy
    has a narrow memory-rate search and an analytic constrained least-squares
    crossover coefficient. Isotonic projection enforces monotone memory rates.
    """
    # Highest intensity at each occupancy anchors a ±10% compute-rate search.
    anchors = [max(points, key=lambda p: p[0] / p[1]) for points in groups.values()]
    initial = _geomean([ops / body for ops, _, body in anchors])
    best = None
    for tick in range(101):
        compute_rate = initial * (0.9 + tick * 0.002)
        fractions = []
        for reference, points in zip(copy_fractions, groups.values(), strict=True):
            candidate = None
            for step in range(71):
                fraction = min(1.0, reference * (0.65 + step * 0.01))
                loss, penalty = _crossover_loss(
                    points, compute_rate, memory_rate * fraction
                )
                trial = (loss, abs(fraction - reference), fraction, penalty)
                if candidate is None or trial < candidate:
                    candidate = trial
            assert candidate is not None
            fractions.append(candidate[2])
        fractions = _isotonic(fractions)
        scores = [
            _crossover_loss(points, compute_rate, memory_rate * fraction)
            for points, fraction in zip(groups.values(), fractions, strict=True)
        ]
        trial = (
            sum(loss for loss, _ in scores),
            compute_rate,
            fractions,
            [penalty for _, penalty in scores],
        )
        if best is None or trial[0] < best[0]:
            best = trial
    assert best is not None
    return best[1], best[2], best[3]


def _crossover_loss(
    points: list[tuple[float, float, float]], compute_rate: float, memory_rate: float
) -> tuple[float, float]:
    """Weighted relative-error least squares for max(C,M)+k*min(C,M)^2/max(C,M)."""
    terms = []
    for ops, traffic, measured in points:
        compute, memory = ops / compute_rate, traffic / memory_rate
        major, minor = max(compute, memory), min(compute, memory)
        terms.append((major / measured, minor * (minor / major) / measured))
    denominator = sum(cross * cross for _, cross in terms)
    penalty = (
        max(
            0.0,
            min(1.0, sum(cross * (1 - base) for base, cross in terms) / denominator),
        )
        if denominator
        else 0.0
    )
    return sum((base + penalty * cross - 1) ** 2 for base, cross in terms), penalty


def fit_refined(
    measurement: dict[str, Any], digest: str
) -> dict[str, CudaTimingCalibration]:
    """Fit explicit arithmetic/copy/streaming profiles; never inspect holdout times."""
    from tools.calibrate_cuda_time import cost_for_case

    if measurement.get("schema") != "generativeqc.cuda-timing-probe.v2":
        raise ValueError("refined fitting requires probe.v2 measurements")
    train = [row for row in measurement["cases"] if row["split"] == "train"]
    if len({row["id"] for row in train}) != len(train):
        raise ValueError("duplicate training case")
    for row in train:
        if not row["wall_seconds"] or any(
            not math.isfinite(x) or x <= 0 for x in row["wall_seconds"]
        ):
            raise ValueError("training times must be finite and positive")
        if (
            not math.isfinite(row["max_absolute_error"])
            or not 0 <= row["max_absolute_error"] <= 2e-12
            or row["local_bytes"] != 0
        ):
            raise ValueError("training numerical/resource acceptance failed")
        if type(row["launch_count"]) is not int or row["launch_count"] <= 0:
            raise ValueError("training launch counts must be positive integers")
    families = {
        family: [row for row in train if row["family"] == family]
        for family in FAMILY_PROFILE
    }
    if not all(families.values()):
        raise ValueError(
            "refined fitting needs launch, copy, arithmetic and mixed training"
        )
    launch, batch = _launch_fit(families["launch"])

    def body(row: dict[str, Any]) -> float:
        result = (statistics.median(row["wall_seconds"]) - batch) / row[
            "launch_count"
        ] - launch
        if result <= 0:
            raise ValueError("training body must exceed launch and batch overhead")
        return result

    def balance(row: dict[str, Any]) -> float:
        sm = measurement["sm_count"]
        return row["grid_blocks"] / (sm * ((row["grid_blocks"] + sm - 1) // sm))

    copy_groups: dict[float, list[float]] = defaultdict(list)
    for row in families["copy"]:
        occupancy = cost_for_case(row, measurement).device_occupancy_upper_bound
        assert occupancy is not None
        copy_groups[occupancy].append(row["bytes_per_launch"] / body(row))
    occupancies = sorted(copy_groups)
    if len(occupancies) < 3 or occupancies[-1] != 1.0:
        raise ValueError(
            "refined memory calibration needs underfill through full occupancy"
        )
    rates = _isotonic([_geomean(copy_groups[occupancy]) for occupancy in occupancies])
    memory_rate = rates[-1]
    fractions = [rate / memory_rate for rate in rates]
    copy_curve = ((0.0, 0.0), *zip(occupancies, fractions, strict=True))
    compute_rate = _geomean(
        [
            row["operations_per_launch"] / body(row) / balance(row)
            for row in families["fma"]
        ]
    )
    block_sizes = {row["block_threads"] for row in train if row["family"] != "launch"}
    if len(block_sizes) != 1:
        raise ValueError("a refined family profile requires one calibrated block size")
    block_size = next(iter(block_sizes))
    common = {
        "device": measurement["device"],
        "architecture": measurement["architecture"],
        "sm_count": measurement["sm_count"],
        "provenance": f"measurement.json sha256:{digest}; tools/calibrate_cuda_time.py; training-only refined fit",
        "effective_compute_ops_per_second": compute_rate,
        "effective_memory_bytes_per_second": memory_rate,
        "launch_seconds": launch,
        "batch_seconds": batch,
        "saturation_occupancy": block_size / measurement["max_threads_per_sm"],
        "uncertainty_fraction": 0.05,
        "model": "roofline-calibrated-overlap.v2",
        "memory_throughput_curve": copy_curve,
        "compute_wave_correction": True,
    }
    workload = f"FP64 FMA=2 ops; {block_size}-thread uniform blocks; warm serial same-stream batches"
    profiles = {
        "arithmetic": CudaTimingCalibration(
            **common, workload=workload + "; independent register FMA chains"
        ),
        "copy": CudaTimingCalibration(
            **common, workload=workload + "; streaming copy; arrays >=4x L2 each"
        ),
    }
    groups: dict[float, list[tuple[float, float, float]]] = {
        occupancy: [] for occupancy in occupancies
    }
    for row in families["mixed"]:
        occupancy = cost_for_case(row, measurement).device_occupancy_upper_bound
        if occupancy not in groups:
            raise ValueError("mixed and copy training need matching occupancy knots")
        groups[occupancy].append(
            (
                row["operations_per_launch"] / balance(row),
                row["bytes_per_launch"],
                body(row),
            )
        )
    if any(
        len({ops / traffic for ops, traffic, _ in points}) < 3
        for points in groups.values()
    ):
        raise ValueError(
            "each mixed occupancy needs at least three training intensities"
        )
    stream_compute, stream_fractions, penalties = _stream_fit(
        groups, memory_rate, fractions
    )
    normalized = [fraction / stream_fractions[-1] for fraction in stream_fractions]
    stream = {
        **common,
        "effective_compute_ops_per_second": stream_compute,
        "effective_memory_bytes_per_second": memory_rate * stream_fractions[-1],
        "memory_throughput_curve": (
            (0.0, 0.0),
            *zip(occupancies, normalized, strict=True),
        ),
        "crossover_penalty_curve": (
            (0.0, penalties[0]),
            *zip(occupancies, penalties, strict=True),
        ),
    }
    profiles["streaming"] = CudaTimingCalibration(
        **stream,
        workload=workload
        + "; dependent FMA grid-stride transform; arrays >=4x L2 each",
    )
    # Training-derived engineering bands remain separate from held-out coverage.
    for name, calibration in profiles.items():
        rows = [row for row in train if FAMILY_PROFILE[row["family"]] == name]
        residuals = []
        for row in rows:
            prediction = estimate_cuda_time(
                cost_for_case(row, measurement), calibration
            ).estimated_seconds
            assert prediction is not None and prediction > 0
            residuals.append(
                abs(statistics.median(row["wall_seconds"]) / prediction - 1)
            )
        band = max(0.05, math.ceil(max(residuals) * 100) / 100)
        if band > 1:
            raise ValueError(
                "refined training residual exceeds representable error band"
            )
        profiles[name] = replace(calibration, uncertainty_fraction=band)
    return profiles


def score_refined(
    measurement: dict[str, Any], profiles: dict[str, CudaTimingCalibration], digest: str
) -> dict[str, Any]:
    """Score frozen profiles, rejecting invalid evidence and retaining gate failures.

    Qualification requires fresh holdouts for every calibrated probe family.
    Partial or training-only collections may be scored but cannot qualify.
    """
    from tools.calibrate_cuda_time import _errors, cost_for_case

    if measurement.get("schema") != "generativeqc.cuda-timing-probe.v2":
        raise ValueError("refined scoring requires probe.v2 measurements")
    rows = measurement["cases"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate qualification case")
    scored = []
    for row in measurement["cases"]:
        if row["split"] not in {"train", "holdout"}:
            raise ValueError("scoring requires explicit train/holdout labels")
        if (
            not math.isfinite(row["max_absolute_error"])
            or not 0 <= row["max_absolute_error"] <= 2e-12
            or row["local_bytes"] != 0
        ):
            raise ValueError("qualification numerical/resource acceptance failed")
        if not row["wall_seconds"] or any(
            not math.isfinite(x) or x <= 0 for x in row["wall_seconds"]
        ):
            raise ValueError("qualification requires positive finite times")
        if type(row["launch_count"]) is not int or row["launch_count"] <= 0:
            raise ValueError("qualification launch counts must be positive integers")
        if row["family"] not in FAMILY_PROFILE:
            raise ValueError("qualification requires a known probe family")
        profile = FAMILY_PROFILE[row["family"]]
        prediction = estimate_cuda_time(
            cost_for_case(row, measurement), profiles[profile]
        )
        measured = statistics.median(row["wall_seconds"])
        assert prediction.estimated_seconds is not None
        scored.append(
            {
                "id": row["id"],
                "family": row["family"],
                "profile": profile,
                "split": row["split"],
                "measured_seconds": measured,
                "predicted_seconds": prediction.estimated_seconds,
                "relative_error": abs(prediction.estimated_seconds / measured - 1),
                "in_engineering_band": prediction.lower_seconds
                <= measured
                <= prediction.upper_seconds,
                "operations": prediction.cost.arithmetic_operation_count,
                "traffic_bytes": prediction.cost.semantic_traffic_bytes,
                "launch_count": row["launch_count"],
                "grid_blocks": row["grid_blocks"],
            }
        )

    def errors(rows: list[dict[str, Any]]) -> dict[str, Any]:
        result = _errors(rows)
        result["passes_accuracy_gates"] = all(
            result[key] <= limit for key, limit in REFINED_GATES.items()
        )
        return result

    heldout = [row for row in scored if row["split"] == "holdout"]
    groups = {
        family: errors([row for row in heldout if row["family"] == family])
        for family in sorted({row["family"] for row in heldout})
    }
    summary = errors(heldout) if heldout else None
    return {
        "schema": "generativeqc.cuda-timing-qualification.v2",
        "measurement_sha256": digest,
        "gates": REFINED_GATES,
        "training": errors([row for row in scored if row["split"] == "train"]),
        "holdout": summary,
        "holdout_by_family": groups,
        "cases": scored,
        "qualified": bool(
            summary
            and summary["passes_accuracy_gates"]
            and set(groups) == set(FAMILY_PROFILE)
            and all(group["passes_accuracy_gates"] for group in groups.values())
        ),
        "scope": "explicit family profiles for the measured FP64 warm-batch domain; not chemistry endpoint prediction",
    }
