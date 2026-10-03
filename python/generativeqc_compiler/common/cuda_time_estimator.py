"""Experimental, GPU-free kernel timing from explicit achieved-rate calibration.

This is a separate layer over static candidate screening, not a promotion policy
or an endpoint predictor. Rates and work counts must use the same operation,
precision, and traffic conventions. No calibration or device topology is guessed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import pairwise

from .cuda_cost_model import StaticCudaCost
from .cuda_target import normalize_cuda_architecture


def _finite_float(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric) or numeric < 0.0 or (positive and numeric == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return numeric


def _unit_interval(value: float, name: str, *, positive: bool = False) -> float:
    numeric = _finite_float(value, name, positive=positive)
    if numeric > 1.0:
        raise ValueError(f"{name} must be in {'(0, 1]' if positive else '[0, 1]'}")
    return numeric


def _count(value: int | None, name: str, *, positive: bool = False) -> None:
    if value is not None and (type(value) is not int or value < int(positive)):
        raise ValueError(
            f"{name} must be a {'positive' if positive else 'non-negative'} integer or None"
        )


@dataclass(frozen=True, slots=True)
class CudaTimingCalibration:
    """Caller-supplied measured rates for a concrete device and workload regime.

    ``workload`` identifies the kernel family, precision, operation convention
    (e.g. FMA counts as two), and cache/traffic regime. ``provenance`` identifies
    retained measurements, including software, clocks, and measurement procedure.
    The caller must ensure these match the candidate; names cannot certify this.

    Rates are achieved whole-device rates, not vendor peaks. The v1 model uses
    ``saturation_occupancy`` for both resources. In v2 a nonempty monotone
    ``memory_throughput_curve`` instead maps occupancy to memory-rate fractions;
    ``crossover_penalty_curve`` maps occupancy to bounded resource-contention
    coefficients. Empty curves retain linear memory scaling and full overlap.
    ``compute_wave_correction`` assumes equal work per block on identical SMs.
    ``batch_seconds`` is a fixed cost per nonempty serial batch, independent of
    launch count. These terms require v2 and explicit matching-family selection.

    The occupancy threshold and engineering ``uncertainty_fraction`` are explicit
    inputs, not fitted defaults or statistical confidence guarantees.
    """

    device: str
    architecture: str
    sm_count: int
    workload: str
    provenance: str
    effective_compute_ops_per_second: float
    effective_memory_bytes_per_second: float
    launch_seconds: float
    saturation_occupancy: float
    uncertainty_fraction: float
    model: str = "roofline-linear-occupancy.v1"
    memory_throughput_curve: tuple[tuple[float, float], ...] = ()
    crossover_penalty_curve: tuple[tuple[float, float], ...] = ()
    batch_seconds: float = 0.0
    compute_wave_correction: bool = False

    def __post_init__(self) -> None:
        for name in ("device", "architecture", "workload", "provenance"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if normalize_cuda_architecture(self.architecture) != self.architecture:
            raise ValueError("architecture must use canonical sm_XX notation")
        if self.sm_count is None:
            raise ValueError("sm_count is required for device calibration")
        _count(self.sm_count, "sm_count", positive=True)
        for name in (
            "effective_compute_ops_per_second",
            "effective_memory_bytes_per_second",
            "launch_seconds",
        ):
            object.__setattr__(
                self, name, _finite_float(getattr(self, name), name, positive=True)
            )
        object.__setattr__(
            self,
            "saturation_occupancy",
            _unit_interval(
                self.saturation_occupancy, "saturation_occupancy", positive=True
            ),
        )
        object.__setattr__(
            self,
            "uncertainty_fraction",
            _unit_interval(self.uncertainty_fraction, "uncertainty_fraction"),
        )
        if self.model not in {
            "roofline-linear-occupancy.v1",
            "roofline-calibrated-overlap.v2",
        }:
            raise ValueError("unsupported CUDA timing model")
        if type(self.compute_wave_correction) is not bool:
            raise TypeError("compute_wave_correction must be a boolean")
        object.__setattr__(
            self, "batch_seconds", _finite_float(self.batch_seconds, "batch_seconds")
        )
        curve = tuple(tuple(point) for point in self.memory_throughput_curve)
        if curve:
            if any(len(point) != 2 for point in curve):
                raise ValueError(
                    "memory_throughput_curve requires occupancy/fraction pairs"
                )
            for x, y in curve:
                _unit_interval(x, "curve occupancy")
                _unit_interval(y, "curve throughput")
            if curve[0] != (0.0, 0.0) or curve[-1] != (1.0, 1.0):
                raise ValueError("memory_throughput_curve must span (0, 0) to (1, 1)")
            if any(
                x1 >= x2 or y1 > y2 or y2 == 0 for (x1, y1), (x2, y2) in pairwise(curve)
            ):
                raise ValueError(
                    "memory_throughput_curve must be monotone with positive throughput"
                )
        object.__setattr__(self, "memory_throughput_curve", curve)
        penalties = tuple(tuple(point) for point in self.crossover_penalty_curve)
        if penalties:
            if any(len(point) != 2 for point in penalties):
                raise ValueError(
                    "crossover_penalty_curve requires occupancy/penalty pairs"
                )
            for x, y in penalties:
                _unit_interval(x, "crossover occupancy")
                _unit_interval(y, "crossover penalty")
            if penalties[0][0] != 0.0 or penalties[-1][0] != 1.0:
                raise ValueError("crossover_penalty_curve must span occupancies 0 to 1")
            if any(x1 >= x2 for (x1, _), (x2, _) in pairwise(penalties)):
                raise ValueError("crossover occupancies must be strictly increasing")
        object.__setattr__(self, "crossover_penalty_curve", penalties)
        if self.model.endswith(".v1") and (
            curve or self.batch_seconds or penalties or self.compute_wave_correction
        ):
            raise ValueError("refined calibration terms require the v2 model")

    def to_payload(self) -> dict[str, object]:
        """Serialize calibration identity and all explicit modeling assumptions."""
        payload = asdict(self)
        version = "v2" if self.model.endswith(".v2") else "v1"
        if version == "v1":
            # Preserve the retained v1 profile byte semantics and replay results.
            for name in (
                "model",
                "memory_throughput_curve",
                "crossover_penalty_curve",
                "batch_seconds",
                "compute_wave_correction",
            ):
                payload.pop(name)
        return {
            "schema": f"generativeqc.compiler.cuda-timing-calibration.{version}",
            **payload,
        }


@dataclass(frozen=True, slots=True)
class CudaTimeEstimate:
    """One homogeneous kernel's estimate, with inputs retained for auditing.

    Component times already include the throughput correction. Work/traffic
    counts cover ALL repetitions; launch_count multiplies only launch latency.
    Missing components and totals remain None. The interval is an engineering
    band around the estimate, not a bound on the actual device execution time.
    ``parallel_scale`` scales compute; ``memory_parallel_scale`` scales memory.
    ``overlap_seconds`` is the modeled time saved versus serialized compute and
    memory, so body time is compute + memory - overlap. ``batch_seconds`` is
    charged once, and is zero for a known no-op.
    """

    calibration: CudaTimingCalibration
    cost: StaticCudaCost
    spill_traffic_bytes: int | None
    estimated_seconds: float | None
    lower_seconds: float | None
    upper_seconds: float | None
    compute_seconds: float | None
    memory_seconds: float | None
    launch_seconds: float | None
    parallelism_fraction: float | None
    parallelism_basis: str | None
    parallel_scale: float | None
    bottleneck: str | None
    diagnostics: tuple[str, ...]
    memory_parallel_scale: float | None = None
    overlap_seconds: float | None = None
    batch_seconds: float = 0.0

    @property
    def device(self) -> str:
        """Device identity from the retained calibration."""
        return self.calibration.device

    def to_payload(self) -> dict[str, object]:
        """Serialize a self-contained report, including static evidence caveats."""
        return {
            **asdict(self),
            "schema": "generativeqc.compiler.cuda-time-estimate."
            + self.calibration.model.rsplit(".", 1)[1],
            "model": self.calibration.model,
            "scope": (
                "experimental homogeneous-kernel engineering estimate; "
                "not an endpoint prediction or a statistical confidence interval; "
                "requires real-device endpoint validation"
            ),
            "device": self.device,
            "calibration": self.calibration.to_payload(),
            "cost": self.cost.to_payload(),
        }


def estimate_cuda_time(
    cost: StaticCudaCost,
    calibration: CudaTimingCalibration,
    *,
    spill_traffic_bytes: int | None = None,
    allow_per_sm_fallback: bool = False,
) -> CudaTimeEstimate:
    """Estimate repeated, identical, serial kernel launches without probing CUDA.

    Work and semantic traffic must be totals across all launches, with the same
    grid/resources per launch. Heterogeneous kernels must be estimated separately:
    max(sum(compute), sum(memory)) loses sequential compute/memory bottlenecks.

    The v1 model is max(ops / compute_rate, bytes / memory_rate) / parallel_scale
    + launches * launch_latency, with scale = min(1, occupancy / saturation).
    The v2 model can scale compute and memory independently. For corrected times
    C and M, body time is max(C,M) + k * min(C,M)^2 / max(C,M), with k in [0,1]
    interpolated from the crossover curve; zero body work costs zero. Fixed
    batch overhead is added once. The optional wave correction models uniform
    blocks finishing on the busiest SM; it does not model arbitrary block costs.
    Occupancy remains an optimistic static upper bound, not measured utilization.

    PTXAS spill bytes are static compiler evidence, NOT dynamic grid traffic.
    Nonzero or unknown static spills require explicit total ``spill_traffic_bytes``
    (excluding bytes already counted in semantic traffic); otherwise memory time
    is unknown. Zero compiler spills allow a zero spill-traffic assumption.

    Whole-device parallelism is required by default. ``allow_per_sm_fallback``
    opts into a disclosed optimistic estimate when global underfill is unknown.
    Contradictory device identities/invalid facts raise; missing evidence returns
    an unknown total while preserving independently computable component times.
    """
    if not isinstance(cost, StaticCudaCost):
        raise TypeError("CUDA timing estimate requires StaticCudaCost")
    if not isinstance(calibration, CudaTimingCalibration):
        raise TypeError("CUDA timing estimate requires CudaTimingCalibration")
    if type(allow_per_sm_fallback) is not bool:
        raise TypeError("allow_per_sm_fallback must be a boolean")
    if cost.architecture != calibration.architecture:
        raise ValueError("cost architecture does not match calibration architecture")
    for name in (
        "arithmetic_operation_count",
        "semantic_traffic_bytes",
        "launch_count",
        "spill_bytes",
        "grid_blocks",
        "resident_blocks_per_sm_upper_bound",
    ):
        _count(getattr(cost, name), name)
    _count(cost.sm_count, "sm_count", positive=True)
    if cost.sm_count is not None and cost.sm_count != calibration.sm_count:
        raise ValueError("cost SM count does not match calibration SM count")
    _count(spill_traffic_bytes, "spill_traffic_bytes")
    for name in ("occupancy_upper_bound", "device_occupancy_upper_bound"):
        value = getattr(cost, name)
        if value is not None:
            _unit_interval(value, name)

    diagnostics = list(cost.diagnostics)
    diagnostics.append(
        "occupancy is an optimistic resource bound, not achieved utilization"
    )
    required = {
        "arithmetic operation count": cost.arithmetic_operation_count,
        "semantic traffic bytes": cost.semantic_traffic_bytes,
        "launch count": cost.launch_count,
    }
    for label, value in required.items():
        if value is None:
            diagnostics.append(f"{label} is unavailable")

    # A known no-op needs no occupancy or spill evidence, but must not hide work.
    no_launches = cost.launch_count == 0
    if no_launches and any(
        value is not None and value > 0
        for value in (
            cost.arithmetic_operation_count,
            cost.semantic_traffic_bytes,
            spill_traffic_bytes,
        )
    ):
        raise ValueError("zero launch count contradicts nonzero work or traffic")
    empty = no_launches and all(value == 0 for value in required.values())

    parallelism = cost.device_occupancy_upper_bound
    basis = "whole-device" if parallelism is not None else None
    if cost.grid_blocks == 0 or cost.resident_blocks_per_sm_upper_bound == 0:
        # In particular, a zero grid with unknown SM count cannot fall back to
        # positive per-SM occupancy and manufacture an executable launch.
        parallelism, basis = 0.0, "no-executable-grid"
    elif parallelism is not None and (
        cost.grid_blocks is None or cost.sm_count is None
    ):
        raise ValueError("whole-device occupancy requires grid size and SM count")
    if parallelism is None and allow_per_sm_fallback:
        parallelism = cost.occupancy_upper_bound
        if parallelism is not None:
            basis = "per-sm-fallback"
            diagnostics.append(
                "using per-SM occupancy without a global underfill correction; "
                "explicit optimistic fallback"
            )
    if not empty:
        if parallelism is None:
            diagnostics.append("whole-device parallelism is unavailable")
        elif parallelism <= 0.0:
            diagnostics.append("known launch shape exposes no executable parallelism")

    parallel_scale = None
    memory_scale = None
    if parallelism is not None and parallelism > 0.0:
        parallel_scale = min(1.0, parallelism / calibration.saturation_occupancy)
        memory_scale = parallel_scale
        if (
            calibration.compute_wave_correction
            and cost.grid_blocks is not None
            and cost.sm_count is not None
        ):
            # Uniform compute-bound blocks share SM-local execution units. A
            # fractional final SM wave completes at the busiest SM, not at the
            # device-average work count. Integer arithmetic keeps huge grids safe.
            resident = cost.resident_blocks_per_sm_upper_bound
            if resident > 0 and cost.occupancy_upper_bound is not None:
                busy_blocks = (cost.grid_blocks + cost.sm_count - 1) // cost.sm_count
                busy_occupancy = (
                    min(busy_blocks, resident) * cost.occupancy_upper_bound / resident
                )
                balance = cost.grid_blocks / (cost.sm_count * busy_blocks)
                parallel_scale = balance * min(
                    1.0, busy_occupancy / calibration.saturation_occupancy
                )
                diagnostics.append(
                    "compute throughput includes uniform-block SM wave imbalance"
                )
        if calibration.memory_throughput_curve:
            for (x0, y0), (x1, y1) in pairwise(calibration.memory_throughput_curve):
                if parallelism <= x1:
                    memory_scale = y0 + (y1 - y0) * ((parallelism - x0) / (x1 - x0))
                    break
            diagnostics.append(
                "memory throughput uses a measured monotone occupancy curve"
            )
        if parallel_scale < 1.0:
            diagnostics.append(
                "effective throughput is linearly reduced below the calibrated saturation occupancy"
            )

    effective_spills = spill_traffic_bytes
    if empty or (effective_spills is None and cost.spill_bytes == 0):
        effective_spills = 0
    if effective_spills is None:
        diagnostics.append(
            "dynamic spill traffic is unavailable; PTXAS spill bytes cannot be "
            "used as total runtime traffic"
        )
    elif spill_traffic_bytes is not None:
        diagnostics.append("caller-supplied total dynamic spill traffic is included")

    def finite_seconds(
        work: int | None, rate: float, scale: float, label: str
    ) -> float | None:
        """Keep out-of-range arithmetic out of the report's strict JSON payload."""
        if work is None:
            return None
        try:
            seconds = work / rate / scale
        except OverflowError:
            seconds = math.inf
        if not math.isfinite(seconds):
            diagnostics.append(f"{label} exceeds the finite timing range")
            return None
        return seconds

    compute_seconds = memory_seconds = None
    if empty:
        compute_seconds = memory_seconds = 0.0
    elif parallel_scale is not None and parallel_scale > 0:
        compute_seconds = finite_seconds(
            cost.arithmetic_operation_count,
            calibration.effective_compute_ops_per_second,
            parallel_scale,
            "compute time",
        )
        traffic = (
            None
            if cost.semantic_traffic_bytes is None or effective_spills is None
            else cost.semantic_traffic_bytes + effective_spills
        )
        if memory_scale is not None and memory_scale > 0:
            memory_seconds = finite_seconds(
                traffic,
                calibration.effective_memory_bytes_per_second,
                memory_scale,
                "memory time",
            )
    # Multiplication avoids overflowing the reciprocal for tiny launch latencies.
    launch_seconds = None
    if cost.launch_count is not None:
        try:
            launch_seconds = cost.launch_count * calibration.launch_seconds
        except OverflowError:
            launch_seconds = math.inf
        if not math.isfinite(launch_seconds):
            diagnostics.append("launch time exceeds the finite timing range")
            launch_seconds = None

    estimated_seconds = lower_seconds = upper_seconds = None
    overlap_seconds = None
    batch_seconds = 0.0 if empty else calibration.batch_seconds
    bottleneck = None
    if (
        compute_seconds is not None
        and memory_seconds is not None
        and launch_seconds is not None
    ):
        # The bounded crossover correction is largest near C == M and decays
        # quadratically when one resource dominates. It stays between a roofline
        # maximum and serialized C+M, without overflowing C+M before subtraction.
        major, minor = (
            max(compute_seconds, memory_seconds),
            min(compute_seconds, memory_seconds),
        )
        penalty = 0.0
        if calibration.crossover_penalty_curve and parallelism is not None:
            for (x0, y0), (x1, y1) in pairwise(calibration.crossover_penalty_curve):
                if parallelism <= x1:
                    penalty = y0 + (y1 - y0) * ((parallelism - x0) / (x1 - x0))
                    break
        correction = penalty * (minor / major) * minor if major else 0.0
        overlap_seconds = minor - correction
        body_seconds = major + correction
        total = body_seconds + launch_seconds + batch_seconds
        upper = total * (1.0 + calibration.uncertainty_fraction)
        if math.isfinite(total) and math.isfinite(upper):
            estimated_seconds = total
            lower_seconds = total * (1.0 - calibration.uncertainty_fraction)
            upper_seconds = upper
            if empty:
                bottleneck = "none"
            elif launch_seconds + batch_seconds > body_seconds:
                bottleneck = "launch"
            elif math.isclose(
                compute_seconds, memory_seconds, rel_tol=1.0e-9, abs_tol=0.0
            ):
                bottleneck = "balanced"
            else:
                bottleneck = "compute" if compute_seconds > memory_seconds else "memory"
        else:
            diagnostics.append(
                "total time or uncertainty band exceeds the finite timing range"
            )

    return CudaTimeEstimate(
        calibration=calibration,
        cost=cost,
        spill_traffic_bytes=effective_spills,
        estimated_seconds=estimated_seconds,
        lower_seconds=lower_seconds,
        upper_seconds=upper_seconds,
        compute_seconds=compute_seconds,
        memory_seconds=memory_seconds,
        launch_seconds=launch_seconds,
        parallelism_fraction=parallelism,
        parallelism_basis=basis,
        parallel_scale=parallel_scale,
        bottleneck=bottleneck,
        diagnostics=tuple(diagnostics),
        memory_parallel_scale=memory_scale,
        overlap_seconds=overlap_seconds,
        batch_seconds=batch_seconds,
    )
