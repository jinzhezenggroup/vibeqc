"""Auditable GPU profitability facts shared by compiler schedule tuners.

This module does not promote a candidate by itself. It normalizes compiler-visible
static and measured resource facts, then provides deterministic ordering keys for
finite search budgets and for candidates already inside an endpoint-noise band.
Missing evidence is kept explicit and ranks behind comparable measured evidence.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

ENDPOINT_NOISE_FRACTION = 0.01
PATHOLOGICAL_REDUCTION_MIN_WARPS = 4


def scalar_reduction_promotion_rejection(
    *,
    output_elements: int,
    reduction_elements: int,
    parallel_width: int,
    alternative: str | None,
) -> str | None:
    """Reject promotion of a serial reduction when a legal parallel lowering exists.

    This is a performance-promotion diagnostic, not scientific legality. Small
    reductions stay eligible because launch/library overhead can dominate them.
    A plan is called pathological only when fewer than one hardware subgroup of
    independent outputs each serializes at least four subgroup-widths of work.
    The caller must supply a concrete legal alternative; otherwise the generic
    scalar implementation remains an admissible fallback.
    """

    for value, name in (
        (output_elements, "output_elements"),
        (reduction_elements, "reduction_elements"),
        (parallel_width, "parallel_width"),
    ):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if parallel_width == 0:
        raise ValueError("parallel_width must be positive")
    if alternative is not None and (
        type(alternative) is not str or not alternative.strip()
    ):
        raise ValueError("alternative must be a nonempty string or None")
    if (
        alternative is None
        or output_elements == 0
        or reduction_elements == 0
        or output_elements >= parallel_width
        or reduction_elements < PATHOLOGICAL_REDUCTION_MIN_WARPS * parallel_width
    ):
        return None
    return (
        f"scalar reduction exposes {output_elements} independent output element(s) "
        f"for reduction extent {reduction_elements}; legal {alternative} lowering exists"
    )


def _optional_count(value: int | None, name: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{name} must be a non-negative integer or None")


def _optional_float(
    value: float | None, name: str, *, unit_interval: bool = False
) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number or None")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number or None")
    if unit_interval and numeric > 1.0:
        raise ValueError(f"{name} must be in [0, 1] or None")


@dataclass(frozen=True, slots=True)
class GpuProfitability:
    """One target-agnostic cost record for a legal GPU candidate."""

    semantic_traffic_bytes: int | None = None
    arithmetic_operation_count: int | None = None
    peak_live_values: int | None = None
    rematerialized_value_count: int | None = None
    estimated_registers_per_thread: int | None = None
    estimated_occupancy_upper_bound: float | None = None
    launch_count: int | None = None
    source_bytes: int | None = None
    precision_cast_read_bytes: int | None = field(default=None, kw_only=True)
    precision_cast_write_bytes: int | None = field(default=None, kw_only=True)
    precision_cast_simultaneous_bytes: int | None = field(default=None, kw_only=True)
    precision_widened_accumulation_terms: int | None = field(default=None, kw_only=True)
    compiled_registers_per_thread: int | None = None
    spill_store_bytes: int | None = None
    spill_load_bytes: int | None = None
    local_bytes: int | None = None
    shared_bytes: int | None = None
    compiled_occupancy_upper_bound: float | None = None
    object_bytes: int | None = None
    compile_seconds: float | None = None
    endpoint_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "semantic_traffic_bytes",
            "arithmetic_operation_count",
            "peak_live_values",
            "rematerialized_value_count",
            "estimated_registers_per_thread",
            "launch_count",
            "source_bytes",
            "precision_cast_read_bytes",
            "precision_cast_write_bytes",
            "precision_cast_simultaneous_bytes",
            "precision_widened_accumulation_terms",
            "compiled_registers_per_thread",
            "spill_store_bytes",
            "spill_load_bytes",
            "local_bytes",
            "shared_bytes",
            "object_bytes",
        ):
            _optional_count(getattr(self, name), name)
        _optional_float(
            self.estimated_occupancy_upper_bound,
            "estimated_occupancy_upper_bound",
            unit_interval=True,
        )
        _optional_float(
            self.compiled_occupancy_upper_bound,
            "compiled_occupancy_upper_bound",
            unit_interval=True,
        )
        _optional_float(self.compile_seconds, "compile_seconds")
        _optional_float(self.endpoint_seconds, "endpoint_seconds")

    @staticmethod
    def _minimize(value: float | None) -> tuple[bool, float]:
        return value is None, 0.0 if value is None else float(value)

    @staticmethod
    def _maximize(value: float | None) -> tuple[bool, float]:
        return value is None, 0.0 if value is None else -float(value)

    @property
    def spill_bytes(self) -> int | None:
        if self.spill_store_bytes is None or self.spill_load_bytes is None:
            return None
        return self.spill_store_bytes + self.spill_load_bytes

    @property
    def precision_cast_bytes(self) -> int | None:
        """Total explicit cast traffic when both transfer directions are known."""

        if (
            self.precision_cast_read_bytes is None
            or self.precision_cast_write_bytes is None
        ):
            return None
        return self.precision_cast_read_bytes + self.precision_cast_write_bytes

    def static_compile_priority(self, generation_index: int) -> tuple[object, ...]:
        """Order legal candidates before compilation without claiming a winner.

        Traffic comes first because it captures endpoint data movement. Explicit
        precision conversions and widened reductions then distinguish otherwise
        similar candidates before register pressure and occupancy. Launches,
        scalar liveness/work, and source size remain deterministic tie-breakers.
        The original generation order is final.
        """

        _optional_count(generation_index, "generation_index")
        return (
            self._minimize(self.semantic_traffic_bytes),
            self._minimize(self.precision_cast_bytes),
            self._minimize(self.precision_cast_simultaneous_bytes),
            self._minimize(self.precision_widened_accumulation_terms),
            self._minimize(self.estimated_registers_per_thread),
            self._maximize(self.estimated_occupancy_upper_bound),
            self._minimize(self.launch_count),
            self._minimize(self.peak_live_values),
            self._minimize(self.arithmetic_operation_count),
            self._minimize(self.source_bytes),
            generation_index,
        )

    def compiled_resource_priority(self) -> tuple[object, ...]:
        """Rank candidates already proven equivalent in endpoint performance.

        This key is only appropriate inside a caller-defined runtime noise band.
        Spills and occupancy lead the resource tie-break; endpoint time is last
        because the caller has already established that those times are tied.
        """

        return (
            self._minimize(self.spill_bytes),
            self._maximize(self.compiled_occupancy_upper_bound),
            self._minimize(self.compiled_registers_per_thread),
            self._minimize(self.local_bytes),
            self._minimize(self.shared_bytes),
            self._minimize(self.launch_count),
            self._minimize(self.semantic_traffic_bytes),
            self._minimize(self.precision_cast_bytes),
            self._minimize(self.precision_cast_simultaneous_bytes),
            self._minimize(self.precision_widened_accumulation_terms),
            self._minimize(self.peak_live_values),
            self._minimize(self.arithmetic_operation_count),
            self._minimize(self.compile_seconds),
            self._minimize(self.source_bytes),
            self._minimize(self.object_bytes),
            self._minimize(self.endpoint_seconds),
        )

    def endpoint_regressions_against(
        self,
        baseline: GpuProfitability,
        *,
        minimum_speedup: float = 1.0,
    ) -> tuple[str, ...]:
        """Return endpoint-profitability failures from complete measured timing."""

        _optional_float(minimum_speedup, "minimum_speedup")
        if minimum_speedup < 1.0:
            raise ValueError("minimum_speedup must be at least one")
        if self.endpoint_seconds is None or baseline.endpoint_seconds is None:
            return ()
        if self.endpoint_seconds <= 0.0 or baseline.endpoint_seconds <= 0.0:
            raise ValueError("endpoint timing must be positive")
        speedup = baseline.endpoint_seconds / self.endpoint_seconds
        if speedup >= minimum_speedup:
            return ()
        reason = (
            f"endpoint speedup {speedup:.6g}x is below "
            f"the required {minimum_speedup:.6g}x"
        )
        return (reason,)

    def resource_regressions_against(
        self,
        baseline: GpuProfitability,
        *,
        endpoint_noise_fraction: float = ENDPOINT_NOISE_FRACTION,
    ) -> tuple[str, ...]:
        """Return measured resource regressions not justified by endpoint speed.

        Relative pressure gates only make a claim when both candidates have
        complete endpoint timing. A speedup larger than the caller's noise
        band may justify a resource trade-off; otherwise new spill traffic or
        register growth that lowers target-estimated occupancy is rejected.
        Missing resource evidence remains unknown rather than being guessed.
        """

        _optional_float(
            endpoint_noise_fraction,
            "endpoint_noise_fraction",
            unit_interval=True,
        )
        if self.endpoint_seconds is None or baseline.endpoint_seconds is None:
            return ()
        if self.endpoint_seconds <= 0.0 or baseline.endpoint_seconds <= 0.0:
            raise ValueError("endpoint timing must be positive")
        if self.endpoint_seconds < baseline.endpoint_seconds * (
            1.0 - float(endpoint_noise_fraction)
        ):
            return ()

        reasons = []
        spill_bytes = self.spill_bytes
        baseline_spill_bytes = baseline.spill_bytes
        if (
            spill_bytes is not None
            and baseline_spill_bytes is not None
            and spill_bytes > baseline_spill_bytes
        ):
            reasons.append(
                "spill traffic grows "
                f"from {baseline_spill_bytes} to {spill_bytes} bytes "
                "without an endpoint win"
            )

        registers = self.compiled_registers_per_thread
        baseline_registers = baseline.compiled_registers_per_thread
        occupancy = self.compiled_occupancy_upper_bound
        baseline_occupancy = baseline.compiled_occupancy_upper_bound
        if (
            registers is not None
            and baseline_registers is not None
            and occupancy is not None
            and baseline_occupancy is not None
            and registers > baseline_registers
            and occupancy < baseline_occupancy
        ):
            reasons.append(
                "registers grow "
                f"from {baseline_registers} to {registers} per thread while "
                "target-estimated occupancy falls "
                f"from {baseline_occupancy:.6g} to {occupancy:.6g} "
                "without an endpoint win"
            )
        return tuple(reasons)

    def to_payload(self) -> dict[str, object]:
        """Serialize every known/unknown fact; do not erase negative evidence."""

        fields = asdict(self)
        return {
            "schema": "vibeqc.compiler.gpu-profitability.v1",
            "static": {
                name: fields[name]
                for name in (
                    "semantic_traffic_bytes",
                    "arithmetic_operation_count",
                    "peak_live_values",
                    "rematerialized_value_count",
                    "estimated_registers_per_thread",
                    "estimated_occupancy_upper_bound",
                    "launch_count",
                    "source_bytes",
                    "precision_cast_read_bytes",
                    "precision_cast_write_bytes",
                    "precision_cast_simultaneous_bytes",
                    "precision_widened_accumulation_terms",
                )
            },
            "compiled": {
                name: fields[name]
                for name in (
                    "compiled_registers_per_thread",
                    "spill_store_bytes",
                    "spill_load_bytes",
                    "local_bytes",
                    "shared_bytes",
                    "compiled_occupancy_upper_bound",
                    "object_bytes",
                    "compile_seconds",
                )
            },
            "endpoint_seconds": self.endpoint_seconds,
        }
