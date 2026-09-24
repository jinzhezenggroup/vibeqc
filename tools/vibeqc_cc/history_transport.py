"""Bounded DIIS-history recycling for diagnosed CC state transport.

History vectors are never trusted by shape alone.  Source amplitudes are
transported through the existing #190 exact/projected owners, while every
retained error vector is recomputed by the target operator before publication.
No source residual or convergence claim crosses this boundary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from vibeqc_compiler.common.arrays import immutable

from .gpu_state import AmplitudeSnapshot
from .projected_transport import project_amplitude_guess
from .state_transport import StateTransport, TransportCompatibility


@dataclass(frozen=True)
class HistoryRecyclePolicy:
    """Bound retained history and target-side amplitude/residual storage."""

    maximum_vectors: int = 8
    maximum_total_elements: int = 16_000_000

    def __post_init__(self) -> None:
        for name in ("maximum_vectors", "maximum_total_elements"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class TargetResidualEvaluator:
    """Target-identity-bound residual evaluator used for recycled DIIS errors."""

    state_identity: str
    evaluate: Callable[[AmplitudeSnapshot], tuple[np.ndarray, np.ndarray]]

    def __post_init__(self) -> None:
        if not isinstance(self.state_identity, str) or not self.state_identity:
            raise ValueError("state_identity must be a nonempty target identity")
        if not callable(self.evaluate):
            raise TypeError("evaluate must be callable")

    def __call__(self, amplitudes: AmplitudeSnapshot) -> tuple[np.ndarray, np.ndarray]:
        return self.evaluate(amplitudes)


@dataclass(frozen=True)
class RecycledHistoryEntry:
    """One target-bound amplitude vector with a freshly evaluated target error."""

    source_index: int
    amplitudes: AmplitudeSnapshot
    residual_singles: np.ndarray
    residual_doubles: np.ndarray
    residual_max_abs: float

    def __post_init__(self) -> None:
        if type(self.source_index) is not int or self.source_index < 0:
            raise ValueError("source_index must be a nonnegative integer")
        if not isinstance(self.amplitudes, AmplitudeSnapshot):
            raise TypeError("history amplitudes must be an AmplitudeSnapshot")
        expected = (self.amplitudes.t1.shape, self.amplitudes.t2.shape)
        frozen = []
        for name, value, shape in zip(
            ("residual_singles", "residual_doubles"),
            (self.residual_singles, self.residual_doubles),
            expected,
            strict=True,
        ):
            if (
                not isinstance(value, np.ndarray)
                or value.dtype != np.float64
                or value.shape != shape
                or not np.isfinite(value).all()
            ):
                raise ValueError(
                    f"{name} must be a finite FP64 array matching target amplitudes"
                )
            frozen.append(immutable(value))
        object.__setattr__(self, "residual_singles", frozen[0])
        object.__setattr__(self, "residual_doubles", frozen[1])
        if not math.isfinite(self.residual_max_abs) or self.residual_max_abs < 0:
            raise ValueError("residual_max_abs must be finite and nonnegative")


@dataclass(frozen=True)
class RecycledHistory:
    """Target-recomputed DIIS history; this is not target convergence evidence."""

    transport_id: str
    source_reference_id: str
    target_reference_id: str
    source_count: int
    dropped_prefix_count: int
    target_residual_evaluations: int
    entries: tuple[RecycledHistoryEntry, ...]
    kind: str = field(init=False, default="target_recomputed_diis_history")

    def __post_init__(self) -> None:
        for name in ("transport_id", "source_reference_id", "target_reference_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty identity")
        if type(self.source_count) is not int or self.source_count < 0:
            raise ValueError("source_count must be a nonnegative integer")
        if (
            type(self.dropped_prefix_count) is not int
            or not 0 <= self.dropped_prefix_count <= self.source_count
        ):
            raise ValueError("invalid dropped_prefix_count")
        if self.target_residual_evaluations != len(self.entries):
            raise ValueError(
                "each retained history vector requires one target residual"
            )
        if self.source_count - self.dropped_prefix_count != len(self.entries):
            raise ValueError("history accounting does not match retained entries")
        if any(
            entry.amplitudes.reference_id != self.target_reference_id
            for entry in self.entries
        ):
            raise ValueError("recycled history must be bound to the target reference")


def _target_elements(transport: StateTransport) -> int:
    nocc = transport.target.nocc
    nvir = transport.target.nvir
    return nocc * nvir + nocc * nocc * nvir * nvir


def _validate_history_source(
    transport: StateTransport, source: AmplitudeSnapshot
) -> None:
    if not isinstance(source, AmplitudeSnapshot):
        raise TypeError("source_history entries must be AmplitudeSnapshot objects")
    if source.reference_id != transport.source.reference_id:
        raise ValueError("history amplitude reference does not match transport source")
    o, v = transport.source.nocc, transport.source.nvir
    if source.t1.shape != (o, v) or source.t2.shape != (o, o, v, v):
        raise ValueError("history amplitude shape does not match transport source")


def _transport_history_amplitudes(
    transport: StateTransport, source: AmplitudeSnapshot
) -> AmplitudeSnapshot:
    _validate_history_source(transport, source)

    if transport.compatibility in (
        TransportCompatibility.identity,
        TransportCompatibility.exact_orbital_rotation,
    ):
        target = transport.rotate_amplitudes(source.t1, source.t2)
    elif transport.compatibility is TransportCompatibility.projected_warm_start:
        target = project_amplitude_guess(transport, source.t1, source.t2).amplitudes
    else:
        raise ValueError("incompatible state transport requires DIIS history reset")

    if not isinstance(target, AmplitudeSnapshot):
        raise TypeError("state transport returned an invalid target amplitude snapshot")
    if target.reference_id != transport.target.reference_id:
        raise ValueError("transported history is not bound to the target reference")
    return target


def recycle_diis_history(
    transport: StateTransport,
    source_history: Sequence[AmplitudeSnapshot],
    target_residual: TargetResidualEvaluator,
    *,
    policy: HistoryRecyclePolicy | None = None,
) -> RecycledHistory:
    """Transport a bounded recent DIIS history and recompute every target error.

    ``source_history`` is ordered oldest to newest.  If it exceeds the configured
    capacity, only the newest vectors are considered.  Source DIIS errors are
    intentionally not accepted by this API: every retained vector calls
    ``target_residual`` after transport, so stale operator images cannot cross a
    model/basis frame boundary.  The result is initialization state only; target
    convergence and energy must still be established by the target solver.
    """
    if not isinstance(transport, StateTransport):
        raise TypeError("transport must be a classified StateTransport")
    selected = HistoryRecyclePolicy() if policy is None else policy
    if not isinstance(selected, HistoryRecyclePolicy):
        raise TypeError("policy must be HistoryRecyclePolicy")
    if not isinstance(source_history, Sequence):
        raise TypeError("source_history must be an ordered sequence")
    if not isinstance(target_residual, TargetResidualEvaluator):
        raise TypeError("target_residual must be a TargetResidualEvaluator")
    if target_residual.state_identity != transport.target.identity:
        raise ValueError(
            "target residual evaluator does not match transport target identity"
        )
    if transport.compatibility is TransportCompatibility.incompatible:
        raise ValueError("incompatible state transport requires DIIS history reset")

    source_count = len(source_history)
    keep = min(source_count, selected.maximum_vectors)
    dropped = source_count - keep
    retained = source_history[dropped:]

    # Account for both amplitudes and freshly recomputed residuals before any
    # transport or target-operator callback allocates output state.
    target_elements = _target_elements(transport)
    required = 2 * keep * target_elements
    if required > selected.maximum_total_elements:
        raise ValueError("recycled DIIS history exceeds the configured element budget")

    # Validate the entire retained input before the first target-operator call.
    # A stale later vector must not perform partial target work for earlier ones.
    for source in retained:
        _validate_history_source(transport, source)

    entries: list[RecycledHistoryEntry] = []
    for offset, source in enumerate(retained, start=dropped):
        target = _transport_history_amplitudes(transport, source)
        residual = target_residual(target)
        if not isinstance(residual, tuple) or len(residual) != 2:
            raise TypeError("target_residual must return (singles, doubles) arrays")
        r1, r2 = residual
        residual_max = max(
            float(np.max(np.abs(r1), initial=0.0))
            if isinstance(r1, np.ndarray)
            else math.inf,
            float(np.max(np.abs(r2), initial=0.0))
            if isinstance(r2, np.ndarray)
            else math.inf,
        )
        entries.append(
            RecycledHistoryEntry(
                source_index=offset,
                amplitudes=target,
                residual_singles=r1,
                residual_doubles=r2,
                residual_max_abs=residual_max,
            )
        )

    return RecycledHistory(
        transport_id=transport.identity,
        source_reference_id=transport.source.reference_id,
        target_reference_id=transport.target.reference_id,
        source_count=source_count,
        dropped_prefix_count=dropped,
        target_residual_evaluations=len(entries),
        entries=tuple(entries),
    )
