"""Bounded projected RCCSD amplitude warm starts.

This module handles only the approximate ``projected_warm_start`` outcome from
:mod:`tools.vibeqc_cc.state_transport`.  It never turns a rectangular orbital
map into an exact-state claim: projected amplitudes are returned as an explicit
initial guess tied to the target reference and must still be refined and
accepted by the target RCCSD residual/energy gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .gpu_state import AmplitudeSnapshot
from .state_transport import StateTransport, TransportCompatibility


@dataclass(frozen=True)
class ProjectedAmplitudePolicy:
    """Resource and map-validity gates for host-side projected warm starts."""

    maximum_elements: int = 16_000_000
    map_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if type(self.maximum_elements) is not int or self.maximum_elements < 1:
            raise ValueError("maximum_elements must be a positive integer")
        try:
            if (
                isinstance(self.map_tolerance, (str, bytes))
                or np.iscomplexobj(self.map_tolerance)
                or np.ndim(self.map_tolerance) != 0
            ):
                raise ValueError("not a real scalar")
            tolerance = float(self.map_tolerance)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("map_tolerance must be a real scalar") from error
        if not math.isfinite(tolerance) or not 0 < tolerance <= 1e-4:
            raise ValueError("map_tolerance must be finite and in (0, 1e-4]")
        # Frozen dataclasses do not detach a caller-owned zero-dimensional array.
        # Retain the validated scalar value, never its mutable container.
        object.__setattr__(self, "map_tolerance", tolerance)


@dataclass(frozen=True)
class ProjectedAmplitudeDiagnostics:
    """Evidence retained with an approximate amplitude proposal."""

    source_elements: int
    target_elements: int
    occupied_rank: int
    virtual_rank: int
    target_virtual_nullity: int
    occupied_max_singular: float
    virtual_max_singular: float
    singles_norm_ratio: float
    doubles_norm_ratio: float


@dataclass(frozen=True)
class ProjectedAmplitudeGuess:
    """Target-bound initial amplitudes that do not assert CC convergence."""

    transport_id: str
    source_reference_id: str
    target_reference_id: str
    amplitudes: AmplitudeSnapshot
    diagnostics: ProjectedAmplitudeDiagnostics
    kind: str = field(init=False, default="projected_warm_start")

    def __post_init__(self) -> None:
        for name in ("transport_id", "source_reference_id", "target_reference_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty identity")
        if self.source_reference_id == self.target_reference_id:
            raise ValueError(
                "projected warm starts require distinct source/target references"
            )
        if not isinstance(self.amplitudes, AmplitudeSnapshot):
            raise TypeError("amplitudes must be an AmplitudeSnapshot")
        if self.amplitudes.reference_id != self.target_reference_id:
            raise ValueError(
                "projected amplitudes must be bound to the target reference"
            )
        if not isinstance(self.diagnostics, ProjectedAmplitudeDiagnostics):
            raise TypeError("diagnostics must be ProjectedAmplitudeDiagnostics")


def _amplitude_elements(nocc: int, nvir: int) -> int:
    singles = nocc * nvir
    doubles = nocc * nocc * nvir * nvir
    return singles + doubles


def _maximum_singular(value: np.ndarray) -> float:
    singular = np.linalg.svd(value, compute_uv=False)
    if not np.isfinite(singular).all():
        raise ValueError("projected amplitude map has nonfinite singular values")
    return float(singular[0]) if singular.size else 0.0


def _norm_ratio(projected: np.ndarray, source: np.ndarray) -> float:
    source_norm = float(np.linalg.norm(source))
    projected_norm = float(np.linalg.norm(projected))
    if not math.isfinite(source_norm) or not math.isfinite(projected_norm):
        raise ValueError("projected amplitude norm is nonfinite")
    if source_norm == 0.0:
        if projected_norm != 0.0:
            raise ValueError("zero source amplitudes projected to a nonzero target")
        return 1.0
    return projected_norm / source_norm


def project_amplitude_guess(
    transport: StateTransport,
    t1: np.ndarray,
    t2: np.ndarray,
    *,
    policy: ProjectedAmplitudePolicy | None = None,
) -> ProjectedAmplitudeGuess:
    """Project source RCCSD amplitudes into a diagnosed target orbital space.

    Rectangular occupied/virtual maps are applied covariantly on every amplitude
    index.  Missing target virtual directions therefore start at exactly zero;
    they are not invented from orbital energies or copied by array position.
    The returned ``AmplitudeSnapshot`` is suitable only as a target warm start.
    A target solve must recompute its own denominators, residuals and energy
    before any physical result is published.
    """
    if not isinstance(transport, StateTransport):
        raise TypeError("transport must be a classified StateTransport")
    if transport.compatibility is not TransportCompatibility.projected_warm_start:
        raise ValueError(
            "project_amplitude_guess requires projected_warm_start compatibility"
        )
    selected = ProjectedAmplitudePolicy() if policy is None else policy
    if not isinstance(selected, ProjectedAmplitudePolicy):
        raise TypeError("policy must be ProjectedAmplitudePolicy")

    source = AmplitudeSnapshot(transport.source.reference_id, t1, t2)
    if source.t1.shape != (transport.source.nocc, transport.source.nvir):
        raise ValueError("source amplitudes do not match the transport identity")

    source_elements = _amplitude_elements(transport.source.nocc, transport.source.nvir)
    target_elements = _amplitude_elements(transport.target.nocc, transport.target.nvir)
    if target_elements > selected.maximum_elements:
        raise ValueError(
            "projected target amplitudes exceed the configured element budget"
        )

    occupied_max = _maximum_singular(transport.occupied_map)
    virtual_max = _maximum_singular(transport.virtual_map)
    if max(occupied_max, virtual_max) > 1.0 + selected.map_tolerance:
        raise ValueError("projected orbital map is not a bounded overlap contraction")

    singles = transport.occupied_map @ source.t1 @ transport.virtual_map.T
    doubles = np.einsum(
        "ki,ijab->kjab", transport.occupied_map, source.t2, optimize=False
    )
    doubles = np.einsum(
        "lj,kjab->klab", transport.occupied_map, doubles, optimize=False
    )
    doubles = np.einsum("ca,klab->klcb", transport.virtual_map, doubles, optimize=False)
    doubles = np.einsum("db,klcb->klcd", transport.virtual_map, doubles, optimize=False)
    target = AmplitudeSnapshot(
        transport.target.reference_id,
        np.ascontiguousarray(singles, dtype=np.float64),
        np.ascontiguousarray(doubles, dtype=np.float64),
    )
    diagnostics = ProjectedAmplitudeDiagnostics(
        source_elements=source_elements,
        target_elements=target_elements,
        occupied_rank=transport.diagnostics.occupied_rank,
        virtual_rank=transport.diagnostics.virtual_rank,
        target_virtual_nullity=transport.diagnostics.target_virtual_nullity,
        occupied_max_singular=occupied_max,
        virtual_max_singular=virtual_max,
        singles_norm_ratio=_norm_ratio(target.t1, source.t1),
        doubles_norm_ratio=_norm_ratio(target.t2, source.t2),
    )
    return ProjectedAmplitudeGuess(
        transport.identity,
        transport.source.reference_id,
        transport.target.reference_id,
        target,
        diagnostics,
    )
