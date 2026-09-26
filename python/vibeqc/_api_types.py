"""Stable public API records shared by the single-point and batch facades.

The execution facades deliberately import these records instead of defining
their own copies.  Keeping the records in a dependency-light module lets
model-resolution and result-translation tests run without loading the native
library or constructing an execution context.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass, field

from .elements import atomic_number as element_number

if typing.TYPE_CHECKING:
    import numpy as np

    from .accuracy import AccuracyAssessment
    from .ks_diagnostics import KsDiagnostic, KsTransportDiagnostic


@dataclass(frozen=True)
class Atom:
    """Validated nucleus data in Bohr coordinates."""

    atomic_number: int
    position: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Own finite coordinates and validate nuclei independently of basis data."""
        z = element_number(self.atomic_number)
        xyz = tuple(float(v) for v in self.position)
        if len(xyz) != 3 or not all(math.isfinite(v) for v in xyz):
            raise ValueError("atom coordinates must be three finite Bohr values")
        object.__setattr__(self, "atomic_number", z)
        object.__setattr__(self, "position", xyz)

    @classmethod
    def from_value(cls, value: Atom | tuple[str | int, typing.Sequence[float]]) -> Atom:
        """Normalize the public ``(element, position)`` tuple form."""
        if isinstance(value, cls):
            return value
        element, position = value
        atomic_number = element_number(element)
        xyz = tuple(float(component) for component in position)
        if len(xyz) != 3:
            raise ValueError("atom coordinates must have three components")
        return cls(atomic_number, xyz)  # type: ignore[arg-type]


@dataclass(frozen=True)
class Primitive:
    exponent: float
    coefficient: float


@dataclass(frozen=True)
class Shell:
    atom_index: int
    angular_momentum: int
    primitives: tuple[Primitive, ...]


@dataclass(frozen=True)
class CorrelationResult:
    """Canonical post-HF components and completed phase diagnostics."""

    reference_energy: float
    opposite_spin_energy: float
    same_spin_energy: float
    minimum_absolute_denominator: float
    reference_residual: float
    numeric_capacity_bytes: int
    energy_tile_count: int
    mo_host_staging: bool
    correlation_owned_device_bytes: int
    correlation_provider_retained_bytes: int
    mo_transfer_bytes: int
    host_to_device_ms: float
    device_to_host_ms: float
    transform_library_ms: float
    tensor_kernel_ms: float
    equation_hash: str
    response_iterations: int
    response_restarts: int
    response_absolute_residual: float
    response_relative_residual: float
    response_workspace_bytes: int
    derivative_workspace_bytes: int
    planned_endpoint_peak_bytes: int
    measured_endpoint_peak_bytes: int
    force_provenance_flags: int
    response_operator_hash: str
    measured_response_workspace_peak_bytes: int
    response_workspace_allocation_count: int
    ccsd_iterations: int
    ccsd_diis_restarts: int
    ccsd_correlation_energy: float
    ccsd_energy_change: float
    ccsd_singles_residual_max: float
    ccsd_doubles_residual_max: float
    ccsd_replay_singles_residual_max: float
    ccsd_replay_doubles_residual_max: float
    ccsd_setup_h2d_bytes: int
    ccsd_scalar_d2h_bytes: int
    ccsd_amplitude_d2h_bytes: int
    ccsd_synchronizations: int
    ccsd_replay_equation_hash: str
    ccsd_t_triples_energy: float
    ccsd_t_virtual_triples: int
    ccsd_t_workspace_bytes: int
    ccsd_t_equation_hash: str


@dataclass(frozen=True)
class Result:
    """Single-system output with separate convergence and stationarity measures."""

    energy: float
    forces: np.ndarray | None
    converged: bool
    iterations: int
    energy_change: float
    density_rms: float
    executed_backend: str
    basis_metadata: dict = field(default_factory=dict)
    accuracy: AccuracyAssessment | None = None
    resource_diagnostics: dict | None = None
    precision: dict | None = None
    correlation: CorrelationResult | None = None
    physical_residual_rms: float | None = None
    ks_diagnostic: KsDiagnostic | None = None
    ks_transport_diagnostic: KsTransportDiagnostic | None = None
    dispersion: object | None = None


@dataclass(frozen=True)
class MethodCapabilities:
    """Executable properties reported by the native method registry."""

    method: str
    family: str
    available: bool
    supports_batch: bool
    supported_properties: frozenset[str]


__all__ = [
    "Atom",
    "CorrelationResult",
    "MethodCapabilities",
    "Primitive",
    "Result",
    "Shell",
]
