"""Internal quadrature/AO development interface; no executable DFT method."""

from importlib import import_module
from typing import TYPE_CHECKING

from .ao import NativeAO, directional_ao_jets, jet_indices
from .density_source import DensitySource, DensityStamp
from .features import density_features, orbital_features, spin_densities
from .grid import (
    ExplicitGrid,
    GridPolicy,
    GridProfile,
    GridSpec,
    MolecularGrid,
    grid_policy_provenance,
    partition_weights,
)
from .nonlocal_integration import (
    FixedDensityNonlocalCorrelation,
    NonlocalGeometry,
    NonlocalIntegral,
)
from .nonlocal_reference import (
    assemble_nonlocal_potential_reference,
    nonlocal_energy_density_reference,
    nonlocal_energy_reference,
    nonlocal_explicit_geometry_derivatives_reference,
    nonlocal_feature_derivatives_reference,
    nonlocal_kernel_matrix_reference,
)
from .xc_schedule import (
    DEVICE_FUSED,
    HOST_UNFUSED,
    GridXcCandidateAssessment,
    GridXcCandidateLimits,
    GridXcCandidateShape,
    GridXcExecutionSchedule,
    GridXcScheduleCandidate,
    GridXcScientificIdentity,
    assess_grid_xc_schedule,
    grid_xc_schedule,
    rank_grid_xc_candidates,
    rank_grid_xc_schedules,
)

if TYPE_CHECKING:
    from .prepared import PreparedGrid, PreparedGridBatch


def __getattr__(name: str) -> object:
    """Activate prepared execution only when that public capability is requested.

    Importing a grid specification executes this package initializer too. Keep
    that metadata path independent of the prepared grid's CUDA/JIT adapters,
    while forwarding explicit requests to the canonical class objects.
    """
    if name in ("PreparedGrid", "PreparedGridBatch"):
        value = getattr(import_module(".prepared", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Preserve discovery of lazily exported prepared-grid classes."""
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "DEVICE_FUSED",
    "HOST_UNFUSED",
    "DensitySource",
    "DensityStamp",
    "ExplicitGrid",
    "FixedDensityNonlocalCorrelation",
    "GridPolicy",
    "GridProfile",
    "GridSpec",
    "GridXcCandidateAssessment",
    "GridXcCandidateLimits",
    "GridXcCandidateShape",
    "GridXcExecutionSchedule",
    "GridXcScheduleCandidate",
    "GridXcScientificIdentity",
    "MolecularGrid",
    "NativeAO",
    "NonlocalGeometry",
    "NonlocalIntegral",
    "PreparedGrid",
    "PreparedGridBatch",
    "assemble_nonlocal_potential_reference",
    "assess_grid_xc_schedule",
    "density_features",
    "directional_ao_jets",
    "grid_policy_provenance",
    "grid_xc_schedule",
    "jet_indices",
    "nonlocal_energy_density_reference",
    "nonlocal_energy_reference",
    "nonlocal_explicit_geometry_derivatives_reference",
    "nonlocal_feature_derivatives_reference",
    "nonlocal_kernel_matrix_reference",
    "orbital_features",
    "partition_weights",
    "rank_grid_xc_candidates",
    "rank_grid_xc_schedules",
    "spin_densities",
]
