"""Thin Python interface to the versioned native VIBEQC ABI."""

from vibeqc_compiler.dft.grid import GridPolicy, GridProfile, GridSpec

from ._cuda_runtime import install_native_loader as _install_native_loader
from .accuracy import (
    AccuracyAssessment,
    ErrorEvidence,
    EvidenceKind,
    ObservableTarget,
    ResolvedModel,
    TargetAccuracy,
    compare_observables,
)
from .basis import BasisProvenance, BasisSet, BasisShell, ElementBasis, load_basis
from .basis_capabilities import basis_capability
from .basis_import import import_bse
from .batch import (
    BatchItemResult,
    BatchResult,
    DensityFittingMetricDiagnostic,
    EigensolverDiagnostic,
    InactiveEigensolverProfileEntry,
    PppsQueueProfile,
    PreparedBatch,
    ShellClassProfileEntry,
)
from .calculator import (
    Atom,
    Calculator,
    CorrelationResult,
    MethodCapabilities,
    Primitive,
    Result,
    Shell,
    method_capabilities,
)
from .dispersion import (
    D3CorrectionBatch,
    D3CorrectionResult,
    D3RuntimeDiagnostic,
    D4CorrectionBatch,
    D4CorrectionResult,
    D4RuntimeDiagnostic,
    evaluate_d3_correction,
    evaluate_d4_correction,
)
from .elements import ElectronState, electron_state
from .fock import FockBuildSpec, FockEvaluation, FockPlan, FockScfResult, FockTerm
from .ks import FunctionalSpec, KsOptions
from .ks_diagnostics import (
    KsDiagnostic,
    KsEnergyComponents,
    KsIteration,
    KsTransportDiagnostic,
)
from .mean_field import (
    FixedDensityExchangeEvaluation,
    FixedDensityMeanField,
    MeanFieldEvaluation,
    assemble_fixed_density_exchange,
    exchange_operator_key,
)
from .overlap import cross_overlap
from .progressive import ProgressiveResult, projected_singlepoint
from .projection import (
    OccupiedProjection,
    ProjectionDiagnostics,
    ProjectionPolicy,
    ProjectionRejected,
    project_density,
    project_occupied,
)
from .r2scan3c import load_r2scan3c_basis
from .resources import (
    ResourceAllocationError,
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourcePlan,
    ResourceRequest,
    ResourceSession,
    plan_resources,
)
from .resources_hf import estimate_hf_resources
from .resources_ks import estimate_ks_resources

_install_native_loader()
del _install_native_loader

__all__ = [
    "AccuracyAssessment",
    "Atom",
    "BasisProvenance",
    "BasisSet",
    "BasisShell",
    "BatchItemResult",
    "BatchResult",
    "Calculator",
    "CorrelationResult",
    "D3CorrectionBatch",
    "D3CorrectionResult",
    "D3RuntimeDiagnostic",
    "D4CorrectionBatch",
    "D4CorrectionResult",
    "D4RuntimeDiagnostic",
    "DensityFittingMetricDiagnostic",
    "EigensolverDiagnostic",
    "ElectronState",
    "ElementBasis",
    "ErrorEvidence",
    "EvidenceKind",
    "FixedDensityExchangeEvaluation",
    "FixedDensityMeanField",
    "FockBuildSpec",
    "FockEvaluation",
    "FockPlan",
    "FockScfResult",
    "FockTerm",
    "FunctionalSpec",
    "GridPolicy",
    "GridProfile",
    "GridSpec",
    "InactiveEigensolverProfileEntry",
    "KsDiagnostic",
    "KsEnergyComponents",
    "KsIteration",
    "KsOptions",
    "KsTransportDiagnostic",
    "MeanFieldEvaluation",
    "MethodCapabilities",
    "ObservableTarget",
    "OccupiedProjection",
    "PppsQueueProfile",
    "PreparedBatch",
    "Primitive",
    "ProgressiveResult",
    "ProjectionDiagnostics",
    "ProjectionPolicy",
    "ProjectionRejected",
    "ResolvedModel",
    "ResourceAllocationError",
    "ResourceBudget",
    "ResourceCandidate",
    "ResourceEstimate",
    "ResourceIdentity",
    "ResourcePlan",
    "ResourceRequest",
    "ResourceSession",
    "Result",
    "Shell",
    "ShellClassProfileEntry",
    "TargetAccuracy",
    "assemble_fixed_density_exchange",
    "basis_capability",
    "compare_observables",
    "cross_overlap",
    "electron_state",
    "estimate_hf_resources",
    "estimate_ks_resources",
    "evaluate_d3_correction",
    "evaluate_d4_correction",
    "exchange_operator_key",
    "import_bse",
    "load_basis",
    "load_r2scan3c_basis",
    "method_capabilities",
    "plan_resources",
    "project_density",
    "project_occupied",
    "projected_singlepoint",
]
__version__ = "0.1.0"
