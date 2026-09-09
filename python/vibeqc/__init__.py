"""Thin Python interface to the versioned native VIBEQC ABI."""

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
    MethodCapabilities,
    Primitive,
    Result,
    Shell,
    method_capabilities,
)
from .elements import ElectronState, electron_state
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

__all__ = [
    "AccuracyAssessment",
    "Atom",
    "BasisProvenance",
    "BasisSet",
    "BasisShell",
    "BatchItemResult",
    "BatchResult",
    "Calculator",
    "DensityFittingMetricDiagnostic",
    "EigensolverDiagnostic",
    "ElectronState",
    "ElementBasis",
    "ErrorEvidence",
    "EvidenceKind",
    "InactiveEigensolverProfileEntry",
    "MethodCapabilities",
    "ObservableTarget",
    "PppsQueueProfile",
    "PreparedBatch",
    "Primitive",
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
    "basis_capability",
    "compare_observables",
    "electron_state",
    "estimate_hf_resources",
    "import_bse",
    "load_basis",
    "method_capabilities",
    "plan_resources",
]
__version__ = "0.1.0"
