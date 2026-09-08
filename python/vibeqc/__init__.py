"""Thin Python interface to the versioned native VIBEQC ABI."""

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

__all__ = [
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
    "InactiveEigensolverProfileEntry",
    "MethodCapabilities",
    "PppsQueueProfile",
    "PreparedBatch",
    "Primitive",
    "Result",
    "Shell",
    "ShellClassProfileEntry",
    "basis_capability",
    "electron_state",
    "import_bse",
    "load_basis",
    "method_capabilities",
]
__version__ = "0.1.0"
