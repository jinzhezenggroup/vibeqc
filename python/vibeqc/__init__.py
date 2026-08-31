"""Thin Python interface to the versioned native VIBEQC ABI."""

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

__all__ = [
    "Atom",
    "BatchItemResult",
    "BatchResult",
    "Calculator",
    "DensityFittingMetricDiagnostic",
    "EigensolverDiagnostic",
    "InactiveEigensolverProfileEntry",
    "MethodCapabilities",
    "PppsQueueProfile",
    "PreparedBatch",
    "Primitive",
    "Result",
    "Shell",
    "ShellClassProfileEntry",
    "method_capabilities",
]
__version__ = "0.1.0"
