"""Thin Python interface to the versioned native VIBEQC ABI."""

from .calculator import (
    Atom,
    Calculator,
    MethodCapabilities,
    Primitive,
    Result,
    Shell,
    method_capabilities,
)
from .batch import (
    BatchItemResult,
    BatchResult,
    EigensolverDiagnostic,
    PppsQueueProfile,
    PreparedBatch,
    ShellClassProfileEntry,
)

__all__ = [
    "Atom",
    "BatchItemResult",
    "BatchResult",
    "Calculator",
    "EigensolverDiagnostic",
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
