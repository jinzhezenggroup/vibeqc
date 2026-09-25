"""Bounded symbolic Array API frontend for TensorIR.

This package is compiler-internal while issue #633 establishes the semantic
contract.  It intentionally advertises only the explicitly listed capability
subset and lowers every captured value to ordinary TensorIR.
"""

from . import namespace
from .array import ExactScalar, VibeArray
from .capabilities import FRONTEND_VERSION, SUPPORTED_FUNCTIONS, capabilities
from .interop import (
    DLPACK_INTEROP_VERSION,
    DLPackDevice,
    DLPackImport,
    DLPackInteropError,
    dlpack_device,
    import_dlpack,
)
from .trace import input_array, trace

__all__ = [
    "DLPACK_INTEROP_VERSION",
    "FRONTEND_VERSION",
    "SUPPORTED_FUNCTIONS",
    "DLPackDevice",
    "DLPackImport",
    "DLPackInteropError",
    "ExactScalar",
    "VibeArray",
    "capabilities",
    "dlpack_device",
    "import_dlpack",
    "input_array",
    "namespace",
    "trace",
]
