"""Bounded symbolic Array API frontend for TensorIR.

This package is compiler-internal while issue #633 establishes the semantic
contract.  It intentionally advertises only the explicitly listed capability
subset and lowers every captured value to ordinary TensorIR.
"""

from . import namespace
from .array import ExactScalar, VibeArray
from .capabilities import FRONTEND_VERSION, SUPPORTED_FUNCTIONS, capabilities
from .trace import input_array, trace

__all__ = [
    "FRONTEND_VERSION",
    "SUPPORTED_FUNCTIONS",
    "ExactScalar",
    "VibeArray",
    "capabilities",
    "input_array",
    "namespace",
    "trace",
]
