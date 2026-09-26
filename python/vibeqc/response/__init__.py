"""Installed method-neutral orbital-response contracts.

Method-specific native providers may still live in development adapters, but
response problem identity and matrix-free RHF/CPKS operator equations are owned
by the installed runtime package.
"""

from .operators import (
    CPKSResponseOperator,
    DenseMatrixResponseOperator,
    RHFResponseOperator,
    cpks_operator_identity,
    rhf_operator_identity,
    validate_rotation_layout,
)
from .problem import (
    ResponseCompatibilityError,
    ResponseProblem,
    ResponseSolveError,
    ResponseUnsupported,
    RotationLayout,
)

__all__ = [
    "CPKSResponseOperator",
    "DenseMatrixResponseOperator",
    "RHFResponseOperator",
    "ResponseCompatibilityError",
    "ResponseProblem",
    "ResponseSolveError",
    "ResponseUnsupported",
    "RotationLayout",
    "cpks_operator_identity",
    "rhf_operator_identity",
    "validate_rotation_layout",
]
