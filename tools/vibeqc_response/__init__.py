"""Shared native response-solver infrastructure for HF/DFT orbital response.

The package intentionally separates three concerns:

* :mod:`problem` owns immutable scientific/compatibility snapshots.
* :mod:`operators` owns matrix-free JVP/VJP actions.
* :mod:`krylov` owns bounded linear-solver and recycling state.

CC-specific right-hand sides and weights remain in their consuming issue.  The
shared contract here is only the orbital-response problem, operator action and
linear solve.
"""

from .backends import CudaDFJKBackend, DenseAOResponseBackend, NativeJKBackend
from .krylov import (
    DiagonalPreconditioner,
    GMRESOptions,
    KrylovRecycleSpace,
    MultiRHSResult,
    SolveResult,
    solve,
    solve_many,
)
from .operators import (
    CPKSResponseOperator,
    DenseMatrixResponseOperator,
    RHFResponseOperator,
)
from .oracle import explicit_rhf_response_matrix, finite_rotation_jvp
from .problem import (
    ResponseCompatibilityError,
    ResponseProblem,
    ResponseSolveError,
    ResponseUnsupported,
    RotationLayout,
)
from .xc import FixedDensityXCDerivativeKernel

__all__ = [
    "CPKSResponseOperator",
    "CudaDFJKBackend",
    "DenseAOResponseBackend",
    "DenseMatrixResponseOperator",
    "DiagonalPreconditioner",
    "FixedDensityXCDerivativeKernel",
    "GMRESOptions",
    "KrylovRecycleSpace",
    "MultiRHSResult",
    "NativeJKBackend",
    "RHFResponseOperator",
    "ResponseCompatibilityError",
    "ResponseProblem",
    "ResponseSolveError",
    "ResponseUnsupported",
    "RotationLayout",
    "SolveResult",
    "explicit_rhf_response_matrix",
    "finite_rotation_jvp",
    "solve",
    "solve_many",
]
