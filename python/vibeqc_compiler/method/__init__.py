"""Canonical method composition above scientific compiler primitives."""

from vibeqc_compiler.common.nonlocal_correlation import (
    NONLOCAL_CORRELATION_VERSION,
    RVV10,
    VV10,
    NonlocalCorrelationSpec,
    UnsupportedNonlocalCorrelation,
    original_nonlocal_correlation,
)

from .dispersion import D3Spec, DispersionCorrectionPrimitive
from .implicit import ImplicitSolveSpec, ImplicitVJPPlan
from .matrix_function import SymmetricMatrixFunctionSpec
from .nonlocal_correlation import NonlocalCorrelationPrimitive
from .spec import (
    METHOD_CATALOG,
    ExactExchangePrimitive,
    MethodIR,
    MethodSpec,
    SemilocalXCPrimitive,
    UnsupportedMethod,
    resolve_method,
)
from .stationary_gradient import (
    IntegralGradientBlock,
    StationaryGradientPlan,
    StationaryMeanField,
)

__all__ = [
    "METHOD_CATALOG",
    "NONLOCAL_CORRELATION_VERSION",
    "RVV10",
    "VV10",
    "D3Spec",
    "DispersionCorrectionPrimitive",
    "ExactExchangePrimitive",
    "ImplicitSolveSpec",
    "ImplicitVJPPlan",
    "IntegralGradientBlock",
    "MethodIR",
    "MethodSpec",
    "NonlocalCorrelationPrimitive",
    "NonlocalCorrelationSpec",
    "SemilocalXCPrimitive",
    "StationaryGradientPlan",
    "StationaryMeanField",
    "SymmetricMatrixFunctionSpec",
    "UnsupportedMethod",
    "UnsupportedNonlocalCorrelation",
    "original_nonlocal_correlation",
    "resolve_method",
]
