"""Canonical method composition above scientific compiler primitives."""

from .dispersion import D3Spec, DispersionCorrectionPrimitive
from .implicit import ImplicitSolveSpec, ImplicitVJPPlan
from .matrix_function import SymmetricMatrixFunctionSpec
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
    "D3Spec",
    "DispersionCorrectionPrimitive",
    "ExactExchangePrimitive",
    "ImplicitSolveSpec",
    "ImplicitVJPPlan",
    "IntegralGradientBlock",
    "MethodIR",
    "MethodSpec",
    "SemilocalXCPrimitive",
    "StationaryGradientPlan",
    "StationaryMeanField",
    "SymmetricMatrixFunctionSpec",
    "UnsupportedMethod",
    "resolve_method",
]
