"""Canonical method composition above scientific compiler primitives."""

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
    "ExactExchangePrimitive",
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
