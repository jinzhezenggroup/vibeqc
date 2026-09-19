"""Canonical method composition above scientific compiler primitives."""

from .dispersion import (
    D3Spec,
    D4Spec,
    DispersionCorrectionPrimitive,
    r2scan3c_d4_eeq,
)
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
    "D4Spec",
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
    "r2scan3c_d4_eeq",
    "resolve_method",
]
