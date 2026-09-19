"""Canonical method composition above scientific compiler primitives."""

from .dispersion import (
    D3_RADII_SHA256,
    D3_TABLE_SHA256,
    D3Spec,
    DispersionCorrectionPrimitive,
    pbe0_d3_bj_spec,
    pbe_d3_bj_spec,
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
    "D3_RADII_SHA256",
    "D3_TABLE_SHA256",
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
    "pbe0_d3_bj_spec",
    "pbe_d3_bj_spec",
    "resolve_method",
]
