"""Canonical method composition above scientific compiler primitives."""

from .dispersion import D3Spec, DispersionCorrectionPrimitive
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

__all__ = [
    "METHOD_CATALOG",
    "D3Spec",
    "DispersionCorrectionPrimitive",
    "ExactExchangePrimitive",
    "MethodIR",
    "MethodSpec",
    "SemilocalXCPrimitive",
    "SymmetricMatrixFunctionSpec",
    "UnsupportedMethod",
    "resolve_method",
]
