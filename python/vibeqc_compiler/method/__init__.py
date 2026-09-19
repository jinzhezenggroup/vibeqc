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
from .xtb import (
    GFN2_PARAMETER_SET,
    XTB_METHOD_CATALOG,
    UnsupportedXtbMethod,
    XtbMethodIR,
    XtbMethodSpec,
    XtbParameterSet,
    XtbPrimitive,
    resolve_xtb_method,
)

__all__ = [
    "METHOD_CATALOG",
    "XTB_METHOD_CATALOG",
    "D3Spec",
    "DispersionCorrectionPrimitive",
    "ExactExchangePrimitive",
    "GFN2_PARAMETER_SET",
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
    "UnsupportedXtbMethod",
    "XtbMethodIR",
    "XtbMethodSpec",
    "XtbParameterSet",
    "XtbPrimitive",
    "resolve_method",
    "resolve_xtb_method",
]
