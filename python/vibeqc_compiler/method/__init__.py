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
from .typecheck import (
    BackendCapability,
    FeatureType,
    MethodTypeError,
    TypedMethodIR,
    infer_feature_types,
    verify_method_ir,
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
    "GFN2_PARAMETER_SET",
    "METHOD_CATALOG",
    "XTB_METHOD_CATALOG",
    "BackendCapability",
    "D3Spec",
    "DispersionCorrectionPrimitive",
    "ExactExchangePrimitive",
    "FeatureType",
    "ImplicitSolveSpec",
    "ImplicitVJPPlan",
    "IntegralGradientBlock",
    "MethodIR",
    "MethodSpec",
    "MethodTypeError",
    "SemilocalXCPrimitive",
    "StationaryGradientPlan",
    "StationaryMeanField",
    "SymmetricMatrixFunctionSpec",
    "TypedMethodIR",
    "UnsupportedMethod",
    "UnsupportedXtbMethod",
    "XtbMethodIR",
    "XtbMethodSpec",
    "XtbParameterSet",
    "XtbPrimitive",
    "infer_feature_types",
    "resolve_method",
    "resolve_xtb_method",
    "verify_method_ir",
]
