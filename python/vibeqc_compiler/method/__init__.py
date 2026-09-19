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
from .typecheck import (
    BackendCapability,
    FeatureType,
    MethodTypeError,
    TypedMethodIR,
    infer_feature_types,
    verify_method_ir,
)

__all__ = [
    "METHOD_CATALOG",
    "NONLOCAL_CORRELATION_VERSION",
    "RVV10",
    "VV10",
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
    "NonlocalCorrelationPrimitive",
    "NonlocalCorrelationSpec",
    "SemilocalXCPrimitive",
    "StationaryGradientPlan",
    "StationaryMeanField",
    "SymmetricMatrixFunctionSpec",
    "TypedMethodIR",
    "UnsupportedMethod",
    "UnsupportedNonlocalCorrelation",
    "infer_feature_types",
    "original_nonlocal_correlation",
    "resolve_method",
    "verify_method_ir",
]
