"""Canonical method composition above scientific compiler primitives."""

from .basis_binding import (
    BasisBinding,
    r2scan3c_def2_mtzvpp_h_ar,
    validate_basis_snapshot,
)
from .correction import CorrectionProvenance, CorrectionResult
from .dispersion import (
    D3Spec,
    D4Spec,
    DispersionCorrectionPrimitive,
    r2scan3c_d4_eeq,
)
from .gcp import GCPSpec, GeometricCounterpoisePrimitive, r2scan3c_gcp
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
    "BasisBinding",
    "CorrectionProvenance",
    "CorrectionResult",
    "D3Spec",
    "D4Spec",
    "DispersionCorrectionPrimitive",
    "ExactExchangePrimitive",
    "FeatureType",
    "GCPSpec",
    "GeometricCounterpoisePrimitive",
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
    "r2scan3c_d4_eeq",
    "r2scan3c_def2_mtzvpp_h_ar",
    "r2scan3c_gcp",
    "resolve_method",
    "resolve_xtb_method",
    "validate_basis_snapshot",
    "verify_method_ir",
]
