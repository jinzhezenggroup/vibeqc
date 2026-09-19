"""Canonical method composition above scientific compiler primitives."""

from vibeqc_compiler.common.nonlocal_correlation import (
    NONLOCAL_CORRELATION_VERSION,
    RVV10,
    VV10,
    NonlocalCorrelationSpec,
    UnsupportedNonlocalCorrelation,
    original_nonlocal_correlation,
)

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
from .nonlocal_correlation import NonlocalCorrelationPrimitive
from .spec import (
    METHOD_CATALOG,
    ExactExchangePrimitive,
    MethodIR,
    MethodSpec,
    RangeSeparatedExchangePrimitive,
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
    "NONLOCAL_CORRELATION_VERSION",
    "RVV10",
    "VV10",
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
    "NonlocalCorrelationPrimitive",
    "NonlocalCorrelationSpec",
    "RangeSeparatedExchangePrimitive",
    "SemilocalXCPrimitive",
    "StationaryGradientPlan",
    "StationaryMeanField",
    "SymmetricMatrixFunctionSpec",
    "TypedMethodIR",
    "UnsupportedMethod",
    "UnsupportedNonlocalCorrelation",
    "UnsupportedXtbMethod",
    "XtbMethodIR",
    "XtbMethodSpec",
    "XtbParameterSet",
    "XtbPrimitive",
    "infer_feature_types",
    "original_nonlocal_correlation",
    "r2scan3c_d4_eeq",
    "r2scan3c_def2_mtzvpp_h_ar",
    "r2scan3c_gcp",
    "resolve_method",
    "resolve_xtb_method",
    "validate_basis_snapshot",
    "verify_method_ir",
]
