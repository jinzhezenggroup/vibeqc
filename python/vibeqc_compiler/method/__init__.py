"""Canonical method composition above scientific compiler primitives.

The public symbols are loaded lazily so build-time-safe method code generators
can import lightweight contracts without importing NumPy/TensorIR.
"""

import typing
from importlib import import_module

_EXPORTS = {
    "D3_RADII_SHA256": ".dispersion",
    "D3_TABLE_SHA256": ".dispersion",
    "pbe0_d3_bj_spec": ".dispersion",
    "pbe_d3_bj_spec": ".dispersion",
    "BasisBinding": ".basis_binding",
    "CorrectionProvenance": ".correction",
    "CorrectionResult": ".correction",
    "GCPSpec": ".gcp",
    "GeometricCounterpoisePrimitive": ".gcp",
    "RangeSeparatedExchangePrimitive": ".spec",
    "r2scan3c_def2_mtzvpp_h_ar": ".basis_binding",
    "r2scan3c_gcp": ".gcp",
    "validate_basis_snapshot": ".basis_binding",
    "BackendCapability": ".typecheck",
    "D3Spec": ".dispersion",
    "D4Spec": ".dispersion",
    "DensityFittingRHFResponsePlan": ".df_hf_response",
    "DispersionCorrectionPrimitive": ".dispersion",
    "ExactExchangePrimitive": ".spec",
    "FeatureType": ".typecheck",
    "GFN2_PARAMETER_SET": ".xtb",
    "ImplicitSolveSpec": ".implicit",
    "ImplicitVJPPlan": ".implicit",
    "IntegralGradientBlock": ".stationary_gradient",
    "METHOD_CATALOG": ".spec",
    "MethodIR": ".spec",
    "MethodSpec": ".spec",
    "MethodTypeError": ".typecheck",
    "NONLOCAL_CORRELATION_VERSION": "vibeqc_compiler.common.nonlocal_correlation",
    "NonlocalCorrelationPrimitive": ".nonlocal_correlation",
    "NonlocalCorrelationSpec": "vibeqc_compiler.common.nonlocal_correlation",
    "RVV10": "vibeqc_compiler.common.nonlocal_correlation",
    "SemilocalXCPrimitive": ".spec",
    "StationaryGradientPlan": ".stationary_gradient",
    "StationaryMeanField": ".stationary_gradient",
    "SymmetricMatrixFunctionSpec": ".matrix_function",
    "TypedMethodIR": ".typecheck",
    "UnsupportedMethod": ".spec",
    "UnsupportedNonlocalCorrelation": "vibeqc_compiler.common.nonlocal_correlation",
    "UnsupportedXtbMethod": ".xtb",
    "VV10": "vibeqc_compiler.common.nonlocal_correlation",
    "XTB_METHOD_CATALOG": ".xtb",
    "XtbMethodIR": ".xtb",
    "XtbMethodSpec": ".xtb",
    "XtbParameterSet": ".xtb",
    "XtbPrimitive": ".xtb",
    "infer_feature_types": ".typecheck",
    "original_nonlocal_correlation": "vibeqc_compiler.common.nonlocal_correlation",
    "r2scan3c_d4_eeq": ".dispersion",
    "resolve_method": ".spec",
    "resolve_xtb_method": ".xtb",
    "verify_method_ir": ".typecheck",
}

__all__ = list(_EXPORTS)


def __getattr__(name: typing.Any) -> typing.Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
