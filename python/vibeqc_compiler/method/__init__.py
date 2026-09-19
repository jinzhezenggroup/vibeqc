"""Canonical method composition above scientific compiler primitives.

The public symbols are loaded lazily so build-time-safe method code generators
can import lightweight contracts without importing NumPy/TensorIR.
"""

from importlib import import_module

_EXPORTS = {
    "BackendCapability": ".typecheck",
    "D3Spec": ".dispersion",
    "DensityFittingRHFResponsePlan": ".df_hf_response",
    "DispersionCorrectionPrimitive": ".dispersion",
    "ExactExchangePrimitive": ".spec",
    "FeatureType": ".typecheck",
    "ImplicitSolveSpec": ".implicit",
    "ImplicitVJPPlan": ".implicit",
    "IntegralGradientBlock": ".stationary_gradient",
    "METHOD_CATALOG": ".spec",
    "MethodIR": ".spec",
    "MethodSpec": ".spec",
    "MethodTypeError": ".typecheck",
    "SemilocalXCPrimitive": ".spec",
    "StationaryGradientPlan": ".stationary_gradient",
    "StationaryMeanField": ".stationary_gradient",
    "SymmetricMatrixFunctionSpec": ".matrix_function",
    "TypedMethodIR": ".typecheck",
    "UnsupportedMethod": ".spec",
    "infer_feature_types": ".typecheck",
    "resolve_method": ".spec",
    "verify_method_ir": ".typecheck",
    "GFN2_PARAMETER_SET": ".xtb",
    "XTB_METHOD_CATALOG": ".xtb",
    "UnsupportedXtbMethod": ".xtb",
    "XtbMethodIR": ".xtb",
    "XtbMethodSpec": ".xtb",
    "XtbParameterSet": ".xtb",
    "XtbPrimitive": ".xtb",
    "resolve_xtb_method": ".xtb",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
