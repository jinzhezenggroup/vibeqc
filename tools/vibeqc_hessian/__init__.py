"""Hessian helpers and a bounded native CPU RHF integration (issue #180).

NativeRHFState and analytic_hessian form the native tools path. Numerical and
PySCF reference oracles are separate explicitly imported validation modules;
none of them supplies a state or derivative to the native Hessian calculation.
"""

from .assembly import assemble_frozen_skeleton, validate_hessian_component
from .numerical import (
    forces_to_gradient,
    hessian_difference,
    hessian_symmetry_error,
    hessian_translation_error,
    numerical_hessian,
)
from .response import build_rhf_nuclear_rhs, metric_density_response_mo
from .weights import two_electron_energy, two_electron_weight, weight_energy

# Loading mathematical helpers does not load the optional native library.
_LAZY = {
    "NativeRHFState": "native",
    "DirectionalRHFResponse": "directional",
    "directional_rhf_response": "directional",
    "generated_directional_first_order": "first_order",
    "generated_directional_first_order_cuda": "first_order_cuda",
    "analytic_hessian": "analytic",
    "build_reference": "analytic",
    "cphf_relaxation": "analytic",
    "nuclear_closed_form": "analytic",
    "provider_components": "analytic",
}

__all__ = [
    "DirectionalRHFResponse",
    "NativeRHFState",
    "analytic_hessian",
    "assemble_frozen_skeleton",
    "build_reference",
    "build_rhf_nuclear_rhs",
    "cphf_relaxation",
    "directional_rhf_response",
    "forces_to_gradient",
    "generated_directional_first_order",
    "generated_directional_first_order_cuda",
    "hessian_difference",
    "hessian_symmetry_error",
    "hessian_translation_error",
    "metric_density_response_mo",
    "nuclear_closed_form",
    "numerical_hessian",
    "provider_components",
    "two_electron_energy",
    "two_electron_weight",
    "validate_hessian_component",
    "weight_energy",
]


def __getattr__(name):
    module = _LAZY.get(name)
    if module is not None:
        import importlib

        return getattr(importlib.import_module(f".{module}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
