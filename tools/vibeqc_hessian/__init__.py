"""Hessian helpers and a bounded native CPU RHF integration (issue #180).

NativeRHFState and analytic_hessian form the native tools path. Numerical and
PySCF reference oracles are separate explicitly imported validation modules;
none of them supplies a state or derivative to the native Hessian calculation.
"""

import typing

from .assembly import assemble_frozen_skeleton, validate_hessian_component
from .numerical import (
    forces_to_gradient,
    hessian_difference,
    hessian_symmetry_error,
    hessian_translation_error,
    numerical_hessian,
)
from .response import (
    build_rhf_nuclear_rhs,
    build_stationary_nuclear_rhs,
    metric_density_response_mo,
)
from .weights import two_electron_energy, two_electron_weight, weight_energy

# Loading mathematical helpers does not load the optional native library.
_LAZY = {
    "NativeRHFState": "native",
    "DirectionalRHFResponse": "directional",
    "directional_rhf_response": "directional",
    "DirectionalRKSResponse": "rks_directional",
    "directional_rks_response": "rks_directional",
    "RHFHVPResult": "hvp",
    "RKSHVPResult": "dft",
    "rks_hvp": "dft",
    "rhf_hvp": "hvp",
    "RHFHVPBlockResult": "block",
    "RHFHessianResult": "block",
    "rhf_hvp_many": "block",
    "rhf_hessian": "block",
    "generated_directional_first_order": "first_order",
    "generated_directional_first_order_cuda": "first_order_cuda",
    "analytic_hessian": "analytic",
    "build_reference": "analytic",
    "cphf_relaxation": "analytic",
    "nuclear_closed_form": "analytic",
    "provider_components": "analytic",
    "StationaryNuclearResponse": "perturbation",
    "StationaryNuclearBatchResponse": "perturbation",
    "solve_stationary_nuclear_perturbation": "perturbation",
    "solve_stationary_nuclear_perturbations": "perturbation",
    "StationaryHVPContext": "stationary_executor",
    "StationaryHVPContributor": "stationary_executor",
    "StationaryPerturbationProvider": "stationary_executor",
    "StationaryResponseDriver": "stationary_executor",
    "StationarySecondOrderExecutor": "stationary_executor",
    "StationarySecondOrderResult": "stationary_executor",
}

__all__ = [
    "DirectionalRHFResponse",
    "DirectionalRKSResponse",
    "NativeRHFState",
    "RHFHVPBlockResult",
    "RHFHVPResult",
    "RKSHVPResult",
    "RHFHessianResult",
    "StationaryHVPContext",
    "StationaryHVPContributor",
    "StationaryNuclearBatchResponse",
    "StationaryNuclearResponse",
    "StationaryPerturbationProvider",
    "StationaryResponseDriver",
    "StationarySecondOrderExecutor",
    "StationarySecondOrderResult",
    "analytic_hessian",
    "assemble_frozen_skeleton",
    "build_reference",
    "build_rhf_nuclear_rhs",
    "build_stationary_nuclear_rhs",
    "cphf_relaxation",
    "directional_rhf_response",
    "directional_rks_response",
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
    "rhf_hessian",
    "rhf_hvp",
    "rhf_hvp_many",
    "rks_hvp",
    "solve_stationary_nuclear_perturbation",
    "solve_stationary_nuclear_perturbations",
    "two_electron_energy",
    "two_electron_weight",
    "validate_hessian_component",
    "weight_energy",
]


def __getattr__(name: typing.Any) -> typing.Any:
    module = _LAZY.get(name)
    if module is not None:
        import importlib

        return getattr(importlib.import_module(f".{module}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
