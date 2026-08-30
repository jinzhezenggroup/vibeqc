"""Compatibility specialization for the original ``dppp`` CUDA pilot."""

from . import cuda_lowering as _implementation

DpppFusedPlan = _implementation.DpppFusedPlan
_specialize_dppp_identifiers = _implementation._specialize_dppp_identifiers
build_dppp_fused_plan = _implementation.build_dppp_fused_plan
dppp_components = _implementation.dppp_components
emit_dppp_fused_cuda = _implementation.emit_dppp_fused_cuda
evaluate_dppp_fused_component = _implementation.evaluate_dppp_fused_component

__all__ = [
    "DpppFusedPlan",
    "_specialize_dppp_identifiers",
    "build_dppp_fused_plan",
    "dppp_components",
    "emit_dppp_fused_cuda",
    "evaluate_dppp_fused_component",
]
