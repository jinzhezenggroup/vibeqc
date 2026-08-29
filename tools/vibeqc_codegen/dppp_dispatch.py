"""Compatibility adapters for historical DPPP and resident-PPPS APIs.

Generic shell compilation lives in :mod:`vibeqc_codegen.cuda_lowering`. This
module intentionally exposes only the original specialization and resident
worker entry points retained by benchmarks and downstream imports.
"""

from .cuda_lowering import (
    DpppFusedPlan,
    _specialize_dppp_identifiers,
    build_dppp_fused_plan,
    dppp_components,
    emit_dppp_fused_cuda,
    emit_ppps_1110_resident_bra_cuda,
    emit_ppps_resident_bra_rys3_cuda,
    evaluate_dppp_fused_component,
)

__all__ = [
    "DpppFusedPlan",
    "_specialize_dppp_identifiers",
    "build_dppp_fused_plan",
    "dppp_components",
    "emit_dppp_fused_cuda",
    "emit_ppps_1110_resident_bra_cuda",
    "emit_ppps_resident_bra_rys3_cuda",
    "evaluate_dppp_fused_component",
]
