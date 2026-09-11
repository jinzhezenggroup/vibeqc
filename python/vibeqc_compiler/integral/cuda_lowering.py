"""Compatibility facade for the decomposed lowering subsystem.

Implementation ownership is in lowering/. Keep these historical callable
exports while repository callers migrate to their semantic owners.
"""

from .lowering.algebra import (
    _emit_packed_force_geometry_algebra_cuda,
    _emit_triple_pair_matchings,
    _emit_weighted_component_gradient_cuda,
    _packed_force_integral,
)
from .lowering.common import (
    _component_axis_expression,
    _component_names,
    _cuda_array_declaration,
    _emitted_component_names,
    _format_cuda_array,
    _generic_component_decode,
    _generic_component_gradient_setup,
    _generic_component_value_setup,
    _generic_task_component_setup,
    _shell_letter,
    _specialize_dppp_identifiers,
    emit_uncached_primitive_geometry_cuda,
)
from .lowering.dispatch import (
    emit_shell_class_fused_cuda,
)
from .lowering.fock import (
    _emit_shell_class_fock_cuda,
    _emit_shell_class_mixed_fock_cuda,
)
from .lowering.fock_component import (
    _emit_rys_component_lane_fock_consumer_cuda,
)
from .lowering.fock_tiled import (
    _emit_packed_fock_consumer_cuda,
    _emit_subgroup_fock_consumer_cuda,
)
from .lowering.force_packed import (
    _emit_packed_force_consumer_cuda,
    _emit_scalar_thread_force_consumer_cuda,
)
from .lowering.force_resident import (
    _emit_ppps_resident_bra_rys3_force_consumer_cuda,
)
from .lowering.force_rys_component import (
    _emit_rys_component_lane_force_consumer_cuda,
)
from .lowering.force_rys_thread import (
    _emit_rys_thread_force_consumer_cuda,
)
from .lowering.force_rys_uniform import (
    _emit_rys_uniform_warp_force_consumer_cuda,
)
from .lowering.force_subgroup import (
    _emit_subgroup_force_consumer_cuda,
)
from .lowering.legacy import (
    DpppFusedPlan,
    build_dppp_fused_plan,
    dppp_components,
    emit_dppp_fused_cuda,
    emit_ppps_resident_bra_rys3_cuda,
    evaluate_dppp_fused_component,
)
from .lowering.selection import (
    _supports_rys_component_lane_fock,
    supports_component_lane_rys,
)

emit_ppps_1110_resident_bra_cuda = emit_ppps_resident_bra_rys3_cuda

__all__ = [
    "DpppFusedPlan",
    "_component_axis_expression",
    "_component_names",
    "_cuda_array_declaration",
    "_emit_packed_fock_consumer_cuda",
    "_emit_packed_force_consumer_cuda",
    "_emit_packed_force_geometry_algebra_cuda",
    "_emit_ppps_resident_bra_rys3_force_consumer_cuda",
    "_emit_rys_component_lane_fock_consumer_cuda",
    "_emit_rys_component_lane_force_consumer_cuda",
    "_emit_rys_thread_force_consumer_cuda",
    "_emit_rys_uniform_warp_force_consumer_cuda",
    "_emit_scalar_thread_force_consumer_cuda",
    "_emit_shell_class_fock_cuda",
    "_emit_shell_class_mixed_fock_cuda",
    "_emit_subgroup_fock_consumer_cuda",
    "_emit_subgroup_force_consumer_cuda",
    "_emit_triple_pair_matchings",
    "_emit_weighted_component_gradient_cuda",
    "_emitted_component_names",
    "_format_cuda_array",
    "_generic_component_decode",
    "_generic_component_gradient_setup",
    "_generic_component_value_setup",
    "_generic_task_component_setup",
    "_packed_force_integral",
    "_shell_letter",
    "_specialize_dppp_identifiers",
    "_supports_rys_component_lane_fock",
    "build_dppp_fused_plan",
    "dppp_components",
    "emit_dppp_fused_cuda",
    "emit_ppps_1110_resident_bra_cuda",
    "emit_ppps_resident_bra_rys3_cuda",
    "emit_shell_class_fused_cuda",
    "emit_uncached_primitive_geometry_cuda",
    "evaluate_dppp_fused_component",
    "supports_component_lane_rys",
]
