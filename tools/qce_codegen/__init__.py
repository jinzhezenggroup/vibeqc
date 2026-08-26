"""Build-time symbolic code generation for shell-class CUDA kernels."""

from .cache import NvrtcCacheSpec, nvrtc_cache_key
from .dppp_dispatch import (
    DpppFusedPlan,
    build_dppp_fused_plan,
    dppp_components,
    emit_dppp_fused_cuda,
    emit_shell_class_fused_cuda,
    evaluate_dppp_fused_component,
)
from .fused_schedule import (
    FusedShellPlan,
    build_fused_shell_plan,
    evaluate_fused_shell_component,
)
from .low_order_force import (
    PSPS_BLOCK_THREADS,
    emit_psps_weighted_force_cuda,
)
from .shell_class import (
    DpppComponentKernel,
    DpppContractionKernel,
    PsssKernel,
    ShellClassComponentKernel,
    ShellClassContractionKernel,
    build_dppp_component_kernel,
    build_dppp_contraction_kernel,
    build_psss_kernel,
    build_shell_class_component_kernel,
    build_shell_class_contraction_kernel,
)
from .shell_spec import (
    DDPS_SPEC,
    DPDS_SPEC,
    DPPP_SPEC,
    FUSED_SHELL_SPEC_BY_NAME,
    FUSED_SHELL_SPECS,
    PSPS_SPEC,
    ShellClassSpec,
    canonical_shell_angular,
    cartesian_components,
    enumerate_fused_shell_specs,
    shell_class_name,
    shell_pair_class,
)

__all__ = [
    "DDPS_SPEC",
    "DPDS_SPEC",
    "DPPP_SPEC",
    "FUSED_SHELL_SPECS",
    "FUSED_SHELL_SPEC_BY_NAME",
    "PSPS_SPEC",
    "DpppComponentKernel",
    "DpppContractionKernel",
    "DpppFusedPlan",
    "FusedShellPlan",
    "NvrtcCacheSpec",
    "PsssKernel",
    "PSPS_BLOCK_THREADS",
    "ShellClassComponentKernel",
    "ShellClassContractionKernel",
    "ShellClassSpec",
    "build_dppp_component_kernel",
    "build_dppp_contraction_kernel",
    "build_dppp_fused_plan",
    "build_fused_shell_plan",
    "build_psss_kernel",
    "build_shell_class_component_kernel",
    "build_shell_class_contraction_kernel",
    "canonical_shell_angular",
    "cartesian_components",
    "dppp_components",
    "emit_dppp_fused_cuda",
    "emit_psps_weighted_force_cuda",
    "emit_shell_class_fused_cuda",
    "enumerate_fused_shell_specs",
    "evaluate_dppp_fused_component",
    "evaluate_fused_shell_component",
    "nvrtc_cache_key",
    "shell_class_name",
    "shell_pair_class",
]
