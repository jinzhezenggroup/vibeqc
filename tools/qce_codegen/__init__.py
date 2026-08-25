"""Build-time symbolic code generation for shell-class CUDA kernels."""

from .cache import NvrtcCacheSpec, nvrtc_cache_key
from .dppp_dispatch import (
    DpppFusedPlan,
    build_dppp_fused_plan,
    dppp_components,
    emit_dppp_fused_cuda,
    evaluate_dppp_fused_component,
)
from .fused_schedule import (
    FusedShellPlan,
    build_fused_shell_plan,
    evaluate_fused_shell_component,
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
    ShellClassSpec,
    cartesian_components,
)

__all__ = [
    "DpppComponentKernel",
    "DpppContractionKernel",
    "DpppFusedPlan",
    "DDPS_SPEC",
    "DPDS_SPEC",
    "DPPP_SPEC",
    "FusedShellPlan",
    "NvrtcCacheSpec",
    "PsssKernel",
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
    "cartesian_components",
    "dppp_components",
    "emit_dppp_fused_cuda",
    "evaluate_dppp_fused_component",
    "evaluate_fused_shell_component",
    "nvrtc_cache_key",
]
