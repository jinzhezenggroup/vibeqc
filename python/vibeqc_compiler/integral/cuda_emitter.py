"""Generic CUDA shell lowering surface.

Production and tuning code depend on this backend-named interface rather than
on historical shell-specific compatibility adapters.
"""

from . import cuda_lowering as _implementation

_emitted_component_names = _implementation._emitted_component_names
_generic_task_component_setup = _implementation._generic_task_component_setup
emit_shell_class_fused_cuda = _implementation.emit_shell_class_fused_cuda
emit_uncached_primitive_geometry_cuda = (
    _implementation.emit_uncached_primitive_geometry_cuda
)

__all__ = [
    "_emitted_component_names",
    "_generic_task_component_setup",
    "emit_shell_class_fused_cuda",
    "emit_uncached_primitive_geometry_cuda",
]
