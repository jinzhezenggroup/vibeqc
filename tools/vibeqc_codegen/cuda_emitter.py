"""Generic CUDA shell lowering surface.

The implementation is retained in ``dppp_dispatch`` while it is decomposed,
but production and tuning code depend on this backend-named interface rather
than on the historical pilot shell class.
"""

from . import dppp_dispatch as _implementation

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
