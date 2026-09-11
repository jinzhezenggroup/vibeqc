"""CUDA scalar emitter compatibility surface; arithmetic is backend-neutral."""

from .scalar_c import ScalarCEmitter, format_constant


class CudaEmitter(ScalarCEmitter):
    """Preserve CUDA emitter behavior and exact source while sharing scalar C lowering."""


__all__ = ["CudaEmitter", "format_constant"]
