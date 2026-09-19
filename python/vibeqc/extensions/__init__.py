"""Versioned programmable VibeQC extension surface.

Built-in methods and user-defined compositions share the canonical compiler
spec/IR contracts. Importing this optional surface is explicit so ordinary
`import vibeqc` does not eagerly import TensorIR development modules.
"""

from . import method, tensor, xc

API_VERSION = 1

__all__ = ["API_VERSION", "method", "tensor", "xc"]
