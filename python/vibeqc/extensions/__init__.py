"""Versioned programmable VibeQC extension surface.

Built-in methods and user-defined compositions share the canonical compiler
spec/IR contracts. Public extension submodules are imported lazily so selecting
one surface does not activate unrelated compiler/codegen modules.
"""

from __future__ import annotations

import importlib
import typing

API_VERSION = 1

_PUBLIC_MODULES = frozenset({"method", "tensor", "xc"})


def __getattr__(name: str) -> typing.Any:
    """Load one public extension surface only when it is requested."""
    if name not in _PUBLIC_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    """Advertise lazy public modules to interactive discovery."""
    return sorted(set(globals()) | _PUBLIC_MODULES)


__all__ = ["API_VERSION", "method", "tensor", "xc"]
