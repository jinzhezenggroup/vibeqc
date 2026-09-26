"""Execution-result translation independent of model resolution.

Only the narrow native descriptor/diagnostic boundary lives here.  The
single-system and batch facades can therefore test status, backend, force, and
correlation translation without constructing a model or a native execution
context.
"""

from __future__ import annotations

import ctypes
import typing

import numpy as np

from . import _native
from ._api_types import CorrelationResult


def read_correlation_result(
    library: ctypes.CDLL,
    owner: ctypes.c_void_p,
    *,
    index: int | None = None,
    context: ctypes.c_void_p | None = None,
) -> CorrelationResult | None:
    """Decode the optional post-HF diagnostic ABI into a stable record."""

    name = (
        "vibeqc_calculation_get_correlation_diagnostic"
        if index is None
        else "vibeqc_batch_get_correlation_diagnostic"
    )
    getter = getattr(library, name)
    diag = _native.CorrelationDiagnostic()
    diag.struct_size = ctypes.sizeof(diag)
    diag.abi_version = _native.ABI_VERSION
    args = (
        (owner, ctypes.byref(diag))
        if index is None
        else (owner, index, ctypes.byref(diag))
    )
    status = getter(*args)
    if status == _native.STATUS_NOT_IMPLEMENTED:
        return None
    _native.check(library, status, context=context)
    values = {
        field_name: getattr(diag, field_name)
        for field_name, _ in diag._fields_
        if field_name not in ("struct_size", "abi_version")
    }
    values["mo_host_staging"] = bool(values["mo_host_staging"])
    for field_name in (
        "equation_hash",
        "response_operator_hash",
        "ccsd_replay_equation_hash",
        "ccsd_t_equation_hash",
    ):
        values[field_name] = values[field_name].decode("ascii")
    return CorrelationResult(**values)


def backend_name(executed_backend: int) -> str:
    """Translate the native backend enum without consulting model metadata."""

    return {
        _native.BACKEND_CPU_REFERENCE: "cpu_reference",
        _native.BACKEND_CUDA: "cuda",
        _native.BACKEND_HYBRID_CUDA: "hybrid_cuda",
    }.get(executed_backend, "unknown")


def copy_force_array(force_storage: typing.Any, atom_count: int) -> np.ndarray | None:
    """Copy a native ``3*N`` force buffer into the public ``(N, 3)`` shape."""

    if force_storage is None:
        return None
    return np.ctypeslib.as_array(force_storage).copy().reshape(atom_count, 3)


def status_message(library: ctypes.CDLL, status: int) -> str:
    """Decode a native status without requiring a calculator or model."""

    return library.vibeqc_status_message(status).decode("utf-8")


__all__ = [
    "backend_name",
    "copy_force_array",
    "read_correlation_result",
    "status_message",
]
