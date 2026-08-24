"""Private ctypes declarations for the QCE C ABI.

Keeping the binding thin ensures Python, Torch, and future JAX callers use the
same native calculation path instead of reimplementing SCF orchestration.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path


ABI_VERSION = 0
STATUS_SUCCESS = 0
STATUS_INVALID_ARGUMENT = 1
STATUS_NOT_IMPLEMENTED = 3
STATUS_SCF_NOT_CONVERGED = 4
METHOD_RHF = 1
METHOD_UHF = 2
METHOD_WB97M_V = 3
METHOD_RCCSD_T = 4
BACKEND_CPU_REFERENCE = 0
BACKEND_CUDA = 1
BACKEND_HYBRID_CUDA = 2
BASIS_CARTESIAN = 0
BASIS_SPHERICAL = 1
BATCH_ENABLE_WARM_STARTS = 1 << 0


class ContextDescriptor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device_id", ctypes.c_int32),
        ("backend", ctypes.c_int),
    ]


class AtomDescriptor(ctypes.Structure):
    _fields_ = [
        ("atomic_number", ctypes.c_int32),
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("z", ctypes.c_double),
    ]


class PrimitiveDescriptor(ctypes.Structure):
    _fields_ = [("exponent", ctypes.c_double), ("coefficient", ctypes.c_double)]


class ShellDescriptor(ctypes.Structure):
    _fields_ = [
        ("atom_index", ctypes.c_uint32),
        ("angular_momentum", ctypes.c_uint32),
        ("primitive_offset", ctypes.c_uint32),
        ("primitive_count", ctypes.c_uint32),
    ]


class SystemDescriptor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("atoms", ctypes.POINTER(AtomDescriptor)),
        ("atom_count", ctypes.c_uint32),
        ("shells", ctypes.POINTER(ShellDescriptor)),
        ("shell_count", ctypes.c_uint32),
        ("primitives", ctypes.POINTER(PrimitiveDescriptor)),
        ("primitive_count", ctypes.c_uint32),
        ("charge", ctypes.c_int32),
        ("multiplicity", ctypes.c_uint32),
        ("basis_representation", ctypes.c_int32),
    ]


class MethodDescriptor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("method", ctypes.c_int),
        ("max_iterations", ctypes.c_uint32),
        ("diis_history", ctypes.c_uint32),
        ("energy_tolerance", ctypes.c_double),
        ("density_tolerance", ctypes.c_double),
        ("screening_tolerance", ctypes.c_double),
    ]


class ResultDescriptor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("energy", ctypes.c_double),
        ("forces", ctypes.POINTER(ctypes.c_double)),
        ("force_count", ctypes.c_uint32),
        ("iterations", ctypes.c_uint32),
        ("energy_change", ctypes.c_double),
        ("density_rms", ctypes.c_double),
        ("converged", ctypes.c_int32),
        ("executed_backend", ctypes.c_int),
    ]


class BatchInputDescriptor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("coordinates", ctypes.POINTER(ctypes.c_double)),
        ("coordinate_count", ctypes.c_uint32),
    ]


class BatchItemResultDescriptor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("status", ctypes.c_int32),
        ("energy", ctypes.c_double),
        ("forces", ctypes.POINTER(ctypes.c_double)),
        ("force_count", ctypes.c_uint32),
        ("iterations", ctypes.c_uint32),
        ("energy_change", ctypes.c_double),
        ("density_rms", ctypes.c_double),
        ("converged", ctypes.c_int32),
        ("executed_backend", ctypes.c_int32),
        ("bucket_id", ctypes.c_uint32),
        ("warm_start_used", ctypes.c_int32),
        ("warm_start_fallback", ctypes.c_int32),
    ]


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    if configured := os.environ.get("QCE_LIBRARY"):
        candidates.append(Path(configured))
    root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [root / "build" / "libqce.so", root / "build" / "libqce.dylib"]
    )
    return candidates


def load_library() -> ctypes.CDLL:
    for candidate in _candidate_paths():
        if candidate.exists():
            library = ctypes.CDLL(str(candidate))
            break
    else:
        raise RuntimeError(
            "QCE native library was not found; set QCE_LIBRARY or build in ./build"
        )

    void_pp = ctypes.POINTER(ctypes.c_void_p)
    library.qce_get_abi_version.restype = ctypes.c_uint32
    library.qce_status_message.argtypes = [ctypes.c_int]
    library.qce_status_message.restype = ctypes.c_char_p
    library.qce_method_available.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int32)]
    library.qce_method_available.restype = ctypes.c_int
    library.qce_context_create.argtypes = [ctypes.POINTER(ContextDescriptor), void_pp]
    library.qce_context_create.restype = ctypes.c_int
    library.qce_context_destroy.argtypes = [ctypes.c_void_p]
    library.qce_system_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(SystemDescriptor),
        void_pp,
    ]
    library.qce_system_create.restype = ctypes.c_int
    library.qce_system_destroy.argtypes = [ctypes.c_void_p]
    library.qce_calculation_prepare.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(MethodDescriptor),
        void_pp,
    ]
    library.qce_calculation_prepare.restype = ctypes.c_int
    library.qce_calculation_destroy.argtypes = [ctypes.c_void_p]
    library.qce_calculation_execute.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ResultDescriptor),
    ]
    library.qce_calculation_execute.restype = ctypes.c_int
    library.qce_batch_prepare.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint32,
        ctypes.POINTER(MethodDescriptor),
        ctypes.c_uint32,
        void_pp,
    ]
    library.qce_batch_prepare.restype = ctypes.c_int
    library.qce_batch_destroy.argtypes = [ctypes.c_void_p]
    library.qce_batch_get_system_count.argtypes = [ctypes.c_void_p]
    library.qce_batch_get_system_count.restype = ctypes.c_uint32
    library.qce_batch_clear_warm_starts.argtypes = [ctypes.c_void_p]
    library.qce_batch_clear_warm_starts.restype = ctypes.c_int
    library.qce_batch_execute.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(BatchInputDescriptor),
        ctypes.c_uint32,
        ctypes.POINTER(BatchItemResultDescriptor),
        ctypes.c_uint32,
    ]
    library.qce_batch_execute.restype = ctypes.c_int
    if library.qce_get_abi_version() != ABI_VERSION:
        raise RuntimeError("QCE Python/native ABI version mismatch")
    return library


def check(library: ctypes.CDLL, status: int) -> None:
    if status != STATUS_SUCCESS:
        message = library.qce_status_message(status).decode("utf-8")
        if status == STATUS_NOT_IMPLEMENTED:
            raise NotImplementedError(f"QCE error {status}: {message}")
        raise RuntimeError(f"QCE error {status}: {message}")
