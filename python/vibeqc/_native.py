"""Private ctypes declarations for the VIBEQC C ABI.

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
STATUS_NOT_CONVERGED = 4
STATUS_SCF_NOT_CONVERGED = 4
METHOD_RHF = 1
METHOD_UHF = 2
METHOD_WB97M_V = 3
METHOD_RCCSD_T = 4
METHOD_FAMILY_HARTREE_FOCK = 1
METHOD_FAMILY_DENSITY_FUNCTIONAL = 2
METHOD_FAMILY_COUPLED_CLUSTER = 3
PROPERTY_ENERGY = 1 << 0
PROPERTY_FORCES = 1 << 1
BACKEND_CPU_REFERENCE = 0
BACKEND_CUDA = 1
BACKEND_HYBRID_CUDA = 2
BASIS_CARTESIAN = 0
BASIS_SPHERICAL = 1
BATCH_ENABLE_WARM_STARTS = 1 << 0
BATCH_ENABLE_SHELL_CLASS_PROFILING = 1 << 1
DIRECT_SHELL_CLASS_COUNT = 55
PPPS_PROFILE_BLOCK_SIZE_COUNT = 4
PPPS_PROFILE_ORIENTATION_COUNT = 2
PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT = 65
EIGENSOLVER_FAMILY_NAMES = (
    "small_native",
    "jacobi_batched",
    "xsyev_batched",
    "graph_native",
)
EIGENSOLVER_SELECTION_SOURCE_NAMES = (
    "dimension_policy",
    "exact_probe",
    "exact_probe_fallback",
)
XSYEV_ELIGIBILITY_REASON_NAMES = (
    "eligible",
    "zero_dimension",
    "invalid_leading_dimension",
    "documented_dimension_limit",
    "solver_batch_limit",
    "documented_product_limit",
)
XSYEV_GRAPH_PROBE_STAGE_NAMES = (
    "none",
    "api_eligibility",
    "select_device",
    "device_identity",
    "create_stream",
    "create_solver",
    "create_parameters",
    "allocate_probe_data",
    "query_workspace",
    "insufficient_device_memory",
    "allocate_workspace",
    "ordinary_execution",
    "ordinary_validation",
    "begin_capture",
    "capture_provider",
    "end_capture",
    "instantiate_device_launch_graph",
    "upload_graph",
    "host_graph_replay",
    "host_graph_validation",
    "device_tail_replay",
    "device_tail_validation",
)


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


class MethodCapabilitiesDescriptor(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("method", ctypes.c_int),
        ("family", ctypes.c_int),
        ("supported_properties", ctypes.c_uint32),
        ("available", ctypes.c_int32),
        ("supports_batch", ctypes.c_int32),
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


class ShellClassProfileEntry(ctypes.Structure):
    _fields_ = [
        ("shell_quartets", ctypes.c_uint64),
        ("tiles", ctypes.c_uint64),
        ("ao_quartets", ctypes.c_uint64),
        ("primitive_quartets", ctypes.c_uint64),
    ]


class PppsQueueProfile(ctypes.Structure):
    _fields_ = [
        ("descriptor_slots", ctypes.c_uint64),
        ("non_empty_descriptors", ctypes.c_uint64),
        ("empty_descriptors", ctypes.c_uint64),
        ("tasks", ctypes.c_uint64),
        ("primitive_work", ctypes.c_uint64),
        ("ket_count_min", ctypes.c_uint32),
        ("ket_count_median", ctypes.c_uint32),
        ("ket_count_p90", ctypes.c_uint32),
        ("ket_count_p99", ctypes.c_uint32),
        ("ket_count_max", ctypes.c_uint32),
        (
            "lane_efficiency",
            ctypes.c_double * PPPS_PROFILE_BLOCK_SIZE_COUNT,
        ),
        ("primitive_warp_efficiency", ctypes.c_double),
        (
            "task_tail_imbalance",
            ctypes.c_double * PPPS_PROFILE_BLOCK_SIZE_COUNT,
        ),
        (
            "primitive_tail_imbalance",
            ctypes.c_double * PPPS_PROFILE_BLOCK_SIZE_COUNT,
        ),
        (
            "orientation_tasks",
            ctypes.c_uint64 * PPPS_PROFILE_ORIENTATION_COUNT,
        ),
        (
            "orientation_primitive_work",
            ctypes.c_uint64 * PPPS_PROFILE_ORIENTATION_COUNT,
        ),
        (
            "bra_primitive_tasks",
            ctypes.c_uint64 * PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT,
        ),
        (
            "bra_primitive_work",
            ctypes.c_uint64 * PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT,
        ),
        (
            "ket_primitive_tasks",
            ctypes.c_uint64 * PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT,
        ),
        (
            "ket_primitive_work",
            ctypes.c_uint64 * PPPS_PROFILE_PRIMITIVE_PAIR_BUCKET_COUNT,
        ),
    ]


class EigensolverDiagnostic(ctypes.Structure):
    _fields_ = [
        ("bucket_id", ctypes.c_uint32),
        ("ordinary_family", ctypes.c_int32),
        ("graph_family", ctypes.c_int32),
        ("selection_source", ctypes.c_int32),
        ("matrix_dimension", ctypes.c_uint64),
        ("physical_system_count", ctypes.c_uint64),
        ("solver_batch_count", ctypes.c_uint64),
        ("api_eligible", ctypes.c_int32),
        ("api_reason", ctypes.c_int32),
        ("matrix_batch_product", ctypes.c_uint64),
        ("probe_failure_stage", ctypes.c_int32),
        ("device_workspace_bytes", ctypes.c_uint64),
        ("host_workspace_bytes", ctypes.c_uint64),
        ("available_device_bytes", ctypes.c_uint64),
        ("device_id", ctypes.c_int32),
        ("device_uuid", ctypes.c_uint8 * 16),
        ("device_name", ctypes.c_char * 256),
        ("compute_capability_major", ctypes.c_int32),
        ("compute_capability_minor", ctypes.c_int32),
        ("cuda_runtime_version", ctypes.c_int32),
        ("cuda_driver_version", ctypes.c_int32),
        ("cusolver_version", ctypes.c_int32),
        ("cuda_error", ctypes.c_int32),
        ("cusolver_error", ctypes.c_int32),
        ("ordinary_execution_passed", ctypes.c_int32),
        ("graph_capture_passed", ctypes.c_int32),
        ("host_graph_replay_passed", ctypes.c_int32),
        ("device_tail_replay_passed", ctypes.c_int32),
        ("graph_eligible", ctypes.c_int32),
        ("maximum_eigenvalue_error", ctypes.c_double),
        ("maximum_residual", ctypes.c_double),
        ("maximum_orthogonality_error", ctypes.c_double),
    ]


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    if configured := os.environ.get("VIBEQC_LIBRARY"):
        candidates.append(Path(configured))
    root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [root / "build" / "libvibeqc.so", root / "build" / "libvibeqc.dylib"]
    )
    return candidates


def load_library() -> ctypes.CDLL:
    for candidate in _candidate_paths():
        if candidate.exists():
            library = ctypes.CDLL(str(candidate))
            break
    else:
        raise RuntimeError(
            "VIBEQC native library was not found; set VIBEQC_LIBRARY or build in ./build"
        )

    void_pp = ctypes.POINTER(ctypes.c_void_p)
    library.vibeqc_get_abi_version.restype = ctypes.c_uint32
    library.vibeqc_status_message.argtypes = [ctypes.c_int]
    library.vibeqc_status_message.restype = ctypes.c_char_p
    library.vibeqc_method_available.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int32)]
    library.vibeqc_method_available.restype = ctypes.c_int
    library.vibeqc_method_get_capabilities.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(MethodCapabilitiesDescriptor),
    ]
    library.vibeqc_method_get_capabilities.restype = ctypes.c_int
    library.vibeqc_context_create.argtypes = [ctypes.POINTER(ContextDescriptor), void_pp]
    library.vibeqc_context_create.restype = ctypes.c_int
    library.vibeqc_context_destroy.argtypes = [ctypes.c_void_p]
    library.vibeqc_system_create.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(SystemDescriptor),
        void_pp,
    ]
    library.vibeqc_system_create.restype = ctypes.c_int
    library.vibeqc_system_destroy.argtypes = [ctypes.c_void_p]
    library.vibeqc_calculation_prepare.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(MethodDescriptor),
        void_pp,
    ]
    library.vibeqc_calculation_prepare.restype = ctypes.c_int
    library.vibeqc_calculation_destroy.argtypes = [ctypes.c_void_p]
    library.vibeqc_calculation_execute.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ResultDescriptor),
    ]
    library.vibeqc_calculation_execute.restype = ctypes.c_int
    library.vibeqc_batch_prepare.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint32,
        ctypes.POINTER(MethodDescriptor),
        ctypes.c_uint32,
        void_pp,
    ]
    library.vibeqc_batch_prepare.restype = ctypes.c_int
    library.vibeqc_batch_destroy.argtypes = [ctypes.c_void_p]
    library.vibeqc_batch_get_system_count.argtypes = [ctypes.c_void_p]
    library.vibeqc_batch_get_system_count.restype = ctypes.c_uint32
    library.vibeqc_batch_get_last_shell_class_profile.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ShellClassProfileEntry),
        ctypes.c_uint32,
    ]
    library.vibeqc_batch_get_last_shell_class_profile.restype = ctypes.c_int
    library.vibeqc_batch_get_last_ppps_queue_profile.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PppsQueueProfile),
    ]
    library.vibeqc_batch_get_last_ppps_queue_profile.restype = ctypes.c_int
    library.vibeqc_batch_get_last_eigensolver_diagnostics.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(EigensolverDiagnostic),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    library.vibeqc_batch_get_last_eigensolver_diagnostics.restype = ctypes.c_int
    library.vibeqc_batch_clear_warm_starts.argtypes = [ctypes.c_void_p]
    library.vibeqc_batch_clear_warm_starts.restype = ctypes.c_int
    library.vibeqc_batch_set_warm_start_updates.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
    ]
    library.vibeqc_batch_set_warm_start_updates.restype = ctypes.c_int
    library.vibeqc_batch_execute.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(BatchInputDescriptor),
        ctypes.c_uint32,
        ctypes.POINTER(BatchItemResultDescriptor),
        ctypes.c_uint32,
    ]
    library.vibeqc_batch_execute.restype = ctypes.c_int
    if library.vibeqc_get_abi_version() != ABI_VERSION:
        raise RuntimeError("VIBEQC Python/native ABI version mismatch")
    return library


def check(library: ctypes.CDLL, status: int) -> None:
    if status != STATUS_SUCCESS:
        message = library.vibeqc_status_message(status).decode("utf-8")
        if status == STATUS_NOT_IMPLEMENTED:
            raise NotImplementedError(f"VIBEQC error {status}: {message}")
        raise RuntimeError(f"VIBEQC error {status}: {message}")
