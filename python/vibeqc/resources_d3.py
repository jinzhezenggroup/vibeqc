"""ResourcePlan adapter for the production two-body D3(BJ) owner.

The formulas mirror src/dft/dispersion/d3_runtime.cpp exactly.  Planning is
metadata-only: no native context, correction batch, coordinates array, or GPU
allocation is created.
"""

from __future__ import annotations

import ctypes
import json
import typing

from vibeqc_compiler.method import D3Spec, DispersionCorrectionPrimitive, MethodIR

from .resources import (
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
    checked_bytes,
)

_D3_MAXIMUM_ATOMS_PER_SYSTEM = 4096
_D3_ELEMENT_COUNT = 86
_D3_PAIR_COUNT = 86 * 87 // 2
_D3_REFERENCE_CN_COUNT = 237
_D3_REFERENCE_C6_COUNT = 28455


class _ElementData(ctypes.Structure):
    _fields_ = [
        ("reference_offset", ctypes.c_uint32),
        ("reference_count", ctypes.c_uint8),
        ("covalent_radius", ctypes.c_double),
        ("r4r2", ctypes.c_double),
    ]


class _PairData(ctypes.Structure):
    _fields_ = [
        ("c6_offset", ctypes.c_uint32),
        ("first_reference_count", ctypes.c_uint8),
        ("second_reference_count", ctypes.c_uint8),
        ("vdw_radius", ctypes.c_double),
    ]


_D3_TABLE_BYTES = (
    _D3_ELEMENT_COUNT * ctypes.sizeof(_ElementData)
    + _D3_PAIR_COUNT * ctypes.sizeof(_PairData)
    + byte_product(_D3_REFERENCE_CN_COUNT, 8)
    + byte_product(_D3_REFERENCE_C6_COUNT, 8)
)


def _d3_spec(graph: MethodIR) -> D3Spec:
    corrections = tuple(
        node
        for node in graph.primitives
        if isinstance(node, DispersionCorrectionPrimitive)
    )
    if len(corrections) != 1 or not isinstance(corrections[0].specification, D3Spec):
        raise NotImplementedError(
            "global ResourcePlan currently supports exactly one production D3(BJ) correction"
        )
    return corrections[0].specification


def _normalized_atomic_numbers(systems: typing.Any) -> tuple[tuple[int, ...], ...]:
    normalized = tuple(tuple(system) for system in systems)
    if not normalized or any(not system for system in normalized):
        raise ValueError("D3 resource planning requires nonempty systems")
    for system in normalized:
        for atomic_number in system:
            if type(atomic_number) is not int:
                raise TypeError("D3 atomic numbers must be integers")
    return normalized


def d3_resource_request(
    systems: typing.Any,
    *,
    method: MethodIR,
    backend: str = "cpu",
    device_id: int = 0,
    maximum_bytes: int = 256 * 1024 * 1024,
    name: str = "d3",
    first_phase: int = 0,
    last_phase: int = 0,
) -> ResourceRequest:
    """Describe the retained production D3(BJ) batch owner without allocating it."""
    if not isinstance(method, MethodIR):
        raise TypeError("D3 resource planning requires a MethodIR")
    spec = _d3_spec(method)
    if backend not in {"cpu", "cuda"}:
        raise ValueError("D3 backend must be 'cpu' or 'cuda'")
    checked_bytes(device_id, "device ordinal")
    checked_bytes(first_phase, "first phase")
    checked_bytes(last_phase, "last phase")
    if last_phase < first_phase:
        raise ValueError("D3 resource lifetime ends before its first use")
    if type(maximum_bytes) is not int or not 0 < maximum_bytes < 2**64:
        raise ValueError("D3 maximum_bytes must be a positive uint64 integer")

    atomic_numbers = _normalized_atomic_numbers(systems)
    counts = tuple(len(system) for system in atomic_numbers)
    total_atoms = checked_bytes(sum(counts), "D3 total atom count")
    maximum_atoms = max(counts)

    controls = {
        "correction_identity": spec.identity,
        "device_id": device_id,
        "inventory_version": 1,
        "maximum_bytes": maximum_bytes,
        "method_ir_identity": method.identity,
        "schedule": "retained-native-d3-bj",
    }
    identity = ResourceIdentity(
        method.identifier,
        "native-d3-bj-v1",
        backend,
        "fp64",
        json.dumps(
            {
                "atomic_numbers": atomic_numbers,
                "counts": counts,
            }
        ),
        ("energy", "forces"),
        json.dumps(controls, sort_keys=True),
    )
    exclusions = (
        "caller-owned inputs and Python objects",
        "allocator bookkeeping, arenas and page rounding",
    )
    if backend == "cuda":
        exclusions += ("CUDA driver/context/modules and pool/page retention",)

    unsupported = None
    if any(z < 1 or z > 86 for system in atomic_numbers for z in system):
        unsupported = "production D3(BJ) supports atomic numbers 1 through 86"
    elif maximum_atoms > _D3_MAXIMUM_ATOMS_PER_SYSTEM:
        unsupported = (
            "production D3(BJ) supports at most "
            f"{_D3_MAXIMUM_ATOMS_PER_SYSTEM} atoms per system"
        )
    elif len(counts) > 2**32 - 1 or total_atoms > 2**32 - 1:
        unsupported = "production D3(BJ) ragged offsets require uint32 atom counts"
    if unsupported is not None:
        return ResourceRequest(
            name,
            identity,
            (),
            exclusions,
            unsupported_reason=unsupported,
        )

    plan_host_bytes = checked_bytes(
        byte_product(len(counts) + 1, 4)
        + byte_product(total_atoms, 4)
        + byte_product(3, total_atoms, 8),
        "D3 plan host bytes",
    )
    execution_host_bytes = checked_bytes(
        byte_product(3, total_atoms, 8)
        + byte_product(2, len(counts))
        + byte_product(len(counts), 4)
        + byte_product(len(counts), 8)
        + byte_product(3, total_atoms, 8),
        "D3 execution host bytes",
    )

    device_bytes = 0
    workspace_bytes = byte_product(
        16,
        maximum_atoms if backend == "cpu" else total_atoms,
        8,
    )
    if backend == "cpu":
        execution_host_bytes = checked_bytes(
            execution_host_bytes + workspace_bytes + byte_product(3, maximum_atoms, 8),
            "D3 CPU execution host bytes",
        )
    else:
        device_bytes = checked_bytes(
            byte_product(len(counts) + 1, 4)
            + byte_product(total_atoms, 4)
            + byte_product(3, total_atoms, 8)
            + byte_product(2, len(counts))
            + byte_product(len(counts), 4)
            + byte_product(len(counts), 8)
            + byte_product(3, total_atoms, 8)
            + workspace_bytes
            + _D3_TABLE_BYTES,
            "D3 device bytes",
        )

    retained_total = checked_bytes(
        plan_host_bytes + execution_host_bytes + device_bytes,
        "D3 retained bytes",
    )
    if retained_total > maximum_bytes:
        return ResourceRequest(
            name,
            identity,
            (),
            exclusions,
            infeasible_reason=(
                f"production D3(BJ) retained owner requires {retained_total} bytes, "
                f"exceeding dispersion maximum_bytes={maximum_bytes}"
            ),
        )

    estimates = [
        ResourceEstimate(
            "D3 retained plan host",
            plan_host_bytes,
            "pageable",
            first_phase,
            last_phase,
            kind="persistent",
        ),
        ResourceEstimate(
            "D3 retained execution host",
            execution_host_bytes,
            "pageable",
            first_phase,
            last_phase,
            kind="persistent",
        ),
    ]
    if backend == "cuda":
        estimates.append(
            ResourceEstimate(
                "D3 retained CUDA owner",
                device_bytes,
                f"device:{device_id}",
                first_phase,
                last_phase,
                kind="persistent",
            )
        )
    candidate = ResourceCandidate(
        f"{backend}-d3-resident",
        "resident",
        tuple(estimates),
        decisions=(
            ("backend", backend),
            ("correction_identity", spec.identity),
            ("maximum_bytes", str(maximum_bytes)),
            ("plan_host_bytes", str(plan_host_bytes)),
            ("execution_host_bytes", str(execution_host_bytes)),
            ("device_bytes", str(device_bytes)),
            ("table_bytes", str(_D3_TABLE_BYTES)),
            ("workspace_bytes", str(workspace_bytes)),
        ),
    )
    return ResourceRequest(name, identity, (candidate,), exclusions)
