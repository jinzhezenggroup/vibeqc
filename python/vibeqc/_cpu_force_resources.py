"""Numeric capacity contract for the bounded CPU stationary force consumer.

The native SCF owner is reserved by resources_ks. This inventory covers the
additional AO owner, snapshot/export copies, derivative provider and generated
consumer staging. Python/compiler objects, loaded code, BLAS/runtime internals
and allocator overhead are excluded; this is not a process RSS bound.
"""

import typing

from .basis import BasisSet
from .resources import checked_bytes

CPU_FORCE_HOST_CAP = 256 << 20


def qualified_basis(basis: typing.Any) -> bool:
    """CPU promotion is specific to s/p ECP records, including their fragments."""
    return (
        isinstance(basis, BasisSet)
        and any(element.ecp_core_electrons for element in basis.elements)
        and all(
            shell.angular_momentum <= 1
            for element in basis.elements
            for shell in element.shells
        )
    )


def cpu_force_inventory(
    basis: typing.Any,
    *,
    grid_points: int,
    ecp_terms: int,
    tile_points: int = 256,
    primitive_tile: int = 128,
    integral_terms: int = 32,
) -> dict[str, int]:
    """Conservative simultaneous numeric capacities; no scientific execution.

    Snapshot factor eight covers native source/state copies, the wire export,
    immutable copies, decoded grid and identity validation. XC factor 128
    covers AO jets (at most 10), immutable/BLAS copies, feature coefficients,
    generated pullback input/output and the previous live tile. Tensor and
    grid adapters each independently enforce an 8 MiB arena cap. Sum phases
    rather than relying on Python's timing of releasing previous locals.
    """
    n, a, p = basis.nao, basis.natom, basis.nprimitive
    if n > 16 or a > 8 or p > 128 or ecp_terms > 128:
        raise ValueError("CPU public force dense-export domain exceeded")
    if grid_points > 1_000_000:
        raise ValueError("CPU public force grid point work budget exceeded")
    packed = 3 * a + 2 * p + 16 * n
    snapshot_values = 256 + 8 * a + 20 * n * n + 8 * n + packed
    snapshot_values += 6 * grid_points + 5 * ecp_terms
    # CPU ECP provider uses its independent fixed 224/44 refined grid.
    # Both value/derivative grids coexist; one radial shell owns AO samples.
    sphere = 2 * 44**2
    ecp = (
        32 * n * n * (1 + 3 * a)
        + 32 * n * (sphere + 16)
        + 384 * (sphere + 224 + n + p + ecp_terms)
        + 4096
        if ecp_terms
        else 0
    )
    blocks = {
        "snapshot_and_ao": 8 * 8 * snapshot_values + 1024 * (1 + a + n + p),
        "xc_tiles": 8 * (128 * tile_points * n + 256 * tile_points + 64 * n * n),
        "integral_staging": 8 * (40 * primitive_tile + 64 * integral_terms + 256 * a),
        "tensor_and_grid_arenas": 32 << 20,
        "ecp_provider": ecp,
        "ecp_export_and_contraction": 8 * (24 * a * n * n + 24 * a * integral_terms),
    }
    return {
        key: checked_bytes(value, "CPU force " + key) for key, value in blocks.items()
    }
