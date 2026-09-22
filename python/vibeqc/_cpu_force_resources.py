"""Numeric capacity contract for the bounded CPU stationary force consumer.

The native SCF owner is reserved by resources_ks. This inventory covers the
additional AO owner, snapshot/export copies, derivative provider and generated
consumer staging. Python/compiler objects, loaded code, BLAS/runtime internals
and allocator overhead are excluded; this is not a process RSS bound.
"""

import typing

from vibeqc_compiler.common.resources import checked_bytes
from vibeqc_compiler.integral.ecp_policy import (
    REFINED_POLAR_POINTS,
    REFINED_RADIAL_POINTS,
)
from vibeqc_compiler.integral.first_derivative_schedule import (
    DISPATCH_ROWS,
    DISPATCH_WIDTH,
)
from vibeqc_compiler.integral.weighted_eri_inputs import PRIMITIVE_RANGE_RECORD

from .basis import BasisSet

CPU_FORCE_HOST_CAP = 256 << 20


def qualified_basis(basis: typing.Any) -> bool:
    """CPU promotion is specific to s/p/d ECP records, including their fragments."""
    return (
        isinstance(basis, BasisSet)
        and any(element.ecp_core_electrons for element in basis.elements)
        and all(
            shell.angular_momentum <= 2
            for element in basis.elements
            for shell in element.shells
        )
    )


def cpu_force_inventory(
    basis: typing.Any,
    *,
    grid_points: int,
    ecp_terms: int,
    nonlocal_correlation: bool = False,
    tile_points: int = 256,
    primitive_tile: int = 128,
    integral_terms: int = 32,
    range_exchange_sources: int = 0,
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
    if type(range_exchange_sources) is not int or not 0 <= range_exchange_sources <= 2:
        raise ValueError("CPU force range-exchange source count must be 0, 1, or 2")
    if range_exchange_sources and any(
        shell.angular_momentum > 1 for shell in basis.shells
    ):
        raise NotImplementedError(
            "CPU RSH stationary force resources currently cover s/p bases only"
        )
    if n > 16 or a > 8 or p > 128 or ecp_terms > 128:
        raise ValueError("CPU public force dense-export domain exceeded")
    if grid_points > 1_000_000:
        raise ValueError("CPU public force grid point work budget exceeded")
    packed = 3 * a + 2 * p + 16 * n
    snapshot_values = 256 + 8 * a + 20 * n * n + 8 * n + packed
    snapshot_values += 6 * grid_points + 5 * ecp_terms
    # CPU ECP provider consumes the shared generated refined-grid policy.
    # Both value/derivative grids coexist; one radial shell owns AO samples.
    sphere = 2 * REFINED_POLAR_POINTS**2
    ecp = (
        32 * n * n * (1 + 3 * a)
        + 32 * n * (sphere + 16)
        + 384 * (sphere + REFINED_RADIAL_POINTS + n + p + ecp_terms)
        + 4096
        if ecp_terms
        else 0
    )
    nonlocal_force = (
        8 * (32 * grid_points + n * n + 3 * a) if nonlocal_correlation else 0
    )

    # The generated range-ERI owner caches one PreparedWeightedEri per
    # (operator, angular-signature, Cartesian component). For s/p AOs the
    # union of public component labels has cardinality 1+3=4, hence at most
    # 4**4=256 plans per SR/LR source when both shell classes are present.
    # Each CPU plan retains one primitive-record staging array, three
    # 13-double result/publication buffers, and one yielded-record scratch.
    angular = {shell.angular_momentum for shell in basis.shells}
    component_labels = sum((l + 1) * (l + 2) // 2 for l in angular)
    range_plan_count = (
        range_exchange_sources * component_labels**4 if range_exchange_sources else 0
    )
    weighted_output_bytes = 13 * 8
    weighted_plan_host_bytes = (
        primitive_tile * PRIMITIVE_RANGE_RECORD.size
        + 3 * weighted_output_bytes
        + PRIMITIVE_RANGE_RECORD.size
    )
    blocks = {
        "snapshot_and_ao": 8 * 8 * snapshot_values + 1024 * (1 + a + n + p),
        "xc_tiles": 8 * (128 * tile_points * n + 256 * tile_points + 64 * n * n),
        "integral_staging": 8 * (40 * primitive_tile + 64 * integral_terms + 256 * a),
        "range_exchange_providers": range_plan_count * weighted_plan_host_bytes,
        # Include construction copies and the retained immutable metadata; the
        # conservative full s/p/d table also covers smaller component domains.
        "component_dispatch": 3 * 8 * (DISPATCH_ROWS * DISPATCH_WIDTH + 3 * n + 46),
        "tensor_and_grid_arenas": 32 << 20,
        "ecp_provider": ecp,
        "ecp_export_and_contraction": 8 * (24 * a * n * n + 24 * a * integral_terms),
        "nonlocal_force": nonlocal_force,
    }
    return {
        key: checked_bytes(value, "CPU force " + key) for key, value in blocks.items()
    }
