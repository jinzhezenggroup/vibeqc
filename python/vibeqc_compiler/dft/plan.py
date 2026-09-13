"""Checked numeric-buffer capacities for unscreened point/AO/derivative tiles."""

import json
from dataclasses import dataclass

from vibeqc_compiler.common.resources import (
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
)

from .ao import jet_indices
from .grid import checked_int


@dataclass(frozen=True)
class TilePlan:
    """Capacity bound, including one detached output set and staging scratch.

    All AOs are active in the unscreened route. Points and optional orbital
    columns tile; even the smallest tile must fit resident D, weighted factors
    and basis metadata. Object
    headers, allocator rounding, CPU BLAS internals and CUDA context/modules/
    stacks are outside the numeric-buffer budget. Retained caller outputs and
    simultaneously prepared batch items need separately summed capacities.
    """

    backend: str
    tile_points: int
    order: int
    nao: int
    host_bytes: int
    device_bytes: int
    allocation_bytes: int
    provider_bytes: int
    active_ao_capacity: int | None = None
    orbital_capacity: tuple[int, int] | None = None
    orbital_tile: int = 0

    @property
    def peak_bytes(self):
        return self.host_bytes + self.device_bytes

    @property
    def orbital_buffers(self):
        """Disjoint device regions; Psi depends on the tile, never all orbitals."""
        if self.orbital_capacity is None:
            return {}
        active = (
            self.nao if self.active_ao_capacity is None else self.active_ao_capacity
        )
        return {
            "orbital_factors": 8 * self.nao * sum(self.orbital_capacity),
            "orbital_pack": 8 * active * self.orbital_tile,
            "orbital_psi": 8 * 4 * self.tile_points * self.orbital_tile,
        }

    def resource_request(self, ingredients, device_id=0):
        """Compose this owner's numeric capacity with XC/other owners under #203."""
        buffers = self.orbital_buffers
        estimates = [ResourceEstimate("grid_host", self.host_bytes, "pageable", 0, 1)]
        estimates.extend(
            ResourceEstimate(name, size, f"device:{device_id}", 0, 1, kind="persistent")
            for name, size in {
                "grid_arena": self.allocation_bytes - sum(buffers.values()),
                **buffers,
            }.items()
        )
        estimates.append(
            ResourceEstimate(
                "cublas_allowance",
                self.provider_bytes,
                f"device:{device_id}",
                0,
                1,
                kind="library",
                accounting="runtime_allowance",
            )
        )
        return ResourceRequest(
            "dft_density_features",
            ResourceIdentity(
                "dft",
                "density_features",
                self.backend,
                "fp64",
                json.dumps(
                    {
                        "nao": self.nao,
                        "active": self.active_ao_capacity,
                        "order": self.order,
                        "points": self.tile_points,
                        "orbitals": self.orbital_capacity,
                        "orbital_tile": self.orbital_tile,
                    }
                ),
                tuple(ingredients),
                "point_and_orbital_tiles",
            ),
            (ResourceCandidate("streamed", "streamed", tuple(estimates)),),
            (
                "Python objects, allocator rounding and CPU BLAS internals",
                "CUDA contexts/modules/stacks beyond the explicit cuBLAS allowance",
                "caller-owned input construction/validation and retained output history",
            ),
        )


def plan_tiles(
    basis,
    *,
    backend="cpu",
    order=1,
    tile_points=256,
    budget_bytes=256 << 20,
    grid=None,
    active_ao_capacity=None,
    orbital_capacity=None,
    orbital_tile=32,
):
    """Fail before evaluation/allocation; no silent point count/backend changes."""
    jets = len(jet_indices(order))
    checked_int(tile_points, "tile points")
    checked_int(budget_bytes, "grid budget", high=2**63 - 1)
    if backend not in ("cpu", "cuda"):
        raise ValueError("unsupported grid backend")
    n, t, a = basis.nao, tile_points, basis.natom
    if active_ao_capacity is not None:
        checked_int(active_ao_capacity, "active AO capacity", high=n)
    m = n if active_ao_capacity is None else active_ao_capacity
    if orbital_capacity is not None:
        if backend != "cuda" or len(orbital_capacity) != 2:
            raise ValueError("CUDA orbital capacity requires two spin counts")
        orbital_capacity = tuple(orbital_capacity)
        for count in orbital_capacity:
            checked_int(count, "orbital capacity", low=0, high=2**31 - 1)
        checked_int(orbital_tile, "orbital tile", high=2**31 - 1)
        orbital_tile = min(orbital_tile, max(1, *orbital_capacity))
    else:
        orbital_tile = 0
    # Point partition scratch is O(tile*atoms), not O(grid*atoms). Charge
    # immutable publication copies, AO validation, feature contractions, D
    # validation/symmetrization and all index/owner arrays conservatively.
    host = (
        basis.numeric_bytes
        + (0 if grid is None else grid.numeric_bytes + grid.setup_scratch_bytes)
        + 8 * (12 * n * n + (2 * jets + 16) * t * m + 8 * t * a + 128 * t)
        + 8192
    )
    if active_ao_capacity is not None:
        host += 8 * (12 * m * m + 2 * m)
    if orbital_capacity is not None:
        # Retained/copy C, occupations, weighted-factor upload staging and
        # detached Psi-sized diagnostic scratch; caller history is excluded.
        host += 8 * (
            4 * (n + 1) * sum(orbital_capacity)
            + 2 * m * orbital_tile
            + 8 * t * orbital_tile
        )
    allocation = provider = 0
    if backend == "cuda":
        elements = basis.packed.size + 2 * n * n + 16 * t + (jets + 8) * t * m
        if active_ao_capacity is not None:
            elements += 2 * n * n + 4 * m * m + m
        if orbital_capacity is not None:
            elements += n * sum(orbital_capacity) + (m + 4 * t) * orbital_tile
        numeric = 8 * elements
        allocation = ((numeric + 255) // 256) * 256 + 256 + (4 << 20)
        provider = 96 << 20
    plan = TilePlan(
        backend,
        t,
        order,
        n,
        host,
        allocation + provider,
        allocation,
        provider,
        active_ao_capacity,
        orbital_capacity,
        orbital_tile,
    )
    if plan.peak_bytes > budget_bytes:
        raise ValueError(
            f"grid tile needs {plan.peak_bytes} numeric bytes; budget is {budget_bytes}"
        )
    return plan
