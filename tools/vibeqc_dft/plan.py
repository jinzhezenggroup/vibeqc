"""Checked numeric-buffer capacities for unscreened point/AO/derivative tiles."""

from dataclasses import dataclass

from .ao import jet_indices
from .grid import checked_int


@dataclass(frozen=True)
class TilePlan:
    """Capacity bound, including one detached output set and staging scratch.

    All AOs are active in the unscreened route. Only the point dimension tiles;
    even the smallest tile must fit resident D and basis metadata. Object
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

    @property
    def peak_bytes(self):
        return self.host_bytes + self.device_bytes


def plan_tiles(
    basis, *, backend="cpu", order=1, tile_points=256, budget_bytes=256 << 20, grid=None
):
    """Fail before evaluation/allocation; no silent point count/backend changes."""
    jets = len(jet_indices(order))
    checked_int(tile_points, "tile points")
    checked_int(budget_bytes, "grid budget", high=2**63 - 1)
    if backend not in ("cpu", "cuda"):
        raise ValueError("unsupported grid backend")
    n, t, a = basis.nao, tile_points, basis.natom
    # Point partition scratch is O(tile*atoms), not O(grid*atoms). Charge
    # immutable publication copies, AO validation, feature contractions, D
    # validation/symmetrization and all index/owner arrays conservatively.
    host = (
        basis.numeric_bytes
        + (0 if grid is None else grid.numeric_bytes + grid.setup_scratch_bytes)
        + 8 * (12 * n * n + (2 * jets + 16) * t * n + 8 * t * a + 128 * t)
        + 8192
    )
    allocation = provider = 0
    if backend == "cuda":
        numeric = 8 * (basis.packed.size + 2 * n * n + 16 * t + (jets + 8) * t * n)
        allocation = ((numeric + 255) // 256) * 256 + 256 + (4 << 20)
        provider = 96 << 20
    plan = TilePlan(
        backend, t, order, n, host, allocation + provider, allocation, provider
    )
    if plan.peak_bytes > budget_bytes:
        raise ValueError(
            f"grid tile needs {plan.peak_bytes} numeric bytes; budget is {budget_bytes}"
        )
    return plan
