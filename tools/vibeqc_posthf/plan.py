"""Checked numeric capacities for resident MO blocks and bounded AO tiles."""

from dataclasses import dataclass
from math import prod

from tools.vibeqc_codegen.shell_signature import checked_index

from .sources import CPU_SOURCE_SCRATCH

PROVIDER_ALLOWANCE = 96 << 20
WORKSPACE_BYTES = 4 << 20


def aligned(n, alignment=256):
    return (n + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class BlockPlan:
    """One full requested MO output, two reusable stage panels and owned C.

    Budgets count FP64 numeric buffers on host and device plus an explicit
    cuBLAS allowance, not interpreter objects, BLAS host workspace, CUDA context
    code/stack, or page rounding. Source recurrence scratch is separately
    charged. No whole molecular AO four-index tensor is planned.
    """

    shape: tuple[int, ...]
    tile_shape: tuple[int, ...]
    coefficient_elements: int
    stage_elements: int
    output_elements: int
    host_bytes: int
    device_bytes: int
    allocation_bytes: int
    backend: str

    @property
    def peak_bytes(self):
        return self.host_bytes + self.device_bytes


def plan_block(snapshot, source, block, *, axis_tile, backend):
    block.validate(snapshot)
    if type(axis_tile) is not int or axis_tile < 1:
        raise ValueError("axis_tile must be positive")
    if backend not in ("cpu", "cuda"):
        raise ValueError("unknown integral transformation backend")
    if (
        source.nbf != snapshot.nmo
        or source.geometry_hash != snapshot.geometry_hash
        or source.basis_hash != snapshot.basis_hash
        or source.representation != snapshot.representation
    ):
        raise ValueError(
            "source geometry/basis/representation does not match the reference"
        )
    tile = (min(axis_tile, max(source.shell_sizes)),) * 4
    m = block.shape
    output = checked_index(prod(m), "MO output elements")
    stages = [prod(tile)] + [prod(tile[k:]) * prod(m[:k]) for k in range(1, 5)]
    stage = checked_index(max(stages), "transformation workspace")
    if backend == "cuda" and stage > (1 << 31) - 1:
        raise ValueError("MO stage exceeds cuBLAS int32 indexing")
    coefficients = checked_index(source.nbf * sum(m), "MO coefficient panels")
    # CPU: tile CG02 response assembly, immutable result publication and two
    # rotating stages coexist conservatively. CUDA also reserves a detached
    # download plus publication copy; arbitrary caller-retained exports are
    # outside provider ownership once returned.
    common = snapshot.numeric_bytes + source.numeric_bytes + CPU_SOURCE_SCRATCH
    host = checked_index(
        common + 8 * (4 * prod(tile) + 2 * stage + 3 * output + 2 * coefficients),
        "host numeric capacity",
    )
    allocation = 0
    if backend == "cuda" and output:
        allocation = (
            aligned(8 * (coefficients + 2 * stage + output)) + 256 + WORKSPACE_BYTES
        )
    device = allocation + PROVIDER_ALLOWANCE if allocation else 0
    return BlockPlan(
        m, tile, coefficients, stage, output, host, device, allocation, backend
    )
