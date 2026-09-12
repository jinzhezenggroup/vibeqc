"""Checked numeric capacities for resident MO blocks and bounded AO tiles."""

from dataclasses import dataclass

from .plan_spec import numeric_capacity

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
    capacity = numeric_capacity(
        nbf=source.nbf,
        reference_bytes=snapshot.numeric_bytes,
        source_bytes=source.numeric_bytes,
        shape=m,
        tile=tile,
        cuda=backend == "cuda",
    )
    if backend == "cuda" and capacity["stage_elements"] > (1 << 31) - 1:
        raise ValueError("MO stage exceeds cuBLAS int32 indexing")
    return BlockPlan(
        m,
        tile,
        capacity["coefficient_elements"],
        capacity["stage_elements"],
        capacity["output_elements"],
        capacity["host_bytes"],
        capacity["device_bytes"],
        capacity["allocation_bytes"],
        backend,
    )
