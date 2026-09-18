"""One fixed-density CPU XC tile composed from existing AO/XC providers.

The complete XC call is intentionally opaque: it already owns density features,
scalar XC and Vxc assembly. This graph does not invent separate native kernels or
claim to fuse them. Only disjoint provider-boundary buffers are described.
"""

from vibeqc_compiler.common.program import PlanCall, ProgramBuffer, ProgramIR
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.resources import byte_product, checked_bytes
from vibeqc_compiler.dft.ao import jet_indices

from .contracts import DiscreteEnergyContract


def fixed_density_tile_program(
    contract,
    *,
    nao,
    tile_points,
    basis_bytes,
    grid_bytes,
    basis_identity,
    native_identity,
):
    """Describe dense synchronous CPU potential execution, without allocating.

    Identities bind the live providers in PreparedXCContractions. The returned
    graph is an inspectable lifetime contract, not a saved executable. Response,
    geometry, CUDA and spatial leases keep their existing independent paths.
    """
    if not isinstance(contract, DiscreteEnergyContract):
        raise TypeError("expected DiscreteEnergyContract")
    if contract.request.observable != "potential":
        raise ValueError("the first ProgramIR slice requires a potential request")
    for name, value in (("nao", nao), ("tile_points", tile_points)):
        checked_bytes(value, name)
        if not value:
            raise ValueError(f"{name} must be positive")
    checked_bytes(basis_bytes, "basis bytes")
    checked_bytes(grid_bytes, "grid bytes")
    for value in (basis_identity, native_identity):
        if not isinstance(value, str) or not value:
            raise ValueError("provider identities must be nonempty strings")
    nspin = 2 if contract.functional.spin == "polarized" else 1
    jets = byte_product(8, len(jet_indices(contract.ao_order)), tile_points, nao)
    contribution = checked_bytes(byte_product(8, nspin, nao, nao) + 24)
    buffers = (
        ProgramBuffer("basis", basis_bytes),
        ProgramBuffer("density", byte_product(8, 2, nao, nao)),
        # Tiles borrow views into the complete quadrature owner. Charge that
        # owner once, not a small view as though it were a disjoint allocation.
        ProgramBuffer("quadrature", grid_bytes),
        ProgramBuffer("jets", jets),
        ProgramBuffer("contribution", contribution),
    )
    return ProgramIR(
        "fixed_density_xc_tile",
        buffers,
        ("basis", "density", "quadrature"),
        (
            PlanCall(
                "collocation",
                "dft.NativeAO.evaluate",
                canonical_hash({"basis": basis_identity, "order": contract.ao_order}),
                ("basis", "quadrature"),
                ("jets",),
            ),
            PlanCall(
                "xc",
                "xc.NativeContractionProgram.evaluate",
                native_identity,
                ("jets", "density", "quadrature"),
                ("contribution",),
            ),
        ),
        ("contribution",),
    )
