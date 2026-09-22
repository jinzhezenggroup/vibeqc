"""Fixed-density CPU XC provider composition and boundary layouts."""

import typing

from vibeqc_compiler.common.layout import DenseLayout
from vibeqc_compiler.common.program import PlanCall, ProgramBuffer, ProgramIR
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.resources import byte_product, checked_bytes
from vibeqc_compiler.dft.ao import jet_indices
from vibeqc_compiler.dft.features import DENSITY_FEATURE_SCALAR_ROWS

from .coefficients import coefficient_program
from .contracts import DiscreteEnergyContract


def _dense(name: typing.Any, shape: typing.Any) -> typing.Any:
    return ProgramBuffer(
        name,
        byte_product(8, *shape),
        layout=DenseLayout(tuple(shape)),
        itemsize=8,
    )


def fixed_density_tile_program(
    contract: typing.Any,
    *,
    nao: typing.Any,
    tile_points: typing.Any,
    basis_bytes: typing.Any,
    grid_bytes: typing.Any,
    basis_identity: typing.Any,
    native_identity: typing.Any,
    packed_features: typing.Any = False,
) -> typing.Any:
    """Describe one synchronous CPU potential tile without owning its runtime."""
    if not isinstance(contract, DiscreteEnergyContract):
        raise TypeError("expected DiscreteEnergyContract")
    if contract.request.observable != "potential":
        raise ValueError("the first ProgramIR slice requires a potential request")
    if type(packed_features) is not bool:
        raise ValueError("packed_features must be bool")
    if packed_features and contract.functional.spin != "polarized":
        raise ValueError("packed feature layout currently requires polarized XC")
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
    njets = len(jet_indices(contract.ao_order))
    buffers = [
        ProgramBuffer("basis", basis_bytes),
        _dense("density", (2, nao, nao)),
        ProgramBuffer("quadrature", grid_bytes),
        _dense("jets", (njets, tile_points, nao)),
    ]
    calls = [
        PlanCall(
            "collocation",
            "dft.NativeAO.evaluate",
            canonical_hash({"basis": basis_identity, "order": contract.ao_order}),
            ("basis", "quadrature"),
            ("jets",),
        )
    ]

    if packed_features:
        scalar_shape = (len(DENSITY_FEATURE_SCALAR_ROWS), tile_points)
        buffers.append(_dense("feature_scalar", scalar_shape))
        feature_writes = ["feature_scalar"]
        if contract.ingredients.family != "lda":
            buffers.append(_dense("feature_gradient", (2, tile_points, 3)))
            feature_writes.append("feature_gradient")
        # Native scalar XC writes the consumer-ready physical derivative ABI:
        # row 0 is energy and rows 1: are the complete functional feature gradient.
        # Inactive derivative rows are explicit zeros, so Vxc can borrow rows 1:
        # without constructing a second dense gradient matrix.
        buffers.append(
            _dense("xc_rows", (1 + len(contract.functional.features), tile_points))
        )
        coefficient_rows = len(
            coefficient_program(
                contract.functional.spin, contract.ingredients.family
            ).roots
        )
        buffers.append(_dense("coefficients", (coefficient_rows, tile_points)))
        calls.append(
            PlanCall(
                "features",
                "dft.density_feature_block",
                canonical_hash(
                    {
                        "abi": "polarized-density-features-v1",
                        "family": contract.ingredients.family,
                        "layout": DenseLayout(scalar_shape).to_payload(),
                    }
                ),
                ("jets", "density"),
                tuple(feature_writes),
            )
        )
        calls.append(
            PlanCall(
                "scalar_xc",
                "xc.NativeContractionProgram.scalar_values_packed",
                native_identity,
                ("feature_scalar",),
                ("xc_rows",),
            )
        )
        vxc_reads = ["jets", "feature_scalar"]
        if contract.ingredients.family != "lda":
            vxc_reads.append("feature_gradient")
        vxc_reads.extend(("xc_rows", "quadrature"))
        calls.append(
            PlanCall(
                "vxc",
                "xc.NativeContractionProgram.potential_from_rows",
                native_identity,
                tuple(vxc_reads),
                ("coefficients", "contribution"),
            )
        )
    else:
        calls.append(
            PlanCall(
                "xc",
                "xc.NativeContractionProgram.evaluate",
                native_identity,
                ("jets", "density", "quadrature"),
                ("contribution",),
            )
        )

    buffers.append(
        ProgramBuffer(
            "contribution",
            checked_bytes(byte_product(8, nspin, nao, nao) + 24),
        )
    )
    return ProgramIR(
        "fixed_density_xc_tile",
        tuple(buffers),
        ("basis", "density", "quadrature"),
        tuple(calls),
        ("contribution",),
    )
