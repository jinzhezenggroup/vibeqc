"""Runtime-bound scalar GFN2 electronic kernels derived from #505 TensorIR.

The full #505 programs own ragged topology, spin packing and reductions. Native
GFN2 runtimes already own those execution policies, so this module factors only
the repeated scalar scientific arithmetic into reusable TensorIR programs.
"""

from __future__ import annotations

from vibeqc_compiler.tensor.ad_program import VJPProgram, transpose_program
from vibeqc_compiler.tensor.ir import Node, add, input_tensor, multiply
from vibeqc_compiler.tensor.program import Program
from vibeqc_compiler.tensor.types import TensorSpec

from .gfn2_electronic_contract import (
    GFN2_DIPOLE_COMPONENTS,
    GFN2_ELECTRONIC_VERSION,
    GFN2_QUADRUPOLE_COMPONENTS,
)

GFN2_ELECTRONIC_RUNTIME_VERSION = "gfn2-electronic-runtime-ir-v2"


def _input(name: str, *, differentiable: bool = False) -> Node:
    return input_tensor(
        name,
        TensorSpec((), role="input", differentiable=differentiable),
    )


def build_gfn2_population_update_program() -> Program:
    """One Mulliken -P*I accumulation used for S/D/Q populations."""

    density = _input("density")
    integral = _input("integral")
    accumulator = _input("accumulator")
    updated = add(
        accumulator,
        multiply(density, integral),
        coefficients=(1, -1),
    )
    return Program(
        {"updated": updated},
        provenance={
            "kind": "gfn2-runtime-population-update",
            "version": GFN2_ELECTRONIC_RUNTIME_VERSION,
            "source": "#505 Gfn2PopulationProgram",
        },
    )


def build_gfn2_core_energy_update_program() -> Program:
    """One density-H0 accumulation from the #505 population program."""

    density = _input("density")
    h0 = _input("h0")
    accumulator = _input("accumulator")
    updated = add(accumulator, multiply(density, h0))
    return Program(
        {"updated": updated},
        provenance={
            "kind": "gfn2-runtime-core-energy-update",
            "version": GFN2_ELECTRONIC_RUNTIME_VERSION,
            "source": "#505 Gfn2PopulationProgram",
        },
    )


def build_gfn2_scalar_hamiltonian_update_program() -> Program:
    """One overlap/scalar-potential Hamiltonian update."""

    overlap = _input("overlap", differentiable=True)
    row_vat = _input("row_vat")
    row_vsh = _input("row_vsh")
    column_vat = _input("column_vat")
    column_vsh = _input("column_vsh")
    accumulator = _input("accumulator")

    # Preserve the established half-overlap and sequential FMA accumulation.
    # Summing potentials first changes overflow/cancellation behavior.
    half_overlap = add(overlap, coefficients=("-1/2",))
    updated = accumulator
    for potential in (row_vat, row_vsh, column_vat, column_vsh):
        updated = add(updated, multiply(half_overlap, potential))
    return Program(
        {"updated": updated},
        provenance={
            "kind": "gfn2-runtime-scalar-hamiltonian-update",
            "version": GFN2_ELECTRONIC_RUNTIME_VERSION,
            "source": "#505 Gfn2ElectronicProgram",
        },
    )


def build_gfn2_multipole_hamiltonian_update_program() -> Program:
    """One directed dipole/quadrupole component Hamiltonian update."""

    forward_integral = _input("forward_integral", differentiable=True)
    reverse_integral = _input("reverse_integral", differentiable=True)
    row_potential = _input("row_potential")
    column_potential = _input("column_potential")
    accumulator = _input("accumulator")

    half_forward = add(forward_integral, coefficients=("-1/2",))
    half_reverse = add(reverse_integral, coefficients=("-1/2",))
    updated = add(accumulator, multiply(half_forward, column_potential))
    updated = add(updated, multiply(half_reverse, row_potential))
    return Program(
        {"updated": updated},
        provenance={
            "kind": "gfn2-runtime-multipole-hamiltonian-update",
            "version": GFN2_ELECTRONIC_RUNTIME_VERSION,
            "source": "#505 Gfn2ElectronicProgram",
            "directed_multipole_origin": "ket-ao-atom",
        },
    )


def build_gfn2_scalar_integral_vjp_program() -> VJPProgram:
    """Generate the overlap adjoint for one canonical AO pair."""

    return transpose_program(
        build_gfn2_scalar_hamiltonian_update_program(),
        ("updated",),
        inputs=("overlap",),
    )


def build_gfn2_multipole_integral_vjp_program() -> VJPProgram:
    """Generate forward/reverse multipole integral adjoints for one AO pair."""

    return transpose_program(
        build_gfn2_multipole_hamiltonian_update_program(),
        ("updated",),
        inputs=("forward_integral", "reverse_integral"),
    )


GFN2_ELECTRONIC_PAIR_VERSION = "gfn2-runtime-electronic-pair-ir-v1"


def _integral_names() -> tuple[str, ...]:
    names = ["overlap"]
    for prefix, components in (
        ("dipole", GFN2_DIPOLE_COMPONENTS),
        ("quadrupole", GFN2_QUADRUPOLE_COMPONENTS),
    ):
        for component in components:
            names.extend(
                (f"{prefix}_forward_{component}", f"{prefix}_reverse_{component}")
            )
    return tuple(names)


GFN2_ELECTRONIC_PAIR_DIFFERENTIABLE_INPUTS = _integral_names()


def build_gfn2_runtime_electronic_pair_primal() -> Program:
    """Build one canonical matrix-pair Hamiltonian shift from #505 science."""

    overlap = _input("overlap", differentiable=True)
    row_scalar = _input("row_scalar_potential")
    column_scalar = _input("column_scalar_potential")
    # Preserve the native finite-range contract: apply -1/2 before products,
    # and do not form a potentially overflowing sum of unscaled potentials.
    half_overlap = add(overlap, coefficients=("-1/2",))
    terms = [multiply(half_overlap, row_scalar), multiply(half_overlap, column_scalar)]

    for prefix, components in (
        ("dipole", GFN2_DIPOLE_COMPONENTS),
        ("quadrupole", GFN2_QUADRUPOLE_COMPONENTS),
    ):
        for component in components:
            forward = _input(f"{prefix}_forward_{component}", differentiable=True)
            reverse = _input(f"{prefix}_reverse_{component}", differentiable=True)
            row_potential = _input(f"{prefix}_row_potential_{component}")
            column_potential = _input(f"{prefix}_column_potential_{component}")
            terms.extend(
                (
                    multiply(add(forward, coefficients=("-1/2",)), column_potential),
                    multiply(add(reverse, coefficients=("-1/2",)), row_potential),
                )
            )

    shift = add(*terms)
    return Program(
        {"shift": shift},
        provenance={
            "kind": "gfn2-runtime-electronic-pair-primal",
            "version": GFN2_ELECTRONIC_PAIR_VERSION,
            "source_graph": GFN2_ELECTRONIC_VERSION,
            "source_issue": 505,
        },
    )


def build_gfn2_runtime_electronic_pair_vjp() -> Program:
    """Generate S/D/Q pair adjoints by reverse-mode AD of the primal graph."""

    primal = build_gfn2_runtime_electronic_pair_primal()
    return transpose_program(
        primal,
        ("shift",),
        inputs=GFN2_ELECTRONIC_PAIR_DIFFERENTIABLE_INPUTS,
    ).program


def build_gfn2_runtime_overlap_vjp() -> Program:
    """Keep the unrestricted spin-only overlap response on the same AD graph."""

    vjp = build_gfn2_runtime_electronic_pair_vjp()
    return Program(
        {"bar_overlap": vjp.outputs["bar_overlap"]},
        vjp.nodes,
        provenance={
            "kind": "gfn2-runtime-electronic-overlap-vjp",
            "version": GFN2_ELECTRONIC_PAIR_VERSION,
            "parent_logical_hash": vjp.logical_hash,
            "generation": "TensorIR reverse AD",
        },
    )
