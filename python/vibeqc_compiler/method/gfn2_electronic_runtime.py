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
