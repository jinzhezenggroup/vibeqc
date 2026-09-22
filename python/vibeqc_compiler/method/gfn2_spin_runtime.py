"""Shell spin polarization with one CPU/CUDA TensorIR arithmetic owner.

The runtime owns ragged traversal and restricted-system zero publication.
Unrestricted populations are packed as charge then magnetization; symmetric
atom-local W gives E = m.T W m / 2 and potential = dE/dm = W m.
"""

from __future__ import annotations

from vibeqc_compiler.tensor.ad_program import transpose_program
from vibeqc_compiler.tensor.ir import Node, add, constant, input_tensor, multiply
from vibeqc_compiler.tensor.program import Program
from vibeqc_compiler.tensor.types import TensorSpec

GFN2_SPIN_RUNTIME_VERSION = "gfn2-spin-runtime-ir-v1"


def _input(name: str, *, differentiable: bool = False) -> Node:
    return input_tensor(
        name, TensorSpec((), role="input", differentiable=differentiable)
    )


def build_gfn2_spin_potential_update() -> Program:
    """One ordered W*m update; native lowering preserves the runtime FMA."""
    return Program(
        {
            "updated": add(
                _input("accumulated"),
                multiply(_input("coupling"), _input("magnetization")),
            )
        },
        provenance={
            "kind": "gfn2-spin-potential-update",
            "version": GFN2_SPIN_RUNTIME_VERSION,
        },
    )


def build_gfn2_spin_energy_update() -> Program:
    """One ordered energy update, scaling magnetization before the FMA."""
    half_m = multiply(constant("1/2"), _input("magnetization"))
    return Program(
        {"updated": add(_input("accumulated"), multiply(half_m, _input("potential")))},
        provenance={
            "kind": "gfn2-spin-energy-update",
            "version": GFN2_SPIN_RUNTIME_VERSION,
        },
    )


def build_gfn2_spin_atom_primal(shells: int) -> Program:
    """Canonical symmetric shell energy for the supported s/p/d atom domain.

    Repeated off-diagonal inputs encode W symmetry explicitly. This graph
    provides the AD witness for the streaming potential used in production.
    """
    if type(shells) is not int or shells not in (1, 2, 3):
        raise ValueError("GFN2 spin atoms require one to three shells")
    magnetization = [_input(f"m_{i}", differentiable=True) for i in range(shells)]
    couplings = {
        (i, j): _input(f"w_{i}_{j}") for i in range(shells) for j in range(i, shells)
    }
    potentials = [
        add(
            *(
                multiply(couplings[min(i, j), max(i, j)], magnetization[j])
                for j in range(shells)
            )
        )
        for i in range(shells)
    ]
    energy = add(
        *(
            multiply(multiply(constant("1/2"), magnetization[i]), potentials[i])
            for i in range(shells)
        )
    )
    return Program(
        {"energy": energy},
        provenance={
            "kind": "gfn2-spin-atom-energy",
            "version": GFN2_SPIN_RUNTIME_VERSION,
            "shells": shells,
        },
    )


def build_gfn2_spin_atom_vjp(shells: int) -> Program:
    """Derive the shell potential from the canonical energy using reverse AD."""
    return transpose_program(
        build_gfn2_spin_atom_primal(shells),
        ("energy",),
        inputs=tuple(f"m_{i}" for i in range(shells)),
    ).program
