"""Runtime-bound scalar GFN2 electronic pair science generated from TensorIR.

The production runtime owns ragged topology, validation, spin layout and SCC
iteration.  This module owns only the fixed-state per-matrix-element science
from #505 so CPU/CUDA consumers do not duplicate the Hamiltonian/VJP formulas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibeqc_compiler.tensor.ad_program import transpose_program
from vibeqc_compiler.tensor.ir import add, input_tensor, multiply
from vibeqc_compiler.tensor.program import Program
from vibeqc_compiler.tensor.types import TensorSpec

from .gfn2_electronic_contract import (
    GFN2_DIPOLE_COMPONENTS,
    GFN2_ELECTRONIC_VERSION,
    GFN2_QUADRUPOLE_COMPONENTS,
)

if TYPE_CHECKING:
    from vibeqc_compiler.tensor.ir import Node

GFN2_ELECTRONIC_PAIR_VERSION = "gfn2-runtime-electronic-pair-ir-v1"


def _input(name: str, *, differentiable: bool = False) -> Node:
    return input_tensor(
        name, TensorSpec((), role="input", differentiable=differentiable)
    )


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
    terms = [multiply(overlap, add(row_scalar, column_scalar))]

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
                    multiply(forward, column_potential),
                    multiply(reverse, row_potential),
                )
            )

    shift = add(*terms, coefficients=("-1/2",) * len(terms))
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