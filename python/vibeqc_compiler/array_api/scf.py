"""First SCF tensor-algebra consumer of the bounded Array frontend.

TensorIR remains the scientific owner of the SCF equation/signature contract.
This module only changes graph construction: ordinary symbolic array
expressions are traced once and disappear into an ordinary TensorIR Program.
"""

from __future__ import annotations

import typing

from vibeqc_compiler.tensor.scf import (
    SCF_TENSOR_VERSION,
    density_expression,
    density_input_specs,
    weighted_density_expression,
    weighted_density_input_specs,
)

from . import namespace as xp
from .trace import trace

if typing.TYPE_CHECKING:
    from vibeqc_compiler.tensor.program import Program


def density_program(
    batch_size: int,
    nbf: int,
    *,
    spin_count: int = 1,
    orbital_count: int | None = None,
) -> Program:
    """Build the validated SCF density DAG through Array frontend capture."""
    specs = density_input_specs(
        batch_size,
        nbf,
        spin_count=spin_count,
        orbital_count=orbital_count,
    )
    return trace(
        lambda coefficients, occupations: {
            "density": density_expression(
                coefficients,
                occupations,
                contract=xp.einsum,
            )
        },
        specs,
        provenance={
            "scf_tensor_version": SCF_TENSOR_VERSION,
            "operation": "density",
            "occupation_semantics": "explicit_weights",
            "construction": "array_frontend",
        },
    )


def weighted_density_program(
    batch_size: int,
    nbf: int,
    *,
    spin_count: int = 1,
    orbital_count: int | None = None,
) -> Program:
    """Build the validated weighted-density DAG through Array frontend capture."""
    specs = weighted_density_input_specs(
        batch_size,
        nbf,
        spin_count=spin_count,
        orbital_count=orbital_count,
    )
    return trace(
        lambda coefficients, occupations, orbital_energies: {
            "weighted_density": weighted_density_expression(
                coefficients,
                occupations,
                orbital_energies,
                contract=xp.einsum,
                multiply_values=xp.multiply,
            )
        },
        specs,
        provenance={
            "scf_tensor_version": SCF_TENSOR_VERSION,
            "operation": "weighted_density",
            "occupation_semantics": "explicit_weights",
            "construction": "array_frontend",
        },
    )
