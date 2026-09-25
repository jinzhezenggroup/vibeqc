"""Backend-neutral SCF tensor algebra shared by generated CPU/CUDA lowering.

This module owns only stateless scientific arithmetic.  Solver state, warm-start
repair, convergence decisions, DIIS history mutation, small linear solves,
provider selection, and ERI recurrence remain runtime/custom-primitive concerns.

Occupations are explicit floating-point weights instead of integer occupied
counts.  Current RHF/UHF semantics are represented exactly by weights 2/1 for
occupied orbitals and 0 otherwise; the same graph can later support validated
fractional occupations without adding scientific branches.
"""

from __future__ import annotations

import typing
from fractions import Fraction
from typing import Literal

from .ir import add, broadcast, einsum, input_tensor, multiply, reduce_sum
from .program import Program
from .types import Index, IndexSpace, TensorSpec

SCF_TENSOR_VERSION = 1
Reference = Literal["restricted", "unrestricted"]
ExactCoefficient = int | str | Fraction
TensorValue = typing.TypeVar("TensorValue")


def _positive(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _fraction(value: ExactCoefficient, name: str) -> Fraction:
    if type(value) not in (int, str, Fraction):
        raise TypeError(
            f"{name} must be an exact integer, Fraction, or rational string"
        )
    result = Fraction(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _reference(reference: Reference) -> tuple[int, Fraction]:
    if reference == "restricted":
        return 1, Fraction(1, 2)
    if reference == "unrestricted":
        return 2, Fraction(1)
    raise ValueError("reference must be 'restricted' or 'unrestricted'")


def _orbital_spaces(
    batch_size: int, spin_count: int, nbf: int, orbital_count: int
) -> tuple[IndexSpace, IndexSpace, IndexSpace, IndexSpace]:
    return (
        IndexSpace("batch", "batch", _positive(batch_size, "batch_size")),
        IndexSpace("spin", "spin", _positive(spin_count, "spin_count")),
        IndexSpace("ao", "ao", _positive(nbf, "nbf")),
        IndexSpace("orbital", "orbital", _positive(orbital_count, "orbital_count")),
    )


def density_input_specs(
    batch_size: int,
    nbf: int,
    *,
    spin_count: int = 1,
    orbital_count: int | None = None,
) -> dict[str, TensorSpec]:
    """Return the canonical typed inputs for SCF density construction."""
    orbital_count = nbf if orbital_count is None else orbital_count
    batch, spin, ao, orbital = _orbital_spaces(
        batch_size, spin_count, nbf, orbital_count
    )
    b, s, p, i = (
        Index("b", batch),
        Index("s", spin),
        Index("p", ao),
        Index("i", orbital),
    )
    return {
        "coefficients": TensorSpec((b, s, p, i), role="input"),
        "occupations": TensorSpec((b, s, i), role="input"),
    }


def weighted_density_input_specs(
    batch_size: int,
    nbf: int,
    *,
    spin_count: int = 1,
    orbital_count: int | None = None,
) -> dict[str, TensorSpec]:
    """Return canonical typed inputs for energy-weighted density construction."""
    specs = density_input_specs(
        batch_size,
        nbf,
        spin_count=spin_count,
        orbital_count=orbital_count,
    )
    return {
        **specs,
        "orbital_energies": TensorSpec(
            specs["occupations"].indices,
            role="input",
        ),
    }


def density_expression(
    coefficients: TensorValue,
    occupations: TensorValue,
    *,
    contract: typing.Callable[..., TensorValue],
) -> TensorValue:
    """Single-source SCF density equation, independent of graph-construction frontend."""
    return contract(
        "bspi,bsi,bsqi->bspq",
        coefficients,
        occupations,
        coefficients,
    )


def weighted_density_expression(
    coefficients: TensorValue,
    occupations: TensorValue,
    orbital_energies: TensorValue,
    *,
    contract: typing.Callable[..., TensorValue],
    multiply_values: typing.Callable[[TensorValue, TensorValue], TensorValue],
) -> TensorValue:
    """Single-source energy-weighted density equation for TensorIR/frontends."""
    weights = multiply_values(occupations, orbital_energies)
    return contract("bspi,bsi,bsqi->bspq", coefficients, weights, coefficients)


def density_program(
    batch_size: int,
    nbf: int,
    *,
    spin_count: int = 1,
    orbital_count: int | None = None,
) -> Program:
    """Build D[b,s,p,q] = sum_i occ[b,s,i] C[b,s,p,i] C[b,s,q,i]."""
    specs = density_input_specs(
        batch_size,
        nbf,
        spin_count=spin_count,
        orbital_count=orbital_count,
    )
    coefficients = input_tensor("coefficients", specs["coefficients"])
    occupations = input_tensor("occupations", specs["occupations"])
    density = density_expression(coefficients, occupations, contract=einsum)
    return Program(
        {"density": density},
        provenance={
            "scf_tensor_version": SCF_TENSOR_VERSION,
            "operation": "density",
            "occupation_semantics": "explicit_weights",
        },
    )


def weighted_density_program(
    batch_size: int,
    nbf: int,
    *,
    spin_count: int = 1,
    orbital_count: int | None = None,
) -> Program:
    """Build W[b,s,p,q] = sum_i occ_i eps_i C[p,i] C[q,i]."""
    specs = weighted_density_input_specs(
        batch_size,
        nbf,
        spin_count=spin_count,
        orbital_count=orbital_count,
    )
    coefficients = input_tensor("coefficients", specs["coefficients"])
    occupations = input_tensor("occupations", specs["occupations"])
    orbital_energies = input_tensor("orbital_energies", specs["orbital_energies"])
    weighted_density = weighted_density_expression(
        coefficients,
        occupations,
        orbital_energies,
        contract=einsum,
        multiply_values=multiply,
    )
    return Program(
        {"weighted_density": weighted_density},
        provenance={
            "scf_tensor_version": SCF_TENSOR_VERSION,
            "operation": "weighted_density",
            "occupation_semantics": "explicit_weights",
        },
    )


def fock_composition_program(
    batch_size: int,
    nbf: int,
    *,
    reference: Reference,
    exact_exchange: ExactCoefficient = 1,
    include_local_potential: bool = False,
) -> Program:
    """Compose F = H + J + V_local - a*K with RHF/UHF exchange conventions.

    The exchange provider supplies K built with the established density
    convention.  Therefore a full-range exact-exchange fraction x contributes
    -x/2 K for the restricted total-density convention and -x K per spin for
    unrestricted density blocks.
    """
    if type(include_local_potential) is not bool:
        raise TypeError("include_local_potential must be a Boolean")
    spin_count, reference_factor = _reference(reference)
    fraction = _fraction(exact_exchange, "exact_exchange")
    batch, spin, ao, _ = _orbital_spaces(batch_size, spin_count, nbf, nbf)
    b, s, p, q = Index("b", batch), Index("s", spin), Index("p", ao), Index("q", ao)
    matrix = TensorSpec((b, p, q), role="input")
    spin_matrix = TensorSpec((b, s, p, q), role="input")
    hcore = input_tensor("hcore", matrix)
    coulomb = input_tensor("coulomb", matrix)
    indices = (b, s, p, q)
    terms = [
        broadcast(hcore, indices, (0, 2, 3)),
        broadcast(coulomb, indices, (0, 2, 3)),
    ]
    coefficients: list[ExactCoefficient] = [1, 1]
    if fraction:
        exchange = input_tensor("exchange", spin_matrix)
        terms.append(exchange)
        coefficients.append(-(fraction * reference_factor))
    if include_local_potential:
        terms.append(input_tensor("local_potential", spin_matrix))
        coefficients.append(1)
    fock = add(*terms, coefficients=coefficients)
    return Program(
        {"fock": fock},
        provenance={
            "scf_tensor_version": SCF_TENSOR_VERSION,
            "operation": "fock_composition",
            "reference": reference,
            "full_range_exact_exchange": str(fraction),
            "include_local_potential": include_local_potential,
        },
    )


def energy_program(
    batch_size: int,
    nbf: int,
    *,
    spin_count: int = 1,
) -> Program:
    """Build E = E_nuc + 1/2 sum_spq D_spq (H_pq + F_spq)."""
    batch, spin, ao, _ = _orbital_spaces(batch_size, spin_count, nbf, nbf)
    b, s, p, q = Index("b", batch), Index("s", spin), Index("p", ao), Index("q", ao)
    density = input_tensor("density", TensorSpec((b, s, p, q), role="input"))
    fock = input_tensor("fock", TensorSpec((b, s, p, q), role="input"))
    hcore = input_tensor("hcore", TensorSpec((b, p, q), role="input"))
    nuclear = input_tensor("nuclear_repulsion", TensorSpec((b,), role="input"))
    hcore_spin = broadcast(hcore, (b, s, p, q), (0, 2, 3))
    electronic = reduce_sum(multiply(density, add(hcore_spin, fock)), axes=(1, 2, 3))
    total = add(nuclear, electronic, coefficients=(1, Fraction(1, 2)))
    return Program(
        {"energy": total},
        provenance={
            "scf_tensor_version": SCF_TENSOR_VERSION,
            "operation": "energy",
        },
    )


def hf_force_program(
    batch_size: int,
    nbf: int,
    *,
    spin_count: int = 1,
    coordinate_count: int = 3,
) -> Program:
    """Build stationary HF forces from nuclear, one-/two-electron, and Pulay sources.

    The supplied density and energy-weighted density are already occupation weighted:
    restricted callers use one spin block with occupation two, while unrestricted
    callers use separate alpha/beta blocks with occupation one. Integral/provider
    response is upstream; this program owns only the final stationary contraction.

    F_bc = -dE_nuc_bc - dE_2e_bc
           - sum_spq D_bspq dH_bcpq
           + sum_spq W_bspq dS_bcpq
    """
    if spin_count not in (1, 2):
        raise ValueError("HF force assembly requires one or two spin blocks")
    batch, spin, ao, _ = _orbital_spaces(batch_size, spin_count, nbf, nbf)
    coordinate = IndexSpace(
        "coordinate", "coordinate", _positive(coordinate_count, "coordinate_count")
    )
    b, s, p, q, c = (
        Index("b", batch),
        Index("s", spin),
        Index("p", ao),
        Index("q", ao),
        Index("c", coordinate),
    )
    spin_matrix = TensorSpec((b, s, p, q), role="input")
    derivative_matrix = TensorSpec((b, c, p, q), role="input")
    derivative_vector = TensorSpec((b, c), role="input")

    density = input_tensor("density", spin_matrix)
    weighted_density = input_tensor("weighted_density", spin_matrix)
    hcore_derivative = input_tensor("hcore_derivative", derivative_matrix)
    overlap_derivative = input_tensor("overlap_derivative", derivative_matrix)
    two_electron = input_tensor("two_electron", derivative_vector)
    nuclear = input_tensor("nuclear_repulsion_derivative", derivative_vector)

    one_electron = einsum("bspq,bcpq->bc", density, hcore_derivative)
    pulay = einsum("bspq,bcpq->bc", weighted_density, overlap_derivative)
    forces = add(
        nuclear,
        two_electron,
        one_electron,
        pulay,
        coefficients=(-1, -1, -1, 1),
    )
    return Program(
        {"forces": forces},
        provenance={
            "scf_tensor_version": SCF_TENSOR_VERSION,
            "operation": "hf_force_assembly",
            "spin_semantics": "occupation_weighted_blocks",
            "convention": "force_is_negative_energy_gradient",
        },
    )


def diis_gram_program(
    batch_size: int,
    history_size: int,
    nbf: int,
    *,
    spin_count: int = 1,
) -> Program:
    """Build the pure residual Gram matrix for a chronological DIIS window.

    Ring ownership, current-residual insertion, normalization, dependent-history
    retirement, and the augmented solve intentionally remain outside TensorIR.
    """
    batch, spin, ao, _ = _orbital_spaces(batch_size, spin_count, nbf, nbf)
    history = IndexSpace("history", "history", _positive(history_size, "history_size"))
    b, h, s, p, q = (
        Index("b", batch),
        Index("h", history),
        Index("s", spin),
        Index("p", ao),
        Index("q", ao),
    )
    residual_history = input_tensor(
        "residual_history", TensorSpec((b, h, s, p, q), role="input")
    )
    gram = einsum("bhspq,bkspq->bhk", residual_history, residual_history)
    return Program(
        {"gram": gram},
        provenance={
            "scf_tensor_version": SCF_TENSOR_VERSION,
            "operation": "diis_gram",
            "history_semantics": "chronological_materialized_window",
        },
    )


def diis_extrapolation_program(
    batch_size: int,
    history_size: int,
    nbf: int,
    *,
    spin_count: int = 1,
) -> Program:
    """Build F_eff[b,s,p,q] = sum_h c[b,h] F_history[b,h,s,p,q]."""
    batch, spin, ao, _ = _orbital_spaces(batch_size, spin_count, nbf, nbf)
    history = IndexSpace("history", "history", _positive(history_size, "history_size"))
    b, h, s, p, q = (
        Index("b", batch),
        Index("h", history),
        Index("s", spin),
        Index("p", ao),
        Index("q", ao),
    )
    fock_history = input_tensor(
        "fock_history", TensorSpec((b, h, s, p, q), role="input")
    )
    coefficients = input_tensor("diis_coefficients", TensorSpec((b, h), role="input"))
    effective_fock = einsum("bhspq,bh->bspq", fock_history, coefficients)
    return Program(
        {"effective_fock": effective_fock},
        provenance={
            "scf_tensor_version": SCF_TENSOR_VERSION,
            "operation": "diis_extrapolation",
            "history_semantics": "chronological_materialized_window",
        },
    )
