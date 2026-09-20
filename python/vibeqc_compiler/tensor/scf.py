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

from fractions import Fraction
from typing import Literal

from .ir import add, broadcast, einsum, input_tensor, multiply, reduce_sum
from .program import Program
from .types import Index, IndexSpace, TensorSpec

SCF_TENSOR_VERSION = 1
Reference = Literal["restricted", "unrestricted"]
ExactCoefficient = int | str | Fraction


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


def density_program(
    batch_size: int,
    nbf: int,
    *,
    spin_count: int = 1,
    orbital_count: int | None = None,
) -> Program:
    """Build D[b,s,p,q] = sum_i occ[b,s,i] C[b,s,p,i] C[b,s,q,i]."""
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
    coefficients = input_tensor("coefficients", TensorSpec((b, s, p, i), role="input"))
    occupations = input_tensor("occupations", TensorSpec((b, s, i), role="input"))
    density = einsum("bspi,bsi,bsqi->bspq", coefficients, occupations, coefficients)
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
    coefficients = input_tensor("coefficients", TensorSpec((b, s, p, i), role="input"))
    occupations = input_tensor("occupations", TensorSpec((b, s, i), role="input"))
    orbital_energies = input_tensor(
        "orbital_energies", TensorSpec((b, s, i), role="input")
    )
    weights = multiply(occupations, orbital_energies)
    weighted_density = einsum(
        "bspi,bsi,bsqi->bspq", coefficients, weights, coefficients
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
