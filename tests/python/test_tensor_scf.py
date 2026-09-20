"""SCF TensorIR algebra parity against the existing handwritten equations."""

from fractions import Fraction

import numpy as np
import pytest
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.tensor import (
    density_program,
    diis_extrapolation_program,
    diis_gram_program,
    energy_program,
    execute,
    fock_composition_program,
    weighted_density_program,
)


def test_density_and_weighted_density_match_rhf_kernel_equations() -> None:
    rng = np.random.default_rng(611)
    coefficients = rng.normal(size=(2, 1, 3, 3))
    occupations = np.array([[[2.0, 2.0, 0.0]], [[2.0, 0.0, 0.0]]])
    orbital_energies = rng.normal(size=(2, 1, 3))

    density = execute(
        density_program(2, 3),
        {"coefficients": coefficients, "occupations": occupations},
    ).outputs["density"]
    weighted = execute(
        weighted_density_program(2, 3),
        {
            "coefficients": coefficients,
            "occupations": occupations,
            "orbital_energies": orbital_energies,
        },
    ).outputs["weighted_density"]

    expected_density = np.einsum(
        "bspi,bsi,bsqi->bspq", coefficients, occupations, coefficients
    )
    expected_weighted = np.einsum(
        "bspi,bsi,bsi,bsqi->bspq",
        coefficients,
        occupations,
        orbital_energies,
        coefficients,
    )
    np.testing.assert_allclose(density, expected_density, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(weighted, expected_weighted, atol=1e-13, rtol=1e-13)


def test_density_uses_explicit_uhf_occupation_weights() -> None:
    rng = np.random.default_rng(612)
    coefficients = rng.normal(size=(1, 2, 3, 3))
    occupations = np.array([[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
    result = execute(
        density_program(1, 3, spin_count=2),
        {"coefficients": coefficients, "occupations": occupations},
    ).outputs["density"]
    expected = np.einsum("bspi,bsi,bsqi->bspq", coefficients, occupations, coefficients)
    np.testing.assert_allclose(result, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize(
    ("reference", "fraction", "spin_count", "exchange_scale"),
    [
        ("restricted", Fraction(1), 1, Fraction(1, 2)),
        ("restricted", Fraction(1, 4), 1, Fraction(1, 8)),
        ("unrestricted", Fraction(1), 2, Fraction(1)),
        ("unrestricted", Fraction(1, 4), 2, Fraction(1, 4)),
    ],
)
def test_fock_composition_matches_current_rhf_uhf_conventions(
    reference: str,
    fraction: Fraction,
    spin_count: int,
    exchange_scale: Fraction,
) -> None:
    rng = np.random.default_rng(613)
    hcore = rng.normal(size=(2, 3, 3))
    coulomb = rng.normal(size=(2, 3, 3))
    exchange = rng.normal(size=(2, spin_count, 3, 3))
    local = rng.normal(size=(2, spin_count, 3, 3))
    program = fock_composition_program(
        2,
        3,
        reference=reference,
        exact_exchange=fraction,
        include_local_potential=True,
    )
    result = execute(
        program,
        {
            "hcore": hcore,
            "coulomb": coulomb,
            "exchange": exchange,
            "local_potential": local,
        },
    ).outputs["fock"]
    expected = (
        hcore[:, None, :, :]
        + coulomb[:, None, :, :]
        - float(exchange_scale) * exchange
        + local
    )
    np.testing.assert_allclose(result, expected, atol=1e-13, rtol=1e-13)


def test_pure_method_fock_does_not_require_exchange_input() -> None:
    rng = np.random.default_rng(614)
    hcore = rng.normal(size=(1, 2, 2))
    coulomb = rng.normal(size=(1, 2, 2))
    local = rng.normal(size=(1, 1, 2, 2))
    method = resolve_method("PBE")
    assert method.full_range_exact_exchange == 0
    program = fock_composition_program(
        1,
        2,
        reference=method.reference,
        exact_exchange=method.full_range_exact_exchange,
        include_local_potential=True,
    )
    result = execute(
        program,
        {"hcore": hcore, "coulomb": coulomb, "local_potential": local},
    ).outputs["fock"]
    np.testing.assert_allclose(
        result, hcore[:, None] + coulomb[:, None] + local, atol=1e-13, rtol=1e-13
    )


def test_method_ir_is_single_source_for_full_range_exchange_fraction() -> None:
    assert resolve_method("PBE").full_range_exact_exchange == 0
    assert resolve_method("PBE0").full_range_exact_exchange == Fraction(1, 4)
    assert resolve_method("B3LYP").full_range_exact_exchange == Fraction(1, 5)
    assert resolve_method("CAM-B3LYP").full_range_exact_exchange == 0


@pytest.mark.parametrize("spin_count", [1, 2])
def test_energy_matches_current_cuda_kernel_equation(spin_count: int) -> None:
    rng = np.random.default_rng(615 + spin_count)
    density = rng.normal(size=(3, spin_count, 2, 2))
    fock = rng.normal(size=(3, spin_count, 2, 2))
    hcore = rng.normal(size=(3, 2, 2))
    nuclear = rng.normal(size=3)
    result = execute(
        energy_program(3, 2, spin_count=spin_count),
        {
            "density": density,
            "fock": fock,
            "hcore": hcore,
            "nuclear_repulsion": nuclear,
        },
    ).outputs["energy"]
    expected = nuclear + 0.5 * np.einsum(
        "bspq,bspq->b",
        density,
        hcore[:, None, :, :] + fock,
    )
    np.testing.assert_allclose(result, expected, atol=1e-13, rtol=1e-13)


def test_diis_pure_tensor_parts_match_fixed_history_equations() -> None:
    rng = np.random.default_rng(618)
    residual = rng.normal(size=(2, 4, 2, 3, 3))
    fock = rng.normal(size=(2, 4, 2, 3, 3))
    coefficients = rng.normal(size=(2, 4))

    gram = execute(
        diis_gram_program(2, 4, 3, spin_count=2),
        {"residual_history": residual},
    ).outputs["gram"]
    effective = execute(
        diis_extrapolation_program(2, 4, 3, spin_count=2),
        {"fock_history": fock, "diis_coefficients": coefficients},
    ).outputs["effective_fock"]

    np.testing.assert_allclose(
        gram,
        np.einsum("bhspq,bkspq->bhk", residual, residual),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        effective,
        np.einsum("bhspq,bh->bspq", fock, coefficients),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: density_program(0, 2),
        lambda: density_program(1, 0),
        lambda: density_program(1, 2, spin_count=0),
        lambda: diis_gram_program(1, 0, 2),
        lambda: fock_composition_program(1, 2, reference="invalid"),
        lambda: fock_composition_program(
            1, 2, reference="restricted", exact_exchange=-1
        ),
    ],
)
def test_scf_tensor_contracts_fail_closed(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()
