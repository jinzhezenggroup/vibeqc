"""Public CPU DF force parity on exact shared basis data, not catalog names."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from vibeqc import Calculator
from vibeqc.basis import BasisSet

if TYPE_CHECKING:
    from collections.abc import Sequence

    from vibeqc.calculator import Result

_WATER = (
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, 1.432, 1.107)),
    ("H", (0.0, -1.432, 1.107)),
)


@pytest.mark.parametrize(
    "method,basis,atoms,multiplicity",
    (
        ("rhf", "sto-3g", (("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))), 1),
        ("rhf", "sto-3g", _WATER, 1),
        ("uhf", "sto-3g", (("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.834))), 2),
        ("rhf", "def2-svp", _WATER, 1),
    ),
    ids=("h2", "water", "oh-open-shell", "water-svp"),
)
def test_weighted_df_public_forces_match_materialized_and_independent_oracle(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    basis: str,
    atoms: Sequence[tuple[str, tuple[float, float, float]]],
    multiplicity: int,
) -> None:
    pyscf = pytest.importorskip("pyscf")

    def calculator() -> Calculator:
        return Calculator(
            method=method,
            basis=basis,
            auxiliary_basis=basis,
            density_fitting="cpu",
            max_iterations=200,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-14,
        )

    results: list[Result] = []
    for flag in ("0", "1"):
        monkeypatch.setenv("VIBEQC_CPU_DF_MATERIALIZED_DERIVATIVES", flag)
        result = calculator().singlepoint(atoms, multiplicity=multiplicity)
        assert result.converged
        assert result.executed_backend == "cpu_reference"
        assert result.forces is not None and np.isfinite(result.forces).all()
        results.append(result)
    fused, materialized = results
    np.testing.assert_allclose(fused.energy, materialized.energy, atol=3e-10, rtol=0)
    np.testing.assert_allclose(fused.forces, materialized.forces, atol=3e-8, rtol=0)
    assert fused.iterations == materialized.iterations

    # Identical names can resolve different rounded basis coefficients in the
    # two catalogs. Feed the oracle the exact orbital/auxiliary primitive data.
    canonical = calculator()._basis
    assert isinstance(canonical, BasisSet)
    matched_basis = {}
    for symbol in {symbol for symbol, _ in atoms}:
        element = canonical.by_element[pyscf.gto.charge(symbol)]
        matched_basis[symbol] = [
            [
                shell.angular_momentum,
                *[
                    [float(exponent), float(coefficient)]
                    for exponent, coefficient in zip(shell.exponents, row, strict=True)
                ],
            ]
            for shell in element.shells
            for row in shell.coefficients
        ]
    molecule = pyscf.gto.M(
        atom=atoms,
        basis=matched_basis,
        unit="Bohr",
        spin=multiplicity - 1,
        cart=True,
        verbose=0,
    )
    reference = (
        pyscf.scf.UHF(molecule) if method == "uhf" else pyscf.scf.RHF(molecule)
    ).density_fit(auxbasis=matched_basis)
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-9
    reference.max_cycle = 200
    reference.kernel()
    assert reference.converged
    gradient = reference.nuc_grad_method()
    gradient.auxbasis_response = True
    expected_force = -gradient.kernel()
    np.testing.assert_allclose(fused.energy, reference.e_tot, atol=5e-9, rtol=0)
    np.testing.assert_allclose(fused.forces, expected_force, atol=3e-7, rtol=0)
    np.testing.assert_allclose(np.sum(fused.forces, axis=0), 0, atol=3e-9, rtol=0)

    monkeypatch.setenv("VIBEQC_CPU_DF_MATERIALIZED_DERIVATIVES", "0")
    for step in (2e-4, 5e-5):
        energies = []
        for sign in (1, -1):
            shifted = [(symbol, list(position)) for symbol, position in atoms]
            shifted[-1][1][2] += sign * step
            probe = calculator().singlepoint(
                shifted, multiplicity=multiplicity, properties=("energy",)
            )
            assert probe.converged
            energies.append(probe.energy)
        finite_difference = -(energies[0] - energies[1]) / (2 * step)
        assert fused.forces[-1, 2] == pytest.approx(finite_difference, abs=2e-6, rel=0)
