"""Fast directional finite-difference gate for the water MP2 gradient."""

import typing

import numpy as np
from vibeqc import Calculator

from tools.vibeqc_mp2.gradient import (
    canonical_energy_adjoint,
    canonical_lagrangian_weights,
    canonical_orbital_rhs,
    dense_molecular_gradient_oracle,
    solve_canonical_orbital_response,
)
from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response import DenseAOResponseBackend, GMRESOptions


def _internal_directions(shape: typing.Any) -> typing.Any:
    """Return two deterministic, translation-free, orthonormal directions."""
    rng = np.random.default_rng(20260914)
    directions = []
    for raw in rng.normal(size=(2, *shape)):
        direction = raw - raw.mean(axis=0, keepdims=True)
        for previous in directions:
            direction -= np.vdot(previous, direction) * previous
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            raise AssertionError("direction construction became singular")
        directions.append(direction / norm)
    return directions


def test_water_complete_gradient_matches_directional_finite_differences() -> None:
    """Keep a two-direction energy-only FD gate on the ordinary CI path.

    The exhaustive water Cartesian check remains in test_mp2_gradient.py and is
    run by scheduled/manual full CI. Ordinary PR/master CI checks two
    independent internal directions at two finite-difference steps, retaining
    both an absolute fine-step accuracy gate and a convergence gate.
    """
    meta, arrays = load_fixture("water")
    arguments = source_arguments(meta)
    reference = fixture_snapshot(meta, arrays)
    occupied = reference.nocc
    eri = arrays["conventional_mo"]
    g = eri[:occupied, occupied:, :occupied, occupied:].transpose(0, 2, 1, 3)
    adjoint = canonical_energy_adjoint(
        g,
        reference.orbital_energies,
        occupied,
        reference_identity=reference.identity,
        hamiltonian_id=reference.hamiltonian_id,
    )
    hcore_mo = (
        reference.coefficients.T @ arrays["conventional_h"] @ reference.coefficients
    )
    orbital = canonical_orbital_rhs(hcore_mo, eri, adjoint, occupied)
    response = solve_canonical_orbital_response(
        reference,
        DenseAOResponseBackend(arrays["ao"]),
        orbital,
        options=GMRESOptions(rtol=1e-12, atol=1e-13, max_iterations=100),
    )
    weights = canonical_lagrangian_weights(hcore_mo, eri, adjoint, response, occupied)
    with NativeSource(**arguments) as source:
        analytic = dense_molecular_gradient_oracle(reference, source, weights)

    calculator = Calculator(
        method="mp2",
        basis=arguments["basis"],
        basis_representation=arguments["representation"],
    )
    atomic_numbers = [value.atomic_number for value in arguments["atoms"]]
    positions = np.asarray(
        [value.position for value in arguments["atoms"]], dtype=float
    )

    for direction in _internal_directions(analytic.shape):
        expected = float(np.vdot(analytic, direction))
        errors = []
        for step in (1e-3, 3e-4):
            plus_atoms = [
                (atomic_number, position.tolist())
                for atomic_number, position in zip(
                    atomic_numbers, positions + step * direction, strict=True
                )
            ]
            minus_atoms = [
                (atomic_number, position.tolist())
                for atomic_number, position in zip(
                    atomic_numbers, positions - step * direction, strict=True
                )
            ]
            plus = calculator.singlepoint(
                plus_atoms, charge=arguments["charge"], properties=("energy",)
            ).energy
            minus = calculator.singlepoint(
                minus_atoms,
                charge=arguments["charge"],
                properties=("energy",),
            ).energy
            finite = (plus - minus) / (2 * step)
            errors.append(abs(finite - expected))
        assert errors[-1] < 1e-6
        assert min(errors) < 1e-7
        assert errors[-1] < errors[0]
