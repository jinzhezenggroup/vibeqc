"""Complete geometric force gates for the CPU GFN2 bootstrap.

Coordinates and forces use the native singlepoint contract: bohr and Eh/bohr.
These gates supplement, rather than replace, the independent upstream goldens
and the tighter OH bond finite-difference test in test_gfn2_xtb.py.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from vibeqc import Calculator

_CASES: dict[
    str,
    tuple[tuple[str, ...], tuple[tuple[float, float, float], ...], int, int],
] = {
    "h3-plus": (
        ("H", "H", "H"),
        ((-0.300, 0.745, 0.110), (-0.471, -0.815, -0.080), (0.941, 0.055, 0.000)),
        1,
        1,
    ),
    "oh-radical": (
        ("O", "H"),
        ((0.130, -0.200, 0.070), (0.480, 0.160, 1.904)),
        0,
        2,
    ),
}


def _calculator() -> Calculator:
    return Calculator(
        method="gfn2-xtb",
        device="cpu",
        max_iterations=300,
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    )


def _molecule(
    symbols: tuple[str, ...], coordinates: NDArray[np.float64]
) -> list[tuple[str, tuple[float, float, float]]]:
    return [
        (symbol, (float(row[0]), float(row[1]), float(row[2])))
        for symbol, row in zip(symbols, coordinates, strict=True)
    ]


def _energy(
    symbols: tuple[str, ...],
    coordinates: NDArray[np.float64],
    charge: int,
    multiplicity: int,
) -> float:
    # Each probe starts a separate calculation and must converge independently.
    result = _calculator().singlepoint(
        _molecule(symbols, coordinates),
        charge=charge,
        multiplicity=multiplicity,
        properties=("energy",),
    )
    assert result.converged
    assert np.isfinite(result.energy)
    return float(result.energy)


def _energy_forces(
    symbols: tuple[str, ...],
    coordinates: NDArray[np.float64],
    charge: int,
    multiplicity: int,
) -> tuple[float, NDArray[np.float64]]:
    result = _calculator().singlepoint(
        _molecule(symbols, coordinates),
        charge=charge,
        multiplicity=multiplicity,
        properties=("energy", "forces"),
    )
    assert result.converged
    assert np.isfinite(result.energy)
    forces = np.asarray(result.forces, dtype=np.float64)
    assert forces.shape == coordinates.shape
    assert np.isfinite(forces).all()
    return float(result.energy), forces


@pytest.mark.parametrize("case", tuple(_CASES))
@pytest.mark.parametrize("step", (2.0e-4, 1.0e-4, 5.0e-5))
def test_gfn2_asymmetric_directional_force_at_multiple_steps(
    case: str, step: float
) -> None:
    symbols, rows, charge, multiplicity = _CASES[case]
    coordinates = np.asarray(rows, dtype=np.float64)
    _, forces = _energy_forces(symbols, coordinates, charge, multiplicity)
    direction = np.array(
        ((0.61, -0.23, 0.17), (-0.37, 0.41, -0.29), (0.19, -0.53, 0.31)),
        dtype=np.float64,
    )[: len(symbols)]
    direction -= direction.mean(axis=0)
    direction /= np.linalg.norm(direction)
    plus = _energy(symbols, coordinates + step * direction, charge, multiplicity)
    minus = _energy(symbols, coordinates - step * direction, charge, multiplicity)
    finite_difference = -(plus - minus) / (2.0 * step)
    projected_force = float(np.sum(forces * direction))
    assert projected_force == pytest.approx(finite_difference, rel=0.0, abs=2.0e-7)


@pytest.mark.parametrize("case", tuple(_CASES))
def test_gfn2_energy_force_geometry_covariance(case: str) -> None:
    symbols, rows, charge, multiplicity = _CASES[case]
    coordinates = np.asarray(rows, dtype=np.float64)
    energy, forces = _energy_forces(symbols, coordinates, charge, multiplicity)
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, rtol=0.0, atol=1.0e-9)

    shift = np.array((1.371, -0.853, 0.619), dtype=np.float64)
    translated_energy, translated_forces = _energy_forces(
        symbols, coordinates + shift, charge, multiplicity
    )
    assert translated_energy == pytest.approx(energy, rel=0.0, abs=2.0e-9)
    np.testing.assert_allclose(translated_forces, forces, rtol=0.0, atol=5.0e-8)

    axis = np.array((1.0, 2.0, 3.0), dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    angle = 0.71
    rotation = (
        np.cos(angle) * np.eye(3)
        + (1.0 - np.cos(angle)) * np.outer(axis, axis)
        + np.sin(angle) * cross
    )
    rotated_energy, rotated_forces = _energy_forces(
        symbols, coordinates @ rotation.T, charge, multiplicity
    )
    assert rotated_energy == pytest.approx(energy, rel=0.0, abs=2.0e-9)
    np.testing.assert_allclose(
        rotated_forces, forces @ rotation.T, rtol=0.0, atol=5.0e-8
    )

    order = np.arange(len(symbols) - 1, -1, -1)
    permuted_symbols = tuple(symbols[int(index)] for index in order)
    permuted_energy, permuted_forces = _energy_forces(
        permuted_symbols, coordinates[order], charge, multiplicity
    )
    assert permuted_energy == pytest.approx(energy, rel=0.0, abs=2.0e-9)
    np.testing.assert_allclose(permuted_forces, forces[order], rtol=0.0, atol=5.0e-8)
