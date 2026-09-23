"""Shared periodic cell contract for the materials roadmap."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vibeqc_compiler.periodic import CELL_SCHEMA, PeriodicCell


def test_periodic_cell_uses_row_vector_lattice_convention() -> None:
    cell = PeriodicCell(((2.0, 0.0, 0.0), (0.5, 3.0, 0.0), (0.2, 0.4, 4.0)))
    fractional = (0.25, 0.5, 0.75)
    cartesian = cell.fractional_to_cartesian(fractional)
    expected = np.asarray(fractional) @ np.asarray(cell.lattice)
    np.testing.assert_allclose(cartesian, expected)
    np.testing.assert_allclose(cell.cartesian_to_fractional(cartesian), fractional)


def test_periodic_cell_reciprocal_and_volume_are_consistent() -> None:
    cell = PeriodicCell(((2.0, 0.0, 0.0), (0.5, 3.0, 0.0), (0.2, 0.4, 4.0)))
    direct = np.asarray(cell.lattice)
    reciprocal = np.asarray(cell.reciprocal_lattice)
    np.testing.assert_allclose(
        direct @ reciprocal.T, 2.0 * math.pi * np.eye(3), rtol=1e-14, atol=2e-15
    )
    assert cell.volume == pytest.approx(24.0)


def test_periodic_cell_identity_is_canonical_and_lattice_sensitive() -> None:
    tuple_cell = PeriodicCell(((2, 0, 0), (0, 3, 0), (0, 0, 4)))
    array_cell = PeriodicCell(np.diag([2.0, 3.0, 4.0]))
    changed = PeriodicCell(((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.5)))
    assert tuple_cell.identity == array_cell.identity
    assert tuple_cell.identity != changed.identity
    assert tuple_cell.to_payload()["schema"] == CELL_SCHEMA


def test_wrapping_is_explicit_deterministic_and_half_open() -> None:
    cell = PeriodicCell(np.diag([2.0, 3.0, 4.0]))
    before = cell.identity
    assert cell.wrap_fractional((-0.25, 1.0, 2.75)) == pytest.approx((0.75, 0.0, 0.75))
    assert cell.wrap_cartesian((-0.5, 3.0, 11.0)) == pytest.approx((1.5, 0.0, 3.0))
    assert cell.identity == before


@pytest.mark.parametrize(
    ("lattice", "message"),
    [
        (((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 0.0, 1.0)), "right-handed"),
        (((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), "right-handed"),
        (((1.0, 0.0, 0.0), (0.0, float("nan"), 0.0), (0.0, 0.0, 1.0)), "finite 3x3"),
    ],
)
def test_invalid_lattices_fail_closed(lattice: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        PeriodicCell(lattice)


def test_partial_periodicity_and_non_bohr_units_fail_closed() -> None:
    lattice = np.eye(3)
    with pytest.raises(NotImplementedError, match="3-D periodicity"):
        PeriodicCell(lattice, periodic_axes=(True, True, False))
    with pytest.raises(ValueError, match="Bohr"):
        PeriodicCell(lattice, units="Angstrom")
    with pytest.raises(TypeError, match="three booleans"):
        PeriodicCell(lattice, periodic_axes=(True, True, 1))  # type: ignore[arg-type]
