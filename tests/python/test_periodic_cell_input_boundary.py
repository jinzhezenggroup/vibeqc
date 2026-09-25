"""Real cell inputs cannot lose complex data or retain mutable unit metadata."""

import numpy as np
import pytest
from vibeqc_compiler.periodic import PeriodicCell


@pytest.mark.parametrize("imaginary", (1.0, float("nan"), float("inf")))
@pytest.mark.parametrize(
    "entry",
    (
        "lattice",
        "fractional_to_cartesian",
        "cartesian_to_fractional",
        "wrap_fractional",
        "wrap_cartesian",
    ),
)
def test_complex_inputs_are_rejected_before_conversion(
    entry: str, imaginary: float
) -> None:
    values = (
        np.eye(3, dtype=np.complex128)
        if entry == "lattice"
        else np.ones(3, dtype=np.complex128)
    )
    values.flat[0] = complex(1.0, imaginary)
    with pytest.raises(ValueError, match="real"):
        if entry == "lattice":
            PeriodicCell(values)
        else:
            getattr(PeriodicCell(np.eye(3)), entry)(values)


@pytest.mark.parametrize("units", (np.array("Bohr"), np.array(["Bohr"])))
def test_mutable_unit_values_are_rejected(units: object) -> None:
    with pytest.raises(ValueError, match="Bohr"):
        PeriodicCell(np.eye(3), units=units)


def test_real_cell_and_payload_are_detached_from_caller() -> None:
    lattice = np.array([[2.0, 0.0, 0.0], [0.5, 3.0, 0.0], [0.2, 0.4, 4.0]])
    axes = [True, True, True]
    cell = PeriodicCell(lattice, periodic_axes=axes)
    identity = cell.identity
    payload = cell.to_payload()
    lattice[0, 0] = 9.0
    axes[0] = False
    payload["lattice_bohr"][0][0] = 10.0
    payload["periodic_axes"][0] = False
    payload["units"] = "Angstrom"
    assert cell.identity == identity
    assert cell.volume == pytest.approx(24.0)
    np.testing.assert_allclose(
        cell.fractional_to_cartesian((0.25, 0.5, 0.75)),
        (0.9, 1.8, 3.0),
        atol=1e-15,
        rtol=0,
    )
    np.testing.assert_allclose(
        cell.cartesian_to_fractional((0.9, 1.8, 3.0)),
        (0.25, 0.5, 0.75),
        atol=1e-15,
        rtol=0,
    )
    np.testing.assert_allclose(
        np.asarray(cell.lattice) @ np.asarray(cell.reciprocal_lattice).T,
        2 * np.pi * np.eye(3),
        atol=2e-15,
        rtol=1e-14,
    )
    assert cell.wrap_fractional((-0.25, 1.0, 2.75)) == (0.75, 0.0, 0.75)
