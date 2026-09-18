"""Coordinate shape and real-value admission before prepared native replay."""

import numpy as np
import pytest
from vibeqc import Calculator

XYZ = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]])
SYSTEM = [("H", position) for position in XYZ]


@pytest.mark.parametrize(
    "coordinates",
    [
        XYZ.T,
        XYZ.ravel(),
        XYZ[None, :, :],
        XYZ.astype(np.complex64) + 1j,
        XYZ.astype(np.complex128),
    ],
    ids=["transposed", "flat", "extra-axis", "complex64", "complex128-real"],
)
def test_invalid_coordinate_structure_never_executes_native(monkeypatch, coordinates):
    with Calculator(device="cpu").prepare_batch([SYSTEM]) as prepared:

        def unexpected_execute(*args):
            pytest.fail("malformed coordinates reached native execution")

        with monkeypatch.context() as guard:
            guard.setattr(prepared._library, "vibeqc_batch_execute", unexpected_execute)
            with pytest.raises(ValueError, match="coordinates.*(shape|real)"):
                prepared.execute([coordinates], strict=True)
        # Admission failures must neither overwrite geometry nor prime warm state.
        result = prepared.execute(strict=True).items[0]
        assert not result.warm_start_used
        assert result.energy == pytest.approx(-1.11671432506255, abs=2e-9)


@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "readonly", "list"])
def test_real_coordinate_layouts_preserve_energy_and_forces(layout):
    values = XYZ.copy()
    values[1, 2] = 0.8
    if layout == "fortran":
        coordinates = np.asfortranarray(values)
    elif layout == "strided":
        storage = np.zeros((2, 6))
        storage[:, ::2] = values
        coordinates = storage[:, ::2]
    elif layout == "readonly":
        coordinates = values.copy()
        coordinates.flags.writeable = False
    elif layout == "list":
        coordinates = values.tolist()
    else:
        coordinates = values.copy()
    calculator = Calculator(device="cpu")
    reference = calculator.singlepoint([("H", position) for position in values])
    with calculator.prepare_batch([SYSTEM]) as prepared:
        actual = prepared.execute([coordinates], strict=True).items[0]
    assert actual.energy == pytest.approx(reference.energy, abs=2e-10)
    np.testing.assert_allclose(actual.forces, reference.forces, atol=2e-9, rtol=0)
    np.testing.assert_array_equal(coordinates, values)


@pytest.mark.parametrize("invalid", [XYZ[:1], [0.0], np.empty(0), np.zeros((3, 3))])
def test_wrong_atom_count_keeps_native_per_item_failure_isolation(invalid):
    with Calculator(device="cpu").prepare_batch([SYSTEM, SYSTEM]) as prepared:
        result = prepared.execute([invalid, XYZ])
        assert result.failure_indices == (0,)
        assert result.items[0].forces is None
        assert result.items[1].succeeded
        assert result.items[1].energy == pytest.approx(-1.11671432506255, abs=2e-9)
