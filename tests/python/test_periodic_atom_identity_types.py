"""Boolean atom identifiers must not become hydrogen through array promotion."""

import numpy as np
import pytest
from vibeqc_compiler.periodic import PeriodicCell, PeriodicSystem


@pytest.mark.parametrize("boolean", [True, np.bool_(True)])
@pytest.mark.parametrize("as_tuple", [False, True])
@pytest.mark.parametrize("slot", [0, 1])
def test_mixed_boolean_atom_identity_is_rejected(
    boolean: object, as_tuple: bool, slot: int
) -> None:
    numbers: list[object] = [8, 1]
    numbers[slot] = boolean
    atoms = tuple(numbers) if as_tuple else numbers
    with pytest.raises(TypeError, match="atomic_numbers"):
        PeriodicSystem(PeriodicCell(np.eye(3)), atoms, np.zeros((2, 3)))


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint64])
def test_integer_atom_identity_and_generation_controls_are_unchanged(
    dtype: type,
) -> None:
    cell = PeriodicCell(np.eye(3))
    system = PeriodicSystem(cell, np.asarray([8, 1], dtype=dtype), np.zeros((2, 3)))
    reference = PeriodicSystem(cell, (8, 1), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    assert system.identity == reference.identity
    assert system.prepared_identity == reference.prepared_identity
    updated = system.with_positions(system.positions_bohr)
    assert updated.identity == system.identity
    assert updated.prepared_identity != system.prepared_identity
