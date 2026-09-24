"""Internal periodic-system identity and invalidation contract."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc_compiler.periodic import SYSTEM_SCHEMA, PeriodicCell, PeriodicSystem


def _system() -> PeriodicSystem:
    return PeriodicSystem(
        cell=PeriodicCell(np.diag([4.0, 5.0, 6.0])),
        atomic_numbers=(8, 1),
        positions_bohr=((0.0, 0.0, 0.0), (4.5, -0.5, 6.5)),
        charge=0,
        spin=1,
    )


def test_periodic_system_identity_is_canonical_and_unit_explicit() -> None:
    tuple_system = _system()
    array_system = PeriodicSystem(
        cell=PeriodicCell(np.diag([4.0, 5.0, 6.0])),
        atomic_numbers=np.asarray([8, 1], dtype=np.int64),
        positions_bohr=np.asarray(((0.0, 0.0, 0.0), (4.5, -0.5, 6.5))),
        spin=1,
    )

    assert tuple_system.identity == array_system.identity
    assert tuple_system.geometry_identity == array_system.geometry_identity
    assert tuple_system.cell_identity == array_system.cell_identity
    payload = tuple_system.to_payload()
    assert payload["schema"] == SYSTEM_SCHEMA
    assert payload["position_units"] == "Bohr"


def test_same_geometry_update_invalidates_only_prepared_generation() -> None:
    system = _system()
    updated = system.with_positions(system.positions_bohr)

    assert updated.identity == system.identity
    assert updated.geometry_identity == system.geometry_identity
    assert updated.cell_identity == system.cell_identity
    assert updated.geometry_generation == system.geometry_generation + 1
    assert updated.cell_generation == system.cell_generation
    assert updated.prepared_identity != system.prepared_identity


def test_geometry_update_changes_geometry_identity_without_touching_cell() -> None:
    system = _system()
    updated = system.with_positions(((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)))

    assert updated.geometry_identity != system.geometry_identity
    assert updated.cell_identity == system.cell_identity
    assert updated.identity != system.identity
    assert updated.geometry_generation == 1
    assert updated.cell_generation == 0


def test_cell_update_preserves_unwrapped_cartesian_atom_identity() -> None:
    system = _system()
    updated = system.with_cell(PeriodicCell(np.diag([8.0, 9.0, 10.0])))

    assert updated.positions_bohr == system.positions_bohr
    assert updated.positions_bohr[1] == (4.5, -0.5, 6.5)
    assert updated.geometry_identity == system.geometry_identity
    assert updated.cell_identity != system.cell_identity
    assert updated.identity != system.identity
    assert updated.geometry_generation == 0
    assert updated.cell_generation == 1


def test_electronic_state_changes_system_but_not_geometry_or_cell_identity() -> None:
    system = _system()
    charged = PeriodicSystem(
        cell=system.cell,
        atomic_numbers=system.atomic_numbers,
        positions_bohr=system.positions_bohr,
        charge=1,
        spin=0,
    )

    assert charged.geometry_identity == system.geometry_identity
    assert charged.cell_identity == system.cell_identity
    assert charged.identity != system.identity


def test_periodic_system_detaches_input_arrays() -> None:
    numbers = np.asarray([8, 1], dtype=np.int64)
    positions = np.asarray(((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)))
    system = PeriodicSystem(PeriodicCell(np.eye(3)), numbers, positions)

    numbers[0] = 1
    positions[0, 0] = 99.0
    assert system.atomic_numbers == (8, 1)
    assert system.positions_bohr[0] == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("atomic_numbers", "positions", "error", "message"),
    [
        ([1.0], [[0.0, 0.0, 0.0]], TypeError, "integer array"),
        ([0], [[0.0, 0.0, 0.0]], ValueError, "1..118"),
        ([119], [[0.0, 0.0, 0.0]], ValueError, "1..118"),
        ([1], [[0.0, 0.0]], ValueError, "1x3"),
        ([1], [[float("nan"), 0.0, 0.0]], ValueError, "finite 1x3"),
        ([1], [[1.0 + 1.0j, 0.0, 0.0]], ValueError, "must be real"),
    ],
)
def test_invalid_atom_or_position_data_fail_closed(
    atomic_numbers: object,
    positions: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        PeriodicSystem(PeriodicCell(np.eye(3)), atomic_numbers, positions)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"charge": True}, TypeError, "charge must be an integer"),
        ({"spin": -1}, ValueError, "spin must be nonnegative"),
        ({"geometry_generation": -1}, ValueError, "geometry_generation"),
        ({"cell_generation": True}, TypeError, "cell_generation"),
    ],
)
def test_invalid_state_or_generation_metadata_fail_closed(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        PeriodicSystem(
            PeriodicCell(np.eye(3)),
            (1,),
            ((0.0, 0.0, 0.0),),
            **kwargs,  # type: ignore[arg-type]
        )
