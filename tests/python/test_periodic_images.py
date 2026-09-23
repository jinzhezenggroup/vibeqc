"""Deterministic bounded lattice-image enumeration for periodic topology."""

from __future__ import annotations

import math

import numpy as np
import pytest
from vibeqc_compiler.periodic import PeriodicCell


def _brute_force_offsets(
    cell: PeriodicCell, cutoff_bohr: float, extent: int
) -> set[tuple[int, int, int]]:
    lattice = np.asarray(cell.lattice, dtype=np.float64)
    inclusive_cutoff = math.nextafter(cutoff_bohr, math.inf)
    accepted: set[tuple[int, int, int]] = set()
    for first in range(-extent, extent + 1):
        for second in range(-extent, extent + 1):
            for third in range(-extent, extent + 1):
                offset = (first, second, third)
                translation = np.asarray(offset, dtype=np.float64) @ lattice
                if (
                    math.hypot(*(float(value) for value in translation))
                    <= inclusive_cutoff
                ):
                    accepted.add(offset)
    return accepted


def test_cubic_image_enumeration_includes_only_translations_within_cutoff() -> None:
    cell = PeriodicCell(np.diag([2.0, 3.0, 4.0]))
    assert cell.lattice_image_offsets(2.01) == (
        (-1, 0, 0),
        (0, 0, 0),
        (1, 0, 0),
    )


def test_triclinic_image_enumeration_is_complete_and_deterministic() -> None:
    cell = PeriodicCell(((2.0, 0.0, 0.0), (0.7, 2.4, 0.0), (0.2, 0.5, 2.8)))
    cutoff = 3.25
    first = cell.lattice_image_offsets(cutoff)
    second = cell.lattice_image_offsets(cutoff)

    assert first == second
    assert first == tuple(sorted(first))
    assert set(first) == _brute_force_offsets(cell, cutoff, extent=5)


def test_image_enumeration_preserves_inverse_pairs_and_cell_identity() -> None:
    cell = PeriodicCell(((2.0, 0.0, 0.0), (0.7, 2.4, 0.0), (0.2, 0.5, 2.8)))
    before = cell.identity
    offsets = set(cell.lattice_image_offsets(4.0))

    assert (0, 0, 0) in offsets
    assert all(
        tuple(-component for component in offset) in offsets for offset in offsets
    )
    assert cell.identity == before


def test_image_enumeration_includes_exact_boundary_translation() -> None:
    cell = PeriodicCell(np.diag([2.0, 3.0, 4.0]))
    offsets = set(cell.lattice_image_offsets(2.0))
    assert (-1, 0, 0) in offsets
    assert (1, 0, 0) in offsets
    assert (0, 1, 0) not in offsets


def test_image_candidate_bound_fails_closed_before_pathological_enumeration() -> None:
    cell = PeriodicCell(np.diag([0.01, 0.01, 0.01]))
    with pytest.raises(ValueError, match="candidate bound"):
        cell.lattice_image_offsets(10.0, max_candidates=10_000)


@pytest.mark.parametrize("cutoff", (-1.0, float("nan"), float("inf")))
def test_image_cutoff_must_be_finite_and_nonnegative(cutoff: float) -> None:
    cell = PeriodicCell(np.eye(3))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        cell.lattice_image_offsets(cutoff)


@pytest.mark.parametrize("max_candidates", (0, -1, True, 2.5))
def test_image_candidate_limit_must_be_a_positive_integer(
    max_candidates: object,
) -> None:
    cell = PeriodicCell(np.eye(3))
    with pytest.raises(ValueError, match="positive integer"):
        cell.lattice_image_offsets(1.0, max_candidates=max_candidates)  # type: ignore[arg-type]
