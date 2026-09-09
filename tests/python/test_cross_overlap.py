"""Native rectangular overlaps against pinned independent libcint fixtures."""

import json
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell
from vibeqc.overlap import cross_overlap
from vibeqc.projection import ProjectionRejected, project_occupied

REFERENCES = json.loads(
    (Path(__file__).parents[1] / "data/basis_projection_reference.json").read_text()
)


@pytest.mark.parametrize("fixture", REFERENCES["overlap"])
def test_rectangular_spdf_blocks_and_transpose_match_libcint(fixture):
    inputs = fixture["inputs"]

    def shells(records):
        return [
            Shell(
                s["atom_index"],
                s["angular_momentum"],
                tuple(Primitive(*p) for p in s["primitives"]),
            )
            for s in records
        ]

    target = Calculator(
        basis=shells(inputs["target_shells"]),
        basis_representation=fixture["target_rep"],
    )
    source = Calculator(
        basis=shells(inputs["source_shells"]),
        basis_representation=fixture["source_rep"],
    )
    actual = cross_overlap(
        target, source, inputs["target_atoms"], source_atoms=inputs["source_atoms"]
    )
    np.testing.assert_allclose(actual, fixture["values"], atol=1e-11, rtol=3e-12)
    transpose = cross_overlap(
        source, target, inputs["source_atoms"], source_atoms=inputs["target_atoms"]
    )
    np.testing.assert_allclose(transpose.T, actual, atol=1e-13)


def test_overlap_budget_is_checked_before_native_evaluation():
    calc = Calculator()
    with pytest.raises(MemoryError, match="maximum_bytes"):
        cross_overlap(
            calc, calc, [("H", (0, 0, 0)), ("H", (0, 0, 1.4))], maximum_bytes=1
        )


def test_diffuse_gaussian_near_duplicates_have_reproducible_projection_rank():
    """A real diffuse Gaussian null mode must be discarded before transport."""
    atoms = [("He", (0, 0, 0))]
    source = Calculator(basis=[Shell(0, 0, (Primitive(0.02, 1.0),))])
    exponents = np.array([0.02, 0.020000002])
    target = Calculator(
        basis=[Shell(0, 0, (Primitive(float(a), 1.0),)) for a in exponents]
    )
    ss = cross_overlap(source, source, atoms)
    tt = cross_overlap(target, target, atoms)
    ts = cross_overlap(target, source, atoms)
    # Closed-form normalized, coincident s-Gaussian overlap is independent of
    # the native recurrence and exposes the tiny eigenvalue in this fixture.
    expected = (
        2
        * np.sqrt(exponents[:, None] * exponents[None, :])
        / (exponents[:, None] + exponents[None, :])
    ) ** 1.5
    np.testing.assert_allclose(tt, expected, atol=1e-14, rtol=0)
    result = project_occupied(ss, tt, ts, [[1.0]])
    assert result.diagnostics.target_metric_rank == 1
    assert result.diagnostics.electron_trace == pytest.approx(2)
    np.testing.assert_allclose(result.density @ tt @ result.density, 2 * result.density)
    with pytest.raises(ProjectionRejected, match="insufficient occupied rank"):
        project_occupied(np.eye(2), tt, np.eye(2), np.eye(2))
