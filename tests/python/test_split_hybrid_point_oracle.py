"""Independent point acceptance and adversarial validation-harness checks."""

import os
from pathlib import Path

import numpy as np
import pytest

from tools.split_hybrid_wide_oracle import oracle_source
from tools.validate_split_hybrid_cuda import (
    comparison,
    empty_spin_neighbors,
    finite_difference_points,
    libxc_density,
    point_inputs,
    validate,
)


def test_point_probe_inputs_have_the_documented_invariants() -> None:
    points, labels, count = point_inputs()
    assert count == 24
    assert len(points) == 50
    assert {
        "vacuum",
        "empty-alpha",
        "empty-beta",
        "zero-gradient",
        "zero-tau",
        "restricted-embedding",
        "antiparallel-gradient",
    } <= set(labels)
    rho = libxc_density(points)
    np.testing.assert_allclose(np.sum(rho[0, 1:4] ** 2, axis=0), points[:, 2])
    np.testing.assert_allclose(np.sum(rho[0, 1:4] * rho[1, 1:4], axis=0), points[:, 3])
    np.testing.assert_allclose(np.sum(rho[1, 1:4] ** 2, axis=0), points[:, 4])
    trials, steps = finite_difference_points(points)
    assert len(trials) == 56 and np.all(steps > 0)
    libxc_density(trials)


def test_point_acceptance_never_hides_nonfinite_values() -> None:
    reference = np.zeros((2, 8))
    assert comparison(reference, reference)["passed"]
    assert not comparison(np.full((2, 8), np.nan), reference)["passed"]
    assert not comparison(reference, np.full((2, 8), np.inf))["passed"]
    assert not comparison(np.ones((2, 8)), reference)["passed"]


def test_empty_spin_neighbors_change_only_one_majority_ulp() -> None:
    points, labels, _ = point_inputs()
    neighbors, names = empty_spin_neighbors(points, labels)
    assert len(neighbors) == len(names) == 6
    for index in (0, 3):
        majority = 1 if names[index] == "empty-alpha" else 0
        assert np.array_equal(neighbors[index], points[labels.index(names[index])])
        for offset, direction in ((1, -np.inf), (2, np.inf)):
            assert neighbors[index + offset, majority] == np.nextafter(
                neighbors[index, majority], direction
            )
            np.testing.assert_array_equal(
                np.delete(neighbors[index + offset], majority),
                np.delete(neighbors[index], majority),
            )


def test_wide_oracle_rejects_nonpinned_archive(tmp_path: Path) -> None:
    archive = tmp_path / "libxc-7.0.0.tar.gz"
    archive.write_bytes(b"unverified upstream archive")
    with pytest.raises(ValueError, match="pinned Libxc 7.0.0"):
        oracle_source(archive)


def test_generated_split_points_match_independent_libxc(tmp_path: Path) -> None:
    # The reference extra is optional for developer unit tests. The CUDA
    # acceptance CLI does not skip or substitute when any dependency is missing.
    pytest.importorskip("pyscf")
    archive = os.environ.get("LIBXC_700_SOURCE_ARCHIVE")
    if not archive:
        pytest.skip("wide point acceptance requires LIBXC_700_SOURCE_ARCHIVE")
    report = validate("host", tmp_path, None, "7.0.0", Path(archive))
    assert report["methods"]["MN15"]["passed"], report
    m06 = report["methods"]["M06-2X"]
    assert m06["passed"] and m06["high_precision_boundary"]["passed"], report
    assert not m06["binary64_full_diagnostic"]["passed"]
    assert report["passed"] == (not report["dirty_tracked_sources"]), report
