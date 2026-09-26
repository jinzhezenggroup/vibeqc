"""Private #163-A point bridge preserves the exact #162 SCF-domain ABI."""

from pathlib import Path

import numpy as np
import pytest
from vibeqc import _native
from vibeqc._ks_snapshot import _scf_xc_points


def test_scf_point_bridge_matches_independent_domain_fixture() -> None:
    data = np.loadtxt(Path(__file__).resolve().parents[1] / "data/xc/scf_domain.tsv")
    library = _native.load_library(device="cpu")
    for pbe in (False, True):
        rows = data[data[:, 0] == int(pbe)]
        rho = rows[:, 2:4].T
        gradient = rows[:, 4:10].reshape(-1, 2, 3).transpose(1, 0, 2)
        expected = rows[:, 10:19]
        actual = _scf_xc_points(library, pbe, rho, gradient)
        packed = np.column_stack(
            [
                actual["energy"],
                actual["rho"].T,
                actual["gradient"].transpose(1, 0, 2).reshape(-1, 6),
            ]
        )
        tolerance = 5.0e-10 * np.abs(expected) + 1.0e-322
        assert np.all(np.isfinite(packed))
        assert np.all(np.abs(packed - expected) <= tolerance)


def test_scf_point_bridge_rejects_layout_and_invalid_domain() -> None:
    library = _native.load_library(device="cpu")
    with pytest.raises(ValueError, match=r"rho\[2,n\]"):
        _scf_xc_points(library, False, np.ones((3, 2)), np.zeros((2, 2, 3)))
    rho = np.array([[-1.0], [1.0]])
    with pytest.raises(RuntimeError, match="numerical failure"):
        _scf_xc_points(library, True, rho, np.zeros((2, 1, 3)))
