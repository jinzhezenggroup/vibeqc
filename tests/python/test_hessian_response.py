"""Checks for the RHF nuclear response RHS convention."""

import numpy as np
import pytest

from tools.vibeqc_hessian import build_rhf_nuclear_rhs, metric_density_response_mo


def test_metric_density_response_uses_closed_shell_occupation_connection() -> None:
    overlap = np.array([[0.2, 0.3, 0.4], [0.3, -0.1, 0.5], [0.4, 0.5, 0.6]])
    expected = -0.5 * (
        overlap * np.array([2.0, 0.0, 0.0])[None, :]
        + np.array([2.0, 0.0, 0.0])[:, None] * overlap
    )
    np.testing.assert_allclose(
        metric_density_response_mo(overlap, nocc=1), expected, rtol=0, atol=0
    )


def test_nuclear_rhs_includes_metric_fock_and_overlap_energy_gap_term() -> None:
    energies = np.array([-0.7, 0.2, 0.8])
    frozen = np.array([[0.0, 0.1, 0.2], [0.3, 0.0, 0.4], [0.5, 0.6, 0.0]])
    overlap = np.array([[0.0, 0.7, 0.8], [0.9, 0.0, 0.2], [0.3, 0.4, 0.0]])
    metric = np.array([[0.0, 0.01, 0.02], [0.03, 0.0, 0.04], [0.05, 0.06, 0.0]])
    rhs = build_rhf_nuclear_rhs(frozen, overlap, metric, energies, nocc=1)
    expected = np.array(
        [[0.3 + 0.03 - 0.5 * (0.2 - 0.7) * 0.7, 0.5 + 0.05 - 0.5 * (0.8 - 0.7) * 0.8]]
    )
    np.testing.assert_allclose(rhs, expected, rtol=0, atol=1e-15)
    assert rhs.shape == (1, 2)


def test_nuclear_rhs_rejects_missing_or_malformed_metric_inputs() -> None:
    values = np.zeros((3, 3))
    with pytest.raises(ValueError, match="finite"):
        bad = values.copy()
        bad[0, 0] = np.nan
        build_rhf_nuclear_rhs(bad, values, values, np.arange(3.0), nocc=1)
    with pytest.raises(ValueError, match="shape"):
        build_rhf_nuclear_rhs(values, values, np.zeros((2, 2)), np.arange(3.0), nocc=1)
    with pytest.raises(ValueError, match="between"):
        metric_density_response_mo(values, nocc=0)
