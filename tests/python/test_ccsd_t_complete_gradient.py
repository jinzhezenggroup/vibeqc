"""Complete VibeQC RCCSD(T) analytic-gradient acceptance for issue #155 B."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from test_cc_complete_gradient import _source

from tools.cc_gradient_fixtures import inputs
from tools.validate_ccsd_t_gradient import analytic_oracle
from tools.vibeqc_cc import CCSDGradientResult, rccsd_t_force


@lru_cache(maxsize=2)
def _runtime(name: str) -> CCSDGradientResult:
    with _source(inputs(name)) as source:
        return rccsd_t_force(source, vir_chunk_size=1)


@pytest.mark.parametrize("name", ("h2o", "nh3"))
def test_complete_ccsdt_gradient_matches_pinned_pyscf(name: str) -> None:
    expected = analytic_oracle(name)["analytic"]
    result = _runtime(name)

    gradient = np.asarray(expected["gradient"], dtype=np.float64)
    np.testing.assert_allclose(result.gradient, gradient, atol=1.0e-6, rtol=0)
    np.testing.assert_allclose(
        result.total_energy,
        expected["total_energy"],
        atol=1.0e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        result.correlation_energy,
        expected["ccsd_correlation_energy"] + expected["triples_energy"],
        atol=1.0e-8,
        rtol=0,
    )
    assert result.diagnostics["triples_gradient"] is True
    assert result.diagnostics["method"] == "standard-canonical-rccsd(t)"
    assert result.diagnostics["native_public_force_capability"] is False
    assert result.lambda_residual <= 1.0e-8
    assert result.z_residual <= 1.0e-9
    assert result.orbital_stationarity <= 1.0e-8


def test_complete_ccsdt_components_close_without_projection() -> None:
    expected = analytic_oracle("h2o")["analytic"]
    result = _runtime("h2o")

    assert set(result.physical_components) == {
        "hf",
        "ccsd_baseline",
        "direct_triples",
        "delta_lambda",
        "triples_denominator",
        "same_space_canonicalization",
        "orbital_response",
        "nuclear",
    }
    np.testing.assert_allclose(
        result.gradient,
        sum(result.physical_components.values()),
        atol=1.0e-10,
        rtol=1.0e-11,
    )
    np.testing.assert_allclose(
        result.gradient.sum(axis=0),
        np.asarray(expected["gradient"]).sum(axis=0),
        atol=2.0e-8,
        rtol=0,
    )
    assert np.linalg.norm(result.physical_components["direct_triples"]) > 1.0e-10
    assert np.linalg.norm(result.physical_components["delta_lambda"]) > 1.0e-10
    assert np.linalg.norm(result.physical_components["triples_denominator"]) > 1.0e-10


def test_complete_ccsdt_endpoint_records_the_single_total_response() -> None:
    result = _runtime("h2o")

    assert (
        result.response_identity
        == result.diagnostics["total_orbital_response_identity"]
    )
    assert result.diagnostics["corrected_lambda_identity"]
    assert result.diagnostics["fixed_orbital_response_identity"]
    assert result.diagnostics["triples_energy"] != 0.0
    assert result.diagnostics["derivative_backend"] == "native-cpu-dense-oracle"
