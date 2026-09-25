"""Qualification-only Libxc density-boundary candidate coverage."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc_compiler.xc.bulk_runtime import (
    PRODUCTION_CANDIDATE_DOMAIN,
    PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
    build_bulk_runtime_program,
)
from vibeqc_compiler.xc.libxc_bulk import read_catalog
from vibeqc_compiler.xc.spec import UnsupportedXC


def _vacuum(program) -> np.ndarray:
    return np.zeros((len(program.spec.features), 1), dtype=np.float64)


def test_density_candidate_screens_exact_vacuum_to_zero() -> None:
    program = build_bulk_runtime_program(
        "LDA_C_VWN_4",
        spin="unpolarized",
        order=1,
        domain=PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
    )
    observed = program.evaluate(_vacuum(program))

    assert observed.shape == (len(program.outputs), 1)
    assert np.array_equal(observed, np.zeros_like(observed))
    assert program.spec.density_threshold is not None
    assert program.spec.density_threshold >= 0.0


def test_older_domains_still_reject_zero_density() -> None:
    for domain in ("libxc-bulk-interior/v1", PRODUCTION_CANDIDATE_DOMAIN):
        program = build_bulk_runtime_program(
            "GGA_X_PBE_SOL",
            spin="polarized",
            order=1,
            domain=domain,
        )
        with pytest.raises(UnsupportedXC, match="strictly positive density"):
            program.evaluate(_vacuum(program))


def test_density_candidate_preserves_active_points_in_mixed_batch() -> None:
    candidate = build_bulk_runtime_program(
        "LDA_C_VWN_4",
        spin="unpolarized",
        order=1,
        domain=PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
    )
    interior = build_bulk_runtime_program(
        "LDA_C_VWN_4",
        spin="unpolarized",
        order=1,
    )
    features = np.array([[0.73, 0.0]], dtype=np.float64)

    observed = candidate.evaluate(features)
    expected_active = interior.evaluate(features[:, :1])

    np.testing.assert_allclose(observed[:, :1], expected_active, rtol=0.0, atol=0.0)
    assert np.array_equal(observed[:, 1:], np.zeros_like(observed[:, 1:]))


def test_density_candidate_admits_empty_spin_and_zero_tau_structurally() -> None:
    program = build_bulk_runtime_program(
        "MGGA_X_R2SCAN01",
        spin="polarized",
        order=1,
        domain=PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
    )
    features = np.ones((len(program.spec.features), 1), dtype=np.float64)
    rows = {name: index for index, name in enumerate(program.spec.features)}
    features[rows["rho_b"], 0] = 0.0
    features[rows["sigma_ab"], 0] = 0.0
    features[rows["sigma_bb"], 0] = 0.0
    features[rows["tau_b"], 0] = 0.0

    checked, active = program.validate_features(features)

    assert np.array_equal(checked, features)
    assert active.tolist() == [True]


def test_density_candidate_rejects_negative_density_and_tau() -> None:
    program = build_bulk_runtime_program(
        "MGGA_X_R2SCAN01",
        spin="polarized",
        order=1,
        domain=PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
    )
    features = np.ones((len(program.spec.features), 1), dtype=np.float64)
    rows = {name: index for index, name in enumerate(program.spec.features)}
    features[rows["sigma_ab"], 0] = 0.0

    negative_density = features.copy()
    negative_density[rows["rho_b"], 0] = -1.0e-30
    with pytest.raises(UnsupportedXC, match="nonnegative density"):
        program.validate_features(negative_density)

    negative_tau = features.copy()
    negative_tau[rows["tau_b"], 0] = -1.0e-30
    with pytest.raises(UnsupportedXC, match="nonnegative tau"):
        program.validate_features(negative_tau)


def test_density_threshold_is_bound_to_pinned_registration_metadata() -> None:
    name = "GGA_X_PBE_SOL"
    program = build_bulk_runtime_program(
        name,
        spin="unpolarized",
        order=1,
        domain=PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
    )
    record = next(
        item for item in read_catalog()["registrations"] if item["name"] == name
    )
    expected = float(record["bindings"]["p_a_dens_threshold"])

    interior = build_bulk_runtime_program(name, spin="unpolarized", order=1)

    assert program.spec.density_threshold == expected
    assert program.spec.to_payload()["density_threshold"] == expected
    assert program.spec.to_payload()["domain"] == PRODUCTION_DENSITY_CANDIDATE_DOMAIN
    assert program.expression_hash != interior.expression_hash
