"""Versioned bulk production candidate domain for zero-gradient qualification."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc_compiler.xc import libxc_bulk
from vibeqc_compiler.xc.bulk_runtime import (
    PRODUCTION_CANDIDATE_DOMAIN,
    build_bulk_runtime_program,
)
from vibeqc_compiler.xc.spec import UnsupportedXC


@pytest.mark.parametrize("spin", ("polarized", "unpolarized"))
def test_production_candidate_admits_physical_zero_sigma_for_evidence(
    spin: str,
) -> None:
    interior = build_bulk_runtime_program("GGA_X_PBE_SOL", spin=spin, order=2)
    candidate = build_bulk_runtime_program(
        "GGA_X_PBE_SOL",
        spin=spin,
        order=2,
        domain=PRODUCTION_CANDIDATE_DOMAIN,
    )

    if spin == "polarized":
        features = np.asarray([[0.7], [0.4], [0.0], [0.0], [0.0]])
    else:
        features = np.asarray([[1.1], [0.0]])

    with pytest.raises(UnsupportedXC, match="positive.*sigma"):
        interior.validate_features(features)

    observed, active = candidate.validate_features(features)
    np.testing.assert_array_equal(observed, features)
    assert active.tolist() == [True]
    assert candidate.spec.domain == PRODUCTION_CANDIDATE_DOMAIN
    assert interior.spec.domain == libxc_bulk.BULK_SEMANTICS
    assert candidate.expression_hash != interior.expression_hash


@pytest.mark.parametrize("spin", ("polarized", "unpolarized"))
def test_production_candidate_still_rejects_negative_or_nonphysical_sigma(
    spin: str,
) -> None:
    candidate = build_bulk_runtime_program(
        "GGA_X_PBE_SOL",
        spin=spin,
        order=2,
        domain=PRODUCTION_CANDIDATE_DOMAIN,
    )

    if spin == "polarized":
        negative = np.asarray([[0.7], [0.4], [-1.0e-12], [0.0], [0.2]])
        indefinite = np.asarray([[0.7], [0.4], [0.1], [0.2], [0.1]])
        with pytest.raises(UnsupportedXC, match="nonnegative same-spin sigma"):
            candidate.validate_features(negative)
        with pytest.raises(UnsupportedXC, match="positive semidefinite"):
            candidate.validate_features(indefinite)
    else:
        negative = np.asarray([[1.1], [-1.0e-12]])
        with pytest.raises(UnsupportedXC, match="nonnegative sigma"):
            candidate.validate_features(negative)


def test_production_candidate_does_not_expand_density_or_tau_endpoints() -> None:
    candidate = build_bulk_runtime_program(
        "MGGA_X_R2SCAN01",
        spin="unpolarized",
        order=2,
        domain=PRODUCTION_CANDIDATE_DOMAIN,
    )

    zero_density = np.asarray([[0.0], [0.0], [0.5]])
    with pytest.raises(UnsupportedXC, match="strictly positive density"):
        candidate.validate_features(zero_density)

    zero_tau = np.asarray([[0.8], [0.0], [0.0]])
    with pytest.raises(UnsupportedXC, match="strictly positive tau"):
        candidate.validate_features(zero_tau)


def test_unknown_runtime_domain_is_fail_closed() -> None:
    with pytest.raises(UnsupportedXC, match="unsupported bulk runtime domain"):
        build_bulk_runtime_program(
            "GGA_X_PBE_SOL",
            spin="polarized",
            order=2,
            domain="libxc-bulk-production-candidate/unknown",
        )
