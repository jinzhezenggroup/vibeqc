"""Generic bulk Libxc point differential -> geometry pullback checks for #1122."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc._dft_gradient import StableGridMotion, _native_ao_atoms
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.method.bulk_gradient import (
    BULK_FORCE_GEOMETRY_SCHEMA,
    resolve_bulk_force_geometry_diagnostic,
)
from vibeqc_compiler.xc import libxc_bulk_capabilities
from vibeqc_compiler.xc.endpoint_capability import ENDPOINT_COVERAGE_SCHEMA
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture

from tools.vibeqc_validation.dft_gradient import finite_difference_xc_directional


def _coverage(
    *rows: tuple[str, str, tuple[str, ...]],
) -> dict:
    return {
        "schema": ENDPOINT_COVERAGE_SCHEMA,
        "coverage": [
            {
                "backend": backend,
                "spin": spin,
                "products": list(products),
            }
            for backend, spin, products in rows
        ],
    }


def _stage(
    capability: libxc_bulk_capabilities.BulkFunctionalCapability,
    stage: str,
    *,
    qualification: dict | None = None,
) -> dict:
    if stage == "production-domain" and qualification is None:
        qualification = capability.production_domain_profile.to_payload()
    payload = {
        "schema": libxc_bulk_capabilities.STAGE_EVIDENCE_SCHEMA,
        "subject_identity": capability.identity,
        "stage": stage,
        "status": "pass",
        "reason": None,
        "evidence": f"test://{capability.name}/{stage}",
    }
    if qualification is not None:
        payload["qualification"] = qualification
    return payload


def _force_evidence(
    capability: libxc_bulk_capabilities.BulkFunctionalCapability,
    *,
    spin: str,
) -> dict:
    return {
        "compiled-cpu": _stage(capability, "compiled-cpu"),
        "production-domain": _stage(capability, "production-domain"),
        "molecular-scf": _stage(
            capability,
            "molecular-scf",
            qualification=_coverage(("cpu", spin, ("energy",))),
        ),
        "forces": _stage(
            capability,
            "forces",
            qualification=_coverage(("cpu", spin, ("forces",))),
        ),
    }


@pytest.mark.parametrize(
    ("name", "spin", "minority_scale"),
    [
        ("LDA_C_VWN_4", "unpolarized", None),
        ("GGA_X_PBE_SOL", "unpolarized", None),
        ("MGGA_X_R2SCAN01", "unpolarized", None),
        ("MGGA_X_R2SCAN01", "polarized", 0.17),
        ("MGGA_X_R2SCAN01", "polarized", 1e-8),
    ],
)
def test_noncurated_bulk_geometry_matches_independent_displaced_energy(
    name: str,
    spin: str,
    minority_scale: float | None,
) -> None:
    base = libxc_bulk_capabilities.functional_capability(name)
    diagnostic = resolve_bulk_force_geometry_diagnostic(
        name,
        spin=spin,
        evidence=_force_evidence(base, spin=spin),
    )
    assert diagnostic.resolution.tau_generalized_ks is name.startswith("MGGA")
    meta, _, _ = load_integration_fixture("h2")
    args = basis_arguments(meta)
    points = np.array(
        [
            [0.27, 0.19, -0.11],
            [0.61, -0.37, 0.23],
            [-0.34, 0.28, 0.47],
            [0.18, -0.52, -0.31],
        ],
        dtype=np.float64,
    )
    weights = np.array([0.17, 0.23, 0.31, 0.29], dtype=np.float64)

    with NativeAO(**args) as basis:
        eye = np.eye(basis.nao, dtype=np.float64)
        density = (
            eye * 0.35
            if spin == "unpolarized"
            else np.stack((eye * 0.31, eye * float(minority_scale)))
        )
        jets = basis.evaluate(points, diagnostic.contraction.contract.ao_order)
        partials = diagnostic.geometry(
            jets,
            density,
            weights,
            ao_atoms=_native_ao_atoms(basis),
            natom=basis.natom,
        )

    rng = np.random.default_rng(1122)
    motion = StableGridMotion(
        topology_identity="bulk-geometry-diagnostic",
        centers=rng.normal(size=(len(args["atoms"]), 3)) * 0.03,
        points=rng.normal(size=points.shape) * 0.02,
        weights=rng.normal(size=weights.shape) * 0.002,
    )
    expected = partials.directional(
        centers=motion.centers,
        points=motion.points,
        weights=motion.weights,
    )
    oracle = finite_difference_xc_directional(
        diagnostic.functional,
        args,
        points,
        weights,
        density,
        motion,
        steps=(1e-3, 3e-4, 1e-4),
        point_energy=diagnostic.point_energy,
    )

    assert oracle.spread < 2e-7
    np.testing.assert_allclose(oracle.stable_estimate, expected, atol=5e-8)
    payload = diagnostic.to_payload()
    assert payload["schema"] == BULK_FORCE_GEOMETRY_SCHEMA
    assert payload["force_resolution"]["identity"] == base.identity
    assert payload["point_execution_domain"] == "libxc-bulk-interior/v1"
    assert payload["claim"] == "interior-fixed-density-geometry-diagnostic"
