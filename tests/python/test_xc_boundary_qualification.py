"""Automatic physical-boundary qualification gates for generated XC."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.boundary import (
    BOUNDARY_SEMANTICS,
    bulk_feature_names,
    semilocal_boundary_probes,
)

from tools.generate_xc_cpu import build_roots

ROOT = Path(__file__).resolve().parents[2]
R2SCAN_REFERENCE = ROOT / "tests/data/xc/boundary/r2scan-zero-minority.json"


@pytest.mark.parametrize("family", ("lda", "gga", "mgga"))
@pytest.mark.parametrize("spin", ("polarized", "unpolarized"))
def test_standard_boundary_suite_matches_bulk_feature_layout(
    family: str, spin: str
) -> None:
    names = bulk_feature_names(family, spin)
    probes = semilocal_boundary_probes(names, spin=spin)
    assert probes
    assert all(probe.feature_names == names for probe in probes)
    assert all(probe.spin == spin for probe in probes)
    assert len({probe.label for probe in probes}) == len(probes)
    assert all(np.isfinite(probe.values).all() for probe in probes)


def test_zero_minority_probe_is_the_issue_1028_physical_endpoint() -> None:
    names = (
        "rho_a",
        "rho_b",
        "sigma_aa",
        "sigma_ab",
        "sigma_bb",
        "tau_a",
        "tau_b",
    )
    probe = next(
        item
        for item in semilocal_boundary_probes(names, spin="polarized")
        if item.label == "zero-minority"
    )
    rho = 0.073
    assert probe.values == pytest.approx(
        (rho, 0.0, 4.0 * rho * rho, 0.0, 0.0, 0.7 * rho, 0.0)
    )


def _r2scan_production_values(features: list[float]) -> np.ndarray | None:
    spec = functional("R2SCAN", spin="polarized")
    outputs = ((), *((index,) for index in range(len(spec.features))))
    graph, roots, _identity = build_roots(spec, outputs, production=True)
    inputs = {
        name: np.asarray([value], dtype=np.float64)
        for name, value in zip(spec.features, features, strict=True)
    }
    try:
        values = evaluate_array_graph(graph, roots, inputs)
    except (ArithmeticError, FloatingPointError, ValueError, ZeroDivisionError):
        return None
    return np.asarray(
        [np.asarray(value, dtype=np.float64).reshape(-1)[0] for value in values]
    )


def test_r2scan_zero_minority_boundary_status_is_machine_readable() -> None:
    reference = json.loads(R2SCAN_REFERENCE.read_text(encoding="utf-8"))
    assert reference["schema"] == "vibeqc.xc-boundary-reference/v1"
    assert reference["boundary_semantics"] == BOUNDARY_SEMANTICS
    assert reference["oracle"]["libxc"] == "7.0.0"
    assert reference["functional"] == "R2SCAN"
    assert reference["spin"] == "polarized"

    passed = True
    for point in reference["points"]:
        actual = _r2scan_production_values(point["features"])
        expected = np.asarray(point["expected"], dtype=np.float64)
        if (
            actual is None
            or actual.shape != expected.shape
            or not np.isfinite(actual).all()
            or not np.allclose(
                actual,
                expected,
                rtol=reference["rtol"],
                atol=reference["atol"],
            )
        ):
            passed = False
            break

    status = "pass" if passed else "fail"
    assert status == reference["expected_current_status"], (
        "boundary qualification changed; update the machine-readable status only "
        "after reviewing the independent Libxc evidence"
    )
