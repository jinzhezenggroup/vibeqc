"""Runtime bridge coverage for pointwise-qualified bulk Libxc Graphs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.xc import build_bulk_runtime_program
from vibeqc_compiler.xc.cuda_emit import XCSchedule, emit_cuda
from vibeqc_compiler.xc.libxc_bulk import build_bulk_program
from vibeqc_compiler.xc.libxc_bulk_capabilities import available_capabilities
from vibeqc_compiler.xc.spec import UnsupportedXC

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    case
    for path in sorted((ROOT / "tests/data/xc/libxc-bulk").glob("*.json"))
    for case in json.loads(path.read_text())["cases"]
]
NAMES = ("LDA_C_VWN_4", "GGA_X_PBE_SOL", "MGGA_X_R2SCAN01")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("spin", ("polarized", "unpolarized"))
def test_bulk_runtime_bridge_matches_independent_pointwise_fixture(
    name: str, spin: str
) -> None:
    runtime = build_bulk_runtime_program(name, spin=spin, order=1)
    full = build_bulk_program(name, spin=spin)
    case = next(case for case in CASES if case["name"] == name and case["spin"] == spin)
    full_features = np.asarray(case["features"], dtype=np.float64).T
    feature_rows = dict(zip(full.features, full_features, strict=True))
    compact = np.stack([feature_rows[name] for name in runtime.spec.features])

    actual = runtime.evaluate(compact).T
    expected_columns = [0] + [
        1 + full.features.index(feature) for feature in runtime.spec.features
    ]
    expected = np.asarray(case["expected"], dtype=np.float64)[:, expected_columns]
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-10)

    payload = runtime.spec.to_payload()
    assert payload["production_admitted"] is False
    assert payload["qualification"] == "pointwise-validated"


def test_tau_mgga_runtime_contract_drops_dead_laplacian_inputs() -> None:
    full = build_bulk_program("MGGA_X_R2SCAN01", spin="polarized")
    runtime = build_bulk_runtime_program("MGGA_X_R2SCAN01", spin="polarized", order=1)
    assert ("lapl_a", "lapl_b") == tuple(
        feature for feature in full.features if feature.startswith("lapl_")
    )
    assert "lapl_a" not in runtime.spec.features
    assert "lapl_b" not in runtime.spec.features
    assert runtime.spec.features[-2:] == ("tau_a", "tau_b")
    assert runtime.spec.ingredients == ("rho", "sigma", "tau")


@pytest.mark.parametrize("name", NAMES)
def test_bulk_runtime_program_uses_common_cuda_execution_contract(name: str) -> None:
    program = build_bulk_runtime_program(name, order=1)
    source, contract, _ = emit_cuda(
        program, XCSchedule(variant="fused", threads=128, group_size=8)
    )
    assert contract["expression_hash"] == program.expression_hash
    assert tuple(contract["features"]) == program.spec.features
    assert contract["functional"]["production_admitted"] is False
    assert f"constexpr size_t XC_INPUTS = {len(program.spec.features)};" in source
    assert f"constexpr size_t XC_OUTPUTS = {len(program.outputs)};" in source


def test_bulk_runtime_bridge_rejects_unsupported_ingredient_family() -> None:
    laplacian = next(
        capability
        for capability in available_capabilities()
        if "laplacian" in capability.required_ingredients
    )
    with pytest.raises(UnsupportedXC, match="unsupported: laplacian"):
        build_bulk_runtime_program(laplacian.name)


def test_bulk_runtime_positive_interior_validator_is_fail_closed() -> None:
    program = build_bulk_runtime_program("GGA_X_PBE_SOL", order=1)
    valid = np.ones((len(program.spec.features), 2), dtype=np.float64)
    if program.spec.spin == "polarized":
        sigma_ab = program.spec.features.index("sigma_ab")
        valid[sigma_ab] = 0.0
    assert program.evaluate(valid).shape == (len(program.outputs), 2)

    invalid = valid.copy()
    invalid[0, 0] = 0.0
    with pytest.raises(UnsupportedXC, match="strictly positive density"):
        program.evaluate(invalid)
