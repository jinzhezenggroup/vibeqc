"""Shared production-domain control evidence for automatic Libxc admission."""

from __future__ import annotations

import pytest
from vibeqc_compiler.xc.production_domain_controls import (
    CONTROL_SEMANTICS,
    run_control_case,
)


@pytest.mark.parametrize(
    "name",
    ("LDA_C_VWN_4", "GGA_X_PBE_SOL", "MGGA_X_R2SCAN01"),
)
@pytest.mark.parametrize("spin", ("polarized", "unpolarized"))
@pytest.mark.parametrize(
    "case_id",
    ("control/lazy-inactive-branch", "control/invalid-nonfinite"),
)
def test_shared_control_cases_produce_exact_pass_rows(
    name: str, spin: str, case_id: str
) -> None:
    row, detail = run_control_case(name, spin=spin, case_id=case_id)

    assert row == {
        "spin": spin,
        "case_id": case_id,
        "status": "pass",
        "outputs": ["energy", "vxc"],
        "reason": None,
    }
    assert detail["status"] == "pass"
    assert detail["control_semantics"] == CONTROL_SEMANTICS


def test_lazy_control_covers_second_derivative_and_both_emitters() -> None:
    _, detail = run_control_case(
        "MGGA_X_R2SCAN01",
        spin="polarized",
        case_id="control/lazy-inactive-branch",
    )

    assert detail["scalar"] == [0.0, 0.0, 2.0]
    assert detail["checks"] == {
        "scalar_exact": True,
        "array_value": True,
        "array_first": True,
        "array_second": True,
        "scalar_emits_branch": True,
        "cuda_emits_branch": True,
        "singular_branch_retained": True,
    }


@pytest.mark.parametrize("spin", ("polarized", "unpolarized"))
def test_nonfinite_control_rejects_every_feature_before_graph_math(spin: str) -> None:
    _, detail = run_control_case(
        "GGA_X_PBE_SOL",
        spin=spin,
        case_id="control/invalid-nonfinite",
    )

    assert detail["rejected"] == detail["expected"]
    assert detail["expected"] == 3 * len(detail["features"])


def test_control_runner_rejects_noncontrol_or_cross_spin_cases() -> None:
    with pytest.raises(ValueError, match="unsupported production-domain control"):
        run_control_case(
            "GGA_X_PBE_SOL",
            spin="polarized",
            case_id="density/near-zero",
        )

    with pytest.raises(ValueError, match="outside the exact profile"):
        run_control_case(
            "GGA_X_PBE_SOL",
            spin="unpolarized",
            case_id="spin/zero-a",
        )
