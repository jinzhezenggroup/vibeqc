"""Production grid accuracy/cost gate against an independently denser quadrature."""

import pytest

from benchmarks.grid_policy_convergence import qualify


def test_production_grid_profiles_converge_against_dense_pyscf() -> None:
    pytest.importorskip(
        "pyscf", reason="independent grid-convergence oracle requires PySCF"
    )
    result = qualify()
    assert result["passed"]
    assert result["profiles"]["pbe-standard"]["passed"]
    assert result["profiles"]["pbe-tight"]["passed"]
    assert result["accuracy_cost_checks"]["pbe_standard_improves_legacy48_force"]
