"""Production grid accuracy/cost gate against an independently denser quadrature."""

import pytest

from benchmarks.grid_policy_convergence import qualify


def test_production_grid_profiles_converge_against_dense_pyscf() -> None:
    pytest.importorskip(
        "pyscf", reason="independent grid-convergence oracle requires PySCF"
    )
    result = qualify()
    assert result["version"] == 2
    assert result["passed"]
    assert result["profiles"]["pbe-standard"]["passed"]
    assert result["profiles"]["pbe-tight"]["passed"]
    assert result["accuracy_cost_checks"]["pbe_standard_improves_radial48_force"]
    for family in ("lda", "pbe"):
        legacy = result["profiles"][f"{family}-legacy-v1"]["spec"]
        assert legacy["version"] == 1
        assert legacy["element_radii"] == ()
    radial = result["profiles"]["pbe-radial48"]["spec"]
    assert radial["version"] == 2
    assert radial["element_radii"]
