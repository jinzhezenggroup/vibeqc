"""Automatic CUDA DF routing must remain on the bounded source-backed path."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_automatic_df_does_not_promote_materialized_resident_owner() -> None:
    source = (ROOT / "src/scf/rhf.cpp").read_text()
    assert "automatic_dense_resident_df_owner" not in source
    assert "preferred_automatic_resident_df_value_peak" not in source

    start = source.index(
        "[[maybe_unused]] DensityFittingScfData prepare_density_fitting_data("
    )
    end = source.index("Matrix build_density_fitting_rhf_fock(", start)
    preparation = source[start:end]
    assert "if (data.resolved_budget.value_bytes != 0U ||" in preparation
    assert "assemble_density_fitting_metadata" in preparation

    start = source.index("CudaDensityFittingPlanPtr make_cuda_density_fitting_plan(")
    end = source.index("ScfResult run_cuda_independent_fock_strategy(", start)
    planner = source[start:end]
    assert "const auto planning_budget = data.resolved_budget.value_bytes;" in planner
    assert (
        "set_cuda_density_fitting_scf_value_budget(owned_plan.get(), planning_budget)"
        in planner
    )


def test_auto_budget_has_no_provider_specific_peak_override() -> None:
    header = (ROOT / "src/scf/df_preparation_budget.hpp").read_text()
    assert "preferred_value_peak_bytes" not in header
    assert (
        "const auto workload_target = "
        "std::max(df_budget_bytes(value_demand + response_demand), min_auto);" in header
    )
