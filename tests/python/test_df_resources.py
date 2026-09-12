"""The common DF adapter uses the production tile planner without a solve."""

import pytest
from vibeqc import Calculator
from vibeqc.resources_df import density_fitting_tile_plan


def test_df_shape_query_composes_fixed_reservation_and_native_tile_shrinking():
    library = Calculator()._library
    shape = (2, 20, 40, 5)
    default = density_fitting_tile_plan(
        library, *shape, budget_bytes=0, fixed_device_bytes=4096
    )
    assert default.batch_tile == 2
    assert default.stores_full_three_center
    lower = density_fitting_tile_plan(
        library,
        *shape,
        budget_bytes=default.peak_workspace_bytes - 1,
        fixed_device_bytes=4096,
    )
    assert lower.peak_workspace_bytes < default.peak_workspace_bytes
    assert not lower.stores_full_three_center
    reserved = density_fitting_tile_plan(
        library, *shape, budget_bytes=0, fixed_device_bytes=8192
    )
    assert reserved.peak_workspace_bytes - default.peak_workspace_bytes == 4096
    with pytest.raises(ValueError, match="cannot hold the metric"):
        density_fitting_tile_plan(
            library, *shape, budget_bytes=4096, fixed_device_bytes=4096
        )


def test_df_shape_query_needs_no_integrals_or_context(monkeypatch):
    import numpy as np

    library = Calculator()._library

    def forbidden(*args, **kwargs):
        pytest.fail("DF shape query allocated a numerical tensor or context")

    monkeypatch.setattr(library, "vibeqc_context_create", forbidden)
    monkeypatch.setattr(library, "vibeqc_batch_execute", forbidden)
    monkeypatch.setattr(np, "empty", forbidden)
    monkeypatch.setattr(np, "zeros", forbidden)
    plan = density_fitting_tile_plan(
        library, 1000, 1000, 1000, 200, budget_bytes=0, fixed_device_bytes=0
    )
    assert plan.peak_workspace_bytes > 10**9
    with pytest.raises(ValueError, match="overflows"):
        density_fitting_tile_plan(
            library, 1, 2**40, 20, 1, budget_bytes=0, fixed_device_bytes=0
        )


def test_generated_residency_uses_complete_source_specific_budget():
    library = Calculator()._library
    generated = density_fitting_tile_plan(
        library,
        1,
        192,
        192,
        40,
        budget_bytes=256 << 20,
        fixed_device_bytes=1 << 20,
        generated_source=True,
    )
    compatibility = density_fitting_tile_plan(
        library,
        1,
        192,
        192,
        40,
        budget_bytes=256 << 20,
        fixed_device_bytes=1 << 20,
    )
    assert generated.stores_full_three_center
    assert generated.peak_workspace_bytes <= 256 << 20
    # The independent host route also needs its raw upload during setup.
    assert not compatibility.stores_full_three_center
    constrained = density_fitting_tile_plan(
        library,
        1,
        192,
        192,
        40,
        budget_bytes=32 << 20,
        fixed_device_bytes=1 << 20,
        generated_source=True,
    )
    assert 8 * 192**3 > constrained.budget_bytes
    assert not constrained.stores_full_three_center
    assert constrained.peak_workspace_bytes <= constrained.budget_bytes
