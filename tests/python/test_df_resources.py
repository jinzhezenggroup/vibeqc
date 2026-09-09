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
