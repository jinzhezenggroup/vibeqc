"""The common DF adapter uses the production tile planner without a solve."""

import pytest
from vibeqc import Calculator
from vibeqc.resources_df import density_fitting_tile_plan


@pytest.mark.parametrize("generated,full_bytes", [(False, 45394), (True, 41298)])
@pytest.mark.parametrize("dense_policy", [None, "dense"])
def test_exchange_reservation_preserves_original_dense_boundaries(
    monkeypatch, generated, full_bytes, dense_policy
):
    """Thresholds were checked against the pre-occupied merged native library."""
    library = Calculator()._library
    if dense_policy is None:
        monkeypatch.delenv("VIBEQC_DF_EXCHANGE", raising=False)
    else:
        monkeypatch.setenv("VIBEQC_DF_EXCHANGE", dense_policy)

    def query(budget):
        return density_fitting_tile_plan(
            library,
            1,
            8,
            8,
            2,
            budget_bytes=budget,
            fixed_device_bytes=0,
            generated_source=generated,
        )

    dense = query(full_bytes)
    assert dense.stores_full_three_center
    assert dense.peak_workspace_bytes == full_bytes
    assert query(25170).peak_workspace_bytes == 25170
    with pytest.raises(ValueError, match="cannot hold the metric"):
        query(25169)

    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "occupied")
    # Two full 8x8 factors, two generation words and an error word per item.
    reserve = 2 * 8 * 8 * 8 + 12
    assert query(0).peak_workspace_bytes == full_bytes + reserve
    assert not query(full_bytes).stores_full_three_center
    assert query(full_bytes + reserve).stores_full_three_center
    with pytest.raises(ValueError, match="cannot hold the metric"):
        query(25170)
    assert query(25170 + reserve).peak_workspace_bytes == 25170 + reserve


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
