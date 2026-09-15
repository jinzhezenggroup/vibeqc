"""The common DF adapter uses the production tile planner without a solve."""

import pytest
from vibeqc import Calculator
from vibeqc.resources_df import density_fitting_tile_plan


@pytest.mark.parametrize(
    "shape",
    [(1, 768, 768, 160), (1, 384, 384, 80), (2, 768, 768, 160), (1, 768, 767, 160)],
)
def test_auto_reserves_only_the_potential_measured_domain(monkeypatch, shape):
    """Reservation precedes device/rank admission and never removes dense capacity."""
    library = Calculator()._library
    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "dense")
    dense = density_fitting_tile_plan(
        library, *shape, budget_bytes=0, fixed_device_bytes=0
    )
    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "auto")
    automatic = density_fitting_tile_plan(
        library, *shape, budget_bytes=0, fixed_device_bytes=0
    )
    expected = 2 * 768 * 768 * 8 + 12 if shape[:3] == (1, 768, 768) else 0
    assert automatic.peak_workspace_bytes - dense.peak_workspace_bytes == expected


@pytest.mark.parametrize("generated,full_bytes", [(False, 1104943), (True, 1100847)])
@pytest.mark.parametrize("dense_policy", [None, "dense"])
def test_exchange_reservation_preserves_full_scratch_and_minimum_boundaries(
    monkeypatch, generated, full_bytes, dense_policy
):
    """All solver owners shift dense/occupied boundaries equally.

    The original 8-AO dense bounds were 45394/41298 bytes. This slice adds
    a serialized 3-matrix AO frame and admitted provider workspace (1058373
    bytes) and both spin final frames (1176 bytes), while the occupied-factor
    differential remains unchanged. Metric and compact solvers now also have
    checked, separate workspace allowances, including their fixed floors.
    """
    library = Calculator()._library
    solver_reserve = 2 * (1 << 20) + 16 * 8 * 8 * 8
    full_bytes += solver_reserve
    minimum_bytes = 1084719 + solver_reserve
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
    assert query(minimum_bytes).peak_workspace_bytes == minimum_bytes
    with pytest.raises(ValueError, match="cannot hold the metric"):
        query(minimum_bytes - 1)

    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "occupied")
    # Two full 8x8 factors, two generation words and an error word per item.
    reserve = 2 * 8 * 8 * 8 + 12
    assert query(0).peak_workspace_bytes == full_bytes + reserve
    constrained = query(full_bytes)
    if generated:
        # Occupied factors can coexist with retained B by reducing K scratch Q.
        assert constrained.stores_full_three_center
        assert constrained.auxiliary_tile < 8
    else:
        assert not constrained.stores_full_three_center
    assert query(full_bytes + reserve).stores_full_three_center
    with pytest.raises(ValueError, match="cannot hold the metric"):
        query(minimum_bytes)
    assert (
        query(minimum_bytes + reserve).peak_workspace_bytes == minimum_bytes + reserve
    )


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


def test_overlap_storage_is_reserved_in_every_cuda_df_candidate():
    """Shape-only admission must charge retained S/X/coordinates for each item."""
    import json

    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    request = Calculator(device="cuda", density_fitting="cuda")._resource_request(
        [atoms] * 4
    )
    # A CPU-only build deliberately publishes no CUDA candidates. Its pure
    # common shape planner is already covered by the tests above.
    if not request.candidates:
        pytest.skip("requires a CUDA-enabled library, no GPU execution")
    for candidate in request.candidates:
        inventory = json.loads(dict(candidate.decisions)["bucket_inventory"])
        for row in inventory:
            expected = 8 * row["batch"] * (2 * row["nbf"] ** 2 + row["coordinates"])
            assert row["overlap_cache_host_bytes"] == expected
            assert row["resident_host_bytes"] >= expected


def test_diis_reservation_precedes_retained_panel_selection():
    """History growth must consume capacity before a retained-B plan picks Q."""
    from vibeqc.resources_df import (
        density_fitting_diis_bytes,
        density_fitting_tile_plan,
    )

    library = Calculator()._library
    assert density_fitting_diis_bytes(library, 4, 768, 0) == 0
    assert density_fitting_diis_bytes(library, 4, 768, 1) == 0
    plans = []
    for history in (2, 8, 12):
        fixed = density_fitting_diis_bytes(library, 1, 768, history) + (1 << 20)
        plan = density_fitting_tile_plan(
            library,
            1,
            768,
            768,
            160,
            budget_bytes=4 << 30,
            fixed_device_bytes=fixed,
            generated_source=True,
        )
        assert plan.peak_workspace_bytes <= 4 << 30
        plans.append(plan)
    assert plans[0].stores_full_three_center and plans[1].stores_full_three_center
    assert plans[0].auxiliary_tile > plans[1].auxiliary_tile
    # The largest history crosses the full-B boundary at this allowance;
    # retained-Q monotonicity does not apply after switching to regeneration.
    assert not plans[2].stores_full_three_center
    # Keep the dimension inside the Python ABI range so native multiplication
    # overflow, rather than argument-range validation, rejects the request.
    with pytest.raises(ValueError, match="overflow"):
        density_fitting_diis_bytes(library, 1, 1 << 32, 8)
