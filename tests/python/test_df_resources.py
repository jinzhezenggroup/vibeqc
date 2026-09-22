"""The common DF adapter uses the production tile planner without a solve."""

import typing

import pytest
from vibeqc import Calculator
from vibeqc.resources_df import density_fitting_tile_plan, density_fitting_value_layout


def test_df_value_layout_contract_is_explicit_without_native_runtime() -> None:
    dense = density_fitting_value_layout(4, 7, "dense")
    packed = density_fitting_value_layout(4, 7, "packed")

    assert dense.storage_elements == 4 * 4 * 7
    assert packed.storage_elements == 10 * 7
    assert packed.dense_elements == dense.storage_elements
    assert packed.identity != dense.identity
    assert packed.to_payload()["triangle"] == "lower"
    with pytest.raises(ValueError, match="dense or packed"):
        density_fitting_value_layout(4, 7, "automatic")


def test_packed_query_preserves_complete_u_when_budget_shrinks(
    monkeypatch: typing.Any,
) -> None:
    """Query native allocation capacities without touching a GPU or tensor."""
    library = Calculator()._library

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("packed shape query created a CUDA context")

    monkeypatch.setattr(library, "vibeqc_context_create", forbidden)

    def query(
        rank: typing.Any = 160, budget: typing.Any = 0, fixed: typing.Any = 4096
    ) -> typing.Any:
        return density_fitting_tile_plan(
            library,
            1,
            768,
            768,
            rank,
            budget_bytes=budget,
            fixed_device_bytes=fixed,
            generated_source=True,
            pair_storage="packed",
        )

    default = query()
    constrained = query(budget=default.peak_workspace_bytes - 1)
    assert constrained.auxiliary_tile < default.auxiliary_tile
    assert constrained.rank_capacity == default.rank_capacity == 160
    assert constrained.projection_capacity_elements >= 768 * 160 * 768
    for plan in (default, constrained, query(rank=0)):
        assert plan.pair_storage == "packed" and plan.stores_full_three_center
        assert (
            plan.raw_factor_bytes
            == plan.stored_factor_bytes
            == 8 * 768 * 769 // 2 * 768
        )
        assert plan.contraction_scratch_bytes == 8 * (
            plan.projection_capacity_elements + 2 * plan.panel_capacity_elements
        )
        assert plan.panel_capacity_elements == 768 * 768 * plan.auxiliary_tile
    assert query(fixed=8192).peak_workspace_bytes - default.peak_workspace_bytes == 4096
    with pytest.raises(ValueError):
        query(budget=1)
    with pytest.raises(ValueError):
        query(rank=769)
    with pytest.raises(ValueError, match="physical generated source"):
        density_fitting_tile_plan(
            library,
            1,
            4,
            5,
            1,
            budget_bytes=0,
            fixed_device_bytes=0,
            pair_storage="packed",
        )


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((1, 768, 768, 160), 2 * 768 * 768 * 8 + 12),
        ((1, 384, 384, 80), 2 * 384 * 384 * 8 + 12),
        ((2, 768, 768, 160), 0),
        ((1, 768, 767, 160), 2 * 768 * 768 * 8 + 12),
        ((2, 384, 384, 80), 0),
        ((1, 384, 383, 80), 2 * 384 * 384 * 8 + 12),
        ((1, 192, 192, 40), 2 * 192 * 192 * 8 + 12),
        ((1, 512, 700, 100), 2 * 512 * 512 * 8 + 12),
        ((1, 8, 13, 8), 0),
        ((1, 1, 1, 1), 0),
    ],
)
def test_auto_reserves_only_authorized_rhf_work_policy_candidates(
    monkeypatch: typing.Any, shape: typing.Any, expected: typing.Any
) -> None:
    """Only a known profitable RHF occupation can charge automatic factor storage."""
    library = Calculator()._library
    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "dense")
    dense = density_fitting_tile_plan(
        library,
        *shape,
        budget_bytes=1 << 40,
        fixed_device_bytes=0,
        rhf_occupied=shape[-1],
    )
    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "auto")
    automatic = density_fitting_tile_plan(
        library,
        *shape,
        budget_bytes=1 << 40,
        fixed_device_bytes=0,
        rhf_occupied=shape[-1],
    )
    # Native and common planners receive the same method-authorized rank.
    assert automatic.peak_workspace_bytes - dense.peak_workspace_bytes == expected


@pytest.mark.parametrize("generated", [False, True])
@pytest.mark.parametrize("rhf_rank", [None, 0, 5, 8])
def test_ineligible_auto_keeps_dense_budget_and_tiles(
    monkeypatch: typing.Any, generated: typing.Any, rhf_rank: typing.Any
) -> None:
    """UHF/unknown, empty and high-rank jobs spend no unused factor reservation."""
    library = Calculator()._library

    def query(budget: typing.Any) -> typing.Any:
        return density_fitting_tile_plan(
            library,
            1,
            8,
            8,
            5,
            budget_bytes=budget,
            fixed_device_bytes=0,
            generated_source=generated,
            rhf_occupied=rhf_rank,
        )

    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "dense")
    budget = query(0).peak_workspace_bytes
    dense = query(budget)
    assert dense.stores_full_three_center
    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "auto")
    assert query(budget) == dense
    monkeypatch.delenv("VIBEQC_DF_EXCHANGE")
    assert query(budget) == dense


def test_optional_auto_factors_do_not_force_streaming(
    monkeypatch: typing.Any,
) -> None:
    """Even eligible RHF retains its dense tiles when optional factors do not fit."""
    library = Calculator()._library

    def query(budget: typing.Any) -> typing.Any:
        return density_fitting_tile_plan(
            library,
            1,
            8,
            8,
            2,
            budget_bytes=budget,
            fixed_device_bytes=0,
            rhf_occupied=2,
        )

    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "dense")
    budget = query(0).peak_workspace_bytes
    dense = query(budget)
    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "auto")
    assert query(budget) == dense
    reserve = 2 * 8 * 8 * 8 + 12
    admitted = query(budget + reserve)
    assert admitted.stores_full_three_center
    assert admitted.peak_workspace_bytes == budget + reserve


@pytest.mark.parametrize("generated,full_bytes", [(False, 1104943), (True, 1100847)])
@pytest.mark.parametrize("occupied_policy", ["occupied"])
def test_exchange_reservation_preserves_full_scratch_and_minimum_boundaries(
    monkeypatch: typing.Any,
    generated: typing.Any,
    full_bytes: typing.Any,
    occupied_policy: typing.Any,
) -> None:
    """All solver owners shift dense/occupied boundaries equally.

    The original 8-AO dense bounds were 45394/41298 bytes. This slice adds
    a serialized 3-matrix AO frame and admitted provider workspace (1058373
    bytes) and both spin final frames (1176 bytes), while the occupied-factor
    differential remains unchanged. Metric and compact solvers now also have
    checked, separate workspace allowances, including their fixed floors.
    Final validation reserves nine serialized matrices, a spectrum and bounded
    reduction storage equally for both exchange policies.
    """
    library = Calculator()._library
    solver_reserve = 2 * (1 << 20) + 16 * 8 * 8 * 8
    solver_reserve += (9 * 8 * 8 + 8) * 8 + (3 * 1 + 1) * 128
    full_bytes += solver_reserve
    minimum_bytes = 1084719 + solver_reserve
    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "dense")

    def query(budget: typing.Any) -> typing.Any:
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

    if occupied_policy is None:
        monkeypatch.delenv("VIBEQC_DF_EXCHANGE", raising=False)
    else:
        monkeypatch.setenv("VIBEQC_DF_EXCHANGE", occupied_policy)
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


def test_df_shape_query_composes_fixed_reservation_and_native_tile_shrinking() -> None:
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


def test_df_shape_query_needs_no_integrals_or_context(
    monkeypatch: typing.Any,
) -> None:
    import numpy as np

    library = Calculator()._library

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
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


def test_generated_residency_uses_complete_source_specific_budget() -> None:
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


def test_generated_source_auto_occupied_requires_complete_q_scratch(
    monkeypatch: typing.Any,
) -> None:
    """Automatic RHF factors are admitted only with the full source lease."""
    library = Calculator()._library
    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "dense")
    dense = density_fitting_tile_plan(
        library,
        1,
        768,
        768,
        160,
        budget_bytes=0,
        fixed_device_bytes=0,
        generated_source=True,
    )
    assert dense.auxiliary_tile == 128
    assert dense.automatic_rhf_rank == 0

    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "auto")
    full_budget = 1 << 40
    complete = density_fitting_tile_plan(
        library,
        1,
        768,
        768,
        160,
        budget_bytes=full_budget,
        fixed_device_bytes=0,
        generated_source=True,
        rhf_occupied=160,
    )
    assert complete.stores_full_three_center
    assert complete.ao_pair_tile == 768 * 768
    assert complete.auxiliary_tile == 768
    assert complete.automatic_rhf_rank == 160

    constrained = density_fitting_tile_plan(
        library,
        1,
        768,
        768,
        160,
        budget_bytes=complete.peak_workspace_bytes - 1,
        fixed_device_bytes=0,
        generated_source=True,
        rhf_occupied=160,
    )
    assert constrained.auxiliary_tile < 768
    assert constrained.automatic_rhf_rank == 0


def test_overlap_storage_is_reserved_in_every_cuda_df_candidate() -> None:
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


def test_packed_inventory_charges_both_owners_and_separates_identity(
    monkeypatch: typing.Any,
) -> None:
    """Representation switches change provenance before any CUDA allocation.

    Compare the explicit packed inventory with the same generated-source dense
    inventory. Every other persistent allocation is shared, so their exact
    difference exposes a missing immutable owner or scratch reservation.
    """
    import json

    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    calc = Calculator(device="cuda", density_fitting="cuda")
    monkeypatch.setenv("VIBEQC_DF_VALUE_STORAGE", "dense")
    dense = calc._resource_request([atoms] * 2)
    if not dense.candidates:
        pytest.skip("requires a CUDA-enabled library, no GPU execution")
    monkeypatch.setenv("VIBEQC_DF_VALUE_STORAGE", "packed")
    packed = calc._resource_request([atoms] * 2)
    assert packed.identity != dense.identity
    assert [candidate.name for candidate in packed.candidates] == ["cuda-df-packed"]
    candidate = packed.candidates[0]
    assert candidate.mode == "resident"
    decisions = dict(candidate.decisions)
    assert decisions["df_pair_storage"] == "packed"
    packed_layout_identities = json.loads(decisions["df_value_layout_identities"])
    assert packed_layout_identities
    dense_source = next(c for c in dense.candidates if c.name == "cuda-df-source")
    dense_layout_identities = json.loads(
        dict(dense_source.decisions)["df_value_layout_identities"]
    )
    assert dense_layout_identities
    assert packed_layout_identities != dense_layout_identities
    dense_rows = json.loads(dict(dense_source.decisions)["bucket_inventory"])
    packed_rows = json.loads(dict(candidate.decisions)["bucket_inventory"])
    for before, after in zip(dense_rows, packed_rows, strict=True):
        b, n, a = (after[key] for key in ("batch", "nbf", "naux"))
        for key in ("energy_tiles", "force_tiles"):
            plan = after[key]
            dense_plan = before[key]
            assert plan["pair_storage"] == "packed"
            assert plan["value_layout_identity"] != dense_plan["value_layout_identity"]
            assert plan["value_layout_elements_per_system"] == n * (n + 1) // 2 * a
            assert plan["dense_equivalent_elements_per_system"] == n * n * a
            assert dense_plan["value_layout_elements_per_system"] == n * n * a
            assert dense_plan["dense_equivalent_elements_per_system"] == n * n * a
            assert dense_plan["bounded_materialization_bytes"] == 0
            assert dense_plan["bounded_conversion_traffic_bytes"] == 0
            q = plan["auxiliary_tile"]
            assert plan["bounded_materialization_bytes"] == n * n * q * 8
            assert plan["bounded_conversion_traffic_bytes"] == 2 * n * n * q * 8
            assert (
                plan["raw_factor_bytes"]
                == plan["stored_factor_bytes"]
                == b * n * (n + 1) // 2 * a * 8
            )
        full = before["energy_tiles"]
        plan = after["energy_tiles"]
        assert full["stores_full_three_center"]
        dense_values = 8 * b * n * n * a + 24 * n * n * full["auxiliary_tile"]
        packed_values = (
            plan["raw_factor_bytes"]
            + plan["stored_factor_bytes"]
            + plan["contraction_scratch_bytes"]
        )
        assert after["resident_device_bytes"] - before["resident_device_bytes"] == (
            packed_values - dense_values
        )
    monkeypatch.setenv("VIBEQC_DF_VALUE_STORAGE", "invalid")
    with pytest.raises(ValueError, match="VIBEQC_DF_VALUE_STORAGE"):
        calc._resource_request([atoms])


def test_diis_reservation_precedes_retained_panel_selection() -> None:
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
