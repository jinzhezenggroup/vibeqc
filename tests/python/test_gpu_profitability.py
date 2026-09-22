"""Shared GPU profitability records and ordering contracts."""

import pytest
from vibeqc_compiler.common.gpu_profitability import (
    GpuProfitability,
    scalar_reduction_promotion_rejection,
)


def test_static_priority_rejects_pressure_growth_without_a_traffic_win() -> None:
    retained_fusion = GpuProfitability(
        semantic_traffic_bytes=4096,
        estimated_registers_per_thread=96,
        estimated_occupancy_upper_bound=0.25,
        launch_count=1,
        source_bytes=900,
        arithmetic_operation_count=90,
        peak_live_values=40,
    )
    rematerialized = GpuProfitability(
        semantic_traffic_bytes=4096,
        estimated_registers_per_thread=48,
        estimated_occupancy_upper_bound=0.5,
        launch_count=2,
        source_bytes=1100,
        arithmetic_operation_count=100,
        peak_live_values=20,
        rematerialized_value_count=4,
    )

    assert rematerialized.static_compile_priority(
        1
    ) < retained_fusion.static_compile_priority(0)


def test_precision_overheads_break_static_ties_without_hiding_total_traffic() -> None:
    strict = GpuProfitability(
        semantic_traffic_bytes=4096,
        precision_cast_read_bytes=0,
        precision_cast_write_bytes=0,
        precision_cast_simultaneous_bytes=0,
        precision_widened_accumulation_terms=0,
    )
    mixed = GpuProfitability(
        semantic_traffic_bytes=4096,
        precision_cast_read_bytes=256,
        precision_cast_write_bytes=128,
        precision_cast_simultaneous_bytes=96,
        precision_widened_accumulation_terms=1024,
    )

    assert strict.precision_cast_bytes == 0
    assert mixed.precision_cast_bytes == 384
    assert strict.static_compile_priority(1) < mixed.static_compile_priority(0)


def test_compiled_priority_never_rewards_spills_for_a_smaller_artifact() -> None:
    healthy = GpuProfitability(
        compiled_registers_per_thread=48,
        spill_store_bytes=0,
        spill_load_bytes=0,
        local_bytes=0,
        shared_bytes=1024,
        compiled_occupancy_upper_bound=0.75,
        source_bytes=1200,
        object_bytes=800,
        compile_seconds=1.1,
        endpoint_seconds=0.01005,
    )
    spilled = GpuProfitability(
        compiled_registers_per_thread=40,
        spill_store_bytes=8,
        spill_load_bytes=8,
        local_bytes=64,
        shared_bytes=0,
        compiled_occupancy_upper_bound=0.8,
        source_bytes=700,
        object_bytes=500,
        compile_seconds=0.8,
        endpoint_seconds=0.01000,
    )

    assert healthy.compiled_resource_priority() < spilled.compiled_resource_priority()


def test_endpoint_profitability_rejects_a_slower_candidate() -> None:
    baseline = GpuProfitability(endpoint_seconds=0.010)
    slower = GpuProfitability(endpoint_seconds=0.012)

    reasons = slower.endpoint_regressions_against(baseline, minimum_speedup=1.02)

    assert reasons == ("endpoint speedup 0.833333x is below the required 1.02x",)


def test_endpoint_profitability_allows_a_qualified_speedup() -> None:
    baseline = GpuProfitability(endpoint_seconds=0.010)
    faster = GpuProfitability(endpoint_seconds=0.009)

    assert faster.endpoint_regressions_against(baseline, minimum_speedup=1.02) == ()


def test_resource_regression_gate_rejects_pressure_growth_inside_noise_band() -> None:
    baseline = GpuProfitability(
        compiled_registers_per_thread=64,
        spill_store_bytes=0,
        spill_load_bytes=0,
        compiled_occupancy_upper_bound=0.75,
        endpoint_seconds=0.01000,
    )
    retained = GpuProfitability(
        compiled_registers_per_thread=96,
        spill_store_bytes=8,
        spill_load_bytes=8,
        compiled_occupancy_upper_bound=0.50,
        endpoint_seconds=0.00995,
    )

    reasons = retained.resource_regressions_against(baseline)

    assert len(reasons) == 2
    assert reasons[0].startswith("spill traffic grows from 0 to 16 bytes")
    assert "registers grow from 64 to 96 per thread" in reasons[1]
    assert "occupancy falls from 0.75 to 0.5" in reasons[1]


def test_resource_regression_gate_allows_measured_endpoint_win() -> None:
    baseline = GpuProfitability(
        compiled_registers_per_thread=64,
        spill_store_bytes=0,
        spill_load_bytes=0,
        compiled_occupancy_upper_bound=0.75,
        endpoint_seconds=0.01000,
    )
    faster = GpuProfitability(
        compiled_registers_per_thread=96,
        spill_store_bytes=8,
        spill_load_bytes=8,
        compiled_occupancy_upper_bound=0.50,
        endpoint_seconds=0.00950,
    )

    assert faster.resource_regressions_against(baseline) == ()


def test_payload_preserves_unknown_evidence_instead_of_guessing() -> None:
    cost = GpuProfitability(
        arithmetic_operation_count=12,
        peak_live_values=5,
        rematerialized_value_count=2,
    )
    payload = cost.to_payload()

    assert payload["schema"] == "vibeqc.compiler.gpu-profitability.v1"
    assert payload["static"]["peak_live_values"] == 5
    assert payload["static"]["precision_cast_read_bytes"] is None
    assert payload["static"]["precision_widened_accumulation_terms"] is None
    assert payload["compiled"]["compiled_registers_per_thread"] is None
    assert payload["endpoint_seconds"] is None


@pytest.mark.parametrize(
    "options",
    [
        {"source_bytes": -1},
        {"estimated_occupancy_upper_bound": 1.01},
        {"compile_seconds": float("nan")},
        {"spill_store_bytes": True},
        {"precision_cast_read_bytes": -1},
        {"precision_widened_accumulation_terms": True},
    ],
)
def test_invalid_profitability_evidence_fails_closed(
    options: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        GpuProfitability(**options)


def test_precision_fields_preserve_positional_compiled_registers() -> None:
    facts = GpuProfitability(None, None, None, None, None, None, None, None, 64)
    assert facts.compiled_registers_per_thread == 64
    assert facts.precision_cast_read_bytes is None
    assert facts.precision_cast_write_bytes is None
    assert facts.precision_cast_simultaneous_bytes is None
    assert facts.precision_widened_accumulation_terms is None


def test_pathological_scalar_reduction_requires_a_concrete_parallel_alternative() -> (
    None
):
    assert (
        scalar_reduction_promotion_rejection(
            output_elements=1,
            reduction_elements=4096,
            parallel_width=32,
            alternative="GEMM",
        )
        == "scalar reduction exposes 1 independent output element(s) for reduction extent 4096; legal GEMM lowering exists"
    )
    assert (
        scalar_reduction_promotion_rejection(
            output_elements=1,
            reduction_elements=4096,
            parallel_width=32,
            alternative=None,
        )
        is None
    )


def test_small_or_already_parallel_reductions_remain_promotion_eligible() -> None:
    assert (
        scalar_reduction_promotion_rejection(
            output_elements=1,
            reduction_elements=127,
            parallel_width=32,
            alternative="cooperative-reduction",
        )
        is None
    )
    assert (
        scalar_reduction_promotion_rejection(
            output_elements=32,
            reduction_elements=4096,
            parallel_width=32,
            alternative="cooperative-reduction",
        )
        is None
    )


@pytest.mark.parametrize(
    "options",
    [
        {"output_elements": -1, "reduction_elements": 1, "parallel_width": 32},
        {"output_elements": 1, "reduction_elements": -1, "parallel_width": 32},
        {"output_elements": 1, "reduction_elements": 1, "parallel_width": 0},
    ],
)
def test_scalar_reduction_diagnostic_rejects_invalid_counts(
    options: dict[str, int],
) -> None:
    with pytest.raises(ValueError):
        scalar_reduction_promotion_rejection(**options, alternative="parallel")
