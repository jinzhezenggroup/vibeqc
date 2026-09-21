"""Shared GPU profitability records and ordering contracts."""

import pytest
from vibeqc_compiler.common.gpu_profitability import GpuProfitability


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


def test_payload_preserves_unknown_evidence_instead_of_guessing() -> None:
    cost = GpuProfitability(
        arithmetic_operation_count=12,
        peak_live_values=5,
        rematerialized_value_count=2,
    )
    payload = cost.to_payload()

    assert payload["schema"] == "vibeqc.compiler.gpu-profitability.v1"
    assert payload["static"]["peak_live_values"] == 5
    assert payload["compiled"]["compiled_registers_per_thread"] is None
    assert payload["endpoint_seconds"] is None


@pytest.mark.parametrize(
    "options",
    [
        {"source_bytes": -1},
        {"estimated_occupancy_upper_bound": 1.01},
        {"compile_seconds": float("nan")},
        {"spill_store_bytes": True},
    ],
)
def test_invalid_profitability_evidence_fails_closed(
    options: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        GpuProfitability(**options)
