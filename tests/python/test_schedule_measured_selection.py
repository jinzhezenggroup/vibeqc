"""Shared measured schedule promotion contracts."""

from __future__ import annotations

import pytest
from vibeqc_compiler.common.gpu_profitability import GpuProfitability
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.schedule import (
    ScheduleContract,
    ScheduleResources,
    ScheduleTopology,
    select_measured_schedule_contract,
)
from vibeqc_compiler.integral.direct_fock_schedule import select_direct_fock_route


def _contract(
    name: str,
    seconds: float | None,
    *,
    fallback: bool = False,
    workload_hash: str = "1" * 64,
    registers: int = 32,
    occupancy: float = 0.75,
    spill_bytes: int = 0,
) -> ScheduleContract:
    return ScheduleContract(
        consumer="integral.direct-fock",
        schedule_hash=canonical_hash({"name": name}),
        workload_hash=workload_hash,
        profile_key="2" * 64,
        target_hash="3" * 64,
        precision_schedule_hash="4" * 64,
        fallback=fallback,
        topology=ScheduleTopology(
            materialization=name,
            residency="device",
            bucket="shell-class",
        ),
        resources=ScheduleResources(registers_per_thread=registers),
        profitability=GpuProfitability(
            compiled_registers_per_thread=registers,
            compiled_occupancy_upper_bound=occupancy,
            spill_store_bytes=spill_bytes,
            spill_load_bytes=0,
            endpoint_seconds=seconds,
        ),
    )


def test_measured_schedule_selection_fails_closed_without_candidate_timing() -> None:
    baseline = _contract("paged", 1.0, fallback=True)
    candidate = _contract("streaming", None)
    assert (
        select_measured_schedule_contract(
            (baseline, candidate),
            minimum_speedup=1.02,
        )
        is baseline
    )


def test_measured_schedule_selection_promotes_a_detectable_endpoint_win() -> None:
    baseline = _contract("paged", 1.0, fallback=True)
    candidate = _contract("streaming", 0.8)
    assert (
        select_measured_schedule_contract(
            (baseline, candidate),
            minimum_speedup=1.02,
        )
        is candidate
    )


def test_measured_schedule_selection_rejects_noise_and_resource_regression() -> None:
    baseline = _contract("paged", 1.0, fallback=True)
    noisy = _contract(
        "streaming",
        0.99,
        registers=64,
        occupancy=0.25,
        spill_bytes=1024,
    )
    assert (
        select_measured_schedule_contract(
            (baseline, noisy),
            minimum_speedup=1.0,
            endpoint_noise_fraction=0.02,
        )
        is baseline
    )


def test_measured_schedule_selection_rejects_cross_workload_comparison() -> None:
    baseline = _contract("paged", 1.0, fallback=True)
    other = _contract("streaming", 0.5, workload_hash="a" * 64)
    with pytest.raises(ValueError, match="share consumer/workload"):
        select_measured_schedule_contract(
            (baseline, other),
            minimum_speedup=1.02,
        )


def test_direct_fock_materialization_consumes_shared_schedule_selection() -> None:
    """The Direct-J/K adapter must not implement a second promotion policy."""

    common = {
        "shell_class": 1,
        "workload_hash": "1" * 64,
        "profile_key": "2" * 64,
        "target_hash": "3" * 64,
        "precision_schedule_hash": "4" * 64,
        "page_size": 8_388_608,
    }
    assert (
        select_direct_fock_route(
            paged=GpuProfitability(endpoint_seconds=1.0),
            streaming=GpuProfitability(endpoint_seconds=0.75),
            **common,
        )
        == "streaming"
    )
    assert (
        select_direct_fock_route(
            paged=GpuProfitability(endpoint_seconds=1.0),
            streaming=GpuProfitability(endpoint_seconds=None),
            **common,
        )
        == "paged"
    )
