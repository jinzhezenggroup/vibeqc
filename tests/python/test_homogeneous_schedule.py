"""Shared compiler scheduling for bounded homogeneous runtime ranges."""

import pytest
from vibeqc_compiler.common.gpu_profitability import GpuProfitability
from vibeqc_compiler.common.homogeneous_schedule import (
    HomogeneousExecution,
    HomogeneousTaskRange,
    HomogeneousTaskSchedule,
)


def test_packet_schedule_is_bounded_stable_and_identified() -> None:
    schedule = HomogeneousTaskSchedule(
        2,
        HomogeneousExecution.ORDINARY,
        GpuProfitability(launch_count=1, shared_bytes=2048),
        "sm_120",
    )
    ranges = (
        HomogeneousTaskRange((0, 1, 0), (2, 3, 2), 40, 12),
        HomogeneousTaskRange((0, 1, 0), (1, 1, 1), 80, 2),
        HomogeneousTaskRange((0, 1, 0), (3, 2, 1), 20, 12),
    )
    packets = schedule.packets(ranges)
    assert tuple(row.signature for row in packets[0]) == ((2, 3, 2), (3, 2, 1))
    assert tuple(row.signature for row in packets[1]) == ((1, 1, 1),)
    assert len(schedule.identity) == 64
    assert schedule.to_payload()["profitability"]["compiled"]["shared_bytes"] == 2048


def test_same_schedule_ir_serves_non_df_integral_ranges() -> None:
    schedule = HomogeneousTaskSchedule(3, HomogeneousExecution.COOPERATIVE)
    four_center = (
        HomogeneousTaskRange((2, 1, 1, 0), (3, 2, 2, 1), 64, 36),
        HomogeneousTaskRange((2, 1, 1, 0), (2, 2, 1, 1), 128, 16),
    )
    assert schedule.packets(four_center) == (four_center,)
    assert schedule.to_payload()["execution"] == "cooperative"


@pytest.mark.parametrize("capacity", [0, -1])
def test_invalid_packet_capacity_fails_closed(capacity: int) -> None:
    with pytest.raises(ValueError):
        HomogeneousTaskSchedule(capacity)


def test_df_signature_packet_policy_is_compiler_owned() -> None:
    from vibeqc_compiler.integral.df_shell_derivatives import (
        DF_SIGNATURE_PACKET_SCHEDULE,
        emit_df_shell_derivatives_cuda,
    )

    source = emit_df_shell_derivatives_cuda(classes=((0, 0, 0),))
    assert (
        f"signature_packet_capacity={DF_SIGNATURE_PACKET_SCHEDULE.packet_capacity};"
        in source
    )
    assert 'signature_packet_execution="ordinary";' in source
