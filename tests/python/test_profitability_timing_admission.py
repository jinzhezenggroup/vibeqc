"""Unavailable timings are unknown; zero-duration measurements are invalid."""

import pytest
from vibeqc_compiler.common.gpu_profitability import GpuProfitability


@pytest.mark.parametrize("candidate,baseline", ((0.0, 1.0), (1.0, 0.0), (0.0, 0.0)))
def test_resource_gate_rejects_zero_endpoint_timings(
    candidate: float, baseline: float
) -> None:
    cost = GpuProfitability(
        endpoint_seconds=candidate, spill_load_bytes=8, spill_store_bytes=8
    )
    peer = GpuProfitability(
        endpoint_seconds=baseline, spill_load_bytes=0, spill_store_bytes=0
    )
    with pytest.raises(ValueError, match="positive"):
        cost.resource_regressions_against(peer)


def test_missing_endpoint_timing_remains_unknown() -> None:
    assert (
        GpuProfitability().resource_regressions_against(
            GpuProfitability(endpoint_seconds=1.0)
        )
        == ()
    )
