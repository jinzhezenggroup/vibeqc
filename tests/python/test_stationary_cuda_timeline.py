from __future__ import annotations

import pytest
from vibeqc._stationary_cuda import _ExclusiveWallTimeline


def test_exclusive_stationary_cuda_timeline_reconciles_without_overlap() -> None:
    ticks = iter((0.0, 1.0, 3.0, 6.0, 10.0))
    timeline = _ExclusiveWallTimeline(clock=lambda: next(ticks))

    timeline.switch("artifact_lookup_compile")
    with timeline.phase("tensorir_weight_execution"):
        pass
    result = timeline.finish()

    assert result["schema"] == "vibeqc.stationary-cuda-exclusive-wall.v1"
    assert result["exclusive_wall_seconds"] == {
        "artifact_lookup_compile": 6.0,
        "preparation": 1.0,
        "tensorir_weight_execution": 3.0,
    }
    assert result["reconciled_seconds"] == pytest.approx(10.0)
    assert result["endpoint_seconds"] == pytest.approx(10.0)
    assert result["reconciliation_error_seconds"] == pytest.approx(0.0)
