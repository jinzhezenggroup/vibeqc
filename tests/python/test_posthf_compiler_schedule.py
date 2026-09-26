"""Compiler-owned generic post-HF source-reuse scheduling."""

import pytest
from vibeqc_compiler.common.source_reuse import (
    SourceReuseRequest,
    ordered_source_reuse_plan,
    uniform_source_reuse_plan,
)


def test_uniform_source_reuse_generalizes_paired_mp2_requests() -> None:
    sequential = uniform_source_reuse_plan(1, 2, 99)
    assert sequential.shared_scan is False
    assert sequential.jobs_per_batch == sequential.provider_requests == 1

    shared = uniform_source_reuse_plan(10, 2, 3)
    assert shared.shared_scan is True
    assert shared.jobs_per_batch == 3
    assert shared.provider_requests == 6


def test_ordered_source_reuse_minimizes_scans_under_retained_outputs() -> None:
    requests = [
        SourceReuseRequest(live_bytes=30, retained_bytes=10),
        SourceReuseRequest(live_bytes=40, retained_bytes=20),
        SourceReuseRequest(live_bytes=50, retained_bytes=30),
    ]
    plan = ordered_source_reuse_plan(
        common_bytes=20,
        fixed_live_bytes=10,
        budget_bytes=105,
        requests=requests,
    )
    assert [(batch.begin, batch.end) for batch in plan.batches] == [(0, 2), (2, 3)]
    assert plan.source_scans == 2
    assert plan.peak_bytes == 100


def test_ordered_source_reuse_rejects_request_that_cannot_fit() -> None:
    with pytest.raises(ValueError, match="exceeds numeric memory budget"):
        ordered_source_reuse_plan(
            common_bytes=20,
            fixed_live_bytes=40,
            budget_bytes=100,
            requests=[SourceReuseRequest(live_bytes=50, retained_bytes=10)],
        )


def test_ordered_source_reuse_requires_live_bytes_to_cover_output() -> None:
    with pytest.raises(ValueError, match="must include its retained output"):
        SourceReuseRequest(live_bytes=7, retained_bytes=8)
