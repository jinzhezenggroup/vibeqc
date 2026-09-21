"""Compiler-owned MP2 source-reuse and residency scheduling."""

import pytest
from vibeqc_compiler.method.mp2_schedule import (
    conventional_reuse_plan,
    ri_mp2_residency_plan,
)


def test_conventional_reuse_batches_paired_consumers() -> None:
    sequential = conventional_reuse_plan(1, 99)
    assert sequential.shared_scan is False
    assert sequential.jobs_per_batch == sequential.provider_requests == 1

    shared = conventional_reuse_plan(10, 3)
    assert shared.shared_scan is True
    assert shared.jobs_per_batch == 3
    assert shared.provider_requests == 6


def test_ri_mp2_residency_matches_native_qualification_points() -> None:
    fixed = 2 << 20
    full = ri_mp2_residency_plan(fixed, 64 << 20, 120, 20, 100, 180)
    assert full.full_resident is True
    assert full.virtual_block == 100
    assert full.j_batch == 8
    assert full.peak_bytes <= 64 << 20

    blocked = ri_mp2_residency_plan(fixed, 8 << 20, 120, 20, 100, 180)
    assert blocked.full_resident is False
    assert blocked.virtual_block == 27
    assert blocked.j_batch == 4
    assert blocked.peak_bytes == 8_364_608


def test_ri_mp2_residency_rejects_impossible_budget() -> None:
    with pytest.raises(ValueError, match="exceeds numeric memory budget"):
        ri_mp2_residency_plan(2 << 20, 2 << 20, 120, 20, 100, 180)
