"""Deterministic MP2 data-residency and source-reuse schedules.

Scientific MP2 equations remain in TensorIR/native consumers.  This module owns
only bounded execution choices that depend on dimensions and numeric capacity:
how many conventional MO requests share one AO source production, and whether
RI-MP2 retains the complete occupied-virtual B tensor or a bounded virtual block.

The generated native header is intentionally a pure transcription of these
choices.  It performs no CUDA probing and contains no molecule/GPU-name policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from vibeqc_compiler.common.resources import MAX_BYTES, checked_bytes


def _add(a: int, b: int) -> int:
    value = a + b
    if value > MAX_BYTES:
        raise OverflowError("MP2 schedule byte count overflow")
    return value


def _mul(a: int, b: int) -> int:
    value = a * b
    if value > MAX_BYTES:
        raise OverflowError("MP2 schedule byte count overflow")
    return value


@dataclass(frozen=True)
class ConventionalReusePlan:
    """Bounded group of MO requests sharing one immutable AO source traversal."""

    shared_scan: bool
    jobs_per_batch: int
    provider_requests: int


def conventional_reuse_plan(
    request_capacity: int, total_jobs: int
) -> ConventionalReusePlan:
    """Plan paired direct/exchange requests without changing job order."""

    checked_bytes(request_capacity, "MP2 provider request capacity")
    checked_bytes(total_jobs, "MP2 total jobs")
    if not total_jobs:
        raise ValueError("MP2 reuse schedule requires at least one job")
    if request_capacity < 2:
        return ConventionalReusePlan(False, 1, 1)
    jobs = min(request_capacity // 2, total_jobs)
    return ConventionalReusePlan(True, jobs, _mul(2, jobs))


@dataclass(frozen=True)
class RiMp2ResidencyPlan:
    """CUDA RI-MP2 B residency/blocking selected from a numeric byte budget."""

    virtual_block: int
    j_batch: int
    peak_bytes: int
    full_resident: bool


def _ri_block_capacity(
    fixed_bytes: int,
    nbf: int,
    occupied: int,
    auxiliaries: int,
    virtual_block: int,
    j_batch: int,
    *,
    full_resident: bool,
) -> int:
    total = fixed_bytes
    # AO->virtual temporary.
    total = _add(total, _mul(8, _mul(_mul(nbf, auxiliaries), virtual_block)))
    # First retained B[Q,i,a] block.
    b_elements = _mul(_mul(auxiliaries, occupied), virtual_block)
    total = _add(total, _mul(8, b_elements))
    if not full_resident:
        total = _add(total, _mul(8, b_elements))
    # Direct fitted-integral block, plus exchange scratch for blocked execution.
    integral_batch = _mul(_mul(j_batch, virtual_block), virtual_block)
    total = _add(total, _mul(8, integral_batch))
    if not full_resident:
        total = _add(total, _mul(8, integral_batch))
    return total


def ri_mp2_residency_plan(
    fixed_bytes: int,
    budget_bytes: int,
    nbf: int,
    occupied: int,
    virtuals: int,
    auxiliaries: int,
) -> RiMp2ResidencyPlan:
    """Prefer complete resident B, otherwise maximize one bounded virtual block."""

    for name, value in (
        ("fixed bytes", fixed_bytes),
        ("budget bytes", budget_bytes),
        ("nbf", nbf),
        ("occupied", occupied),
        ("virtuals", virtuals),
        ("auxiliaries", auxiliaries),
    ):
        checked_bytes(value, name)
    if not nbf or not occupied or not virtuals or not auxiliaries or occupied >= nbf:
        raise ValueError("invalid CUDA RI-MP2 block-planner dimensions")

    preferred_j = max(1, min(occupied, 8))
    full_peak = _ri_block_capacity(
        fixed_bytes,
        nbf,
        occupied,
        auxiliaries,
        virtuals,
        preferred_j,
        full_resident=True,
    )
    resident_j = preferred_j
    while full_peak > budget_bytes and resident_j > 1:
        resident_j = (resident_j + 1) // 2
        full_peak = _ri_block_capacity(
            fixed_bytes,
            nbf,
            occupied,
            auxiliaries,
            virtuals,
            resident_j,
            full_resident=True,
        )
    if full_peak <= budget_bytes:
        return RiMp2ResidencyPlan(virtuals, resident_j, full_peak, True)

    best = 0
    best_j = 1
    lower, upper = 1, virtuals - 1
    while lower <= upper:
        middle = lower + (upper - lower) // 2
        j_batch = preferred_j
        while j_batch > 1:
            peak = _ri_block_capacity(
                fixed_bytes,
                nbf,
                occupied,
                auxiliaries,
                middle,
                j_batch,
                full_resident=False,
            )
            if peak <= budget_bytes:
                break
            j_batch = (j_batch + 1) // 2
        peak = _ri_block_capacity(
            fixed_bytes,
            nbf,
            occupied,
            auxiliaries,
            middle,
            j_batch,
            full_resident=False,
        )
        if peak <= budget_bytes:
            best, best_j = middle, j_batch
            lower = middle + 1
        else:
            upper = middle - 1
    if not best:
        raise ValueError(
            "CUDA RI-MP2 resident/blocked B transform exceeds numeric memory budget"
        )
    peak = _ri_block_capacity(
        fixed_bytes,
        nbf,
        occupied,
        auxiliaries,
        best,
        best_j,
        full_resident=False,
    )
    return RiMp2ResidencyPlan(best, best_j, peak, False)


def native_header() -> str:
    """Emit the checked native schedule used by conventional and RI MP2."""

    return r"""// Generated from vibeqc_compiler.method.mp2_schedule; do not edit.
#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include "posthf/capacity.hpp"
namespace vibeqc::mp2::generated {
struct ConventionalReusePlan {
  bool shared_scan;
  std::size_t jobs_per_batch;
  std::size_t provider_requests;
};
inline ConventionalReusePlan conventional_reuse_plan(std::size_t request_capacity,
                                                       std::size_t total_jobs) {
  if (!total_jobs) throw std::invalid_argument("MP2 reuse schedule requires at least one job");
  if (request_capacity < 2) return {false, 1, 1};
  const auto jobs = std::min(request_capacity / 2, total_jobs);
  return {true, jobs, posthf::checked_mul(2, jobs)};
}
struct RiMp2ResidencyPlan {
  std::size_t virtual_block;
  std::size_t j_batch;
  std::size_t peak_bytes;
  bool full_resident;
};
inline std::size_t ri_block_capacity(std::size_t fixed, std::size_t n,
                                     std::size_t no, std::size_t na,
                                     std::size_t virtual_block,
                                     std::size_t j_batch, bool full_resident) {
  auto total = fixed;
  total = posthf::checked_add(
      total, posthf::checked_mul(8, posthf::checked_mul(
          posthf::checked_mul(n, na), virtual_block)));
  const auto b_elements =
      posthf::checked_mul(posthf::checked_mul(na, no), virtual_block);
  total = posthf::checked_add(total, posthf::checked_mul(8, b_elements));
  if (!full_resident)
    total = posthf::checked_add(total, posthf::checked_mul(8, b_elements));
  const auto integral_batch =
      posthf::checked_mul(posthf::checked_mul(j_batch, virtual_block), virtual_block);
  total = posthf::checked_add(total, posthf::checked_mul(8, integral_batch));
  if (!full_resident)
    total = posthf::checked_add(total, posthf::checked_mul(8, integral_batch));
  return total;
}
inline RiMp2ResidencyPlan ri_mp2_residency_plan(
    std::size_t fixed, std::size_t budget, std::size_t n, std::size_t no,
    std::size_t nv, std::size_t na) {
  if (!n || !no || !nv || !na || no >= n)
    throw std::invalid_argument("invalid CUDA RI-MP2 block-planner dimensions");
  const std::size_t preferred_j = std::max<std::size_t>(1, std::min<std::size_t>(no, 8));
  auto resident_j = preferred_j;
  auto full_peak = ri_block_capacity(fixed, n, no, na, nv, resident_j, true);
  while (full_peak > budget && resident_j > 1) {
    resident_j = (resident_j + 1) / 2;
    full_peak = ri_block_capacity(fixed, n, no, na, nv, resident_j, true);
  }
  if (full_peak <= budget) return {nv, resident_j, full_peak, true};
  std::size_t best = 0, best_j = 1, lower = 1, upper = nv - 1;
  while (lower <= upper) {
    const std::size_t middle = lower + (upper - lower) / 2;
    std::size_t j_batch = preferred_j;
    while (j_batch > 1 &&
           ri_block_capacity(fixed, n, no, na, middle, j_batch, false) > budget)
      j_batch = (j_batch + 1) / 2;
    const auto peak =
        ri_block_capacity(fixed, n, no, na, middle, j_batch, false);
    if (peak <= budget) {
      best = middle;
      best_j = j_batch;
      lower = middle + 1;
    } else {
      upper = middle - 1;
    }
  }
  if (!best)
    throw std::length_error(
        "CUDA RI-MP2 resident/blocked B transform exceeds numeric memory budget");
  return {best, best_j,
          ri_block_capacity(fixed, n, no, na, best, best_j, false), false};
}
}  // namespace vibeqc::mp2::generated
"""
