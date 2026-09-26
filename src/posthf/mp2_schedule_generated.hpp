// Generated from vibeqc_compiler.method.mp2_schedule; do not edit.
#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include "posthf/capacity.hpp"
#include "posthf/source_reuse_schedule_generated.hpp"
namespace vibeqc::mp2::generated {
struct ConventionalReusePlan {
  bool shared_scan;
  std::size_t jobs_per_batch;
  std::size_t provider_requests;
};
inline ConventionalReusePlan conventional_reuse_plan(std::size_t request_capacity,
                                                       std::size_t total_jobs) {
  const auto plan =
      posthf::generated::uniform_source_reuse_plan(request_capacity, 2, total_jobs);
  return {plan.shared_scan, plan.jobs_per_batch, plan.provider_requests};
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
