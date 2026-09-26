// Generated from tools/vibeqc_posthf/plan_spec.py; do not change formulas here.
#pragma once
#include <algorithm>
#include <array>

#include "posthf/capacity.hpp"
namespace vibeqc::posthf {
struct NumericBlockPlan {
  std::size_t output_elements;
  std::size_t tile_elements;
  std::size_t coefficient_elements;
  std::size_t stage_elements;
  std::size_t host_bytes;
  std::size_t aligned_numeric;
  std::size_t allocation_bytes;
  std::size_t device_bytes;
};
inline NumericBlockPlan numeric_block_plan(std::size_t nbf, std::size_t reference_bytes,
                                           std::size_t source_bytes,
                                           const std::array<std::size_t, 4>& shape,
                                           const std::array<std::size_t, 4>& tile, bool cuda) {
  const auto m0 = shape[0], m1 = shape[1], m2 = shape[2], m3 = shape[3];
  const auto t0 = tile[0], t1 = tile[1], t2 = tile[2], t3 = tile[3];
  const std::size_t output_elements = checked_mul(checked_mul(checked_mul(m0, m1), m2), m3);
  const std::size_t tile_elements = checked_mul(checked_mul(checked_mul(t0, t1), t2), t3);
  const std::size_t coefficient_elements =
      checked_mul(nbf, checked_add(checked_add(checked_add(m0, m1), m2), m3));
  const std::size_t stage_elements = std::max<std::size_t>(
      {tile_elements, checked_mul(checked_mul(checked_mul(t1, t2), t3), m0),
       checked_mul(checked_mul(checked_mul(t2, t3), m0), m1),
       checked_mul(checked_mul(checked_mul(t3, m0), m1), m2), output_elements});
  const std::size_t host_bytes = checked_add(
      checked_add(checked_add(reference_bytes, source_bytes), 8388608ULL),
      checked_mul(8ULL, checked_add(checked_add(checked_add(checked_mul(4ULL, tile_elements),
                                                            checked_mul(2ULL, stage_elements)),
                                                checked_mul(3ULL, output_elements)),
                                    checked_mul(2ULL, coefficient_elements))));
  const std::size_t aligned_numeric = checked_mul(
      (checked_add(checked_mul(8ULL, checked_add(checked_add(coefficient_elements,
                                                             checked_mul(2ULL, stage_elements)),
                                                 output_elements)),
                   255ULL) /
       256ULL),
      256ULL);
  const std::size_t allocation_bytes =
      ((cuda && output_elements) ? checked_add(checked_add(aligned_numeric, 256ULL), 4194304ULL)
                                 : 0ULL);
  const std::size_t device_bytes =
      (allocation_bytes ? checked_add(allocation_bytes, 100663296ULL) : 0ULL);
  return {output_elements, tile_elements,   coefficient_elements, stage_elements,
          host_bytes,      aligned_numeric, allocation_bytes,     device_bytes};
}
}  // namespace vibeqc::posthf
