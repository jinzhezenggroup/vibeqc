#ifndef VIBEQC_SCF_GENERATED_SHELL_TASK_HPP
#define VIBEQC_SCF_GENERATED_SHELL_TASK_HPP

#include <cstdint>

namespace vibeqc::scf::detail {

/** Stable geometry-cache ABI shared by handwritten and generated kernels. */
struct GeneratedPrimitivePairData {
  double exponent_sum;
  double reduced_exponent;
  double product_center[3];
  double weighted_coefficient;
  double first_product_scale;
  double second_product_scale;
};

static_assert(sizeof(GeneratedPrimitivePairData) == 8 * sizeof(double));

/** Stable task ABI shared by CUDA compaction and generated kernel shards. */
struct GeneratedShellTask {
  std::uint64_t primitive_begin[4];
  std::uint64_t primitive_end[4];
  std::uint64_t ao_begin[4];
  std::uint64_t ao_coefficient_begin[4];
  std::uint64_t density_offset;
  std::uint64_t spin_offset;
  std::uint32_t matrix_order;
  std::uint32_t shell_pair[2];
  // Bit n is set when canonical slots 2*n and 2*n+1 reverse the cache's
  // shell_pair_first/shell_pair_second order.
  std::uint32_t reversed_shell_pair_mask;
  std::uint32_t shell[4];
  std::uint32_t atom[4];
};

}  // namespace vibeqc::scf::detail

#endif
