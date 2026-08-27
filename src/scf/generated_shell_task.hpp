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

/**
 * Descriptor for one resident canonical ``ppps`` bra shell pair.
 *
 * The descriptor owns all active ``p s`` ket tasks for one ``p p`` bra pair.
 * ``ket_begin`` and ``ket_count`` index a device-side, bra-grouped array of
 * ``GeneratedShellTask`` records.  They are deliberately 32-bit: the direct
 * shell-pair task builder already bounds one bucket below ``UINT32_MAX`` and
 * the compact resident allocation is at most one record per active ppps
 * shell quartet.  Keeping this ABI to three words makes descriptor traffic
 * negligible compared with the generated recurrence.
 */
struct GeneratedPppsResidentTask {
  std::uint32_t bra_pair;
  std::uint32_t ket_begin;
  std::uint32_t ket_count;
};

static_assert(sizeof(GeneratedPppsResidentTask) == 12);
static_assert(alignof(GeneratedPppsResidentTask) == alignof(std::uint32_t));

}  // namespace vibeqc::scf::detail

#endif
