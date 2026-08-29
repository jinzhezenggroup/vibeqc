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

/** Stable density maxima consumed by generated shell-pair stream kernels. */
struct GeneratedShellPairDensityBounds {
  double coulomb;
  double exchange_alpha;
  double exchange_beta;
};

static_assert(sizeof(GeneratedShellPairDensityBounds) == 3 * sizeof(double));

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
 * Device topology view for fixed-storage generated direct-J/K streaming.
 *
 * ``pair_order`` is grouped first by the ten angular shell-pair classes and
 * then by physical system. ``pair_class_offsets`` stores absolute offsets for
 * every ``[pair_class][system]`` segment and therefore has
 * ``10 * (batch_size + 1)`` entries. Generated kernels keep one bra pair
 * resident while traversing the matching ket segment, mirroring GPU4PySCF's
 * bounded per-worker queue without materializing the shell-quartet product.
 */
struct GeneratedShellPairStream {
  std::int32_t batch_size;
  std::uint32_t matrix_order;
  const std::int64_t* system_shell_offsets;
  const std::int64_t* system_shell_pair_offsets;
  const std::int32_t* shell_atoms;
  const std::uint8_t* shell_angular;
  const std::int64_t* shell_direct_ao_offsets;
  const std::int64_t* shell_primitive_offsets;
  const std::int32_t* shell_pair_systems;
  const std::int32_t* shell_pair_first;
  const std::int32_t* shell_pair_second;
  const std::uint32_t* pair_order;
  const std::uint32_t* pair_class_offsets;
  const double* shell_pair_bounds;
  const GeneratedShellPairDensityBounds* shell_pair_density_bounds;
  const std::uint8_t* active;
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
