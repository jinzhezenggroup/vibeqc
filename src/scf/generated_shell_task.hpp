#ifndef QCE_SCF_GENERATED_SHELL_TASK_HPP
#define QCE_SCF_GENERATED_SHELL_TASK_HPP

#include <cstdint>

namespace qce::scf::detail {

/** Stable task ABI shared by CUDA compaction and generated kernel shards. */
struct GeneratedShellTask {
  std::uint64_t primitive_begin[4];
  std::uint64_t primitive_end[4];
  std::uint64_t ao_begin[4];
  std::uint64_t ao_coefficient_begin[4];
  std::uint64_t density_offset;
  std::uint64_t spin_offset;
  std::uint32_t matrix_order;
  std::uint32_t shell[4];
  std::uint32_t atom[4];
};

}  // namespace qce::scf::detail

#endif
