#ifndef VIBEQC_SCF_CUDA_ONE_ELECTRON_KERNEL_ABI_CUH
#define VIBEQC_SCF_CUDA_ONE_ELECTRON_KERNEL_ABI_CUH

#include "scf/cuda/one_electron_values.cuh"

/** Scalar/pointer CUDA entry ABI for one-electron value and derivative kernels.
 *
 * Keep aggregates on the host launcher side only. Device entries and the device
 * call graph receive individual scalar/pointer values so PTX consumers never
 * need to reconstruct heterogeneous pointer/integer records from local memory.
 */
#define VIBEQC_ONE_ELECTRON_VIEW_KERNEL_PARAMETERS                                       \
  std::int32_t batch_size, std::int32_t nbf, std::size_t shell_pair_count,               \
      const std::int64_t *atom_offsets, const std::int32_t *atomic_numbers,              \
      const double *positions, const std::int32_t *shell_atoms,                          \
      const std::int64_t *shell_ao_offsets, const std::int64_t *shell_primitive_offsets, \
      const std::int32_t *shell_pair_first, const std::int32_t *shell_pair_second,       \
      const std::int32_t *ao_shells, const std::uint8_t *ao_term_counts,                 \
      const std::uint8_t *ao_term_angular, const double *ao_term_coefficients,           \
      const double *primitive_exponents, const double *primitive_coefficients

#define VIBEQC_ONE_ELECTRON_VIEW_KERNEL_ARGUMENTS                                                \
  batch.batch_size, batch.nbf, batch.shell_pair_count, batch.atom_offsets, batch.atomic_numbers, \
      batch.positions, batch.shell_atoms, batch.shell_ao_offsets, batch.shell_primitive_offsets, \
      batch.shell_pair_first, batch.shell_pair_second, batch.ao_shells, batch.ao_term_counts,    \
      batch.ao_term_angular, batch.ao_term_coefficients, batch.primitive_exponents,              \
      batch.primitive_coefficients

#endif
