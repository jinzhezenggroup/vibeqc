#ifndef VIBEQC_SCF_CUDA_ONE_ELECTRON_KERNEL_ABI_CUH
#define VIBEQC_SCF_CUDA_ONE_ELECTRON_KERNEL_ABI_CUH

#include "scf/cuda/one_electron_values.cuh"

namespace vibeqc::scf::cuda_kernel_abi {

/** Rebuild the host launcher view from scalar/pointer CUDA kernel parameters.
 *
 * Some CUDA-compatible PTX consumers do not support aggregate .param loads.
 * Keep the public host launcher typed while making the device entry ABI
 * scalar/pointer-only. The local view is caller-owned and never returned by
 * value across a device-function boundary.
 */
__device__ __forceinline__ void bind_one_electron_view(
    OneElectronDeviceView& view, std::int32_t batch_size, std::int32_t nbf,
    std::size_t shell_pair_count, const std::int64_t* atom_offsets,
    const std::int32_t* atomic_numbers, const double* positions, const std::int32_t* shell_atoms,
    const std::int64_t* shell_ao_offsets, const std::int64_t* shell_primitive_offsets,
    const std::int32_t* shell_pair_first, const std::int32_t* shell_pair_second,
    const std::int32_t* ao_shells, const std::uint8_t* ao_term_counts,
    const std::uint8_t* ao_term_angular, const double* ao_term_coefficients,
    const double* primitive_exponents, const double* primitive_coefficients) {
  view.batch_size = batch_size;
  view.nbf = nbf;
  view.shell_pair_count = shell_pair_count;
  view.atom_offsets = atom_offsets;
  view.atomic_numbers = atomic_numbers;
  view.positions = positions;
  view.shell_atoms = shell_atoms;
  view.shell_ao_offsets = shell_ao_offsets;
  view.shell_primitive_offsets = shell_primitive_offsets;
  view.shell_pair_first = shell_pair_first;
  view.shell_pair_second = shell_pair_second;
  view.ao_shells = ao_shells;
  view.ao_term_counts = ao_term_counts;
  view.ao_term_angular = ao_term_angular;
  view.ao_term_coefficients = ao_term_coefficients;
  view.primitive_exponents = primitive_exponents;
  view.primitive_coefficients = primitive_coefficients;
}

}  // namespace vibeqc::scf::cuda_kernel_abi

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

#define VIBEQC_BIND_ONE_ELECTRON_VIEW(view)                                                        \
  ::vibeqc::scf::cuda_kernel_abi::bind_one_electron_view(                                          \
      view, batch_size, nbf, shell_pair_count, atom_offsets, atomic_numbers, positions,            \
      shell_atoms, shell_ao_offsets, shell_primitive_offsets, shell_pair_first, shell_pair_second, \
      ao_shells, ao_term_counts, ao_term_angular, ao_term_coefficients, primitive_exponents,       \
      primitive_coefficients)

#endif
