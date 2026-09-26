#ifndef VIBEQC_SCF_CUDA_ONE_ELECTRON_VALUES_CUH
#define VIBEQC_SCF_CUDA_ONE_ELECTRON_VALUES_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::scf {

/** Non-owning device view of existing normalized basis topology and geometry.
 * AO expansion terms retain the basis layer's Cartesian/spherical order.
 * All shell, primitive, atom and AO indices are global within the packed batch.
 * Positions and nuclear charges must describe the current invocation; this
 * layer retains no geometry cache or allocation of its own.
 */
struct OneElectronDeviceView {
  std::int32_t batch_size, nbf;
  std::size_t shell_pair_count;
  const std::int64_t* atom_offsets;
  const std::int32_t* atomic_numbers;
  const double* positions;
  const std::int32_t* shell_atoms;
  const std::int64_t* shell_ao_offsets;
  const std::int64_t* shell_primitive_offsets;
  const std::int32_t* shell_pair_first;
  const std::int32_t* shell_pair_second;
  const std::int32_t* ao_shells;
  const std::uint8_t* ao_term_counts;
  const std::uint8_t* ao_term_angular;
  const double* ao_term_coefficients;
  const double* primitive_exponents;
  const double* primitive_coefficients;
};

/** Launch normalized S/Hcore, optionally retaining separate T/V diagnostics.
 * Schedule 0 assigns each lower-triangular AO pair to a thread; schedule 1
 * assigns each shell pair to a warp with component lanes. One owner writes
 * both symmetric entries (the diagonal once). Matrices use a full per-system
 * layout; symmetry makes row/column-major storage equivalent here.
 * The caller supplies existing output buffers. No allocation or host/device
 * synchronization occurs, making this launch safe inside a prepared plan.
 */
cudaError_t launch_generated_one_electron_values(
    const OneElectronDeviceView& batch, const std::int32_t* pair_first,
    const std::int32_t* pair_second, std::size_t pair_count, unsigned schedule, double* overlap,
    double* hcore, cudaStream_t stream, double* kinetic = nullptr, double* attraction = nullptr);

}  // namespace vibeqc::scf

#endif
