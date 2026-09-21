#include <limits>

#include "generated_one_electron_policy.cuh"
#include "molecule/basis.hpp"
#include "runtime/cuda_ao_pairs.cuh"
#include "scf/cuda/one_electron_kernel_abi.cuh"
#include "scf/cuda/one_electron_values.cuh"

namespace vibeqc::scf {
namespace {
using Policy = generated_one_electron::ValuePolicy;
namespace pairs = runtime::cuda_ao_pairs;
constexpr std::size_t kTerms = molecule::kMaximumAoExpansionTerms;

__global__ void thread_pairs_flat(VIBEQC_ONE_ELECTRON_VIEW_KERNEL_PARAMETERS,
                                  const std::int32_t* pair_first, const std::int32_t* pair_second,
                                  std::size_t pair_count, double* overlap, double* hcore,
                                  double* kinetic, double* attraction) {
  pairs::thread_pairs_body<Policy, kTerms>(
      batch_size, nbf, atom_offsets, atomic_numbers, positions, shell_atoms,
      shell_primitive_offsets, ao_shells, ao_term_counts, ao_term_angular, ao_term_coefficients,
      primitive_exponents, primitive_coefficients, pair_first, pair_second, pair_count, overlap,
      hcore, kinetic, attraction);
}

__global__ void shell_warp_pairs_flat(VIBEQC_ONE_ELECTRON_VIEW_KERNEL_PARAMETERS, double* overlap,
                                      double* hcore, double* kinetic, double* attraction) {
  pairs::shell_warp_pairs_body<Policy, kTerms>(
      nbf, shell_pair_count, atom_offsets, atomic_numbers, positions, shell_atoms, shell_ao_offsets,
      shell_primitive_offsets, shell_pair_first, shell_pair_second, ao_shells, ao_term_counts,
      ao_term_angular, ao_term_coefficients, primitive_exponents, primitive_coefficients, overlap,
      hcore, kinetic, attraction);
}

}  // namespace

cudaError_t launch_generated_one_electron_values(
    const OneElectronDeviceView& batch, const std::int32_t* pair_first,
    const std::int32_t* pair_second, std::size_t pair_count, unsigned schedule, double* overlap,
    double* hcore, cudaStream_t stream, double* kinetic, double* attraction) {
  if (batch.batch_size <= 0 || batch.nbf <= 0 || !overlap || !hcore || schedule > 1)
    return cudaErrorInvalidValue;
  constexpr unsigned threads = 128;
  const std::size_t tasks = schedule == 1 ? batch.shell_pair_count
                                          : static_cast<std::size_t>(batch.batch_size) * pair_count;
  if (tasks == 0 || (schedule == 0 && (!pair_first || !pair_second || pair_count == 0)))
    return cudaErrorInvalidValue;
  const std::size_t tasks_per_block = schedule == 1 ? threads / 32 : threads;
  if ((tasks - 1) / tasks_per_block >= std::numeric_limits<int>::max())
    return cudaErrorInvalidValue;
  const unsigned blocks = static_cast<unsigned>((tasks - 1) / tasks_per_block + 1);
  if (schedule == 1) {
    shell_warp_pairs_flat<<<blocks, threads, 0, stream>>>(VIBEQC_ONE_ELECTRON_VIEW_KERNEL_ARGUMENTS,
                                                          overlap, hcore, kinetic, attraction);
  } else {
    thread_pairs_flat<<<blocks, threads, 0, stream>>>(VIBEQC_ONE_ELECTRON_VIEW_KERNEL_ARGUMENTS,
                                                      pair_first, pair_second, pair_count, overlap,
                                                      hcore, kinetic, attraction);
  }
  return cudaPeekAtLastError();
}

}  // namespace vibeqc::scf