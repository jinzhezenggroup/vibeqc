#include <limits>

#include "generated_one_electron_policy.cuh"
#include "molecule/basis.hpp"
#include "runtime/cuda_ao_pairs.cuh"
#include "scf/cuda/one_electron_values.cuh"

namespace vibeqc::scf {
namespace {

using Policy = generated_one_electron::ValuePolicy;
namespace pairs = runtime::cuda_ao_pairs;
constexpr std::size_t kTerms = molecule::kMaximumAoExpansionTerms;

}  // namespace

cudaError_t launch_generated_one_electron_values(
    const OneElectronDeviceView& batch, const std::int32_t* pair_first,
    const std::int32_t* pair_second, std::size_t pair_count, unsigned schedule, double* overlap,
    double* hcore, cudaStream_t stream, double* kinetic, double* attraction) {
  constexpr unsigned threads = 128;
  if (batch.batch_size <= 0 || batch.nbf <= 0 || !overlap || !hcore || schedule > 1)
    return cudaErrorInvalidValue;
  const std::size_t tasks = schedule == 1 ? batch.shell_pair_count
                                          : static_cast<std::size_t>(batch.batch_size) * pair_count;
  const unsigned tasks_per_block = schedule == 1 ? threads / 32 : threads;
  if (tasks == 0 || (tasks - 1) / tasks_per_block >= std::numeric_limits<int>::max())
    return cudaErrorInvalidValue;
  const unsigned blocks = static_cast<unsigned>((tasks - 1) / tasks_per_block + 1);
  if (schedule == 1) {
    pairs::shell_warp_pairs<Policy, kTerms><<<blocks, threads, 0, stream>>>(
        batch, pairs::Outputs<Policy::channels>{{overlap, hcore, kinetic, attraction}});
  } else {
    pairs::thread_pairs<Policy, kTerms><<<blocks, threads, 0, stream>>>(
        batch, pair_first, pair_second, pair_count,
        pairs::Outputs<Policy::channels>{{overlap, hcore, kinetic, attraction}});
  }
  return cudaPeekAtLastError();
}

}  // namespace vibeqc::scf
