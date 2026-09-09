#include <limits>

#include "generated_one_electron_values.cuh"
#include "molecule/basis.hpp"
#include "scf/cuda/one_electron_values.cuh"

namespace vibeqc::scf {
namespace {

namespace generated = generated_one_electron;
constexpr std::size_t kTerms = molecule::kMaximumAoExpansionTerms;

struct Values {
  double overlap{}, kinetic{}, attraction{};
};

/** Contract normalized public AOs without duplicating a shell radial norm.
 * A single primitive traversal accumulates S/T/V, and pair geometry lives
 * outside the on-device nuclear loop. Spherical AOs use the existing sparse
 * normalized Cartesian expansions; no new transform convention is introduced.
 */
__device__ Values contracted(const OneElectronDeviceView& batch, std::int64_t i, std::int64_t j) {
  const auto si = batch.ao_shells[i], sj = batch.ao_shells[j];
  const auto system = i / batch.nbf;
  const double* A = batch.positions + 3 * batch.shell_atoms[si];
  const double* B = batch.positions + 3 * batch.shell_atoms[sj];
  Values result;
  for (auto a = batch.shell_primitive_offsets[si]; a < batch.shell_primitive_offsets[si + 1]; ++a) {
    for (auto b = batch.shell_primitive_offsets[sj]; b < batch.shell_primitive_offsets[sj + 1];
         ++b) {
      const auto pair =
          generated::make_pair(batch.primitive_exponents[a], batch.primitive_exponents[b], A[0],
                               A[1], A[2], B[0], B[1], B[2]);
      const double primitive_weight =
          batch.primitive_coefficients[a] * batch.primitive_coefficients[b];
      for (unsigned ti = 0; ti < batch.ao_term_counts[i]; ++ti) {
        const auto term_i = i * kTerms + ti;
        const auto* ai = batch.ao_term_angular + 3 * term_i;
        const auto first = generated::component_index(ai[0], ai[1], ai[2]);
        for (unsigned tj = 0; tj < batch.ao_term_counts[j]; ++tj) {
          const auto term_j = j * kTerms + tj;
          const auto* aj = batch.ao_term_angular + 3 * term_j;
          const auto second = generated::component_index(aj[0], aj[1], aj[2]);
          const double weight = primitive_weight * batch.ao_term_coefficients[term_i] *
                                batch.ao_term_coefficients[term_j];
          const auto st = generated::overlap_kinetic(pair, first, second);
          double v = 0.0;
          for (auto atom = batch.atom_offsets[system]; atom < batch.atom_offsets[system + 1];
               ++atom) {
            const double* C = batch.positions + 3 * atom;
            v += batch.atomic_numbers[atom] *
                 generated::attraction(pair, first, second, C[0], C[1], C[2]);
          }
          result.overlap += weight * st.overlap;
          result.kinetic += weight * st.kinetic;
          result.attraction += weight * v;
        }
      }
    }
  }
  return result;
}

__device__ void write_values(const OneElectronDeviceView& batch, std::int64_t i, std::int64_t j,
                             double* overlap, double* hcore, double* kinetic, double* attraction) {
  const auto values = contracted(batch, i, j);
  const std::size_t n = batch.nbf;
  const std::size_t system = i / n;
  const std::size_t row = i % n, column = j % n;
  const std::size_t offset = system * n * n;
  const auto first = offset + row * n + column, second = offset + column * n + row;
  overlap[first] = values.overlap;
  hcore[first] = values.kinetic + values.attraction;
  if (kinetic) kinetic[first] = values.kinetic;
  if (attraction) attraction[first] = values.attraction;
  if (row != column) {
    overlap[second] = values.overlap;
    hcore[second] = values.kinetic + values.attraction;
    if (kinetic) kinetic[second] = values.kinetic;
    if (attraction) attraction[second] = values.attraction;
  }
}

__global__ void thread_values(OneElectronDeviceView batch, const std::int32_t* pair_first,
                              const std::int32_t* pair_second, std::size_t pair_count,
                              double* overlap, double* hcore, double* kinetic, double* attraction) {
  const std::size_t task = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (task >= static_cast<std::size_t>(batch.batch_size) * pair_count) return;
  const std::int64_t base = (task / pair_count) * batch.nbf;
  const std::size_t pair = task % pair_count;
  write_values(batch, base + pair_first[pair], base + pair_second[pair], overlap, hcore, kinetic,
               attraction);
}

__global__ void shell_warp_values(OneElectronDeviceView batch, double* overlap, double* hcore,
                                  double* kinetic, double* attraction) {
  const std::size_t task = (std::size_t{blockIdx.x} * blockDim.x + threadIdx.x) / 32;
  if (task >= batch.shell_pair_count) return;
  const auto si = batch.shell_pair_first[task], sj = batch.shell_pair_second[task];
  const auto begin_i = batch.shell_ao_offsets[si], begin_j = batch.shell_ao_offsets[sj];
  const auto count_i = batch.shell_ao_offsets[si + 1] - begin_i;
  const auto count_j = batch.shell_ao_offsets[sj + 1] - begin_j;
  for (std::int64_t component = threadIdx.x % 32; component < count_i * count_j; component += 32) {
    const auto i = begin_i + component / count_j, j = begin_j + component % count_j;
    if (si == sj && i < j) continue;
    write_values(batch, i, j, overlap, hcore, kinetic, attraction);
  }
}

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
    shell_warp_values<<<blocks, threads, 0, stream>>>(batch, overlap, hcore, kinetic, attraction);
  } else {
    thread_values<<<blocks, threads, 0, stream>>>(batch, pair_first, pair_second, pair_count,
                                                  overlap, hcore, kinetic, attraction);
  }
  return cudaPeekAtLastError();
}

}  // namespace vibeqc::scf
