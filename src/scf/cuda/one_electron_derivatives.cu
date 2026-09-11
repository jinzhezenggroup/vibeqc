#include <cmath>
#include <limits>

#include "generated_one_electron_derivatives.cuh"
#include "molecule/basis.hpp"
#include "scf/cuda/one_electron_derivatives.cuh"

namespace vibeqc::scf {
namespace {
namespace generated = generated_one_electron_derivatives;
constexpr std::size_t kTerms = molecule::kMaximumAoExpansionTerms;

__device__ double pair_weight(const double* matrix, double scale, std::size_t offset, std::size_t i,
                              std::size_t j, std::size_t n) {
  if (!matrix) return 0.0;
  const double weight = matrix[offset + i * n + j];
  return scale * (i == j ? weight : weight + matrix[offset + j * n + i]);
}

/** Evaluate each primitive/component once and fuse its three center responses.
 * Atom ownership is applied after mathematical differentiation: coincident or
 * shared A/B/C atoms retain every contribution, including cancellation. Only
 * pair-center accumulators live in registers; each external nuclear response
 * goes directly to the bounded global gradient buffer.
 */
__device__ void contract_pair(const OneElectronDeviceView& batch, std::int64_t i, std::int64_t j,
                              const OneElectronWeightView& weights, double sign, double* gradient) {
  const std::size_t n = batch.nbf, system = i / n, offset = system * n * n;
  const double ws = pair_weight(weights.overlap, weights.overlap_scale, offset, i % n, j % n, n);
  const double wt = pair_weight(weights.kinetic, weights.kinetic_scale, offset, i % n, j % n, n);
  const double wv =
      pair_weight(weights.attraction, weights.attraction_scale, offset, i % n, j % n, n);
  if (ws == 0.0 && wt == 0.0 && wv == 0.0) return;
  const auto si = batch.ao_shells[i], sj = batch.ao_shells[j];
  const auto atom_a = batch.shell_atoms[si], atom_b = batch.shell_atoms[sj];
  const double* A = batch.positions + 3 * atom_a;
  const double* B = batch.positions + 3 * atom_b;
  double first[3]{}, second[3]{};
  for (auto a = batch.shell_primitive_offsets[si]; a < batch.shell_primitive_offsets[si + 1]; ++a) {
    for (auto b = batch.shell_primitive_offsets[sj]; b < batch.shell_primitive_offsets[sj + 1];
         ++b) {
      const auto pair =
          generated::make_pair(batch.primitive_exponents[a], batch.primitive_exponents[b], A[0],
                               A[1], A[2], B[0], B[1], B[2]);
      const double radial = batch.primitive_coefficients[a] * batch.primitive_coefficients[b];
      for (unsigned ti = 0; ti < batch.ao_term_counts[i]; ++ti) {
        const auto term_i = i * kTerms + ti;
        const auto* ai = batch.ao_term_angular + 3 * term_i;
        const auto ca = generated::component_index(ai[0], ai[1], ai[2]);
        for (unsigned tj = 0; tj < batch.ao_term_counts[j]; ++tj) {
          const auto term_j = j * kTerms + tj;
          const auto* aj = batch.ao_term_angular + 3 * term_j;
          const auto cb = generated::component_index(aj[0], aj[1], aj[2]);
          const double norm = sign * radial * batch.ao_term_coefficients[term_i] *
                              batch.ao_term_coefficients[term_j];
          if (ws != 0.0 || wt != 0.0) {
            const auto st = generated::overlap_kinetic_gradient(pair, ca, cb);
            for (unsigned axis = 0; axis < 3; ++axis) {
              const double value = norm * (ws * st.first[axis] + wt * st.second[axis]);
              first[axis] += value;
              second[axis] -= value;
            }
          }
          if (wv != 0.0) {
            for (auto atom = batch.atom_offsets[system]; atom < batch.atom_offsets[system + 1];
                 ++atom) {
              const double* C = batch.positions + 3 * atom;
              const auto v = generated::attraction_gradient(pair, ca, cb, C[0], C[1], C[2]);
              const double factor = norm * wv * batch.atomic_numbers[atom];
              for (unsigned axis = 0; axis < 3; ++axis) {
                const double da = factor * v.first[axis], db = factor * v.second[axis];
                first[axis] += da;
                second[axis] += db;
                atomicAdd(gradient + 3 * atom + axis, -da - db);
              }
            }
          }
        }
      }
    }
  }
  for (unsigned axis = 0; axis < 3; ++axis) {
    atomicAdd(gradient + 3 * atom_a + axis, first[axis]);
    atomicAdd(gradient + 3 * atom_b + axis, second[axis]);
  }
}

__global__ void thread_gradient(OneElectronDeviceView batch, const std::int32_t* first,
                                const std::int32_t* second, std::size_t count,
                                OneElectronWeightView weights, const std::uint8_t* active,
                                double sign, double* gradient) {
  const std::size_t task = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (task >= static_cast<std::size_t>(batch.batch_size) * count) return;
  const auto system = task / count;
  if (active && !active[system]) return;
  const auto base = system * batch.nbf, pair = task % count;
  contract_pair(batch, base + first[pair], base + second[pair], weights, sign, gradient);
}

__global__ void shell_warp_gradient(OneElectronDeviceView batch, OneElectronWeightView weights,
                                    const std::uint8_t* active, double sign, double* gradient) {
  const std::size_t task = (std::size_t{blockIdx.x} * blockDim.x + threadIdx.x) / 32;
  if (task >= batch.shell_pair_count) return;
  const auto si = batch.shell_pair_first[task], sj = batch.shell_pair_second[task];
  const auto begin_i = batch.shell_ao_offsets[si], begin_j = batch.shell_ao_offsets[sj];
  if (active && !active[begin_i / batch.nbf]) return;
  const auto ni = batch.shell_ao_offsets[si + 1] - begin_i,
             nj = batch.shell_ao_offsets[sj + 1] - begin_j;
  for (std::int64_t c = threadIdx.x % 32; c < ni * nj; c += 32) {
    const auto i = begin_i + c / nj, j = begin_j + c % nj;
    if (si == sj && i < j) continue;
    contract_pair(batch, i, j, weights, sign, gradient);
  }
}

__global__ void serial_gradient(OneElectronDeviceView batch, OneElectronWeightView weights,
                                const std::uint8_t* active, double sign, double* gradient) {
  const std::size_t system = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (system >= static_cast<std::size_t>(batch.batch_size) || (active && !active[system])) return;
  const auto base = system * batch.nbf;
  for (std::int32_t i = 0; i < batch.nbf; ++i)
    for (std::int32_t j = 0; j <= i; ++j)
      contract_pair(batch, base + i, base + j, weights, sign, gradient);
}
}  // namespace

cudaError_t launch_generated_one_electron_gradient(
    const OneElectronDeviceView& batch, const std::int32_t* pair_first,
    const std::int32_t* pair_second, std::size_t pair_count, const OneElectronWeightView& weights,
    const std::uint8_t* active, unsigned schedule, double output_sign, double* gradient,
    cudaStream_t stream) {
  if (batch.batch_size <= 0 || batch.nbf <= 0 || !gradient || schedule > 2 ||
      !std::isfinite(output_sign) || !std::isfinite(weights.overlap_scale) ||
      !std::isfinite(weights.kinetic_scale) || !std::isfinite(weights.attraction_scale))
    return cudaErrorInvalidValue;
  constexpr unsigned threads = 128;
  if (schedule == 0 && (!pair_first || !pair_second || pair_count == 0))
    return cudaErrorInvalidValue;
  // Validate products before task/grid calculations or matrix offset indexing.
  const auto systems = static_cast<std::size_t>(batch.batch_size);
  const auto n = static_cast<std::size_t>(batch.nbf);
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (n > maximum / n || systems > maximum / (n * n) ||
      (schedule == 0 && pair_count > maximum / systems))
    return cudaErrorInvalidValue;
  if (schedule == 1 && (!batch.shell_pair_first || !batch.shell_pair_second))
    return cudaErrorInvalidValue;
  const std::size_t tasks = schedule == 2 ? static_cast<std::size_t>(batch.batch_size)
                            : schedule == 1
                                ? batch.shell_pair_count
                                : static_cast<std::size_t>(batch.batch_size) * pair_count;
  const unsigned per_block = schedule == 1 ? threads / 32 : threads;
  if (tasks == 0 || (tasks - 1) / per_block >= std::numeric_limits<int>::max())
    return cudaErrorInvalidValue;
  const unsigned blocks = static_cast<unsigned>((tasks - 1) / per_block + 1);
  if (schedule == 2)
    serial_gradient<<<blocks, threads, 0, stream>>>(batch, weights, active, output_sign, gradient);
  else if (schedule == 1)
    shell_warp_gradient<<<blocks, threads, 0, stream>>>(batch, weights, active, output_sign,
                                                        gradient);
  else
    thread_gradient<<<blocks, threads, 0, stream>>>(batch, pair_first, pair_second, pair_count,
                                                    weights, active, output_sign, gradient);
  return cudaPeekAtLastError();
}

}  // namespace vibeqc::scf
