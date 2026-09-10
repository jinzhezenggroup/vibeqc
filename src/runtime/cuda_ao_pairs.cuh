#ifndef VIBEQC_RUNTIME_CUDA_AO_PAIRS_CUH
#define VIBEQC_RUNTIME_CUDA_AO_PAIRS_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::runtime::cuda_ao_pairs {

/** Borrowed full symmetric matrices; a null channel suppresses that output.
 * A unique lower-triangular pair owner writes both entries, the diagonal once.
 * The generated policy defines channel meaning and any linear combinations.
 */
template <unsigned Channels>
struct Outputs {
  double* data[Channels];
};

template <unsigned Channels>
__device__ void store_symmetric(Outputs<Channels> outputs, const double (&values)[Channels],
                                std::size_t first, std::size_t second) {
#pragma unroll
  for (unsigned c = 0; c < Channels; ++c) {
    if (outputs.data[c]) {
      outputs.data[c][first] = values[c];
      if (first != second) outputs.data[c][second] = values[c];
    }
  }
}

/** Traverse normalized primitive and sparse Cartesian AO-expansion pairs.
 * View indices are global within a packed batch. TermCapacity is the basis
 * layer's stride, not an operator/shell specialization. No radial normalization
 * is applied here: primitive and expansion coefficients already include it.
 * Policy owns scalar geometry, operator evaluation and accumulated channels.
 * Keeping primitive preparation outside the term loops allows geometry reuse.
 */
template <class Policy, std::size_t TermCapacity, class View>
__device__ typename Policy::Accumulator contract(const View& batch, std::int64_t i,
                                                 std::int64_t j) {
  const auto si = batch.ao_shells[i], sj = batch.ao_shells[j];
  const double* A = batch.positions + 3 * batch.shell_atoms[si];
  const double* B = batch.positions + 3 * batch.shell_atoms[sj];
  typename Policy::Accumulator result{};
  for (auto a = batch.shell_primitive_offsets[si]; a < batch.shell_primitive_offsets[si + 1]; ++a) {
    for (auto b = batch.shell_primitive_offsets[sj]; b < batch.shell_primitive_offsets[sj + 1];
         ++b) {
      const auto pair =
          Policy::prepare(batch.primitive_exponents[a], batch.primitive_exponents[b], A, B);
      const double primitive_weight =
          batch.primitive_coefficients[a] * batch.primitive_coefficients[b];
      for (unsigned ti = 0; ti < batch.ao_term_counts[i]; ++ti) {
        const auto term_i = i * TermCapacity + ti;
        const auto first = Policy::component(batch.ao_term_angular + 3 * term_i);
        for (unsigned tj = 0; tj < batch.ao_term_counts[j]; ++tj) {
          const auto term_j = j * TermCapacity + tj;
          const auto second = Policy::component(batch.ao_term_angular + 3 * term_j);
          const double weight = primitive_weight * batch.ao_term_coefficients[term_i] *
                                batch.ao_term_coefficients[term_j];
          Policy::accumulate(result, pair, first, second, weight, batch, i / batch.nbf);
        }
      }
    }
  }
  return result;
}

template <class Policy, std::size_t TermCapacity, class View>
__device__ void evaluate_pair(const View& batch, std::int64_t i, std::int64_t j,
                              Outputs<Policy::channels> outputs) {
  const auto result = contract<Policy, TermCapacity>(batch, i, j);
  const std::size_t n = batch.nbf, system = i / n;
  const std::size_t row = i % n, column = j % n, offset = system * n * n;
  double values[Policy::channels];
  Policy::project(result, values);
  store_symmetric(outputs, values, offset + row * n + column, offset + column * n + row);
}

/** Thread and shell-warp schedules share exactly the same pair evaluator.
 * AO-pair maps are local to one system. Shell-pair maps and AO offsets are
 * already expanded across the batch. Maps must contain each pair exactly once;
 * their construction/validation belongs to the caller's prepared topology.
 */
template <class Policy, std::size_t TermCapacity, class View>
__global__ void thread_pairs(View batch, const std::int32_t* pair_first,
                             const std::int32_t* pair_second, std::size_t pair_count,
                             Outputs<Policy::channels> outputs) {
  const std::size_t task = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (task >= static_cast<std::size_t>(batch.batch_size) * pair_count) return;
  const std::int64_t base = (task / pair_count) * batch.nbf;
  const std::size_t pair = task % pair_count;
  evaluate_pair<Policy, TermCapacity>(batch, base + pair_first[pair], base + pair_second[pair],
                                      outputs);
}

template <class Policy, std::size_t TermCapacity, class View>
__global__ void shell_warp_pairs(View batch, Outputs<Policy::channels> outputs) {
  const std::size_t task = (std::size_t{blockIdx.x} * blockDim.x + threadIdx.x) / 32;
  if (task >= batch.shell_pair_count) return;
  const auto si = batch.shell_pair_first[task], sj = batch.shell_pair_second[task];
  const auto begin_i = batch.shell_ao_offsets[si], begin_j = batch.shell_ao_offsets[sj];
  const auto count_i = batch.shell_ao_offsets[si + 1] - begin_i;
  const auto count_j = batch.shell_ao_offsets[sj + 1] - begin_j;
  for (std::int64_t component = threadIdx.x % 32; component < count_i * count_j; component += 32) {
    const auto i = begin_i + component / count_j, j = begin_j + component % count_j;
    if (si == sj && i < j) continue;
    evaluate_pair<Policy, TermCapacity>(batch, i, j, outputs);
  }
}

}  // namespace vibeqc::runtime::cuda_ao_pairs

#endif
