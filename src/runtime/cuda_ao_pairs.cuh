#ifndef VIBEQC_RUNTIME_CUDA_AO_PAIRS_CUH
#define VIBEQC_RUNTIME_CUDA_AO_PAIRS_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace vibeqc::runtime::cuda_ao_pairs {

template <unsigned Channels, std::size_t... Indices, class... OutputPointers>
__device__ __forceinline__ void store_symmetric_impl(const double (&values)[Channels],
                                                     std::size_t first, std::size_t second,
                                                     std::index_sequence<Indices...>,
                                                     OutputPointers... outputs) {
  static_assert(sizeof...(OutputPointers) == Channels);
  ((void)(outputs == nullptr ? 0
                             : (outputs[first] = values[Indices],
                                first == second ? 0 : (outputs[second] = values[Indices], 0))),
   ...);
}

template <unsigned Channels, class... OutputPointers>
__device__ __forceinline__ void store_symmetric(const double (&values)[Channels], std::size_t first,
                                                std::size_t second, OutputPointers... outputs) {
  static_assert(sizeof...(OutputPointers) == Channels);
  store_symmetric_impl(values, first, second, std::index_sequence_for<OutputPointers...>{},
                       outputs...);
}

template <class Policy, std::size_t TermCapacity>
__device__ void contract(std::int32_t nbf, const std::int64_t* atom_offsets,
                         const std::int32_t* atomic_numbers, const double* positions,
                         const std::int32_t* shell_atoms,
                         const std::int64_t* shell_primitive_offsets, const std::int32_t* ao_shells,
                         const std::uint8_t* ao_term_counts, const std::uint8_t* ao_term_angular,
                         const double* ao_term_coefficients, const double* primitive_exponents,
                         const double* primitive_coefficients, std::int64_t i, std::int64_t j,
                         typename Policy::Accumulator& result) {
  const auto si = ao_shells[i], sj = ao_shells[j];
  const double* A = positions + 3 * shell_atoms[si];
  const double* B = positions + 3 * shell_atoms[sj];
  result = typename Policy::Accumulator{};
  for (auto a = shell_primitive_offsets[si]; a < shell_primitive_offsets[si + 1]; ++a) {
    for (auto b = shell_primitive_offsets[sj]; b < shell_primitive_offsets[sj + 1]; ++b) {
      typename Policy::PrimitiveGeometry pair{};
      Policy::prepare(pair, primitive_exponents[a], primitive_exponents[b], A, B);
      const double primitive_weight = primitive_coefficients[a] * primitive_coefficients[b];
      for (unsigned ti = 0; ti < ao_term_counts[i]; ++ti) {
        const auto term_i = i * TermCapacity + ti;
        const auto first = Policy::component(ao_term_angular + 3 * term_i);
        for (unsigned tj = 0; tj < ao_term_counts[j]; ++tj) {
          const auto term_j = j * TermCapacity + tj;
          const auto second = Policy::component(ao_term_angular + 3 * term_j);
          const double weight =
              primitive_weight * ao_term_coefficients[term_i] * ao_term_coefficients[term_j];
          Policy::accumulate(result, pair, first, second, weight, atom_offsets, atomic_numbers,
                             positions, i / nbf);
        }
      }
    }
  }
}

template <class Policy, std::size_t TermCapacity, class... OutputPointers>
__device__ void evaluate_pair(std::int32_t nbf, const std::int64_t* atom_offsets,
                              const std::int32_t* atomic_numbers, const double* positions,
                              const std::int32_t* shell_atoms,
                              const std::int64_t* shell_primitive_offsets,
                              const std::int32_t* ao_shells, const std::uint8_t* ao_term_counts,
                              const std::uint8_t* ao_term_angular,
                              const double* ao_term_coefficients, const double* primitive_exponents,
                              const double* primitive_coefficients, std::int64_t i, std::int64_t j,
                              OutputPointers... outputs) {
  static_assert(sizeof...(OutputPointers) == Policy::channels);
  typename Policy::Accumulator result{};
  contract<Policy, TermCapacity>(nbf, atom_offsets, atomic_numbers, positions, shell_atoms,
                                 shell_primitive_offsets, ao_shells, ao_term_counts,
                                 ao_term_angular, ao_term_coefficients, primitive_exponents,
                                 primitive_coefficients, i, j, result);
  const std::size_t n = nbf, system = i / n;
  const std::size_t row = i % n, column = j % n, offset = system * n * n;
  double values[Policy::channels];
  Policy::project(result, values);
  store_symmetric(values, offset + row * n + column, offset + column * n + row, outputs...);
}

template <class Policy, std::size_t TermCapacity, class... OutputPointers>
__device__ __forceinline__ void thread_pairs_body(
    std::int32_t batch_size, std::int32_t nbf, const std::int64_t* atom_offsets,
    const std::int32_t* atomic_numbers, const double* positions, const std::int32_t* shell_atoms,
    const std::int64_t* shell_primitive_offsets, const std::int32_t* ao_shells,
    const std::uint8_t* ao_term_counts, const std::uint8_t* ao_term_angular,
    const double* ao_term_coefficients, const double* primitive_exponents,
    const double* primitive_coefficients, const std::int32_t* pair_first,
    const std::int32_t* pair_second, std::size_t pair_count, OutputPointers... outputs) {
  static_assert(sizeof...(OutputPointers) == Policy::channels);
  const std::size_t task = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (task >= static_cast<std::size_t>(batch_size) * pair_count) return;
  const std::int64_t base = (task / pair_count) * nbf;
  const std::size_t pair = task % pair_count;
  evaluate_pair<Policy, TermCapacity>(
      nbf, atom_offsets, atomic_numbers, positions, shell_atoms, shell_primitive_offsets, ao_shells,
      ao_term_counts, ao_term_angular, ao_term_coefficients, primitive_exponents,
      primitive_coefficients, base + pair_first[pair], base + pair_second[pair], outputs...);
}

template <class Policy, std::size_t TermCapacity, class... OutputPointers>
__device__ __forceinline__ void shell_warp_pairs_body(
    std::int32_t nbf, std::size_t shell_pair_count, const std::int64_t* atom_offsets,
    const std::int32_t* atomic_numbers, const double* positions, const std::int32_t* shell_atoms,
    const std::int64_t* shell_ao_offsets, const std::int64_t* shell_primitive_offsets,
    const std::int32_t* shell_pair_first, const std::int32_t* shell_pair_second,
    const std::int32_t* ao_shells, const std::uint8_t* ao_term_counts,
    const std::uint8_t* ao_term_angular, const double* ao_term_coefficients,
    const double* primitive_exponents, const double* primitive_coefficients,
    OutputPointers... outputs) {
  static_assert(sizeof...(OutputPointers) == Policy::channels);
  const std::size_t task = (std::size_t{blockIdx.x} * blockDim.x + threadIdx.x) / 32;
  if (task >= shell_pair_count) return;
  const auto si = shell_pair_first[task], sj = shell_pair_second[task];
  const auto begin_i = shell_ao_offsets[si], begin_j = shell_ao_offsets[sj];
  const auto count_i = shell_ao_offsets[si + 1] - begin_i;
  const auto count_j = shell_ao_offsets[sj + 1] - begin_j;
  for (std::int64_t component = threadIdx.x % 32; component < count_i * count_j; component += 32) {
    const auto i = begin_i + component / count_j, j = begin_j + component % count_j;
    if (si == sj && i < j) continue;
    evaluate_pair<Policy, TermCapacity>(nbf, atom_offsets, atomic_numbers, positions, shell_atoms,
                                        shell_primitive_offsets, ao_shells, ao_term_counts,
                                        ao_term_angular, ao_term_coefficients, primitive_exponents,
                                        primitive_coefficients, i, j, outputs...);
  }
}

}  // namespace vibeqc::runtime::cuda_ao_pairs

#endif
