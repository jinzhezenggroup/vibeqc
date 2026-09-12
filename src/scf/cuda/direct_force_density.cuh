#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_eri_symmetry.cuh"
#include "scf/cuda/matrix_index.cuh"

// Retained direct force density contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace vibeqc::scf::cuda_execution {

/** Exact symmetry-reduced density coefficient for one force AO quartet. */
template <bool Unrestricted>
__device__ __forceinline__ double direct_force_density_coefficient(
    std::size_t n, std::size_t physical_offset, std::size_t spin_offset, const double* density,
    std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
  const std::size_t matrix_size = n * n;
  double coefficient = 0.0;
  for (unsigned permutation = 0; permutation < 8; ++permutation) {
    if (!unique_eri_symmetry_permutation(permutation, i, j, k, l)) {
      continue;
    }
    std::size_t a = 0;
    std::size_t b = 0;
    std::size_t c = 0;
    std::size_t d = 0;
    eri_symmetry_permutation(permutation, i, j, k, l, a, b, c, d);
    const std::size_t ab = matrix_index(a, b, n);
    const std::size_t ac = matrix_index(a, c, n);
    const std::size_t cd = matrix_index(c, d, n);
    const std::size_t bd = matrix_index(b, d, n);
    if constexpr (Unrestricted) {
      const double total_ab = density[spin_offset + ab] + density[spin_offset + matrix_size + ab];
      const double total_cd = density[spin_offset + cd] + density[spin_offset + matrix_size + cd];
      coefficient += 0.5 * total_ab * total_cd;
      coefficient -=
          0.5 * (density[spin_offset + ac] * density[spin_offset + bd] +
                 density[spin_offset + matrix_size + ac] * density[spin_offset + matrix_size + bd]);
    } else {
      coefficient += 0.5 * density[physical_offset + ab] * density[physical_offset + cd] -
                     0.25 * density[physical_offset + ac] * density[physical_offset + bd];
    }
  }
  return coefficient;
}

}  // namespace vibeqc::scf::cuda_execution
