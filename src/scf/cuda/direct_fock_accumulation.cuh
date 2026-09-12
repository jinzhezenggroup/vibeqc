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

// Retained direct fock accumulation contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace vibeqc::scf::cuda_execution {

/** Scatter one symmetry-canonical ERI into the direct RHF/UHF Fock matrix. */
template <bool Unrestricted>
__device__ __forceinline__ void accumulate_direct_fock_integral(
    std::size_t n, std::size_t physical_offset, std::size_t spin_offset, const double* density,
    double* fock, std::size_t i, std::size_t j, std::size_t k, std::size_t l, double integral) {
  const std::size_t matrix_size = n * n;
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
      const double alpha_cd = density[spin_offset + cd];
      const double beta_cd = density[spin_offset + matrix_size + cd];
      const double total_cd = alpha_cd + beta_cd;
      if (total_cd != 0.0) {
        atomicAdd(fock + spin_offset + ab, total_cd * integral);
        atomicAdd(fock + spin_offset + matrix_size + ab, total_cd * integral);
      }
      const double alpha_bd = density[spin_offset + bd];
      const double beta_bd = density[spin_offset + matrix_size + bd];
      if (alpha_bd != 0.0) {
        atomicAdd(fock + spin_offset + ac, -alpha_bd * integral);
      }
      if (beta_bd != 0.0) {
        atomicAdd(fock + spin_offset + matrix_size + ac, -beta_bd * integral);
      }
    } else {
      const double density_cd = density[physical_offset + cd];
      const double density_bd = density[physical_offset + bd];
      if (density_cd != 0.0) {
        atomicAdd(fock + physical_offset + ab, density_cd * integral);
      }
      if (density_bd != 0.0) {
        atomicAdd(fock + physical_offset + ac, -0.5 * density_bd * integral);
      }
    }
  }
}

}  // namespace vibeqc::scf::cuda_execution
