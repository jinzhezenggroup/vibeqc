#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Retained direct eri symmetry contraction helpers.
// Borrow immutable metadata and density/output views; host plans own lifetime.

namespace vibeqc::scf::cuda_execution {

__device__ inline void eri_symmetry_permutation(unsigned permutation, std::size_t i, std::size_t j,
                                                std::size_t k, std::size_t l, std::size_t& a,
                                                std::size_t& b, std::size_t& c, std::size_t& d) {
  switch (permutation) {
    case 0:
      a = i;
      b = j;
      c = k;
      d = l;
      break;
    case 1:
      a = j;
      b = i;
      c = k;
      d = l;
      break;
    case 2:
      a = i;
      b = j;
      c = l;
      d = k;
      break;
    case 3:
      a = j;
      b = i;
      c = l;
      d = k;
      break;
    case 4:
      a = k;
      b = l;
      c = i;
      d = j;
      break;
    case 5:
      a = l;
      b = k;
      c = i;
      d = j;
      break;
    case 6:
      a = k;
      b = l;
      c = j;
      d = i;
      break;
    default:
      a = l;
      b = k;
      c = j;
      d = i;
      break;
  }
}

/** Test uniqueness directly from the canonical pair symmetries. */
__device__ inline bool unique_eri_symmetry_permutation(unsigned permutation, std::size_t i,
                                                       std::size_t j, std::size_t k,
                                                       std::size_t l) {
  const bool pair_swapped = permutation >= 4;
  const bool first_pair_diagonal = pair_swapped ? k == l : i == j;
  const bool second_pair_diagonal = pair_swapped ? i == j : k == l;
  if ((permutation & 1U) != 0 && first_pair_diagonal) return false;
  if ((permutation & 2U) != 0 && second_pair_diagonal) return false;
  return !pair_swapped || i != k || j != l;
}

}  // namespace vibeqc::scf::cuda_execution
