#ifndef VIBEQC_RUNTIME_STRIDED_RANGE_HPP
#define VIBEQC_RUNTIME_STRIDED_RANGE_HPP

#include <cstddef>

namespace vibeqc::runtime {
/** Borrowed flat weights mapped onto a two-dimensional strided destination.
 * A complete row contains row_length entries; a final row may be partial.
 * Keeping both strides permits bounded transposed tiles without materializing
 * a second dense tensor. Callers validate dimensions before device execution.
 */
struct StridedRange {
  std::size_t offset{}, row_length{1}, row_stride{1}, column_stride{1};
#if defined(__CUDACC__)
  __host__ __device__
#endif
      constexpr std::size_t index(std::size_t element) const {
    return offset + (element / row_length) * row_stride + (element % row_length) * column_stride;
  }
};
}  // namespace vibeqc::runtime
#endif
