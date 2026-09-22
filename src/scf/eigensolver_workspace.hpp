#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace vibeqc::scf {

/** Shape-only bound for one serialized ordinary FP64 Xsyevd workspace.
 * Both host and device provider queries must fit before allocation. The fixed
 * floor covers small matrices; spins/items reuse the same workspace in order.
 * This is an admission bound, not an estimate of opaque library allocations.
 */
inline std::size_t ordinary_eigensolver_workspace_allowance(std::size_t n) {
  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  constexpr std::size_t fixed = 1024U * 1024U;
  if (n == 0 || n > maximum / n || n * n > (maximum - fixed) / (16U * sizeof(double)))
    throw std::overflow_error("ordinary eigensolver workspace size overflows");
  return fixed + 16U * n * n * sizeof(double);
}

}  // namespace vibeqc::scf
