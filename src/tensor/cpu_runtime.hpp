#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// Generated FP64 programs own the sizes. Only publish after every SSA value
// passes its finite check. The budget includes adapter staging and snapshots.
namespace vibeqc_tensor_cpu {
template <class Evaluate>
int run(const double* input, size_t ni, double* output, size_t no, size_t budget,
        size_t expected_input, size_t expected_output, size_t arena_count, size_t required_bytes,
        Evaluate evaluate) noexcept {
  if (ni != expected_input || no != expected_output || budget < required_bytes || (ni && !input) ||
      (no && !output))
    return -1;
  try {
    for (size_t i = 0; i < ni; ++i)
      if (!std::isfinite(input[i])) return -2;
    std::vector<double> arena(arena_count);
    if (!evaluate(input, arena.data())) return -2;
    // Generated evaluation stores the detached outputs at the arena tail.
    if (no) std::copy_n(arena.data() + arena_count - no, no, output);
    return 0;
  } catch (...) {
    return -3;
  }
}
}  // namespace vibeqc_tensor_cpu
