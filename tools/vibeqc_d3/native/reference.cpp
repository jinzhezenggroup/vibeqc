// SPDX-License-Identifier: GPL-3.0-or-later
// Diagnostic C ABI only, intentionally not linked into libvibeqc.
#include <algorithm>
#include <vector>

#include "d3_bj_reference.hpp"
extern "C" int vibeqc_d3_reference(std::int64_t n, const double* input, const double* parameters,
                                   double* output) {
  if (n < 1 || n > 512 || !input || !parameters || !output) return 1;
  try {
    std::vector<double> scratch(16 * n), candidate(1 + 3 * n);
    if (!vibeqc_d3_baseline::evaluate(n, input, parameters, candidate.data(), scratch.data()))
      return 2;
    std::copy(candidate.begin(), candidate.end(), output);
    return 0;
  } catch (...) {
    return 3;
  }
}

extern "C" int vibeqc_d3_backend() { return 0; }
