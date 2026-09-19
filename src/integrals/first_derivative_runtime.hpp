#pragma once

#include <cmath>
#include <cstddef>

namespace vibeqc::integrals {
/** Bounded CPU primitive reduction shared by generated first derivatives.
 * Each record has four exponents, twelve center coordinates, and one fixed
 * external weight (including primitive/component normalization). Mathematical
 * centers remain distinct; the consumer scatters the twelve outputs to atoms.
 * No allocation or derivative algebra lives in this execution template.
 */
template <class Evaluate>
int first_derivative_records(const double* records, std::size_t count, double* output,
                             Evaluate evaluate) {
  if (!records || !output || !count || count > 4096) return 1;
  double sum[12]{};
  for (std::size_t i = 0; i < count; ++i) {
    const auto* record = records + 17 * i;
    for (unsigned j = 0; j < 17; ++j)
      if (!std::isfinite(record[j])) return 1;
    double value[12]{};
    if (!evaluate(record, record + 4, value)) return 1;
    for (unsigned j = 0; j < 12; ++j) sum[j] += record[16] * value[j];
  }
  for (double value : sum)
    if (!std::isfinite(value)) return 1;
  for (unsigned j = 0; j < 12; ++j) output[j] = sum[j];
  return 0;
}
}  // namespace vibeqc::integrals
