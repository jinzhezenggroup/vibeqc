#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vibeqc::integrals {
// Execution only: Program supplies the compiler-generated value/derivative DAG.
// A record contains positive exponents, Bohr centers, then a fixed coefficient.
// Both invalid inputs and late numerical failures leave output unchanged.
template <class Program>
int contract_first_components(const double* records, std::size_t count, std::size_t stride,
                              double* output, std::size_t output_size) {
  constexpr auto inputs = Program::inputs;
  constexpr auto width = Program::outputs;
  constexpr auto components = Program::components;
  if (!output || (count && !records) || stride != inputs + 1 || output_size != components * width ||
      count > std::numeric_limits<std::size_t>::max() / stride)
    return 1;
  std::array<double, components * width> accumulated{};
  for (std::size_t record = 0; record < count; ++record) {
    const auto* input = records + record * stride;
    for (std::size_t j = 0; j < stride; ++j)
      if (!std::isfinite(input[j]) || (j < Program::exponents && input[j] <= 0)) return 1;
    for (std::size_t component = 0; component < components; ++component) {
      std::array<double, width> candidate{};
      if (!Program::evaluate(input, component, candidate.data())) return 5;
      for (std::size_t j = 0; j < width; ++j) {
        auto& sum = accumulated[component * width + j];
        const double term = input[inputs] * candidate[j];
        sum += term;
        if (!std::isfinite(candidate[j]) || !std::isfinite(term) || !std::isfinite(sum)) return 5;
      }
    }
  }
  for (std::size_t j = 0; j < output_size; ++j) output[j] = accumulated[j];
  return 0;
}
}  // namespace vibeqc::integrals
