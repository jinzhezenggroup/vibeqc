#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace vibeqc::integrals {

// Lane-parallel execution only. Program owns the generated scientific DAG.
// Active AoS records are packed into an internal SoA tile; inactive tail lanes
// receive finite, positive-exponent padding and have zero contraction weight.
template <class Program>
int contract_first_components_cpu_lanes(const double* records, std::size_t count,
                                        std::size_t stride, double* output,
                                        std::size_t output_size) {
  constexpr auto inputs = Program::inputs;
  constexpr auto width = Program::outputs;
  constexpr auto components = Program::components;
  constexpr auto lanes = Program::lanes;
  if (!output || (count && !records) || stride != inputs + 1 || output_size != components * width ||
      count > std::numeric_limits<std::size_t>::max() / stride)
    return 1;

  alignas(64) std::array<double, inputs * lanes> packed{};
  alignas(64) std::array<double, lanes> weights{};
  alignas(64) std::array<double, width * lanes> candidate{};
  std::array<double, components * width> accumulated{};

  for (std::size_t base = 0; base < count; base += lanes) {
    const std::size_t active = (count - base < lanes) ? count - base : lanes;
    for (std::size_t lane = 0; lane < lanes; ++lane) {
      if (lane < active) {
        const auto* record = records + (base + lane) * stride;
        for (std::size_t j = 0; j < stride; ++j)
          if (!std::isfinite(record[j]) || (j < Program::exponents && record[j] <= 0)) return 1;
        for (std::size_t field = 0; field < inputs; ++field)
          packed[field * lanes + lane] = record[field];
        weights[lane] = record[inputs];
      } else {
        for (std::size_t field = 0; field < inputs; ++field)
          packed[field * lanes + lane] = field < Program::exponents ? 1.0 : 0.0;
        weights[lane] = 0.0;
      }
    }

    for (std::size_t component = 0; component < components; ++component) {
      if (!Program::evaluate(packed.data(), component, candidate.data())) return 5;
      for (std::size_t out = 0; out < width; ++out) {
        auto& sum = accumulated[component * width + out];
        for (std::size_t lane = 0; lane < active; ++lane) {
          const double value = candidate[out * lanes + lane];
          const double term = weights[lane] * value;
          sum += term;
          if (!std::isfinite(value) || !std::isfinite(term) || !std::isfinite(sum)) return 5;
        }
      }
    }
  }

  for (std::size_t j = 0; j < output_size; ++j) output[j] = accumulated[j];
  return 0;
}

}  // namespace vibeqc::integrals
