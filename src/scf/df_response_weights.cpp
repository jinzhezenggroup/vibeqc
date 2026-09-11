#include "scf/df_response_weights.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "scf/density_fitting.hpp"

namespace vibeqc::scf {
DensityFittingResponseWeightResources contract_density_fitting_response_weights(
    std::size_t n, std::size_t a, const std::vector<double>& metric,
    const std::vector<double>& inverse, std::span<const DensityFittingDensityResponse> terms,
    double relative_threshold, std::size_t maximum_bytes, std::size_t maximum_auxiliary_tile,
    const std::function<void(std::size_t, std::span<double>)>& read_values,
    const std::function<void(unsigned, runtime::StridedRange, std::span<const double>)>& consume) {
  const auto maximum = std::numeric_limits<std::size_t>::max() / sizeof(double);
  if (!std::isfinite(relative_threshold) || relative_threshold <= 0 || relative_threshold >= 1 ||
      !n || !a || !maximum_bytes || terms.empty() || n > maximum / n || a > maximum / a ||
      n * n > maximum / a || terms.size() > maximum / a || metric.size() != a * a ||
      inverse.size() != a * a || !read_values || !consume)
    throw std::invalid_argument("invalid DF response weight dimensions or budget");
  const auto matrix = n * n, metric_elements = a * a;
  for (const auto& term : terms) {
    if (term.density.size() != matrix || !std::isfinite(term.coulomb_coefficient) ||
        !std::isfinite(term.exchange_coefficient))
      throw std::invalid_argument("invalid HF density response term");
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        if (!std::isfinite(term.density[i * n + j]) ||
            std::abs(term.density[i * n + j] - term.density[j * n + i]) > 1e-10)
          throw std::invalid_argument("HF response requires finite symmetric densities");
  }
  // The spectral map owns several dense metric matrices. Reserve a conservative
  // bound for its eigensolver, temporaries, result and retained-subspace mask.
  // Auxiliary weight blocks are released before that distinct peak occurs.
  const long double spectral_bytes = sizeof(double) * (10.0L * metric_elements + 4.0L * a);
  const long double fixed_bytes = sizeof(double) * (static_cast<long double>(metric_elements) +
                                                    3.0L * matrix + 2.0L * terms.size() * a);
  if (spectral_bytes > maximum_bytes ||
      fixed_bytes + 2.0L * matrix * sizeof(double) > maximum_bytes)
    throw std::bad_alloc();
  const auto available =
      static_cast<std::size_t>((maximum_bytes - fixed_bytes) / (2.0L * matrix * sizeof(double)));
  const auto tile = std::min({a, available, maximum_auxiliary_tile ? maximum_auxiliary_tile : a});
  DensityFittingResponseWeightResources resources;
  resources.auxiliary_tile = tile;
  resources.host_peak_bytes = static_cast<std::size_t>(
      std::max(spectral_bytes, fixed_bytes + 2.0L * tile * matrix * sizeof(double)));
  std::vector<double> bar_inverse(metric_elements, 0.0);
  {
    std::vector<double> values(matrix), temporary(matrix), response(matrix);
    std::vector<double> raw_block(tile * matrix), weight_block(tile * matrix);
    std::vector<double> charges(terms.size() * a, 0.0), potentials(terms.size() * a, 0.0);
    const auto read = [&](std::size_t auxiliary, std::span<double> output) {
      read_values(auxiliary, output);
      ++resources.value_slices;
      if (!std::all_of(output.begin(), output.end(), [](double x) { return std::isfinite(x); }))
        throw std::runtime_error("nonfinite DF values in HF response");
    };
    for (std::size_t q = 0; q < a; ++q) {
      read(q, values);
      for (std::size_t t = 0; t < terms.size(); ++t)
        if (terms[t].coulomb_coefficient != 0)
          for (std::size_t ij = 0; ij < matrix; ++ij)
            charges[t * a + q] += terms[t].density[ij] * values[ij];
    }
    for (std::size_t t = 0; t < terms.size(); ++t)
      for (std::size_t p = 0; p < a; ++p)
        for (std::size_t q = 0; q < a; ++q)
          potentials[t * a + p] += inverse[p * a + q] * charges[t * a + q];

    for (std::size_t begin = 0; begin < a; begin += tile) {
      const auto count = std::min(tile, a - begin);
      std::fill(weight_block.begin(), weight_block.end(), 0.0);
      for (std::size_t p = 0; p < count; ++p) {
        read(begin + p, std::span<double>(raw_block).subspan(p * matrix, matrix));
        for (std::size_t t = 0; t < terms.size(); ++t) {
          const double coefficient = terms[t].coulomb_coefficient;
          if (coefficient == 0) continue;
          for (std::size_t ij = 0; ij < matrix; ++ij)
            weight_block[p * matrix + ij] +=
                coefficient * terms[t].density[ij] * potentials[t * a + begin + p];
          for (std::size_t q = 0; q < a; ++q)
            bar_inverse[(begin + p) * a + q] +=
                0.5 * coefficient * charges[t * a + begin + p] * charges[t * a + q];
        }
      }
      for (std::size_t q = 0; q < a; ++q) {
        read(q, values);
        for (const auto& term : terms) {
          if (term.exchange_coefficient == 0) continue;
          // R_Q = D^T A_Q D. Symmetric HF densities make Q_PQ=A_P:R_Q
          // symmetric, so both appearances of A give the factor two below.
          std::fill(temporary.begin(), temporary.end(), 0.0);
          std::fill(response.begin(), response.end(), 0.0);
          for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
              for (std::size_t k = 0; k < n; ++k)
                temporary[i * n + j] += values[i * n + k] * term.density[k * n + j];
          for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
              for (std::size_t k = 0; k < n; ++k)
                response[i * n + j] += term.density[k * n + i] * temporary[k * n + j];
          for (std::size_t p = 0; p < count; ++p) {
            const double scale = -2 * term.exchange_coefficient * inverse[(begin + p) * a + q];
            double quadratic = 0.0;
            for (std::size_t ij = 0; ij < matrix; ++ij) {
              quadratic += raw_block[p * matrix + ij] * response[ij];
              weight_block[p * matrix + ij] += scale * response[ij];
            }
            bar_inverse[(begin + p) * a + q] -= term.exchange_coefficient * quadratic;
          }
        }
      }
      consume(0, {begin, matrix, 1, a},
              std::span<const double>(weight_block).first(count * matrix));
      ++resources.weight_tiles;
    }
  }
  const auto bar_metric =
      density_fitting_metric_inverse_response(metric, inverse, bar_inverse, a, relative_threshold);
  consume(1, {}, bar_metric);
  ++resources.weight_tiles;
  return resources;
}
}  // namespace vibeqc::scf
