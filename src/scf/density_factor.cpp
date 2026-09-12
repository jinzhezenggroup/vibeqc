#include "scf/density_factor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace vibeqc::scf {
namespace {
bool valid_identity(DensityFactorIdentity identity) {
  return identity.basis && identity.reference && identity.orbital_generation &&
         identity.density_generation;
}
std::size_t checked_product(std::size_t first, std::size_t second) {
  if (second && first > std::numeric_limits<std::size_t>::max() / second)
    throw std::invalid_argument("density-factor dimensions overflow size_t");
  return first * second;
}
}  // namespace

OccupiedDensityFactor::OccupiedDensityFactor(DensityFactorIdentity identity, DensityFactorSpin spin,
                                             std::size_t nbf, std::span<const double> coefficients,
                                             std::span<const double> occupations)
    : identity_(identity), spin_(spin), nbf_(nbf) {
  if (!valid_identity(identity) || !nbf || occupations.size() > nbf ||
      coefficients.size() != checked_product(nbf, occupations.size()))
    throw std::invalid_argument("invalid occupied density-factor identity or dimensions");
  if (spin != DensityFactorSpin::Restricted && spin != DensityFactorSpin::Alpha &&
      spin != DensityFactorSpin::Beta)
    throw std::invalid_argument("invalid occupied density-factor spin channel");
  const double occupation = spin == DensityFactorSpin::Restricted ? 2.0 : 1.0;
  for (double value : occupations)
    if (value != occupation)
      throw std::invalid_argument(
          "occupied density factor requires canonical nonnegative occupations");
  for (double value : coefficients)
    if (!std::isfinite(value))
      throw std::invalid_argument("occupied density factor has nonfinite coefficients");
  occupations_.assign(occupations.begin(), occupations.end());
  values_.resize(coefficients.size());
  const auto scale = std::sqrt(occupation);
  for (std::size_t i = 0; i < coefficients.size(); ++i) values_[i] = scale * coefficients[i];
  density_.assign(checked_product(nbf, nbf), 0.0);
  for (std::size_t i = 0; i < nbf; ++i)
    for (std::size_t j = 0; j < nbf; ++j)
      for (std::size_t o = 0; o < rank(); ++o)
        density_[i * nbf + j] +=
            occupation * coefficients[i * rank() + o] * coefficients[j * rank() + o];
  if (!std::all_of(values_.begin(), values_.end(), [](double x) { return std::isfinite(x); }) ||
      !std::all_of(density_.begin(), density_.end(), [](double x) { return std::isfinite(x); }))
    throw std::invalid_argument("occupied density factor overflows its numerical representation");
}

bool OccupiedDensityFactor::matches(DensityFactorIdentity identity, DensityFactorSpin spin,
                                    std::span<const double> density) const noexcept {
  return identity == identity_ && spin == spin_ && density.size() == density_.size() &&
         std::equal(density.begin(), density.end(), density_.begin());
}

std::optional<std::vector<double>> occupied_density_fitting_exchange(
    const DensityFittingThreeCenter& tensor, const OccupiedDensityFactor* factor,
    DensityFactorIdentity identity, DensityFactorSpin spin, std::span<const double> density) {
  if (!factor || tensor.nbf != factor->nbf() || !factor->matches(identity, spin, density))
    return std::nullopt;
  const auto n = tensor.nbf, rank = factor->rank();
  const auto matrix = checked_product(n, n);
  if (!tensor.naux || tensor.values.size() != checked_product(matrix, tensor.naux))
    throw std::invalid_argument("occupied RI-K tensor dimensions are invalid");
  std::vector<double> exchange(matrix, 0.0), panel(checked_product(n, rank));
  for (std::size_t q = 0; q < tensor.naux; ++q) {
    std::fill(panel.begin(), panel.end(), 0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        for (std::size_t o = 0; o < rank; ++o)
          panel[i * rank + o] +=
              tensor.values[(i * n + j) * tensor.naux + q] * factor->values()[j * rank + o];
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        for (std::size_t o = 0; o < rank; ++o)
          exchange[i * n + j] += panel[i * rank + o] * panel[j * rank + o];
  }
  return exchange;
}

}  // namespace vibeqc::scf
