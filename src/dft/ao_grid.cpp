#include "dft/ao_grid.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <stdexcept>

#include "molecule/basis.hpp"

namespace vibeqc::dft {
namespace {
std::size_t multiply(std::size_t a, std::size_t b) {
  if (b && a > SIZE_MAX / b) throw std::invalid_argument("AO tile size overflow");
  return a * b;
}

// Repeated polynomial differentiation is independent of the CUDA Leibniz
// formula. Keep the Gaussian exponential outside the polynomial recurrence.
double differentiated_power(unsigned l, unsigned derivative, double alpha, double x) {
  std::array<double, 7> coefficients{};
  coefficients[l] = 1;
  unsigned degree = l;
  for (unsigned d = 0; d < derivative; ++d) {
    std::array<double, 7> next{};
    for (unsigned k = 0; k <= degree; ++k) {
      if (k) next[k - 1] += k * coefficients[k];
      next[k + 1] -= 2 * alpha * coefficients[k];
    }
    coefficients = next;
    ++degree;
  }
  double result = coefficients[degree];
  while (degree) result = result * x + coefficients[--degree];
  return result;
}
}  // namespace

AoBasis::AoBasis(const core::System& system) {
  natom = system.atoms.size();
  nao = molecule::ao_count(system);
  for (const auto& shell : system.shells) {
    if (shell.angular_momentum > 3) throw std::invalid_argument("AO grids support through f");
    nprimitive += shell.primitives.size();
  }
  if (!natom || !nao || natom > INT_MAX || nao > INT_MAX || nprimitive > INT_MAX)
    throw std::invalid_argument("invalid AO grid basis dimensions");
  packed.resize(3 * natom + 2 * nprimitive + 16 * nao, 0);
  for (std::size_t a = 0; a < natom; ++a)
    std::copy(system.atoms[a].position.begin(), system.atoms[a].position.end(),
              packed.begin() + 3 * a);
  auto* primitives = packed.data() + 3 * natom;
  auto* aos = primitives + 2 * nprimitive;
  std::size_t primitive_begin = 0, ao = 0;
  for (const auto& shell : system.shells) {
    for (std::size_t p = 0; p < shell.primitives.size(); ++p) {
      const auto& primitive = shell.primitives[p];
      if (!std::isfinite(primitive.exponent) || !std::isfinite(primitive.coefficient))
        throw std::invalid_argument("nonfinite normalized AO primitive");
      primitives[2 * (primitive_begin + p)] = primitive.exponent;
      primitives[2 * (primitive_begin + p) + 1] = primitive.coefficient;
    }
    for (const auto& expansion :
         molecule::ao_expansions(shell.angular_momentum, system.basis_representation)) {
      auto* record = aos + 16 * ao++;
      record[0] = shell.atom_index;
      record[1] = primitive_begin;
      record[2] = shell.primitives.size();
      record[3] = expansion.size();
      for (std::size_t t = 0; t < expansion.size(); ++t) {
        for (unsigned k = 0; k < 3; ++k) record[4 + 4 * t + k] = expansion[t].component[k];
        record[7 + 4 * t] = expansion[t].coefficient *
                            molecule::cartesian_component_normalization(expansion[t].component);
      }
    }
    primitive_begin += shell.primitives.size();
  }
}

void AoBasis::evaluate(const double* points, std::size_t npoint, unsigned order,
                       std::size_t ao_begin, std::size_t count, double* output,
                       std::size_t elements, const std::size_t* ao_ids) const {
  if (order > 3 || ao_begin > nao || count > nao - ao_begin)
    throw std::invalid_argument("invalid AO jet order or AO slice");
  const std::size_t jets = (order + 1) * (order + 2) * (order + 3) / 6;
  if (elements != multiply(multiply(jets, npoint), count) || (npoint && !points) ||
      (elements && !output))
    throw std::invalid_argument("invalid AO jet buffer");
  // A spatial mask selects records before scientific evaluation. Requiring
  // sorted unique IDs makes local D[I,I] and matrix scatter unambiguous.
  if (ao_ids)
    for (std::size_t i = 0; i < count; ++i)
      if (ao_ids[i] >= nao || (i && ao_ids[i] <= ao_ids[i - 1]))
        throw std::invalid_argument("invalid selected AO map");
  for (std::size_t i = 0; i < multiply(3, npoint); ++i)
    if (!std::isfinite(points[i])) throw std::invalid_argument("nonfinite grid point");
  const auto* primitives = packed.data() + 3 * natom;
  const auto* aos = primitives + 2 * nprimitive;
  std::size_t jet = 0;
  for (unsigned degree = 0; degree <= order; ++degree) {
    for (const auto& derivative : molecule::cartesian_components(degree)) {
      for (std::size_t point = 0; point < npoint; ++point) {
        for (std::size_t ao = 0; ao < count; ++ao) {
          const auto* record = aos + 16 * (ao_ids ? ao_ids[ao] : ao_begin + ao);
          const auto atom = static_cast<std::size_t>(record[0]);
          std::array<double, 3> r{};
          double r2 = 0;
          for (unsigned k = 0; k < 3; ++k) {
            r[k] = points[3 * point + k] - packed[3 * atom + k];
            r2 += r[k] * r[k];
          }
          double value = 0;
          const auto first = static_cast<std::size_t>(record[1]);
          const auto end = first + static_cast<std::size_t>(record[2]);
          for (auto p = first; p < end; ++p) {
            const double alpha = primitives[2 * p];
            const double radial = primitives[2 * p + 1] * std::exp(-alpha * r2);
            // Exact exponential underflow contributes zero, without evaluating
            // potentially overflowing far-field polynomial factors.
            if (radial == 0) continue;
            for (unsigned t = 0; t < static_cast<unsigned>(record[3]); ++t) {
              double term = radial * record[7 + 4 * t];
              for (unsigned k = 0; k < 3; ++k)
                term *= differentiated_power(static_cast<unsigned>(record[4 + 4 * t + k]),
                                             derivative[k], alpha, r[k]);
              value += term;
            }
          }
          if (!std::isfinite(value)) throw std::runtime_error("nonfinite AO jet result");
          output[(jet * npoint + point) * count + ao] = value;
        }
      }
      ++jet;
    }
  }
}
}  // namespace vibeqc::dft
