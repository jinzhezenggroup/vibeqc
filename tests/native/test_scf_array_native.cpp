#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "scf/reference/mean_field.hpp"

namespace {
using vibeqc::scf::reference::Matrix;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

Matrix density_oracle(const Matrix& coefficients, std::size_t n, std::size_t occupied,
                      double weight) {
  Matrix result(n * n, 0.0);
  for (std::size_t mu = 0; mu < n; ++mu)
    for (std::size_t nu = 0; nu < n; ++nu)
      for (std::size_t orbital = 0; orbital < occupied; ++orbital)
        result[mu * n + nu] +=
            weight * coefficients[mu * n + orbital] * coefficients[nu * n + orbital];
  return result;
}

Matrix weighted_oracle(const Matrix& coefficients, const std::vector<double>& energies,
                       std::size_t n, std::size_t occupied, double weight) {
  Matrix result(n * n, 0.0);
  for (std::size_t mu = 0; mu < n; ++mu)
    for (std::size_t nu = 0; nu < n; ++nu)
      for (std::size_t orbital = 0; orbital < occupied; ++orbital)
        result[mu * n + nu] += weight * energies[orbital] * coefficients[mu * n + orbital] *
                               coefficients[nu * n + orbital];
  return result;
}

void exact_equal(const Matrix& actual, const Matrix& expected, const char* message) {
  require(actual.size() == expected.size(), "SCF array-native result size changed");
  for (std::size_t i = 0; i < actual.size(); ++i) require(actual[i] == expected[i], message);
}
}  // namespace

int main() {
  try {
    constexpr std::size_t n = 4;
    const Matrix coefficients{
        0.75, -0.25, 0.125, 0.5, -0.4, 0.3, -0.2, 0.1, 0.2, 0.6, -0.1, 0.7, -0.3, 0.15, 0.55, -0.45,
    };
    const std::vector<double> energies{-1.2, -0.35, 0.4, 1.1};

    for (const auto [occupied, weight] :
         std::array<std::pair<std::size_t, double>, 4>{{{0, 2.0}, {1, 2.0}, {3, 2.0}, {2, 1.0}}}) {
      const auto density =
          vibeqc::scf::reference::density_from_orbitals(coefficients, n, occupied, weight);
      const auto weighted = vibeqc::scf::reference::energy_weighted_density(coefficients, energies,
                                                                            n, occupied, weight);
      exact_equal(density, density_oracle(coefficients, n, occupied, weight),
                  "generated Array density changed FP64 evaluation");
      exact_equal(weighted, weighted_oracle(coefficients, energies, n, occupied, weight),
                  "generated Array weighted density changed FP64 evaluation");
    }

    std::cout << "Array frontend native SCF density parity passed\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
