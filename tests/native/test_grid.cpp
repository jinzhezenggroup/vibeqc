// Analytic Gaussian derivative gate, independent of the Python/libcint fixtures.
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>

#include "dft/ao_grid.hpp"
#include "molecule/basis.hpp"

int main() {
  try {
    vibeqc::core::System system;
    system.atoms = {{2, {0, 0, 0}}};
    system.shells = {{0, 0, {{0.7, 1}}}};
    std::string detail;
    if (vibeqc::molecule::validate_and_normalize(system, detail) != VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail);
    vibeqc::dft::AoBasis basis(system);
    // Destroy the original topology to test that the evaluator owns its state.
    system.atoms.clear();
    system.shells.clear();
    const std::array<double, 3> point{0.3, -0.2, 0.1};
    std::array<double, 20> jets{};
    basis.evaluate(point.data(), 1, 3, 0, 1, jets.data(), jets.size());
    const double a = 0.7;
    const double value = std::pow(2 * a / std::numbers::pi, 0.75) * std::exp(-a * 0.14);
    std::size_t index = 0;
    for (unsigned degree = 0; degree <= 3; ++degree) {
      for (const auto& d : vibeqc::molecule::cartesian_components(degree)) {
        double exact = value;
        for (unsigned k = 0; k < 3; ++k) {
          const double x = point[k];
          if (d[k] == 1) exact *= -2 * a * x;
          if (d[k] == 2) exact *= 4 * a * a * x * x - 2 * a;
          if (d[k] == 3) exact *= 12 * a * a * x - 8 * a * a * a * x * x * x;
        }
        if (std::abs(jets[index++] - exact) > 1e-13)
          throw std::runtime_error("analytic Gaussian jet mismatch");
      }
    }
    basis.evaluate(nullptr, 0, 3, 1, 0, nullptr, 0);
    bool rejected = false;
    try {
      basis.evaluate(point.data(), std::numeric_limits<std::size_t>::max(), 3, 0, 1, jets.data(),
                     20);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected) throw std::runtime_error("AO tile overflow was accepted");
    std::cout << "Owned Gaussian AO jets through order three passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
