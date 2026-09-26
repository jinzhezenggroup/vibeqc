#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "scf/density_factor.hpp"
#include "scf/reference/mean_field.hpp"

namespace {
void require(bool value, const char* reason) {
  if (!value) throw std::runtime_error(reason);
}
template <class F>
void invalid(F&& operation) {
  try {
    operation();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("invalid occupied factor was accepted");
}
}  // namespace

int main() {
  try {
    using namespace vibeqc::scf;
    const DensityFactorIdentity identity{11, 29, 3, 5};
    constexpr std::size_t n = 4, a = 3;
    DensityFittingThreeCenter tensor{n, a, a, std::vector<double>(n * n * a)};
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        for (std::size_t q = 0; q < a; ++q)
          tensor.values[(i * n + j) * a + q] = std::sin(double((i + 1) * (j + 1) + q));
    // Nonorthogonal AO coefficients are intentional: Euclidean orbital
    // orthogonality is not a precondition of D=B B^T in a nonorthogonal basis.
    std::vector<double> orbitals(n * n);
    for (std::size_t i = 0; i < orbitals.size(); ++i) orbitals[i] = std::cos(double(i + 1)) / 3;
    for (auto spin :
         {DensityFactorSpin::Restricted, DensityFactorSpin::Alpha, DensityFactorSpin::Beta}) {
      const double occupation = spin == DensityFactorSpin::Restricted ? 2 : 1;
      for (std::size_t rank : {0U, 1U, 2U, 4U}) {
        std::vector<double> coefficients, occupations(rank, occupation);
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t o = 0; o < rank; ++o) coefficients.push_back(orbitals[i * n + o]);
        OccupiedDensityFactor factor(identity, spin, n, coefficients, occupations);
        const auto density = reference::density_from_orbitals(orbitals, n, rank, occupation);
        require(factor.matches(identity, spin, density),
                "SCF density witness changed evaluation order");
        const auto expected = build_density_fitting_rhf_jk(tensor, density, {false, true}).exchange;
        const auto actual =
            occupied_density_fitting_exchange(tensor, &factor, identity, spin, density);
        require(actual.has_value(), "compatible factor fell back");
        for (std::size_t i = 0; i < expected.size(); ++i)
          require(std::abs(actual->at(i) - expected[i]) < 3e-14, "occupied/dense RI-K mismatch");
        require(!occupied_density_fitting_exchange(tensor, nullptr, identity, spin, density),
                "external density unexpectedly used occupied RI-K");
        for (unsigned field = 0; field < 4; ++field) {
          auto stale = identity;
          if (field == 0) ++stale.basis;
          if (field == 1) ++stale.reference;
          if (field == 2) ++stale.orbital_generation;
          if (field == 3) ++stale.density_generation;
          require(!occupied_density_fitting_exchange(tensor, &factor, stale, spin, density),
                  "stale factor generation/source was accepted");
        }
        const auto other_spin =
            spin == DensityFactorSpin::Alpha ? DensityFactorSpin::Beta : DensityFactorSpin::Alpha;
        require(!occupied_density_fitting_exchange(tensor, &factor, identity, other_spin, density),
                "wrong spin factor was accepted");
        auto changed = density;
        changed[0] += 1e-12;
        require(!occupied_density_fitting_exchange(tensor, &factor, identity, spin, changed),
                "same labels authorized a different density");
        changed[0] = std::numeric_limits<double>::quiet_NaN();
        require(!factor.matches(identity, spin, changed), "nonfinite density matched");
      }
    }
    const std::vector<double> coefficients(n, 0.1);
    for (double occupation : {-1.0, 0.0, 0.5, 1.0, std::numeric_limits<double>::quiet_NaN()})
      invalid([&] {
        OccupiedDensityFactor f(identity, DensityFactorSpin::Restricted, n, coefficients,
                                std::vector<double>{occupation});
      });
    invalid([&] {
      OccupiedDensityFactor f({}, DensityFactorSpin::Alpha, n, coefficients,
                              std::vector<double>{1});
    });
    invalid([&] {
      OccupiedDensityFactor f(identity, DensityFactorSpin::Alpha, n, {}, std::vector<double>{1});
    });
    std::cout << "occupied-factor identity, fallback and dense RI-K parity passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
