#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dft/dispersion/d3_atm.hpp"

namespace {
using vibeqc::dft::dispersion::d3_atm_workspace_elements;
using vibeqc::dft::dispersion::d3_host_tables;
using vibeqc::dft::dispersion::D3ATMParameters;
using vibeqc::dft::dispersion::D3Status;
using vibeqc::dft::dispersion::evaluate_d3_bj_atm;

constexpr std::array<std::int32_t, 4> kNumbers{6, 8, 7, 1};
constexpr std::array<double, 12> kCoordinates{0.0, 0.0, 0.0, 2.5,  0.1, 0.0,
                                              0.6, 2.7, 0.2, -1.2, 0.8, 2.4};
// Independent oracle: dftd3 1.6.0 RationalDampingParam with
// {s6=1, s8=.7875, a1=.4289, a2=4.4407, alp=14}, evaluated as s9=1 minus s9=0.
constexpr double kOracleEnergy = 3.3280853885481534e-08;
constexpr std::array<double, 12> kOracleGradient{
    3.72472344498586336e-09,  -1.85479558309227088e-08, -3.61045307549533526e-08,
    2.60708265967875448e-08,  -2.63615669102038128e-08, 2.56660793687209677e-09,
    -3.96599503356355378e-09, 5.05725536818215585e-08,  -1.60861908260602298e-08,
    -2.58295550081963018e-08, -5.66303094069503694e-09, 4.96241136441330152e-08};

struct Result {
  D3Status status{};
  double energy{};
  std::array<double, 12> gradient{};
};

Result evaluate(const std::array<double, 12>& xyz, D3ATMParameters parameters, bool gradient) {
  std::vector<double> workspace(d3_atm_workspace_elements(kNumbers.size()));
  Result out{};
  out.status = evaluate_d3_bj_atm(kNumbers.size(), kNumbers.data(), xyz.data(), parameters,
                                  d3_host_tables(), workspace.data(), workspace.size(), &out.energy,
                                  gradient ? out.gradient.data() : nullptr);
  return out;
}

void require_close(double actual, double expected, double tolerance, const char* message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << message << ": expected " << expected << ", got " << actual << '\n';
    throw std::runtime_error(message);
  }
}

void test_external_oracle() {
  const auto result = evaluate(kCoordinates, D3ATMParameters{1.0, 0.0, 0.0, 0.0}, true);
  if (result.status != D3Status::success) throw std::runtime_error("ATM oracle evaluation failed");
  require_close(result.energy, kOracleEnergy, 2.0e-18, "ATM oracle energy mismatch");
  for (std::size_t i = 0; i < result.gradient.size(); ++i)
    require_close(result.gradient[i], kOracleGradient[i], 2.0e-17, "ATM oracle gradient mismatch");
}

void test_finite_difference(D3ATMParameters parameters, double tolerance) {
  const auto analytic = evaluate(kCoordinates, parameters, true);
  if (analytic.status != D3Status::success)
    throw std::runtime_error("analytic ATM evaluation failed");
  constexpr double step = 1.0e-5;
  for (std::size_t q = 0; q < kCoordinates.size(); ++q) {
    auto plus = kCoordinates, minus = kCoordinates;
    plus[q] += step;
    minus[q] -= step;
    const auto ep = evaluate(plus, parameters, false);
    const auto em = evaluate(minus, parameters, false);
    if (ep.status != D3Status::success || em.status != D3Status::success)
      throw std::runtime_error("finite-difference ATM evaluation failed");
    const double numeric = (ep.energy - em.energy) / (2.0 * step);
    require_close(analytic.gradient[q], numeric, tolerance, "ATM finite-difference mismatch");
  }
}

void test_zero_scaling() {
  const auto result = evaluate(kCoordinates, D3ATMParameters{0.0, 0.0, 0.0, 0.0}, true);
  if (result.status != D3Status::success || result.energy != 0.0 ||
      !std::all_of(result.gradient.begin(), result.gradient.end(),
                   [](double x) { return x == 0.0; }))
    throw std::runtime_error("s9=0 must disable the standalone ATM contribution exactly");
}

}  // namespace

int main() {
  try {
    test_external_oracle();
    test_finite_difference(D3ATMParameters{1.0, 0.0, 0.0, 0.0}, 3.0e-12);
    // Exercise all three smooth-cutoff product-rule terms as well as CN response.
    test_finite_difference(D3ATMParameters{1.0, 0.0, 4.5, 2.0}, 3.0e-12);
    test_zero_scaling();
    std::cout << "D3(BJ)-ATM oracle, analytic-gradient, smooth-cutoff, and s9=0 tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
