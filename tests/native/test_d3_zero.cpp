#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dft/dispersion/d3_model.hpp"

namespace {
using vibeqc::dft::dispersion::d3_host_tables;
using vibeqc::dft::dispersion::d3_pair_term;
using vibeqc::dft::dispersion::d3_zero_workspace_elements;
using vibeqc::dft::dispersion::D3Damping;
using vibeqc::dft::dispersion::D3ModelParameters;
using vibeqc::dft::dispersion::D3PairTerm;
using vibeqc::dft::dispersion::D3Status;
using vibeqc::dft::dispersion::D3ZeroParameters;
using vibeqc::dft::dispersion::evaluate_d3_zero;

constexpr std::array<std::int32_t, 4> kNumbers{6, 8, 7, 1};
constexpr std::array<double, 12> kCoordinates{0.0, 0.0, 0.0, 2.5,  0.1, 0.0,
                                              0.6, 2.7, 0.2, -1.2, 0.8, 2.4};

struct Oracle {
  D3ZeroParameters parameters;
  double energy;
  std::array<double, 12> gradient;
};

// Independent simple-dftd3 1.4.0 oracle at the repository-pinned parameter
// revision, with ATM explicitly disabled (s9=0).
constexpr Oracle kPbeOracle{
    {1.0, 0.722, 1.217, 1.0, 14.0, 0.0, 0.0, 0.0},
    -2.585095064181989e-4,
    {-8.589549018568496e-05, 6.384438209854598e-05, 1.804755086632350e-04, -1.532347867723881e-04,
     7.361363487402514e-05, 7.626987717305705e-05, 1.124628703587642e-05, -8.424167738597969e-05,
     2.800093182458554e-05, 2.278839899221966e-04, -5.321633958659144e-05, -2.847463176608776e-04}};

// Independent dftd3 1.6.0 ZeroDampingParam(method="pbe0", atm=False)
// oracle at the same geometry. Coordinates are bohr; energy/gradient are
// Hartree and Hartree/bohr.
constexpr Oracle kPbe0Oracle{
    {1.0, 0.928, 1.287, 1.0, 14.0, 0.0, 0.0, 0.0},
    -3.0546188757657987e-4,
    {-1.0134734075343157e-04, 7.555506460862482e-05, 2.1322571013266314e-04,
     -1.5939358235838346e-04, 8.332486543062817e-05, 7.578382712325813e-05, 1.3211801614035793e-05,
     -1.0067348683521795e-04, 3.357846633457028e-05, 2.4752912149777925e-04,
     -5.8206443204035034e-05, -3.225880035904916e-04}};

constexpr Oracle kSlaterDiracOracle{
    {1.0, -1.957, 0.999, 0.697, 14.0, 0.0, 0.0, 0.0},
    1.8354418869906384e-2,
    {3.705073075728348e-03, -7.326828792164451e-03, -1.333819401190468e-02, 8.118386754879386e-03,
     -1.061823483738393e-02, 5.949873810480457e-04, -5.769683369674431e-03, 1.581685325418789e-02,
     -1.043678153203872e-03, -6.053776460933303e-03, 2.128210375360490e-03, 1.378688478406050e-02}};

struct Result {
  D3Status status{};
  double energy{};
  std::array<double, 12> gradient{};
};

Result evaluate(const std::array<double, 12>& xyz, D3ZeroParameters parameters, bool gradient) {
  std::vector<double> workspace(d3_zero_workspace_elements(kNumbers.size()));
  Result out{};
  out.status = evaluate_d3_zero(kNumbers.size(), kNumbers.data(), xyz.data(), parameters,
                                d3_host_tables(), workspace.data(), workspace.size(), &out.energy,
                                gradient ? out.gradient.data() : nullptr);
  return out;
}

void require_close(double actual, double expected, double tolerance, const char* message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << message << ": expected " << expected << ", got " << actual << "\n";
    throw std::runtime_error(message);
  }
}

void test_external_oracle(const Oracle& oracle) {
  const auto result = evaluate(kCoordinates, oracle.parameters, true);
  if (result.status != D3Status::success)
    throw std::runtime_error("D3(0) oracle evaluation failed");
  require_close(result.energy, oracle.energy, 3.0e-16, "D3(0) oracle energy mismatch");
  for (std::size_t i = 0; i < result.gradient.size(); ++i)
    require_close(result.gradient[i], oracle.gradient[i], 5.0e-15,
                  "D3(0) oracle gradient mismatch");
}

void test_finite_difference(D3ZeroParameters parameters, double tolerance) {
  const auto analytic = evaluate(kCoordinates, parameters, true);
  if (analytic.status != D3Status::success)
    throw std::runtime_error("analytic D3(0) evaluation failed");
  constexpr double step = 1.0e-5;
  for (std::size_t q = 0; q < kCoordinates.size(); ++q) {
    auto plus = kCoordinates, minus = kCoordinates;
    plus[q] += step;
    minus[q] -= step;
    const auto ep = evaluate(plus, parameters, false);
    const auto em = evaluate(minus, parameters, false);
    if (ep.status != D3Status::success || em.status != D3Status::success)
      throw std::runtime_error("finite-difference D3(0) evaluation failed");
    const double numeric = (ep.energy - em.energy) / (2.0 * step);
    require_close(analytic.gradient[q], numeric, tolerance, "D3(0) finite-difference mismatch");
  }
}

void test_multistep_finite_difference(D3ZeroParameters parameters, double tolerance) {
  const auto analytic = evaluate(kCoordinates, parameters, true);
  if (analytic.status != D3Status::success)
    throw std::runtime_error("analytic multistep D3(0) evaluation failed");
  for (double step : {2.0e-4, 7.0e-5, 2.0e-5}) {
    for (std::size_t q = 0; q < kCoordinates.size(); ++q) {
      auto plus = kCoordinates, minus = kCoordinates;
      plus[q] += step;
      minus[q] -= step;
      const auto ep = evaluate(plus, parameters, false);
      const auto em = evaluate(minus, parameters, false);
      if (ep.status != D3Status::success || em.status != D3Status::success)
        throw std::runtime_error("multistep finite-difference D3(0) evaluation failed");
      const double numeric = (ep.energy - em.energy) / (2.0 * step);
      require_close(analytic.gradient[q], numeric, tolerance,
                    "D3(0) multistep finite-difference mismatch");
    }
  }
}

void test_translation_invariance() {
  auto shifted = kCoordinates;
  for (std::size_t atom = 0; atom < kNumbers.size(); ++atom) {
    shifted[3 * atom] += 2.3;
    shifted[3 * atom + 1] -= 1.7;
    shifted[3 * atom + 2] += 0.4;
  }
  const auto reference = evaluate(kCoordinates, kPbeOracle.parameters, true);
  const auto translated = evaluate(shifted, kPbeOracle.parameters, true);
  require_close(translated.energy, reference.energy, 3.0e-15, "D3(0) translation energy mismatch");
  std::array<double, 3> total{};
  for (std::size_t q = 0; q < reference.gradient.size(); ++q) {
    require_close(translated.gradient[q], reference.gradient[q], 3.0e-13,
                  "D3(0) translation gradient mismatch");
    total[q % 3] += reference.gradient[q];
  }
  for (double component : total)
    require_close(component, 0.0, 3.0e-13, "D3(0) total gradient must vanish");
}

void test_cutoff_product_rule() {
  auto parameters = kPbeOracle.parameters;
  parameters.pair_cutoff = 4.5;
  parameters.pair_switch_width = 2.0;
  test_finite_difference(parameters, 4.0e-10);
}

void test_overflowing_damping_keeps_representable_weighted_results() {
  // Exact binary inputs: x=6*2^(4*alpha) overflows, while x^-1*r^-power
  // and its radial derivative are normal representable FP64 values.
  for (int power : {6, 8}) {
    const double exponent = power + 254.0;
    const auto term = vibeqc::dft::dispersion::d3_zero_detail::damped_inverse_power(
        std::ldexp(1.0, -16), std::ldexp(1.0, -32), std::ldexp(1.0, -12), exponent, power);
    const double expected = std::ldexp(1.0 / 6.0, 12 * power - 1016);
    const double expected_derivative = std::ldexp(254.0 / 6.0, 12 * power - 984);
    if (!std::isfinite(term.value) || !std::isfinite(term.derivative_over_distance) ||
        std::abs(term.value / expected - 1.0) > 1.0e-12 ||
        std::abs(term.derivative_over_distance / expected_derivative - 1.0) > 1.0e-12)
      throw std::runtime_error("D3(0) overflowed damping lost a representable result");
  }
}

void test_pair_helper_overflowing_damping_keeps_representable_results() {
  constexpr std::array<std::int32_t, 2> numbers{6, 6};
  const auto tables = d3_host_tables();
  const double r = std::ldexp(1.0, -16);
  const double r2 = std::ldexp(1.0, -32);
  const double r0 =
      tables.pairs[vibeqc::dft::dispersion::d3_detail::pair_index(numbers[0], numbers[1])]
          .vdw_radius;
  const double rr =
      3.0 * tables.elements[numbers[0] - 1].r4r2 * tables.elements[numbers[1] - 1].r4r2;

  D3ModelParameters model{};
  model.damping = D3Damping::zero;
  model.zero = kPbeOracle.parameters;
  model.zero.alpha6 = 260.0;
  model.zero.rs6 = std::ldexp(1.0, -12) / r0;
  model.zero.rs8 = std::ldexp(1.0, -12) / r0;

  D3PairTerm term{};
  model.zero.s6 = 1.0;
  model.zero.s8 = 0.0;
  if (!d3_pair_term(0, 1, numbers.data(), r2, model, tables, term) || !term.included)
    throw std::runtime_error("D3(0) pair helper rejected representable inverse-sixth result");
  const double expected6 = std::ldexp(1.0 / 6.0, 12 * 6 - 1016);
  const double expected6_derivative = std::ldexp(254.0 / 6.0, 12 * 6 - 984);
  if (!std::isfinite(term.damping) || !std::isfinite(term.radial_derivative_over_distance) ||
      std::abs(term.damping / expected6 - 1.0) > 1.0e-12 ||
      std::abs(term.radial_derivative_over_distance / expected6_derivative - 1.0) > 1.0e-12)
    throw std::runtime_error("D3(0) pair helper lost representable inverse-sixth result");

  model.zero.s6 = 0.0;
  model.zero.s8 = 1.0;
  if (!d3_pair_term(0, 1, numbers.data(), r2, model, tables, term) || !term.included)
    throw std::runtime_error("D3(0) pair helper rejected representable inverse-eighth result");
  const double expected8 = rr * std::ldexp(1.0 / 6.0, 12 * 8 - 1016);
  const double expected8_derivative = rr * std::ldexp(254.0 / 6.0, 12 * 8 - 984);
  if (!std::isfinite(term.damping) || !std::isfinite(term.radial_derivative_over_distance) ||
      std::abs(term.damping / expected8 - 1.0) > 1.0e-12 ||
      std::abs(term.radial_derivative_over_distance / expected8_derivative - 1.0) > 1.0e-12)
    throw std::runtime_error("D3(0) pair helper lost representable inverse-eighth result");
}

void test_parameter_validation() {
  auto parameters = kPbeOracle.parameters;
  parameters.rs6 = 0.0;
  if (evaluate(kCoordinates, parameters, true).status != D3Status::invalid_argument)
    throw std::runtime_error("zero rs6 must fail");
  parameters = kPbeOracle.parameters;
  parameters.rs8 = -1.0;
  if (evaluate(kCoordinates, parameters, true).status != D3Status::invalid_argument)
    throw std::runtime_error("negative rs8 must fail");
  parameters = kPbeOracle.parameters;
  parameters.alpha6 = std::numeric_limits<double>::infinity();
  if (evaluate(kCoordinates, parameters, true).status != D3Status::invalid_argument)
    throw std::runtime_error("non-finite alpha6 must fail");
}

}  // namespace

int main() {
  try {
    test_external_oracle(kPbeOracle);
    test_external_oracle(kPbe0Oracle);
    test_external_oracle(kSlaterDiracOracle);
    test_finite_difference(kPbeOracle.parameters, 8.0e-11);
    test_finite_difference(kSlaterDiracOracle.parameters, 2.0e-9);
    test_multistep_finite_difference(kPbeOracle.parameters, 2.0e-9);
    test_multistep_finite_difference(kPbe0Oracle.parameters, 2.0e-9);
    test_translation_invariance();
    test_cutoff_product_rule();
    test_parameter_validation();
    test_overflowing_damping_keeps_representable_weighted_results();
    test_pair_helper_overflowing_damping_keeps_representable_results();
    std::cout << "D3(0) independent-oracle, analytic-gradient, multistep-FD and invariance tests "
                 "passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
