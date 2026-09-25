#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>

#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/initial_guess/overlap.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/reference/observation.hpp"

namespace {
using namespace vibeqc;
using namespace scf::initial_guess;
using namespace scf::reference;

unsigned solves{};
std::size_t begin(const char* name, std::size_t) noexcept {
  if (std::string_view(name) == "reference_eigensolve") ++solves;
  return 0;
}
void end(std::size_t, int) noexcept {}
const observation::Observer observer{begin, end};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void close(const Matrix& actual, const Matrix& expected) {
  require(actual.size() == expected.size(), "matrix shape changed");
  for (std::size_t i = 0; i < actual.size(); ++i)
    require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 1e-13,
            "initial-density convention changed");
}
template <class Function>
void invalid(Function&& function) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("invalid initial density was accepted");
}

void check_initial_density_contract() {
  // Diagonal, nonidentity S gives analytically known eigenpairs. Expected
  // densities below are independent of the production preparation helpers.
  integrals::IntegralData ints;
  ints.nbf = 3;
  ints.overlap = {2, 0, 0, 0, 1, 0, 0, 0, .5};
  ints.hcore = {-2, 0, 0, 0, -.5, 0, 0, 0, .25};
  const Matrix x{std::sqrt(.5), 0, 0, 0, 1, 0, 0, 0, std::sqrt(2.)};
  core::System system;
  system.electron_count = 2;
  std::optional<EigenResult> a, b;
  solves = 0;
  const auto cold = prepare_initial_density(system, ints, x, 1, nullptr, a);
  require(solves == 1 && a, "cold RHF must construct its core frame once");
  close(cold, {1, 0, 0, 0, 0, 0, 0, 0, 0});
  const auto [alpha, beta] = prepare_initial_uhf_density(ints, x, 2, 1, nullptr, a, b);
  require(solves == 2 && a && b, "cold UHF must share one core solve");
  close(alpha, {.5, 0, 0, 0, 1, 0, 0, 0, 0});
  // Historical 45-degree beta HOMO/LUMO mixing remains a cold-only choice.
  close(beta, {.25, std::sqrt(.125), 0, std::sqrt(.125), .5, 0, 0, 0, 0});

  const Matrix hcore = ints.hcore;
  std::fill(ints.hcore.begin(), ints.hcore.end(), std::numeric_limits<double>::quiet_NaN());
  const Matrix raw{2, .2, 0, .4, 0, 0, 0, 0, 0};
  solves = 0;
  close(prepare_initial_density(system, ints, x, 1, &raw, a), {1, .15, 0, .15, 0, 0, 0, 0, 0});
  require(solves == 0 && !a, "warm RHF must not compute or retain a core frame");
  Matrix spin = raw;
  spin.insert(spin.end(), raw.begin(), raw.end());
  auto [warm_alpha, warm_beta] = prepare_initial_uhf_density(ints, x, 2, 0, &spin, a, b);
  close(warm_alpha, {1, .15, 0, .15, 0, 0, 0, 0, 0});
  close(warm_beta, Matrix(9, 0));
  require(solves == 0 && !a && !b, "warm empty-spin UHF must clear both core frames");

  // A consumer needing orbitals requests them explicitly, after validating D.
  ints.hcore = hcore;
  prepare_initial_uhf_density(ints, x, 2, 1, &spin, a, b, InitialOrbitalRequest::RequireCoreFrame);
  require(solves == 1 && a && b, "explicit warm frame request did not solve");
  close(a->vectors, b->vectors);
  close(density_from_orbitals(b->vectors, 3, 1, 1), {.5, 0, 0, 0, 0, 0, 0, 0, 0});
  prepare_initial_density(system, ints, x, 1, &raw, a, InitialOrbitalRequest::RequireCoreFrame);
  require(solves == 2 && a, "explicit RHF warm frame request did not solve");

  for (const Matrix bad :
       {Matrix(2, 1), Matrix(9, 0), Matrix(9, std::numeric_limits<double>::infinity())}) {
    a = EigenResult{{1}, {1}};
    solves = 0;
    invalid([&] { prepare_initial_density(system, ints, x, 1, &bad, a); });
    require(solves == 0 && !a, "failed RHF validation leaked a frame or solved hcore");
  }
  for (const Matrix bad :
       {Matrix(2, 1), Matrix(18, 0), Matrix(18, std::numeric_limits<double>::quiet_NaN())}) {
    a = b = EigenResult{{1}, {1}};
    solves = 0;
    invalid([&] { prepare_initial_uhf_density(ints, x, 2, 1, &bad, a, b); });
    require(solves == 0 && !a && !b, "failed UHF validation leaked a frame or solved hcore");
  }
  // A failed item cannot poison a subsequent independent cold request.
  prepare_initial_density(system, ints, x, 1, nullptr, a);
  require(solves == 1 && a, "cold retry did not rebuild a valid frame");
}

void check_charge_guided_lowdin_contract() {
  core::System system;
  system.atoms = {{1, {0, 0, 0}}, {1, {0, 0, 1}}};
  system.shells = {{0, 0, {}}, {1, 0, {}}};
  system.charge = 0;
  system.electron_count = 2;

  integrals::IntegralData ints;
  ints.nbf = 2;
  ints.overlap = {4, 0, 0, 1};
  const Matrix x{.5, 0, 0, 1};
  const Matrix base{.25, 0, 0, 1};

  // The base has one Loewdin electron on each atom. A +0.5/-0.5 xTB charge
  // target therefore moves half an electron from atom 0 to atom 1.
  close(charge_guided_lowdin_density(system, ints, x, base, std::array{.5, -.5}),
        {.125, 0, 0, 1.5});
  close(charge_guided_lowdin_density(system, ints, x, base, std::array{0.0, 0.0}), base);

  invalid([&] { charge_guided_lowdin_density(system, ints, x, base, std::array{0.0}); });
  invalid([&] { charge_guided_lowdin_density(system, ints, x, base, std::array{.2, .2}); });
  invalid([&] { charge_guided_lowdin_density(system, ints, x, base, std::array{1.2, -1.2}); });
}


void check_occupied_projection_contract() {
  integrals::IntegralData target;
  target.nbf = 2;
  target.overlap = {1, 0, 0, 1};
  const Matrix x{1, 0, 0, 1};
  const Matrix source_overlap{1, 0, 0, 1};
  const Matrix cross{1, 0, 0, 1};
  const double inv_sqrt_two = 1.0 / std::sqrt(2.0);
  // Coefficients are columns. The occupied source vector is a normalized
  // 45-degree rotation; projecting into an identical target must preserve it.
  const Matrix source_coefficients{inv_sqrt_two, -inv_sqrt_two,
                                   inv_sqrt_two, inv_sqrt_two};
  const auto projected = project_occupied_density(target, x, source_overlap, cross, 2,
                                                   source_coefficients, 1);
  close(projected.density, {1, 1, 1, 1});
  require(std::abs(projected.minimum_projected_norm - 1.0) < 1e-13 &&
              projected.projection_residual < 1e-13 &&
              projected.source_metric_orthogonality_error < 1e-13 &&
              projected.target_metric_orthogonality_error < 1e-13,
          "identity occupied projection diagnostics changed");

  const Matrix rank_lost_cross{0, 0, 0, 0};
  invalid([&] {
    project_occupied_density(target, x, source_overlap, rank_lost_cross, 2,
                             source_coefficients, 1);
  });
  const Matrix nonorthogonal_coefficients{2, 0, 0, 1};
  invalid([&] {
    project_occupied_density(target, x, source_overlap, cross, 2,
                             nonorthogonal_coefficients, 1);
  });
}

void check_overlap_cache_contract() {
  core::System system;
  system.atoms.push_back({1, {0, 0, 0}});
  OverlapOrthogonalizer cache;
  Matrix overlap{2, 0, 0, 0, 1, 0, 0, 0, .5};
  solves = 0;
  close(cache.get(system, overlap, 3), {std::sqrt(.5), 0, 0, 0, 1, 0, 0, 0, std::sqrt(2.)});
  require(solves == 1, "cold overlap cache missed its actual solve");
  cache.get(system, overlap, 3);
  require(solves == 1, "identical overlap was diagonalized again");
  require(cache.numeric_capacity_bytes() == (2 * 9 + 3) * sizeof(double),
          "overlap cache did not account for both matrices and its geometry");
  // Same dimensions and geometry must not hide different overlap values.
  // A stale-X mutation that drops the S comparison fails this analytic gate.
  overlap[0] = 4;
  close(cache.get(system, overlap, 3), {.5, 0, 0, 0, 1, 0, 0, 0, std::sqrt(2.)});
  require(solves == 2, "changed overlap reused stale X");
  system.atoms[0].position[0] = .1;
  cache.get(system, overlap, 3);
  require(solves == 3, "changed geometry did not replace the cached frame");
  overlap[0] = 9e-11;
  bool rejected = false;
  try {
    cache.get(system, overlap, 3);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected && cache.numeric_capacity_bytes() == 0,
          "singular overlap changed its policy or retained stale state");
  overlap[0] = 1e-10;  // The existing strict '<' policy admits the boundary.
  const auto& x = cache.get(system, overlap, 3);
  close(multiply(transpose(x, 3), multiply(overlap, x, 3), 3), identity(3));
  const auto previous = solves;
  cache.get(system, overlap, 3);
  require(solves == previous, "successful retry did not retain its new X");
}
void check_borrowed_setup_operation() {
  // Analytic diagonal eigenpairs qualify callback forwarding without using
  // either reference decomposition or the production CUDA provider as oracle.
  unsigned calls = 0;
  EigenOperation eigen = [&](const Matrix& matrix, const Matrix* overlap, const Matrix* x,
                             std::size_t n) {
    ++calls;
    require(n == 3 && (overlap == nullptr) == (x == nullptr), "setup callback shape changed");
    std::vector<std::pair<double, std::size_t>> order;
    for (std::size_t i = 0; i < n; ++i)
      order.emplace_back(matrix[i * n + i] * (x ? (*x)[i * n + i] * (*x)[i * n + i] : 1), i);
    std::sort(order.begin(), order.end());
    EigenResult result;
    result.vectors.assign(n * n, 0);
    for (std::size_t column = 0; column < n; ++column) {
      const auto [value, row] = order[column];
      result.values.push_back(value);
      result.vectors[row * n + column] = x ? (*x)[row * n + row] : 1;
    }
    return result;
  };
  core::System system;
  system.electron_count = 2;
  system.atoms.push_back({1, {0, 0, 0}});
  integrals::IntegralData ints;
  ints.nbf = 3;
  ints.overlap = {2, 0, 0, 0, 1, 0, 0, 0, .5};
  ints.hcore = {-2, 0, 0, 0, -.5, 0, 0, 0, .25};
  OverlapOrthogonalizer cache;
  solves = 0;
  const auto x = cache.get(system, ints.overlap, 3, eigen);
  close(x, {std::sqrt(.5), 0, 0, 0, 1, 0, 0, 0, std::sqrt(2.)});
  cache.get(system, ints.overlap, 3, eigen);
  require(calls == 1 && solves == 0, "cached X retained or repeated a provider call");
  std::optional<EigenResult> alpha, beta;
  const auto density = prepare_initial_density(system, ints, x, 1, nullptr, alpha,
                                               InitialOrbitalRequest::ColdDensityOnly, eigen);
  close(density, {1, 0, 0, 0, 0, 0, 0, 0, 0});
  auto spins = prepare_initial_uhf_density(ints, x, 2, 1, nullptr, alpha, beta,
                                           InitialOrbitalRequest::ColdDensityOnly, eigen);
  close(spins.second, {.25, std::sqrt(.125), 0, std::sqrt(.125), .5, 0, 0, 0, 0});
  require(calls == 3 && solves == 0, "cold setup bypassed the borrowed operation");
  EigenOperation fail = [](const Matrix&, const Matrix*, const Matrix*,
                           std::size_t) -> EigenResult {
    throw std::runtime_error("intentional setup failure");
  };
  prepare_initial_density(system, ints, x, 1, &density, alpha,
                          InitialOrbitalRequest::ColdDensityOnly, fail);
  require(!alpha, "warm setup kept an unused core frame");
  try {
    prepare_initial_density(system, ints, x, 1, nullptr, alpha,
                            InitialOrbitalRequest::ColdDensityOnly, fail);
    require(false, "setup swallowed the provider failure");
  } catch (const std::runtime_error& error) {
    require(std::string_view(error.what()) == "intentional setup failure" && !alpha,
            "failed setup published stale orbitals");
  }
  system.atoms[0].position[0] = .1;
  try {
    cache.get(system, ints.overlap, 3, fail);
    require(false, "changed setup swallowed the provider failure");
  } catch (const std::runtime_error&) {
    require(cache.numeric_capacity_bytes() == 0, "failed setup kept old X");
  }
  ints.overlap[0] = 1e-10;
  cache.get(system, ints.overlap, 3, eigen);
  ints.overlap[0] = std::nextafter(1e-10, 0.0);
  bool rejected = false;
  try {
    cache.get(system, ints.overlap, 3, eigen);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected && cache.numeric_capacity_bytes() == 0, "borrowed setup changed the cutoff");
}
}  // namespace

int main() {
  observation::active = &observer;
  try {
    check_initial_density_contract();
    check_charge_guided_lowdin_contract();
    check_occupied_projection_contract();
    check_overlap_cache_contract();
    check_borrowed_setup_operation();
    observation::active = nullptr;
    return 0;
  } catch (const std::exception& error) {
    observation::active = nullptr;
    std::cerr << error.what() << '\n';
    return 1;
  }
}
