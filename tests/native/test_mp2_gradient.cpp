#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "posthf/mp2_gradient.hpp"
#include "posthf/native_provider.hpp"
#include "response/native_gmres.hpp"
#include "scf/mean_field.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::size_t eri_index(std::size_t n, std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
  return ((p * n + q) * n + r) * n + s;
}

std::size_t g_index(std::size_t no, std::size_t nv, std::size_t i, std::size_t j, std::size_t a,
                    std::size_t b) {
  return ((i * no + j) * nv + a) * nv + b;
}

double mp2_energy(std::span<const double> g, std::span<const double> eps, std::size_t no) {
  const auto nv = eps.size() - no;
  double energy = 0.0;
  for (std::size_t i = 0; i < no; ++i)
    for (std::size_t j = 0; j < no; ++j)
      for (std::size_t a = 0; a < nv; ++a)
        for (std::size_t b = 0; b < nv; ++b) {
          const double direct = g[g_index(no, nv, i, j, a, b)];
          const double exchange = g[g_index(no, nv, i, j, b, a)];
          const double denominator = eps[i] + eps[j] - eps[no + a] - eps[no + b];
          energy += (2.0 * direct * direct - direct * exchange) / denominator;
        }
  return energy;
}

std::vector<double> symmetric_eri(std::size_t n) {
  std::vector<double> values(n * n * n * n);
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s) {
          const auto first = std::min(p, q) * n + std::max(p, q);
          const auto second = std::min(r, s) * n + std::max(r, s);
          const auto lo = std::min(first, second), hi = std::max(first, second);
          values[eri_index(n, p, q, r, s)] = 0.004 * (1 + lo + 3 * hi);
        }
  return values;
}

std::vector<double> fock(std::span<const double> h, std::span<const double> eri, std::size_t n,
                         std::size_t no) {
  std::vector<double> result(h.begin(), h.end());
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t i = 0; i < no; ++i)
        result[p * n + q] += 2.0 * eri[eri_index(n, p, q, i, i)] - eri[eri_index(n, p, i, i, q)];
  return result;
}

std::vector<double> multiply(std::span<const double> left, std::span<const double> right,
                             std::size_t n) {
  std::vector<double> result(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t k = 0; k < n; ++k) result[i * n + j] += left[i * n + k] * right[k * n + j];
  return result;
}

std::vector<double> transpose(std::span<const double> matrix, std::size_t n) {
  std::vector<double> result(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) result[i * n + j] = matrix[j * n + i];
  return result;
}

std::vector<double> inverse(std::span<const double> matrix, std::size_t n) {
  std::vector<double> augmented(n * 2 * n);
  const auto stride = 2 * n;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) augmented[i * stride + j] = matrix[i * n + j];
    augmented[i * stride + n + i] = 1.0;
  }
  for (std::size_t column = 0; column < n; ++column) {
    std::size_t pivot = column;
    for (std::size_t row = column + 1; row < n; ++row)
      if (std::abs(augmented[row * stride + column]) > std::abs(augmented[pivot * stride + column]))
        pivot = row;
    require(std::abs(augmented[pivot * stride + column]) > 1e-14, "singular test matrix");
    for (std::size_t j = 0; j < stride; ++j)
      std::swap(augmented[column * stride + j], augmented[pivot * stride + j]);
    const double scale = augmented[column * stride + column];
    for (std::size_t j = 0; j < stride; ++j) augmented[column * stride + j] /= scale;
    for (std::size_t row = 0; row < n; ++row) {
      if (row == column) continue;
      const double factor = augmented[row * stride + column];
      for (std::size_t j = 0; j < stride; ++j)
        augmented[row * stride + j] -= factor * augmented[column * stride + j];
    }
  }
  std::vector<double> result(n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) result[i * n + j] = augmented[i * stride + n + j];
  return result;
}

std::vector<double> rotated_eri(std::span<const double> eri, std::span<const double> rotation,
                                std::size_t n) {
  std::vector<double> result(n * n * n * n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t k = 0; k < n; ++k)
        for (std::size_t l = 0; l < n; ++l)
          for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = 0; q < n; ++q)
              for (std::size_t r = 0; r < n; ++r)
                for (std::size_t s = 0; s < n; ++s)
                  result[eri_index(n, i, j, k, l)] += eri[eri_index(n, p, q, r, s)] *
                                                      rotation[p * n + i] * rotation[q * n + j] *
                                                      rotation[r * n + k] * rotation[s * n + l];
  return result;
}

void energy_adjoint_matches_independent_finite_difference() {
  constexpr std::size_t no = 1, nv = 2;
  const std::vector<double> g{0.12, -0.04, 0.07, 0.09};
  const std::vector<double> eps{-0.8, 0.2, 0.55};
  const std::vector<double> dg{0.3, -0.2, 0.5, 0.1};
  const std::vector<double> de{-0.4, 0.2, 0.6};
  const auto adjoint = vibeqc::mp2::canonical_energy_adjoint(g, eps, no, 1e-10);
  double reverse = 0.0;
  for (std::size_t i = 0; i < g.size(); ++i) reverse += adjoint.integrals_iajb[i] * dg[i];
  for (std::size_t i = 0; i < eps.size(); ++i) reverse += adjoint.orbital_energies[i] * de[i];
  double previous = std::numeric_limits<double>::infinity();
  for (double step : {1e-3, 1e-4, 1e-5}) {
    auto plus_g = g, minus_g = g, plus_e = eps, minus_e = eps;
    for (std::size_t i = 0; i < g.size(); ++i) {
      plus_g[i] += step * dg[i];
      minus_g[i] -= step * dg[i];
    }
    for (std::size_t i = 0; i < eps.size(); ++i) {
      plus_e[i] += step * de[i];
      minus_e[i] -= step * de[i];
    }
    const double error = std::abs(
        (mp2_energy(plus_g, plus_e, no) - mp2_energy(minus_g, minus_e, no)) / (2.0 * step) -
        reverse);
    require(error < previous, "energy-adjoint finite difference did not improve");
    previous = error;
  }
  require(previous < 1e-9, "energy adjoint failed the independent directional derivative");
  const auto a01 = g_index(no, nv, 0, 0, 0, 1);
  const double d01 = eps[0] + eps[0] - eps[1] - eps[2];
  const double expected =
      (4.0 * g[a01] - g[g_index(no, nv, 0, 0, 1, 0)]) / d01 - g[g_index(no, nv, 0, 0, 1, 0)] / d01;
  require(std::abs(adjoint.integrals_iajb[a01] - expected) < 1e-14,
          "exchange cotangent was not transposed back and accumulated");
}

struct Fixture {
  std::size_t n{3}, no{1};
  std::vector<double> eps{-0.9, 0.25, 0.7};
  std::vector<double> eri{symmetric_eri(n)};
  std::vector<double> h;
  std::vector<double> g;

  Fixture() : h(n * n), g(no * no * (n - no) * (n - no)) {
    auto mean_field = fock(h, eri, n, no);
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q)
        h[p * n + q] = (p == q ? eps[p] : 0.0) - mean_field[p * n + q];
    for (std::size_t a = 0; a < n - no; ++a)
      for (std::size_t b = 0; b < n - no; ++b)
        g[g_index(no, n - no, 0, 0, a, b)] = eri[eri_index(n, 0, no + a, 0, no + b)];
  }
};

vibeqc::core::System h2() {
  vibeqc::core::System system;
  system.atoms = {{1, {0, 0, -0.7}}, {1, {0, 0, 0.7}}};
  const std::vector<vibeqc::core::Primitive> primitives{
      {3.42525091, 0.1543289673}, {0.62391373, 0.5353281423}, {0.1688554, 0.4446345422}};
  system.shells = {{0, 0, primitives}, {1, 0, primitives}};
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "H2 setup failed");
  return system;
}

void streamed_provider_matches_dense_oracle() {
  const auto system = h2();
  vibeqc::scf::ScfOptions options;
  options.export_physical_reference = true;
  options.compute_forces = false;
  options.screening_tolerance = 0.0;
  options.energy_tolerance = options.density_tolerance = 1e-11;
  options.reference_memory_budget_bytes = 256ULL << 20;
  const auto hf = vibeqc::scf::run_rhf(system, options);
  require(hf.converged && hf.reference, "H2 reference did not converge");
  const auto& reference = *hf.reference;
  vibeqc::posthf::RawSource source(system);
  vibeqc::posthf::NativeBlockProvider provider(source, reference, 256ULL << 20, 1);
  std::vector<std::size_t> orbitals(reference.nbf);
  for (std::size_t p = 0; p < reference.nbf; ++p) orbitals[p] = p;
  const auto eri = provider.get({orbitals, orbitals, orbitals, orbitals});
  std::vector<double> hcore_mo(reference.nbf * reference.nbf);
  for (std::size_t p = 0; p < reference.nbf; ++p)
    for (std::size_t q = 0; q < reference.nbf; ++q)
      for (std::size_t mu = 0; mu < reference.nbf; ++mu)
        for (std::size_t nu = 0; nu < reference.nbf; ++nu)
          hcore_mo[p * reference.nbf + q] += reference.coefficients[mu * reference.nbf + p] *
                                             reference.hcore[mu * reference.nbf + nu] *
                                             reference.coefficients[nu * reference.nbf + q];
  const auto virtuals = reference.nbf - reference.nocc;
  std::vector<double> g(reference.nocc * reference.nocc * virtuals * virtuals);
  for (std::size_t i = 0; i < reference.nocc; ++i)
    for (std::size_t j = 0; j < reference.nocc; ++j)
      for (std::size_t a = 0; a < virtuals; ++a)
        for (std::size_t b = 0; b < virtuals; ++b)
          g[g_index(reference.nocc, virtuals, i, j, a, b)] =
              eri[eri_index(reference.nbf, i, reference.nocc + a, j, reference.nocc + b)];
  const auto adjoint =
      vibeqc::mp2::canonical_energy_adjoint(g, reference.orbital_energies, reference.nocc, 1e-10);
  const auto dense = vibeqc::mp2::canonical_orbital_rhs(hcore_mo, eri, adjoint, 1e-10);
  const auto streamed =
      vibeqc::mp2::canonical_orbital_rhs_streamed(reference, hcore_mo, provider, adjoint, 1e-10);
  auto close = [](std::span<const double> first, std::span<const double> second) {
    if (first.size() != second.size()) return false;
    for (std::size_t i = 0; i < first.size(); ++i)
      if (std::abs(first[i] - second[i]) > 2e-11) return false;
    return true;
  };
  require(close(streamed.energy_gradient, dense.energy_gradient) &&
              close(streamed.response_rhs, dense.response_rhs) &&
              close(streamed.one_electron, dense.one_electron) &&
              close(streamed.two_electron, dense.two_electron),
          "streamed native provider path differs from the dense oracle");
  const double response_denominator =
      reference.orbital_energies[reference.nocc] - reference.orbital_energies[0] +
      4.0 * eri[eri_index(reference.nbf, reference.nocc, 0, reference.nocc, 0)] -
      eri[eri_index(reference.nbf, reference.nocc, reference.nocc, 0, 0)] -
      eri[eri_index(reference.nbf, reference.nocc, 0, 0, reference.nocc)];
  const std::array<double, 1> response{streamed.response_rhs[0] / response_denominator};
  const auto dense_weights =
      vibeqc::mp2::canonical_lagrangian_weights(hcore_mo, eri, adjoint, response, 1e-10);
  const auto streamed_weights = vibeqc::mp2::canonical_lagrangian_weights_streamed(
      reference, hcore_mo, provider, adjoint, response, 1e-10);
  require(close(streamed_weights.one_electron, dense_weights.one_electron) &&
              close(streamed_weights.two_electron, dense_weights.two_electron) &&
              close(streamed_weights.overlap, dense_weights.overlap) &&
              std::abs(streamed_weights.stationarity_residual -
                       dense_weights.stationarity_residual) < 2e-11,
          "streamed relaxed weights differ from the dense oracle");
  auto stale = reference;
  bool rejected = false;
  try {
    (void)vibeqc::mp2::canonical_orbital_rhs_streamed(stale, hcore_mo, provider, adjoint, 1e-10);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "streamed provider accepted a copied/stale reference owner");
#if !VIBEQC_HAS_CUDA
  rejected = false;
  try {
    (void)vibeqc::mp2::canonical_orbital_rhs_streamed(reference, hcore_mo, provider, adjoint, 1e-10,
                                                      true, 7);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "streamed orbital response ignored the requested CUDA backend");
#endif
}

double rotated_mp2_energy(const Fixture& fixture, std::span<const double> direction, double step) {
  const auto n = fixture.n, no = fixture.no;
  std::vector<double> generator(n * n), left(n * n), right(n * n);
  for (std::size_t p = 0; p < n; ++p) left[p * n + p] = right[p * n + p] = 1.0;
  for (std::size_t a = 0; a < n - no; ++a) {
    generator[a + no] = direction[a];
    generator[(a + no) * n] = -direction[a];
  }
  for (std::size_t i = 0; i < n * n; ++i) {
    left[i] -= 0.5 * step * generator[i];
    right[i] += 0.5 * step * generator[i];
  }
  const auto rotation = multiply(inverse(left, n), right, n);
  const auto transformed_h = multiply(multiply(transpose(rotation, n), fixture.h, n), rotation, n);
  const auto transformed_eri = rotated_eri(fixture.eri, rotation, n);
  const auto transformed_fock = fock(transformed_h, transformed_eri, n, no);
  std::vector<double> eps(n), g((n - no) * (n - no));
  for (std::size_t p = 0; p < n; ++p) eps[p] = transformed_fock[p * n + p];
  for (std::size_t a = 0; a < n - no; ++a)
    for (std::size_t b = 0; b < n - no; ++b)
      g[g_index(no, n - no, 0, 0, a, b)] = transformed_eri[eri_index(n, 0, no + a, 0, no + b)];
  return mp2_energy(g, eps, no);
}

void orbital_rhs_and_relaxed_weights_match_independent_oracles() {
  const Fixture fixture;
  const auto adjoint =
      vibeqc::mp2::canonical_energy_adjoint(fixture.g, fixture.eps, fixture.no, 1e-10);
  const auto orbital = vibeqc::mp2::canonical_orbital_rhs(fixture.h, fixture.eri, adjoint, 1e-10);
  const std::array<double, 2> direction{0.31, -0.27};
  double reverse = 0.0;
  for (std::size_t i = 0; i < direction.size(); ++i)
    reverse += orbital.energy_gradient[i] * direction[i];
  std::array<double, 3> errors{};
  std::size_t error_index = 0;
  for (double step : {1e-3, 1e-4, 1e-5}) {
    const double finite = (rotated_mp2_energy(fixture, direction, step) -
                           rotated_mp2_energy(fixture, direction, -step)) /
                          (2.0 * step);
    const double error = std::abs(finite - reverse);
    errors[error_index++] = error;
  }
  require(errors[1] < errors[0] && errors[1] < 1e-10 && errors[2] < 1e-10,
          "orbital gradient has the wrong sign or missing terms");

  const auto nv = fixture.n - fixture.no;
  std::vector<double> response_matrix(nv * nv);
  for (std::size_t a = 0; a < nv; ++a)
    for (std::size_t b = 0; b < nv; ++b)
      response_matrix[a * nv + b] =
          (fixture.eps[fixture.no + a] - fixture.eps[0]) * (a == b) +
          4.0 * fixture.eri[eri_index(fixture.n, fixture.no + a, 0, fixture.no + b, 0)] -
          fixture.eri[eri_index(fixture.n, fixture.no + a, fixture.no + b, 0, 0)] -
          fixture.eri[eri_index(fixture.n, fixture.no + a, 0, 0, fixture.no + b)];
  const auto response_inverse = inverse(response_matrix, nv);
  std::vector<double> z(nv);
  for (std::size_t a = 0; a < nv; ++a)
    for (std::size_t b = 0; b < nv; ++b)
      z[a] += response_inverse[a * nv + b] * orbital.response_rhs[b];
  const auto weights =
      vibeqc::mp2::canonical_lagrangian_weights(fixture.h, fixture.eri, adjoint, z, 1e-10);
  require(weights.stationarity_residual < 1e-10,
          "relaxed weights are not stationary with the independent Z-vector");
  for (std::size_t p = 0; p < fixture.n; ++p)
    for (std::size_t q = 0; q < fixture.n; ++q)
      require(
          std::abs(weights.overlap[p * fixture.n + q] - weights.overlap[q * fixture.n + p]) < 1e-14,
          "overlap weight is not symmetric");
}

void invalid_inputs_and_resource_boundaries() {
  bool rejected = false;
  const std::array<double, 1> unit_integral{1.0};
  const std::array<double, 2> zero_gap{0.0, 0.0};
  try {
    (void)vibeqc::mp2::canonical_energy_adjoint(unit_integral, zero_gap, 1, 1e-10);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "near-zero MP2 denominator was accepted");
  rejected = false;
  const std::array<double, 1> nonfinite_integral{std::numeric_limits<double>::quiet_NaN()};
  const std::array<double, 2> separated_energies{-1.0, 1.0};
  try {
    (void)vibeqc::mp2::canonical_energy_adjoint(nonfinite_integral, separated_energies, 1, 1e-10);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "nonfinite MP2 integral was accepted");

  auto options = vibeqc::response::GmresOptions{};
  options.restart = 3;
  options.max_iterations = 10;
  const auto response = vibeqc::response::prepare_gmres(4, options);
  const auto probe = vibeqc::mp2::conventional_gradient_plan(
      4, 2, 4096, response, 3, 9, 9 * sizeof(double),
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()));
  require(probe.peak_bytes > probe.response_bytes && probe.shell_cotangent_bytes > 0,
          "gradient resource plan omitted a simultaneous owner");
  require(probe.shell_cotangent_bytes == 81 * sizeof(double),
          "shell-quartet cotangent ownership is not isolated");
  require(probe.derivative_staging_bytes == 485 * sizeof(double),
          "derivative staging double-counts the shell-quartet cotangent");
  const auto cuda_probe = vibeqc::mp2::conventional_gradient_plan(
      4, 2, 4096, response, 3, 9, 9 * sizeof(double),
      static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()), 12345);
  require(cuda_probe.derivative_backend_staging_bytes == 12345,
          "gradient resource plan omitted CUDA consumer staging");
  require(cuda_probe.peak_bytes == probe.peak_bytes + 12345,
          "CUDA consumer staging was not charged exactly once");
  const auto exact = vibeqc::mp2::conventional_gradient_plan(4, 2, 4096, response, 3, 9,
                                                             9 * sizeof(double), probe.peak_bytes);
  require(exact.peak_bytes == probe.peak_bytes, "exact resource budget changed the plan");
  rejected = false;
  try {
    (void)vibeqc::mp2::conventional_gradient_plan(4, 2, 4096, response, 3, 9, 9 * sizeof(double),
                                                  probe.peak_bytes - 1);
  } catch (const std::length_error&) {
    rejected = true;
  }
  require(rejected, "one-byte-short gradient budget was accepted");
  rejected = false;
  try {
    (void)vibeqc::mp2::conventional_gradient_plan(4, 2, 1, response,
                                                  std::numeric_limits<std::size_t>::max(), 3, 24,
                                                  std::numeric_limits<std::size_t>::max());
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  require(rejected, "gradient resource arithmetic overflow was accepted");
}
}  // namespace

int main() {
  try {
    energy_adjoint_matches_independent_finite_difference();
    orbital_rhs_and_relaxed_weights_match_independent_oracles();
    streamed_provider_matches_dense_oracle();
    invalid_inputs_and_resource_boundaries();
    std::cout << "MP2 native gradient contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
