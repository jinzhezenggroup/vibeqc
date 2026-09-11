#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "molecule/basis.hpp"
#include "scf/fleet.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/fock_provider.hpp"
#include "scf/mean_field.hpp"

namespace {
using namespace vibeqc::scf;
using vibeqc::integrals::IntegralData;

void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void close(double a, double b, double tolerance, const char* message) {
  if (!std::isfinite(a) || std::abs(a - b) > tolerance)
    throw std::runtime_error(std::string(message) + ": " + std::to_string(a) + " vs " +
                             std::to_string(b));
}
void matrix(const std::vector<double>& a, const std::vector<double>& b, const char* message,
            double tolerance = 3e-13) {
  require(a.size() == b.size(), message);
  for (std::size_t i = 0; i < a.size(); ++i) close(a[i], b[i], tolerance, message);
}
template <typename F>
void rejected(F function, const char* message) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

DensityFittingScfData fitted_data(double displacement, double cutoff) {
  DensityFittingScfData data;
  data.metric_relative_threshold = cutoff;
  auto& raw = data.raw;
  raw.nbf = 2;
  raw.naux = 2;
  raw.ncoord = 1;
  raw.metric = {2.0, 0.0, 0.0, 0.2};
  // At cutoff=0.15 the smaller eigenvalue is discarded but nonzero. Off-diagonal
  // dM mixes retained/discarded eigenvectors and detects the incorrect -M+ dM M+ response.
  raw.metric_derivative = {0.03, 0.04, 0.04, -0.02};
  raw.three_center = {1.3, 0.2, 0.3, -0.1, 0.3, -0.1, 0.8, 0.6};
  raw.three_center_derivative = {0.04, -0.03, 0.01, 0.05, 0.01, 0.05, -0.06, 0.02};
  for (std::size_t i = 0; i < raw.metric.size(); ++i)
    raw.metric[i] += displacement * raw.metric_derivative[i];
  for (std::size_t i = 0; i < raw.three_center.size(); ++i)
    raw.three_center[i] += displacement * raw.three_center_derivative[i];
  data.three_center = orthonormalize_density_fitting_three_center(
      raw.three_center, 2, factor_density_fitting_metric(raw.metric, 2, cutoff));
  return data;
}

IntegralData exact_data(double displacement) {
  IntegralData data;
  data.nbf = 2;
  data.ncoord = 1;
  const double pair[3][3]{{1.3, 0.2, 0.4}, {0.2, 0.6, -0.1}, {0.4, -0.1, 0.9}};
  const double derivative[3][3]{{0.11, -0.03, 0.07}, {-0.03, 0.05, 0.02}, {0.07, 0.02, -0.04}};
  const std::size_t index[]{0, 1, 1, 2};
  for (auto a : index)
    for (auto b : index) {
      data.eri.push_back(pair[a][b] + displacement * derivative[a][b]);
      data.eri_derivative.push_back(derivative[a][b]);
    }
  return data;
}

std::vector<double> fitted_eri(const DensityFittingScfData& data) {
  const auto inverse =
      density_fitting_metric_pseudoinverse(data.raw, data.metric_relative_threshold);
  std::vector<double> eri(16);
  // Tiny independent four-index oracle from raw A M+ A^T, not the production B/GEMM contraction.
  for (std::size_t ij = 0; ij < 4; ++ij)
    for (std::size_t kl = 0; kl < 4; ++kl)
      for (std::size_t p = 0; p < 2; ++p)
        for (std::size_t q = 0; q < 2; ++q)
          eri[ij * 4 + kl] += data.raw.three_center[ij * 2 + p] * inverse[p * 2 + q] *
                              data.raw.three_center[kl * 2 + q];
  return eri;
}

DirectJkMatrices oracle(const FockBuildSpec& spec, const IntegralData& exact,
                        const DensityFittingScfData& fitted, const std::vector<double>& a,
                        const std::vector<double>& b) {
  const auto df = fitted_eri(fitted);
  const auto& j_eri = spec.coulomb.approximation == FockApproximation::Exact ? exact.eri : df;
  const auto& k_eri = spec.exchange.approximation == FockApproximation::Exact ? exact.eri : df;
  DirectJkMatrices result;
  result.nbf = 2;
  if (spec.coulomb.present) result.coulomb.resize(4);
  if (spec.exchange.present) {
    result.exchange_alpha.resize(4);
    if (!b.empty()) result.exchange_beta.resize(4);
  }
  for (std::size_t i = 0; i < 2; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      for (std::size_t k = 0; k < 2; ++k)
        for (std::size_t l = 0; l < 2; ++l) {
          if (spec.coulomb.present)
            result.coulomb[i * 2 + j] += (a[k * 2 + l] + (b.empty() ? 0.0 : b[k * 2 + l])) *
                                         j_eri[((i * 2 + j) * 2 + k) * 2 + l];
          if (spec.exchange.present) {
            result.exchange_alpha[i * 2 + j] += a[k * 2 + l] * k_eri[((i * 2 + k) * 2 + j) * 2 + l];
            if (!b.empty())
              result.exchange_beta[i * 2 + j] +=
                  b[k * 2 + l] * k_eri[((i * 2 + k) * 2 + j) * 2 + l];
          }
        }
  return result;
}

CpuFockPlanView plan(FockBuildSpec spec, const IntegralData& exact,
                     const DensityFittingScfData& df) {
  const CpuFockProviderView exact_provider(exact), df_provider(df);
  return {resolve_fock_build(spec, FockBackend::Cpu, 1e-12, df.metric_relative_threshold), 2, 1,
          spec.coulomb.approximation == FockApproximation::Exact ? exact_provider : df_provider,
          spec.exchange.approximation == FockApproximation::Exact ? exact_provider : df_provider};
}

double reference_energy(const FockBuildSpec& spec, double displacement, double cutoff,
                        const std::vector<double>& a, const std::vector<double>& b) {
  const auto jk = oracle(spec, exact_data(displacement), fitted_data(displacement, cutoff), a, b);
  double energy = 0.0;
  for (std::size_t ij = 0; ij < 4; ++ij) {
    if (spec.coulomb.present)
      energy +=
          0.5 * spec.coulomb.coefficient * (a[ij] + (b.empty() ? 0.0 : b[ij])) * jk.coulomb[ij];
    if (spec.exchange.present) {
      energy += 0.5 * spec.exchange.coefficient * a[ij] * jk.exchange_alpha[ij];
      if (!b.empty()) energy += 0.5 * spec.exchange.coefficient * b[ij] * jk.exchange_beta[ij];
    }
  }
  return energy;
}

void combinations() {
  for (double cutoff : {1e-10, 0.15})
    for (bool uhf : {false, true})
      for (bool j_df : {false, true})
        for (bool k_df : {false, true})
          for (bool j : {false, true})
            for (bool k : {false, true}) {
              auto spec = make_hf_fock_spec(uhf ? FockSpin::Unrestricted : FockSpin::Restricted);
              spec.coulomb = {j, 1.7, FockOperator::FullRange, 0.0,
                              j_df ? FockApproximation::DensityFitted : FockApproximation::Exact};
              spec.exchange = {k, uhf ? -0.7 : 0.23, FockOperator::FullRange, 0.0,
                               k_df ? FockApproximation::DensityFitted : FockApproximation::Exact};
              const std::vector<double> a{1.2, 0.3, 0.3, 0.7};
              const std::vector<double> b =
                  uhf ? std::vector<double>{0.1, -0.05, -0.05, 0.4} : std::vector<double>{};
              const auto exact = exact_data(0.0);
              const auto df = fitted_data(0.0, cutoff);
              const auto prepared = plan(spec, exact, df);
              const auto actual = prepared.build(a, b);
              const auto expected = oracle(spec, exact, df, a, b);
              matrix(actual.coulomb, expected.coulomb, "independent J");
              matrix(actual.exchange_alpha, expected.exchange_alpha, "independent alpha K");
              matrix(actual.exchange_beta, expected.exchange_beta, "independent beta K");
              const auto fock = assemble_fock(prepared.strategy(), std::vector<double>(4), actual);
              double assembled_energy = 0.0;
              for (std::size_t ij = 0; ij < 4; ++ij)
                assembled_energy +=
                    0.5 * (a[ij] * fock.alpha[ij] + (uhf ? b[ij] * fock.beta[ij] : 0.0));
              const double energy = reference_energy(spec, 0.0, cutoff, a, b);
              close(contract_fock_energy(prepared.strategy(), actual, a, b), energy, 5e-13,
                    "energy weights");
              close(assembled_energy, energy, 5e-13, "Fock/energy agreement");
              constexpr double step = 1e-4;
              const double fd = (reference_energy(spec, step, cutoff, a, b) -
                                 reference_energy(spec, -step, cutoff, a, b)) /
                                (2.0 * step);
              close(prepared.energy_derivative(a, b).at(0), fd, 2e-9,
                    "mixed-provider derivative/metric response");
            }
}

void preflight() {
  const auto exact = exact_data(0.0);
  const auto df = fitted_data(0.0, 1e-10);
  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  spec.coulomb.approximation = FockApproximation::DensityFitted;
  const auto resolved = resolve_fock_build(spec, FockBackend::Cpu);
  const CpuFockProviderView j(df), k(exact);
  rejected([&] { (void)CpuFockPlanView(resolved, 2, 1, {}, k); }, "missing J accepted");
  rejected([&] { (void)CpuFockPlanView(resolved, 2, 1, k, j); }, "swapped approximations accepted");
  rejected([&] { (void)CpuFockPlanView(resolved, 3, 1, j, k); }, "AO mismatch accepted");
  rejected([&] { (void)CpuFockPlanView(resolved, 2, 2, j, k); }, "coordinate mismatch accepted");
  auto invalid = df;
  invalid.raw.metric_derivative.clear();
  rejected([&] { (void)CpuFockPlanView(resolved, 2, 1, CpuFockProviderView(invalid), k); },
           "missing derivative accepted during value preflight");
  invalid = df;
  invalid.metric_relative_threshold = 1e-7;
  rejected([&] { (void)CpuFockPlanView(resolved, 2, 1, CpuFockProviderView(invalid), k); },
           "value/derivative cutoff mismatch accepted");
  const auto prepared = plan(spec, exact, df);
  const std::vector<double> density{1.2, 0.3, 0.3, 0.7};
  const auto before = prepared.build(density);
  rejected([&] { (void)prepared.build({1.0}); }, "invalid density accepted");
  matrix(prepared.build(density).coulomb, before.coulomb, "failed execution changed shared state");
  spec.coulomb.coefficient = spec.exchange.coefficient = 0.0;
  const auto zero = plan(spec, exact, df);
  const auto raw = zero.build(density);
  require(raw.coulomb.size() == 4 && raw.exchange_alpha.size() == 4, "zero weight lost raw terms");
  close(zero.energy_derivative(density)[0], 0.0, 0.0, "zero-weight response");
  spec.derivative_order = 0;
  const auto energy_only = plan(spec, exact, df);
  rejected([&] { (void)energy_only.energy_derivative(density); },
           "unrequested derivative executed");
  const auto capability =
      fock_provider_capabilities(FockApproximation::DensityFitted, FockBackend::Cpu);
  require(capability.independent_terms && capability.arbitrary_coefficients &&
              !capability.legacy_adapter_only,
          "CPU DF capabilities do not describe independent providers");
}

void molecular_endpoints() {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0, 0, {{3.42525091, 0.15432897}, {0.62391373, 0.53532814}, {0.16885540, 0.44463454}}},
      {1, 0, {{3.42525091, 0.15432897}, {0.62391373, 0.53532814}, {0.16885540, 0.44463454}}}};
  system.electron_count = 2;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "molecular basis validation failed");
  for (bool uhf : {false, true})
    for (bool j_df : {false, true})
      for (bool k_df : {false, true}) {
        ScfOptions options;
        options.energy_tolerance = 1e-12;
        options.density_tolerance = 1e-10;
        auto spec = make_hf_fock_spec(uhf ? FockSpin::Unrestricted : FockSpin::Restricted);
        spec.coulomb.approximation =
            j_df ? FockApproximation::DensityFitted : FockApproximation::Exact;
        spec.exchange.approximation =
            k_df ? FockApproximation::DensityFitted : FockApproximation::Exact;
        options.resolved_fock_build = resolve_fock_build(spec, FockBackend::Cpu);
        const auto result = run_cpu_fock_strategy(system, nullptr, options);
        require(result.converged && result.forces.size() == 6, "mixed molecular SCF/forces failed");
        const auto warm = run_cpu_fock_strategy(system, nullptr, options, &result.density);
        require(warm.converged && warm.initial_density_used, "mixed-provider warm replay failed");
        close(warm.energy, result.energy, 1e-11, "warm energy drift");
        // SCF energy/density tolerances do not imply bitwise force equality.
        matrix(warm.forces, result.forces, "warm force drift", 1e-9);
        constexpr double step = 1e-4;
        auto plus = system, minus = system;
        plus.atoms[1].position[2] += step;
        minus.atoms[1].position[2] -= step;
        options.compute_forces = false;
        const auto ep = run_cpu_fock_strategy(plus, nullptr, options, &result.density);
        const auto em = run_cpu_fock_strategy(minus, nullptr, options, &result.density);
        require(ep.converged && em.converged && ep.forces.empty() && em.forces.empty(),
                "changed-geometry energy-only solve failed");
        close(result.forces[5], -(ep.energy - em.energy) / (2 * step), 2e-8,
              "mixed-provider stationary molecular force");
      }

  auto helium = system;
  helium.atoms = {{2, {0.0, 0.0, 0.0}}};
  helium.shells.resize(1);
  ScfOptions options;
  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  spec.coulomb.approximation = FockApproximation::DensityFitted;
  options.resolved_fock_build = resolve_fock_build(spec, FockBackend::Cpu);
  FleetPlan fleet({system, helium, system}, VIBEQC_METHOD_RHF, options, true, false, false, false,
                  0);
  const auto first = fleet.execute({});
  for (const auto& item : first)
    require(item.status == VIBEQC_STATUS_SUCCESS, "mixed ragged fleet failed");
  std::vector<std::optional<std::vector<double>>> bad(3);
  bad[1] = std::vector<double>{0.0};
  const auto isolated = fleet.execute(bad);
  require(isolated[1].status == VIBEQC_STATUS_INVALID_ARGUMENT, "malformed fleet item accepted");
  for (std::size_t i : {0U, 2U}) {
    require(isolated[i].status == VIBEQC_STATUS_SUCCESS && isolated[i].warm_start_used,
            "failed neighbor damaged mixed-provider warm state");
    close(isolated[i].scf.energy, first[i].scf.energy, 1e-10, "ragged replay energy drift");
  }
  const auto recovered = fleet.execute({});
  require(recovered[1].status == VIBEQC_STATUS_SUCCESS && recovered[1].warm_start_used,
          "rejected coordinates replaced previous warm state");
}

void prepared_identity() {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {{0, 0, {{0.8, 1.0}}}, {1, 0, {{0.6, 1.0}}}};
  system.electron_count = 2;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "prepared identity fixture failed");
  const auto exact = resolve_fock_build(make_hf_fock_spec(FockSpin::Restricted), FockBackend::Cpu);
  PreparedFockPlan plan(system, nullptr, exact);
  const std::vector<double> density{0.8, 0.1, 0.1, 0.6};
  const auto before = plan.build(density);
  require(plan.matches(system, nullptr, exact, -1, 0), "identical source failed compatibility");
  auto changed = system;
  changed.atoms[1].position[2] += 0.1;
  require(!plan.matches(changed, nullptr, exact, -1, 0), "stale geometry accepted");
  changed = system;
  changed.shells[0].primitives[0].exponent *= 1.2;
  require(!plan.matches(changed, nullptr, exact, -1, 0), "same-shaped changed basis accepted");
  require(plan.matches(system, &changed, exact, -1, 0), "unused auxiliary changed exact identity");
  auto spec = exact.spec;
  spec.exchange.coefficient = -0.2;
  require(!plan.matches(system, nullptr, resolve_fock_build(spec, FockBackend::Cpu), -1, 0),
          "changed Fock coefficient accepted as reusable source");
  spec = exact.spec;
  spec.coulomb.approximation = FockApproximation::DensityFitted;
  const auto mixed = resolve_fock_build(spec, FockBackend::Cpu);
  PreparedFockPlan fitted(system, &system, mixed);
  require(!fitted.matches(system, &changed, mixed, -1, 0), "changed auxiliary basis accepted");
  require(!fitted.matches(system, &system, resolve_fock_build(spec, FockBackend::Cpu, 1e-12, 1e-6),
                          -1, 0),
          "changed metric cutoff accepted");
  spec.derivative_order = 0;
  PreparedFockPlan values(system, &system, resolve_fock_build(spec, FockBackend::Cpu));
  rejected([&] { (void)values.energy_derivative(density); }, "value-only source returned response");
  matrix(plan.build(density).coulomb, before.coulomb, "prepared owner changed after other sources");
}
}  // namespace

int main() {
  try {
    combinations();
    preflight();
    molecular_endpoints();
    prepared_identity();
    std::cout << "CPU Fock providers: independent raw matrices, energy, metric response, preflight "
                 "PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
