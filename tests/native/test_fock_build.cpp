#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/fleet.hpp"
#include "scf/fock_build.hpp"
#include "scf/mean_field.hpp"

namespace {
using namespace vibeqc::scf;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, const std::string& message) {
  require(std::isfinite(actual) && std::abs(actual - expected) < 2.0e-13,
          message + ": actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
}

void require_matrix(std::span<const double> actual, std::span<const double> expected,
                    const std::string& message) {
  require(actual.size() == expected.size(), message + ": wrong matrix size");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    require_close(actual[i], expected[i], message + " at element " + std::to_string(i));
  }
}

template <typename Function>
void require_rejected(Function&& function, const std::string& message) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error(message);
}

// Six independent values fix all eightfold ERI permutations. Expected J/K and
// derivative results below are pinned hand contractions, not another call to
// the production contraction or a duplicate of its optimized loop ordering.
std::array<double, 16> eri_fixture(bool derivative = false) {
  const std::array<double, 9> pairs =
      derivative ? std::array<double, 9>{0.11, -0.03, 0.07, -0.03, 0.05, 0.02, 0.07, 0.02, -0.04}
                 : std::array<double, 9>{1.3, 0.2, 0.4, 0.2, 0.6, -0.1, 0.4, -0.1, 0.9};
  const std::array<std::size_t, 4> pair_index{0, 1, 1, 2};
  std::array<double, 16> values{};
  for (std::size_t ij = 0; ij < 4; ++ij) {
    for (std::size_t kl = 0; kl < 4; ++kl) {
      values[ij * 4 + kl] = pairs[pair_index[ij] * 3 + pair_index[kl]];
    }
  }
  return values;
}

const std::array<double, 4> hcore{-1.0, 0.12, 0.12, -0.8};
const std::array<double, 4> density{1.2, 0.3, 0.3, 0.7};
const std::array<double, 4> alpha_density{0.7, 0.2, 0.2, 0.1};
const std::array<double, 4> beta_density{0.1, -0.05, -0.05, 0.4};
const std::array<double, 4> zero_density{};

void verify_restricted_raw_and_assembly() {
  const auto strategy =
      resolve_fock_build(make_hf_fock_spec(FockSpin::Restricted), FockBackend::Cpu);
  const auto eri = eri_fixture();
  const auto jk = build_exact_direct_jk(strategy, 2, eri, density);
  require(jk.nbf == 2 && jk.exchange_beta.empty(), "RHF raw result spin/shape is wrong");
  require_matrix(jk.coulomb, std::array{1.96, 0.53, 0.53, 1.05}, "RHF raw J");
  require_matrix(jk.exchange_alpha, std::array{2.10, 0.47, 0.47, 1.29}, "RHF raw K");
  const auto fock = assemble_fock(strategy, hcore, jk);
  require_matrix(fock.alpha, std::array{-0.09, 0.415, 0.415, -0.395}, "RHF Fock");
  require(fock.beta.empty(), "RHF assembly emitted an unrequested beta Fock");
  double two_electron_energy = 0.0;
  for (std::size_t i = 0; i < density.size(); ++i) {
    two_electron_energy += 0.5 * density[i] * (fock.alpha[i] - hcore[i]);
  }
  require_close(two_electron_energy, 0.77625, "RHF density convention or double counting");
  require_close(contract_exact_direct_energy_derivative(strategy, 2, eri_fixture(true), density),
                0.0695, "RHF derivative weights");

  // A nonsymmetric density exposes exchange index/layout errors that a
  // physical symmetric SCF density cannot distinguish.
  const std::array<double, 4> nonsymmetric{1.2, 0.31, -0.07, 0.7};
  const auto nonsymmetric_jk = build_exact_direct_jk(strategy, 2, eri, nonsymmetric);
  require_matrix(nonsymmetric_jk.coulomb, std::array{1.888, 0.314, 0.314, 1.086},
                 "nonsymmetric-density J");
  require_matrix(nonsymmetric_jk.exchange_alpha, std::array{2.028, 0.252, 0.328, 1.326},
                 "nonsymmetric-density K must retain its orientation");
}

void verify_unrestricted_and_closed_shell_limit() {
  const auto strategy =
      resolve_fock_build(make_hf_fock_spec(FockSpin::Unrestricted), FockBackend::Cpu);
  const auto eri = eri_fixture();
  const auto jk = build_exact_direct_jk(strategy, 2, eri, alpha_density, beta_density);
  require_matrix(jk.coulomb, std::array{1.30, 0.29, 0.29, 0.74}, "UHF total-density J");
  require_matrix(jk.exchange_alpha, std::array{1.05, 0.33, 0.33, 0.47}, "UHF alpha K");
  require_matrix(jk.exchange_beta, std::array{0.35, -0.07, -0.07, 0.43}, "UHF beta K");
  const auto fock = assemble_fock(strategy, hcore, jk);
  require_matrix(fock.alpha, std::array{-0.75, 0.08, 0.08, -0.53}, "UHF alpha Fock");
  require_matrix(fock.beta, std::array{-0.05, 0.48, 0.48, -0.49}, "UHF beta Fock");
  require_close(contract_exact_direct_energy_derivative(strategy, 2, eri_fixture(true),
                                                        alpha_density, beta_density),
                0.02965, "UHF derivative must use total J and separate spin K");

  const std::array<double, 4> half_density{0.6, 0.15, 0.15, 0.35};
  const auto closed_jk = build_exact_direct_jk(strategy, 2, eri, half_density, half_density);
  const auto closed_fock = assemble_fock(strategy, hcore, closed_jk);
  const auto rhf_strategy =
      resolve_fock_build(make_hf_fock_spec(FockSpin::Restricted), FockBackend::Cpu);
  const auto rhf_fock =
      assemble_fock(rhf_strategy, hcore, build_exact_direct_jk(rhf_strategy, 2, eri, density));
  require_matrix(closed_fock.alpha, rhf_fock.alpha, "closed-shell UHF alpha/RHF limit");
  require_matrix(closed_fock.beta, rhf_fock.alpha, "closed-shell UHF beta/RHF limit");
  require_close(contract_exact_direct_energy_derivative(strategy, 2, eri_fixture(true),
                                                        half_density, half_density),
                0.0695, "closed-shell UHF/RHF derivative limit");

  const auto polarized = build_exact_direct_jk(strategy, 2, eri, alpha_density, zero_density);
  require_matrix(polarized.coulomb, std::array{1.03, 0.37, 0.37, 0.33}, "polarized J");
  require_matrix(polarized.exchange_beta, zero_density, "empty beta occupation produced exchange");
}

void verify_independent_terms_and_coefficients() {
  const auto eri = eri_fixture();
  for (const auto spin : {FockSpin::Restricted, FockSpin::Unrestricted}) {
    const auto initial = make_hf_fock_spec(spin);
    const std::span<const double> first =
        spin == FockSpin::Restricted ? std::span<const double>(density) : alpha_density;
    const std::span<const double> second =
        spin == FockSpin::Restricted ? std::span<const double>() : beta_density;
    for (const unsigned requested : {0U, 1U, 2U}) {
      auto spec = initial;
      spec.coulomb.present = (requested & 1U) != 0;
      spec.exchange.present = (requested & 2U) != 0;
      const auto strategy = resolve_fock_build(spec, FockBackend::Cpu);
      const auto jk = build_exact_direct_jk(
          strategy, 2, requested == 0 ? std::span<const double>() : eri, first, second);
      require(jk.coulomb.empty() == !spec.coulomb.present, "absent J was allocated");
      require(jk.exchange_alpha.empty() == !spec.exchange.present, "absent alpha K was allocated");
      require(jk.exchange_beta.empty() == (!spec.exchange.present || spin == FockSpin::Restricted),
              "absent beta K was allocated");
      const auto fock = assemble_fock(strategy, hcore, jk);
      const std::array<double, 3> derivative_rhf{0.0, 0.124, -0.0545};
      const std::array<double, 3> derivative_uhf{0.0, 0.05625, -0.0266};
      require_close(
          contract_exact_direct_energy_derivative(
              strategy, 2, requested == 0 ? std::span<const double>() : eri_fixture(true), first,
              second),
          spin == FockSpin::Restricted ? derivative_rhf[requested] : derivative_uhf[requested],
          "independently absent J/K derivative");
      if (requested == 1) {
        const std::array<double, 4> expected = spin == FockSpin::Restricted
                                                   ? std::array{0.96, 0.65, 0.65, 0.25}
                                                   : std::array{0.30, 0.41, 0.41, -0.06};
        require_matrix(fock.alpha, expected, "J-only Fock");
        if (spin == FockSpin::Unrestricted) require_matrix(fock.beta, expected, "UHF J-only beta");
      } else if (requested == 2) {
        require_matrix(fock.alpha,
                       spin == FockSpin::Restricted ? std::array{-2.05, -0.115, -0.115, -1.445}
                                                    : std::array{-2.05, -0.21, -0.21, -1.27},
                       "K-only Fock");
        if (spin == FockSpin::Unrestricted) {
          require_matrix(fock.beta, std::array{-1.35, 0.19, 0.19, -1.23}, "UHF K-only beta");
        }
      }
      if (requested == 0) {
        require_matrix(fock.alpha, hcore, "both terms absent must return hcore");
        if (spin == FockSpin::Unrestricted) {
          require_matrix(fock.beta, hcore, "both terms absent UHF beta must return hcore");
        }
      }
    }
  }
  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  spec.coulomb.coefficient = 1.7;
  spec.exchange.coefficient = 0.23;
  const auto strategy = resolve_fock_build(spec, FockBackend::Cpu);
  const auto jk = build_exact_direct_jk(strategy, 2, eri, density);
  require_matrix(jk.coulomb, std::array{1.96, 0.53, 0.53, 1.05}, "coefficients leaked into raw J");
  require_matrix(jk.exchange_alpha, std::array{2.10, 0.47, 0.47, 1.29},
                 "coefficients leaked into raw K");
  require_matrix(assemble_fock(strategy, hcore, jk).alpha,
                 std::array{2.815, 1.1291, 1.1291, 1.2817}, "general coefficients applied twice");
  require_close(contract_exact_direct_energy_derivative(strategy, 2, eri_fixture(true), density),
                0.23587, "general coefficients not shared with derivative dispatch");
  require_rejected(
      [&] { require_exact_direct_strategy(strategy, FockSpin::Restricted, FockBackend::Cpu); },
      "HF entry accepted a different mathematical method");
}

void verify_unrestricted_coefficients_and_capabilities() {
  auto spec = make_hf_fock_spec(FockSpin::Unrestricted);
  spec.coulomb.coefficient = 0.4;
  spec.exchange.coefficient = -0.7;
  const auto strategy = resolve_fock_build(spec, FockBackend::Cpu);
  const auto jk = build_exact_direct_jk(strategy, 2, eri_fixture(), alpha_density, beta_density);
  const auto fock = assemble_fock(strategy, hcore, jk);
  require_matrix(fock.alpha, std::array{-1.215, 0.005, 0.005, -0.833},
                 "general UHF alpha coefficients");
  require_matrix(fock.beta, std::array{-0.725, 0.285, 0.285, -0.805},
                 "general UHF beta coefficients");
  require_close(contract_exact_direct_energy_derivative(strategy, 2, eri_fixture(true),
                                                        alpha_density, beta_density),
                0.00388, "general UHF derivative coefficients");

  spec.coulomb.coefficient = 0.0;
  spec.exchange.coefficient = 0.0;
  const auto zero_weight = resolve_fock_build(spec, FockBackend::Cpu);
  const auto zero_jk =
      build_exact_direct_jk(zero_weight, 2, eri_fixture(), alpha_density, beta_density);
  require(
      !zero_jk.coulomb.empty() && !zero_jk.exchange_alpha.empty() && !zero_jk.exchange_beta.empty(),
      "a requested zero-coefficient raw term was confused with an absent term");
  const auto zero_fock = assemble_fock(zero_weight, hcore, zero_jk);
  require_matrix(zero_fock.alpha, hcore, "zero-weight UHF alpha Fock");
  require_matrix(zero_fock.beta, hcore, "zero-weight UHF beta Fock");

  const auto cpu = fock_provider_capabilities(FockApproximation::Exact, FockBackend::Cpu);
  const auto cuda = fock_provider_capabilities(FockApproximation::Exact, FockBackend::Cuda);
  const auto fitted =
      fock_provider_capabilities(FockApproximation::DensityFitted, FockBackend::Cuda);
  require(cpu.restricted && cpu.unrestricted && cpu.full_range && !cpu.short_range &&
              !cpu.long_range && cpu.maximum_derivative_order == 1 &&
              cpu.maximum_angular_momentum == 3 && cpu.cartesian && cpu.spherical &&
              cpu.independent_terms && cpu.arbitrary_coefficients && !cpu.legacy_adapter_only,
          "CPU exact capabilities misrepresent the executable contract");
  require(!cuda.independent_terms && !cuda.arbitrary_coefficients && !cuda.legacy_adapter_only,
          "CUDA advertised independent terms before its fused consumer supports them");
  require(fitted.legacy_adapter_only && !fitted.independent_terms && !fitted.arbitrary_coefficients,
          "legacy fitted adapter advertised a migrated independent provider");
  require_rejected(
      [&] {
        (void)fock_provider_capabilities(static_cast<FockApproximation>(99), FockBackend::Cpu);
      },
      "capability query accepted an unknown approximation");
  auto forged_precision = strategy;
  forged_precision.precision = static_cast<FockPrecision>(99);
  require(forged_precision != strategy, "precision missing from execution identity");
  require_rejected(
      [&] {
        (void)build_exact_direct_jk(forged_precision, 2, eri_fixture(), alpha_density,
                                    beta_density);
      },
      "exact provider accepted an unimplemented precision");
}

void verify_preflight_and_approximation_identity() {
  const auto exact_spec = make_hf_fock_spec(FockSpin::Restricted);
  const auto exact = resolve_fock_build(exact_spec, FockBackend::Cpu);
  require_exact_direct_strategy(exact, FockSpin::Restricted, FockBackend::Cpu);
  for (const auto op : {FockOperator::ShortRange, FockOperator::LongRange}) {
    // Both J and K capability checks must occur during resolution, before any
    // integral provider executes or returns one successful partial term.
    for (const bool exchange : {false, true}) {
      auto unsupported = exact_spec;
      auto& term = exchange ? unsupported.exchange : unsupported.coulomb;
      term.op = op;
      term.omega = 0.4;
      require_rejected([&] { (void)resolve_fock_build(unsupported, FockBackend::Cpu); },
                       "unsupported range operator survived preflight");
    }
  }
  auto second_derivative = exact_spec;
  second_derivative.derivative_order = 2;
  require_rejected([&] { (void)resolve_fock_build(second_derivative, FockBackend::Cpu); },
                   "unsupported derivative order survived preflight");
  auto energy_only = exact_spec;
  energy_only.derivative_order = 0;
  const auto energy_strategy = resolve_fock_build(energy_only, FockBackend::Cpu);
  (void)build_exact_direct_jk(energy_strategy, 2, eri_fixture(), density);
  require_rejected(
      [&] {
        (void)contract_exact_direct_energy_derivative(energy_strategy, 2, eri_fixture(true),
                                                      density);
      },
      "energy-only strategy silently supplied derivatives");
  require_rejected(
      [&] {
        require_exact_direct_strategy(energy_strategy, FockSpin::Restricted, FockBackend::Cpu);
      },
      "HF entry accepted an incomplete energy/force strategy");

  for (const bool exchange : {false, true}) {
    auto mixed = exact_spec;
    (exchange ? mixed.exchange : mixed.coulomb).approximation = FockApproximation::DensityFitted;
    require_rejected([&] { (void)resolve_fock_build(mixed, FockBackend::Cuda); },
                     "unimplemented mixed exact/DF approximation was silently selected");
  }
  const auto fitted = resolve_fock_build(
      make_hf_fock_spec(FockSpin::Restricted, FockApproximation::DensityFitted), FockBackend::Cuda);
  require(fitted.legacy_density_fitting &&
              fitted.spec.coulomb.approximation == FockApproximation::DensityFitted &&
              fitted.spec.exchange.approximation == FockApproximation::DensityFitted,
          "resolved fitted provider lost approximation identity");
  require_rejected([&] { (void)build_exact_direct_jk(fitted, 2, eri_fixture(), density); },
                   "exact provider consumed a fitted energy strategy");
  require_rejected(
      [&] { (void)contract_exact_direct_energy_derivative(fitted, 2, eri_fixture(true), density); },
      "fitted energy silently received an exact derivative");
  require_rejected(
      [&] { require_exact_direct_strategy(fitted, FockSpin::Restricted, FockBackend::Cuda); },
      "exact HF entry accepted a fitted provider");
  require(exact.spec.coulomb.approximation == FockApproximation::Exact &&
              exact.spec.exchange.approximation == FockApproximation::Exact,
          "default provider resolution changed the declared approximation");

  auto custom_cuda = exact_spec;
  custom_cuda.exchange.coefficient = -0.2;
  require_rejected([&] { (void)resolve_fock_build(custom_cuda, FockBackend::Cuda); },
                   "CUDA fixed-HF schedule accepted general exchange coefficients");
  custom_cuda = exact_spec;
  custom_cuda.exchange.present = false;
  require_rejected([&] { (void)resolve_fock_build(custom_cuda, FockBackend::Cuda); },
                   "CUDA fixed-HF schedule silently computed unrequested K");
}

vibeqc::core::System hydrogen_molecule() {
  vibeqc::core::System system;
  system.atoms = {{1, {0.0, 0.0, -0.7}}, {1, {0.0, 0.0, 0.7}}};
  system.shells = {
      {0, 0, {{3.42525091, 0.15432897}, {0.62391373, 0.53532814}, {0.16885540, 0.44463454}}},
      {1, 0, {{3.42525091, 0.15432897}, {0.62391373, 0.53532814}, {0.16885540, 0.44463454}}},
  };
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "SCF preflight H2 fixture normalization failed: " + detail);
  return system;
}

template <typename Function>
void verify_entry_rejects_mismatched_strategies(FockSpin spin, FockBackend backend,
                                                Function&& evaluate, const std::string& entry) {
  ScfOptions options;
  const auto spec = make_hf_fock_spec(spin);
  const auto other_backend = backend == FockBackend::Cpu ? FockBackend::Cuda : FockBackend::Cpu;
  options.resolved_fock_build =
      resolve_fock_build(spec, other_backend, options.screening_tolerance);
  require_rejected([&] { evaluate(options); }, entry + " accepted the wrong resolved backend");

  const auto other_spin =
      spin == FockSpin::Restricted ? FockSpin::Unrestricted : FockSpin::Restricted;
  options.resolved_fock_build =
      resolve_fock_build(make_hf_fock_spec(other_spin), backend, options.screening_tolerance);
  require_rejected([&] { evaluate(options); }, entry + " accepted the wrong resolved spin");

  options.resolved_fock_build = resolve_fock_build(spec, backend, 1.0e-8);
  require_rejected([&] { evaluate(options); }, entry + " accepted mismatched resolved screening");
}

void verify_scf_entry_preflight() {
  // A valid normalized molecule ensures that an unrelated input failure cannot
  // satisfy the invalid_argument assertion for a mismatched resolved strategy.
  const auto system = hydrogen_molecule();
  verify_entry_rejects_mismatched_strategies(
      FockSpin::Restricted, FockBackend::Cpu,
      [&](const ScfOptions& options) { (void)run_rhf(system, options); }, "CPU RHF");
  verify_entry_rejects_mismatched_strategies(
      FockSpin::Unrestricted, FockBackend::Cpu,
      [&](const ScfOptions& options) { (void)run_uhf(system, options); }, "CPU UHF");

#if VIBEQC_HAS_CUDA
  // Strategy preflight must reject before CUDA initialization. These cases run
  // even on a CUDA-enabled build without a usable GPU, and must not be skipped.
  verify_entry_rejects_mismatched_strategies(
      FockSpin::Restricted, FockBackend::Cuda,
      [&](const ScfOptions& options) { (void)run_rhf_cuda(system, options, 0); }, "CUDA RHF");
  verify_entry_rejects_mismatched_strategies(
      FockSpin::Unrestricted, FockBackend::Cuda,
      [&](const ScfOptions& options) { (void)run_uhf_cuda(system, options, 0); }, "CUDA UHF");
#endif

  verify_entry_rejects_mismatched_strategies(
      FockSpin::Restricted, FockBackend::Cpu,
      [&](const ScfOptions& options) {
        FleetPlan plan({system}, VIBEQC_METHOD_RHF, options, false, false, false, false, 0);
      },
      "CPU RHF fleet");
}

void verify_identity_and_invalid_inputs() {
  auto spec = make_hf_fock_spec(FockSpin::Restricted);
  const auto cpu = resolve_fock_build(spec, FockBackend::Cpu, 1.0e-12, 1.0e-10);
  const auto other_unused_metric = resolve_fock_build(spec, FockBackend::Cpu, 1.0e-12, 1.0e-7);
  require(cpu == other_unused_metric && cpu.metric_relative_threshold == 0.0,
          "unused DF threshold changed exact-provider identity");
  const auto cuda = resolve_fock_build(spec, FockBackend::Cuda, 1.0e-12, 1.0e-10);
  require(cpu.spec == cuda.spec && cpu != cuda && cpu.backend != cuda.backend &&
              cpu.schedule != cuda.schedule,
          "mathematical semantics and backend schedule identities were conflated");
  // Existing post-HF density export uses zero to disable screening.
  for (const auto backend : {FockBackend::Cpu, FockBackend::Cuda}) {
    const auto unscreened = resolve_fock_build(spec, backend, 0.0);
    require_exact_direct_strategy(unscreened, FockSpin::Restricted, backend);
    require(unscreened.screening_tolerance == 0.0,
            "zero screening must preserve the unscreened reference request");
  }
  const auto screened = resolve_fock_build(spec, FockBackend::Cpu, 1.0e-8, 1.0e-10);
  require(cpu.spec == screened.spec && cpu != screened, "screening change retained stale identity");
  auto scaled = spec;
  scaled.exchange.coefficient = -0.3;
  require(cpu.spec != resolve_fock_build(scaled, FockBackend::Cpu).spec,
          "exchange coefficient missing from mathematical identity");
  auto absent = spec;
  absent.exchange.present = false;
  auto irrelevant = absent;
  irrelevant.exchange.op = FockOperator::ShortRange;
  irrelevant.exchange.omega = 0.4;
  irrelevant.exchange.approximation = FockApproximation::DensityFitted;
  irrelevant.exchange.coefficient = 0.8;
  require(resolve_fock_build(absent, FockBackend::Cpu) ==
              resolve_fock_build(irrelevant, FockBackend::Cpu),
          "absent operator metadata did not normalize to one identity");

  auto malformed = spec;
  malformed.version = 2;
  require_rejected([&] { (void)resolve_fock_build(malformed, FockBackend::Cpu); },
                   "unknown schema accepted");
  malformed = spec;
  malformed.spin = static_cast<FockSpin>(99);
  require_rejected([&] { (void)resolve_fock_build(malformed, FockBackend::Cpu); },
                   "unknown spin accepted");
  malformed = spec;
  malformed.exchange.op = static_cast<FockOperator>(99);
  require_rejected([&] { (void)resolve_fock_build(malformed, FockBackend::Cpu); },
                   "unknown operator accepted");
  malformed = spec;
  malformed.exchange.approximation = static_cast<FockApproximation>(99);
  require_rejected([&] { (void)resolve_fock_build(malformed, FockBackend::Cpu); },
                   "unknown approximation accepted");
  require_rejected([&] { (void)resolve_fock_build(spec, static_cast<FockBackend>(99)); },
                   "unknown backend accepted");
  for (double invalid :
       {-0.1, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    malformed = spec;
    malformed.exchange.omega = invalid;
    require_rejected([&] { (void)resolve_fock_build(malformed, FockBackend::Cpu); },
                     "invalid range parameter accepted");
    require_rejected([&] { (void)resolve_fock_build(spec, FockBackend::Cpu, invalid); },
                     "invalid screening accepted");
    require(cpu == resolve_fock_build(spec, FockBackend::Cpu, 1.0e-12, invalid),
            "unused metric threshold changed or rejected an exact strategy");
    require_rejected(
        [&] {
          (void)resolve_fock_build(
              make_hf_fock_spec(FockSpin::Restricted, FockApproximation::DensityFitted),
              FockBackend::Cuda, 1.0e-12, invalid);
        },
        "invalid fitted metric threshold accepted");
  }
  malformed = spec;
  malformed.exchange.omega = 0.4;
  require_rejected([&] { (void)resolve_fock_build(malformed, FockBackend::Cpu); },
                   "full-range operator accepted a nonzero range parameter");
  malformed = absent;
  malformed.exchange.coefficient = std::numeric_limits<double>::quiet_NaN();
  require_rejected([&] { (void)resolve_fock_build(malformed, FockBackend::Cpu); },
                   "absent term hid a nonfinite coefficient");

  require_rejected([&] { (void)build_exact_direct_jk(cpu, 2, eri_fixture(), std::array{1.0}); },
                   "raw direct provider accepted a truncated density");
  require_rejected([&] { (void)build_exact_direct_jk(cpu, 2, std::array{1.0}, density); },
                   "raw direct provider accepted a truncated ERI tensor");
  require_rejected([&] { (void)build_exact_direct_jk(cpu, 2, eri_fixture(), density, density); },
                   "restricted raw provider accepted a beta density");
  const auto unrestricted =
      resolve_fock_build(make_hf_fock_spec(FockSpin::Unrestricted), FockBackend::Cpu);
  require_rejected([&] { (void)build_exact_direct_jk(unrestricted, 2, eri_fixture(), density); },
                   "unrestricted raw provider accepted a missing beta density");
  auto nonfinite = density;
  nonfinite[1] = std::numeric_limits<double>::quiet_NaN();
  require_rejected([&] { (void)build_exact_direct_jk(cpu, 2, eri_fixture(), nonfinite); },
                   "raw direct provider accepted a nonfinite density");
  const auto valid_jk = build_exact_direct_jk(cpu, 2, eri_fixture(), density);
  require_rejected([&] { (void)assemble_fock(cpu, std::array{1.0}, valid_jk); },
                   "Fock assembly accepted a truncated hcore");
  require_rejected(
      [&] { require_exact_direct_strategy(cpu, FockSpin::Unrestricted, FockBackend::Cpu); },
      "HF entry accepted the wrong spin strategy");
  require_rejected(
      [&] { require_exact_direct_strategy(cpu, FockSpin::Restricted, FockBackend::Cuda); },
      "HF entry accepted the wrong backend strategy");
}
}  // namespace

int main() {
  try {
    verify_restricted_raw_and_assembly();
    verify_unrestricted_and_closed_shell_limit();
    verify_independent_terms_and_coefficients();
    verify_unrestricted_coefficients_and_capabilities();
    verify_preflight_and_approximation_identity();
    verify_scf_entry_preflight();
    verify_identity_and_invalid_inputs();
    std::cout
        << "exact Fock providers: raw J/K, spin, terms, derivatives, preflight, identity PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
