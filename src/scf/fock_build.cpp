#include "scf/fock_build.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace vibeqc::scf {
namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
bool valid(FockSpin value) {
  return value == FockSpin::Restricted || value == FockSpin::Unrestricted;
}
bool valid(FockBackend value) { return value == FockBackend::Cpu || value == FockBackend::Cuda; }
bool valid(FockApproximation value) {
  return value == FockApproximation::Exact || value == FockApproximation::DensityFitted ||
         value == FockApproximation::SeminumericalCosx;
}
bool valid(FockOperator value) {
  return value == FockOperator::FullRange || value == FockOperator::ShortRange ||
         value == FockOperator::LongRange;
}
void canonicalize(FockTermSpec& term) {
  require(valid(term.op) && valid(term.approximation), "unknown Fock term operator/provider");
  require(std::isfinite(term.coefficient) && std::isfinite(term.omega) && term.omega >= 0.0,
          "Fock coefficients and nonnegative range parameters must be finite");
  if (!term.present) {
    term = {false, 0.0, FockOperator::FullRange, 0.0, FockApproximation::Exact};
    return;
  }
  if (term.op == FockOperator::FullRange) {
    require(term.omega == 0.0, "full-range Fock terms require omega=0");
  } else {
    require(term.approximation == FockApproximation::Exact,
            "range-separated Fock terms currently require the exact provider");
  }
  if (term.approximation == FockApproximation::SeminumericalCosx) {
    const auto& cosx = term.cosx;
    require(cosx.version == 1 && cosx.grid_version == 1 && cosx.radial_points > 0 &&
                cosx.angular_polar > 0 && cosx.angular_azimuth > 0 &&
                cosx.partition_iterations > 0 && std::isfinite(cosx.coincident_tolerance) &&
                cosx.coincident_tolerance > 0.0,
            "COSX v1 requires a complete finite explicit grid identity");
    for (double radius : cosx.element_radii)
      require(std::isfinite(radius) && radius >= 0.0,
              "COSX element radii must be finite and nonnegative");
    require(cosx.symmetrize && !cosx.overlap_fitting && !cosx.screening,
            "COSX v1 is explicitly symmetrized, unfitted and unscreened");
  } else {
    // Irrelevant approximation metadata must not perturb exact/DF identities.
    term.cosx = {};
  }
  // Eliminate negative zero from serialized mathematical identities.
  if (term.coefficient == 0.0) term.coefficient = 0.0;
  if (term.op == FockOperator::FullRange) term.omega = 0.0;
}
bool standard_hf_terms(const FockBuildSpec& spec, FockApproximation approximation) {
  return spec.coulomb.present && spec.exchange.present && spec.coulomb.coefficient == 1.0 &&
         spec.exchange.coefficient == (spec.spin == FockSpin::Restricted ? -0.5 : -1.0) &&
         spec.coulomb.op == FockOperator::FullRange &&
         spec.exchange.op == FockOperator::FullRange && spec.coulomb.omega == 0.0 &&
         spec.exchange.omega == 0.0 && spec.coulomb.approximation == approximation &&
         spec.exchange.approximation == approximation;
}
void require_cpu_exact_consumer(const ResolvedFockBuild& strategy) {
  validate_resolved_fock_build(strategy);
  require(strategy.spec.version == 1 && valid(strategy.spec.spin) &&
              strategy.spec.derivative_order <= 1 && strategy.backend == FockBackend::Cpu &&
              strategy.schedule == FockSchedule::CpuReference &&
              strategy.precision == FockPrecision::Float64 && !strategy.legacy_density_fitting,
          "exact raw Fock consumer requires a resolved CPU exact strategy");
  for (const auto* term : {&strategy.spec.coulomb, &strategy.spec.exchange}) {
    if (!term->present) continue;
    require(std::isfinite(term->coefficient) && term->approximation == FockApproximation::Exact,
            "exact raw Fock consumer cannot execute another provider");
    require(term->op == FockOperator::FullRange ? term->omega == 0.0
                                                : std::isfinite(term->omega) && term->omega >= 0.0,
            "exact raw Fock consumer has inconsistent operator/range identity");
  }
  require(!(strategy.spec.coulomb.present && strategy.spec.exchange.present &&
            strategy.spec.exchange.op != FockOperator::FullRange),
          "one dense exact tensor cannot serve Coulomb and range exchange together");
}
std::size_t matrix_size(std::size_t nbf) {
  require(nbf > 0 && nbf <= std::numeric_limits<std::size_t>::max() / nbf,
          "invalid Fock matrix dimension");
  return nbf * nbf;
}
void validate_densities(FockSpin spin, std::size_t count, std::span<const double> density,
                        std::span<const double> beta) {
  require(density.size() == count, "Fock density dimensions do not match the AO basis");
  require(spin == FockSpin::Unrestricted ? beta.size() == count : beta.empty(),
          "Fock spin-density layout does not match the requested spin semantics");
  for (double value : density) require(std::isfinite(value), "nonfinite Fock density");
  for (double value : beta) require(std::isfinite(value), "nonfinite Fock spin density");
}

constexpr FockProviderCapabilities supported_fock_domain() {
  FockProviderCapabilities capabilities;
  capabilities.restricted = true;
  capabilities.unrestricted = true;
  capabilities.full_range = true;
  capabilities.maximum_derivative_order = 1;
  capabilities.maximum_angular_momentum = 3;
  capabilities.cartesian = true;
  capabilities.spherical = true;
  capabilities.batching = true;
  capabilities.coulomb = true;
  capabilities.exchange = true;
  capabilities.independent_terms = true;
  capabilities.arbitrary_coefficients = true;
  return capabilities;
}

constexpr FockProviderCapabilities cpu_exact_fock_domain() {
  auto capabilities = supported_fock_domain();
  capabilities.short_range = true;
  capabilities.long_range = true;
  return capabilities;
}

constexpr FockProviderCapabilities cosx_fock_domain() {
  FockProviderCapabilities capabilities;
  capabilities.restricted = true;
  capabilities.unrestricted = true;
  capabilities.full_range = true;
  capabilities.maximum_derivative_order = 0;
  capabilities.maximum_angular_momentum = 3;
  capabilities.cartesian = true;
  capabilities.spherical = true;
  capabilities.batching = false;
  capabilities.coulomb = false;
  capabilities.exchange = true;
  capabilities.independent_terms = true;
  capabilities.arbitrary_coefficients = true;
  return capabilities;
}

constexpr FockProviderRegistration make_registration(
    std::string_view name, FockApproximation approximation, runtime::ProviderBackend backend,
    runtime::ProviderAvailability availability, std::string_view reason,
    std::string_view provenance, FockProviderCapabilities capabilities = supported_fock_domain()) {
  return {{"scf.fock", name, 1, backend},
          {approximation, capabilities},
          availability,
          runtime::ProviderFallback::None,
          runtime::ProviderRequirement::PreparedState | runtime::ProviderRequirement::Resources,
          reason,
          provenance};
}

#if VIBEQC_HAS_CUDA
constexpr auto kCudaAvailability = runtime::ProviderAvailability::Executable;
constexpr std::string_view kCudaReason{};
constexpr auto kCosxCudaAvailability = runtime::ProviderAvailability::Executable;
constexpr std::string_view kCosxCudaReason{};
#else
constexpr auto kCudaAvailability = runtime::ProviderAvailability::NotBuilt;
constexpr std::string_view kCudaReason = "CUDA support was not compiled into this build";
constexpr auto kCosxCudaAvailability = runtime::ProviderAvailability::NotBuilt;
constexpr std::string_view kCosxCudaReason = "CUDA support was not compiled into this build";
#endif

constexpr std::array<FockProviderRegistration, 6> kFockProviders{{
    make_registration("cpu.exact", FockApproximation::Exact, runtime::ProviderBackend::Cpu,
                      runtime::ProviderAvailability::Executable, {}, "src/scf/fock_provider.cpp",
                      cpu_exact_fock_domain()),
    make_registration("cpu.df", FockApproximation::DensityFitted, runtime::ProviderBackend::Cpu,
                      runtime::ProviderAvailability::Executable, {}, "src/scf/fock_provider.cpp"),
    make_registration("cuda.exact", FockApproximation::Exact, runtime::ProviderBackend::Cuda,
                      kCudaAvailability, kCudaReason, "src/scf/cuda_fock_provider.cpp"),
    make_registration("cuda.df", FockApproximation::DensityFitted, runtime::ProviderBackend::Cuda,
                      kCudaAvailability, kCudaReason, "src/scf/cuda_fock_provider.cpp"),
    make_registration("cpu.cosx", FockApproximation::SeminumericalCosx,
                      runtime::ProviderBackend::Cpu, runtime::ProviderAvailability::Unavailable,
                      "the CPU COSX path is a correctness oracle, not a Fock provider",
                      "src/dft/cosx_reference.cpp", cosx_fock_domain()),
    make_registration("cuda.cosx", FockApproximation::SeminumericalCosx,
                      runtime::ProviderBackend::Cuda, kCosxCudaAvailability, kCosxCudaReason,
                      "src/dft/cosx_fock_provider.cpp", cosx_fock_domain()),
}};

runtime::ProviderBackend registry_backend(FockBackend backend) {
  return backend == FockBackend::Cpu ? runtime::ProviderBackend::Cpu
                                     : runtime::ProviderBackend::Cuda;
}

}  // namespace

const FockProviderRegistration& fock_provider_registration(FockApproximation approximation,
                                                           FockBackend backend) {
  require(valid(approximation) && valid(backend), "unknown Fock provider/backend");
  const auto execution_backend = registry_backend(backend);
  for (const auto& provider : kFockProviders)
    if (provider.domain.approximation == approximation &&
        provider.identity.backend == execution_backend)
      return provider;
  throw std::logic_error("valid Fock provider/backend has no registry entry");
}

FockProviderCapabilities fock_provider_capabilities(FockApproximation approximation,
                                                    FockBackend backend) {
  const auto& provider = fock_provider_registration(approximation, backend);
  FockProviderCapabilities capabilities;
  capabilities.provider_version = provider.identity.version;
  if (!runtime::provider_executable(provider)) return capabilities;
  capabilities = provider.domain.capabilities;
  capabilities.available = true;
  capabilities.provider_version = provider.identity.version;
  return capabilities;
}

void require_fock_provider_executable(FockApproximation approximation, FockBackend backend) {
  const auto& provider = fock_provider_registration(approximation, backend);
  if (!runtime::provider_executable(provider))
    throw std::runtime_error(runtime::provider_diagnostic(provider, "Fock preparation"));
}

FockCosxSpec make_cosx_v1_spec(std::size_t radial_points, std::size_t angular_polar,
                               std::size_t angular_azimuth, unsigned partition_iterations,
                               double coincident_tolerance, std::array<double, 119> element_radii) {
  FockCosxSpec spec;
  spec.version = 1;
  spec.grid_version = 1;
  spec.radial_points = radial_points;
  spec.angular_polar = angular_polar;
  spec.angular_azimuth = angular_azimuth;
  spec.partition_iterations = partition_iterations;
  spec.coincident_tolerance = coincident_tolerance;
  spec.element_radii = std::move(element_radii);
  spec.symmetrize = true;
  return spec;
}

FockBuildSpec make_hf_fock_spec(FockSpin spin, FockApproximation approximation) {
  require(valid(spin) && (approximation == FockApproximation::Exact ||
                          approximation == FockApproximation::DensityFitted),
          "HF helper requires an exact or density-fitted J/K approximation");
  FockBuildSpec spec;
  spec.spin = spin;
  spec.coulomb.approximation = approximation;
  spec.exchange.approximation = approximation;
  spec.exchange.coefficient = spin == FockSpin::Restricted ? -0.5 : -1.0;
  return spec;
}

ResolvedFockBuild resolve_fock_build(FockBuildSpec spec, FockBackend backend,
                                     double screening_tolerance, double metric_relative_threshold) {
  require(spec.version == 1, "unsupported FockBuildSpec version");
  require(valid(spec.spin) && valid(backend), "unknown Fock spin/backend");
  require(spec.derivative_order <= 1, "second Fock derivatives are not implemented");
  require(std::isfinite(screening_tolerance) && screening_tolerance >= 0.0,
          "Fock screening tolerance must be nonnegative and finite");
  canonicalize(spec.coulomb);
  canonicalize(spec.exchange);
  auto validate_term = [&](const FockTermSpec& term, bool coulomb) {
    if (!term.present) return;
    if (coulomb && term.op != FockOperator::FullRange)
      throw std::invalid_argument("range-separated Coulomb is not implemented");
    const auto& capability =
        fock_provider_registration(term.approximation, backend).domain.capabilities;
    require(coulomb ? capability.coulomb : capability.exchange,
            coulomb ? "requested Fock provider does not implement Coulomb"
                    : "requested Fock provider does not implement exchange");
    require(spec.spin == FockSpin::Restricted ? capability.restricted : capability.unrestricted,
            "requested Fock spin convention is outside the provider domain");
    require(spec.derivative_order <= capability.maximum_derivative_order,
            "requested Fock derivative order is outside the provider domain");
    require(term.op == FockOperator::FullRange    ? capability.full_range
            : term.op == FockOperator::ShortRange ? capability.short_range
                                                  : capability.long_range,
            "requested Fock operator is outside the provider domain");
  };
  validate_term(spec.coulomb, true);
  validate_term(spec.exchange, false);
  if (spec.derivative_order && spec.exchange.present && spec.exchange.op != FockOperator::FullRange)
    throw std::invalid_argument(
        "range-separated Fock derivatives are not integrated into the common provider");
  const bool fitted =
      (spec.coulomb.present && spec.coulomb.approximation == FockApproximation::DensityFitted) ||
      (spec.exchange.present && spec.exchange.approximation == FockApproximation::DensityFitted);
  const bool cosx =
      spec.exchange.present && spec.exchange.approximation == FockApproximation::SeminumericalCosx;
  if (fitted) {
    require(std::isfinite(metric_relative_threshold) && metric_relative_threshold > 0.0 &&
                metric_relative_threshold < 1.0,
            "DF metric relative threshold must lie strictly between zero and one");
  }
  ResolvedFockBuild result;
  result.spec = std::move(spec);
  result.backend = backend;
  const bool fitted_hf = standard_hf_terms(result.spec, FockApproximation::DensityFitted);
  if (backend == FockBackend::Cuda)
    result.schedule = fitted_hf ? FockSchedule::LegacyDensityFitting
                      : standard_hf_terms(result.spec, FockApproximation::Exact)
                          ? FockSchedule::CudaFused
                          : FockSchedule::CudaIndependent;
  else
    result.schedule = fitted || cosx ? (fitted_hf ? FockSchedule::LegacyDensityFitting
                                                  : FockSchedule::CpuIndependent)
                                     : FockSchedule::CpuReference;
  result.screening_tolerance = screening_tolerance;
  result.metric_relative_threshold = fitted ? metric_relative_threshold : 0.0;
  result.legacy_density_fitting = fitted_hf;
  return result;
}

void validate_resolved_fock_build(const ResolvedFockBuild& strategy) {
  require(
      strategy == resolve_fock_build(strategy.spec, strategy.backend, strategy.screening_tolerance,
                                     strategy.metric_relative_threshold),
      "Fock execution state differs from its resolved mathematical request");
}

void require_exact_direct_strategy(const ResolvedFockBuild& strategy, FockSpin spin,
                                   FockBackend backend) {
  require(valid(spin) && valid(backend) && strategy.spec.version == 1 &&
              strategy.spec.spin == spin && strategy.backend == backend &&
              strategy.spec.derivative_order == 1 && strategy.precision == FockPrecision::Float64 &&
              !strategy.legacy_density_fitting &&
              standard_hf_terms(strategy.spec, FockApproximation::Exact) &&
              strategy.schedule == (backend == FockBackend::Cpu ? FockSchedule::CpuReference
                                                                : FockSchedule::CudaFused) &&
              std::isfinite(strategy.screening_tolerance) && strategy.screening_tolerance >= 0.0,
          "HF direct energy/force execution requires its resolved exact standard HF strategy");
}

DirectJkMatrices build_exact_direct_jk(const ResolvedFockBuild& strategy, std::size_t nbf,
                                       std::span<const double> eri, std::span<const double> density,
                                       std::span<const double> beta) {
  require_cpu_exact_consumer(strategy);
  const std::size_t count = matrix_size(nbf);
  validate_densities(strategy.spec.spin, count, density, beta);
  if (!strategy.spec.coulomb.present && !strategy.spec.exchange.present) {
    DirectJkMatrices result;
    result.nbf = nbf;
    return result;
  }
  require(count <= std::numeric_limits<std::size_t>::max() / count && eri.size() == count * count,
          "exact Fock ERI dimensions do not match the AO basis");
  const bool unrestricted = strategy.spec.spin == FockSpin::Unrestricted;
  DirectJkMatrices result;
  result.nbf = nbf;
  if (strategy.spec.coulomb.present) result.coulomb.resize(count);
  if (strategy.spec.exchange.present) {
    result.exchange_alpha.resize(count);
    if (unrestricted) result.exchange_beta.resize(count);
  }
  for (std::size_t i = 0; i < nbf; ++i) {
    for (std::size_t j = 0; j < nbf; ++j) {
      double coulomb = 0.0, exchange_alpha = 0.0, exchange_beta = 0.0;
      for (std::size_t k = 0; k < nbf; ++k) {
        for (std::size_t l = 0; l < nbf; ++l) {
          const std::size_t kl = k * nbf + l;
          const double alpha = density[kl];
          const double beta_value = unrestricted ? beta[kl] : 0.0;
          if (strategy.spec.coulomb.present)
            coulomb += (unrestricted ? alpha + beta_value : alpha) *
                       eri[((i * nbf + j) * nbf + k) * nbf + l];
          if (strategy.spec.exchange.present) {
            const double value = eri[((i * nbf + k) * nbf + j) * nbf + l];
            exchange_alpha += alpha * value;
            if (unrestricted) exchange_beta += beta_value * value;
          }
        }
      }
      const std::size_t ij = i * nbf + j;
      if (strategy.spec.coulomb.present) result.coulomb[ij] = coulomb;
      if (strategy.spec.exchange.present) {
        result.exchange_alpha[ij] = exchange_alpha;
        if (unrestricted) result.exchange_beta[ij] = exchange_beta;
      }
    }
  }
  return result;
}

FockMatrices assemble_fock(const ResolvedFockBuild& strategy, std::span<const double> hcore,
                           const DirectJkMatrices& jk) {
  validate_resolved_fock_build(strategy);
  const std::size_t count = matrix_size(jk.nbf);
  const bool unrestricted = strategy.spec.spin == FockSpin::Unrestricted;
  require(hcore.size() == count, "Fock Hcore dimensions do not match the AO basis");
  require(
      jk.coulomb.size() == (strategy.spec.coulomb.present ? count : 0) &&
          jk.exchange_alpha.size() == (strategy.spec.exchange.present ? count : 0) &&
          jk.exchange_beta.size() == (strategy.spec.exchange.present && unrestricted ? count : 0),
      "raw J/K outputs do not match the resolved Fock terms");
  FockMatrices result;
  result.alpha.assign(hcore.begin(), hcore.end());
  if (unrestricted) result.beta = result.alpha;
  for (std::size_t ij = 0; ij < count; ++ij) {
    const double j =
        strategy.spec.coulomb.present ? strategy.spec.coulomb.coefficient * jk.coulomb[ij] : 0.0;
    const double ka = strategy.spec.exchange.present
                          ? strategy.spec.exchange.coefficient * jk.exchange_alpha[ij]
                          : 0.0;
    result.alpha[ij] += j + ka;
    if (unrestricted) {
      const double kb = strategy.spec.exchange.present
                            ? strategy.spec.exchange.coefficient * jk.exchange_beta[ij]
                            : 0.0;
      result.beta[ij] += j + kb;
    }
  }
  return result;
}

FockEnergyComponents contract_fock_energy_components(const ResolvedFockBuild& strategy,
                                                     const DirectJkMatrices& jk,
                                                     std::span<const double> density,
                                                     std::span<const double> beta) {
  validate_resolved_fock_build(strategy);
  const std::size_t count = matrix_size(jk.nbf);
  validate_densities(strategy.spec.spin, count, density, beta);
  const bool unrestricted = strategy.spec.spin == FockSpin::Unrestricted;
  require(
      jk.coulomb.size() == (strategy.spec.coulomb.present ? count : 0) &&
          jk.exchange_alpha.size() == (strategy.spec.exchange.present ? count : 0) &&
          jk.exchange_beta.size() == (strategy.spec.exchange.present && unrestricted ? count : 0),
      "raw J/K outputs do not match the resolved Fock energy terms");
  FockEnergyComponents result;
  for (std::size_t ij = 0; ij < density.size(); ++ij) {
    const double total = density[ij] + (unrestricted ? beta[ij] : 0.0);
    if (strategy.spec.coulomb.present)
      result.coulomb += 0.5 * total * strategy.spec.coulomb.coefficient * jk.coulomb[ij];
    if (strategy.spec.exchange.present) {
      result.exchange +=
          0.5 * density[ij] * strategy.spec.exchange.coefficient * jk.exchange_alpha[ij];
      if (unrestricted)
        result.exchange +=
            0.5 * beta[ij] * strategy.spec.exchange.coefficient * jk.exchange_beta[ij];
    }
  }
  return result;
}

double contract_fock_energy(const ResolvedFockBuild& strategy, const DirectJkMatrices& jk,
                            std::span<const double> density, std::span<const double> beta) {
  return contract_fock_energy_components(strategy, jk, density, beta).total();
}

double contract_exact_direct_energy_derivative(const ResolvedFockBuild& strategy, std::size_t nbf,
                                               std::span<const double> eri_derivative,
                                               std::span<const double> density,
                                               std::span<const double> beta) {
  require(strategy.spec.derivative_order == 1, "Fock first derivatives were not requested");
  return contract_fock_energy(
      strategy, build_exact_direct_jk(strategy, nbf, eri_derivative, density, beta), density, beta);
}

}  // namespace vibeqc::scf
