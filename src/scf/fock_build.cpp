#include "scf/fock_build.hpp"

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
  return value == FockApproximation::Exact || value == FockApproximation::DensityFitted;
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
  require(term.op == FockOperator::FullRange,
          "short-/long-range Fock providers are not implemented");
  require(term.omega == 0.0, "full-range Fock terms require omega=0");
  // Eliminate negative zero from serialized mathematical identities.
  if (term.coefficient == 0.0) term.coefficient = 0.0;
  term.omega = 0.0;
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
    require(std::isfinite(term->coefficient) && term->op == FockOperator::FullRange &&
                term->omega == 0.0 && term->approximation == FockApproximation::Exact,
            "exact raw Fock consumer cannot execute another operator/provider");
  }
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

}  // namespace

FockProviderCapabilities fock_provider_capabilities(FockApproximation approximation,
                                                    FockBackend backend) {
  require(valid(approximation) && valid(backend), "unknown Fock provider/backend");
  FockProviderCapabilities capabilities;
  capabilities.independent_terms = true;
  capabilities.arbitrary_coefficients = capabilities.independent_terms;
  capabilities.legacy_adapter_only = false;
  return capabilities;
}

FockBuildSpec make_hf_fock_spec(FockSpin spin, FockApproximation approximation) {
  require(valid(spin) && valid(approximation), "unknown HF spin/provider");
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
  const bool fitted = spec.coulomb.approximation == FockApproximation::DensityFitted ||
                      spec.exchange.approximation == FockApproximation::DensityFitted;
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
    result.schedule =
        fitted ? (fitted_hf ? FockSchedule::LegacyDensityFitting : FockSchedule::CpuIndependent)
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

double contract_fock_energy(const ResolvedFockBuild& strategy, const DirectJkMatrices& jk,
                            std::span<const double> density, std::span<const double> beta) {
  validate_resolved_fock_build(strategy);
  const std::size_t count = matrix_size(jk.nbf);
  validate_densities(strategy.spec.spin, count, density, beta);
  const bool unrestricted = strategy.spec.spin == FockSpin::Unrestricted;
  require(
      jk.coulomb.size() == (strategy.spec.coulomb.present ? count : 0) &&
          jk.exchange_alpha.size() == (strategy.spec.exchange.present ? count : 0) &&
          jk.exchange_beta.size() == (strategy.spec.exchange.present && unrestricted ? count : 0),
      "raw J/K outputs do not match the resolved Fock energy terms");
  double result = 0.0;
  for (std::size_t ij = 0; ij < density.size(); ++ij) {
    const double total = density[ij] + (unrestricted ? beta[ij] : 0.0);
    if (strategy.spec.coulomb.present)
      result += 0.5 * total * strategy.spec.coulomb.coefficient * jk.coulomb[ij];
    if (strategy.spec.exchange.present) {
      result += 0.5 * density[ij] * strategy.spec.exchange.coefficient * jk.exchange_alpha[ij];
      if (unrestricted)
        result += 0.5 * beta[ij] * strategy.spec.exchange.coefficient * jk.exchange_beta[ij];
    }
  }
  return result;
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
