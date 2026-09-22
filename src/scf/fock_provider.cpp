#include "scf/fock_provider.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "scf/cuda_fock_provider.hpp"

namespace vibeqc::scf {
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
std::size_t product(std::size_t a, std::size_t b) {
  require(b == 0 || a <= std::numeric_limits<std::size_t>::max() / b,
          "Fock provider dimensions overflow");
  return a * b;
}
void finite(const std::vector<double>& values) {
  for (double value : values) require(std::isfinite(value), "nonfinite Fock provider input");
}
FockBuildSpec selected(FockBuildSpec spec, bool j, bool k) {
  spec.coulomb.present = j;
  spec.exchange.present = k;
  return spec;
}
}  // namespace

CpuFockProviderView::CpuFockProviderView(const integrals::IntegralData& exact,
                                         const std::vector<double>* range_eri,
                                         FockOperator range_operator, double range_omega)
    : exact_(&exact),
      range_eri_(range_eri),
      range_operator_(range_operator),
      range_omega_(range_omega) {}
CpuFockProviderView::CpuFockProviderView(const DensityFittingScfData& fitted) : fitted_(&fitted) {}
FockApproximation CpuFockProviderView::approximation() const {
  return exact_ ? FockApproximation::Exact : FockApproximation::DensityFitted;
}
std::size_t CpuFockProviderView::nbf() const { return exact_ ? exact_->nbf : fitted_->raw.nbf; }
std::size_t CpuFockProviderView::ncoord() const {
  return exact_ ? exact_->ncoord : fitted_->raw.ncoord;
}

void CpuFockProviderView::validate(const ResolvedFockBuild& strategy) const {
  const std::size_t matrix = product(nbf(), nbf());
  require(nbf() > 0, "empty Fock provider AO basis");
  if (exact_) {
    const std::size_t count = product(matrix, matrix);
    const bool needs_full =
        strategy.spec.coulomb.present ||
        (strategy.spec.exchange.present && strategy.spec.exchange.op == FockOperator::FullRange);
    if (needs_full) {
      require(exact_->eri.size() == count, "exact Fock provider ERI shape mismatch");
      finite(exact_->eri);
    }
    if (strategy.spec.exchange.present && strategy.spec.exchange.op != FockOperator::FullRange) {
      require(range_eri_ && range_operator_ == strategy.spec.exchange.op &&
                  range_omega_ == strategy.spec.exchange.omega && range_eri_->size() == count,
              "range-separated Fock provider operator/omega/tensor mismatch");
      finite(*range_eri_);
    }
    if (strategy.spec.derivative_order) {
      require(exact_->eri_derivative.size() == product(ncoord(), count),
              "exact Fock provider derivative shape mismatch");
      finite(exact_->eri_derivative);
    }
  } else {
    const auto& raw = fitted_->raw;
    const auto& b = fitted_->three_center;
    const std::size_t metric = product(raw.naux, raw.naux);
    const std::size_t tensor = product(matrix, raw.naux);
    require(raw.naux > 0 && raw.metric.size() == metric && raw.three_center.size() == tensor &&
                b.nbf == nbf() && b.naux == raw.naux && b.values.size() == tensor &&
                b.effective_rank > 0 && b.effective_rank <= raw.naux,
            "DF Fock provider tensor shape/rank mismatch");
    require(fitted_->metric_relative_threshold == strategy.metric_relative_threshold,
            "DF value/response metric cutoff differs from the resolved provider");
    finite(raw.metric);
    finite(raw.three_center);
    finite(b.values);
    if (!b.auxiliary_major_values.empty()) {
      require(b.auxiliary_major_values.size() == tensor,
              "DF Fock provider Q-major cache shape mismatch");
      finite(b.auxiliary_major_values);
    }
    if (strategy.spec.derivative_order) {
      const bool materialized = raw.metric_derivative.size() == product(ncoord(), metric) &&
                                raw.three_center_derivative.size() == product(ncoord(), tensor);
      const bool weighted = raw.metric_derivative.empty() && raw.three_center_derivative.empty() &&
                            fitted_->df_gradient_orbital.has_value() &&
                            fitted_->df_gradient_auxiliary.has_value();
      require(materialized || weighted, "DF Fock provider derivative shape/source mismatch");
      if (materialized) {
        finite(raw.metric_derivative);
        finite(raw.three_center_derivative);
      }
    }
  }
}

DirectJkMatrices CpuFockProviderView::build(FockBuildSpec spec, const std::vector<double>& density,
                                            const std::vector<double>& beta) const {
  if (exact_) {
    const auto strategy = resolve_fock_build(spec, FockBackend::Cpu);
    if (!spec.exchange.present || spec.exchange.op == FockOperator::FullRange)
      return build_exact_direct_jk(strategy, nbf(), exact_->eri, density, beta);
    DirectJkMatrices result;
    result.nbf = nbf();
    if (spec.coulomb.present) {
      auto j =
          build_exact_direct_jk(resolve_fock_build(selected(spec, true, false), FockBackend::Cpu),
                                nbf(), exact_->eri, density, beta);
      result.coulomb = std::move(j.coulomb);
    }
    auto k =
        build_exact_direct_jk(resolve_fock_build(selected(spec, false, true), FockBackend::Cpu),
                              nbf(), *range_eri_, density, beta);
    result.exchange_alpha = std::move(k.exchange_alpha);
    result.exchange_beta = std::move(k.exchange_beta);
    return result;
  }
  const JkTermSelection terms{spec.coulomb.present, spec.exchange.present};
  if (spec.spin == FockSpin::Restricted) {
    auto jk = build_density_fitting_rhf_jk(fitted_->three_center, density, terms);
    return {jk.nbf, std::move(jk.coulomb), std::move(jk.exchange), {}};
  }
  auto jk = build_density_fitting_uhf_jk(fitted_->three_center, density, beta, terms);
  return {jk.nbf, std::move(jk.coulomb), std::move(jk.alpha_exchange), std::move(jk.beta_exchange)};
}

std::vector<double> CpuFockProviderView::derivative(FockBuildSpec spec,
                                                    const std::vector<double>& density,
                                                    const std::vector<double>& beta) const {
  if (exact_) {
    const auto strategy = resolve_fock_build(spec, FockBackend::Cpu);
    const std::size_t eri_size = product(product(nbf(), nbf()), product(nbf(), nbf()));
    std::vector<double> result(ncoord());
    for (std::size_t coordinate = 0; coordinate < result.size(); ++coordinate)
      result[coordinate] = contract_exact_direct_energy_derivative(
          strategy, nbf(),
          std::span(exact_->eri_derivative).subspan(coordinate * eri_size, eri_size), density,
          beta);
    return result;
  }
  const JkCoefficients coefficients{spec.coulomb.present ? spec.coulomb.coefficient : 0.0,
                                    spec.exchange.present ? spec.exchange.coefficient : 0.0};
  const bool weighted =
      fitted_->raw.metric_derivative.empty() && fitted_->raw.three_center_derivative.empty() &&
      fitted_->df_gradient_orbital.has_value() && fitted_->df_gradient_auxiliary.has_value();
  if (spec.spin == FockSpin::Restricted) {
    if (weighted)
      return build_density_fitting_rhf_weighted_gradient(
                 *fitted_->df_gradient_orbital, *fitted_->df_gradient_auxiliary, fitted_->raw,
                 density, fitted_->metric_relative_threshold, coefficients)
          .derivative;
    return build_density_fitting_rhf_gradient(fitted_->raw, density,
                                              fitted_->metric_relative_threshold, coefficients)
        .derivative;
  }
  if (weighted)
    return build_density_fitting_uhf_weighted_gradient(
               *fitted_->df_gradient_orbital, *fitted_->df_gradient_auxiliary, fitted_->raw,
               density, beta, fitted_->metric_relative_threshold, coefficients)
        .derivative;
  return build_density_fitting_uhf_gradient(fitted_->raw, density, beta,
                                            fitted_->metric_relative_threshold, coefficients)
      .derivative;
}

template <class Provider>
BasicFockPlanView<Provider>::BasicFockPlanView(ResolvedFockBuild strategy, std::size_t nbf,
                                               std::size_t ncoord, std::optional<Provider> coulomb,
                                               std::optional<Provider> exchange)
    : strategy_(std::move(strategy)),
      nbf_(nbf),
      ncoord_(ncoord),
      coulomb_(coulomb),
      exchange_(exchange) {
  validate_resolved_fock_build(strategy_);
  require(strategy_.backend == Provider::backend, "Fock plan/provider backend mismatch");
  require(nbf_ > 0, "empty Fock plan AO basis");
  (void)product(nbf_, nbf_);
  auto bind = [&](const FockTermSpec& term, std::optional<Provider>& provider) {
    if (!term.present) {
      provider.reset();  // Irrelevant data cannot affect an absent term's identity or validation.
      return;
    }
    require(provider.has_value(), "missing requested Fock provider");
    require(provider->approximation() == term.approximation && provider->nbf() == nbf_ &&
                provider->ncoord() == ncoord_,
            "Fock provider semantics/dimensions mismatch");
    provider->validate(strategy_);
  };
  bind(strategy_.spec.coulomb, coulomb_);
  bind(strategy_.spec.exchange, exchange_);
}

template <class Provider>
void BasicFockPlanView<Provider>::validate_density(const std::vector<double>& density,
                                                   const std::vector<double>& beta,
                                                   bool derivative) const {
  require(density.size() == nbf_ * nbf_ &&
              (strategy_.spec.spin == FockSpin::Restricted ? beta.empty()
                                                           : beta.size() == density.size()),
          "Fock provider density/spin layout mismatch");
  finite(density);
  finite(beta);
  if (coulomb_) coulomb_->validate_density(density, beta, derivative);
  if (exchange_ && exchange_ != coulomb_) exchange_->validate_density(density, beta, derivative);
}

template <class Provider>
DirectJkMatrices BasicFockPlanView<Provider>::build(const std::vector<double>& density,
                                                    const std::vector<double>& beta) const {
  validate_density(density, beta, false);
  if (coulomb_ && exchange_ && *coulomb_ == *exchange_)
    return coulomb_->build(strategy_.spec, density, beta);
  DirectJkMatrices result;
  result.nbf = nbf_;
  if (coulomb_)
    result.coulomb = coulomb_->build(selected(strategy_.spec, true, false), density, beta).coulomb;
  if (exchange_) {
    auto jk = exchange_->build(selected(strategy_.spec, false, true), density, beta);
    result.exchange_alpha = std::move(jk.exchange_alpha);
    result.exchange_beta = std::move(jk.exchange_beta);
  }
  return result;
}

template <class Provider>
std::vector<double> BasicFockPlanView<Provider>::energy_derivative(
    const std::vector<double>& density, const std::vector<double>& beta) const {
  require(strategy_.spec.derivative_order == 1, "Fock derivatives were not requested");
  validate_density(density, beta, true);
  if (coulomb_ && exchange_ && *coulomb_ == *exchange_)
    return coulomb_->derivative(strategy_.spec, density, beta);
  std::vector<double> result(ncoord_);
  if (coulomb_) result = coulomb_->derivative(selected(strategy_.spec, true, false), density, beta);
  if (exchange_) {
    const auto response =
        exchange_->derivative(selected(strategy_.spec, false, true), density, beta);
    for (std::size_t coordinate = 0; coordinate < ncoord_; ++coordinate)
      result[coordinate] += response[coordinate];
  }
  return result;
}

template class BasicFockPlanView<CpuFockProviderView>;
template class BasicFockPlanView<CudaFockProviderView>;

}  // namespace vibeqc::scf
