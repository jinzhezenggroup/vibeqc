#include "scf/cuda_fock_provider.hpp"

#include <cmath>
#include <stdexcept>

#include "scf/df_response_weights.hpp"

namespace vibeqc::scf {
namespace {
void require(bool condition, const char* detail) {
  if (!condition) throw std::invalid_argument(detail);
}
void checked(vibeqc_status status, const std::string& detail) {
  if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status == VIBEQC_STATUS_INVALID_ARGUMENT) throw std::invalid_argument(detail);
  if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}
}  // namespace

CudaFockProviderView::CudaFockProviderView(CudaDirectJkPlan* exact, std::size_t item)
    : exact_(exact), item_(item) {
  require(exact != nullptr, "null exact CUDA Fock provider");
}
CudaFockProviderView::CudaFockProviderView(CudaDensityFittingJkPlan* fitted,
                                           const DensityFittingScfData& data, std::size_t item)
    : fitted_(fitted), data_(&data), item_(item) {
  require(fitted != nullptr, "null fitted CUDA Fock provider");
}
FockApproximation CudaFockProviderView::approximation() const {
  return exact_ ? FockApproximation::Exact : FockApproximation::DensityFitted;
}
std::size_t CudaFockProviderView::nbf() const {
  return exact_ ? cuda_direct_jk_plan_diagnostic(exact_).nbf : data_->raw.nbf;
}
std::size_t CudaFockProviderView::ncoord() const {
  return exact_ ? cuda_direct_jk_plan_diagnostic(exact_).coordinates_per_item : data_->raw.ncoord;
}
void CudaFockProviderView::validate(const ResolvedFockBuild& strategy) const {
  if (exact_) {
    const auto info = cuda_direct_jk_plan_diagnostic(exact_);
    require(item_ < info.batch_size && info.derivative_order >= strategy.spec.derivative_order &&
                info.screening_tolerance == strategy.screening_tolerance,
            "CUDA direct Fock item/capability/screening mismatch");
  } else {
    require(cuda_density_fitting_jk_plan_matches(fitted_, item_, nbf(), data_->raw.naux,
                                                 strategy.metric_relative_threshold) &&
                data_->metric_relative_threshold == strategy.metric_relative_threshold,
            "CUDA DF Fock item/dimensions/cutoff mismatch");
    if (strategy.spec.derivative_order)
      require(data_->df_gradient_orbital && data_->df_gradient_auxiliary &&
                  data_->df_gradient_budget > 0 &&
                  ncoord() == 3 * data_->df_gradient_orbital->atoms.size(),
              "CUDA DF Fock source lacks matching generated derivative metadata");
  }
}
void CudaFockProviderView::validate_density(const std::vector<double>& density,
                                            const std::vector<double>& beta,
                                            bool derivative) const {
  if (!fitted_ || !derivative) return;
  for (const auto* spin : {&density, &beta})
    if (!spin->empty())
      for (std::size_t i = 0; i < nbf(); ++i)
        for (std::size_t j = 0; j < nbf(); ++j)
          require(std::abs((*spin)[i * nbf() + j] - (*spin)[j * nbf() + i]) <= 1e-10,
                  "generated CUDA DF response requires symmetric densities");
}
DirectJkMatrices CudaFockProviderView::build(FockBuildSpec spec, const std::vector<double>& density,
                                             const std::vector<double>& beta) const {
  DirectJkMatrices out;
  out.nbf = nbf();
  std::string detail;
  vibeqc_status status;
  if (exact_)
    status = execute_cuda_direct_jk_item(exact_, item_, spec, density, beta, out.coulomb,
                                         out.exchange_alpha, out.exchange_beta, detail);
  else if (spec.spin == FockSpin::Restricted)
    status = execute_cuda_density_fitting_rhf_jk_item(
        fitted_, item_, density, out.coulomb, out.exchange_alpha, detail,
        {spec.coulomb.present, spec.exchange.present});
  else
    status = execute_cuda_density_fitting_uhf_jk_item(
        fitted_, item_, density, beta, out.coulomb, out.exchange_alpha, out.exchange_beta, detail,
        {spec.coulomb.present, spec.exchange.present});
  checked(status, detail);
  return out;
}
std::vector<double> CudaFockProviderView::derivative(FockBuildSpec spec,
                                                     const std::vector<double>& density,
                                                     const std::vector<double>& beta) const {
  std::vector<double> out(ncoord());
  std::string detail;
  if (exact_) {
    checked(
        execute_cuda_direct_energy_derivative_item(exact_, item_, spec, density, beta, out, detail),
        detail);
    return out;
  }
  const double cj = spec.coulomb.present ? spec.coulomb.coefficient : 0.0;
  // The shared generated response uses -cK*Q:M+, whereas Fock assembly uses
  // +FockExchange*K and energy supplies its independent factor of one half.
  const double ck = spec.exchange.present ? -0.5 * spec.exchange.coefficient : 0.0;
  if (cj == 0.0 && ck == 0.0) return out;
  std::vector<double> total;
  std::vector<DensityFittingDensityResponse> terms;
  if (spec.spin == FockSpin::Unrestricted) {
    if (cj != 0.0) {
      total.resize(density.size());
      for (std::size_t i = 0; i < total.size(); ++i) total[i] = density[i] + beta[i];
      terms.push_back({total, cj, 0.0});
    }
    if (ck != 0.0) {
      terms.push_back({density, 0.0, ck});
      terms.push_back({beta, 0.0, ck});
    }
  } else {
    terms.push_back({density, cj, ck});
  }
  const auto staging = total.size() * sizeof(double);
  if (staging >= data_->df_gradient_budget) throw std::bad_alloc();
  checked(execute_cuda_density_fitting_generated_force_response(
              fitted_, item_, *data_->df_gradient_orbital, *data_->df_gradient_auxiliary,
              data_->raw.three_center, data_->raw.metric, terms, data_->df_gradient_mapping,
              data_->df_gradient_budget - staging, 0, out, detail),
          detail);
  if (out.size() != ncoord()) throw std::runtime_error("CUDA DF response coordinate mismatch");
  return out;
}
}  // namespace vibeqc::scf
