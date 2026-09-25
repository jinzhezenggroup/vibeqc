#include "scf/cuda_fock_execution.hpp"

#include "scf/cuda_direct_jk_device.hpp"

namespace vibeqc::scf {
namespace {

bool exact_full_range(const FockTermSpec& term) noexcept {
  return !term.present ||
         (term.approximation == FockApproximation::Exact && term.op == FockOperator::FullRange);
}

bool exact_value_exchange(const FockTermSpec& term) noexcept {
  if (!term.present) return true;
  if (term.approximation != FockApproximation::Exact) return false;
  if (term.op == FockOperator::FullRange) return true;
  return (term.op == FockOperator::ShortRange || term.op == FockOperator::LongRange) &&
         term.omega >= 0.0;
}

}  // namespace

PreparedCudaFockBinding prepared_cuda_fock_binding(const PreparedFockPlan& plan) noexcept {
  const auto& strategy = plan.strategy();
  if (strategy.backend != FockBackend::Cuda || strategy.spec.derivative_order != 0 ||
      !exact_full_range(strategy.spec.coulomb) || !exact_value_exchange(strategy.spec.exchange))
    return {};

  auto* source = plan.cuda_direct_source();
  if (!source) return {};
  const auto diagnostic = cuda_direct_jk_plan_diagnostic(source);
  if (!diagnostic.nbf) return {};

  return {cuda_direct_jk_device(source), cuda_direct_jk_stream(source), source, diagnostic.nbf};
}

vibeqc_status enqueue_prepared_cuda_fock(const PreparedFockPlan& plan, const double* density,
                                         const double* beta, std::size_t matrix_elements,
                                         double* coulomb, double* alpha_exchange,
                                         double* beta_exchange, int* numerical_error,
                                         bool mixed_coulomb, std::string& detail) {
  const auto binding = prepared_cuda_fock_binding(plan);
  if (!binding) {
    detail = "prepared CUDA Fock owner has no single-provider resident value execution";
    return VIBEQC_STATUS_NOT_IMPLEMENTED;
  }

  auto* source = plan.cuda_direct_source();
  if (!source) {
    detail = "prepared CUDA Fock source became unavailable";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  return mixed_coulomb ? enqueue_cuda_direct_jk_device_mixed_j(
                             source, plan.strategy().spec, density, beta, matrix_elements, coulomb,
                             alpha_exchange, beta_exchange, numerical_error, detail)
                       : enqueue_cuda_direct_jk_device(source, plan.strategy().spec, density, beta,
                                                       matrix_elements, coulomb, alpha_exchange,
                                                       beta_exchange, numerical_error, detail);
}

}  // namespace vibeqc::scf
