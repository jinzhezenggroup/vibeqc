#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "scf/cuda/df_jk_internal.hpp"
#include "scf/cuda/df_jk_kernels.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"

namespace vibeqc::scf {
using namespace cuda_df;

// Host/item/device adapters share the same J/K builders and term selection.
namespace {

vibeqc_status prepare_outputs(std::size_t elements, std::vector<double>& first,
                              std::vector<double>& second, std::string& detail,
                              JkTermSelection terms) {
  try {
    first.assign(terms.coulomb ? elements : 0, 0.0);
    second.assign(terms.exchange ? elements : 0, 0.0);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA DF J/K output failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  return VIBEQC_STATUS_SUCCESS;
}

bool validate_execution_input(const CudaDensityFittingJkPlan* plan,
                              const std::vector<double>& density, std::string& detail) {
  if (plan == nullptr) {
    detail = "CUDA DF J/K plan is null";
    return false;
  }
  const std::size_t expected = plan->batch_size * plan->matrix_elements;
  if (density.size() != expected || !finite_values(density)) {
    detail = "CUDA DF density dimensions or values are invalid";
    return false;
  }
  return true;
}

}  // namespace

DensityFactorIdentity cuda_density_fitting_factor_identity(
    const CudaDensityFittingJkPlan* plan, std::size_t system, std::uint64_t orbital_generation,
    std::uint64_t density_generation) noexcept {
  if (!plan || system >= plan->batch_size || !orbital_generation || !density_generation) return {};
  return {plan->factor_basis_identity, system + 1, orbital_generation, density_generation};
}

vibeqc_status execute_cuda_density_fitting_occupied_exchange(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& density, DensityFactorSpin spin,
    std::span<const CudaOccupiedDensityInput> factors, std::vector<double>& exchange,
    std::vector<std::uint8_t>& selected, std::string& detail) {
  detail.clear();
  if (!validate_execution_input(plan, density, detail)) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (!factors.empty() && factors.size() != plan->batch_size) {
    detail = "occupied DF factor list must cover the batch or be empty";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    exchange.assign(density.size(), 0);
    selected.assign(plan->batch_size, 0);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for occupied DF K outputs failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  auto error = cudaSetDevice(plan->device_id);
  if (error == cudaSuccess)
    error = cudaMemcpyAsync(plan->primary_density, density.data(), density.size() * sizeof(double),
                            cudaMemcpyHostToDevice, plan->stream);
  if (error != cudaSuccess)
    return cuda_failure(error, "upload occupied DF density witness", detail);
  for (std::size_t system = 0; system < plan->batch_size; ++system) {
    const auto input = factors.empty() ? CudaOccupiedDensityInput{} : factors[system];
    const auto expected = cuda_density_fitting_factor_identity(
        plan, system, input.expected.orbital_generation, input.expected.density_generation);
    const auto* factor = input.factor;
    const bool compatible =
        factor && factor->nbf() == plan->nbf && expected == input.expected &&
        factor->matches(
            expected, spin,
            std::span(density).subspan(system * plan->matrix_elements, plan->matrix_elements));
    vibeqc_status status;
    if (compatible) {
      // K owns the transpose staging after density upload. A bounded B fits
      // because rank<=nbf; no extra persistent/device factor allocation here.
      auto* staged = plan->exchange_density_column_major + system * plan->matrix_elements;
      if (factor->rank())
        error = cudaMemcpyAsync(staged, factor->values().data(), factor->values().size_bytes(),
                                cudaMemcpyHostToDevice, plan->stream);
      if (error != cudaSuccess) return cuda_failure(error, "upload occupied DF factor", detail);
      status = build_occupied_exchange(*plan, system, staged, factor->rank(), false, 1,
                                       plan->alpha_exchange, detail);
      selected[system] = status == VIBEQC_STATUS_SUCCESS;
    } else {
      status = build_exchange(*plan, plan->primary_density, plan->alpha_exchange, detail, false,
                              system, system + 1);
    }
    if (status != VIBEQC_STATUS_SUCCESS) return status;
  }
  error = cudaMemcpyAsync(exchange.data(), plan->alpha_exchange, exchange.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, plan->stream);
  if (error == cudaSuccess) error = cudaStreamSynchronize(plan->stream);
  return error == cudaSuccess ? VIBEQC_STATUS_SUCCESS
                              : cuda_failure(error, "finish occupied DF exchange", detail);
}

vibeqc_status execute_cuda_density_fitting_rhf_jk(CudaDensityFittingJkPlan* plan,
                                                  const std::vector<double>& density,
                                                  std::vector<double>& coulomb,
                                                  std::vector<double>& exchange,
                                                  std::string& detail, JkTermSelection terms) {
  detail.clear();
  if (!validate_execution_input(plan, density, detail)) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t elements = plan->batch_size * plan->matrix_elements;
  vibeqc_status status = prepare_outputs(elements, coulomb, exchange, detail, terms);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  if (!terms.coulomb && !terms.exchange) return VIBEQC_STATUS_SUCCESS;
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  const std::size_t bytes = elements * sizeof(double);
  cuda_error = cudaMemcpyAsync(plan->primary_density, density.data(), bytes, cudaMemcpyHostToDevice,
                               plan->stream);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "upload RHF CUDA DF density", detail);
  }
  status =
      terms.coulomb ? build_coulomb(*plan, plan->primary_density, detail) : VIBEQC_STATUS_SUCCESS;
  if (terms.exchange && status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, plan->primary_density, plan->alpha_exchange, detail);
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  if (terms.coulomb) {
    cuda_error =
        cudaMemcpyAsync(coulomb.data(), plan->coulomb, bytes, cudaMemcpyDeviceToHost, plan->stream);
  }
  if (terms.exchange && cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(exchange.data(), plan->alpha_exchange, bytes,
                                 cudaMemcpyDeviceToHost, plan->stream);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaStreamSynchronize(plan->stream);
  }
  return cuda_error == cudaSuccess ? VIBEQC_STATUS_SUCCESS
                                   : cuda_failure(cuda_error, "finish RHF CUDA DF J/K", detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_jk(CudaDensityFittingJkPlan* plan,
                                                  const std::vector<double>& alpha_density,
                                                  const std::vector<double>& beta_density,
                                                  std::vector<double>& coulomb,
                                                  std::vector<double>& alpha_exchange,
                                                  std::vector<double>& beta_exchange,
                                                  std::string& detail, JkTermSelection terms) {
  detail.clear();
  if (!validate_execution_input(plan, alpha_density, detail) ||
      !validate_execution_input(plan, beta_density, detail)) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t elements = plan->batch_size * plan->matrix_elements;
  vibeqc_status status = prepare_outputs(elements, coulomb, alpha_exchange, detail, terms);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  try {
    beta_exchange.assign(terms.exchange ? elements : 0, 0.0);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA DF beta exchange output failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  if (!terms.coulomb && !terms.exchange) return VIBEQC_STATUS_SUCCESS;
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  const std::size_t bytes = elements * sizeof(double);
  cuda_error = cudaMemcpyAsync(plan->primary_density, alpha_density.data(), bytes,
                               cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(plan->secondary_density, beta_density.data(), bytes,
                                 cudaMemcpyHostToDevice, plan->stream);
  }
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "upload UHF CUDA DF densities", detail);
  }
  if (terms.coulomb) {
    launch_sum_spin_density_kernel(blocks_for(elements), kThreads, 0, plan->stream, elements,
                                   plan->primary_density, plan->secondary_density,
                                   plan->total_density);
    cuda_error = cudaPeekAtLastError();
    if (cuda_error != cudaSuccess) {
      return cuda_failure(cuda_error, "sum UHF CUDA DF density", detail);
    }
  }
  status =
      terms.coulomb ? build_coulomb(*plan, plan->total_density, detail) : VIBEQC_STATUS_SUCCESS;
  if (terms.exchange && status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, plan->primary_density, plan->alpha_exchange, detail);
  }
  if (terms.exchange && status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, plan->secondary_density, plan->beta_exchange, detail);
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  if (terms.coulomb) {
    cuda_error =
        cudaMemcpyAsync(coulomb.data(), plan->coulomb, bytes, cudaMemcpyDeviceToHost, plan->stream);
  }
  if (terms.exchange && cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(alpha_exchange.data(), plan->alpha_exchange, bytes,
                                 cudaMemcpyDeviceToHost, plan->stream);
  }
  if (terms.exchange && cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(beta_exchange.data(), plan->beta_exchange, bytes,
                                 cudaMemcpyDeviceToHost, plan->stream);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaStreamSynchronize(plan->stream);
  }
  return cuda_error == cudaSuccess ? VIBEQC_STATUS_SUCCESS
                                   : cuda_failure(cuda_error, "finish UHF CUDA DF J/K", detail);
}

vibeqc_status execute_cuda_density_fitting_rhf_jk_item(CudaDensityFittingJkPlan* plan,
                                                       std::size_t system,
                                                       const std::vector<double>& density,
                                                       std::vector<double>& coulomb,
                                                       std::vector<double>& exchange,
                                                       std::string& detail, JkTermSelection terms) {
  detail.clear();
  coulomb.clear();
  exchange.clear();
  if (plan == nullptr || system >= plan->batch_size || density.size() != plan->matrix_elements ||
      !finite_values(density)) {
    detail = "CUDA DF RHF item density dimensions or values are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (!terms.coulomb && !terms.exchange) return VIBEQC_STATUS_SUCCESS;
  if (plan->batch_size == 1U) {
    return execute_cuda_density_fitting_rhf_jk(plan, density, coulomb, exchange, detail, terms);
  }
  const std::size_t batch_elements = plan->batch_size * plan->matrix_elements;
  const std::size_t item_bytes = plan->matrix_elements * sizeof(double);
  const std::size_t batch_bytes = batch_elements * sizeof(double);
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  cuda_error = cudaMemsetAsync(plan->primary_density, 0, batch_bytes, plan->stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(plan->primary_density + system * plan->matrix_elements,
                                 density.data(), item_bytes, cudaMemcpyHostToDevice, plan->stream);
  }
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "upload bounded RHF CUDA DF item", detail);
  }
  vibeqc_status status =
      terms.coulomb ? build_coulomb(*plan, plan->primary_density, detail) : VIBEQC_STATUS_SUCCESS;
  if (terms.exchange && status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, plan->primary_density, plan->alpha_exchange, detail);
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  try {
    coulomb.resize(terms.coulomb ? plan->matrix_elements : 0);
    exchange.resize(terms.exchange ? plan->matrix_elements : 0);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for bounded RHF CUDA DF item failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  const std::size_t offset_bytes = system * item_bytes;
  if (terms.coulomb) {
    cuda_error = cudaMemcpyAsync(
        coulomb.data(), reinterpret_cast<const unsigned char*>(plan->coulomb) + offset_bytes,
        item_bytes, cudaMemcpyDeviceToHost, plan->stream);
  }
  if (terms.exchange && cuda_error == cudaSuccess) {
    cuda_error =
        cudaMemcpyAsync(exchange.data(),
                        reinterpret_cast<const unsigned char*>(plan->alpha_exchange) + offset_bytes,
                        item_bytes, cudaMemcpyDeviceToHost, plan->stream);
  }
  if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(plan->stream);
  return cuda_error == cudaSuccess
             ? VIBEQC_STATUS_SUCCESS
             : cuda_failure(cuda_error, "finish bounded RHF CUDA DF item", detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_jk_item(
    CudaDensityFittingJkPlan* plan, std::size_t system, const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density, std::vector<double>& coulomb,
    std::vector<double>& alpha_exchange, std::vector<double>& beta_exchange, std::string& detail,
    JkTermSelection terms) {
  detail.clear();
  coulomb.clear();
  alpha_exchange.clear();
  beta_exchange.clear();
  if (plan == nullptr || system >= plan->batch_size ||
      alpha_density.size() != plan->matrix_elements ||
      beta_density.size() != plan->matrix_elements || !finite_values(alpha_density) ||
      !finite_values(beta_density)) {
    detail = "CUDA DF UHF item density dimensions or values are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (!terms.coulomb && !terms.exchange) return VIBEQC_STATUS_SUCCESS;
  if (plan->batch_size == 1U) {
    return execute_cuda_density_fitting_uhf_jk(plan, alpha_density, beta_density, coulomb,
                                               alpha_exchange, beta_exchange, detail, terms);
  }
  const std::size_t batch_elements = plan->batch_size * plan->matrix_elements;
  const std::size_t item_bytes = plan->matrix_elements * sizeof(double);
  const std::size_t batch_bytes = batch_elements * sizeof(double);
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  cuda_error = cudaMemsetAsync(plan->primary_density, 0, batch_bytes, plan->stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemsetAsync(plan->secondary_density, 0, batch_bytes, plan->stream);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error =
        cudaMemcpyAsync(plan->primary_density + system * plan->matrix_elements,
                        alpha_density.data(), item_bytes, cudaMemcpyHostToDevice, plan->stream);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error =
        cudaMemcpyAsync(plan->secondary_density + system * plan->matrix_elements,
                        beta_density.data(), item_bytes, cudaMemcpyHostToDevice, plan->stream);
  }
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "upload bounded UHF CUDA DF item", detail);
  }
  if (terms.coulomb) {
    launch_sum_spin_density_kernel(blocks_for(batch_elements), kThreads, 0, plan->stream,
                                   batch_elements, plan->primary_density, plan->secondary_density,
                                   plan->total_density);
    cuda_error = cudaPeekAtLastError();
    if (cuda_error != cudaSuccess) {
      return cuda_failure(cuda_error, "sum bounded UHF CUDA DF item density", detail);
    }
  }
  vibeqc_status status =
      terms.coulomb ? build_coulomb(*plan, plan->total_density, detail) : VIBEQC_STATUS_SUCCESS;
  if (terms.exchange && status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, plan->primary_density, plan->alpha_exchange, detail);
  }
  if (terms.exchange && status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, plan->secondary_density, plan->beta_exchange, detail);
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  try {
    coulomb.resize(terms.coulomb ? plan->matrix_elements : 0);
    alpha_exchange.resize(terms.exchange ? plan->matrix_elements : 0);
    beta_exchange.resize(terms.exchange ? plan->matrix_elements : 0);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for bounded UHF CUDA DF item failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  const std::size_t offset_bytes = system * item_bytes;
  if (terms.coulomb) {
    cuda_error = cudaMemcpyAsync(
        coulomb.data(), reinterpret_cast<const unsigned char*>(plan->coulomb) + offset_bytes,
        item_bytes, cudaMemcpyDeviceToHost, plan->stream);
  }
  if (terms.exchange && cuda_error == cudaSuccess) {
    cuda_error =
        cudaMemcpyAsync(alpha_exchange.data(),
                        reinterpret_cast<const unsigned char*>(plan->alpha_exchange) + offset_bytes,
                        item_bytes, cudaMemcpyDeviceToHost, plan->stream);
  }
  if (terms.exchange && cuda_error == cudaSuccess) {
    cuda_error =
        cudaMemcpyAsync(beta_exchange.data(),
                        reinterpret_cast<const unsigned char*>(plan->beta_exchange) + offset_bytes,
                        item_bytes, cudaMemcpyDeviceToHost, plan->stream);
  }
  if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(plan->stream);
  return cuda_error == cudaSuccess
             ? VIBEQC_STATUS_SUCCESS
             : cuda_failure(cuda_error, "finish bounded UHF CUDA DF item", detail);
}

vibeqc_status execute_cuda_density_fitting_rhf_jk_device(CudaDensityFittingJkPlan* plan,
                                                         const double* density, double* coulomb,
                                                         double* exchange, std::string& detail,
                                                         JkTermSelection terms,
                                                         FockMatrixLayout density_layout) {
  detail.clear();
  if ((density_layout != FockMatrixLayout::RowMajor &&
       density_layout != FockMatrixLayout::ColumnMajor) ||
      plan == nullptr || density == nullptr || (terms.coulomb && coulomb == nullptr) ||
      (terms.exchange && exchange == nullptr)) {
    detail = "CUDA DF device RHF J/K pointers are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (!terms.coulomb && !terms.exchange) return VIBEQC_STATUS_SUCCESS;
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  vibeqc_status status =
      terms.coulomb ? build_coulomb(*plan, density, detail) : VIBEQC_STATUS_SUCCESS;
  if (terms.exchange && status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, density, plan->alpha_exchange, detail,
                            density_layout == FockMatrixLayout::ColumnMajor);
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  const std::size_t bytes = plan->batch_size * plan->matrix_elements * sizeof(double);
  if (terms.coulomb && coulomb != plan->coulomb) {
    cuda_error =
        cudaMemcpyAsync(coulomb, plan->coulomb, bytes, cudaMemcpyDeviceToDevice, plan->stream);
  }
  if (terms.exchange && cuda_error == cudaSuccess && exchange != plan->alpha_exchange) {
    cuda_error = cudaMemcpyAsync(exchange, plan->alpha_exchange, bytes, cudaMemcpyDeviceToDevice,
                                 plan->stream);
  }
  return cuda_error == cudaSuccess
             ? VIBEQC_STATUS_SUCCESS
             : cuda_failure(cuda_error, "copy CUDA DF device J/K outputs", detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_jk_device(
    CudaDensityFittingJkPlan* plan, const double* alpha_density, const double* beta_density,
    double* coulomb, double* alpha_exchange, double* beta_exchange, std::string& detail,
    JkTermSelection terms, FockMatrixLayout density_layout) {
  detail.clear();
  if ((density_layout != FockMatrixLayout::RowMajor &&
       density_layout != FockMatrixLayout::ColumnMajor) ||
      plan == nullptr || alpha_density == nullptr || beta_density == nullptr ||
      (terms.coulomb && coulomb == nullptr) ||
      (terms.exchange && (alpha_exchange == nullptr || beta_exchange == nullptr))) {
    detail = "CUDA DF device UHF J/K pointers are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (!terms.coulomb && !terms.exchange) return VIBEQC_STATUS_SUCCESS;
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  if (terms.coulomb) {
    launch_sum_spin_density_kernel(blocks_for(plan->batch_size * plan->matrix_elements), kThreads,
                                   0, plan->stream, plan->batch_size * plan->matrix_elements,
                                   alpha_density, beta_density, plan->total_density);
    cuda_error = cudaPeekAtLastError();
    if (cuda_error != cudaSuccess) {
      return cuda_failure(cuda_error, "sum CUDA DF device UHF density", detail);
    }
  }
  vibeqc_status status =
      terms.coulomb ? build_coulomb(*plan, plan->total_density, detail) : VIBEQC_STATUS_SUCCESS;
  if (terms.exchange && status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, alpha_density, plan->alpha_exchange, detail,
                            density_layout == FockMatrixLayout::ColumnMajor);
  }
  if (terms.exchange && status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, beta_density, plan->beta_exchange, detail,
                            density_layout == FockMatrixLayout::ColumnMajor);
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  const std::size_t bytes = plan->batch_size * plan->matrix_elements * sizeof(double);
  if (terms.coulomb && coulomb != plan->coulomb) {
    cuda_error =
        cudaMemcpyAsync(coulomb, plan->coulomb, bytes, cudaMemcpyDeviceToDevice, plan->stream);
  }
  if (terms.exchange && cuda_error == cudaSuccess && alpha_exchange != plan->alpha_exchange) {
    cuda_error = cudaMemcpyAsync(alpha_exchange, plan->alpha_exchange, bytes,
                                 cudaMemcpyDeviceToDevice, plan->stream);
  }
  if (terms.exchange && cuda_error == cudaSuccess && beta_exchange != plan->beta_exchange) {
    cuda_error = cudaMemcpyAsync(beta_exchange, plan->beta_exchange, bytes,
                                 cudaMemcpyDeviceToDevice, plan->stream);
  }
  return cuda_error == cudaSuccess
             ? VIBEQC_STATUS_SUCCESS
             : cuda_failure(cuda_error, "copy CUDA DF device UHF J/K outputs", detail);
}

}  // namespace vibeqc::scf
