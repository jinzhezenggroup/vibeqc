#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "scf/cuda_density_fitting.hpp"

namespace vibeqc::scf {
namespace {

constexpr unsigned kThreads = 256;

bool checked_multiply(std::size_t first, std::size_t second,
                      std::size_t& product) {
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  product = first * second;
  return true;
}

bool checked_bytes(std::size_t elements, std::size_t& bytes) {
  return checked_multiply(elements, sizeof(double), bytes);
}

bool finite_values(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

unsigned blocks_for(std::size_t elements) {
  return static_cast<unsigned>((elements + kThreads - 1) / kThreads);
}

vibeqc_status cuda_failure(cudaError_t error, const char* operation,
                           std::string& detail) {
  detail = std::string(operation) + ": " + cudaGetErrorString(error);
  return error == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                            : VIBEQC_STATUS_CUDA_ERROR;
}

vibeqc_status blas_failure(cublasStatus_t status, const char* operation,
                           std::string& detail) {
  detail = std::string(operation) + " failed with cuBLAS status " +
           std::to_string(static_cast<int>(status));
  return status == CUBLAS_STATUS_ALLOC_FAILED ? VIBEQC_STATUS_OUT_OF_MEMORY
                                              : VIBEQC_STATUS_CUDA_ERROR;
}

vibeqc_status solver_failure(cusolverStatus_t status, const char* operation,
                             std::string& detail) {
  detail = std::string(operation) + " failed with cuSOLVER status " +
           std::to_string(static_cast<int>(status));
  return status == CUSOLVER_STATUS_ALLOC_FAILED ? VIBEQC_STATUS_OUT_OF_MEMORY
                                                : VIBEQC_STATUS_CUDA_ERROR;
}

__global__ void symmetrize_metrics_kernel(std::size_t dimension,
                                          double* metrics) {
  const std::size_t column =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const std::size_t system = blockIdx.z;
  if (row >= dimension || column >= dimension || row > column) return;
  const std::size_t offset = system * dimension * dimension;
  const std::size_t first = offset + row * dimension + column;
  const std::size_t second = offset + column * dimension + row;
  const double symmetric = 0.5 * (metrics[first] + metrics[second]);
  metrics[first] = symmetric;
  metrics[second] = symmetric;
}

__global__ void scale_eigenvectors_kernel(std::size_t matrix_elements,
                                          std::size_t dimension,
                                          const double* eigenvectors,
                                          const double* scales,
                                          double* scaled_eigenvectors) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= matrix_elements) return;
  const std::size_t local = element % (dimension * dimension);
  const std::size_t system = element / (dimension * dimension);
  const std::size_t column = local / dimension;
  scaled_eigenvectors[element] =
      eigenvectors[element] * scales[system * dimension + column];
}

__global__ void sum_spin_density_kernel(std::size_t elements,
                                        const double* alpha, const double* beta,
                                        double* total) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= elements) return;
  total[element] = alpha[element] + beta[element];
}

__global__ void gather_auxiliary_tile_kernel(
    std::size_t matrix_elements, std::size_t naux, std::size_t system,
    std::size_t auxiliary_begin, std::size_t auxiliary_count,
    const double* three_center, double* tile) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t tile_elements = auxiliary_count * matrix_elements;
  if (element >= tile_elements) return;
  const std::size_t auxiliary = element / matrix_elements;
  const std::size_t pair = element % matrix_elements;
  tile[element] = three_center[(system * matrix_elements + pair) * naux +
                               auxiliary_begin + auxiliary];
}

__global__ void reduce_exchange_tile_kernel(std::size_t matrix_elements,
                                            std::size_t auxiliary_count,
                                            std::size_t system,
                                            const double* contributions,
                                            double* exchange) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= matrix_elements) return;
  double value = 0.0;
  for (std::size_t auxiliary = 0; auxiliary < auxiliary_count; ++auxiliary) {
    value += contributions[auxiliary * matrix_elements + element];
  }
  exchange[system * matrix_elements + element] += value;
}

struct SetupBuffers {
  double* metrics{};
  double* eigenvalues{};
  double* scales{};
  double* scaled_eigenvectors{};
  double* inverse_square_roots{};
  double* raw_three_center{};
  double* solver_workspace{};
  int* solver_info{};

  ~SetupBuffers() {
    (void)cudaFree(metrics);
    (void)cudaFree(eigenvalues);
    (void)cudaFree(scales);
    (void)cudaFree(scaled_eigenvectors);
    (void)cudaFree(inverse_square_roots);
    (void)cudaFree(raw_three_center);
    (void)cudaFree(solver_workspace);
    (void)cudaFree(solver_info);
  }
};

}  // namespace

struct CudaDensityFittingJkPlan {
  int device_id{-1};
  std::size_t batch_size{};
  std::size_t nbf{};
  std::size_t naux{};
  std::size_t matrix_elements{};
  std::size_t tensor_elements_per_system{};
  std::size_t auxiliary_tile{};
  cudaStream_t stream{};
  cublasHandle_t blas{};
  cusolverDnHandle_t solver{};
  double* three_center{};
  double* primary_density{};
  double* secondary_density{};
  double* total_density{};
  double* auxiliary_density{};
  double* coulomb{};
  double* alpha_exchange{};
  double* beta_exchange{};
  double* auxiliary_tile_values{};
  double* exchange_intermediate{};
  double* exchange_contributions{};
};

namespace {

void release(CudaDensityFittingJkPlan& plan) noexcept {
  if (plan.device_id >= 0) (void)cudaSetDevice(plan.device_id);
  (void)cudaFree(plan.three_center);
  (void)cudaFree(plan.primary_density);
  (void)cudaFree(plan.secondary_density);
  (void)cudaFree(plan.total_density);
  (void)cudaFree(plan.auxiliary_density);
  (void)cudaFree(plan.coulomb);
  (void)cudaFree(plan.alpha_exchange);
  (void)cudaFree(plan.beta_exchange);
  (void)cudaFree(plan.auxiliary_tile_values);
  (void)cudaFree(plan.exchange_intermediate);
  (void)cudaFree(plan.exchange_contributions);
  if (plan.solver != nullptr) (void)cusolverDnDestroy(plan.solver);
  if (plan.blas != nullptr) (void)cublasDestroy(plan.blas);
  if (plan.stream != nullptr) (void)cudaStreamDestroy(plan.stream);
  plan = {};
}

vibeqc_status fail_plan(CudaDensityFittingJkPlan* plan, vibeqc_status status) {
  if (plan != nullptr) {
    release(*plan);
    delete plan;
  }
  return status;
}

vibeqc_status allocate_device(void** pointer, std::size_t bytes,
                              const char* description, std::string& detail) {
  const cudaError_t error = cudaMalloc(pointer, bytes);
  return error == cudaSuccess ? VIBEQC_STATUS_SUCCESS
                              : cuda_failure(error, description, detail);
}

vibeqc_status build_coulomb(CudaDensityFittingJkPlan& plan,
                            const double* density, std::string& detail) {
  const int batch_size = static_cast<int>(plan.batch_size);
  const int matrix_elements = static_cast<int>(plan.matrix_elements);
  const int naux = static_cast<int>(plan.naux);
  const long long tensor_stride =
      static_cast<long long>(plan.tensor_elements_per_system);
  const long long matrix_stride = static_cast<long long>(plan.matrix_elements);
  const long long auxiliary_stride = static_cast<long long>(plan.naux);
  const double one = 1.0;
  const double zero = 0.0;
  cublasStatus_t blas_status = cublasDgemmStridedBatched(
      plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, naux, 1, matrix_elements, &one,
      plan.three_center, naux, tensor_stride, density, matrix_elements,
      matrix_stride, &zero, plan.auxiliary_density, naux, auxiliary_stride,
      batch_size);
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return blas_failure(blas_status, "DF auxiliary-density contraction",
                        detail);
  }
  blas_status = cublasDgemmStridedBatched(
      plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, matrix_elements, 1, naux, &one,
      plan.three_center, naux, tensor_stride, plan.auxiliary_density, naux,
      auxiliary_stride, &zero, plan.coulomb, matrix_elements, matrix_stride,
      batch_size);
  return blas_status == CUBLAS_STATUS_SUCCESS
             ? VIBEQC_STATUS_SUCCESS
             : blas_failure(blas_status, "DF Coulomb contraction", detail);
}

vibeqc_status build_exchange(CudaDensityFittingJkPlan& plan,
                             const double* density, double* exchange,
                             std::string& detail) {
  const std::size_t output_elements = plan.batch_size * plan.matrix_elements;
  cudaError_t cuda_error = cudaMemsetAsync(
      exchange, 0, output_elements * sizeof(double), plan.stream);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "zero DF exchange output", detail);
  }

  const int nbf = static_cast<int>(plan.nbf);
  const long long matrix_stride = static_cast<long long>(plan.matrix_elements);
  const double one = 1.0;
  const double zero = 0.0;
  for (std::size_t system = 0; system < plan.batch_size; ++system) {
    const double* system_density = density + system * plan.matrix_elements;
    for (std::size_t auxiliary_begin = 0; auxiliary_begin < plan.naux;
         auxiliary_begin += plan.auxiliary_tile) {
      const std::size_t auxiliary_count =
          std::min(plan.auxiliary_tile, plan.naux - auxiliary_begin);
      const std::size_t tile_elements = auxiliary_count * plan.matrix_elements;
      gather_auxiliary_tile_kernel<<<blocks_for(tile_elements), kThreads, 0,
                                     plan.stream>>>(
          plan.matrix_elements, plan.naux, system, auxiliary_begin,
          auxiliary_count, plan.three_center, plan.auxiliary_tile_values);
      cuda_error = cudaPeekAtLastError();
      if (cuda_error != cudaSuccess) {
        return cuda_failure(cuda_error, "gather DF exchange tile", detail);
      }

      const int tile_count = static_cast<int>(auxiliary_count);
      cublasStatus_t blas_status = cublasDgemmStridedBatched(
          plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, nbf, nbf, nbf, &one,
          system_density, nbf, 0, plan.auxiliary_tile_values, nbf,
          matrix_stride, &zero, plan.exchange_intermediate, nbf, matrix_stride,
          tile_count);
      if (blas_status != CUBLAS_STATUS_SUCCESS) {
        return blas_failure(blas_status, "DF exchange first GEMM", detail);
      }
      blas_status = cublasDgemmStridedBatched(
          plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, nbf, nbf, nbf, &one,
          plan.auxiliary_tile_values, nbf, matrix_stride,
          plan.exchange_intermediate, nbf, matrix_stride, &zero,
          plan.exchange_contributions, nbf, matrix_stride, tile_count);
      if (blas_status != CUBLAS_STATUS_SUCCESS) {
        return blas_failure(blas_status, "DF exchange second GEMM", detail);
      }
      reduce_exchange_tile_kernel<<<blocks_for(plan.matrix_elements), kThreads,
                                    0, plan.stream>>>(
          plan.matrix_elements, auxiliary_count, system,
          plan.exchange_contributions, exchange);
      cuda_error = cudaPeekAtLastError();
      if (cuda_error != cudaSuccess) {
        return cuda_failure(cuda_error, "reduce DF exchange tile", detail);
      }
    }
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status prepare_outputs(std::size_t elements, std::vector<double>& first,
                              std::vector<double>& second,
                              std::string& detail) {
  try {
    first.assign(elements, 0.0);
    second.assign(elements, 0.0);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA DF J/K output failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  return VIBEQC_STATUS_SUCCESS;
}

bool validate_execution_input(const CudaDensityFittingJkPlan* plan,
                              const std::vector<double>& density,
                              std::string& detail) {
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

vibeqc_status create_cuda_density_fitting_jk_plan(
    int device_id, std::size_t batch_size, std::size_t nbf, std::size_t naux,
    const std::vector<double>& metrics, const std::vector<double>& three_center,
    double relative_threshold, std::size_t auxiliary_tile,
    CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail) {
  detail.clear();
  diagnostics.clear();
  if (plan == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  *plan = nullptr;
  if (batch_size == 0 || nbf == 0 || naux == 0 || !(relative_threshold > 0.0) ||
      !(relative_threshold < 1.0) ||
      batch_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      batch_size > 65535 ||
      nbf > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      naux > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    detail = "CUDA DF plan dimensions or metric threshold are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  std::size_t matrix_elements = 0;
  std::size_t metric_elements = 0;
  std::size_t tensor_elements_per_system = 0;
  std::size_t all_matrix_elements = 0;
  std::size_t all_metric_elements = 0;
  std::size_t all_tensor_elements = 0;
  if (!checked_multiply(nbf, nbf, matrix_elements) ||
      !checked_multiply(naux, naux, metric_elements) ||
      !checked_multiply(matrix_elements, naux, tensor_elements_per_system) ||
      !checked_multiply(batch_size, matrix_elements, all_matrix_elements) ||
      !checked_multiply(batch_size, metric_elements, all_metric_elements) ||
      !checked_multiply(batch_size, tensor_elements_per_system,
                        all_tensor_elements) ||
      matrix_elements >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      metrics.size() != all_metric_elements ||
      three_center.size() != all_tensor_elements || !finite_values(metrics) ||
      !finite_values(three_center)) {
    detail = "CUDA DF plan buffers have invalid dimensions or values";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  auxiliary_tile =
      auxiliary_tile == 0 ? std::min<std::size_t>(naux, 32) : auxiliary_tile;
  if (auxiliary_tile > naux ||
      auxiliary_tile >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    detail = "CUDA DF auxiliary tile is invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  std::size_t matrix_bytes = 0;
  std::size_t metric_bytes = 0;
  std::size_t tensor_bytes = 0;
  std::size_t auxiliary_bytes = 0;
  std::size_t tile_elements = 0;
  std::size_t tile_bytes = 0;
  if (!checked_bytes(all_matrix_elements, matrix_bytes) ||
      !checked_bytes(all_metric_elements, metric_bytes) ||
      !checked_bytes(all_tensor_elements, tensor_bytes) ||
      !checked_multiply(batch_size, naux, tile_elements) ||
      !checked_bytes(tile_elements, auxiliary_bytes) ||
      !checked_multiply(auxiliary_tile, matrix_elements, tile_elements) ||
      !checked_bytes(tile_elements, tile_bytes)) {
    detail = "CUDA DF plan storage overflows size_t";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  auto* candidate = new (std::nothrow) CudaDensityFittingJkPlan{};
  if (candidate == nullptr) return VIBEQC_STATUS_OUT_OF_MEMORY;
  candidate->device_id = device_id;
  candidate->batch_size = batch_size;
  candidate->nbf = nbf;
  candidate->naux = naux;
  candidate->matrix_elements = matrix_elements;
  candidate->tensor_elements_per_system = tensor_elements_per_system;
  candidate->auxiliary_tile = auxiliary_tile;

  cuda_error =
      cudaStreamCreateWithFlags(&candidate->stream, cudaStreamNonBlocking);
  if (cuda_error != cudaSuccess) {
    return fail_plan(candidate,
                     cuda_failure(cuda_error, "create CUDA DF stream", detail));
  }
  cublasStatus_t blas_status = cublasCreate(&candidate->blas);
  if (blas_status == CUBLAS_STATUS_SUCCESS) {
    blas_status = cublasSetStream(candidate->blas, candidate->stream);
  }
  if (blas_status == CUBLAS_STATUS_SUCCESS) {
    blas_status =
        cublasSetPointerMode(candidate->blas, CUBLAS_POINTER_MODE_HOST);
  }
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return fail_plan(
        candidate,
        blas_failure(blas_status, "initialize CUDA DF cuBLAS", detail));
  }
  cusolverStatus_t solver_status = cusolverDnCreate(&candidate->solver);
  if (solver_status == CUSOLVER_STATUS_SUCCESS) {
    solver_status = cusolverDnSetStream(candidate->solver, candidate->stream);
  }
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    return fail_plan(
        candidate,
        solver_failure(solver_status, "initialize CUDA DF cuSOLVER", detail));
  }

  auto allocate_permanent = [&](double** pointer, std::size_t bytes,
                                const char* description) {
    return allocate_device(reinterpret_cast<void**>(pointer), bytes,
                           description, detail);
  };
  vibeqc_status status =
      allocate_permanent(&candidate->three_center, tensor_bytes,
                         "allocate transformed CUDA DF tensor");
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->primary_density, matrix_bytes,
                                "allocate primary CUDA DF density");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->secondary_density, matrix_bytes,
                                "allocate secondary CUDA DF density");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->total_density, matrix_bytes,
                                "allocate total CUDA DF density");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->auxiliary_density, auxiliary_bytes,
                                "allocate CUDA DF auxiliary density");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->coulomb, matrix_bytes,
                                "allocate CUDA DF Coulomb matrices");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->alpha_exchange, matrix_bytes,
                                "allocate CUDA DF alpha exchange matrices");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->beta_exchange, matrix_bytes,
                                "allocate CUDA DF beta exchange matrices");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->auxiliary_tile_values, tile_bytes,
                                "allocate CUDA DF auxiliary tile");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->exchange_intermediate, tile_bytes,
                                "allocate CUDA DF exchange intermediate");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->exchange_contributions, tile_bytes,
                                "allocate CUDA DF exchange contributions");
  }
  if (status != VIBEQC_STATUS_SUCCESS) return fail_plan(candidate, status);

  SetupBuffers setup;
  auto allocate_setup = [&](void** pointer, std::size_t bytes,
                            const char* description) {
    return allocate_device(pointer, bytes, description, detail);
  };
  status = allocate_setup(reinterpret_cast<void**>(&setup.metrics),
                          metric_bytes, "allocate CUDA DF metric eigensystem");
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(reinterpret_cast<void**>(&setup.eigenvalues),
                            batch_size * naux * sizeof(double),
                            "allocate CUDA DF metric eigenvalues");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(reinterpret_cast<void**>(&setup.scales),
                            batch_size * naux * sizeof(double),
                            "allocate CUDA DF metric eigenvalue scales");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(
        reinterpret_cast<void**>(&setup.scaled_eigenvectors), metric_bytes,
        "allocate scaled CUDA DF metric eigenvectors");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(
        reinterpret_cast<void**>(&setup.inverse_square_roots), metric_bytes,
        "allocate CUDA DF metric inverse square roots");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(reinterpret_cast<void**>(&setup.raw_three_center),
                            tensor_bytes,
                            "allocate raw CUDA DF three-center tensor");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(reinterpret_cast<void**>(&setup.solver_info),
                            batch_size * sizeof(int),
                            "allocate CUDA DF solver status");
  }
  if (status != VIBEQC_STATUS_SUCCESS) return fail_plan(candidate, status);

  cuda_error = cudaMemcpyAsync(setup.metrics, metrics.data(), metric_bytes,
                               cudaMemcpyHostToDevice, candidate->stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(setup.raw_three_center, three_center.data(),
                                 tensor_bytes, cudaMemcpyHostToDevice,
                                 candidate->stream);
  }
  if (cuda_error != cudaSuccess) {
    return fail_plan(
        candidate,
        cuda_failure(cuda_error, "upload CUDA DF setup tensors", detail));
  }
  const dim3 symmetric_threads(16, 16, 1);
  const dim3 symmetric_blocks(
      static_cast<unsigned>((naux + symmetric_threads.x - 1) /
                            symmetric_threads.x),
      static_cast<unsigned>((naux + symmetric_threads.y - 1) /
                            symmetric_threads.y),
      static_cast<unsigned>(batch_size));
  symmetrize_metrics_kernel<<<symmetric_blocks, symmetric_threads, 0,
                              candidate->stream>>>(naux, setup.metrics);
  cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return fail_plan(
        candidate,
        cuda_failure(cuda_error, "symmetrize CUDA DF metrics", detail));
  }

  int solver_lwork = 0;
  solver_status = cusolverDnDsyevd_bufferSize(
      candidate->solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      static_cast<int>(naux), setup.metrics, static_cast<int>(naux),
      setup.eigenvalues, &solver_lwork);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    return fail_plan(candidate,
                     solver_failure(solver_status,
                                    "size CUDA DF metric eigensolver", detail));
  }
  if (solver_lwork <= 0) {
    detail = "CUDA DF metric eigensolver returned an invalid workspace size";
    return fail_plan(candidate, VIBEQC_STATUS_CUDA_ERROR);
  }
  status =
      allocate_setup(reinterpret_cast<void**>(&setup.solver_workspace),
                     static_cast<std::size_t>(solver_lwork) * sizeof(double),
                     "allocate CUDA DF metric solver workspace");
  if (status != VIBEQC_STATUS_SUCCESS) return fail_plan(candidate, status);
  for (std::size_t system = 0; system < batch_size; ++system) {
    solver_status = cusolverDnDsyevd(
        candidate->solver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        static_cast<int>(naux), setup.metrics + system * metric_elements,
        static_cast<int>(naux), setup.eigenvalues + system * naux,
        setup.solver_workspace, solver_lwork, setup.solver_info + system);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      return fail_plan(
          candidate,
          solver_failure(solver_status, "diagonalize CUDA DF metric", detail));
    }
  }

  std::vector<double> eigenvalues;
  std::vector<double> scales;
  std::vector<int> solver_info;
  try {
    eigenvalues.resize(batch_size * naux);
    scales.assign(batch_size * naux, 0.0);
    solver_info.resize(batch_size);
    diagnostics.resize(batch_size);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA DF metric diagnostics failed";
    return fail_plan(candidate, VIBEQC_STATUS_OUT_OF_MEMORY);
  }
  cuda_error = cudaMemcpyAsync(eigenvalues.data(), setup.eigenvalues,
                               eigenvalues.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, candidate->stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(solver_info.data(), setup.solver_info,
                                 solver_info.size() * sizeof(int),
                                 cudaMemcpyDeviceToHost, candidate->stream);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaStreamSynchronize(candidate->stream);
  }
  if (cuda_error != cudaSuccess) {
    return fail_plan(
        candidate,
        cuda_failure(cuda_error, "read CUDA DF metric eigensystem", detail));
  }
  for (std::size_t system = 0; system < batch_size; ++system) {
    if (solver_info[system] != 0) {
      detail = "CUDA DF metric eigensolver did not converge for system " +
               std::to_string(system);
      return fail_plan(candidate, VIBEQC_STATUS_CUDA_ERROR);
    }
    const std::size_t offset = system * naux;
    const double largest = eigenvalues[offset + naux - 1];
    if (!(largest > 0.0) || !std::isfinite(largest)) {
      detail = "CUDA DF metric has no finite positive eigenspace for system " +
               std::to_string(system);
      return fail_plan(candidate, VIBEQC_STATUS_INVALID_ARGUMENT);
    }
    auto& diagnostic = diagnostics[system];
    diagnostic.absolute_threshold = relative_threshold * largest;
    double smallest_retained = largest;
    for (std::size_t item = 0; item < naux; ++item) {
      const double value = eigenvalues[offset + item];
      if (!std::isfinite(value)) {
        detail = "CUDA DF metric eigensolver returned a non-finite eigenvalue";
        return fail_plan(candidate, VIBEQC_STATUS_CUDA_ERROR);
      }
      if (value <= diagnostic.absolute_threshold) continue;
      scales[offset + item] = 1.0 / std::sqrt(value);
      ++diagnostic.effective_rank;
      smallest_retained = std::min(smallest_retained, value);
    }
    if (diagnostic.effective_rank == 0) {
      detail = "CUDA DF metric threshold removed every auxiliary direction";
      return fail_plan(candidate, VIBEQC_STATUS_INVALID_ARGUMENT);
    }
    diagnostic.condition_number = largest / smallest_retained;
  }

  cuda_error = cudaMemcpyAsync(setup.scales, scales.data(),
                               scales.size() * sizeof(double),
                               cudaMemcpyHostToDevice, candidate->stream);
  if (cuda_error != cudaSuccess) {
    return fail_plan(
        candidate,
        cuda_failure(cuda_error, "upload CUDA DF metric scales", detail));
  }
  scale_eigenvectors_kernel<<<blocks_for(all_metric_elements), kThreads, 0,
                              candidate->stream>>>(all_metric_elements, naux,
                                                   setup.metrics, setup.scales,
                                                   setup.scaled_eigenvectors);
  cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return fail_plan(
        candidate,
        cuda_failure(cuda_error, "scale CUDA DF metric eigenvectors", detail));
  }

  const double one = 1.0;
  const double zero = 0.0;
  blas_status = cublasDgemmStridedBatched(
      candidate->blas, CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(naux),
      static_cast<int>(naux), static_cast<int>(naux), &one,
      setup.scaled_eigenvectors, static_cast<int>(naux),
      static_cast<long long>(metric_elements), setup.metrics,
      static_cast<int>(naux), static_cast<long long>(metric_elements), &zero,
      setup.inverse_square_roots, static_cast<int>(naux),
      static_cast<long long>(metric_elements), static_cast<int>(batch_size));
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return fail_plan(
        candidate,
        blas_failure(blas_status,
                     "construct CUDA DF metric inverse square root", detail));
  }
  blas_status = cublasDgemmStridedBatched(
      candidate->blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(naux),
      static_cast<int>(matrix_elements), static_cast<int>(naux), &one,
      setup.inverse_square_roots, static_cast<int>(naux),
      static_cast<long long>(metric_elements), setup.raw_three_center,
      static_cast<int>(naux),
      static_cast<long long>(tensor_elements_per_system), &zero,
      candidate->three_center, static_cast<int>(naux),
      static_cast<long long>(tensor_elements_per_system),
      static_cast<int>(batch_size));
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return fail_plan(
        candidate,
        blas_failure(blas_status, "transform CUDA DF three-center tensor",
                     detail));
  }
  cuda_error = cudaStreamSynchronize(candidate->stream);
  if (cuda_error != cudaSuccess) {
    return fail_plan(
        candidate,
        cuda_failure(cuda_error, "finish CUDA DF plan preparation", detail));
  }

  (void)cusolverDnDestroy(candidate->solver);
  candidate->solver = nullptr;
  *plan = candidate;
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status execute_cuda_density_fitting_rhf_jk(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& density,
    std::vector<double>& coulomb, std::vector<double>& exchange,
    std::string& detail) {
  detail.clear();
  if (!validate_execution_input(plan, density, detail)) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t elements = plan->batch_size * plan->matrix_elements;
  vibeqc_status status = prepare_outputs(elements, coulomb, exchange, detail);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  const std::size_t bytes = elements * sizeof(double);
  cuda_error = cudaMemcpyAsync(plan->primary_density, density.data(), bytes,
                               cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "upload RHF CUDA DF density", detail);
  }
  status = build_coulomb(*plan, plan->primary_density, detail);
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, plan->primary_density, plan->alpha_exchange,
                            detail);
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  cuda_error = cudaMemcpyAsync(coulomb.data(), plan->coulomb, bytes,
                               cudaMemcpyDeviceToHost, plan->stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(exchange.data(), plan->alpha_exchange, bytes,
                                 cudaMemcpyDeviceToHost, plan->stream);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaStreamSynchronize(plan->stream);
  }
  return cuda_error == cudaSuccess
             ? VIBEQC_STATUS_SUCCESS
             : cuda_failure(cuda_error, "finish RHF CUDA DF J/K", detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_jk(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density, std::vector<double>& coulomb,
    std::vector<double>& alpha_exchange, std::vector<double>& beta_exchange,
    std::string& detail) {
  detail.clear();
  if (!validate_execution_input(plan, alpha_density, detail) ||
      !validate_execution_input(plan, beta_density, detail)) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t elements = plan->batch_size * plan->matrix_elements;
  vibeqc_status status =
      prepare_outputs(elements, coulomb, alpha_exchange, detail);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  try {
    beta_exchange.assign(elements, 0.0);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA DF beta exchange output failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  const std::size_t bytes = elements * sizeof(double);
  cuda_error = cudaMemcpyAsync(plan->primary_density, alpha_density.data(),
                               bytes, cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(plan->secondary_density, beta_density.data(),
                                 bytes, cudaMemcpyHostToDevice, plan->stream);
  }
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "upload UHF CUDA DF densities", detail);
  }
  sum_spin_density_kernel<<<blocks_for(elements), kThreads, 0, plan->stream>>>(
      elements, plan->primary_density, plan->secondary_density,
      plan->total_density);
  cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "sum UHF CUDA DF density", detail);
  }
  status = build_coulomb(*plan, plan->total_density, detail);
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, plan->primary_density, plan->alpha_exchange,
                            detail);
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, plan->secondary_density, plan->beta_exchange,
                            detail);
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  cuda_error = cudaMemcpyAsync(coulomb.data(), plan->coulomb, bytes,
                               cudaMemcpyDeviceToHost, plan->stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(alpha_exchange.data(), plan->alpha_exchange,
                                 bytes, cudaMemcpyDeviceToHost, plan->stream);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(beta_exchange.data(), plan->beta_exchange,
                                 bytes, cudaMemcpyDeviceToHost, plan->stream);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaStreamSynchronize(plan->stream);
  }
  return cuda_error == cudaSuccess
             ? VIBEQC_STATUS_SUCCESS
             : cuda_failure(cuda_error, "finish UHF CUDA DF J/K", detail);
}

void destroy_cuda_density_fitting_jk_plan(
    CudaDensityFittingJkPlan* plan) noexcept {
  if (plan == nullptr) return;
  release(*plan);
  delete plan;
}

}  // namespace vibeqc::scf
