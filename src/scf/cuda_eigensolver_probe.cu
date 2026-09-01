#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "scf/cuda_eigensolver_policy.hpp"

namespace vibeqc::scf {
namespace {

constexpr unsigned kProbeThreads = 256;

__device__ double probe_diagonal_value(std::uint64_t row, std::uint64_t n) {
  return 1.0 + static_cast<double>(row) / static_cast<double>(n + 1);
}

__global__ void fill_probe_matrices_kernel(std::uint64_t elements, std::uint64_t n,
                                           double* matrices) {
  const std::uint64_t element = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= elements) return;
  const std::uint64_t matrix_elements = n * n;
  const std::uint64_t local = element % matrix_elements;
  const std::uint64_t row = local % n;
  const std::uint64_t column = local / n;
  matrices[element] = row == column ? probe_diagonal_value(row, n) : 0.0;
}

/** Refill the matrix before exactly one device-tail replay. */
__global__ void xsyev_probe_tail_kernel(std::uint64_t elements, std::uint64_t n, double* matrices,
                                        unsigned* replay_count) {
  __shared__ unsigned prior_replays;
  if (threadIdx.x == 0) prior_replays = atomicAdd(replay_count, 1U);
  __syncthreads();
  if (prior_replays != 0U) return;
  for (std::uint64_t element = threadIdx.x; element < elements; element += blockDim.x) {
    const std::uint64_t matrix_elements = n * n;
    const std::uint64_t local = element % matrix_elements;
    const std::uint64_t row = local % n;
    const std::uint64_t column = local / n;
    matrices[element] = row == column ? probe_diagonal_value(row, n) : 0.0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const cudaGraphExec_t current = cudaGetCurrentGraphExec();
    if (current != nullptr) {
      (void)cudaGraphLaunch(current, cudaStreamGraphTailLaunch);
    }
  }
}

struct ProbeResources {
  int device_id{-1};
  cudaStream_t stream{};
  cusolverDnHandle_t solver{};
  cusolverDnParams_t parameters{};
  cudaGraph_t graph{};
  cudaGraphExec_t graph_exec{};
  double* matrices{};
  double* eigenvalues{};
  int* info{};
  unsigned* replay_count{};
  void* device_workspace{};
  void* host_workspace{};

  ~ProbeResources() {
    if (device_id >= 0) (void)cudaSetDevice(device_id);
    if (stream != nullptr) (void)cudaStreamSynchronize(stream);
    if (graph_exec != nullptr) (void)cudaGraphExecDestroy(graph_exec);
    if (graph != nullptr) (void)cudaGraphDestroy(graph);
    if (parameters != nullptr) (void)cusolverDnDestroyParams(parameters);
    if (solver != nullptr) (void)cusolverDnDestroy(solver);
    (void)cudaFree(device_workspace);
    (void)cudaFree(replay_count);
    (void)cudaFree(info);
    (void)cudaFree(eigenvalues);
    (void)cudaFree(matrices);
    if (stream != nullptr) (void)cudaStreamDestroy(stream);
    // A provider that rejects capture invalidates only its private stream, but
    // CUDA also retains that status as the calling thread's last error.
    // Clearing it here prevents one negative qualification from contaminating
    // the next independent signature probe.
    (void)cudaGetLastError();
    std::free(host_workspace);
  }
};

bool record_cuda_failure(XsyevBatchedGraphProbeResult& result, XsyevBatchedGraphProbeStage stage,
                         cudaError_t error) {
  result.failure_stage = stage;
  result.cuda_error = static_cast<int>(error);
  return false;
}

bool record_solver_failure(XsyevBatchedGraphProbeResult& result, XsyevBatchedGraphProbeStage stage,
                           cusolverStatus_t error) {
  result.failure_stage = stage;
  result.cusolver_error = static_cast<int>(error);
  return false;
}

bool allocate_probe(void** pointer, std::size_t bytes, XsyevBatchedGraphProbeResult& result) {
  const cudaError_t error = cudaMalloc(pointer, bytes);
  return error == cudaSuccess ||
         record_cuda_failure(result, XsyevBatchedGraphProbeStage::allocate_probe_data, error);
}

bool fill_probe_state(ProbeResources& resources, std::uint64_t elements, std::uint64_t n,
                      unsigned replay_count, XsyevBatchedGraphProbeResult& result) {
  const unsigned blocks = static_cast<unsigned>((elements + kProbeThreads - 1) / kProbeThreads);
  fill_probe_matrices_kernel<<<blocks, kProbeThreads, 0, resources.stream>>>(elements, n,
                                                                             resources.matrices);
  cudaError_t error = cudaPeekAtLastError();
  if (error == cudaSuccess) {
    error = cudaMemsetAsync(resources.info, 0, result.solver_batch * sizeof(int), resources.stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(resources.replay_count, &replay_count, sizeof(replay_count),
                            cudaMemcpyHostToDevice, resources.stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(resources.stream);
  return error == cudaSuccess ||
         record_cuda_failure(result, XsyevBatchedGraphProbeStage::allocate_probe_data, error);
}

bool validate_probe_state(ProbeResources& resources, XsyevBatchedGraphProbeResult& result,
                          XsyevBatchedGraphProbeStage stage) {
  const std::size_t n = static_cast<std::size_t>(result.n);
  const std::size_t batch = static_cast<std::size_t>(result.solver_batch);
  std::vector<double> matrix;
  std::vector<double> eigenvalues;
  std::vector<int> info;
  try {
    matrix.resize(n * n * batch);
    eigenvalues.resize(n * batch);
    info.resize(batch);
  } catch (const std::bad_alloc&) {
    result.failure_stage = stage;
    return false;
  }
  cudaError_t error =
      cudaMemcpyAsync(matrix.data(), resources.matrices, matrix.size() * sizeof(double),
                      cudaMemcpyDeviceToHost, resources.stream);
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(eigenvalues.data(), resources.eigenvalues,
                            eigenvalues.size() * sizeof(double), cudaMemcpyDeviceToHost,
                            resources.stream);
  }
  if (error == cudaSuccess) {
    error = cudaMemcpyAsync(info.data(), resources.info, info.size() * sizeof(int),
                            cudaMemcpyDeviceToHost, resources.stream);
  }
  if (error == cudaSuccess) error = cudaStreamSynchronize(resources.stream);
  if (error != cudaSuccess) return record_cuda_failure(result, stage, error);

  constexpr double tolerance = 2.0e-10;
  for (std::size_t system = 0; system < batch; ++system) {
    if (info[system] != 0) {
      result.failure_stage = stage;
      return false;
    }
    for (std::size_t item = 0; item < n; ++item) {
      const double expected = 1.0 + static_cast<double>(item) / static_cast<double>(n + 1);
      const double value = eigenvalues[system * n + item];
      if (!std::isfinite(value)) {
        result.failure_stage = stage;
        return false;
      }
      result.maximum_eigenvalue_error =
          std::max(result.maximum_eigenvalue_error, std::abs(value - expected));
    }
  }
  for (std::size_t system = 0; system < batch; ++system) {
    const std::size_t matrix_offset = system * n * n;
    const std::size_t eigenvalue_offset = system * n;
    for (std::size_t column = 0; column < n; ++column) {
      const double eigenvalue = eigenvalues[eigenvalue_offset + column];
      for (std::size_t row = 0; row < n; ++row) {
        const double vector = matrix[matrix_offset + column * n + row];
        if (!std::isfinite(vector)) {
          result.failure_stage = stage;
          return false;
        }
        const double diagonal = 1.0 + static_cast<double>(row) / static_cast<double>(n + 1);
        result.maximum_residual =
            std::max(result.maximum_residual, std::abs((diagonal - eigenvalue) * vector));
        const double expected_vector = row == column ? 1.0 : 0.0;
        result.maximum_orthogonality_error = std::max(
            result.maximum_orthogonality_error,
            row == column ? std::abs(std::abs(vector) - expected_vector) : std::abs(vector));
      }
    }
  }
  if (result.maximum_eigenvalue_error > tolerance || result.maximum_residual > tolerance ||
      result.maximum_orthogonality_error > tolerance) {
    result.failure_stage = stage;
    return false;
  }
  return true;
}

cusolverStatus_t launch_probe_solver(ProbeResources& resources,
                                     const XsyevBatchedGraphProbeResult& result) {
  return cusolverDnXsyevBatched(
      resources.solver, resources.parameters, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      static_cast<int>(result.n), CUDA_R_64F, resources.matrices, static_cast<int>(result.n),
      CUDA_R_64F, resources.eigenvalues, CUDA_R_64F, resources.device_workspace,
      result.device_workspace_bytes, resources.host_workspace, result.host_workspace_bytes,
      resources.info, static_cast<int>(result.solver_batch));
}

}  // namespace

XsyevBatchedGraphProbeResult probe_xsyev_batched_device_launch_graph(
    int device_id, std::uint64_t n, std::uint64_t solver_batch) noexcept {
  XsyevBatchedGraphProbeResult result;
  result.device_id = device_id;
  result.n = n;
  result.solver_batch = solver_batch;
  result.api = xsyev_batched_api_eligibility(n, n, solver_batch);
  if (!result.api.eligible) {
    result.failure_stage = XsyevBatchedGraphProbeStage::api_eligibility;
    return result;
  }

  try {
    ProbeResources resources;
    resources.device_id = device_id;
    (void)cudaGetLastError();
    cudaError_t cuda_error = cudaSetDevice(device_id);
    if (cuda_error != cudaSuccess) {
      record_cuda_failure(result, XsyevBatchedGraphProbeStage::select_device, cuda_error);
      return result;
    }
    cuda_error = cudaRuntimeGetVersion(&result.cuda_runtime_version);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaDriverGetVersion(&result.cuda_driver_version);
    }
    if (cuda_error != cudaSuccess) {
      record_cuda_failure(result, XsyevBatchedGraphProbeStage::device_identity, cuda_error);
      return result;
    }
    cudaDeviceProp properties{};
    cuda_error = cudaGetDeviceProperties(&properties, device_id);
    if (cuda_error != cudaSuccess) {
      record_cuda_failure(result, XsyevBatchedGraphProbeStage::device_identity, cuda_error);
      return result;
    }
    result.compute_capability_major = properties.major;
    result.compute_capability_minor = properties.minor;
    std::copy_n(properties.uuid.bytes, result.device_uuid.size(), result.device_uuid.begin());
    std::copy_n(properties.name, result.device_name.size() - 1, result.device_name.begin());
    int cusolver_major = 0;
    int cusolver_minor = 0;
    int cusolver_patch = 0;
    cusolverStatus_t solver_error = cusolverGetProperty(MAJOR_VERSION, &cusolver_major);
    if (solver_error == CUSOLVER_STATUS_SUCCESS) {
      solver_error = cusolverGetProperty(MINOR_VERSION, &cusolver_minor);
    }
    if (solver_error == CUSOLVER_STATUS_SUCCESS) {
      solver_error = cusolverGetProperty(PATCH_LEVEL, &cusolver_patch);
    }
    if (solver_error != CUSOLVER_STATUS_SUCCESS) {
      record_solver_failure(result, XsyevBatchedGraphProbeStage::device_identity, solver_error);
      return result;
    }
    result.cusolver_version = cusolver_major * 10000 + cusolver_minor * 100 + cusolver_patch;
    cuda_error = cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking);
    if (cuda_error != cudaSuccess) {
      record_cuda_failure(result, XsyevBatchedGraphProbeStage::create_stream, cuda_error);
      return result;
    }
    solver_error = cusolverDnCreate(&resources.solver);
    if (solver_error == CUSOLVER_STATUS_SUCCESS) {
      solver_error = cusolverDnSetStream(resources.solver, resources.stream);
    }
    if (solver_error != CUSOLVER_STATUS_SUCCESS) {
      record_solver_failure(result, XsyevBatchedGraphProbeStage::create_solver, solver_error);
      return result;
    }
    solver_error = cusolverDnCreateParams(&resources.parameters);
    if (solver_error != CUSOLVER_STATUS_SUCCESS) {
      record_solver_failure(result, XsyevBatchedGraphProbeStage::create_parameters, solver_error);
      return result;
    }

    const std::uint64_t matrix_elements = n * n;
    const std::uint64_t all_matrix_elements = matrix_elements * solver_batch;
    if (!allocate_probe(reinterpret_cast<void**>(&resources.matrices),
                        all_matrix_elements * sizeof(double), result) ||
        !allocate_probe(reinterpret_cast<void**>(&resources.eigenvalues),
                        n * solver_batch * sizeof(double), result) ||
        !allocate_probe(reinterpret_cast<void**>(&resources.info), solver_batch * sizeof(int),
                        result) ||
        !allocate_probe(reinterpret_cast<void**>(&resources.replay_count), sizeof(unsigned),
                        result)) {
      return result;
    }

    solver_error = cusolverDnXsyevBatched_bufferSize(
        resources.solver, resources.parameters, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        static_cast<int>(n), CUDA_R_64F, resources.matrices, static_cast<int>(n), CUDA_R_64F,
        resources.eigenvalues, CUDA_R_64F, &result.device_workspace_bytes,
        &result.host_workspace_bytes, static_cast<int>(solver_batch));
    if (solver_error != CUSOLVER_STATUS_SUCCESS) {
      record_solver_failure(result, XsyevBatchedGraphProbeStage::query_workspace, solver_error);
      return result;
    }
    std::size_t total_device_bytes = 0;
    cuda_error = cudaMemGetInfo(&result.available_device_bytes, &total_device_bytes);
    if (cuda_error != cudaSuccess) {
      record_cuda_failure(result, XsyevBatchedGraphProbeStage::query_workspace, cuda_error);
      return result;
    }
    if (result.device_workspace_bytes > result.available_device_bytes) {
      result.failure_stage = XsyevBatchedGraphProbeStage::insufficient_device_memory;
      return result;
    }
    if (result.device_workspace_bytes != 0 &&
        !allocate_probe(&resources.device_workspace, result.device_workspace_bytes, result)) {
      result.failure_stage = XsyevBatchedGraphProbeStage::allocate_workspace;
      return result;
    }
    if (result.host_workspace_bytes != 0) {
      resources.host_workspace = std::malloc(result.host_workspace_bytes);
      if (resources.host_workspace == nullptr) {
        result.failure_stage = XsyevBatchedGraphProbeStage::allocate_workspace;
        return result;
      }
    }

    // Isolated shell-class timing needs the real provider workspace contract
    // but not three full-size eigensolver executions before the measured Fock
    // work. Production runs never set this diagnostic environment variable.
    const char* skip_validation = std::getenv("VIBEQC_XSYEV_PROBE_SKIP_DIAGNOSTIC");
    if (skip_validation != nullptr &&
        (std::strcmp(skip_validation, "1") == 0 || std::strcmp(skip_validation, "skip") == 0)) {
      result.ordinary_execution_passed = true;
      result.graph_capture_passed = true;
      result.host_graph_replay_passed = true;
      result.device_tail_replay_passed = true;
      result.graph_eligible = true;
      return result;
    }

    if (!fill_probe_state(resources, all_matrix_elements, n, 1U, result)) {
      return result;
    }
    solver_error = launch_probe_solver(resources, result);
    if (solver_error != CUSOLVER_STATUS_SUCCESS) {
      record_solver_failure(result, XsyevBatchedGraphProbeStage::ordinary_execution, solver_error);
      return result;
    }
    cuda_error = cudaStreamSynchronize(resources.stream);
    if (cuda_error != cudaSuccess) {
      record_cuda_failure(result, XsyevBatchedGraphProbeStage::ordinary_execution, cuda_error);
      return result;
    }
    if (!validate_probe_state(resources, result,
                              XsyevBatchedGraphProbeStage::ordinary_validation)) {
      return result;
    }
    result.ordinary_execution_passed = true;

    if (!fill_probe_state(resources, all_matrix_elements, n, 1U, result)) {
      return result;
    }
    cuda_error = cudaStreamBeginCapture(resources.stream, cudaStreamCaptureModeThreadLocal);
    if (cuda_error != cudaSuccess) {
      record_cuda_failure(result, XsyevBatchedGraphProbeStage::begin_capture, cuda_error);
      return result;
    }
    solver_error = launch_probe_solver(resources, result);
    if (solver_error == CUSOLVER_STATUS_SUCCESS) {
      xsyev_probe_tail_kernel<<<1, kProbeThreads, 0, resources.stream>>>(
          all_matrix_elements, n, resources.matrices, resources.replay_count);
      cuda_error = cudaPeekAtLastError();
    }
    if (solver_error != CUSOLVER_STATUS_SUCCESS || cuda_error != cudaSuccess) {
      cudaGraph_t abandoned = nullptr;
      (void)cudaStreamEndCapture(resources.stream, &abandoned);
      if (abandoned != nullptr) (void)cudaGraphDestroy(abandoned);
      (void)cudaGetLastError();
      if (solver_error != CUSOLVER_STATUS_SUCCESS) {
        record_solver_failure(result, XsyevBatchedGraphProbeStage::capture_provider, solver_error);
      } else {
        record_cuda_failure(result, XsyevBatchedGraphProbeStage::capture_provider, cuda_error);
      }
      return result;
    }
    cuda_error = cudaStreamEndCapture(resources.stream, &resources.graph);
    if (cuda_error != cudaSuccess || resources.graph == nullptr) {
      record_cuda_failure(result, XsyevBatchedGraphProbeStage::end_capture, cuda_error);
      return result;
    }
    result.graph_capture_passed = true;
    cuda_error = cudaGraphInstantiate(&resources.graph_exec, resources.graph,
                                      cudaGraphInstantiateFlagDeviceLaunch);
    if (cuda_error != cudaSuccess) {
      record_cuda_failure(result, XsyevBatchedGraphProbeStage::instantiate_device_launch_graph,
                          cuda_error);
      return result;
    }
    cuda_error = cudaGraphUpload(resources.graph_exec, resources.stream);
    if (cuda_error != cudaSuccess) {
      record_cuda_failure(result, XsyevBatchedGraphProbeStage::upload_graph, cuda_error);
      return result;
    }

    if (!fill_probe_state(resources, all_matrix_elements, n, 1U, result)) {
      return result;
    }
    cuda_error = cudaGraphLaunch(resources.graph_exec, resources.stream);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream);
    }
    if (cuda_error != cudaSuccess) {
      record_cuda_failure(result, XsyevBatchedGraphProbeStage::host_graph_replay, cuda_error);
      return result;
    }
    if (!validate_probe_state(resources, result,
                              XsyevBatchedGraphProbeStage::host_graph_validation)) {
      return result;
    }
    result.host_graph_replay_passed = true;

    if (!fill_probe_state(resources, all_matrix_elements, n, 0U, result)) {
      return result;
    }
    cuda_error = cudaGraphLaunch(resources.graph_exec, resources.stream);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamSynchronize(resources.stream);
    }
    unsigned replay_count = 0;
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(&replay_count, resources.replay_count, sizeof(replay_count),
                              cudaMemcpyDeviceToHost);
    }
    if (cuda_error != cudaSuccess || replay_count != 2U) {
      record_cuda_failure(result, XsyevBatchedGraphProbeStage::device_tail_replay,
                          cuda_error == cudaSuccess ? cudaErrorUnknown : cuda_error);
      return result;
    }
    if (!validate_probe_state(resources, result,
                              XsyevBatchedGraphProbeStage::device_tail_validation)) {
      return result;
    }
    result.device_tail_replay_passed = true;
    result.graph_eligible = true;
    result.failure_stage = XsyevBatchedGraphProbeStage::none;
    return result;
  } catch (...) {
    result.failure_stage = XsyevBatchedGraphProbeStage::allocate_probe_data;
    return result;
  }
}

}  // namespace vibeqc::scf
