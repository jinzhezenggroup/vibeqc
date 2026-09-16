#include "scf/cuda/df_eigensystem.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <utility>

#include "runtime/cuda_component_trace.hpp"
#include "runtime/host_component_trace.hpp"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda/df_scf_library.hpp"
#include "scf/cuda/eigensolver.hpp"
#include "scf/cuda_density_fitting_eigen.hpp"
#include "scf/cuda_density_fitting_final_state.hpp"

namespace vibeqc::scf::cuda_df {
namespace {
/** One ordinary frame is serialized across the bucket's items/spins. It never
 * aliases a captured SCF buffer, occupied factor or another item's result. */
struct OrdinaryEigensystem {
  int device{-1};
  std::size_t n{}, workspace_bytes{}, host_workspace_bytes{}, device_bytes{};
  double* matrix{};
  double* x{};
  double* temporary{};
  double* values{};
  int* info{};
  std::uint8_t* active{};
  void* workspace{};
  std::vector<unsigned char> host_workspace;
  ~OrdinaryEigensystem() {
    if (device >= 0) (void)cudaSetDevice(device);
    for (void* pointer : {static_cast<void*>(matrix), static_cast<void*>(x),
                          static_cast<void*>(temporary), static_cast<void*>(values),
                          static_cast<void*>(info), static_cast<void*>(active), workspace})
      (void)runtime::resource_cuda_free(pointer);
  }
};

struct StreamDrain {
  cudaStream_t stream{};
  bool active{true};
  ~StreamDrain() {
    if (active) (void)cudaStreamSynchronize(stream);
  }
};

bool symmetric_finite(const std::vector<double>& matrix, std::size_t n) {
  if (matrix.size() != n * n || !finite_values(matrix)) return false;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = i + 1; j < n; ++j)
      if (std::abs(matrix[i * n + j] - matrix[j * n + i]) >
          1e-12 * std::max({1.0, std::abs(matrix[i * n + j]), std::abs(matrix[j * n + i])}))
        return false;
  return true;
}

std::vector<double> column_major(const std::vector<double>& matrix, std::size_t n) {
  std::vector<double> packed(matrix.size());
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t column = 0; column < n; ++column)
      packed[column * n + row] = matrix[row * n + column];
  return packed;
}

vibeqc_status prepare(CudaDensityFittingJkPlan& plan, OrdinaryEigensystem*& state,
                      std::string& detail) {
  state = static_cast<OrdinaryEigensystem*>(plan.ordinary_eigensystem);
  if (state) return VIBEQC_STATUS_SUCCESS;
  auto candidate = std::make_unique<OrdinaryEigensystem>();
  candidate->device = plan.device_id;
  candidate->n = plan.nbf;
  const auto n = plan.nbf;
  const auto matrix_bytes = n * n * sizeof(double);
  auto allocate = [&](void** target, std::size_t bytes, const char* name) {
    const auto status = allocate_device(target, bytes, name, detail);
    if (status == VIBEQC_STATUS_SUCCESS) candidate->device_bytes += bytes;
    return status;
  };
  for (double** target : {&candidate->matrix, &candidate->x, &candidate->temporary}) {
    const auto status = allocate(reinterpret_cast<void**>(target), matrix_bytes,
                                 "allocate ordinary CUDA DF AO frame");
    if (status != VIBEQC_STATUS_SUCCESS) return status;
  }
  auto status = allocate(reinterpret_cast<void**>(&candidate->values), n * sizeof(double),
                         "allocate ordinary CUDA DF eigenvalues");
  if (status == VIBEQC_STATUS_SUCCESS)
    status = allocate(reinterpret_cast<void**>(&candidate->info), sizeof(int),
                      "allocate ordinary CUDA DF solver info");
  if (status == VIBEQC_STATUS_SUCCESS)
    status = allocate(reinterpret_cast<void**>(&candidate->active), sizeof(std::uint8_t),
                      "allocate ordinary CUDA DF active mask");
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  const auto sized = cusolverDnXsyevd_bufferSize(
      plan.solver, plan.solver_parameters, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      static_cast<std::int64_t>(n), CUDA_R_64F, candidate->matrix, static_cast<std::int64_t>(n),
      CUDA_R_64F, candidate->values, CUDA_R_64F, &candidate->workspace_bytes,
      &candidate->host_workspace_bytes);
  if (sized != CUSOLVER_STATUS_SUCCESS)
    return solver_failure(sized, "size ordinary CUDA DF AO eigensolver", detail);
  // The shape planner reserves this allowance without querying a GPU. Check
  // both actual provider requests before either allocation; never borrow an
  // opaque-library allowance or silently exceed the admitted numeric capacity.
  const auto allowance = df_eigen_workspace_allowance(n);
  if (candidate->workspace_bytes > allowance || candidate->host_workspace_bytes > allowance) {
    detail = "ordinary CUDA DF eigensolver exceeds its admitted workspace allowance: device=" +
             std::to_string(candidate->workspace_bytes) +
             ", host=" + std::to_string(candidate->host_workspace_bytes) +
             ", allowance=" + std::to_string(allowance);
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  if (candidate->workspace_bytes) {
    status = allocate(&candidate->workspace, candidate->workspace_bytes,
                      "allocate ordinary CUDA DF solver workspace");
    if (status != VIBEQC_STATUS_SUCCESS) return status;
  }
  candidate->host_workspace.resize(candidate->host_workspace_bytes);
  state = candidate.release();
  plan.ordinary_eigensystem = state;
  return VIBEQC_STATUS_SUCCESS;
}
}  // namespace

void destroy_ordinary_eigensystem(void*& opaque) noexcept {
  delete static_cast<OrdinaryEigensystem*>(opaque);
  opaque = nullptr;
}
}  // namespace vibeqc::scf::cuda_df

namespace vibeqc::scf {
vibeqc_status solve_cuda_density_fitting_eigen(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& matrix,
    const std::vector<double>* overlap, const std::vector<double>* orthogonalizer,
    std::vector<double>& eigenvalues, std::vector<double>& coefficients,
    CudaDfEigenDiagnostic& diagnostic, std::string& detail, std::size_t system_index) {
  using namespace cuda_df;
  namespace trace = runtime::cuda_trace;
  namespace host = runtime::host_trace;
  diagnostic = {};
  detail.clear();
  if (df_eigen_outputs_alias(matrix, overlap, orthogonalizer, eigenvalues, coefficients)) {
    detail = "ordinary CUDA DF eigen inputs and outputs must not alias";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  eigenvalues.clear();
  coefficients.clear();
  if (!plan || system_index >= plan->batch_size || plan->nbf == 0 ||
      plan->nbf > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      (overlap == nullptr) != (orthogonalizer == nullptr)) {
    detail = "invalid ordinary CUDA DF eigensolver plan or generalized inputs";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const auto n = plan->nbf;
  if (!symmetric_finite(matrix, n) ||
      (overlap && (!symmetric_finite(*overlap, n) || !symmetric_finite(*orthogonalizer, n)))) {
    detail = "ordinary CUDA DF eigensolve requires finite symmetric AO inputs";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  auto error = cudaSetDevice(plan->device_id);
  if (error != cudaSuccess) return cuda_failure(error, "select ordinary DF eigen device", detail);
  // Host validation and synchronous publication are deliberately outside a
  // captured SCF graph. Reject capture before allocating or submitting work.
  cudaStreamCaptureStatus capture{};
  error = cudaStreamIsCapturing(plan->stream, &capture);
  if (error != cudaSuccess) return cuda_failure(error, "query ordinary DF eigen stream", detail);
  if (capture != cudaStreamCaptureStatusNone) {
    detail = "ordinary CUDA DF eigen operations require a noncapturing stream";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  trace::TraceOperation operation(
      "production_eigensolve", plan->stream,
      {1, n, plan->naux, plan->integral_source != nullptr, plan->streamed, system_index});
  host::Region endpoint("production_eigensystem", n);
  try {
    (void)df_eigen_device_reservation(n);  // Check all shape arithmetic before allocation.
    diagnostic.workspace_reused = plan->ordinary_eigensystem != nullptr;
    OrdinaryEigensystem* state{};
    auto status = trace::trace_call("workspace_setup", plan->stream,
                                    [&] { return prepare(*plan, state, detail); });
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    diagnostic.device_bytes = state->device_bytes;
    diagnostic.host_workspace_bytes = state->host_workspace_bytes;
    auto packed = column_major(matrix, n);
    auto packed_x = orthogonalizer ? column_major(*orthogonalizer, n) : std::vector<double>{};
    std::vector<double> values(n);
    // Declare the drain after every asynchronous host buffer so failure paths
    // synchronize before those buffers are destroyed, even without profiling.
    StreamDrain drain{plan->stream};
    const auto bytes = n * n * sizeof(double);
    {
      trace::TraceRegion upload("host_to_device", plan->stream);
      error = cudaMemcpyAsync(state->matrix, packed.data(), bytes, cudaMemcpyHostToDevice,
                              plan->stream);
      if (error == cudaSuccess && orthogonalizer)
        error =
            cudaMemcpyAsync(state->x, packed_x.data(), bytes, cudaMemcpyHostToDevice, plan->stream);
      if (error == cudaSuccess) error = cudaMemsetAsync(state->active, 1, 1, plan->stream);
      trace::trace_counter("host_to_device_bytes", bytes * (orthogonalizer ? 2 : 1));
    }
    if (error != cudaSuccess) return cuda_failure(error, "upload ordinary DF eigen inputs", detail);
    if (orthogonalizer) {
      trace::TraceRegion transform("generalized_transform", plan->stream);
      status = scf_gemm(*plan, false, 1, n, state->matrix, state->x, state->temporary, detail);
      if (status == VIBEQC_STATUS_SUCCESS)
        status = scf_gemm(*plan, true, 1, n, state->x, state->temporary, state->matrix, detail);
      if (status != VIBEQC_STATUS_SUCCESS) return status;
    }
    const cuda_execution::EigensolverResources resources{
        plan->stream,
        plan->solver,
        plan->solver_parameters,
        nullptr,
        state->workspace,
        state->workspace_bytes,
        state->host_workspace.empty() ? nullptr : state->host_workspace.data(),
        state->host_workspace_bytes};
    {
      host::Region actual_solve("device_eigensolve", n);
      trace::TraceRegion solve("device_eigensolve", plan->stream);
      runtime::df_progress::label("eigen_provider", "cusolverDnXsyevd");
      diagnostic.solver_calls = 1;
      trace::trace_counter("device_eigensolves", 1);
      status = cuda_execution::launch_solver(resources, CudaEigensolverFamily::xsyevd,
                                             static_cast<int>(n), 1, state->matrix, nullptr,
                                             state->values, 0, state->info, state->active);
    }
    if (status != VIBEQC_STATUS_SUCCESS) {
      detail = "ordinary CUDA DF Xsyevd submission failed";
      return status;
    }
    auto* output = state->matrix;
    if (orthogonalizer) {
      trace::TraceRegion transform("coefficient_backtransform", plan->stream);
      status = scf_gemm(*plan, false, 1, n, state->x, state->matrix, state->temporary, detail);
      if (status != VIBEQC_STATUS_SUCCESS) return status;
      output = state->temporary;
    }
    {
      trace::TraceRegion download("device_to_host", plan->stream);
      error = cudaMemcpyAsync(packed.data(), output, bytes, cudaMemcpyDeviceToHost, plan->stream);
      if (error == cudaSuccess)
        error = cudaMemcpyAsync(values.data(), state->values, n * sizeof(double),
                                cudaMemcpyDeviceToHost, plan->stream);
      if (error == cudaSuccess)
        error = cudaMemcpyAsync(&diagnostic.solver_info, state->info, sizeof(int),
                                cudaMemcpyDeviceToHost, plan->stream);
      if (error == cudaSuccess) error = cudaStreamSynchronize(plan->stream);
      trace::trace_counter("device_to_host_bytes", bytes + n * sizeof(double) + sizeof(int));
    }
    if (error != cudaSuccess) return cuda_failure(error, "download ordinary DF eigenframe", detail);
    drain.active = false;
    auto vectors = column_major(packed, n);  // Transpose the layout, not the mathematical frame.
    {
      host::Region validation("eigenframe_validation", n);
      if (!validate_cuda_density_fitting_eigen_frame(plan, matrix, overlap, values, vectors,
                                                     diagnostic, detail))
        return VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
    eigenvalues = std::move(values);
    coefficients = std::move(vectors);
    return VIBEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "ordinary CUDA DF eigensystem allocation failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::overflow_error& error) {
    detail = error.what();
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::runtime_error& error) {
    detail = error.what();
    return VIBEQC_STATUS_CUDA_ERROR;
  }
}
}  // namespace vibeqc::scf
