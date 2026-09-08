// Prepared TensorIR executor support. Scientific equations are emitted from
// TensorIR; this header owns CUDA resources, error boundaries and measurements.
#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace vibeqc_tensor {
using I = int64_t;

inline void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
inline void blas_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS status " + std::to_string(status));
}
inline void error_text(char* out, size_t size, const char* text) noexcept {
  if (out && size) std::snprintf(out, size, "%s", text);
}

// Switching is scoped to preparation/destruction. Execution rejects a caller
// device mismatch instead of running pointers against another CUDA context.
struct DeviceGuard {
  int previous = 0;
  explicit DeviceGuard(int device) {
    cuda_check(cudaGetDevice(&previous));
    cuda_check(cudaSetDevice(device));
  }
  ~DeviceGuard() { cudaSetDevice(previous); }
};

struct Metrics {
  uint64_t owned_device_bytes = 0;
  uint64_t provider_retained_bytes = 0;
  uint64_t prepare_device_delta = 0;
  uint64_t observed_device_delta = 0;
  double device_ms = 0;
  double input_ms = 0;
  double output_ms = 0;
  double packing_ms = 0;
  double library_ms = 0;
  double kernel_ms = 0;
};

struct Context {
  int device = 0;
  unsigned char* arena = nullptr;
  int* error = nullptr;
  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  cudaEvent_t begin = nullptr, end = nullptr, section_begin = nullptr, section_end = nullptr;
  Metrics metrics;
  size_t free_before_prepare = 0;
  std::mutex mutex;

  // All allocations and event/handle creation happen here. No run() path
  // allocates buffers or creates cuBLAS handles, even for partial tiles.
  void prepare(int ordinal, int major, int minor, size_t bytes, size_t error_offset,
               size_t library_offset, size_t library_bytes, size_t provider_bytes,
               bool needs_blas) {
    device = ordinal;
    DeviceGuard guard(device);
    cudaDeviceProp property{};
    cuda_check(cudaGetDeviceProperties(&property, device));
    if (property.major != major || property.minor != minor)
      throw std::runtime_error("tensor plan/device architecture mismatch");
    size_t before = 0, after = 0, total = 0;
    cuda_check(cudaMemGetInfo(&before, &total));
    free_before_prepare = before;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (needs_blas) {
      // A supplied workspace does not release cuBLAS's internal retained
      // allocations (64 MiB plus small buffers on the audited provider).
      // Charge a configurable allowance and check it before allocating tensor
      // storage. Preparation/destruction are serialized by the Python owner
      // so another owned handle cannot distort this conservative device delta.
      size_t provider_before = 0, provider_after = 0;
      cuda_check(cudaMemGetInfo(&provider_before, &total));
      blas_check(cublasCreate(&handle));
      cuda_check(cudaMemGetInfo(&provider_after, &total));
      metrics.provider_retained_bytes =
          provider_before > provider_after ? provider_before - provider_after : 0;
      if (metrics.provider_retained_bytes > provider_bytes)
        throw std::runtime_error(
            "cuBLAS retained allocations exceed the provider allowance: observed " +
            std::to_string(metrics.provider_retained_bytes) + " bytes, allowed " +
            std::to_string(provider_bytes));
    }
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&arena), bytes));
    error = reinterpret_cast<int*>(arena + error_offset);
    metrics.owned_device_bytes = bytes;
    if (needs_blas) {
      blas_check(cublasSetStream(handle, stream));
      blas_check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
      blas_check(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
      blas_check(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED));
      // SetStream resets the workspace; install our counted workspace
      // only after the final stream binding. A zero-byte workspace is a
      // valid conservative path; retained provider storage is budgeted above.
      blas_check(cublasSetWorkspace(handle, arena + library_offset, library_bytes));
    }
    cuda_check(cudaEventCreate(&begin));
    cuda_check(cudaEventCreate(&end));
    cuda_check(cudaEventCreate(&section_begin));
    cuda_check(cudaEventCreate(&section_end));
    cuda_check(cudaMemGetInfo(&after, &total));
    metrics.prepare_device_delta = before > after ? before - after : 0;
  }

  uint64_t device_delta() const {
    size_t available = 0, total = 0;
    cuda_check(cudaMemGetInfo(&available, &total));
    return free_before_prepare > available ? free_before_prepare - available : 0;
  }

  void check_device() const {
    int current = -1;
    cuda_check(cudaGetDevice(&current));
    if (current != device) throw std::runtime_error("tensor plan/current device mismatch");
  }
  template <class F>
  void section(bool profile, double& ms, F operation) {
    if (profile) cuda_check(cudaEventRecord(section_begin, stream));
    operation();
    if (profile) {
      cuda_check(cudaEventRecord(section_end, stream));
      cuda_check(cudaEventSynchronize(section_end));
      float elapsed = 0;
      cuda_check(cudaEventElapsedTime(&elapsed, section_begin, section_end));
      ms += elapsed;
    }
  }
  ~Context() {
    // Cleanup remains nonthrowing, including partial preparation failures.
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (stream) cudaStreamSynchronize(stream);
    if (handle) cublasDestroy(handle);
    if (begin) cudaEventDestroy(begin);
    if (end) cudaEventDestroy(end);
    if (section_begin) cudaEventDestroy(section_begin);
    if (section_end) cudaEventDestroy(section_end);
    if (arena) cudaFree(arena);
    if (stream) cudaStreamDestroy(stream);
    cudaSetDevice(previous);
  }
};

__device__ inline double finite(double value, int* error, int node) {
  if (!isfinite(value)) atomicCAS(error, 0, node + 1);
  return value;
}
__device__ inline double quotient(double a, double b, int* error, int node) {
  if (b == 0.0) {
    atomicCAS(error, 0, -(node + 1));
    return 0.0;
  }
  return finite(__ddiv_rn(a, b), error, node);
}
inline unsigned blocks(I count, int threads) {
  // A grid-stride loop bounds the grid, including on older CUDA devices.
  return static_cast<unsigned>(std::min<I>((count + threads - 1) / threads, 65535));
}
__global__ void check_scale(double* values, I count, double scale, int* error, int node) {
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += I(blockDim.x) * gridDim.x) {
    double value = finite(values[i], error, node);
    values[i] = finite(__dmul_rn(value, scale), error, node);
  }
}

// Row-major C = op(A) op(B) is column-major C^T = op(B)^T op(A)^T.
// Arguments expose every transpose, leading dimension, stride and beta.
inline void gemm(Context& context, char a_trans, char b_trans, int m, int n, int k, const double* a,
                 const double* b, double* c, I a_stride, I b_stride, I c_stride, int batches,
                 double beta) {
  const double alpha = 1.0;
  const auto ta = a_trans == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  const auto tb = b_trans == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int lda = a_trans == 'N' ? k : m;
  const int ldb = b_trans == 'N' ? n : k;
  if (batches == 1) {
    blas_check(cublasDgemm(context.handle, tb, ta, n, m, k, &alpha, b, ldb, a, lda, &beta, c, n));
  } else {
    blas_check(cublasDgemmStridedBatched(context.handle, tb, ta, n, m, k, &alpha, b, ldb, b_stride,
                                         a, lda, a_stride, &beta, c, n, c_stride, batches));
  }
}
}  // namespace vibeqc_tensor
