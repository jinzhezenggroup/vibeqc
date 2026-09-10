/** Generic retained Cholesky-prefix projection and dense matrix contractions.
 *
 * Raw integral columns and pivot/PSD decisions are supplied by the existing
 * host provider. No shell/operator formula is implemented here. Packing and
 * cuBLAS implement B[P] D B[P] and B[P] trace(B[P] D), reusing the common owned
 * CUDA context/arena/provider workspace. Execution allocates no device memory.
 */
#include <climits>
#include <cmath>

#include "../tensor/cuda_runtime.cuh"

namespace {
using namespace vibeqc_tensor;
constexpr size_t workspace_bytes = 4U << 20;
constexpr size_t provider_bytes = 96U << 20;

size_t add(size_t a, size_t b) {
  if (a > SIZE_MAX - b) throw std::overflow_error("low-rank allocation overflow");
  return a + b;
}
size_t mul(size_t a, size_t b) {
  if (b && a > SIZE_MAX / b) throw std::overflow_error("low-rank allocation overflow");
  return a * b;
}

struct Plan {
  Context context;
  size_t n{}, pairs{}, capacity{}, rank{};
  double *factors{}, *raw{}, *schur{}, *density{}, *total{}, *physical{}, *stage{}, *output{},
      *scalar{};
  bool healthy = true;

  void check(size_t expected_rank) {
    context.check_device();
    if (!healthy) throw std::runtime_error("low-rank CUDA state failed; close and rebuild");
    if (rank != expected_rank) throw std::invalid_argument("CPU/CUDA factor generation differs");
  }
};

template <class F>
int guarded(char* error, size_t size, F fn) noexcept {
  try {
    fn();
    return 0;
  } catch (const std::exception& ex) {
    error_text(error, size, ex.what());
    return 1;
  } catch (...) {
    error_text(error, size, "unknown low-rank CUDA failure");
    return 1;
  }
}

void finite_host(const double* data, size_t count) {
  if (!data && count) throw std::invalid_argument("null low-rank host input");
  for (size_t i = 0; i < count; ++i)
    if (!std::isfinite(data[i])) throw std::invalid_argument("nonfinite low-rank input");
}

void checked_kernel(Plan& p) {
  cuda_check(cudaGetLastError());
  int error = 0;
  cuda_check(cudaMemcpyAsync(&error, p.context.error, sizeof(error), cudaMemcpyDeviceToHost,
                             p.context.stream));
  cuda_check(cudaStreamSynchronize(p.context.stream));
  if (error) throw std::runtime_error("nonfinite low-rank CUDA arithmetic");
}

__global__ void project(const double* factors, const double* raw, double* out, size_t pairs,
                        size_t rank, size_t pivot, int* error) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < pairs; i += blockDim.x * gridDim.x) {
    double value = raw[i];
    // A deterministic rank-order subtraction preserves the actual retained
    // prefix. The old factors are never recalculated from a dense matrix.
    for (size_t k = 0; k < rank; ++k)
      value = __dsub_rn(value, __dmul_rn(factors[k * pairs + i], factors[k * pairs + pivot]));
    out[i] = finite(value, error, 0);
  }
}

__global__ void expand_pair(const double* factors, double* physical, size_t n, size_t pairs,
                            size_t rank) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n * n; i += blockDim.x * gridDim.x) {
    const size_t row = i / n, column = i % n;
    const size_t hi = row > column ? row : column, lo = row > column ? column : row;
    const size_t pair = hi * (hi + 1) / 2 + lo;
    physical[i] = factors[rank * pairs + pair] / (row == column ? 1.0 : 1.4142135623730951);
  }
}

__global__ void total_density(const double* density, double* total, size_t elements,
                              unsigned spins) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x)
    total[i] = density[i] + (spins == 2 ? density[elements + i] : 0.0);
}

__global__ void add_coulomb(const double* physical, const double* scalar, double* output,
                            size_t count, int* error) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x)
    output[i] = finite(output[i] + physical[i] * scalar[0], error, 1);
}

__global__ void symmetric_output(double* output, size_t n, unsigned matrices, int* error) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < matrices * n * n;
       i += blockDim.x * gridDim.x) {
    const size_t block = i / (n * n), row = (i / n) % n, column = i % n;
    if (row <= column) {
      const size_t other = block * n * n + column * n + row;
      const double value = finite(0.5 * output[i] + 0.5 * output[other], error, 2);
      output[i] = output[other] = value;
    }
  }
}
}  // namespace

extern "C" {
int posthf_cholesky_create_v1(int device, int major, int minor, size_t n, size_t capacity,
                              size_t expected_bytes, void** out, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!out) throw std::invalid_argument("null low-rank output handle");
    *out = nullptr;
    const size_t square = mul(n, n), pairs = mul(n, add(n, 1)) / 2;
    if (!n || square > INT_MAX || capacity > pairs)
      throw std::invalid_argument("invalid low-rank CUDA dimensions");
    const size_t numeric =
        mul(8, add(add(mul(capacity, pairs), mul(2, pairs)), add(mul(8, square), 1)));
    const size_t error_offset = mul(add(numeric, 255) / 256, 256);
    const size_t library_offset = add(error_offset, 256);
    const size_t bytes = add(library_offset, workspace_bytes);
    if (bytes != expected_bytes)
      throw std::invalid_argument("low-rank native/Python budget differs");
    auto p = std::make_unique<Plan>();
    p->n = n;
    p->pairs = pairs;
    p->capacity = capacity;
    p->context.prepare(device, major, minor, bytes, error_offset, library_offset, workspace_bytes,
                       provider_bytes, true);
    double* base = reinterpret_cast<double*>(p->context.arena);
    p->factors = base;
    p->raw = p->factors + capacity * pairs;
    p->schur = p->raw + pairs;
    p->density = p->schur + pairs;
    p->total = p->density + 2 * square;
    p->physical = p->total + square;
    p->stage = p->physical + square;
    p->output = p->stage + square;
    p->scalar = p->output + 3 * square;
    *out = p.release();
  });
}

void posthf_cholesky_destroy_v1(void* handle) { delete static_cast<Plan*>(handle); }

int posthf_cholesky_project_v1(void* handle, size_t rank, size_t pivot, const double* raw,
                               double* output, size_t count, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!handle || !output) throw std::invalid_argument("null low-rank project request");
    auto& p = *static_cast<Plan*>(handle);
    std::lock_guard<std::mutex> lock(p.context.mutex);
    p.check(rank);
    if (count != p.pairs || pivot >= p.pairs || rank >= p.capacity)
      throw std::invalid_argument("invalid low-rank projection dimensions");
    finite_host(raw, count);
    cuda_check(cudaMemsetAsync(p.context.error, 0, sizeof(int), p.context.stream));
    p.context.section(true, p.context.metrics.input_ms, [&] {
      cuda_check(cudaMemcpyAsync(p.raw, raw, count * 8, cudaMemcpyHostToDevice, p.context.stream));
    });
    p.context.section(true, p.context.metrics.kernel_ms, [&] {
      project<<<blocks(count, 128), 128, 0, p.context.stream>>>(p.factors, p.raw, p.schur, count,
                                                                rank, pivot, p.context.error);
    });
    checked_kernel(p);
    p.context.section(true, p.context.metrics.output_ms, [&] {
      cuda_check(
          cudaMemcpyAsync(output, p.schur, count * 8, cudaMemcpyDeviceToHost, p.context.stream));
    });
  });
}

int posthf_cholesky_commit_v1(void* handle, size_t rank, const double* column, size_t count,
                              char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!handle) throw std::invalid_argument("null low-rank commit request");
    auto& p = *static_cast<Plan*>(handle);
    std::lock_guard<std::mutex> lock(p.context.mutex);
    p.check(rank);
    if (count != p.pairs || rank >= p.capacity)
      throw std::invalid_argument("invalid low-rank commit dimensions");
    finite_host(column, count);
    // A device transport failure makes this mirror unusable; no later call
    // may assume that a partially transferred factor column was committed.
    p.healthy = false;
    p.context.section(true, p.context.metrics.input_ms, [&] {
      cuda_check(cudaMemcpyAsync(p.factors + rank * count, column, count * 8,
                                 cudaMemcpyHostToDevice, p.context.stream));
    });
    ++p.rank;
    p.healthy = true;
  });
}

int posthf_cholesky_jk_v1(void* handle, size_t rank, const double* density, unsigned spins,
                          double* output, size_t count, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!handle || !output) throw std::invalid_argument("null low-rank J/K request");
    auto& p = *static_cast<Plan*>(handle);
    std::lock_guard<std::mutex> lock(p.context.mutex);
    p.check(rank);
    const size_t square = p.n * p.n;
    if ((spins != 1 && spins != 2) || count != (spins + 1) * square)
      throw std::invalid_argument("invalid low-rank J/K dimensions");
    finite_host(density, spins * square);
    for (size_t spin = 0; spin < spins; ++spin)
      for (size_t row = 0; row < p.n; ++row)
        for (size_t column = 0; column < row; ++column)
          if (density[spin * square + row * p.n + column] !=
              density[spin * square + column * p.n + row])
            throw std::invalid_argument("low-rank CUDA density must be symmetric");
    cuda_check(cudaMemsetAsync(p.context.error, 0, sizeof(int), p.context.stream));
    cuda_check(cudaMemsetAsync(p.output, 0, count * 8, p.context.stream));
    p.context.section(true, p.context.metrics.input_ms, [&] {
      cuda_check(cudaMemcpyAsync(p.density, density, spins * square * 8, cudaMemcpyHostToDevice,
                                 p.context.stream));
    });
    p.context.section(true, p.context.metrics.kernel_ms, [&] {
      total_density<<<blocks(square, 128), 128, 0, p.context.stream>>>(p.density, p.total, square,
                                                                       spins);
    });
    const double one = 1.0, zero = 0.0;
    for (size_t k = 0; k < rank; ++k) {
      p.context.section(true, p.context.metrics.packing_ms, [&] {
        expand_pair<<<blocks(square, 128), 128, 0, p.context.stream>>>(p.factors, p.physical, p.n,
                                                                       p.pairs, k);
      });
      p.context.section(true, p.context.metrics.library_ms, [&] {
        blas_check(cublasSetPointerMode(p.context.handle, CUBLAS_POINTER_MODE_DEVICE));
        blas_check(cublasDdot(p.context.handle, static_cast<int>(square), p.physical, 1, p.total, 1,
                              p.scalar));
        blas_check(cublasSetPointerMode(p.context.handle, CUBLAS_POINTER_MODE_HOST));
        for (unsigned spin = 0; spin < spins; ++spin) {
          // Symmetric host matrices also represent the same column-major
          // matrices. The intermediate B*D is column-major, then (B*D)*B.
          const int n = static_cast<int>(p.n);
          blas_check(cublasDgemm(p.context.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one,
                                 p.physical, n, p.density + spin * square, n, &zero, p.stage, n));
          blas_check(cublasDgemm(p.context.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, p.stage,
                                 n, p.physical, n, &one, p.output + (spin + 1) * square, n));
        }
      });
      p.context.section(true, p.context.metrics.kernel_ms, [&] {
        add_coulomb<<<blocks(square, 128), 128, 0, p.context.stream>>>(
            p.physical, p.scalar, p.output, square, p.context.error);
      });
    }
    symmetric_output<<<blocks(count, 128), 128, 0, p.context.stream>>>(p.output, p.n, spins + 1,
                                                                       p.context.error);
    checked_kernel(p);
    p.context.section(true, p.context.metrics.output_ms, [&] {
      cuda_check(
          cudaMemcpyAsync(output, p.output, count * 8, cudaMemcpyDeviceToHost, p.context.stream));
    });
  });
}

int posthf_cholesky_metrics_v1(void* handle, Metrics* out, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!handle || !out) throw std::invalid_argument("null low-rank metrics request");
    auto& p = *static_cast<Plan*>(handle);
    std::lock_guard<std::mutex> lock(p.context.mutex);
    p.check(p.rank);
    *out = p.context.metrics;
    out->observed_device_delta = p.context.device_delta();
  });
}
}  // extern "C"
