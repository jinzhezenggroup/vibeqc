// Intrusive, bounded input capture for #409; never an endpoint timing. This preload shim leaves
// cuBLAS inputs untouched and never provides endpoint timings. It binds each Gram input U to the
// occupied-factor C used by the immediately preceding projection.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
struct Projection {
  int auxiliary, rank, orbital, leading;
  cublasOperation_t operation;
  const double* coefficients;
  const double* factors;
};
std::mutex capture_mutex;
std::unordered_map<const double*, Projection> projections;
unsigned sequence = 0;
unsigned graph_capture_skips = 0;

template <class Function>
Function original(const char* name) {
  auto* symbol = dlsym(RTLD_NEXT, name);
  if (!symbol) throw std::runtime_error(std::string("missing original symbol: ") + name);
  return reinterpret_cast<Function>(symbol);
}
void cuda_check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void write(const std::filesystem::path& path, const std::vector<double>& data) {
  if (std::filesystem::exists(path)) throw std::runtime_error("refusing to overwrite capture");
  std::ofstream stream(path, std::ios::binary);
  stream.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
  if (!stream) throw std::runtime_error("capture write failed");
}
int requested_size() {
  const char* value = std::getenv("VIBEQC_PACKED_CAPTURE_N");
  return value ? std::atoi(value) : 0;
}
}  // namespace

extern "C" cublasStatus_t cublasDgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
    const double* alpha, const double* a, int lda, long long stride_a, const double* b, int ldb,
    long long stride_b, const double* beta, double* c, int ldc, long long stride_c, int batches) {
  try {
    static auto call = original<decltype(&cublasDgemmStridedBatched)>("cublasDgemmStridedBatched");
    auto status = call(handle, ta, tb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c,
                       ldc, stride_c, batches);
    if (status == CUBLAS_STATUS_SUCCESS && k == requested_size() && batches == k &&
        ta == CUBLAS_OP_N && stride_b == 0 && ldc == m &&
        stride_c == static_cast<long long>(m) * n &&
        ((tb == CUBLAS_OP_N && ldb == k) || (tb == CUBLAS_OP_T && ldb == n))) {
      std::lock_guard lock(capture_mutex);
      projections[c] = {m, n, k, ldb, tb, b, a};
    }
    return status;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Gram capture projection: %s\n", error.what());
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
}

extern "C" cublasStatus_t cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
                                         cublasOperation_t trans, int n, int k, const double* alpha,
                                         const double* a, int lda, const double* beta, double* c,
                                         int ldc) {
  try {
    static auto call = original<decltype(&cublasDsyrk_v2)>("cublasDsyrk_v2");
    std::lock_guard lock(capture_mutex);
    const char* directory = std::getenv("VIBEQC_PACKED_CAPTURE_DIR");
    const auto projection = projections.find(a);
    // Retain at most four eager occupied builds. Graph construction is skipped
    // below because a host snapshot/synchronization would invalidate capture.
    // The accompanying native trace identifies seed versus final/eager roles.
    const bool capture = directory && sequence < 4 && n == requested_size() &&
                         uplo == CUBLAS_FILL_MODE_LOWER && trans == CUBLAS_OP_T && lda == k &&
                         ldc == n && projection != projections.end();
    if (!capture) return call(handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    cublasPointerMode_t pointer_mode;
    cudaStream_t stream;
    if (cublasGetPointerMode(handle, &pointer_mode) != CUBLAS_STATUS_SUCCESS ||
        pointer_mode != CUBLAS_POINTER_MODE_HOST ||
        cublasGetStream(handle, &stream) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("capture requires the existing host-scalar stream contract");
    cudaStreamCaptureStatus capture_status;
    cuda_check(cudaStreamIsCapturing(stream, &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
      ++graph_capture_skips;
      return call(handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    }
    const auto p = projection->second;
    if (p.orbital != n || static_cast<long long>(p.auxiliary) * p.rank != k ||
        (*alpha != 1 && *alpha != 2) || *beta != 0)
      throw std::runtime_error("occupied Gram/projection identity mismatch");
    const auto elements = static_cast<std::size_t>(n) * k;
    if (elements > (2ULL << 30) / sizeof(double))
      throw std::runtime_error("bounded capture exceeds 2 GiB per U");
    std::vector<double> u(elements), physical(static_cast<std::size_t>(n) * p.rank);
    cuda_check(
        cudaMemcpyAsync(u.data(), a, u.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(physical.data(), p.coefficients, physical.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    const auto status = call(handle, uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    std::vector<double> triangular(static_cast<std::size_t>(n) * n);
    cuda_check(cudaMemcpyAsync(triangular.data(), c, triangular.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    // Only the written lower triangle is part of the reference. Mirror it on
    // the host; production performs its own unchanged device mirror afterward.
    for (int column = 0; column < n; ++column)
      for (int row = 0; row < column; ++row)
        triangular[row + static_cast<std::size_t>(column) * n] =
            triangular[column + static_cast<std::size_t>(row) * n];
    std::vector<double> coefficients(physical.size());
    for (int row = 0; row < n; ++row)
      for (int orbital = 0; orbital < p.rank; ++orbital)
        coefficients[static_cast<std::size_t>(row) * p.rank + orbital] =
            p.operation == CUBLAS_OP_N ? physical[row + static_cast<std::size_t>(orbital) * n]
                                       : physical[orbital + static_cast<std::size_t>(row) * p.rank];
    std::filesystem::create_directories(directory);
    auto prefix = std::filesystem::path(directory) / std::to_string(sequence++);
    // B is immutable over these eager builds. Stream one copy to disk in
    // bounded host chunks; a later reader verifies physical AO symmetry.
    if (sequence == 1) {
      const auto path = std::filesystem::path(directory) / "b.bin";
      if (std::filesystem::exists(path)) throw std::runtime_error("B capture exists");
      std::ofstream factors(path, std::ios::binary);
      const auto count = static_cast<std::size_t>(n) * n * p.auxiliary;
      if (count > (4ULL << 30) / sizeof(double))
        throw std::runtime_error("B capture exceeds 4 GiB");
      std::vector<double> chunk(1 << 20);
      for (std::size_t start = 0; start < count; start += chunk.size()) {
        const auto length = std::min(chunk.size(), count - start);
        cuda_check(cudaMemcpyAsync(chunk.data(), p.factors + start, length * sizeof(double),
                                   cudaMemcpyDeviceToHost, stream));
        cuda_check(cudaStreamSynchronize(stream));
        factors.write(reinterpret_cast<const char*>(chunk.data()), length * sizeof(double));
      }
      if (!factors) throw std::runtime_error("B capture write failed");
    }
    write(prefix.string() + "-u.bin", u);
    write(prefix.string() + "-c.bin", coefficients);
    write(prefix.string() + "-k.bin", triangular);
    std::ofstream metadata(prefix.string() + ".json");
    metadata << "{\"n\":" << n << ",\"length\":" << k << ",\"leading_dimension\":" << lda
             << ",\"auxiliary\":" << p.auxiliary << ",\"rank\":" << p.rank
             << ",\"weight\":" << *alpha
             << ",\"graph_capture_skips_before_snapshot\":" << graph_capture_skips
             << ",\"u_order\":\"column-major (length,n)\",\"c_order\":\"row-major (n,rank)\","
                "\"k_order\":\"column-major (n,n), mirrored\"}\n";
    if (!metadata) throw std::runtime_error("capture metadata write failed");
    return status;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Gram capture: %s\n", error.what());
    return CUBLAS_STATUS_INTERNAL_ERROR;
  }
}
