/** Bounded adapter to CG05's generated native DF value source.
 * The host metric and explicit D2H raw tiles make placement observable. The
 * source's own basis packing/setup accounting is preserved in diagnostics.
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "posthf/raw_source.hpp"
#include "scf/cuda_density_fitting.hpp"

namespace {
using namespace vibeqc::scf;
void check(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
struct DFSource {
  CudaDensityFittingIntegralSource* source = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t begin = nullptr, end = nullptr;
  double generation_ms = 0, transfer_ms = 0;
  double* tile = nullptr;
  int device = 0;
  size_t nbf = 0, naux = 0, capacity = 0;
  std::vector<double> metric;
  std::mutex mutex;
  ~DFSource() {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (stream) cudaStreamSynchronize(stream);
    if (tile) cudaFree(tile);
    if (begin) cudaEventDestroy(begin);
    if (end) cudaEventDestroy(end);
    destroy_cuda_density_fitting_integral_source(source);
    if (stream) cudaStreamDestroy(stream);
    cudaSetDevice(previous);
  }
};
template <class F>
int guarded(char* error, size_t size, F f) noexcept {
  try {
    f();
    return 0;
  } catch (const std::exception& e) {
    if (error && size) std::snprintf(error, size, "%s", e.what());
    return 1;
  }
}
}  // namespace
extern "C" {
int vibeqc_posthf_df_create_v1(void* raw, int device, size_t capacity, size_t budget, void** out,
                               size_t* diagnostic, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!out) throw std::invalid_argument("null DF output");
    *out = nullptr;
    if (!raw || !capacity || capacity > SIZE_MAX / 8 || !diagnostic)
      throw std::invalid_argument("invalid DF source");
    const auto& base = *static_cast<vibeqc::posthf::RawSource*>(raw);
    if (capacity * 8 > budget) throw std::invalid_argument("DF tile exceeds source budget");
    auto p = std::make_unique<DFSource>();
    p->device = device;
    p->capacity = capacity;
    check(cudaSetDevice(device));
    std::string detail;
    if (create_cuda_density_fitting_integral_source(device, {base.orbital()}, {base.auxiliary()},
                                                    &p->source, p->metric, p->nbf, p->naux,
                                                    detail) != VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail);
    const auto placement = cuda_density_fitting_integral_source_diagnostic(p->source);
    if (std::strcmp(placement.value_backend, "generated_rys") != 0)
      throw std::runtime_error("generated DF values required; source selected another backend");
    const size_t host =
        cuda_density_fitting_integral_source_host_peak_bytes(p->source) + p->metric.capacity() * 8;
    const size_t device_bytes =
        cuda_density_fitting_integral_source_device_bytes(p->source) + capacity * 8;
    // The existing source reports its setup capacity after preparation. Reject
    // and release it before exposing a usable provider if that scope exceeds
    // the separate source budget. Transformation budgets are planned earlier.
    if (host > budget || device_bytes > budget - host)
      throw std::runtime_error("DF source setup exceeds its separate budget");
    check(cudaStreamCreateWithFlags(&p->stream, cudaStreamNonBlocking));
    check(cudaEventCreate(&p->begin));
    check(cudaEventCreate(&p->end));
    check(cudaMalloc(reinterpret_cast<void**>(&p->tile), capacity * 8));
    diagnostic[0] = host;
    diagnostic[1] = device_bytes;
    diagnostic[2] = p->nbf;
    diagnostic[3] = p->naux;
    *out = p.release();
  });
}
void vibeqc_posthf_df_destroy_v1(void* p) { delete static_cast<DFSource*>(p); }
int vibeqc_posthf_df_read_v1(void* pointer, int kind, const size_t* b, const size_t* n, double* out,
                             size_t elements, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !b || !n || !out) throw std::invalid_argument("null DF tile");
    auto& p = *static_cast<DFSource*>(pointer);
    std::lock_guard<std::mutex> lock(p.mutex);
    check(cudaSetDevice(p.device));
    if (kind == 3) {
      if (b[0] > p.naux || n[0] > p.naux - b[0] || b[1] > p.naux || n[1] > p.naux - b[1] ||
          n[0] && n[1] > SIZE_MAX / n[0] || n[0] * n[1] != elements)
        throw std::invalid_argument("invalid DF metric tile");
      for (size_t i = 0; i < n[0]; ++i)
        std::copy_n(p.metric.data() + (b[0] + i) * p.naux + b[1], n[1], out + i * n[1]);
      return;
    }
    if (kind != 4 || b[0] > p.nbf || n[0] > p.nbf - b[0] || b[1] > p.nbf || n[1] > p.nbf - b[1] ||
        b[2] > p.naux || n[2] > p.naux - b[2])
      throw std::invalid_argument("invalid generated DF tile");
    size_t product = 1;
    for (unsigned i = 0; i < 3; ++i) {
      if (n[i] && product > SIZE_MAX / n[i]) throw std::overflow_error("DF tile overflow");
      product *= n[i];
    }
    if (product != elements || elements > p.capacity)
      throw std::invalid_argument("DF tile exceeds prepared capacity");
    if (!elements) return;
    std::string detail;
    check(cudaEventRecord(p.begin, p.stream));
    for (size_t i = 0; i < n[0]; ++i) {
      if (generate_cuda_density_fitting_raw_tile(p.source, 0, (b[0] + i) * p.nbf + b[1], n[1], b[2],
                                                 n[2], -1, p.stream, p.tile + i * n[1] * n[2],
                                                 detail) != VIBEQC_STATUS_SUCCESS)
        throw std::runtime_error(detail);
    }
    check(cudaEventRecord(p.end, p.stream));
    check(cudaEventSynchronize(p.end));
    float milliseconds = 0;
    check(cudaEventElapsedTime(&milliseconds, p.begin, p.end));
    p.generation_ms += milliseconds;
    check(cudaEventRecord(p.begin, p.stream));
    check(cudaMemcpyAsync(out, p.tile, elements * 8, cudaMemcpyDeviceToHost, p.stream));
    check(cudaEventRecord(p.end, p.stream));
    check(cudaEventSynchronize(p.end));
    check(cudaEventElapsedTime(&milliseconds, p.begin, p.end));
    p.transfer_ms += milliseconds;
  });
}
int vibeqc_posthf_df_metrics_v1(void* pointer, double* values, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !values) throw std::invalid_argument("null DF source metrics");
    auto& p = *static_cast<DFSource*>(pointer);
    std::lock_guard<std::mutex> lock(p.mutex);
    values[0] = p.generation_ms;
    values[1] = p.transfer_ms;
  });
}
}
