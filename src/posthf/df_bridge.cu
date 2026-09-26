/** Bounded adapter to CG05's generated native DF value source.
 * The host metric and explicit D2H raw tiles make placement observable. The
 * source's own basis packing/setup accounting is preserved in diagnostics.
 */
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "posthf/raw_source.hpp"
#include "runtime/cuda_resources.cuh"
#include "scf/cuda_density_fitting.hpp"

namespace {
using namespace vibeqc::scf;
void check(cudaError_t s) {
  if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
struct DFSource {
  CudaDensityFittingIntegralSource* source = nullptr;
  vibeqc::runtime::OwnedCudaStream stream;
  vibeqc::runtime::OwnedCudaEvent begin, end;
  vibeqc::runtime::OwnedCudaBuffer<double> tile;
  double generation_ms = 0, transfer_ms = 0, endpoint_ms = 0;
  std::uint64_t d2h_bytes = 0, host_staged_tiles = 0;
  std::uint64_t generated_bytes_snapshot = 0, generated_tiles_snapshot = 0;
  bool device_handoff = false;
  int device = 0;
  size_t nbf = 0, naux = 0, capacity = 0;
  std::vector<double> metric;
  std::mutex mutex;
  ~DFSource() {
    int previous = 0;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    if (stream) cudaStreamSynchronize(stream.get());
    tile.reset();
    begin.reset();
    end.reset();
    destroy_cuda_density_fitting_integral_source(source);
    source = nullptr;
    stream.reset();
    cudaSetDevice(previous);
  }
};

struct DFJkPlan {
  CudaDensityFittingJkPlan* plan = nullptr;
  std::uint64_t generated_bytes_begin = 0, generated_tiles_begin = 0;
  std::uint64_t density_h2d_bytes = 0, result_d2h_bytes = 0, executions = 0;
  double endpoint_ms = 0;
  size_t nbf = 0, naux = 0;

  ~DFJkPlan() {
    if (plan) destroy_cuda_density_fitting_jk_plan(plan);
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
    p->stream.create(device);
    p->begin.create(device);
    p->end.create(device);
    p->tile.allocate(device, capacity, p->stream.get());
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
    if (!p.source || p.device_handoff)
      throw std::runtime_error(
          "generated DF source has been handed off to a device-resident consumer");
    size_t product = 1;
    for (unsigned i = 0; i < 3; ++i) {
      if (n[i] && product > SIZE_MAX / n[i]) throw std::overflow_error("DF tile overflow");
      product *= n[i];
    }
    if (product != elements || elements > p.capacity)
      throw std::invalid_argument("DF tile exceeds prepared capacity");
    if (!elements) return;
    std::string detail;
    const auto endpoint_begin = std::chrono::steady_clock::now();
    p.begin.record(p.stream.get());
    for (size_t i = 0; i < n[0]; ++i) {
      if (generate_cuda_density_fitting_raw_tile(
              p.source, 0, (b[0] + i) * p.nbf + b[1], n[1], b[2], n[2], -1, p.stream.get(),
              p.tile.get() + i * n[1] * n[2], detail) != VIBEQC_STATUS_SUCCESS)
        throw std::runtime_error(detail);
    }
    p.end.record(p.stream.get());
    p.end.synchronize();
    float milliseconds = p.end.elapsed_since(p.begin);
    p.generation_ms += milliseconds;
    p.begin.record(p.stream.get());
    check(cudaMemcpyAsync(out, p.tile.get(), elements * 8, cudaMemcpyDeviceToHost, p.stream.get()));
    p.end.record(p.stream.get());
    p.end.synchronize();
    milliseconds = p.end.elapsed_since(p.begin);
    p.transfer_ms += milliseconds;
    p.d2h_bytes += elements * sizeof(double);
    ++p.host_staged_tiles;
    p.endpoint_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - endpoint_begin)
            .count();
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

int vibeqc_posthf_df_metrics_v2(void* pointer, std::uint64_t* counters, size_t counter_count,
                                double* values, size_t value_count, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !counters || counter_count < 5 || !values || value_count < 3)
      throw std::invalid_argument("invalid DF source metrics");
    auto& p = *static_cast<DFSource*>(pointer);
    std::lock_guard<std::mutex> lock(p.mutex);
    auto generated = CudaDensityFittingSourceCounters{};
    if (p.source) {
      generated = cuda_density_fitting_integral_source_counters(p.source);
      p.generated_bytes_snapshot = generated.generated_value_bytes;
      p.generated_tiles_snapshot = generated.generated_value_tiles;
    }
    counters[0] = p.generated_bytes_snapshot;
    counters[1] = p.d2h_bytes;
    counters[2] = p.generated_tiles_snapshot;
    counters[3] = p.host_staged_tiles;
    counters[4] = p.device_handoff ? 1U : 0U;
    values[0] = p.generation_ms;
    values[1] = p.transfer_ms;
    values[2] = p.endpoint_ms;
  });
}

int vibeqc_posthf_df_rhf_jk_plan_create_v1(void* pointer, double threshold, void** out,
                                           double* diagnostics, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !out || !diagnostics || !(threshold > 0.0) || !(threshold < 1.0))
      throw std::invalid_argument("invalid device-resident DF J/K plan request");
    *out = nullptr;
    auto& p = *static_cast<DFSource*>(pointer);
    std::lock_guard<std::mutex> lock(p.mutex);
    if (!p.source || p.device_handoff)
      throw std::runtime_error("generated DF source is not available for device handoff");
    check(cudaSetDevice(p.device));
    if (p.stream) check(cudaStreamSynchronize(p.stream.get()));
    const auto before = cuda_density_fitting_integral_source_counters(p.source);
    p.generated_bytes_snapshot = before.generated_value_bytes;
    p.generated_tiles_snapshot = before.generated_value_tiles;
    auto candidate = std::make_unique<DFJkPlan>();
    candidate->generated_bytes_begin = before.generated_value_bytes;
    candidate->generated_tiles_begin = before.generated_value_tiles;
    std::vector<CudaDensityFittingMetricDiagnostic> plan_diagnostics;
    std::string detail;
    const auto status = create_cuda_density_fitting_jk_plan_from_source(
        p.device, &p.source, 1U, p.nbf, p.naux, p.metric, threshold, 0U, 0U, &candidate->plan,
        plan_diagnostics, detail);
    p.device_handoff = true;
    if (status != VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail.empty() ? "device-resident DF J/K preparation failed"
                                              : detail);
    candidate->nbf = p.nbf;
    candidate->naux = p.naux;
    if (plan_diagnostics.empty())
      throw std::runtime_error("device-resident DF J/K preparation returned no diagnostics");
    const auto& d = plan_diagnostics.front();
    diagnostics[0] = static_cast<double>(p.nbf);
    diagnostics[1] = static_cast<double>(p.naux);
    diagnostics[2] = static_cast<double>(d.device_resident_bytes);
    diagnostics[3] = static_cast<double>(d.peak_device_bytes);
    diagnostics[4] = static_cast<double>(d.host_resident_bytes);
    diagnostics[5] = threshold;
    // The compatibility staging buffer is no longer reachable after ownership
    // transfer. Release it so it cannot inflate the production path footprint.
    p.tile.reset();
    p.begin.reset();
    p.end.reset();
    p.stream.reset();
    *out = candidate.release();
  });
}

int vibeqc_posthf_df_rhf_jk_plan_execute_v1(void* pointer, const double* density, size_t elements,
                                            double* coulomb, double* exchange, char* error,
                                            size_t size) {
  return guarded(error, size, [&] {
    if (!pointer) throw std::invalid_argument("null device-resident DF J/K plan");
    auto& p = *static_cast<DFJkPlan*>(pointer);
    if (!p.plan || !density || !coulomb || !exchange || elements != p.nbf * p.nbf)
      throw std::invalid_argument("invalid device-resident DF J/K execution request");
    const auto begin = std::chrono::steady_clock::now();
    std::vector<double> density_vector(density, density + elements);
    std::vector<double> coulomb_vector, exchange_vector;
    std::string detail;
    const auto status = execute_cuda_density_fitting_rhf_jk(p.plan, density_vector, coulomb_vector,
                                                            exchange_vector, detail);
    if (status != VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error(detail.empty() ? "device-resident DF J/K execution failed" : detail);
    if (coulomb_vector.size() != elements || exchange_vector.size() != elements)
      throw std::runtime_error("device-resident DF J/K output size mismatch");
    std::copy(coulomb_vector.begin(), coulomb_vector.end(), coulomb);
    std::copy(exchange_vector.begin(), exchange_vector.end(), exchange);
    p.density_h2d_bytes += elements * sizeof(double);
    p.result_d2h_bytes += 2U * elements * sizeof(double);
    ++p.executions;
    p.endpoint_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
  });
}

int vibeqc_posthf_df_rhf_jk_plan_metrics_v1(void* pointer, std::uint64_t* counters,
                                            size_t counter_count, double* values,
                                            size_t value_count, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !counters || counter_count < 7 || !values || value_count < 1)
      throw std::invalid_argument("invalid device-resident DF J/K metrics");
    auto& p = *static_cast<DFJkPlan*>(pointer);
    const auto now = cuda_density_fitting_jk_plan_source_counters(p.plan);
    counters[0] = now.generated_value_bytes - p.generated_bytes_begin;
    counters[1] = now.generated_value_tiles - p.generated_tiles_begin;
    counters[2] = 0U;  // raw DF D2H
    counters[3] = 0U;  // raw/derived DF H2D
    counters[4] = p.density_h2d_bytes;
    counters[5] = p.result_d2h_bytes;
    counters[6] = p.executions;
    values[0] = p.endpoint_ms;
  });
}

void vibeqc_posthf_df_rhf_jk_plan_destroy_v1(void* pointer) {
  delete static_cast<DFJkPlan*>(pointer);
}
}
