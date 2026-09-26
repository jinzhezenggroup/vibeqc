// Generic prepared XC runtime. Scientific kernels are generated before including
// this file; no functional formula or Python callback exists in this ABI.
#pragma once

#include <limits>

namespace {
struct XCPlan {
  Context context;
  size_t capacity = 0;
  double* input = nullptr;
  double* output = nullptr;
};
template <class F>
int xc_guarded(char* error, size_t size, F function) noexcept {
  try {
    function();
    return 0;
  } catch (const std::exception& exception) {
    error_text(error, size, exception.what());
    return 1;
  } catch (...) {
    error_text(error, size, "unknown XC runtime failure");
    return 1;
  }
}
}  // namespace
extern "C" {
const char* xc_identity_v1() { return XC_IDENTITY; }
int xc_create_v1(int ordinal, int major, int minor, size_t capacity, size_t budget, void** result,
                 char* error, size_t size) {
  return xc_guarded(error, size, [&] {
    if (!result) throw std::invalid_argument("null XC plan output");
    *result = nullptr;
    if (!capacity ||
        capacity > (std::numeric_limits<size_t>::max() - 256) / (8 * (XC_INPUTS + XC_OUTPUTS)))
      throw std::invalid_argument("invalid XC capacity");
    const size_t data = 8 * capacity * (XC_INPUTS + XC_OUTPUTS);
    const size_t bytes = data + 256;
    if (bytes > budget) throw std::invalid_argument("XC device budget exceeded");
    auto plan = std::make_unique<XCPlan>();
    plan->capacity = capacity;
    plan->context.prepare(ordinal, major, minor, bytes, data, 0, 0, 0, false);
    plan->input = reinterpret_cast<double*>(plan->context.arena);
    plan->output = plan->input + capacity * XC_INPUTS;
    *result = plan.release();
  });
}
void xc_destroy_v1(void* pointer) { delete static_cast<XCPlan*>(pointer); }
int xc_run_v1(void* pointer, const double* input, size_t npoint, double* output, char* error,
              size_t size) {
  return xc_guarded(error, size, [&] {
    if (!pointer) throw std::invalid_argument("null XC plan");
    auto& plan = *static_cast<XCPlan*>(pointer);
    auto& context = plan.context;
    std::lock_guard<std::mutex> lock(context.mutex);
    context.check_device();
    if (npoint > plan.capacity || (npoint && (!input || !output)))
      throw std::invalid_argument("invalid XC tile");
    if (!npoint) return;
    // Physical-domain checks are owned by the Python preparation interface.
    // This private ABI independently rejects nonfinite host data and output.
    for (size_t i = 0; i < npoint * XC_INPUTS; ++i)
      if (!std::isfinite(input[i])) throw std::invalid_argument("nonfinite XC input");
    context.section(true, context.metrics.input_ms, [&] {
      cuda_check(cudaMemcpyAsync(plan.input, input, npoint * XC_INPUTS * 8, cudaMemcpyHostToDevice,
                                 context.stream));
      cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    });
    context.section(true, context.metrics.kernel_ms, [&] {
      xc_launch(plan.input, plan.output, npoint, context.error, context.stream);
    });
    int failure = 0;
    context.section(true, context.metrics.output_ms, [&] {
      cuda_check(cudaMemcpyAsync(output, plan.output, npoint * XC_OUTPUTS * 8,
                                 cudaMemcpyDeviceToHost, context.stream));
      cuda_check(cudaMemcpyAsync(&failure, context.error, sizeof(int), cudaMemcpyDeviceToHost,
                                 context.stream));
    });
    if (failure) throw std::runtime_error("nonfinite XC output " + std::to_string(failure - 1));
  });
}
int xc_metrics_v1(void* pointer, Metrics* metrics, int* versions, char* error, size_t size) {
  return xc_guarded(error, size, [&] {
    if (!pointer || !metrics || !versions) throw std::invalid_argument("null XC metrics");
    auto& context = static_cast<XCPlan*>(pointer)->context;
    std::lock_guard<std::mutex> lock(context.mutex);
    context.check_device();
    *metrics = context.metrics;
    metrics->observed_device_delta = context.device_delta();
    cuda_check(cudaRuntimeGetVersion(versions));
    cuda_check(cudaDriverGetVersion(versions + 1));
  });
}
}
