/** Resident conventional MO block with four cuBLAS AO-axis transformations.
 * The AO source is explicitly host staged; no tensor arithmetic occurs there.
 * Reuses CG09's private stream, cuBLAS provider guard and owned arena lifecycle.
 */
#include <array>
#include <climits>

#include "../tensor/cuda_runtime.cuh"

namespace {
using namespace vibeqc_tensor;
struct Transform {
  Context context;
  size_t nbf{}, stage{}, output{}, coefficients{};
  std::array<size_t, 4> m{}, tile{}, c_offset{};
  double *c{}, *first{}, *second{}, *result{};
};
size_t mul(size_t a, size_t b) {
  if (b && a > SIZE_MAX / b) throw std::overflow_error("post-HF allocation overflow");
  return a * b;
}
size_t add(size_t a, size_t b) {
  if (a > SIZE_MAX - b) throw std::overflow_error("post-HF allocation overflow");
  return a + b;
}
template <class F>
int guarded(char* error, size_t size, F fn) noexcept {
  try {
    fn();
    return 0;
  } catch (const DeviceAllocationError& e) {
    error_text(error, size, e.what());
    return 2;
  } catch (const std::bad_alloc& e) {
    error_text(error, size, e.what());
    return 2;
  } catch (const std::length_error& e) {
    error_text(error, size, e.what());
    return 2;
  } catch (const std::exception& e) {
    error_text(error, size, e.what());
    return 1;
  } catch (...) {
    error_text(error, size, "unknown post-HF CUDA error");
    return 1;
  }
}
}  // namespace
extern "C" {
int posthf_cuda_create_v1(int device, size_t nbf, const size_t* m, const size_t* tile,
                          const double* coefficients, size_t expected_bytes, void** out,
                          char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!out) throw std::invalid_argument("null output handle");
    *out = nullptr;
    if (!nbf || !m || !tile || !coefficients) throw std::invalid_argument("invalid MO plan");
    auto p = std::make_unique<Transform>();
    p->nbf = nbf;
    p->output = 1;
    size_t input = 1;
    for (unsigned k = 0; k < 4; ++k) {
      if (!m[k] || m[k] > nbf || !tile[k] || tile[k] > nbf)
        throw std::invalid_argument("invalid MO/tile dimensions");
      p->m[k] = m[k];
      p->tile[k] = tile[k];
      p->c_offset[k] = p->coefficients;
      p->coefficients = add(p->coefficients, mul(nbf, m[k]));
      p->output = mul(p->output, m[k]);
      input = mul(input, tile[k]);
    }
    p->stage = input;
    for (unsigned k = 0; k < 4; ++k) {
      input = mul(input / tile[k], m[k]);
      p->stage = std::max(p->stage, input);
    }
    if (p->stage > INT_MAX || nbf > INT_MAX)
      throw std::invalid_argument("MO stage exceeds cuBLAS int32 indexing");
    const size_t numeric = mul(8, add(add(p->coefficients, mul(2, p->stage)), p->output));
    const size_t error_offset = mul(add(numeric, 255) / 256, 256);
    const size_t workspace = add(error_offset, 256), bytes = add(workspace, 4U << 20);
    if (bytes != expected_bytes)
      throw std::invalid_argument("native/Python MO allocation plan mismatch");
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device));
    p->context.prepare(device, prop.major, prop.minor, bytes, error_offset, workspace, 4U << 20,
                       96U << 20, true);
    p->c = reinterpret_cast<double*>(p->context.arena);
    p->first = p->c + p->coefficients;
    p->second = p->first + p->stage;
    p->result = p->second + p->stage;
    p->context.section(true, p->context.metrics.input_ms, [&] {
      cuda_check(cudaMemcpyAsync(p->c, coefficients, p->coefficients * 8, cudaMemcpyHostToDevice,
                                 p->context.stream));
      cuda_check(cudaMemsetAsync(p->result, 0, p->output * 8, p->context.stream));
      cuda_check(cudaMemsetAsync(p->context.error, 0, sizeof(int), p->context.stream));
    });
    *out = p.release();
  });
}
void posthf_cuda_destroy_v1(void* pointer) { delete static_cast<Transform*>(pointer); }
int posthf_cuda_add_v1(void* pointer, const double* values, const size_t* begin,
                       const size_t* counts, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !values || !begin || !counts) throw std::invalid_argument("null MO tile");
    auto& p = *static_cast<Transform*>(pointer);
    auto& ctx = p.context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    std::array<size_t, 4> shape{};
    size_t elements = 1;
    for (unsigned k = 0; k < 4; ++k) {
      if (!counts[k] || counts[k] > p.tile[k] || begin[k] > p.nbf || counts[k] > p.nbf - begin[k])
        throw std::invalid_argument("MO tile outside prepared bounds");
      shape[k] = counts[k];
      elements = mul(elements, counts[k]);
    }
    ctx.section(true, ctx.metrics.input_ms, [&] {
      cuda_check(
          cudaMemcpyAsync(p.first, values, elements * 8, cudaMemcpyHostToDevice, ctx.stream));
    });
    double *in = p.first, *out = p.second;
    ctx.section(true, ctx.metrics.library_ms, [&] {
      for (unsigned k = 0; k < 4; ++k) {
        const int dim = shape[0], rest = elements / shape[0], columns = p.m[k];
        const double alpha = 1, beta = 0;
        // Row-major input[AO,rest]^T @ C[AO,MO] -> output[rest,MO].
        // Column-major views reverse the product and transpose only input.
        blas_check(cublasDgemm(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_T, columns, rest, dim, &alpha,
                               p.c + p.c_offset[k] + begin[k] * p.m[k], columns, in, rest, &beta,
                               out, columns));
        elements = mul(rest, p.m[k]);
        for (unsigned axis = 0; axis < 3; ++axis) shape[axis] = shape[axis + 1];
        shape[3] = p.m[k];
        std::swap(in, out);
      }
      const double one = 1;
      blas_check(cublasDaxpy(ctx.handle, p.output, &one, in, 1, p.result, 1));
    });
    ctx.section(true, ctx.metrics.kernel_ms, [&] {
      check_scale<<<blocks(p.output, 256), 256, 0, ctx.stream>>>(p.result, p.output, 1, ctx.error,
                                                                 0);
      cuda_check(cudaGetLastError());
    });
    int invalid = 0;
    cuda_check(
        cudaMemcpyAsync(&invalid, ctx.error, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
    cuda_check(cudaStreamSynchronize(ctx.stream));
    if (invalid) throw std::runtime_error("nonfinite MO transformation");
  });
}
int posthf_cuda_download_v1(void* pointer, double* out, size_t elements, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !out) throw std::invalid_argument("null MO download");
    auto& p = *static_cast<Transform*>(pointer);
    auto& ctx = p.context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    if (elements != p.output) throw std::invalid_argument("MO download size mismatch");
    ctx.section(true, ctx.metrics.output_ms, [&] {
      cuda_check(cudaMemcpyAsync(out, p.result, elements * 8, cudaMemcpyDeviceToHost, ctx.stream));
    });
  });
}
int posthf_cuda_metrics_v1(void* pointer, Metrics* out, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !out) throw std::invalid_argument("null MO metrics");
    auto& ctx = static_cast<Transform*>(pointer)->context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    *out = ctx.metrics;
    out->observed_device_delta = ctx.device_delta();
    out->device_ms = out->input_ms + out->output_ms + out->library_ms + out->kernel_ms;
  });
}
void* posthf_cuda_pointer_v1(void* pointer) {
  return pointer ? static_cast<Transform*>(pointer)->result : nullptr;
}
int posthf_cuda_versions_v1(void* pointer, int* values, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !values) throw std::invalid_argument("null CUDA version request");
    auto& ctx = static_cast<Transform*>(pointer)->context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    cuda_check(cudaRuntimeGetVersion(values));
    cuda_check(cudaDriverGetVersion(values + 1));
    blas_check(cublasGetVersion(ctx.handle, values + 2));
  });
}
}
