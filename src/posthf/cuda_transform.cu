/** Resident conventional MO block with four cuBLAS AO-axis transformations.
 * The AO source is explicitly host staged; no tensor arithmetic occurs there.
 * Reuses CG09's private stream, cuBLAS provider guard and owned arena lifecycle.
 */
#include <array>
#include <climits>
#include <vector>

#include "../tensor/cuda_runtime.cuh"

namespace {
using namespace vibeqc_tensor;
struct Transform {
  Context context;
  size_t nbf{}, stage{}, output{}, coefficients{};
  std::array<size_t, 4> m{}, tile{}, c_offset{};
  double *c{}, *first{}, *second{}, *result{};
  bool validated = true;
  bool failed = false;
};
using vibeqc::runtime::size_add;
using vibeqc::runtime::size_mul;
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
// Exception-only fence: queued copies must stop borrowing host storage before
// a failed API call returns. Successful profiled sections already drain.
struct StreamDrain {
  cudaStream_t stream;
  bool active = true;
  ~StreamDrain() {
    if (active) (void)cudaStreamSynchronize(stream);
  }
};
void validate(Transform& p) {
  if (p.failed) throw std::runtime_error("MO accumulation failed; recreate the transform");
  if (p.validated) return;
  auto& ctx = p.context;
  ctx.section(true, ctx.metrics.kernel_ms, [&] {
    check_scale<<<blocks(p.output, 256), 256, 0, ctx.stream>>>(p.result, p.output, 1, ctx.error, 0);
    cuda_check(cudaGetLastError());
  });
  int invalid = 0;
  StreamDrain readback_drain{ctx.stream};
  cuda_check(cudaMemcpyAsync(&invalid, ctx.error, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
  cuda_check(cudaStreamSynchronize(ctx.stream));
  readback_drain.active = false;
  if (invalid) throw std::runtime_error("nonfinite MO transformation");
  p.validated = true;
}

struct BatchState {
  size_t stage{}, output{}, coefficients{};
  std::array<size_t, 4> m{}, c_offset{};
  double *c{}, *first{}, *second{}, *result{};
};
struct BatchTransform {
  Context context;
  size_t nbf{};
  std::array<size_t, 4> tile{};
  std::vector<BatchState> states;
  double* raw{};
  bool validated = true;
  bool failed = false;
};
void validate(BatchTransform& p) {
  if (p.failed) throw std::runtime_error("MO batch accumulation failed; recreate the transform");
  if (p.validated) return;
  auto& ctx = p.context;
  ctx.section(true, ctx.metrics.kernel_ms, [&] {
    for (size_t request = 0; request < p.states.size(); ++request) {
      const auto& state = p.states[request];
      check_scale<<<blocks(state.output, 256), 256, 0, ctx.stream>>>(
          state.result, state.output, 1, ctx.error, static_cast<int>(request));
      cuda_check(cudaGetLastError());
    }
  });
  int invalid = 0;
  StreamDrain readback_drain{ctx.stream};
  cuda_check(cudaMemcpyAsync(&invalid, ctx.error, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
  cuda_check(cudaStreamSynchronize(ctx.stream));
  readback_drain.active = false;
  if (invalid) throw std::runtime_error("nonfinite MO batch transformation");
  p.validated = true;
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
      p->coefficients = size_add(p->coefficients, size_mul(nbf, m[k]));
      p->output = size_mul(p->output, m[k]);
      input = size_mul(input, tile[k]);
    }
    p->stage = input;
    for (unsigned k = 0; k < 4; ++k) {
      input = size_mul(input / tile[k], m[k]);
      p->stage = std::max(p->stage, input);
    }
    if (p->stage > INT_MAX || nbf > INT_MAX)
      throw std::invalid_argument("MO stage exceeds cuBLAS int32 indexing");
    const size_t numeric =
        size_mul(8, size_add(size_add(p->coefficients, size_mul(2, p->stage)), p->output));
    const size_t error_offset = size_mul(size_add(numeric, 255) / 256, 256);
    const size_t workspace = size_add(error_offset, 256), bytes = size_add(workspace, 4U << 20);
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
      elements = size_mul(elements, counts[k]);
    }
    if (p.failed) throw std::runtime_error("MO accumulation failed; recreate the transform");
    // Invalidate before any submission: section timing/fences may fail after
    // DAXPY has already changed the accumulator, including to a finite partial.
    p.validated = false;
    p.failed = true;
    StreamDrain accumulation_drain{ctx.stream};
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
        elements = size_mul(rest, p.m[k]);
        for (unsigned axis = 0; axis < 3; ++axis) shape[axis] = shape[axis + 1];
        shape[3] = p.m[k];
        std::swap(in, out);
      }
      const double one = 1;
      blas_check(cublasDaxpy(ctx.handle, p.output, &one, in, 1, p.result, 1));
    });
    p.failed = false;
    accumulation_drain.active = false;
  });
}
int posthf_cuda_validate_v1(void* pointer, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer) throw std::invalid_argument("null MO validation");
    auto& p = *static_cast<Transform*>(pointer);
    auto& ctx = p.context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    validate(p);
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
    validate(p);
    StreamDrain download_drain{ctx.stream};
    ctx.section(true, ctx.metrics.output_ms, [&] {
      cuda_check(cudaMemcpyAsync(out, p.result, elements * 8, cudaMemcpyDeviceToHost, ctx.stream));
    });
    download_drain.active = false;
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
  if (!pointer) return nullptr;
  try {
    auto& p = *static_cast<Transform*>(pointer);
    std::lock_guard<std::mutex> lock(p.context.mutex);
    p.context.check_device();
    validate(p);
    return p.result;
  } catch (...) {
    // Retain the legacy pointer ABI without letting native callers bypass the
    // publication boundary. The status-returning validate API reports details.
    return nullptr;
  }
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

int posthf_cuda_batch_create_v1(int device, size_t nbf, size_t request_count, const size_t* shapes,
                                const size_t* tile, const double* coefficients,
                                size_t maximum_bytes, void** out, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!out) throw std::invalid_argument("null batch output handle");
    *out = nullptr;
    if (!nbf || !request_count || request_count > static_cast<size_t>(INT_MAX) || !shapes ||
        !tile || !coefficients || !maximum_bytes)
      throw std::invalid_argument("invalid MO batch plan");
    auto p = std::make_unique<BatchTransform>();
    p->nbf = nbf;
    size_t tile_elements = 1;
    for (unsigned axis = 0; axis < 4; ++axis) {
      if (!tile[axis] || tile[axis] > nbf)
        throw std::invalid_argument("invalid MO batch tile dimensions");
      p->tile[axis] = tile[axis];
      tile_elements = size_mul(tile_elements, tile[axis]);
    }
    if (nbf > INT_MAX) throw std::invalid_argument("MO batch exceeds cuBLAS int32 indexing");

    p->states.resize(request_count);
    size_t coefficient_elements = 0;
    size_t numeric_elements = 0;
    for (size_t request = 0; request < request_count; ++request) {
      auto& state = p->states[request];
      size_t transformed = tile_elements;
      state.output = 1;
      for (unsigned axis = 0; axis < 4; ++axis) {
        const auto m = shapes[4 * request + axis];
        if (!m || m > nbf) throw std::invalid_argument("invalid MO batch block dimensions");
        state.m[axis] = m;
        state.c_offset[axis] = state.coefficients;
        state.coefficients = size_add(state.coefficients, size_mul(nbf, m));
        state.output = size_mul(state.output, m);
      }
      state.stage = tile_elements;
      for (unsigned axis = 0; axis < 4; ++axis) {
        transformed = size_mul(transformed / tile[axis], state.m[axis]);
        state.stage = std::max(state.stage, transformed);
      }
      if (state.stage > INT_MAX)
        throw std::invalid_argument("MO batch stage exceeds cuBLAS int32 indexing");
      coefficient_elements = size_add(coefficient_elements, state.coefficients);
      numeric_elements =
          size_add(numeric_elements, size_add(size_mul(2, state.stage), state.output));
    }
    numeric_elements = size_add(numeric_elements, coefficient_elements);
    const size_t numeric = size_mul(8, numeric_elements);
    const size_t error_offset = size_mul(size_add(numeric, 255) / 256, 256);
    const size_t workspace = size_add(error_offset, 256);
    const size_t bytes = size_add(workspace, 4U << 20);
    if (bytes > maximum_bytes)
      throw std::length_error("shared MO batch allocation exceeds admitted capacity");

    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device));
    p->context.prepare(device, prop.major, prop.minor, bytes, error_offset, workspace, 4U << 20,
                       96U << 20, true);
    auto* base = reinterpret_cast<double*>(p->context.arena);
    size_t coefficient_cursor = 0;
    auto* scratch = base + coefficient_elements;
    for (auto& state : p->states) {
      state.c = base + coefficient_cursor;
      coefficient_cursor = size_add(coefficient_cursor, state.coefficients);
      state.first = scratch;
      scratch += state.stage;
      state.second = scratch;
      scratch += state.stage;
      state.result = scratch;
      scratch += state.output;
    }
    p->raw = p->states.front().second;
    p->context.section(true, p->context.metrics.input_ms, [&] {
      cuda_check(cudaMemcpyAsync(base, coefficients, coefficient_elements * 8,
                                 cudaMemcpyHostToDevice, p->context.stream));
      for (const auto& state : p->states)
        cuda_check(cudaMemsetAsync(state.result, 0, state.output * 8, p->context.stream));
      cuda_check(cudaMemsetAsync(p->context.error, 0, sizeof(int), p->context.stream));
    });
    *out = p.release();
  });
}
void posthf_cuda_batch_destroy_v1(void* pointer) { delete static_cast<BatchTransform*>(pointer); }
int posthf_cuda_batch_add_v1(void* pointer, const double* values, const size_t* begin,
                             const size_t* counts, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !values || !begin || !counts) throw std::invalid_argument("null MO batch tile");
    auto& p = *static_cast<BatchTransform*>(pointer);
    auto& ctx = p.context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    std::array<size_t, 4> shape{};
    size_t elements = 1;
    for (unsigned axis = 0; axis < 4; ++axis) {
      if (!counts[axis] || counts[axis] > p.tile[axis] || begin[axis] > p.nbf ||
          counts[axis] > p.nbf - begin[axis])
        throw std::invalid_argument("MO batch tile outside prepared bounds");
      shape[axis] = counts[axis];
      elements = size_mul(elements, counts[axis]);
    }
    if (p.failed) throw std::runtime_error("MO batch accumulation failed; recreate the transform");
    p.validated = false;
    p.failed = true;
    StreamDrain accumulation_drain{ctx.stream};
    ctx.section(true, ctx.metrics.input_ms, [&] {
      cuda_check(cudaMemcpyAsync(p.raw, values, elements * 8, cudaMemcpyHostToDevice, ctx.stream));
    });
    ctx.section(true, ctx.metrics.library_ms, [&] {
      for (auto& state : p.states) {
        const int dim = static_cast<int>(shape[0]);
        const int rest = static_cast<int>(elements / shape[0]);
        const int columns = static_cast<int>(state.m[0]);
        const double alpha = 1, beta = 0;
        blas_check(cublasDgemm(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_T, columns, rest, dim, &alpha,
                               state.c + state.c_offset[0] + begin[0] * state.m[0], columns, p.raw,
                               rest, &beta, state.first, columns));
      }
      for (auto& state : p.states) {
        auto transformed_shape = shape;
        auto transformed_elements = size_mul(elements / shape[0], state.m[0]);
        for (unsigned axis = 0; axis < 3; ++axis)
          transformed_shape[axis] = transformed_shape[axis + 1];
        transformed_shape[3] = state.m[0];
        double* in = state.first;
        double* out_state = state.second;
        for (unsigned k = 1; k < 4; ++k) {
          const int dim = static_cast<int>(transformed_shape[0]);
          const int rest = static_cast<int>(transformed_elements / transformed_shape[0]);
          const int columns = static_cast<int>(state.m[k]);
          const double alpha = 1, beta = 0;
          blas_check(cublasDgemm(ctx.handle, CUBLAS_OP_N, CUBLAS_OP_T, columns, rest, dim, &alpha,
                                 state.c + state.c_offset[k] + begin[k] * state.m[k], columns, in,
                                 rest, &beta, out_state, columns));
          transformed_elements = size_mul(rest, state.m[k]);
          for (unsigned axis = 0; axis < 3; ++axis)
            transformed_shape[axis] = transformed_shape[axis + 1];
          transformed_shape[3] = state.m[k];
          std::swap(in, out_state);
        }
        const double one = 1;
        blas_check(
            cublasDaxpy(ctx.handle, static_cast<int>(state.output), &one, in, 1, state.result, 1));
      }
    });
    p.failed = false;
    accumulation_drain.active = false;
  });
}
int posthf_cuda_batch_download_v1(void* pointer, double* const* outputs, const size_t* elements,
                                  size_t request_count, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !outputs || !elements) throw std::invalid_argument("null MO batch download");
    auto& p = *static_cast<BatchTransform*>(pointer);
    auto& ctx = p.context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    if (request_count != p.states.size())
      throw std::invalid_argument("MO batch download request count mismatch");
    for (size_t request = 0; request < request_count; ++request)
      if (!outputs[request] || elements[request] != p.states[request].output)
        throw std::invalid_argument("MO batch download size mismatch");
    validate(p);
    StreamDrain download_drain{ctx.stream};
    ctx.section(true, ctx.metrics.output_ms, [&] {
      for (size_t request = 0; request < request_count; ++request)
        cuda_check(cudaMemcpyAsync(outputs[request], p.states[request].result,
                                   elements[request] * 8, cudaMemcpyDeviceToHost, ctx.stream));
    });
    download_drain.active = false;
  });
}
int posthf_cuda_batch_metrics_v1(void* pointer, Metrics* out, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !out) throw std::invalid_argument("null MO batch metrics");
    auto& ctx = static_cast<BatchTransform*>(pointer)->context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    *out = ctx.metrics;
    out->observed_device_delta = ctx.device_delta();
    out->device_ms = out->input_ms + out->output_ms + out->library_ms + out->kernel_ms;
  });
}
}
