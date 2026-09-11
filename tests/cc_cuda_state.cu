// Real-device test adapter only; production state is a #146 reservation.
#include "cc/cuda_state.cuh"

extern "C" int test_cc_state(int elements, int history, const double* errors, const double* vectors,
                             double* result, double* norm, int* diis_status, char* error,
                             size_t error_size) {
  using namespace vibeqc_tensor;
  try {
    if (elements < 1 || history < 2 || history > 20) throw std::invalid_argument("test shape");
    const size_t h = history, n = elements, partials = blocks(elements, 256);
    const size_t count = 2 * n * h + h * h + (h + 1) * (h + 1) + (h + 1) + n + partials + 1;
    const size_t error_offset = (count * 8 + 255) / 256 * 256;
    Context ctx;
    cudaDeviceProp property{};
    cuda_check(cudaGetDeviceProperties(&property, 0));
    ctx.prepare(0, property.major, property.minor, error_offset + 256, error_offset, error_offset,
                0, 96ULL << 20, true);
    auto* e = reinterpret_cast<double*>(ctx.arena);
    auto* v = e + n * h;
    auto* gram = v + n * h;
    auto* system = gram + h * h;
    auto* coefficients = system + (h + 1) * (h + 1);
    auto* out = coefficients + h + 1;
    auto* scratch = out + n;
    auto* max_norm = scratch + partials;
    auto* state = ctx.error + 1;
    cuda_check(cudaMemcpyAsync(e, errors, n * h * 8, cudaMemcpyHostToDevice, ctx.stream));
    cuda_check(cudaMemcpyAsync(v, vectors, n * h * 8, cudaMemcpyHostToDevice, ctx.stream));
    cuda_check(cudaMemsetAsync(ctx.error, 0, sizeof(int), ctx.stream));
    cuda_check(cudaMemsetAsync(out, 0, n * 8, ctx.stream));
    vibeqc::cc::diis_gram(ctx, e, elements, history, gram);
    vibeqc::cc::diis_coefficients<<<1, 1, 0, ctx.stream>>>(gram, history, system, coefficients,
                                                           state);
    vibeqc::cc::diis_combine<<<blocks(elements, 256), 256, 0, ctx.stream>>>(
        v, coefficients, elements, history, state, out, ctx.error);
    vibeqc::cc::residual_partials<<<partials, 256, 0, ctx.stream>>>(e, elements, scratch,
                                                                    ctx.error);
    vibeqc::cc::residual_finish<<<1, 1, 0, ctx.stream>>>(scratch, partials, max_norm);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpyAsync(result, out, n * 8, cudaMemcpyDeviceToHost, ctx.stream));
    cuda_check(cudaMemcpyAsync(norm, max_norm, 8, cudaMemcpyDeviceToHost, ctx.stream));
    cuda_check(
        cudaMemcpyAsync(diis_status, state, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
    int arithmetic = 0;
    cuda_check(
        cudaMemcpyAsync(&arithmetic, ctx.error, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
    cuda_check(cudaStreamSynchronize(ctx.stream));
    if (arithmetic) throw std::runtime_error("nonfinite test arithmetic");
    return 0;
  } catch (const std::exception& e) {
    error_text(error, error_size, e.what());
    return 1;
  }
}
