// Validation adapter for the actual native streamed CUDA source-response owner.
// Inputs may be deliberately unphysical to expose transposes and masked errors.
#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <vector>

#include "posthf/df_mo_response.hpp"
#include "runtime/cuda_resources.cuh"

namespace {
struct Blas {
  cublasHandle_t value{};
  ~Blas() {
    if (value) (void)cublasDestroy(value);
  }
};
void check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("probe BLAS setup failed");
}
}  // namespace

extern "C" int df_mo_response_probe(std::size_t n, std::size_t q, const double* const* input,
                                    double* const* output, std::size_t budget,
                                    std::size_t caller_bytes, int failure, std::size_t* counts,
                                    char* error, std::size_t error_size) noexcept {
  using namespace generativeqc;
  try {
    const auto nn = n * n, qq = q * q, full = nn * q, small = nn + qq;
    std::vector<double> bar_raw(full), bar_c(nn), bar_root(qq);
    runtime::CudaDeviceScope scope(0);
    runtime::OwnedCudaStream stream(0), wrong_stream(0);
    Blas blas;
    check(cublasCreate(&blas.value));
    check(cublasSetStream(blas.value, failure == 4 ? wrong_stream.get() : stream.get()));
    constexpr std::size_t workspace_bytes = 4ULL << 20;
    runtime::OwnedCudaBuffer<unsigned char> workspace(0, workspace_bytes, stream.get());
    check(cublasSetWorkspace(blas.value, workspace.get(), workspace_bytes));
    if (failure == 5) check(cublasSetPointerMode(blas.value, CUBLAS_POINTER_MODE_DEVICE));
    runtime::OwnedCudaBuffer<double> data(0, 2 * full + small, stream.get());
    double* raw = data.get();
    double* c = raw + full;
    double* root = c + nn;
    double* bar = root + qq;
    const std::array<double*, 4> device{raw, c, root, bar};
    const std::array<std::size_t, 4> sizes{full, nn, qq, full};
    for (std::size_t i = 0; i < 4; ++i)
      runtime::cuda_resource_check(cudaMemcpyAsync(device[i], input[i], sizes[i] * sizeof(double),
                                                   cudaMemcpyHostToDevice, stream.get()));
    runtime::cuda_resource_check(cudaStreamSynchronize(stream.get()));
    std::size_t reads = 0, consumed = 0;
    const auto result = posthf::pullback_df_mo_source_cuda(
        {n, q, c, root, bar}, 0, stream.get(), blas.value,
        [&](std::size_t mu, double* row, cudaStream_t owner_stream) {
          if (mu != reads++ % n || owner_stream != stream.get())
            throw std::runtime_error("probe source order/stream mismatch");
          if (failure == 1 && reads == n + 1)
            throw std::runtime_error("injected second-pass source failure");
          runtime::cuda_resource_check(cudaMemcpyAsync(row, raw + mu * n * q,
                                                       n * q * sizeof(double),
                                                       cudaMemcpyDeviceToDevice, owner_stream));
        },
        [&](std::size_t mu, const double* row, cudaStream_t owner_stream) {
          if (mu != consumed++ || owner_stream != stream.get())
            throw std::runtime_error("probe response row order/stream mismatch");
          runtime::cuda_resource_check(cudaMemcpyAsync(bar_raw.data() + mu * n * q, row,
                                                       n * q * sizeof(double),
                                                       cudaMemcpyDeviceToHost, owner_stream));
          if (failure == 2)
            throw std::runtime_error("injected consume failure after queued download");
        },
        [&](const double* dc, const double* dw, cudaStream_t owner_stream) {
          runtime::cuda_resource_check(cudaMemcpyAsync(bar_c.data(), dc, nn * sizeof(double),
                                                       cudaMemcpyDeviceToHost, owner_stream));
          runtime::cuda_resource_check(cudaMemcpyAsync(bar_root.data(), dw, qq * sizeof(double),
                                                       cudaMemcpyDeviceToHost, owner_stream));
          if (failure == 3)
            throw std::runtime_error("injected finish failure after queued downloads");
        },
        budget, caller_bytes + workspace_bytes + sizeof(double) * (5 * full + 3 * small));
    const std::array<const std::vector<double>*, 3> values{&bar_raw, &bar_c, &bar_root};
    for (std::size_t i = 0; i < 3; ++i) std::copy(values[i]->begin(), values[i]->end(), output[i]);
    const std::size_t work[]{result.source_rows,
                             result.source_values,
                             result.output_rows,
                             result.output_values,
                             result.gemms,
                             result.contraction_summands,
                             result.owned_device_bytes,
                             result.numeric_capacity_bytes,
                             result.scalar_d2h_bytes};
    std::copy(std::begin(work), std::end(work), counts);
    return 0;
  } catch (const std::exception& e) {
    if (error && error_size) std::snprintf(error, error_size, "%s", e.what());
    return 1;
  }
}
