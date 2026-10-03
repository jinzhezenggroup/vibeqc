#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "df_mo_source_generated.hpp"
#include "posthf/capacity.hpp"
#include "posthf/df_mo_response.hpp"
#include "runtime/cuda_resources.cuh"

namespace generativeqc::posthf {
namespace {
std::size_t bytes(std::size_t count) { return checked_mul(count, sizeof(double)); }

// Generic arithmetic audit, separate from compiler-owned contractions. Every
// GEMM contributes to one sticky flag so a later projection cannot hide failure.
__global__ void audit_finite(const double* values, std::size_t count, int* error) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(gridDim.x) * blockDim.x)
    if (!isfinite(values[i])) atomicExch(error, 1);
}
}  // namespace

CudaDFMOSourceResponseDiagnostic pullback_df_mo_source_cuda(
    CudaDFMOSourceResponseView view, int device, cudaStream_t stream, cublasHandle_t blas,
    const CudaDFSourceRead& read, const CudaDFSourceConsume& consume,
    const CudaDFSourceFinish& finish, std::size_t budget, std::size_t caller_bytes) {
  const auto n = view.nbf, q = view.naux;
  if (!n || !q || !view.coefficients || !view.inverse_root || !view.bar_whitened || device < 0 ||
      !stream || !blas || !read || !consume || !finish || !budget)
    throw std::invalid_argument("invalid native CUDA DF MO source response request");
  const auto work = generated::df_mo_source_response_work(n, q);
  const auto row = checked_mul(n, q), nn = checked_mul(n, n), qq = checked_mul(q, q);
  const auto full = checked_mul(n, row);
  if (std::max({n, q, nn, row}) > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("DF MO source response exceeds BLAS indexing");
  const auto small = checked_add(nn, qq), owned = checked_add(work.scratch_values, small);
  CudaDFMOSourceResponseDiagnostic result;
  result.owned_device_bytes = checked_add(bytes(owned), sizeof(int));
  result.numeric_capacity_bytes = checked_add(
      caller_bytes, checked_add(result.owned_device_bytes, bytes(checked_add(full, small))));
  if (result.numeric_capacity_bytes > budget)
    throw std::length_error("DF MO source response exceeds complete numeric budget");
  runtime::CudaDeviceScope scope(device);
  cudaStream_t handle_stream{};
  cublasPointerMode_t mode{};
  if (cublasGetStream(blas, &handle_stream) != CUBLAS_STATUS_SUCCESS || handle_stream != stream ||
      cublasGetPointerMode(blas, &mode) != CUBLAS_STATUS_SUCCESS ||
      mode != CUBLAS_POINTER_MODE_HOST)
    throw std::invalid_argument("DF MO source response BLAS stream/scalar ownership mismatch");
  // The host error destination outlives every stream-dependent allocation.
  int failed = 0;
  runtime::OwnedCudaBuffer<double> storage(device, owned, stream);
  runtime::OwnedCudaBuffer<int> error(device, 1, stream);
  auto* first = storage.get();
  auto* transformed = first + full;
  auto* raw_row = transformed + full;
  auto* bar_row = raw_row + row;
  auto* bar_c = bar_row + row;
  auto* bar_root = bar_c + nn;
  runtime::cuda_resource_check(cudaMemsetAsync(error.get(), 0, sizeof(int), stream));
  generated::pullback_df_mo_source(
      n, q, view.coefficients, view.inverse_root, view.bar_whitened, first, transformed, bar_row,
      bar_c, bar_root,
      [&](std::size_t mu) {
        read(mu, raw_row, stream);
        ++result.source_rows;
        result.source_values = checked_add(result.source_values, row);
        return raw_row;
      },
      [&](std::size_t mu, const double* values) {
        consume(mu, values, stream);
        ++result.output_rows;
        result.output_values = checked_add(result.output_values, row);
      },
      [&](char ta, char tb, std::size_t m, std::size_t columns, std::size_t k, const double* a,
          const double* b, double* c, bool accumulate) {
        const double one = 1, beta = accumulate ? 1 : 0;
        const auto status = cublasDgemm(
            blas, ta == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T, tb == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T,
            static_cast<int>(m), static_cast<int>(columns), static_cast<int>(k), &one, a,
            static_cast<int>(ta == 'N' ? m : k), b, static_cast<int>(tb == 'N' ? k : columns),
            &beta, c, static_cast<int>(m));
        if (status != CUBLAS_STATUS_SUCCESS)
          throw std::runtime_error("native DF MO source response GEMM failed");
        const auto count = checked_mul(m, columns);
        const auto blocks =
            static_cast<unsigned>(std::min<std::size_t>((count + 255) / 256, 65535));
        audit_finite<<<blocks, 256, 0, stream>>>(c, count, error.get());
        runtime::cuda_resource_check(cudaGetLastError());
        ++result.gemms;
        result.contraction_summands =
            checked_add(result.contraction_summands, checked_mul(count, k));
      });
  if (result.source_rows != work.source_rows || result.source_values != work.raw_values ||
      result.output_rows != work.output_rows || result.output_values != work.output_values ||
      result.gemms != work.gemms || result.contraction_summands != work.contraction_summands)
    throw std::logic_error("DF MO source reverse execution differs from compiler work");
  finish(bar_c, bar_root, stream);
  runtime::cuda_resource_check(
      cudaMemcpyAsync(&failed, error.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
  runtime::cuda_resource_check(cudaStreamSynchronize(stream));
  result.scalar_d2h_bytes = sizeof(int);
  if (failed) throw std::runtime_error("nonfinite native DF MO source response arithmetic");
  return result;
}
}  // namespace generativeqc::posthf
