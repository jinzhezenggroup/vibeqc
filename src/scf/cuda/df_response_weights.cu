#include <algorithm>
#include <limits>

#include "generated_df_hf_response.cuh"
#include "generated_symmetric_matrix_function.cuh"
#include "runtime/cuda_component_trace.hpp"
#include "scf/cuda/df_jk_kernels.hpp"
#include "scf/cuda/df_packed_values.hpp"
#include "scf/cuda/df_response_weights.cuh"
#include "scf/cuda/df_scf_kernels.hpp"

namespace vibeqc::scf {
namespace {
constexpr unsigned threads = 128;
unsigned blocks(std::size_t size) { return static_cast<unsigned>((size + threads - 1) / threads); }

/** Scalar kernels preserve their original per-output summation order. The
 * exchange metric dot uses the plan's cuBLAS GEMV below, with this scalar
 * kernel retained for ablation. Both routes use the same bounded raw panel;
 * integral generation and the derivative consumer are unchanged.
 */
__global__ void inverse_kernel(std::size_t a, const double* x, double* inverse) {
  const auto ij = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (ij >= a * a) return;
  double value = 0;
  for (std::size_t k = 0; k < a; ++k) value += x[(ij / a) * a + k] * x[(ij % a) * a + k];
  inverse[ij] = value;
}

/** Scale projected AO factors before rotating them back to the public axis.
 * Constructing M^-1 as X X^T first loses significant digits through the later
 * raw-A contraction on practical JKFIT bases. Keep the eigendirections separate
 * until their factors have been divided by the corresponding eigenvalue.
 */
__global__ void divide_factor_eigenvalues(std::size_t matrix, std::size_t a,
                                          const double* eigenvalues, double* factors) {
  const auto element = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (element < matrix * a) factors[element] /= eigenvalues[element / matrix];
}

/** Charge vectors have the eigenvalue index contiguous within each density. */
__global__ void divide_charge_eigenvalues(std::size_t a, std::size_t terms,
                                          const double* eigenvalues, double* charges) {
  const auto element = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (element < a * terms) charges[element] /= eigenvalues[element % a];
}

__global__ void charge_kernel(std::size_t matrix, std::size_t a, std::size_t q, std::size_t terms,
                              const double* densities, const double* values, double* charges) {
  const auto t = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (t >= terms) return;
  double value = 0;
  for (std::size_t ij = 0; ij < matrix; ++ij) value += densities[t * matrix + ij] * values[ij];
  charges[t * a + q] = value;
}

__global__ void potential_kernel(std::size_t a, std::size_t terms, const double* inverse,
                                 const double* charges, double* potentials) {
  const auto tp = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (tp >= terms * a) return;
  const auto t = tp / a, p = tp % a;
  double value = 0;
  for (std::size_t q = 0; q < a; ++q) value += inverse[p * a + q] * charges[t * a + q];
  potentials[tp] = value;
}

/** Fold the physical Coulomb adjoint directly into public lower AO pairs.
 * Reading both density entries retains the dense adapter's exact convention,
 * even when its accepted symmetric input differs by floating-point roundoff.
 */
__global__ void packed_coulomb_weights(std::size_t n, std::size_t begin, std::size_t count,
                                       double coefficient, const double* density,
                                       const double* potential, double* weights) {
  const auto k = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  const auto pairs = n * (n + 1) / 2;
  if (k >= count * pairs) return;
  const auto pair = k % pairs;
  auto i = static_cast<std::size_t>((sqrt(8.0 * pair + 1) - 1) * .5);
  while (i * (i + 1) / 2 > pair) --i;
  while ((i + 1) * (i + 2) / 2 <= pair) ++i;
  const auto j = pair - i * (i + 1) / 2;
  const double density_pair = density[i * n + j] + (i == j ? 0 : density[j * n + i]);
  weights[k] += coefficient * density_pair * potential[begin + k / pairs];
}

/** Scatter one bounded rectangular GEMM block into folded lower AO pairs.
 * Columns of the BLAS output label consecutive public AO rows. Only diagonal
 * blocks contain unused upper entries; no dense AO panel is ever constructed.
 * The producer symmetrizes U before C U C^T, so doubling its off-diagonal
 * entries contracts precisely the symmetric part of the original adjoint.
 */
__global__ void add_packed_exchange_block(std::size_t begin, std::size_t rows, std::size_t columns,
                                          const double* block, double* weights,
                                          std::size_t count = 1, std::size_t pair_stride = 0) {
  const auto k = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (k >= count * rows * columns) return;
  const auto panel = k / (rows * columns), element = k % (rows * columns);
  const auto i = begin + element / columns, j = element % columns;
  if (j <= i) weights[panel * pair_stride + i * (i + 1) / 2 + j] += (i == j ? 1 : 2) * block[k];
}

/** R_Q = D^T A_Q D for a symmetric physical density, in two cubic products. */
__global__ void right_density_kernel(std::size_t n, const double* values, const double* density,
                                     double* temporary) {
  const auto ij = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (ij >= n * n) return;
  double value = 0;
  for (std::size_t k = 0; k < n; ++k) value += values[(ij / n) * n + k] * density[k * n + ij % n];
  temporary[ij] = value;
}

__global__ void left_density_kernel(std::size_t n, const double* density, const double* temporary,
                                    double* response) {
  const auto ij = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (ij >= n * n) return;
  double value = 0;
  for (std::size_t k = 0; k < n; ++k) value += density[k * n + ij / n] * temporary[k * n + ij % n];
  response[ij] = value;
}

__global__ void symmetrize_kernel(std::size_t a, double* values, std::size_t count = 1) {
  const auto element = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (element >= count * a * a) return;
  const auto ij = element % (a * a);
  values += element / (a * a) * a * a;
  if (ij / a >= ij % a) return;
  const auto ji = (ij % a) * a + ij / a;
  values[ij] = values[ji] = .5 * (values[ij] + values[ji]);
}
}  // namespace

std::size_t cuda_df_response_workspace_elements(std::size_t n, std::size_t a, std::size_t terms,
                                                std::size_t tile) {
  return 4 * a * a + (3 + 2 * tile) * n * n + 2 * terms * a;
}

/** Full-rank response with inverse-applied factors before quadratic products.
 * Forming a raw A Gram and then applying two ill-conditioned metric inverses
 * amplifies its rounding error. B_P=sum_Q A_Q M^-1_QP instead gives
 * bar_A_P=cJ*D*c_P-2*cK*D^T*B_P*D and
 * bar_M_PQ=-cJ*c_P*c_Q/2+cK*(D^T*B_Q*D):B_P directly.
 *
 * Two already charged complete panels are required. An owned full tile reuses
 * its raw panel as weights only after the B transform; a resident borrow keeps
 * raw immutable and lends its other two buffers. No new allocation, source
 * regeneration, or host numerical work is introduced by this route.
 */
static cudaError_t contract_full_rank_response(
    std::size_t n, std::size_t a, std::span<const DensityFittingDensityResponse> terms,
    const double* densities, CudaDfMetricView metric, std::size_t tile, double* workspace,
    cudaStream_t stream, cublasHandle_t blas, bool serial_metric_dot, bool blas_products,
    const std::function<void(std::size_t, double*)>& read_values,
    const CudaDfResponseConsumer& consume, const CudaDfResponseBuffers* borrowed,
    std::span<const double> raw_host) {
  const auto matrix = n * n, aa = a * a;
  auto* inverse = workspace;
  auto* bar_metric = inverse + aa;
  auto* values = workspace + 4 * aa;
  auto* temporary = values + matrix;
  auto* response = temporary + matrix;
  auto* charges = response + matrix;
  auto* potentials = charges + terms.size() * a;
  auto* first_panel = potentials + terms.size() * a;
  auto* raw = borrowed ? borrowed->raw_auxiliary_major : first_panel;
  auto* fitted = borrowed ? borrowed->exchange_response : first_panel + matrix * a;
  auto* weights = borrowed ? borrowed->staging_weights : first_panel;
  const auto ni = static_cast<int>(n), ai = static_cast<int>(a), mi = static_cast<int>(matrix);
  const double one = 1, zero = 0;
  const auto checked = [](cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw CudaDfResponseBlasFailure{status};
  };
  if (borrowed && !borrowed->resident_raw.data) {
    auto error = cudaMemcpyAsync(weights, raw_host.data(), raw_host.size_bytes(),
                                 cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) return error;
    cuda_df::launch_gather_auxiliary_tile_kernel(blocks(matrix * a), threads, 0, stream, matrix, a,
                                                 0, 0, a, weights, raw);
    runtime::cuda_trace::trace_counter("raw_value_upload_bytes", raw_host.size_bytes());
    runtime::cuda_trace::trace_counter("raw_value_bulk_uploads", 1);
  } else if (!borrowed) {
    for (std::size_t q = 0; q < a; ++q) read_values(q, raw + q * matrix);
  }
  {
    runtime::cuda_trace::TraceRegion transform("response_inverse_applied_factors", stream);
    checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, mi, ai, ai, &one, raw, mi,
                        metric.eigenvectors, ai, &zero, fitted, mi));
    divide_factor_eigenvalues<<<blocks(matrix * a), threads, 0, stream>>>(
        matrix, a, metric.eigenvalues, fitted);
    checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, mi, ai, ai, &one, fitted, mi,
                        metric.eigenvectors, ai, &zero, weights, mi));
    // Owned raw storage may be overwritten after the first projection; a
    // borrowed raw tensor stays immutable because weights is separate there.
    // The first projection buffer is now dead and becomes the A-adjoint panel.
    std::swap(fitted, weights);
    runtime::cuda_trace::trace_counter("response_inverse_applied_factor_elements", matrix * a);
    runtime::cuda_trace::trace_counter("response_inverse_applied_factor_gemms", 2);
    runtime::cuda_trace::trace_counter("response_inverse_applied_factor_flops", 4 * matrix * aa);
  }
  auto error = cudaMemsetAsync(weights, 0, matrix * a * sizeof(double), stream);
  if (error != cudaSuccess) return error;
  error = cudaMemsetAsync(bar_metric, 0, aa * sizeof(double), stream);
  if (error != cudaSuccess) return error;
  // Charge and metric adjoints must use the same fitted factors as exchange.
  // Applying M^-1 to the raw charges separately loses this consistency through
  // a different cancellation order on practical ill-conditioned auxiliary bases.
  if (blas_products) {
    checked(generated::df_rhf_charge_contract(blas, mi, ai, 0, ai, static_cast<int>(terms.size()),
                                              densities, fitted, potentials));
    runtime::cuda_trace::trace_counter("response_charge_blas_dots", terms.size() * a);
  } else {
    for (std::size_t q = 0; q < a; ++q)
      charge_kernel<<<blocks(terms.size()), threads, 0, stream>>>(
          matrix, a, q, terms.size(), densities, fitted + q * matrix, potentials);
    runtime::cuda_trace::trace_counter("response_charge_scalar_dots", terms.size() * a);
  }
  runtime::cuda_trace::trace_counter("response_charge_dot_elements", terms.size() * a * matrix);
  for (std::size_t t = 0; t < terms.size(); ++t) {
    if (terms[t].coulomb_coefficient != 0) {
      coulomb_weights_kernel<<<blocks(matrix * a), threads, 0, stream>>>(
          matrix, a, 0, a, terms[t].coulomb_coefficient, densities + t * matrix, potentials + t * a,
          weights);
      coulomb_metric_kernel<<<blocks(aa), threads, 0, stream>>>(a, -terms[t].coulomb_coefficient,
                                                                potentials + t * a, bar_metric);
    }
    const double coefficient = terms[t].exchange_coefficient;
    if (coefficient == 0) continue;
    for (std::size_t q = 0; q < a; ++q) {
      if (blas_products) {
        checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, ni, ni, ni, &one,
                            densities + t * matrix, ni, fitted + q * matrix, ni, &zero, temporary,
                            ni));
        checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, ni, ni, ni, &one, temporary, ni,
                            densities + t * matrix, ni, &zero, response, ni));
        runtime::cuda_trace::trace_counter("response_density_blas_products", 2);
      } else {
        right_density_kernel<<<blocks(matrix), threads, 0, stream>>>(
            n, fitted + q * matrix, densities + t * matrix, temporary);
        left_density_kernel<<<blocks(matrix), threads, 0, stream>>>(n, densities + t * matrix,
                                                                    temporary, response);
        runtime::cuda_trace::trace_counter("response_density_scalar_products", 2);
      }
      runtime::cuda_trace::trace_counter("response_ao_matrix_products", 2);
      fitted_exchange_weights_kernel<<<blocks(matrix), threads, 0, stream>>>(
          matrix, coefficient, response, weights + q * matrix);
      if (serial_metric_dot) {
        exchange_metric_kernel<<<blocks(a), threads, 0, stream>>>(matrix, a, 0, a, q, -coefficient,
                                                                  fitted, response, bar_metric);
        runtime::cuda_trace::trace_counter("response_metric_serial_dots", a);
      } else {
        checked(cublasDgemv(blas, CUBLAS_OP_T, mi, ai, &coefficient, fitted, mi, response, 1, &one,
                            bar_metric + q, ai));
        runtime::cuda_trace::trace_counter("response_metric_blas_dots", a);
      }
    }
  }
  symmetrize_kernel<<<blocks(aa), threads, 0, stream>>>(a, bar_metric);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  runtime::cuda_trace::trace_counter("response_full_rank_factor_first", 1);
  for (std::size_t begin = 0; begin < a; begin += tile) {
    const auto count = std::min(tile, a - begin);
    runtime::cuda_trace::trace_counter("response_auxiliary_blocks", 1);
    consume({0, {begin, matrix, 1, a}, count * matrix, weights + begin * matrix});
  }
  consume({1, {}, aa, bar_metric});
  return cudaSuccess;
}

/** Bounded weight panels with a borrowed forward source or complete fitted B.
 * Keep W_P while replacing the other panel with each fitted B_Q, and form
 * bar_M_PQ=-W_P:B_Q/2. Both inverse applications precede the quadratic product;
 * multiplying a mixed W_P:A_Q product by the inverse afterwards is unstable.
 * The diagonal block consumes the original B_P before it is overwritten.
 * The reader counts repeated fitted projections and any source regeneration.
 * A complete fitted tensor permits direct reads and one metric GEMM per panel.
 */
static cudaError_t contract_full_rank_panels(
    std::size_t n, std::size_t a, std::span<const DensityFittingDensityResponse> terms,
    const double* densities, std::size_t tile, double* workspace, cudaStream_t stream,
    cublasHandle_t blas, bool serial_metric_dot, bool blas_products,
    const std::function<void(std::size_t, std::size_t, double*)>& read_fitted,
    const CudaDfResponseConsumer& consume, const double* all_fitted = nullptr) {
  const auto matrix = n * n, aa = a * a;
  auto* bar_metric = workspace + aa;
  auto* temporary = workspace + 4 * aa + matrix;
  auto* response = temporary + matrix;
  auto* potentials = response + matrix;
  // The first AO scratch matrix is reserved for packed-source unpacking.
  auto* fitted_panel = potentials + 2 * terms.size() * a;
  auto* weights = fitted_panel + (all_fitted ? a : tile) * matrix;
  const auto ni = static_cast<int>(n), ai = static_cast<int>(a), mi = static_cast<int>(matrix);
  const double one = 1, zero = 0, half = -.5;
  const auto checked = [](cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw CudaDfResponseBlasFailure{status};
  };
  auto error = cudaMemsetAsync(bar_metric, 0, aa * sizeof(double), stream);
  if (error != cudaSuccess) return error;
  for (std::size_t begin = 0; begin < a; begin += tile) {
    const auto count = std::min(tile, a - begin);
    if (!all_fitted) read_fitted(begin, count, fitted_panel);
    const double* fitted = all_fitted ? all_fitted + begin * matrix : fitted_panel;
    error = cudaMemsetAsync(weights, 0, count * matrix * sizeof(double), stream);
    if (error != cudaSuccess) return error;
    if (blas_products) {
      checked(generated::df_rhf_charge_contract(
          blas, mi, ai, static_cast<int>(begin), static_cast<int>(count),
          static_cast<int>(terms.size()), densities, fitted, potentials));
      runtime::cuda_trace::trace_counter("response_charge_blas_dots", terms.size() * count);
    } else {
      for (std::size_t p = 0; p < count; ++p)
        charge_kernel<<<blocks(terms.size()), threads, 0, stream>>>(
            matrix, a, begin + p, terms.size(), densities, fitted + p * matrix, potentials);
      runtime::cuda_trace::trace_counter("response_charge_scalar_dots", terms.size() * count);
    }
    runtime::cuda_trace::trace_counter("response_charge_dot_elements",
                                       terms.size() * count * matrix);
    for (std::size_t t = 0; t < terms.size(); ++t) {
      if (terms[t].coulomb_coefficient != 0)
        coulomb_weights_kernel<<<blocks(count * matrix), threads, 0, stream>>>(
            matrix, a, begin, count, terms[t].coulomb_coefficient, densities + t * matrix,
            potentials + t * a, weights);
      const double coefficient = terms[t].exchange_coefficient;
      if (coefficient == 0) continue;
      for (std::size_t p = 0; p < count; ++p) {
        if (blas_products) {
          checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, ni, ni, ni, &one,
                              densities + t * matrix, ni, fitted + p * matrix, ni, &zero, temporary,
                              ni));
          checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, ni, ni, ni, &one, temporary, ni,
                              densities + t * matrix, ni, &zero, response, ni));
          runtime::cuda_trace::trace_counter("response_density_blas_products", 2);
        } else {
          right_density_kernel<<<blocks(matrix), threads, 0, stream>>>(
              n, fitted + p * matrix, densities + t * matrix, temporary);
          left_density_kernel<<<blocks(matrix), threads, 0, stream>>>(n, densities + t * matrix,
                                                                      temporary, response);
          runtime::cuda_trace::trace_counter("response_density_scalar_products", 2);
        }
        runtime::cuda_trace::trace_counter("response_ao_matrix_products", 2);
        fitted_exchange_weights_kernel<<<blocks(matrix), threads, 0, stream>>>(
            matrix, coefficient, response, weights + p * matrix);
      }
    }
    const auto contract_metric = [&](std::size_t qbegin, std::size_t qcount) {
      if (serial_metric_dot) {
        for (std::size_t q = 0; q < qcount; ++q)
          exchange_metric_kernel<<<blocks(count), threads, 0, stream>>>(
              matrix, a, begin, count, qbegin + q, .5, weights, fitted + q * matrix, bar_metric);
        runtime::cuda_trace::trace_counter("response_metric_serial_dots", count * qcount);
      } else {
        // Column-major [Q,P] writes the row-major [P,Q] block with leading a.
        checked(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(qcount),
                            static_cast<int>(count), mi, &half, fitted, mi, weights, mi, &zero,
                            bar_metric + begin * a + qbegin, ai));
        runtime::cuda_trace::trace_counter("response_metric_blas_gemms", 1);
        runtime::cuda_trace::trace_counter("response_metric_blas_dots", count * qcount);
      }
    };
    if (all_fitted) {
      fitted = all_fitted;
      contract_metric(0, a);
    } else {
      contract_metric(begin, count);
      for (std::size_t qbegin = 0; qbegin < a; qbegin += tile) {
        if (qbegin == begin) continue;
        const auto qcount = std::min(tile, a - qbegin);
        read_fitted(qbegin, qcount, fitted_panel);
        contract_metric(qbegin, qcount);
      }
    }
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    consume({0, {begin, matrix, 1, a}, count * matrix, weights});
    runtime::cuda_trace::trace_counter("response_auxiliary_blocks", 1);
  }
  symmetrize_kernel<<<blocks(aa), threads, 0, stream>>>(a, bar_metric);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  runtime::cuda_trace::trace_counter(
      all_fitted ? "response_full_rank_single_tensor" : "response_full_rank_bounded_factor_first",
      1);
  consume({1, {}, aa, bar_metric});
  return cudaSuccess;
}

/** Fit one owned tensor in place without keeping a second complete tensor.
 * An auxiliary transform never mixes distinct AO pairs. Project a disjoint
 * pair batch into two currently dead metric matrices before overwriting its
 * original raw rows. The fixed scratch is O(Naux^2), already charged, and raw
 * integral production is one pass regardless of the derivative panel count.
 */
static cudaError_t fit_full_tensor_in_place(
    std::size_t n, std::size_t a, std::size_t terms, CudaDfMetricView metric, double* workspace,
    cudaStream_t stream, cublasHandle_t blas,
    const std::function<void(std::size_t, double*)>& read_values, double*& fitted) {
  const auto matrix = n * n, aa = a * a;
  auto* projected = workspace + 2 * aa;
  fitted = workspace + 4 * aa + 3 * matrix + 2 * terms * a;
  for (std::size_t q = 0; q < a; ++q) read_values(q, fitted + q * matrix);
  const auto pair_tile = std::min(matrix, 2 * a);
  const auto mi = static_cast<int>(matrix), ai = static_cast<int>(a);
  const double one = 1, zero = 0;
  runtime::cuda_trace::TraceRegion transform("response_inverse_applied_factors", stream);
  for (std::size_t begin = 0; begin < matrix; begin += pair_tile) {
    const auto count = std::min(pair_tile, matrix - begin);
    const auto rows = static_cast<int>(count);
    auto status = cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, rows, ai, ai, &one, fitted + begin,
                              mi, metric.eigenvectors, ai, &zero, projected, rows);
    if (status != CUBLAS_STATUS_SUCCESS) throw CudaDfResponseBlasFailure{status};
    divide_factor_eigenvalues<<<blocks(count * a), threads, 0, stream>>>(
        count, a, metric.eigenvalues, projected);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    status = cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, rows, ai, ai, &one, projected, rows,
                         metric.eigenvectors, ai, &zero, fitted + begin, mi);
    if (status != CUBLAS_STATUS_SUCCESS) throw CudaDfResponseBlasFailure{status};
    runtime::cuda_trace::trace_counter("response_inverse_applied_factor_gemms", 2);
  }
  runtime::cuda_trace::trace_counter("response_inverse_applied_factor_elements", matrix * a);
  runtime::cuda_trace::trace_counter("response_inverse_applied_factor_flops", 4 * matrix * aa);
  return cudaSuccess;
}

/** Borrow the resident plan's raw view and two full J/K temporaries.
 * The legacy provider instead uploads raw A and transposes it to [Q,ij]. For each exchange density,
 * compute every R_Q=D^T A_Q D once, then perform both all-auxiliary contractions
 * with GEMM. The raw tensor remains intact across spin terms. In particular,
 * E[P,Q]=A_P:R_Q still contains discarded metric directions; the same spectral
 * Frechet map as the bounded panel route handles those directions below.
 */
static cudaError_t contract_resident_response(
    std::size_t n, std::size_t a, std::span<const DensityFittingDensityResponse> terms,
    const double* densities, CudaDfMetricView metric, std::size_t tile, double* workspace,
    cudaStream_t stream, cublasHandle_t blas, const CudaDfResponseBuffers& buffers,
    std::span<const double> raw_host, const CudaDfResponseConsumer& consume) {
  const auto matrix = n * n, aa = a * a;
  auto* inverse = workspace;
  auto* bar_inverse = inverse + aa;
  auto* metric_temp = bar_inverse + aa;
  auto* transformed = metric_temp + aa;
  auto* temporary = transformed + aa;
  // The common bridge reserves at least three AO matrices; only one is used
  // by the serial Q projections here. Neither charges nor potentials alias it.
  auto* charges = temporary + 3 * matrix;
  auto* potentials = charges + terms.size() * a;
  auto* weights = buffers.staging_weights;
  auto* raw = buffers.raw_auxiliary_major;
  auto* response = buffers.exchange_response;
  const auto ni = static_cast<int>(n), ai = static_cast<int>(a);
  const auto mi = static_cast<int>(matrix), ti = static_cast<int>(terms.size());
  const double one = 1.0, zero = 0.0;
  const auto blas_check = [](cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw CudaDfResponseBlasFailure{status};
  };
  if (!buffers.resident_raw.data) {
    runtime::cuda_trace::TraceRegion upload("raw_value_resident_upload", stream);
    const auto error = cudaMemcpyAsync(weights, raw_host.data(), raw_host.size_bytes(),
                                       cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) return error;
    runtime::cuda_trace::trace_counter("raw_value_upload_bytes", raw_host.size_bytes());
    runtime::cuda_trace::trace_counter("raw_value_bulk_uploads", 1);
  }
  if (!buffers.resident_raw.data) {
    runtime::cuda_trace::TraceRegion transpose("raw_value_resident_transpose", stream);
    // Host [ij,Q] is column-major [Q,ij]; turn it into column-major [ij,Q].
    // CuMetal exposes a subset of cuBLAS without GEAM. A dependent capability
    // check keeps NVIDIA's fast transpose and reuses the existing J/K gather
    // on such providers, without adding another layout or derivative kernel.
    const auto transpose_raw = [&](auto handle) {
      if constexpr (requires {
                      cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, mi, ai, &one, weights, ai,
                                  &zero, weights, ai, raw, mi);
                    }) {
        blas_check(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, mi, ai, &one, weights, ai, &zero,
                               weights, ai, raw, mi));
      } else {
        cuda_df::launch_gather_auxiliary_tile_kernel(blocks(matrix * a), threads, 0, stream, matrix,
                                                     a, 0, 0, a, weights, raw);
      }
    };
    transpose_raw(blas);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    runtime::cuda_trace::trace_counter("raw_resident_transpose_elements", matrix * a);
  }
  runtime::cuda_trace::TraceRegion metric_inverse("metric_inverse", stream);
  inverse_kernel<<<blocks(aa), threads, 0, stream>>>(a, metric.inverse_square_root, inverse);
  metric_inverse.finish();
  auto error = cudaMemsetAsync(bar_inverse, 0, aa * sizeof(double), stream);
  if (error != cudaSuccess) return error;
  // Raw has been transposed before this same-stream overwrite of its upload
  // destination. That destination now accumulates the complete response W.
  error = cudaMemsetAsync(weights, 0, matrix * a * sizeof(double), stream);
  if (error != cudaSuccess) return error;
  runtime::cuda_trace::TraceRegion coulomb("coulomb_response", stream);
  {
    runtime::cuda_trace::TraceRegion charge_dot("coulomb_response_charge_dot", stream);
    blas_check(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, ai, ti, mi, &one, raw, mi, densities, mi,
                           &zero, charges, ai));
    runtime::cuda_trace::trace_counter("response_charge_blas_dots", terms.size() * a);
    runtime::cuda_trace::trace_counter("response_charge_dot_elements", terms.size() * a * matrix);
  }
  potential_kernel<<<blocks(terms.size() * a), threads, 0, stream>>>(a, terms.size(), inverse,
                                                                     charges, potentials);
  for (std::size_t t = 0; t < terms.size(); ++t) {
    if (terms[t].coulomb_coefficient == 0) continue;
    coulomb_metric_kernel<<<blocks(aa), threads, 0, stream>>>(a, terms[t].coulomb_coefficient,
                                                              charges + t * a, bar_inverse);
    coulomb_weights_kernel<<<blocks(matrix * a), threads, 0, stream>>>(
        matrix, a, 0, a, terms[t].coulomb_coefficient, densities + t * matrix, potentials + t * a,
        weights);
  }
  coulomb.finish();
  for (std::size_t t = 0; t < terms.size(); ++t) {
    const auto coefficient = terms[t].exchange_coefficient;
    if (coefficient == 0) continue;
    for (std::size_t q = 0; q < a; ++q) {
      runtime::cuda_trace::TraceRegion products("exchange_response_matrix_products", stream);
      // Preserve the old row/column-major contract without assuming bitwise
      // symmetry of either the physical density or a computed AO slice.
      blas_check(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, ni, ni, ni, &one,
                             densities + t * matrix, ni, raw + q * matrix, ni, &zero, temporary,
                             ni));
      blas_check(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, ni, ni, ni, &one, temporary, ni,
                             densities + t * matrix, ni, &zero, response + q * matrix, ni));
      runtime::cuda_trace::trace_counter("response_ao_matrix_products", 2);
      runtime::cuda_trace::trace_counter("response_density_blas_products", 2);
      runtime::cuda_trace::trace_counter("response_resident_exchange_columns", 1);
    }
    {
      runtime::cuda_trace::TraceRegion dot("exchange_response_metric_gemm", stream);
      const double alpha = -coefficient;
      // E row-major[P,Q] is column-major[Q,P]: R^T A writes precisely that
      // transpose, including both spin terms and the prior Coulomb adjoint.
      blas_check(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, ai, ai, mi, &alpha, response, mi, raw,
                             mi, &one, bar_inverse, ai));
      runtime::cuda_trace::trace_counter("response_metric_blas_dots", aa);
      runtime::cuda_trace::trace_counter("response_metric_blas_gemms", 1);
    }
    {
      runtime::cuda_trace::TraceRegion contraction("exchange_response_weight_gemm", stream);
      const double alpha = -2 * coefficient;
      // W^T=R^T (M+)^T. The existing row-major inverse is already the required
      // column-major transpose; explicit ordering retains its index contract.
      blas_check(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, mi, ai, ai, &alpha, response, mi,
                             inverse, ai, &one, weights, mi));
      runtime::cuda_trace::trace_counter("response_weight_blas_gemms", 1);
    }
  }
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  for (std::size_t begin = 0; begin < a; begin += tile) {
    const auto count = std::min(tile, a - begin);
    runtime::cuda_trace::trace_counter("response_auxiliary_blocks", 1);
    consume({0, {begin, matrix, 1, a}, count * matrix, weights + begin * matrix});
  }
  runtime::cuda_trace::TraceRegion reverse("metric_frechet_response", stream);
  tensor::launch_symmetric_pseudoinverse_vjp(a, metric.eigenvectors, metric.eigenvalues,
                                             metric.relative_threshold, bar_inverse, metric_temp,
                                             transformed, bar_inverse, stream);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  reverse.finish();
  consume({1, {}, aa, bar_inverse});
  return cudaSuccess;
}

/** Exact low-rank contraction for generation-validated canonical densities.
 * C is column-major and D=w C C^T. Raw slices keep every metric direction:
 * T_Q=C^T A_Q C, U_P=sum_Q V_PQ T_Q. The metric adjoint is -cK*w^2*T^T*T;
 * Only bounded dense or folded AO-pair panels reach generated derivatives.
 * Packed expansion uses lower rectangular blocks of C U_P C^T, never a dense
 * panel followed by compression. The scalar derivative equations are unchanged.
 * All rank-squared storage borrows already charged resident J/K buffers.
 */
static cudaError_t contract_occupied_response(
    std::size_t n, std::size_t a, std::span<const DensityFittingDensityResponse> terms,
    const double* densities, CudaDfMetricView metric, std::size_t tile, double* workspace,
    cudaStream_t stream, cublasHandle_t blas, const CudaDfResponseBuffers& buffers,
    std::span<const double> raw_host, const CudaDfResponseConsumer& consume, bool packed_pairs,
    std::span<const std::int64_t> auxiliary_shell_offsets, std::size_t ao_block_rows,
    const std::function<void(std::size_t, double*)>& read_values = {},
    bool factorized_exchange = false) {
  const auto matrix = n * n, aa = a * a;
  const auto pair_stride = packed_pairs ? n * (n + 1) / 2 : matrix;
  auto* inverse = workspace;
  auto* bar_inverse = inverse + aa;
  auto* metric_temp = bar_inverse + aa;
  auto* transformed = metric_temp + aa;
  auto* temporary = transformed + aa;
  // Every T projection has been consumed before derivative panels begin.
  // Its borrowed buffer can then hold a fixed-capacity pseudo-density panel,
  // avoiding small panels without allocating or retaining a complete W tensor.
  auto* weights = buffers.exchange_response;
  auto* charges = temporary + 3 * matrix;
  auto* potentials = charges + terms.size() * a;
  auto* raw = buffers.raw_auxiliary_major;
  auto* projected = buffers.exchange_response;
  auto* transformed_projected = buffers.staging_weights;
  const auto* packed_raw = buffers.resident_packed_raw.data;
  const auto ni = static_cast<int>(n), ai = static_cast<int>(a), mi = static_cast<int>(matrix);
  const double one = 1, zero = 0;
  const auto checked = [](cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw CudaDfResponseBlasFailure{status};
  };
  if (!read_values && !buffers.resident_raw.data && !packed_raw) {
    runtime::cuda_trace::TraceRegion upload("raw_value_resident_upload", stream);
    auto error = cudaMemcpyAsync(transformed_projected, raw_host.data(), raw_host.size_bytes(),
                                 cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) return error;
    runtime::cuda_trace::trace_counter("raw_value_upload_bytes", raw_host.size_bytes());
    runtime::cuda_trace::trace_counter("raw_value_bulk_uploads", 1);
  }
  if (!read_values && !buffers.resident_raw.data && !packed_raw) {
    runtime::cuda_trace::TraceRegion transpose("raw_value_resident_transpose", stream);
    // The existing gather supports both NVIDIA and providers without GEAM.
    cuda_df::launch_gather_auxiliary_tile_kernel(blocks(matrix * a), threads, 0, stream, matrix, a,
                                                 0, 0, a, transformed_projected, raw);
  }
  if (!metric.full_rank)
    inverse_kernel<<<blocks(aa), threads, 0, stream>>>(a, metric.inverse_square_root, inverse);
  auto error = cudaMemsetAsync(bar_inverse, 0, aa * sizeof(double), stream);
  if (error != cudaSuccess) return error;
  if (read_values) {
    // Source-driven projection: each raw AO slice feeds every charge and spin
    // before eviction. Keep C^T A_Q C, not A itself, across the auxiliary axis.
    // The third AO temporary is raw; the first is disjoint A*C scratch.
    runtime::cuda_trace::TraceRegion products("streamed_occupied_raw_projection", stream);
    for (std::size_t q = 0; q < a; ++q) {
      read_values(q, raw);
      checked(cublasDgemv(blas, CUBLAS_OP_T, mi, static_cast<int>(terms.size()), &one, densities,
                          mi, raw, 1, &zero, charges + q, ai));
      std::size_t offset = 0;
      for (std::size_t t = 0; t < terms.size(); ++t) {
        const auto& factor = buffers.occupied_factors[t];
        const auto r = factor.rank, rr = r * r;
        if (!r || terms[t].exchange_coefficient == 0) continue;
        const auto ri = static_cast<int>(r);
        checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, ni, ri, ni, &one, raw, ni,
                            factor.coefficients, ni, &zero, temporary, ni));
        checked(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, ri, ri, ni, &one, factor.coefficients,
                            ni, temporary, ni, &zero, transformed_projected + offset + q * rr, ri));
        offset += a * rr;
        runtime::cuda_trace::trace_counter("response_occupied_projection_blas_calls", 2);
        runtime::cuda_trace::trace_counter("response_occupied_projection_products", 2);
      }
    }
    runtime::cuda_trace::trace_counter("response_streamed_occupied_raw_passes", 1);
    runtime::cuda_trace::trace_counter("response_streamed_occupied_charge_gemvs", a);
  }
  {
    runtime::cuda_trace::TraceRegion charge("coulomb_response_charge_dot", stream);
    if (read_values) {
      // Charges were formed during the single raw-source pass above.
    } else if (packed_raw) {
      // Unit-weight raw pairs contract D_mn+D_nm; triangular storage never
      // discards an antisymmetric density component by reading one triangle.
      for (std::size_t t = 0; t < terms.size(); ++t) {
        cuda_df::launch_pack_df_density(stream, n, densities + t * matrix, temporary);
        checked(cublasDgemv(blas, CUBLAS_OP_N, ai,
                            static_cast<int>(buffers.resident_packed_raw.pair_count), &one,
                            packed_raw, ai, temporary, 1, &zero, charges + t * a, 1));
      }
    } else {
      checked(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, ai, static_cast<int>(terms.size()), mi,
                          &one, raw, mi, densities, mi, &zero, charges, ai));
    }
    if (metric.full_rank) {
      // Preserve weak metric directions until after applying the charge
      // projection. Forming X X^T first also destabilizes occupied response.
      const auto ti = static_cast<int>(terms.size());
      checked(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, ai, ti, ai, &one, metric.eigenvectors, ai,
                          charges, ai, &zero, potentials, ai));
      divide_charge_eigenvalues<<<blocks(a * terms.size()), threads, 0, stream>>>(
          a, terms.size(), metric.eigenvalues, potentials);
      checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, ai, ti, ai, &one, metric.eigenvectors, ai,
                          potentials, ai, &zero, charges, ai));
      std::swap(charges, potentials);
      runtime::cuda_trace::trace_counter("response_occupied_charge_inverse_gemms", 2);
    } else {
      potential_kernel<<<blocks(terms.size() * a), threads, 0, stream>>>(a, terms.size(), inverse,
                                                                         charges, potentials);
    }
    for (std::size_t t = 0; t < terms.size(); ++t)
      if (terms[t].coulomb_coefficient != 0)
        coulomb_metric_kernel<<<blocks(aa), threads, 0, stream>>>(
            a, metric.full_rank ? -terms[t].coulomb_coefficient : terms[t].coulomb_coefficient,
            (metric.full_rank ? potentials : charges) + t * a, bar_inverse);
  }
  std::size_t retained = 0;
  for (std::size_t t = 0; t < terms.size(); ++t) {
    const auto& factor = buffers.occupied_factors[t];
    const auto r = factor.rank, rr = r * r;
    if (!r || terms[t].exchange_coefficient == 0) continue;
    const auto ri = static_cast<int>(r), rri = static_cast<int>(rr);
    const double coefficient =
        terms[t].exchange_coefficient * factor.density_scale * factor.density_scale;
    {
      runtime::cuda_trace::TraceRegion products("exchange_response_occupied_products", stream);
      if (read_values) {
        // Keep the existing eigenfactor inverse ordering. Its input and output
        // alternate between these disjoint intervals; previous spin factors
        // remain below retained while the reusable projection is overwritten.
        error = cudaMemcpyAsync(projected, transformed_projected + retained,
                                a * rr * sizeof(double), cudaMemcpyDeviceToDevice, stream);
        if (error != cudaSuccess) return error;
        runtime::cuda_trace::trace_counter("response_streamed_raw_projection_copy_bytes",
                                           a * rr * sizeof(double));
      } else if (buffers.final_occupied_projection) {
        // U is column-major (a*r,n); right multiplication by its exact C
        // yields G_whitened[a,i,j]. The tail is disjoint from G_raw[ij,a].
        // No new O(n*a*r) allocation or raw/retained-subspace approximation.
        auto* whitened = projected + buffers.exchange_capacity() - a * rr;
        checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(a * r), ri, ni, &one,
                            buffers.final_occupied_projection, static_cast<int>(a * r),
                            factor.coefficients, ni, &zero, whitened, static_cast<int>(a * r)));
        // Reuse the existing positive-eigenvector scaling primitive. The
        // owner admits only full rank, so Q sqrt(lambda) Q^T is M^(1/2).
        cuda_df::launch_density_exchange_factor(stream, a, a, metric.eigenvectors,
                                                metric.eigenvalues, metric_temp);
        checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, ai, ai, ai, &one, metric_temp, ai,
                            metric.eigenvectors, ai, &zero, transformed, ai));
        checked(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, rri, ai, ai, &one, whitened, ai,
                            transformed, ai, &zero, projected, rri));
        runtime::cuda_trace::trace_counter("response_final_projection_reused", 1);
        runtime::cuda_trace::trace_counter("response_final_projection_retained_bytes",
                                           n * r * a * sizeof(double));
        runtime::cuda_trace::trace_counter("response_occupied_projection_blas_calls", 3);
        runtime::cuda_trace::trace_counter("response_occupied_projection_products", 3);
        runtime::cuda_trace::trace_counter("response_occupied_projection_flops",
                                           2 * a * n * rr + 2 * a * a * (a + rr));
      } else if (packed_raw) {
        if (n * r <= (buffers.staging_capacity() - retained) / a &&
            2 * rr <= buffers.exchange_capacity() / a &&
            a * r <= static_cast<std::size_t>(std::numeric_limits<int>::max())) {
          // Contract immutable raw A directly, including discarded metric
          // directions. The all-Q U has the same layout as occupied K; a
          // disjoint tail receives C^T A C before conversion to [Q,i,j].
          auto* u = transformed_projected + retained;
          auto* raw_projected = projected + buffers.exchange_capacity() - a * rr;
          cuda_df::launch_project_packed_df(stream, n, a, r, true, packed_raw, factor.coefficients,
                                            u);
          checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(a * r), ri, ni, &one,
                              u, static_cast<int>(a * r), factor.coefficients, ni, &zero,
                              raw_projected, static_cast<int>(a * r)));
          cuda_df::launch_gather_auxiliary_tile_kernel(blocks(a * rr), threads, 0, stream, rr, a, 0,
                                                       0, a, raw_projected, projected);
          runtime::cuda_trace::trace_counter("response_packed_raw_projection", 1);
        } else {
          // Unknown/saturated spin ranks use one exact dense raw slice. Both
          // scratch and the retained preceding spin projections stay bounded.
          for (std::size_t q = 0; q < a; ++q) {
            cuda_df::launch_unpack_df_values(stream, n, a, 0, n, q, 1, true, packed_raw, raw);
            checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, ni, ri, ni, &one, raw, ni,
                                factor.coefficients, ni, &zero, temporary, ni));
            checked(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, ri, ri, ni, &one,
                                factor.coefficients, ni, temporary, ni, &zero, projected + q * rr,
                                ri));
          }
          runtime::cuda_trace::trace_counter("response_packed_raw_projection_slices", a);
        }
      } else if (buffers.batch_products &&
                 n * a <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
                 n * r <= (buffers.staging_capacity() - retained) / a) {
        // Concatenate all raw AO slices: C^T [A_0 ... A_(a-1)] is one
        // larger GEMM. Its (r,n) slices then share C in a batched product.
        // The U destination is free until the following metric contraction;
        // previous spin U factors live strictly below retained.
        auto* projected_once = transformed_projected + retained;
        checked(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, ri, static_cast<int>(n * a), ni, &one,
                            factor.coefficients, ni, raw, ni, &zero, projected_once, ri));
        checked(cublasDgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, ri, ri, ni, &one,
                                          projected_once, ri, n * r, factor.coefficients, ni, 0,
                                          &zero, projected, ri, rr, ai));
        runtime::cuda_trace::trace_counter("response_occupied_projection_blas_calls", 2);
        runtime::cuda_trace::trace_counter("response_occupied_projection_products", a + 1);
        runtime::cuda_trace::trace_counter("response_occupied_projection_temporary_bytes",
                                           n * r * a * sizeof(double));
      } else
        for (std::size_t q = 0; q < a; ++q) {
          checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, ni, ri, ni, &one, raw + q * matrix,
                              ni, factor.coefficients, ni, &zero, temporary, ni));
          checked(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, ri, ri, ni, &one, factor.coefficients,
                              ni, temporary, ni, &zero, projected + q * rr, ri));
          runtime::cuda_trace::trace_counter("response_occupied_projection_blas_calls", 2);
          runtime::cuda_trace::trace_counter("response_occupied_projection_products", 2);
        }
      if (!buffers.final_occupied_projection) {
        runtime::cuda_trace::trace_counter("response_final_projection_reconstructed", 1);
        runtime::cuda_trace::trace_counter("response_occupied_projection_flops",
                                           2 * a * (n * n * r + n * rr));
      }
    }
    {
      runtime::cuda_trace::TraceRegion project("exchange_response_occupied_weight_gemm", stream);
      if (metric.full_rank) {
        auto* fitted = transformed_projected + retained;
        checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, rri, ai, ai, &one, projected, rri,
                            metric.eigenvectors, ai, &zero, fitted, rri));
        divide_factor_eigenvalues<<<blocks(rr * a), threads, 0, stream>>>(rr, a, metric.eigenvalues,
                                                                          fitted);
        checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, rri, ai, ai, &one, fitted, rri,
                            metric.eigenvectors, ai, &zero, projected, rri));
        // Retain each spin's final factors in the original disjoint staging
        // interval. The next spin reuses projected; it cannot overwrite them.
        error = cudaMemcpyAsync(fitted, projected, rr * a * sizeof(double),
                                cudaMemcpyDeviceToDevice, stream);
        if (error != cudaSuccess) return error;
        runtime::cuda_trace::trace_counter("response_occupied_inverse_gemms", 2);
        runtime::cuda_trace::trace_counter("response_occupied_inverse_flops", 4 * rr * aa);
        runtime::cuda_trace::trace_counter("response_occupied_factor_copy_bytes",
                                           rr * a * sizeof(double));
      } else {
        checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, rri, ai, ai, &one, projected, rri,
                            inverse, ai, &zero, transformed_projected + retained, rri));
      }
    }
    {
      runtime::cuda_trace::TraceRegion dot("exchange_response_occupied_metric_gemm", stream);
      // Full rank permits the Gram of U=M^-1 T directly. A truncated metric
      // still needs raw T in its spectral map to retain subspace motion.
      const double alpha = metric.full_rank ? coefficient : -coefficient;
      const auto* metric_factors = metric.full_rank ? transformed_projected + retained : projected;
      checked(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, ai, ai, rri, &alpha, metric_factors, rri,
                          metric_factors, rri, &one, bar_inverse, ai));
    }
    retained += a * rr;
    runtime::cuda_trace::trace_counter("response_occupied_rank", r);
  }
  runtime::cuda_trace::trace_counter("response_occupied_projected_elements", retained);
  runtime::cuda_trace::trace_counter("response_occupied_projected_bytes",
                                     retained * sizeof(double));
  runtime::cuda_trace::trace_counter("response_packed_pairs", packed_pairs);
  runtime::cuda_trace::trace_counter("response_pseudo_density_capacity_elements",
                                     tile * pair_stride);
  runtime::cuda_trace::trace_counter("response_dense_weight_panel_elements",
                                     packed_pairs ? 0 : tile * matrix);
  std::size_t peak_count = 0, block_peak_elements = 0;
  for (std::size_t begin = 0; begin < a;) {
    auto count = std::min(tile, a - begin);
    if (packed_pairs && !auxiliary_shell_offsets.empty()) {
      // Keep the byte cap while avoiding repeated primitive work for a shell
      // cut by an arbitrary auxiliary AO boundary. A shell wider than the cap
      // remains split and is still handled exactly by the derivative consumer.
      auto end = std::upper_bound(auxiliary_shell_offsets.begin(), auxiliary_shell_offsets.end(),
                                  begin + count);
      if (end != auxiliary_shell_offsets.begin() && static_cast<std::size_t>(*--end) > begin)
        count = *end - begin;
    }
    peak_count = std::max(peak_count, count);
    error = cudaMemsetAsync(weights, 0, count * pair_stride * sizeof(double), stream);
    if (error != cudaSuccess) return error;
    CudaDfResponsePanel fused_panel{};
    std::size_t offset = 0;
    for (std::size_t t = 0; t < terms.size(); ++t) {
      if (terms[t].coulomb_coefficient != 0) {
        if (packed_pairs)
          packed_coulomb_weights<<<blocks(count * pair_stride), threads, 0, stream>>>(
              n, begin, count, terms[t].coulomb_coefficient, densities + t * matrix,
              potentials + t * a, weights);
        else
          coulomb_weights_kernel<<<blocks(count * matrix), threads, 0, stream>>>(
              matrix, a, begin, count, terms[t].coulomb_coefficient, densities + t * matrix,
              potentials + t * a, weights);
      }
      const auto& factor = buffers.occupied_factors[t];
      const auto r = factor.rank, rr = r * r;
      if (!r || terms[t].exchange_coefficient == 0) continue;
      const auto ri = static_cast<int>(r);
      const double alpha =
          -2 * terms[t].exchange_coefficient * factor.density_scale * factor.density_scale;
      runtime::cuda_trace::TraceRegion expand("exchange_response_pseudo_density_products", stream);
      const auto per_panel = pair_stride + n * r + n * std::min(n, ao_block_rows);
      if (buffers.batch_products &&
          r * count <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
          per_panel <= buffers.exchange_capacity() / count) {
        // Projected T is dead after the metric/weight GEMMs. Its former
        // buffer now holds disjoint bounded W, CU and rectangular block
        // panels. Capacity is checked together, including both live spins.
        // Full rectangular output keeps its original orientation; packed
        // output folds only the explicitly symmetrized low-rank adjoint.
        auto* u = transformed_projected + offset + begin * rr;
        auto* cu = weights + count * pair_stride;
        auto* block = cu + count * n * r;
        if (packed_pairs)
          symmetrize_kernel<<<blocks(count * rr), threads, 0, stream>>>(r, u, count);
        checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, ni, static_cast<int>(r * count), ri,
                            &one, factor.coefficients, ni, u, ri, &zero, cu, ni));
        runtime::cuda_trace::trace_counter("response_pseudo_density_products", 1);
        runtime::cuda_trace::trace_counter("response_pseudo_density_blas_calls", 1);
        runtime::cuda_trace::trace_counter("response_pseudo_density_flops", 2 * count * n * rr);
        if (factorized_exchange && packed_pairs && terms.size() == 1) {
          // Candidate A for #419: retain the exact C*U factor until the shell
          // consumer. The materialized packed panel still carries Coulomb; the
          // exchange contribution is reconstructed by an exact rank dot for
          // each public AO pair that the derivative kernel actually consumes.
          fused_panel = {2,
                         {begin, pair_stride, 1, a},
                         count * pair_stride,
                         weights,
                         factor.coefficients,
                         cu,
                         r,
                         alpha};
          runtime::cuda_trace::trace_counter("response_factorized_exchange_panels", 1);
          runtime::cuda_trace::trace_counter("response_factorized_exchange_rank", r);
          runtime::cuda_trace::trace_counter("response_factorized_exchange_projected_elements",
                                             count * n * r);
          offset += a * rr;
          continue;
        }
        if (packed_pairs) {
          for (std::size_t row = 0; row < n; row += ao_block_rows) {
            const auto rows = std::min(ao_block_rows, n - row), columns = row + rows;
            block_peak_elements = std::max(block_peak_elements, count * rows * columns);
            checked(cublasDgemmStridedBatched(
                blas, CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(columns), static_cast<int>(rows),
                ri, &alpha, factor.coefficients, ni, 0, cu + row, ni, n * r, &zero, block,
                static_cast<int>(columns), rows * columns, static_cast<int>(count)));
            add_packed_exchange_block<<<blocks(count * rows * columns), threads, 0, stream>>>(
                row, rows, columns, block, weights, count, pair_stride);
            runtime::cuda_trace::trace_counter("response_pseudo_density_products", count);
            runtime::cuda_trace::trace_counter("response_pseudo_density_blas_calls", 1);
            runtime::cuda_trace::trace_counter("response_pseudo_density_flops",
                                               2 * count * rows * columns * r);
            runtime::cuda_trace::trace_counter("response_pseudo_density_rectangular_elements",
                                               count * rows * columns);
          }
        } else {
          checked(cublasDgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_T, ni, ni, ri, &alpha, cu,
                                            ni, n * r, factor.coefficients, ni, 0, &one, weights,
                                            ni, matrix, static_cast<int>(count)));
          runtime::cuda_trace::trace_counter("response_pseudo_density_products", count);
          runtime::cuda_trace::trace_counter("response_pseudo_density_blas_calls", 1);
          runtime::cuda_trace::trace_counter("response_pseudo_density_flops",
                                             2 * count * n * n * r);
          runtime::cuda_trace::trace_counter("response_pseudo_density_rectangular_elements",
                                             count * matrix);
        }
        runtime::cuda_trace::trace_counter("response_pseudo_density_batched_panels", 1);
        offset += a * rr;
        continue;
      }
      for (std::size_t p = 0; p < count; ++p) {
        auto* u = transformed_projected + offset + (begin + p) * rr;
        if (packed_pairs) symmetrize_kernel<<<blocks(rr), threads, 0, stream>>>(r, u);
        checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, ni, ri, ri, &one, factor.coefficients,
                            ni, u, ri, &zero, temporary, ni));
        runtime::cuda_trace::trace_counter("response_pseudo_density_products", 1);
        runtime::cuda_trace::trace_counter("response_pseudo_density_flops", 2 * n * rr);
        if (packed_pairs) {
          // temporary holds C U (n*rank); the following existing AO scratch
          // holds at most n*ao_block_rows entries, bounded by its existing
          // n*n capacity, and is reused before the next block.
          // Computing rows [i,i+b) against columns [0,i+b) avoids every upper
          // off-diagonal block, with only b*(b-1)/2 extra diagonal entries.
          auto* block = temporary + matrix;
          for (std::size_t row = 0; row < n; row += ao_block_rows) {
            const auto rows = std::min(ao_block_rows, n - row), columns = row + rows;
            block_peak_elements = std::max(block_peak_elements, rows * columns);
            checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(columns),
                                static_cast<int>(rows), ri, &alpha, factor.coefficients, ni,
                                temporary + row, ni, &zero, block, static_cast<int>(columns)));
            add_packed_exchange_block<<<blocks(rows * columns), threads, 0, stream>>>(
                row, rows, columns, block, weights + p * pair_stride);
            runtime::cuda_trace::trace_counter("response_pseudo_density_products", 1);
            runtime::cuda_trace::trace_counter("response_pseudo_density_flops",
                                               2 * rows * columns * r);
            runtime::cuda_trace::trace_counter("response_pseudo_density_rectangular_elements",
                                               rows * columns);
          }
        } else {
          checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, ni, ni, ri, &alpha, temporary, ni,
                              factor.coefficients, ni, &one, weights + p * matrix, ni));
          runtime::cuda_trace::trace_counter("response_pseudo_density_products", 1);
          runtime::cuda_trace::trace_counter("response_pseudo_density_flops", 2 * n * n * r);
          runtime::cuda_trace::trace_counter("response_pseudo_density_rectangular_elements",
                                             matrix);
        }
      }
      offset += a * rr;
    }
    runtime::cuda_trace::trace_counter("response_auxiliary_blocks", 1);
    if (fused_panel.factorized_exchange())
      consume(fused_panel);
    else
      consume({packed_pairs ? 2U : 0U, {begin, pair_stride, 1, a}, count * pair_stride, weights});
    begin += count;
  }
  runtime::cuda_trace::trace_counter("response_pseudo_density_peak_elements",
                                     peak_count * pair_stride);
  runtime::cuda_trace::trace_counter("response_pseudo_density_peak_bytes",
                                     peak_count * pair_stride * sizeof(double));
  runtime::cuda_trace::trace_counter("response_pseudo_density_block_peak_elements",
                                     block_peak_elements);
  runtime::cuda_trace::trace_counter("response_pseudo_density_block_rows",
                                     packed_pairs ? ao_block_rows : 0);
  // Small explicit domains can fit in one panel; report literal full-domain
  // storage separately for dense and packed weights instead of hiding it.
  runtime::cuda_trace::trace_counter("response_full_weight_tensor_elements",
                                     !packed_pairs && peak_count == a ? matrix * a : 0);
  runtime::cuda_trace::trace_counter("response_full_packed_weight_tensor_elements",
                                     packed_pairs && peak_count == a ? pair_stride * a : 0);
  runtime::cuda_trace::TraceRegion reverse("metric_frechet_response", stream);
  if (!metric.full_rank) {
    tensor::launch_symmetric_pseudoinverse_vjp(a, metric.eigenvectors, metric.eigenvalues,
                                               metric.relative_threshold, bar_inverse, metric_temp,
                                               transformed, bar_inverse, stream);
  } else {
    runtime::cuda_trace::trace_counter("response_full_rank_occupied_factor_first", 1);
    if (read_values)
      runtime::cuda_trace::trace_counter("response_streamed_occupied_factor_first", 1);
    symmetrize_kernel<<<blocks(aa), threads, 0, stream>>>(a, bar_inverse);
  }
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  reverse.finish();
  consume({1, {}, aa, bar_inverse});
  return cudaSuccess;
}

cudaError_t contract_cuda_df_response_weights(
    std::size_t n, std::size_t a, std::span<const DensityFittingDensityResponse> terms,
    const double* densities, CudaDfMetricView metric, std::size_t tile, double* workspace,
    cudaStream_t stream, cublasHandle_t blas, bool serial_metric_dot, bool blas_products,
    const std::function<void(std::size_t, double*)>& read_values,
    const CudaDfResponseConsumer& consume, const CudaDfResponseBuffers* borrowed,
    std::span<const double> raw_host, bool packed_pairs,
    std::span<const std::int64_t> auxiliary_shell_offsets, std::size_t packed_block_rows,
    const std::function<void(std::size_t, std::size_t, double*)>& read_fitted,
    bool single_fitted_tensor, const CudaDfResponseBuffers* streamed_occupied,
    bool factorized_exchange) {
  if (streamed_occupied) {
    if (!metric.full_rank || borrowed || read_fitted || single_fitted_tensor || !read_values ||
        !streamed_occupied->occupied_response || (packed_pairs && !packed_block_rows) ||
        (factorized_exchange && (!packed_pairs || terms.size() != 1)))
      return cudaErrorInvalidValue;
    return contract_occupied_response(n, a, terms, densities, metric, tile, workspace, stream, blas,
                                      *streamed_occupied, {}, consume, packed_pairs,
                                      auxiliary_shell_offsets, packed_block_rows, read_values,
                                      factorized_exchange);
  }
  if (single_fitted_tensor) {
    if (!metric.full_rank || borrowed || read_fitted || packed_pairs) return cudaErrorInvalidValue;
    double* fitted = nullptr;
    const auto error = fit_full_tensor_in_place(n, a, terms.size(), metric, workspace, stream, blas,
                                                read_values, fitted);
    if (error != cudaSuccess) return error;
    return contract_full_rank_panels(n, a, terms, densities, tile, workspace, stream, blas,
                                     serial_metric_dot, blas_products, {}, consume, fitted);
  }
  if (packed_pairs && (!borrowed || !borrowed->occupied_response || !packed_block_rows))
    return cudaErrorInvalidValue;
  if (borrowed && borrowed->occupied_response)
    return contract_occupied_response(
        n, a, terms, densities, metric, tile, workspace, stream, blas, *borrowed, raw_host, consume,
        packed_pairs, auxiliary_shell_offsets, packed_block_rows, {}, factorized_exchange);
  // A validated resident fitted reader takes precedence even for a full-width
  // panel. Capacity is not permission to discard immutable forward values and
  // regenerate raw integrals. Borrowed/raw and rank-truncated routes stay below.
  if (metric.full_rank && !borrowed && read_fitted)
    return contract_full_rank_panels(n, a, terms, densities, tile, workspace, stream, blas,
                                     serial_metric_dot, blas_products, read_fitted, consume);
  if (metric.full_rank && (borrowed || tile == a))
    return contract_full_rank_response(n, a, terms, densities, metric, tile, workspace, stream,
                                       blas, serial_metric_dot, blas_products, read_values, consume,
                                       borrowed, raw_host);
  if (borrowed)
    return contract_resident_response(n, a, terms, densities, metric, tile, workspace, stream, blas,
                                      *borrowed, raw_host, consume);
  const auto matrix = n * n, aa = a * a;
  auto* inverse = workspace;
  auto* bar_inverse = inverse + aa;
  auto* metric_temp = bar_inverse + aa;
  auto* transformed = metric_temp + aa;
  auto* values = transformed + aa;
  auto* temporary = values + matrix;
  auto* response = temporary + matrix;
  auto* raw = response + matrix;
  auto* weights = raw + tile * matrix;
  auto* charges = weights + tile * matrix;
  auto* potentials = charges + terms.size() * a;
  runtime::cuda_trace::TraceRegion metric_inverse("metric_inverse", stream);
  inverse_kernel<<<blocks(aa), threads, 0, stream>>>(a, metric.inverse_square_root, inverse);
  metric_inverse.finish();
  runtime::cuda_trace::TraceRegion coulomb("coulomb_response", stream);
  auto error = cudaMemsetAsync(bar_inverse, 0, aa * sizeof(double), stream);
  if (error != cudaSuccess) return error;
  for (std::size_t q = 0; q < a; ++q) {
    // The first response panel is already reserved in this workspace. Fill
    // it while forming charges, then retain those exact raw values for both
    // exchange weights and metric response. Discarded metric directions make
    // reconstructing raw A from a retained transformed tensor unsafe here.
    auto* charge_values = q < tile ? raw + q * matrix : values;
    read_values(q, charge_values);
    runtime::cuda_trace::TraceRegion charge_dot("coulomb_response_charge_dot", stream);
    if (blas_products) {
      // Packed D[t,ij] is column-major [ij,t]. Keep each spin/total charge
      // in its existing auxiliary-strided destination without a host scalar.
      const double alpha = 1.0, beta = 0.0;
      const auto status =
          cublasDgemv(blas, CUBLAS_OP_T, static_cast<int>(matrix), static_cast<int>(terms.size()),
                      &alpha, densities, static_cast<int>(matrix), charge_values, 1, &beta,
                      charges + q, static_cast<int>(a));
      if (status != CUBLAS_STATUS_SUCCESS) throw CudaDfResponseBlasFailure{status};
      runtime::cuda_trace::trace_counter("response_charge_blas_dots", terms.size());
    } else {
      charge_kernel<<<blocks(terms.size()), threads, 0, stream>>>(
          matrix, a, q, terms.size(), densities, charge_values, charges);
      runtime::cuda_trace::trace_counter("response_charge_scalar_dots", terms.size());
    }
    runtime::cuda_trace::trace_counter("response_charge_dot_elements", terms.size() * matrix);
  }
  potential_kernel<<<blocks(terms.size() * a), threads, 0, stream>>>(a, terms.size(), inverse,
                                                                     charges, potentials);
  for (std::size_t t = 0; t < terms.size(); ++t)
    if (terms[t].coulomb_coefficient != 0)
      coulomb_metric_kernel<<<blocks(aa), threads, 0, stream>>>(a, terms[t].coulomb_coefficient,
                                                                charges + t * a, bar_inverse);
  coulomb.finish();
  for (std::size_t begin = 0; begin < a; begin += tile) {
    runtime::cuda_trace::trace_counter("response_auxiliary_blocks", 1);
    const auto count = std::min(tile, a - begin);
    error = cudaMemsetAsync(weights, 0, count * matrix * sizeof(double), stream);
    if (error != cudaSuccess) return error;
    if (begin != 0) {
      for (std::size_t p = 0; p < count; ++p) read_values(begin + p, raw + p * matrix);
    } else {
      runtime::cuda_trace::trace_counter("raw_value_cache_hits", count);
      runtime::cuda_trace::trace_counter("raw_value_reuse_bytes", count * matrix * sizeof(double));
    }
    runtime::cuda_trace::TraceRegion coulomb_weights("coulomb_response_weights", stream);
    for (std::size_t t = 0; t < terms.size(); ++t)
      if (terms[t].coulomb_coefficient != 0)
        coulomb_weights_kernel<<<blocks(count * matrix), threads, 0, stream>>>(
            matrix, a, begin, count, terms[t].coulomb_coefficient, densities + t * matrix,
            potentials + t * a, weights);
    coulomb_weights.finish();
    for (std::size_t q = 0; q < a; ++q) {
      const double* exchange_values = values;
      if (q >= begin && q - begin < count) {
        // A later raw panel overwrites this storage only after consume() has
        // enqueued the preceding derivative contraction on the same stream.
        exchange_values = raw + (q - begin) * matrix;
        runtime::cuda_trace::trace_counter("raw_value_cache_hits", 1);
        runtime::cuda_trace::trace_counter("raw_value_reuse_bytes", matrix * sizeof(double));
      } else {
        read_values(q, values);
      }
      for (std::size_t t = 0; t < terms.size(); ++t) {
        const double coefficient = terms[t].exchange_coefficient;
        if (coefficient == 0) continue;
        runtime::cuda_trace::TraceRegion products("exchange_response_matrix_products", stream);
        runtime::cuda_trace::trace_counter("response_ao_matrix_products", 2);
        if (blas_products) {
          // Row-major T=A_Q D and R=D^T T become T^T=D^T A_Q^T and
          // R^T=T^T D. Explicit transpose flags preserve the original index
          // contract without depending on exact floating-point symmetry.
          const auto dimension = static_cast<int>(n);
          const double alpha = 1.0, beta = 0.0;
          auto status = cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, dimension, dimension, dimension,
                                    &alpha, densities + t * matrix, dimension, exchange_values,
                                    dimension, &beta, temporary, dimension);
          if (status != CUBLAS_STATUS_SUCCESS) throw CudaDfResponseBlasFailure{status};
          status = cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T, dimension, dimension, dimension,
                               &alpha, temporary, dimension, densities + t * matrix, dimension,
                               &beta, response, dimension);
          if (status != CUBLAS_STATUS_SUCCESS) throw CudaDfResponseBlasFailure{status};
          runtime::cuda_trace::trace_counter("response_density_blas_products", 2);
        } else {
          right_density_kernel<<<blocks(matrix), threads, 0, stream>>>(
              n, exchange_values, densities + t * matrix, temporary);
          left_density_kernel<<<blocks(matrix), threads, 0, stream>>>(n, densities + t * matrix,
                                                                      temporary, response);
          runtime::cuda_trace::trace_counter("response_density_scalar_products", 2);
        }
        products.finish();
        runtime::cuda_trace::TraceRegion contractions("exchange_response_weights_and_metric",
                                                      stream);
        exchange_weights_kernel<<<blocks(count * matrix), threads, 0, stream>>>(
            matrix, a, begin, count, q, coefficient, inverse, response, weights);
        runtime::cuda_trace::TraceRegion metric_dot("exchange_response_metric_dot", stream);
        if (serial_metric_dot) {
          runtime::cuda_trace::trace_counter("response_metric_serial_dots", count);
          exchange_metric_kernel<<<blocks(count), threads, 0, stream>>>(
              matrix, a, begin, count, q, coefficient, raw, response, bar_inverse);
        } else {
          // raw[P,ij] is also column-major [ij,P]. Its transpose contracts
          // every retained P against R_Q without a serial AO-pair loop per P.
          // The output stride writes bar_inverse[P,Q] in its existing layout;
          // beta=1 preserves Coulomb and previous spin contributions.
          const double alpha = -coefficient, beta = 1.0;
          const auto status =
              cublasDgemv(blas, CUBLAS_OP_T, static_cast<int>(matrix), static_cast<int>(count),
                          &alpha, raw, static_cast<int>(matrix), response, 1, &beta,
                          bar_inverse + begin * a + q, static_cast<int>(a));
          if (status != CUBLAS_STATUS_SUCCESS) throw CudaDfResponseBlasFailure{status};
          runtime::cuda_trace::trace_counter("response_metric_blas_dots", count);
        }
      }
    }
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    consume({0, {begin, matrix, 1, a}, count * matrix, weights});
  }
  runtime::cuda_trace::TraceRegion metric_response("metric_frechet_response", stream);
  tensor::launch_symmetric_pseudoinverse_vjp(a, metric.eigenvectors, metric.eigenvalues,
                                             metric.relative_threshold, bar_inverse, metric_temp,
                                             transformed, bar_inverse, stream);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  metric_response.finish();
  consume({1, {}, aa, bar_inverse});
  return cudaSuccess;
}
}  // namespace vibeqc::scf
