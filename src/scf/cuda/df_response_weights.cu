#include <algorithm>
#include <limits>

#include "runtime/cuda_component_trace.hpp"
#include "scf/cuda/df_jk_kernels.hpp"
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

__global__ void coulomb_weights_kernel(std::size_t matrix, std::size_t a, std::size_t begin,
                                       std::size_t count, double coefficient, const double* density,
                                       const double* potential, double* weights) {
  const auto pi = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (pi >= count * matrix) return;
  weights[pi] += coefficient * density[pi % matrix] * potential[begin + pi / matrix];
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

__global__ void coulomb_metric_kernel(std::size_t a, double coefficient, const double* charges,
                                      double* bar_inverse) {
  const auto pq = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (pq < a * a) bar_inverse[pq] += .5 * coefficient * charges[pq / a] * charges[pq % a];
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

__global__ void exchange_weights_kernel(std::size_t matrix, std::size_t a, std::size_t begin,
                                        std::size_t count, std::size_t q, double coefficient,
                                        const double* inverse, const double* response,
                                        double* weights) {
  const auto pi = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (pi >= count * matrix) return;
  weights[pi] += -2 * coefficient * inverse[(begin + pi / matrix) * a + q] * response[pi % matrix];
}

__global__ void exchange_metric_kernel(std::size_t matrix, std::size_t a, std::size_t begin,
                                       std::size_t count, std::size_t q, double coefficient,
                                       const double* raw, const double* response,
                                       double* bar_inverse) {
  const auto p = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (p >= count) return;
  double value = 0;
  for (std::size_t ij = 0; ij < matrix; ++ij) value += raw[p * matrix + ij] * response[ij];
  bar_inverse[(begin + p) * a + q] -= coefficient * value;
}

/** Eigenvectors are cuSOLVER column-major; all response matrices are row-major.
 * Stages compute sym(E) Q, Q^T temp with the divided differences, Q Ehat,
 * then temp Q^T. These include finite discarded eigenvalues, which a simple
 * -M+ E M+ formula would omit and give incorrect rank-deficient forces.
 */
__global__ void metric_response_kernel(std::size_t a, unsigned stage, CudaDfMetricView metric,
                                       const double* input, double* output) {
  const auto ij = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (ij >= a * a) return;
  const auto i = ij / a, j = ij % a;
  const auto* q = metric.eigenvectors;
  double value = 0;
  for (std::size_t k = 0; k < a; ++k) {
    if (stage == 0) value += .5 * (input[i * a + k] + input[k * a + i]) * q[k + j * a];
    if (stage == 1) value += q[k + i * a] * input[k * a + j];
    if (stage == 2) value += q[i + k * a] * input[k * a + j];
    if (stage == 3) value += input[i * a + k] * q[j + k * a];
  }
  if (stage == 1) {
    const double li = metric.eigenvalues[i], lj = metric.eigenvalues[j];
    const double cutoff = metric.relative_threshold * metric.eigenvalues[a - 1];
    const bool ki = li > cutoff, kj = lj > cutoff;
    double divided = 0;
    if (ki && kj)
      divided = -1 / (li * lj);
    else if (ki != kj)
      divided = ((ki ? 1 / li : 0) - (kj ? 1 / lj : 0)) / (li - lj);
    value *= divided;
  }
  output[ij] = value;
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
    std::span<const double> raw_host,
    const std::function<void(unsigned, runtime::StridedRange, std::size_t, const double*)>&
        consume) {
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
    consume(0, {begin, matrix, 1, a}, count * matrix, weights + begin * matrix);
  }
  runtime::cuda_trace::TraceRegion reverse("metric_frechet_response", stream);
  metric_response_kernel<<<blocks(aa), threads, 0, stream>>>(a, 0, metric, bar_inverse,
                                                             metric_temp);
  metric_response_kernel<<<blocks(aa), threads, 0, stream>>>(a, 1, metric, metric_temp,
                                                             transformed);
  metric_response_kernel<<<blocks(aa), threads, 0, stream>>>(a, 2, metric, transformed,
                                                             metric_temp);
  metric_response_kernel<<<blocks(aa), threads, 0, stream>>>(a, 3, metric, metric_temp,
                                                             bar_inverse);
  symmetrize_kernel<<<blocks(aa), threads, 0, stream>>>(a, bar_inverse);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  reverse.finish();
  consume(1, {}, aa, bar_inverse);
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
    std::span<const double> raw_host,
    const std::function<void(unsigned, runtime::StridedRange, std::size_t, const double*)>& consume,
    bool packed_pairs, std::span<const std::int64_t> auxiliary_shell_offsets,
    std::size_t ao_block_rows) {
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
  const auto ni = static_cast<int>(n), ai = static_cast<int>(a), mi = static_cast<int>(matrix);
  const double one = 1, zero = 0;
  const auto checked = [](cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) throw CudaDfResponseBlasFailure{status};
  };
  if (!buffers.resident_raw.data) {
    runtime::cuda_trace::TraceRegion upload("raw_value_resident_upload", stream);
    auto error = cudaMemcpyAsync(transformed_projected, raw_host.data(), raw_host.size_bytes(),
                                 cudaMemcpyHostToDevice, stream);
    if (error != cudaSuccess) return error;
    runtime::cuda_trace::trace_counter("raw_value_upload_bytes", raw_host.size_bytes());
    runtime::cuda_trace::trace_counter("raw_value_bulk_uploads", 1);
  }
  if (!buffers.resident_raw.data) {
    runtime::cuda_trace::TraceRegion transpose("raw_value_resident_transpose", stream);
    // The existing gather supports both NVIDIA and providers without GEAM.
    cuda_df::launch_gather_auxiliary_tile_kernel(blocks(matrix * a), threads, 0, stream, matrix, a,
                                                 0, 0, a, transformed_projected, raw);
  }
  inverse_kernel<<<blocks(aa), threads, 0, stream>>>(a, metric.inverse_square_root, inverse);
  auto error = cudaMemsetAsync(bar_inverse, 0, aa * sizeof(double), stream);
  if (error != cudaSuccess) return error;
  {
    runtime::cuda_trace::TraceRegion charge("coulomb_response_charge_dot", stream);
    checked(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, ai, static_cast<int>(terms.size()), mi,
                        &one, raw, mi, densities, mi, &zero, charges, ai));
    potential_kernel<<<blocks(terms.size() * a), threads, 0, stream>>>(a, terms.size(), inverse,
                                                                       charges, potentials);
    for (std::size_t t = 0; t < terms.size(); ++t)
      if (terms[t].coulomb_coefficient != 0)
        coulomb_metric_kernel<<<blocks(aa), threads, 0, stream>>>(a, terms[t].coulomb_coefficient,
                                                                  charges + t * a, bar_inverse);
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
      if (buffers.final_occupied_projection) {
        // U is column-major (a*r,n); right multiplication by its exact C
        // yields G_whitened[a,i,j]. The tail is disjoint from G_raw[ij,a].
        // No new O(n*a*r) allocation or raw/retained-subspace approximation.
        auto* whitened = projected + buffers.elements_per_buffer - a * rr;
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
      } else if (buffers.batch_products &&
                 n * a <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
                 n * r <= (buffers.elements_per_buffer - retained) / a) {
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
      if (!buffers.final_occupied_projection)
        runtime::cuda_trace::trace_counter("response_occupied_projection_flops",
                                           2 * a * (n * n * r + n * rr));
    }
    {
      runtime::cuda_trace::TraceRegion dot("exchange_response_occupied_metric_gemm", stream);
      const double alpha = -coefficient;
      checked(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, ai, ai, rri, &alpha, projected, rri,
                          projected, rri, &one, bar_inverse, ai));
    }
    {
      runtime::cuda_trace::TraceRegion project("exchange_response_occupied_weight_gemm", stream);
      checked(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, rri, ai, ai, &one, projected, rri,
                          inverse, ai, &zero, transformed_projected + retained, rri));
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
                                     packed_pairs ? 0 : a * matrix);
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
          per_panel <= buffers.elements_per_buffer / count) {
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
    consume(packed_pairs ? 2 : 0, {begin, pair_stride, 1, a}, count * pair_stride, weights);
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
  metric_response_kernel<<<blocks(aa), threads, 0, stream>>>(a, 0, metric, bar_inverse,
                                                             metric_temp);
  metric_response_kernel<<<blocks(aa), threads, 0, stream>>>(a, 1, metric, metric_temp,
                                                             transformed);
  metric_response_kernel<<<blocks(aa), threads, 0, stream>>>(a, 2, metric, transformed,
                                                             metric_temp);
  metric_response_kernel<<<blocks(aa), threads, 0, stream>>>(a, 3, metric, metric_temp,
                                                             bar_inverse);
  symmetrize_kernel<<<blocks(aa), threads, 0, stream>>>(a, bar_inverse);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  reverse.finish();
  consume(1, {}, aa, bar_inverse);
  return cudaSuccess;
}

cudaError_t contract_cuda_df_response_weights(
    std::size_t n, std::size_t a, std::span<const DensityFittingDensityResponse> terms,
    const double* densities, CudaDfMetricView metric, std::size_t tile, double* workspace,
    cudaStream_t stream, cublasHandle_t blas, bool serial_metric_dot, bool blas_products,
    const std::function<void(std::size_t, double*)>& read_values,
    const std::function<void(unsigned, runtime::StridedRange, std::size_t, const double*)>& consume,
    const CudaDfResponseBuffers* borrowed, std::span<const double> raw_host, bool packed_pairs,
    std::span<const std::int64_t> auxiliary_shell_offsets, std::size_t packed_block_rows) {
  if (packed_pairs && (!borrowed || !borrowed->occupied_response || !packed_block_rows))
    return cudaErrorInvalidValue;
  if (borrowed && borrowed->occupied_response)
    return contract_occupied_response(n, a, terms, densities, metric, tile, workspace, stream, blas,
                                      *borrowed, raw_host, consume, packed_pairs,
                                      auxiliary_shell_offsets, packed_block_rows);
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
    consume(0, {begin, matrix, 1, a}, count * matrix, weights);
  }
  runtime::cuda_trace::TraceRegion metric_response("metric_frechet_response", stream);
  metric_response_kernel<<<blocks(aa), threads, 0, stream>>>(a, 0, metric, bar_inverse,
                                                             metric_temp);
  metric_response_kernel<<<blocks(aa), threads, 0, stream>>>(a, 1, metric, metric_temp,
                                                             transformed);
  metric_response_kernel<<<blocks(aa), threads, 0, stream>>>(a, 2, metric, transformed,
                                                             metric_temp);
  metric_response_kernel<<<blocks(aa), threads, 0, stream>>>(a, 3, metric, metric_temp,
                                                             bar_inverse);
  symmetrize_kernel<<<blocks(aa), threads, 0, stream>>>(a, bar_inverse);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  metric_response.finish();
  consume(1, {}, aa, bar_inverse);
  return cudaSuccess;
}
}  // namespace vibeqc::scf
