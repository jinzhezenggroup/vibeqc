#include <algorithm>

#include "runtime/cuda_component_trace.hpp"
#include "scf/cuda/df_response_weights.cuh"

namespace vibeqc::scf {
namespace {
constexpr unsigned threads = 128;
unsigned blocks(std::size_t size) { return static_cast<unsigned>((size + threads - 1) / threads); }

/** All small contractions keep a fixed summation order per output element.
 * They intentionally avoid atomics and unbounded GEMM workspace. Integral
 * generation and the derivative consumer remain the existing #142/#143 paths.
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

__global__ void symmetrize_kernel(std::size_t a, double* values) {
  const auto ij = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (ij >= a * a || ij / a >= ij % a) return;
  const auto ji = (ij % a) * a + ij / a;
  values[ij] = values[ji] = .5 * (values[ij] + values[ji]);
}
}  // namespace

std::size_t cuda_df_response_workspace_elements(std::size_t n, std::size_t a, std::size_t terms,
                                                std::size_t tile) {
  return 4 * a * a + (3 + 2 * tile) * n * n + 2 * terms * a;
}

cudaError_t contract_cuda_df_response_weights(
    std::size_t n, std::size_t a, std::span<const DensityFittingDensityResponse> terms,
    const double* densities, CudaDfMetricView metric, std::size_t tile, double* workspace,
    cudaStream_t stream, const std::function<void(std::size_t, double*)>& read_values,
    const std::function<void(unsigned, runtime::StridedRange, std::size_t, const double*)>&
        consume) {
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
    read_values(q, values);
    charge_kernel<<<blocks(terms.size()), threads, 0, stream>>>(matrix, a, q, terms.size(),
                                                                densities, values, charges);
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
    for (std::size_t p = 0; p < count; ++p) read_values(begin + p, raw + p * matrix);
    runtime::cuda_trace::TraceRegion coulomb_weights("coulomb_response_weights", stream);
    for (std::size_t t = 0; t < terms.size(); ++t)
      if (terms[t].coulomb_coefficient != 0)
        coulomb_weights_kernel<<<blocks(count * matrix), threads, 0, stream>>>(
            matrix, a, begin, count, terms[t].coulomb_coefficient, densities + t * matrix,
            potentials + t * a, weights);
    coulomb_weights.finish();
    for (std::size_t q = 0; q < a; ++q) {
      read_values(q, values);
      for (std::size_t t = 0; t < terms.size(); ++t) {
        const double coefficient = terms[t].exchange_coefficient;
        if (coefficient == 0) continue;
        runtime::cuda_trace::TraceRegion products("exchange_response_matrix_products", stream);
        runtime::cuda_trace::trace_counter("response_ao_matrix_products", 2);
        right_density_kernel<<<blocks(matrix), threads, 0, stream>>>(
            n, values, densities + t * matrix, temporary);
        left_density_kernel<<<blocks(matrix), threads, 0, stream>>>(n, densities + t * matrix,
                                                                    temporary, response);
        products.finish();
        runtime::cuda_trace::TraceRegion contractions("exchange_response_weights_and_metric",
                                                      stream);
        exchange_weights_kernel<<<blocks(count * matrix), threads, 0, stream>>>(
            matrix, a, begin, count, q, coefficient, inverse, response, weights);
        exchange_metric_kernel<<<blocks(count), threads, 0, stream>>>(
            matrix, a, begin, count, q, coefficient, raw, response, bar_inverse);
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
