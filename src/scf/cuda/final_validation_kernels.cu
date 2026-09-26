#include <cmath>

#include "scf/cuda/final_validation_kernels.hpp"

namespace vibeqc::scf::cuda_df {
namespace {
constexpr unsigned threads = validation_threads;
__device__ void add(double value, double& sum, double& correction) {
  const double next = sum + value;
  correction += fabs(sum) >= fabs(value) ? (sum - next) + value : (value - next) + sum;
  sum = next;
}
__device__ ValidationPartial merge(ValidationPartial a, const ValidationPartial& b) {
  a.norm_f = hypot(a.norm_f, b.norm_f);
  a.norm_c = hypot(a.norm_c, b.norm_c);
  a.norm_rhs = hypot(a.norm_rhs, b.norm_rhs);
  a.norm_residual = hypot(a.norm_residual, b.norm_residual);
  a.norm_density = hypot(a.norm_density, b.norm_density);
  a.eigen = fmax(a.eigen, b.eigen);
  a.metric = fmax(a.metric, b.metric);
  a.canonical = fmax(a.canonical, b.canonical);
  a.density = fmax(a.density, b.density);
  a.idempotency = fmax(a.idempotency, b.idempotency);
  a.commutator = fmax(a.commutator, b.commutator);
  a.electrons += b.electrons;
  a.energy += b.energy;
  a.invalid |= b.invalid;
  return a;
}
__device__ void store(ValidationPartial local, ValidationPartial* output) {
  __shared__ ValidationPartial shared[threads];
  // Every lane writes a fully initialized packet before any lane reads it.
  shared[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = threads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] = merge(shared[threadIdx.x], shared[threadIdx.x + stride]);
    __syncthreads();
  }
  if (!threadIdx.x) output[blockIdx.x] = shared[0];
}
__global__ void fock_kernel(std::size_t n, const double* h, const double* j, const double* exchange,
                            double weight, double* f) {
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n * n;
       k += gridDim.x * blockDim.x) {
    const auto transposed = (k % n) * n + k / n;
    f[k] = h[k] + (j[transposed] - weight * exchange[transposed]);
  }
}
__global__ void eigen_kernel(ValidationInputs in, const double* fc, const double* sc,
                             const double* gram, const double* canonical,
                             ValidationPartial* output) {
  ValidationPartial r{};
  if ((in.info && *in.info != 0) || (in.generation && *in.generation != in.expected_generation))
    r.invalid = validation_input_failure;
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < in.n * in.n;
       k += gridDim.x * blockDim.x) {
    const auto column = k / in.n, row = k % in.n;
    const double rhs = sc[k] * in.values[column], residual = fc[k] - rhs;
    const double metric = gram[k] - (row == column ? 1.0 : 0.0);
    r.invalid |= !isfinite(in.f[k]) || !isfinite(in.c[k]) || !isfinite(in.s[k]) ||
                 !isfinite(in.values[column]) || !isfinite(fc[k]) || !isfinite(sc[k]) ||
                 !isfinite(gram[k]) || !isfinite(rhs) || !isfinite(residual) ||
                 (column && in.values[column] < in.values[column - 1]);
    if (in.expected_density && (!isfinite(in.d[k]) || in.d[k] != in.expected_density[k]))
      r.invalid |= validation_input_failure;
    if (in.generation && (!isfinite(in.c[k]) || !isfinite(in.values[column])))
      r.invalid |= validation_input_failure;
    if (in.physical_fock) {
      const double other = in.f[column + row * in.n];
      if (!isfinite(in.f[k]) ||
          fabs(in.f[k] - other) > 1e-12 * fmax(1.0, fmax(fabs(in.f[k]), fabs(other))))
        r.invalid |= validation_input_failure;
    }
    r.norm_f = hypot(r.norm_f, in.f[k]);
    r.norm_c = hypot(r.norm_c, in.c[k]);
    r.norm_rhs = hypot(r.norm_rhs, rhs);
    r.norm_residual = hypot(r.norm_residual, residual);
    r.eigen = fmax(r.eigen, fabs(residual));
    r.metric = fmax(r.metric, fabs(metric));
    if (canonical) {
      const double error = canonical[k] - (row == column ? in.values[row] : 0.0);
      r.invalid |= !isfinite(canonical[k]) || !isfinite(error);
      r.canonical = fmax(r.canonical, fabs(error));
    }
  }
  store(r, output);
}
__global__ void density_kernel(ValidationInputs in, const double* reconstructed, const double* ds,
                               const double* dsd, ValidationPartial* output) {
  ValidationPartial r{};
  double ec = 0, tc = 0;
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < in.n * in.n;
       k += gridDim.x * blockDim.x) {
    const double drift = reconstructed[k] - in.d[k];
    const double idempotency = dsd[k] - in.weight * in.d[k];
    const double energy = in.d[k] * (.5 * in.h[k] + .5 * in.f[k]);
    r.invalid |= !isfinite(reconstructed[k]) || !isfinite(ds[k]) || !isfinite(dsd[k]) ||
                 !isfinite(drift) || !isfinite(idempotency) || !isfinite(energy);
    r.norm_density = hypot(r.norm_density, drift);
    r.density = fmax(r.density, fabs(drift));
    r.idempotency = fmax(r.idempotency, fabs(idempotency));
    if (k / in.n == k % in.n) add(ds[k], r.electrons, tc);
    add(energy, r.energy, ec);
  }
  r.electrons += tc;
  r.energy += ec;
  store(r, output);
}
__global__ void commutator_kernel(std::size_t n, const double* fds, const double* sdf,
                                  ValidationPartial* output) {
  ValidationPartial r{};
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n * n;
       k += gridDim.x * blockDim.x) {
    const double value = fds[k] - sdf[k];
    r.invalid |= !isfinite(fds[k]) || !isfinite(sdf[k]) || !isfinite(value);
    r.commutator = fmax(r.commutator, fabs(value));
  }
  store(r, output);
}
__global__ void finish_kernel(const ValidationPartial* partial, ValidationPartial* result,
                              unsigned stages, unsigned blocks) {
  ValidationPartial r{};
  double energy = 0, electrons = 0, ec = 0, tc = 0;
  for (unsigned k = 0; k < stages * blocks; ++k) {
    r = merge(r, partial[k]);
    add(partial[k].energy, energy, ec);
    add(partial[k].electrons, electrons, tc);
  }
  r.energy = energy + ec;
  r.electrons = electrons + tc;
  *result = r;
}
__global__ void columns_kernel(std::size_t n, std::size_t occupied, const double* c,
                               const double* values, double weight, double* columns) {
  for (std::size_t k = blockIdx.x * blockDim.x + threadIdx.x; k < n * n;
       k += gridDim.x * blockDim.x) {
    const auto column = k / n;
    columns[k] = column < occupied ? c[k] * weight * (values ? values[column] : 1.0) : 0.0;
  }
}
}  // namespace
void launch_validation_fock(cudaStream_t s, std::size_t n, const double* h, const double* j,
                            const double* k, double weight, double* f) {
  fock_kernel<<<validation_block_count(n), threads, 0, s>>>(n, h, j, k, weight, f);
}
void launch_validation_eigen(cudaStream_t s, ValidationInputs in, const double* fc,
                             const double* sc, const double* gram, const double* canonical,
                             ValidationPartial* p) {
  eigen_kernel<<<validation_block_count(in.n), threads, 0, s>>>(in, fc, sc, gram, canonical, p);
}
void launch_validation_density(cudaStream_t s, ValidationInputs in, const double* rec,
                               const double* ds, const double* dsd, ValidationPartial* p) {
  density_kernel<<<validation_block_count(in.n), threads, 0, s>>>(in, rec, ds, dsd, p);
}
void launch_validation_commutator(cudaStream_t s, std::size_t n, const double* fds,
                                  const double* sdf, ValidationPartial* p) {
  commutator_kernel<<<validation_block_count(n), threads, 0, s>>>(n, fds, sdf, p);
}
void launch_validation_finish(cudaStream_t s, ValidationPartial* p, ValidationPartial* r,
                              unsigned stages, unsigned blocks) {
  finish_kernel<<<1, 1, 0, s>>>(p, r, stages, blocks);
}
void launch_validation_columns(cudaStream_t s, std::size_t n, std::size_t occupied, const double* c,
                               const double* values, double weight, double* out) {
  columns_kernel<<<validation_block_count(n), threads, 0, s>>>(n, occupied, c, values, weight, out);
}
}  // namespace vibeqc::scf::cuda_df
