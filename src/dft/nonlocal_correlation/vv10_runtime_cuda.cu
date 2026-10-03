#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dft/grid_task_view.cuh"
#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "generated_nonlocal_pair_native.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/cuda_resources.cuh"

namespace generativeqc::dft::nlc {
namespace {

// Scientific pair formulas and output-demand closures have one compiler owner.
using PairKernelValues = generated::PairValues;

template <Vv10Variant Variant, bool Features, bool Geometry>
__device__ PairKernelValues pair_kernel_values(double r2, double wi, double wj, double ki,
                                               double kj, double row_inverse_kappa) {
  if constexpr (Variant == Vv10Variant::vv10 && Features)
    return generated::pair_values_vv10_rational<Geometry>(r2, wi, wj, ki, kj, row_inverse_kappa);
  return generated::pair_values<Variant, Features, Geometry, true>(r2, wi, wj, ki, kj,
                                                                   row_inverse_kappa);
}

unsigned launch_blocks(std::size_t count, unsigned threads) {
  const auto blocks = 1 + (count - 1) / threads;
  if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("nonlocal CUDA launch grid overflow");
  return static_cast<unsigned>(blocks);
}

template <Vv10Variant Variant, bool Features>
__global__ void local_scales_kernel(std::size_t npoint, double b, double c, const double* weights,
                                    const double* density, const double* gradient, double* omega,
                                    double* kappa, double* domega_drho, double* domega_dsigma,
                                    double* dkappa_drho, double* weighted_density, int* failed) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= npoint) return;
  const double gx = gradient[3 * i];
  const double gy = gradient[3 * i + 1];
  const double gz = gradient[3 * i + 2];
  const double sigma = gx * gx + gy * gy + gz * gz;
  const double rho = density[i];
  const auto local = generated::local_scales_cuda<Variant, Features>(rho, sigma, b, c);
  omega[i] = local.omega;
  kappa[i] = local.kappa;
  weighted_density[i] = weights[i] * rho;
  // Preserve the inactive -0 marker, but do not create one when an active
  // negative integration weight underflows in the density product.
  if (weighted_density[i] == 0.0 && weights[i] != 0.0) weighted_density[i] = 0.0;
  if (!isfinite(omega[i]) || !isfinite(kappa[i]) || kappa[i] <= 0.0 ||
      !isfinite(weighted_density[i]))
    atomicExch(failed, 1);
  if constexpr (Features) {
    domega_drho[i] = local.domega_drho;
    domega_dsigma[i] = local.domega_dsigma;
    dkappa_drho[i] = local.dkappa_drho;
    if (!isfinite(domega_drho[i]) || !isfinite(domega_dsigma[i]) || !isfinite(dkappa_drho[i]))
      atomicExch(failed, 1);
  }
  generated::precondition_local_scales_cuda<Variant>(omega[i], kappa[i]);
  if constexpr (Variant == Vv10Variant::rvv10) {
    if (!isfinite(omega[i]) || !isfinite(kappa[i])) atomicExch(failed, 1);
  }
}

__global__ void count_active_partner_blocks_kernel(std::size_t npoint,
                                                   const double* weighted_density,
                                                   std::uint64_t* block_offsets) {
  constexpr unsigned kWarpSize = 32;
  __shared__ unsigned warp_counts[4];
  const auto j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool active = j < npoint && weighted_density[j] != 0.0;
  const unsigned mask = __ballot_sync(0xffffffffu, active);
  const unsigned lane = threadIdx.x % kWarpSize;
  const unsigned warp = threadIdx.x / kWarpSize;
  if (lane == 0) warp_counts[warp] = __popc(mask);
  __syncthreads();
  if (threadIdx.x == 0) {
    std::uint64_t count = 0;
    for (unsigned w = 0; w < blockDim.x / kWarpSize; ++w) count += warp_counts[w];
    block_offsets[blockIdx.x] = count;
  }
}

__global__ void prefix_active_partner_blocks_kernel(std::size_t block_count,
                                                    std::uint64_t* block_offsets,
                                                    std::uint64_t* active_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  std::uint64_t offset = 0;
  for (std::size_t block = 0; block < block_count; ++block) {
    const auto count = block_offsets[block];
    block_offsets[block] = offset;
    offset += count;
  }
  *active_count = offset;
}

__global__ void scatter_active_partners_ordered_kernel(std::size_t npoint,
                                                       const double* weighted_density,
                                                       const std::uint64_t* block_offsets,
                                                       std::uint64_t* active_indices) {
  constexpr unsigned kWarpSize = 32;
  __shared__ unsigned warp_counts[4];
  __shared__ unsigned warp_offsets[4];
  const auto j = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool active = j < npoint && weighted_density[j] != 0.0;
  const unsigned mask = __ballot_sync(0xffffffffu, active);
  const unsigned lane = threadIdx.x % kWarpSize;
  const unsigned warp = threadIdx.x / kWarpSize;
  if (lane == 0) warp_counts[warp] = __popc(mask);
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned offset = 0;
    for (unsigned w = 0; w < blockDim.x / kWarpSize; ++w) {
      warp_offsets[w] = offset;
      offset += warp_counts[w];
    }
  }
  __syncthreads();
  if (!active) return;
  const unsigned lower_mask = lane == 0 ? 0u : (mask & ((1u << lane) - 1u));
  const auto rank = warp_offsets[warp] + __popc(lower_mask);
  active_indices[block_offsets[blockIdx.x] + rank] = static_cast<std::uint64_t>(j);
}

// A positive-zero row is still observable through potential/weight response;
// only the negative-zero density-screen marker is excluded. Absolute coordinate
// bounds imply r2 <= 3*2^30 < 2^32 for every admitted pair, including roundoff.
// This is sufficient, not necessary: translated/extreme grids use the fallback.
__global__ void admit_molecular_pair_domain_kernel(std::size_t npoint, const double* points,
                                                   const double* density, const double* omega,
                                                   const double* kappa, const double* domega_drho,
                                                   const double* domega_dsigma,
                                                   const double* dkappa_drho,
                                                   const double* weighted_density, int* rejected) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= npoint) return;
  const double factor = weighted_density[i];
  if (factor == 0.0 && signbit(factor)) return;
  const bool admitted = fabs(points[3 * i]) <= 0x1p14 && fabs(points[3 * i + 1]) <= 0x1p14 &&
                        fabs(points[3 * i + 2]) <= 0x1p14 && omega[i] >= 0x1p-32 &&
                        omega[i] <= 0x1p32 && kappa[i] >= 0x1p-32 && kappa[i] <= 0x1p32 &&
                        fabs(factor) <= 0x1p64 && fabs(density[i]) <= 0x1p128 &&
                        fabs(domega_drho[i]) <= 0x1p128 && fabs(domega_dsigma[i]) <= 0x1p128 &&
                        fabs(dkappa_drho[i]) <= 0x1p128;
  if (!admitted) atomicExch(rejected, 1);
}

template <Vv10Variant Variant, bool Features, bool Geometry, bool MaskZeroRows,
          bool Prevalidated = false>
__global__ void pair_kernel_ordered(
    std::size_t row_offset, std::size_t row_count, double coefficient, const double* points,
    const double* density, const double* omega, const double* kappa, const double* domega_drho,
    const double* domega_dsigma, const double* dkappa_drho, const double* weighted_density,
    const std::uint64_t* active_indices, const std::uint64_t* active_count, double beta,
    double* energy_terms, double* vrho, double* vsigma, double* point_derivative,
    double* weight_derivative, int* failed, const int* bounds_rejected = nullptr) {
  static_assert(!Prevalidated || (Variant == Vv10Variant::vv10 && Features && MaskZeroRows));
  // Both launch variants read the same stream-ordered device predicate. Exactly
  // one traverses pairs, with no host readback or change to reduction order.
  if (bounds_rejected != nullptr && ((*bounds_rejected == 0) != Prevalidated)) return;
  const auto lane = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (lane >= row_count) return;
  const auto i = row_offset + lane;

  const double xi = points[3 * i];
  const double yi = points[3 * i + 1];
  const double zi = points[3 * i + 2];
  const double wi = omega[i];
  const double ki = kappa[i];
  const double rhoi = density[i];
  const double weighted_i = weighted_density[i];
  const double domega_rhoi = Features ? domega_drho[i] : 0.0;
  const double domega_sigmai = Features ? domega_dsigma[i] : 0.0;
  const double dkappa_rhoi = Features ? dkappa_drho[i] : 0.0;
  if constexpr (MaskZeroRows) {
    // Only negative zero denotes a density-screened row. Finite negative
    // quadrature weights and active positive-zero rows retain their derivatives.
    if (weighted_i == 0.0 && signbit(weighted_i)) {
      energy_terms[i] = 0.0;
      if constexpr (Features) {
        vrho[i] = 0.0;
        vsigma[i] = 0.0;
      }
      if constexpr (Geometry) {
        point_derivative[3 * i] = 0.0;
        point_derivative[3 * i + 1] = 0.0;
        point_derivative[3 * i + 2] = 0.0;
        weight_derivative[i] = 0.0;
      }
      return;
    }
  }
  double row_inverse_kappa = 0.0;
  if constexpr (Variant == Vv10Variant::rvv10 && Features)
    row_inverse_kappa = 1.0 / (6.0 * rhoi * dkappa_rhoi);

  double sum_phi = 0.0;
  double sum_rho = 0.0;
  double sum_sigma = 0.0;
  double coordinate_sum[3]{0.0, 0.0, 0.0};
  const auto nactive = *active_count;
  // The preflight proved every pair and row-chain bound. Only this admitted
  // specialization changes feature summation rounding; the general route keeps
  // its original one-pass chain, without a rejected-prefix replay.
  for (std::uint64_t slot = 0; slot < nactive; ++slot) {
    const auto j = static_cast<std::size_t>(active_indices[slot]);
    const double factor = weighted_density[j];
    const double dx = points[3 * j] - xi;
    const double dy = points[3 * j + 1] - yi;
    const double dz = points[3 * j + 2] - zi;
    const double r2 = dx * dx + dy * dy + dz * dz;
    const auto pair = [&]() {
      if constexpr (Prevalidated)
        return generated::pair_values_vv10_admitted<Geometry>(r2, wi, omega[j], ki, kappa[j]);
      else
        return pair_kernel_values<Variant, Features, Geometry>(r2, wi, omega[j], ki, kappa[j],
                                                               row_inverse_kappa);
    }();
    sum_phi += factor * pair.phi;
    if constexpr (Features) {
      if constexpr (Prevalidated) {
        sum_rho += factor * pair.dphi_dkappa;
        sum_sigma += factor * pair.dphi_domega;
      } else {
        const double dphi_drho = pair.dphi_domega * domega_rhoi + pair.dphi_dkappa * dkappa_rhoi;
        const double dphi_dsigma = pair.dphi_domega * domega_sigmai;
        sum_rho += factor * dphi_drho;
        sum_sigma += factor * dphi_dsigma;
      }
    }
    if constexpr (Geometry) {
      const double radial = -2.0 * factor * pair.dphi_dr2;
      coordinate_sum[0] += radial * dx;
      coordinate_sum[1] += radial * dy;
      coordinate_sum[2] += radial * dz;
    }
  }

  energy_terms[i] = coefficient * weighted_i * (beta + 0.5 * sum_phi);
  bool nonfinite = !isfinite(energy_terms[i]);
  if constexpr (Features) {
    if constexpr (Prevalidated) {
      generated::row_feature_values(sum_phi, sum_sigma, sum_rho, rhoi, domega_rhoi, domega_sigmai,
                                    dkappa_rhoi, beta, coefficient, vrho[i], vsigma[i]);
    } else {
      vrho[i] = coefficient * (beta + sum_phi + rhoi * sum_rho);
      vsigma[i] = coefficient * rhoi * sum_sigma;
    }
    nonfinite = nonfinite || !isfinite(vrho[i]) || !isfinite(vsigma[i]);
  }
  if constexpr (Geometry) {
    point_derivative[3 * i] = coefficient * weighted_i * coordinate_sum[0];
    point_derivative[3 * i + 1] = coefficient * weighted_i * coordinate_sum[1];
    point_derivative[3 * i + 2] = coefficient * weighted_i * coordinate_sum[2];
    weight_derivative[i] = coefficient * rhoi * (beta + sum_phi);
    nonfinite = nonfinite || !isfinite(point_derivative[3 * i]) ||
                !isfinite(point_derivative[3 * i + 1]) || !isfinite(point_derivative[3 * i + 2]) ||
                !isfinite(weight_derivative[i]);
  }
  if (nonfinite) atomicExch(failed, 1);
}

template <Vv10Variant Variant, bool Features, bool Geometry, bool MaskZeroRows>
void launch_pair_rows_impl(const Vv10CudaDeviceLayout& layout, cudaStream_t stream,
                           double coefficient, const double* points, const double* density,
                           const double* omega, const double* kappa, const double* domega_drho,
                           const double* domega_dsigma, const double* dkappa_drho,
                           const double* weighted_density, const std::uint64_t* active_indices,
                           const std::uint64_t* active_count, double beta, double* energy_terms,
                           double* vrho, double* vsigma, double* point_derivative,
                           double* weight_derivative, int* failed,
                           const int* bounds_rejected = nullptr) {
  constexpr unsigned threads = 128;
  const auto max_rows_per_launch =
      static_cast<std::size_t>(std::numeric_limits<int>::max()) * threads;
  for (std::size_t first = 0; first < layout.point_count; first += max_rows_per_launch) {
    const auto count = std::min(max_rows_per_launch, layout.point_count - first);
    const auto blocks = launch_blocks(count, threads);
    pair_kernel_ordered<Variant, Features, Geometry, MaskZeroRows><<<blocks, threads, 0, stream>>>(
        first, count, coefficient, points, density, omega, kappa, domega_drho, domega_dsigma,
        dkappa_drho, weighted_density, active_indices, active_count, beta, energy_terms, vrho,
        vsigma, point_derivative, weight_derivative, failed, bounds_rejected);
    runtime::cuda_resource_check(cudaGetLastError());
    if constexpr (Variant == Vv10Variant::vv10 && Features && MaskZeroRows) {
      if (bounds_rejected != nullptr) {
        pair_kernel_ordered<Variant, Features, Geometry, MaskZeroRows, true>
            <<<blocks, threads, 0, stream>>>(first, count, coefficient, points, density, omega,
                                             kappa, domega_drho, domega_dsigma, dkappa_drho,
                                             weighted_density, active_indices, active_count, beta,
                                             energy_terms, vrho, vsigma, point_derivative,
                                             weight_derivative, failed, bounds_rejected);
        runtime::cuda_resource_check(cudaGetLastError());
      }
    }
  }
}

template <Vv10Variant Variant, bool Features, bool Geometry>
void launch_pair_rows(const Vv10CudaDeviceLayout& layout, cudaStream_t stream, double coefficient,
                      const double* points, const double* density, const double* omega,
                      const double* kappa, const double* domega_drho, const double* domega_dsigma,
                      const double* dkappa_drho, const double* weighted_density,
                      const std::uint64_t* active_indices, const std::uint64_t* active_count,
                      double beta, double* energy_terms, double* vrho, double* vsigma,
                      double* point_derivative, double* weight_derivative, int* failed,
                      const int* bounds_rejected = nullptr) {
  if (layout.mask_zero_weight_rows)
    launch_pair_rows_impl<Variant, Features, Geometry, true>(
        layout, stream, coefficient, points, density, omega, kappa, domega_drho, domega_dsigma,
        dkappa_drho, weighted_density, active_indices, active_count, beta, energy_terms, vrho,
        vsigma, point_derivative, weight_derivative, failed, bounds_rejected);
  else
    launch_pair_rows_impl<Variant, Features, Geometry, false>(
        layout, stream, coefficient, points, density, omega, kappa, domega_drho, domega_dsigma,
        dkappa_drho, weighted_density, active_indices, active_count, beta, energy_terms, vrho,
        vsigma, point_derivative, weight_derivative, failed, bounds_rejected);
}

constexpr unsigned kOrderedEnergyLoadThreads = 128U;

__global__ void reduce_energy_ordered_kernel(std::size_t npoint, const double* energy_terms,
                                             double* energy, int* failed) {
  if (blockIdx.x != 0) return;
  __shared__ double staged[kOrderedEnergyLoadThreads];
  double sum = 0.0;
  for (std::size_t base = 0; base < npoint; base += kOrderedEnergyLoadThreads) {
    const std::size_t index = base + threadIdx.x;
    if (index < npoint) staged[threadIdx.x] = energy_terms[index];
    __syncthreads();
    if (threadIdx.x == 0) {
      const std::size_t width =
          npoint - base < kOrderedEnergyLoadThreads ? npoint - base : kOrderedEnergyLoadThreads;
      // Preserve the historical left-to-right FP64 additions exactly. Only
      // global-memory acquisition is cooperative; arithmetic association and
      // one-rounding-per-add semantics are unchanged.
      for (std::size_t i = 0; i < width; ++i) sum += staged[i];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (!isfinite(sum)) atomicExch(failed, 1);
    *energy = sum;
  }
}

__global__ void molecular_domain_kernel(std::size_t npoint, double threshold, const double* weights,
                                        const double* density, const double* gradient,
                                        double* effective_weights, double* effective_density,
                                        double* effective_gradient, int* failed) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < npoint;
       i += std::size_t(blockDim.x) * gridDim.x) {
    // The private force owner uses in-place effective rho/gradient storage.
    // Keep every input load before the first output store for this point.
    const double rho = density[i];
    const double gx = gradient[3 * i];
    const double gy = gradient[3 * i + 1];
    const double gz = gradient[3 * i + 2];
    const double weight = weights[i];
    const bool valid = isfinite(rho) && rho >= 0.0 && isfinite(gx) && isfinite(gy) &&
                       isfinite(gz) && isfinite(weight);
    if (!valid) atomicExch(failed, 1);
    const bool inactive = !valid || rho < threshold;
    // Distinguish screened density rows from active signed-zero weights without
    // allocating an additional point mask.
    effective_weights[i] = inactive ? -0.0 : (weight == 0.0 ? 0.0 : weight);
    effective_density[i] = inactive ? 1.0 : rho;
    effective_gradient[3 * i] = inactive ? 0.0 : gx;
    effective_gradient[3 * i + 1] = inactive ? 0.0 : gy;
    effective_gradient[3 * i + 2] = inactive ? 0.0 : gz;
  }
}

__global__ void collect_total_features_kernel(generativeqc::dft::GridTaskView view,
                                              std::size_t offset, std::size_t total_points,
                                              double* density, double* gradient, int* failed) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= view.npoint) return;
  if (offset > total_points || view.npoint > total_points - offset || !view.features) {
    atomicExch(failed, 1);
    return;
  }
  const auto np = view.npoint;
  const auto out = offset + i;
  // A finite feature buffer may still belong to a failed producer generation.
  // Carry its sticky status before reading features, and initialize safe padding
  // for the downstream domain kernel. Seed publication will poison every row.
  if (view.error && *view.error) {
    atomicExch(failed, 1);
    density[out] = 0.0;
    gradient[3 * out] = 0.0;
    gradient[3 * out + 1] = 0.0;
    gradient[3 * out + 2] = 0.0;
    return;
  }
  const double rho = view.features[i] + view.features[5 * np + i];
  const double gx = view.features[np + i] + view.features[6 * np + i];
  const double gy = view.features[2 * np + i] + view.features[7 * np + i];
  const double gz = view.features[3 * np + i] + view.features[8 * np + i];
  const bool valid = isfinite(rho) && rho >= 0.0 && isfinite(gx) && isfinite(gy) && isfinite(gz);
  if (!valid) atomicExch(failed, 1);
  density[out] = valid ? rho : 0.0;
  gradient[3 * out] = valid ? gx : 0.0;
  gradient[3 * out + 1] = valid ? gy : 0.0;
  gradient[3 * out + 2] = valid ? gz : 0.0;
}

__global__ void pack_force_seeds_kernel(std::size_t npoint, const double* effective_weights,
                                        const double* point_derivative, double* seeds,
                                        const int* collect_error, const int* domain_error,
                                        const int* pair_error) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= npoint) return;
  bool failed = *collect_error != 0 || *domain_error != 0 || *pair_error != 0;
  const double px = point_derivative[3 * i];
  const double py = point_derivative[3 * i + 1];
  const double pz = point_derivative[3 * i + 2];
  failed = failed || !isfinite(effective_weights[i]) || !isfinite(seeds[i]) ||
           !isfinite(seeds[npoint + i]) || !isfinite(px) || !isfinite(py) || !isfinite(pz) ||
           !isfinite(seeds[5 * npoint + i]);
  if (failed) {
    const double poison = __longlong_as_double(0x7ff8000000000000ULL);
    for (std::size_t row = 0; row < 6; ++row) seeds[row * npoint + i] = poison;
    return;
  }
  // Only the inactive negative-zero marker erases the force seeds.
  if (effective_weights[i] == 0.0 && signbit(effective_weights[i])) {
    for (std::size_t row = 0; row < 6; ++row) seeds[row * npoint + i] = 0.0;
    return;
  }
  seeds[2 * npoint + i] = px;
  seeds[3 * npoint + i] = py;
  seeds[4 * npoint + i] = pz;
}

}  // namespace

void enqueue_vv10_molecular_domain_cuda(cudaStream_t stream, std::size_t point_count,
                                        double density_threshold, const double* weights,
                                        const double* density, const double* density_gradient,
                                        double* effective_weights, double* effective_density,
                                        double* effective_density_gradient, int* numerical_error) {
  if (stream == nullptr || !point_count || !std::isfinite(density_threshold) ||
      density_threshold <= 0.0 || weights == nullptr || density == nullptr ||
      density_gradient == nullptr || effective_weights == nullptr || effective_density == nullptr ||
      effective_density_gradient == nullptr || numerical_error == nullptr)
    throw std::invalid_argument("invalid resident molecular VV10 domain request");
  runtime::cuda_resource_check(cudaMemsetAsync(numerical_error, 0, sizeof(int), stream));
  constexpr unsigned threads = 128;
  molecular_domain_kernel<<<launch_blocks(point_count, threads), threads, 0, stream>>>(
      point_count, density_threshold, weights, density, density_gradient, effective_weights,
      effective_density, effective_density_gradient, numerical_error);
  runtime::cuda_resource_check(cudaGetLastError());
}

void enqueue_vv10_collect_total_features_cuda(cudaStream_t stream,
                                              const generativeqc::dft::GridTaskView& view,
                                              std::size_t offset, std::size_t total_points,
                                              double* density, double* density_gradient,
                                              int* numerical_error) {
  if (stream == nullptr || view.version != 1 || !view.npoint || !view.features ||
      view.stream != stream || offset > total_points || view.npoint > total_points - offset ||
      density == nullptr || density_gradient == nullptr || numerical_error == nullptr)
    throw std::invalid_argument("invalid resident VV10 feature collection request");
  constexpr unsigned threads = 128;
  collect_total_features_kernel<<<launch_blocks(view.npoint, threads), threads, 0, stream>>>(
      view, offset, total_points, density, density_gradient, numerical_error);
  runtime::cuda_resource_check(cudaGetLastError());
}

void enqueue_vv10_pack_force_seeds_cuda(cudaStream_t stream, std::size_t point_count,
                                        const double* effective_weights,
                                        const double* point_derivative, double* seeds,
                                        const int* collect_error, const int* domain_error,
                                        const int* pair_error) {
  if (stream == nullptr || !point_count || effective_weights == nullptr ||
      point_derivative == nullptr || seeds == nullptr || collect_error == nullptr ||
      domain_error == nullptr || pair_error == nullptr)
    throw std::invalid_argument("invalid resident VV10 force seed pack request");
  constexpr unsigned threads = 128;
  pack_force_seeds_kernel<<<launch_blocks(point_count, threads), threads, 0, stream>>>(
      point_count, effective_weights, point_derivative, seeds, collect_error, domain_error,
      pair_error);
  runtime::cuda_resource_check(cudaGetLastError());
}

Vv10CudaDeviceLayout vv10_cuda_device_layout(std::size_t point_count, std::size_t tile_points,
                                             bool features, bool geometry,
                                             bool mask_zero_weight_rows) {
  if (!point_count || !tile_points)
    throw std::invalid_argument("resident VV10 CUDA layout requires nonzero point/tile counts");
  constexpr std::size_t partner_threads = 128;
  const auto partner_blocks = 1 + (point_count - 1) / partner_threads;
  const auto arrays = std::size_t{4} + (features ? 3u : 0u);
  auto slots =
      runtime::size_mul(arrays, point_count, "resident VV10 CUDA workspace extent overflow");
  slots = runtime::size_add(slots, point_count,
                            "resident VV10 CUDA active-partner index extent overflow");
  slots = runtime::size_add(slots, partner_blocks,
                            "resident VV10 CUDA active-partner block extent overflow");
  slots = runtime::size_add(slots, std::size_t{1},
                            "resident VV10 CUDA active-partner count extent overflow");
  static_assert(sizeof(double) == sizeof(std::uint64_t));
  return {point_count,
          std::min(tile_points, point_count),
          runtime::size_mul(slots, sizeof(double), "resident VV10 CUDA workspace byte overflow"),
          features,
          geometry,
          mask_zero_weight_rows};
}

void enqueue_vv10_cuda_device(const Vv10CudaDeviceLayout& layout, Vv10Parameters parameters,
                              int device_id, cudaStream_t stream, const double* points_xyz,
                              const double* weights, const double* density,
                              const double* density_gradient, void* workspace,
                              std::size_t workspace_bytes, double* energy, double* vrho,
                              double* vsigma, double* point_derivative, double* weight_derivative,
                              int* numerical_error) {
  // The resident entry point can be called without Vv10Plan::prepare. Preserve
  // that owner's scientific parameter domain before touching the caller stream.
  if ((parameters.variant != Vv10Variant::vv10 && parameters.variant != Vv10Variant::rvv10) ||
      !std::isfinite(parameters.b) || parameters.b <= 0.0 || !std::isfinite(parameters.c) ||
      parameters.c <= 0.0 || !std::isfinite(parameters.coefficient) ||
      parameters.coefficient <= 0.0)
    throw std::invalid_argument(
        "resident VV10 CUDA parameters must have a supported variant and finite positive values");
  const auto canonical =
      vv10_cuda_device_layout(layout.point_count, layout.tile_points, layout.features,
                              layout.geometry, layout.mask_zero_weight_rows);
  if (layout.point_count != canonical.point_count || layout.tile_points != canonical.tile_points ||
      layout.workspace_bytes != canonical.workspace_bytes ||
      layout.mask_zero_weight_rows != canonical.mask_zero_weight_rows ||
      workspace_bytes < layout.workspace_bytes)
    throw std::invalid_argument("resident VV10 CUDA layout/workspace mismatch");
  if (device_id < 0 || stream == nullptr || points_xyz == nullptr || weights == nullptr ||
      density == nullptr || density_gradient == nullptr || workspace == nullptr ||
      energy == nullptr || numerical_error == nullptr)
    throw std::invalid_argument("resident VV10 CUDA execution received a null owner/input");
  if (reinterpret_cast<std::uintptr_t>(workspace) % alignof(double))
    throw std::invalid_argument("resident VV10 CUDA workspace is misaligned");
  if (layout.features != (vrho != nullptr && vsigma != nullptr) ||
      (vrho == nullptr) != (vsigma == nullptr))
    throw std::invalid_argument("resident VV10 CUDA feature outputs disagree with layout");
  if (layout.geometry != (point_derivative != nullptr && weight_derivative != nullptr) ||
      (point_derivative == nullptr) != (weight_derivative == nullptr))
    throw std::invalid_argument("resident VV10 CUDA geometry outputs disagree with layout");

  runtime::CudaDeviceScope device(device_id);
  auto* cursor = static_cast<double*>(workspace);
  auto take = [&](std::size_t count) {
    auto* out = cursor;
    cursor += count;
    return out;
  };
  const auto npoint = layout.point_count;
  double* omega = take(npoint);
  double* kappa = take(npoint);
  double* domega_drho = layout.features ? take(npoint) : nullptr;
  double* domega_dsigma = layout.features ? take(npoint) : nullptr;
  double* dkappa_drho = layout.features ? take(npoint) : nullptr;
  double* weighted_density = take(npoint);
  double* energy_terms = take(npoint);
  constexpr unsigned partner_threads = 128;
  const auto partner_blocks = launch_blocks(npoint, partner_threads);
  auto* active_indices = reinterpret_cast<std::uint64_t*>(take(npoint));
  auto* block_offsets = reinterpret_cast<std::uint64_t*>(take(partner_blocks));
  auto* active_count = reinterpret_cast<std::uint64_t*>(take(1));
  const auto expected_end =
      static_cast<double*>(workspace) + layout.workspace_bytes / sizeof(double);
  if (cursor != expected_end)
    throw std::logic_error("resident VV10 CUDA workspace partition mismatch");

  runtime::cuda_resource_check(cudaMemsetAsync(numerical_error, 0, sizeof(int), stream));
  constexpr unsigned threads = 128;
  const auto blocks = launch_blocks(npoint, threads);
  if (parameters.variant == Vv10Variant::rvv10) {
    if (layout.features)
      local_scales_kernel<Vv10Variant::rvv10, true><<<blocks, threads, 0, stream>>>(
          npoint, parameters.b, parameters.c, weights, density, density_gradient, omega, kappa,
          domega_drho, domega_dsigma, dkappa_drho, weighted_density, numerical_error);
    else
      local_scales_kernel<Vv10Variant::rvv10, false><<<blocks, threads, 0, stream>>>(
          npoint, parameters.b, parameters.c, weights, density, density_gradient, omega, kappa,
          domega_drho, domega_dsigma, dkappa_drho, weighted_density, numerical_error);
  } else {
    if (layout.features)
      local_scales_kernel<Vv10Variant::vv10, true><<<blocks, threads, 0, stream>>>(
          npoint, parameters.b, parameters.c, weights, density, density_gradient, omega, kappa,
          domega_drho, domega_dsigma, dkappa_drho, weighted_density, numerical_error);
    else
      local_scales_kernel<Vv10Variant::vv10, false><<<blocks, threads, 0, stream>>>(
          npoint, parameters.b, parameters.c, weights, density, density_gradient, omega, kappa,
          domega_drho, domega_dsigma, dkappa_drho, weighted_density, numerical_error);
  }
  runtime::cuda_resource_check(cudaGetLastError());

  // Build one stable device-resident partner list. Block counts are reduced in
  // block order and each warp scatters active lanes in increasing j order, so
  // every row preserves the historical ordered summation while skipping
  // density-screened/zero-factor partners entirely.
  count_active_partner_blocks_kernel<<<partner_blocks, partner_threads, 0, stream>>>(
      npoint, weighted_density, block_offsets);
  runtime::cuda_resource_check(cudaGetLastError());
  prefix_active_partner_blocks_kernel<<<1, 1, 0, stream>>>(partner_blocks, block_offsets,
                                                           active_count);
  runtime::cuda_resource_check(cudaGetLastError());
  scatter_active_partners_ordered_kernel<<<partner_blocks, partner_threads, 0, stream>>>(
      npoint, weighted_density, block_offsets, active_indices);
  runtime::cuda_resource_check(cudaGetLastError());

  // Compaction's block offsets are dead after scatter, and one slot always
  // exists. Borrow its first word for a predicate on this same stream. No new
  // allocation, retained state, or cross-invocation identity is introduced.
  int* bounds_rejected = nullptr;
  if (parameters.variant == Vv10Variant::vv10 && layout.features && layout.mask_zero_weight_rows &&
      npoint <= (std::uint64_t{1} << 32) && std::fabs(parameters.coefficient) <= 0x1p32) {
    bounds_rejected = reinterpret_cast<int*>(block_offsets);
    runtime::cuda_resource_check(cudaMemsetAsync(bounds_rejected, 0, sizeof(int), stream));
    admit_molecular_pair_domain_kernel<<<blocks, threads, 0, stream>>>(
        npoint, points_xyz, density, omega, kappa, domega_drho, domega_dsigma, dkappa_drho,
        weighted_density, bounds_rejected);
    runtime::cuda_resource_check(cudaGetLastError());
  }

  const double beta = std::pow(3.0 / (parameters.b * parameters.b), 0.75) / 32.0;

  // Specialize the O(N^2) pair loop by scientific variant and requested
  // outputs. Resident rows launch as one flat 1D domain (with a finite
  // grid-limit fallback), avoiding logical-tile division/modulo and inactive
  // lanes while preserving each row's ordered j traversal and final reduction.
  if (parameters.variant == Vv10Variant::rvv10) {
    if (layout.features) {
      if (layout.geometry)
        launch_pair_rows<Vv10Variant::rvv10, true, true>(
            layout, stream, parameters.coefficient, points_xyz, density, omega, kappa, domega_drho,
            domega_dsigma, dkappa_drho, weighted_density, active_indices, active_count, beta,
            energy_terms, vrho, vsigma, point_derivative, weight_derivative, numerical_error,
            bounds_rejected);
      else
        launch_pair_rows<Vv10Variant::rvv10, true, false>(
            layout, stream, parameters.coefficient, points_xyz, density, omega, kappa, domega_drho,
            domega_dsigma, dkappa_drho, weighted_density, active_indices, active_count, beta,
            energy_terms, vrho, vsigma, point_derivative, weight_derivative, numerical_error,
            bounds_rejected);
    } else if (layout.geometry) {
      launch_pair_rows<Vv10Variant::rvv10, false, true>(
          layout, stream, parameters.coefficient, points_xyz, density, omega, kappa, domega_drho,
          domega_dsigma, dkappa_drho, weighted_density, active_indices, active_count, beta,
          energy_terms, vrho, vsigma, point_derivative, weight_derivative, numerical_error,
          bounds_rejected);
    } else {
      launch_pair_rows<Vv10Variant::rvv10, false, false>(
          layout, stream, parameters.coefficient, points_xyz, density, omega, kappa, domega_drho,
          domega_dsigma, dkappa_drho, weighted_density, active_indices, active_count, beta,
          energy_terms, vrho, vsigma, point_derivative, weight_derivative, numerical_error,
          bounds_rejected);
    }
  } else {
    if (layout.features) {
      if (layout.geometry)
        launch_pair_rows<Vv10Variant::vv10, true, true>(
            layout, stream, parameters.coefficient, points_xyz, density, omega, kappa, domega_drho,
            domega_dsigma, dkappa_drho, weighted_density, active_indices, active_count, beta,
            energy_terms, vrho, vsigma, point_derivative, weight_derivative, numerical_error,
            bounds_rejected);
      else
        launch_pair_rows<Vv10Variant::vv10, true, false>(
            layout, stream, parameters.coefficient, points_xyz, density, omega, kappa, domega_drho,
            domega_dsigma, dkappa_drho, weighted_density, active_indices, active_count, beta,
            energy_terms, vrho, vsigma, point_derivative, weight_derivative, numerical_error,
            bounds_rejected);
    } else if (layout.geometry) {
      launch_pair_rows<Vv10Variant::vv10, false, true>(
          layout, stream, parameters.coefficient, points_xyz, density, omega, kappa, domega_drho,
          domega_dsigma, dkappa_drho, weighted_density, active_indices, active_count, beta,
          energy_terms, vrho, vsigma, point_derivative, weight_derivative, numerical_error,
          bounds_rejected);
    } else {
      launch_pair_rows<Vv10Variant::vv10, false, false>(
          layout, stream, parameters.coefficient, points_xyz, density, omega, kappa, domega_drho,
          domega_dsigma, dkappa_drho, weighted_density, active_indices, active_count, beta,
          energy_terms, vrho, vsigma, point_derivative, weight_derivative, numerical_error,
          bounds_rejected);
    }
  }
  reduce_energy_ordered_kernel<<<1, kOrderedEnergyLoadThreads, 0, stream>>>(
      npoint, energy_terms, energy, numerical_error);
  runtime::cuda_resource_check(cudaGetLastError());
}

void execute_vv10_cuda(const double* points_xyz, const double* weights, const double* density,
                       const double* density_gradient, std::size_t npoint, std::size_t tile_points,
                       Vv10Parameters parameters, int device_id, double& energy, double* vrho,
                       double* vsigma, double* point_derivative, double* weight_derivative) {
  if ((point_derivative == nullptr) != (weight_derivative == nullptr))
    throw std::invalid_argument("nonlocal geometry outputs must be requested together");
  if ((vrho == nullptr) != (vsigma == nullptr))
    throw std::invalid_argument("nonlocal feature outputs must be requested together");
  const bool features = vrho != nullptr;
  const bool geometry = point_derivative != nullptr;
  const auto layout = vv10_cuda_device_layout(npoint, tile_points, features, geometry);
  const auto io_arrays = std::size_t{8} + (features ? 2u : 0u) + (geometry ? 4u : 0u);
  const auto io_doubles =
      runtime::size_mul(io_arrays, npoint, "VV10 CUDA resident I/O extent overflow");
  const auto workspace_doubles = layout.workspace_bytes / sizeof(double);
  const auto doubles = runtime::size_add(
      runtime::size_add(io_doubles, workspace_doubles, "VV10 CUDA arena extent overflow"),
      std::size_t{1}, "VV10 CUDA failure-slot extent overflow");

  int host_failed = 0;
  runtime::CudaDeviceScope device(device_id);
  runtime::OwnedCudaStream stream(device_id);
  runtime::OwnedCudaBuffer<double> arena(device_id, doubles, stream.get());
  double* cursor = arena.get();
  auto take = [&](std::size_t count) {
    double* out = cursor;
    cursor += count;
    return out;
  };
  double* d_points = take(3 * npoint);
  double* d_weights = take(npoint);
  double* d_density = take(npoint);
  double* d_gradient = take(3 * npoint);
  double* d_vrho = features ? take(npoint) : nullptr;
  double* d_vsigma = features ? take(npoint) : nullptr;
  double* d_point_derivative = geometry ? take(3 * npoint) : nullptr;
  double* d_weight_derivative = geometry ? take(npoint) : nullptr;
  double* workspace = take(workspace_doubles);
  int* failed = reinterpret_cast<int*>(take(1));
  if (cursor != arena.get() + doubles)
    throw std::logic_error("nonlocal CUDA arena layout mismatch");
  // The ordered reduction runs after every pair kernel, so this scalar may
  // safely reuse the first scratch double without extending the historical
  // provider-owned device bound.
  double* d_energy = workspace;

  const auto copy_h2d = [&](double* destination, const double* source, std::size_t count) {
    runtime::cuda_resource_check(cudaMemcpyAsync(destination, source, count * sizeof(double),
                                                 cudaMemcpyHostToDevice, stream.get()));
  };
  copy_h2d(d_points, points_xyz, 3 * npoint);
  copy_h2d(d_weights, weights, npoint);
  copy_h2d(d_density, density, npoint);
  copy_h2d(d_gradient, density_gradient, 3 * npoint);

  enqueue_vv10_cuda_device(layout, parameters, device_id, stream.get(), d_points, d_weights,
                           d_density, d_gradient, workspace, layout.workspace_bytes, d_energy,
                           d_vrho, d_vsigma, d_point_derivative, d_weight_derivative, failed);

  runtime::cuda_resource_check(
      cudaMemcpyAsync(&energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost, stream.get()));
  if (vrho)
    runtime::cuda_resource_check(cudaMemcpyAsync(vrho, d_vrho, npoint * sizeof(double),
                                                 cudaMemcpyDeviceToHost, stream.get()));
  if (vsigma)
    runtime::cuda_resource_check(cudaMemcpyAsync(vsigma, d_vsigma, npoint * sizeof(double),
                                                 cudaMemcpyDeviceToHost, stream.get()));
  if (geometry) {
    runtime::cuda_resource_check(cudaMemcpyAsync(point_derivative, d_point_derivative,
                                                 3 * npoint * sizeof(double),
                                                 cudaMemcpyDeviceToHost, stream.get()));
    runtime::cuda_resource_check(cudaMemcpyAsync(weight_derivative, d_weight_derivative,
                                                 npoint * sizeof(double), cudaMemcpyDeviceToHost,
                                                 stream.get()));
  }
  runtime::cuda_resource_check(
      cudaMemcpyAsync(&host_failed, failed, sizeof(int), cudaMemcpyDeviceToHost, stream.get()));
  stream.synchronize();
  if (host_failed) throw std::overflow_error("nonfinite nonlocal CUDA result");
  if (!std::isfinite(energy)) throw std::overflow_error("nonfinite nonlocal CUDA energy");
}

}  // namespace generativeqc::dft::nlc
