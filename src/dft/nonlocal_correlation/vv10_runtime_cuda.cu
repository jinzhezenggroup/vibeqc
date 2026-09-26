#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/cuda_resources.cuh"

namespace vibeqc::dft::nlc {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

__device__ double pair_kernel(double r2, double wi, double wj, double ki, double kj,
                              Vv10Variant variant) {
  if (variant == Vv10Variant::rvv10) {
    const double zi = wi / ki * r2 + 1.0;
    const double zj = wj / kj * r2 + 1.0;
    return -1.5 / (pow(ki * kj, 1.5) * zi * zj * (zi + zj));
  }
  const double gi = wi * r2 + ki;
  const double gj = wj * r2 + kj;
  return -1.5 / (gi * gj * (gi + gj));
}

unsigned launch_blocks(std::size_t count, unsigned threads) {
  const auto blocks = 1 + (count - 1) / threads;
  if (blocks > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("nonlocal CUDA launch grid overflow");
  return static_cast<unsigned>(blocks);
}

__global__ void local_scales_kernel(std::size_t npoint, double b, double c, const double* weights,
                                    const double* density, const double* gradient, double* omega,
                                    double* kappa, double* domega_drho, double* domega_dsigma,
                                    double* dkappa_drho, double* weighted_density,
                                    bool want_features, int* failed) {
  const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= npoint) return;
  const double gx = gradient[3 * i];
  const double gy = gradient[3 * i + 1];
  const double gz = gradient[3 * i + 2];
  const double sigma = gx * gx + gy * gy + gz * gz;
  const double rho = density[i];
  const double ratio = sigma / (rho * rho);
  omega[i] = sqrt(c * ratio * ratio + (4.0 * kPi / 3.0) * rho);
  kappa[i] = b * 1.5 * kPi * pow(rho / (9.0 * kPi), 1.0 / 6.0);
  weighted_density[i] = weights[i] * rho;
  if (!isfinite(omega[i]) || !isfinite(kappa[i]) || kappa[i] <= 0.0 ||
      !isfinite(weighted_density[i]))
    atomicExch(failed, 1);
  if (!want_features) return;
  domega_drho[i] = ((4.0 * kPi / 3.0) - 4.0 * c * sigma * sigma / pow(rho, 5.0)) / (2.0 * omega[i]);
  domega_dsigma[i] = c * sigma / (omega[i] * pow(rho, 4.0));
  dkappa_drho[i] = kappa[i] / (6.0 * rho);
  if (!isfinite(domega_drho[i]) || !isfinite(domega_dsigma[i]) || !isfinite(dkappa_drho[i]))
    atomicExch(failed, 1);
}

__global__ void pair_kernel_ordered(std::size_t row_offset, std::size_t tile_points,
                                    std::size_t blocks_per_tile, std::size_t npoint,
                                    Vv10Parameters parameters, const double* points,
                                    const double* density, const double* omega, const double* kappa,
                                    const double* domega_drho, const double* domega_dsigma,
                                    const double* dkappa_drho, const double* weighted_density,
                                    double beta, double* energy_terms, double* vrho, double* vsigma,
                                    double* point_derivative, double* weight_derivative,
                                    int* failed) {
  const auto tile = static_cast<std::size_t>(blockIdx.x) / blocks_per_tile;
  const auto lane =
      (static_cast<std::size_t>(blockIdx.x) % blocks_per_tile) * blockDim.x + threadIdx.x;
  const auto i = row_offset + tile * tile_points + lane;
  if (lane >= tile_points || i >= npoint) return;
  double sum_phi = 0.0;
  double sum_rho = 0.0;
  double sum_sigma = 0.0;
  double coordinate_sum[3]{0.0, 0.0, 0.0};
  for (std::size_t j = 0; j < npoint; ++j) {
    const double dx = points[3 * j] - points[3 * i];
    const double dy = points[3 * j + 1] - points[3 * i + 1];
    const double dz = points[3 * j + 2] - points[3 * i + 2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    const double phi = pair_kernel(r2, omega[i], omega[j], kappa[i], kappa[j], parameters.variant);
    const double factor = weighted_density[j];
    sum_phi += factor * phi;
    if (vrho) {
      double dphi_domega = 0.0;
      double dphi_dkappa = 0.0;
      if (parameters.variant == Vv10Variant::rvv10) {
        const double zi = 1.0 + omega[i] / kappa[i] * r2;
        const double zj = 1.0 + omega[j] / kappa[j] * r2;
        const double factor_z = 1.0 / zi + 1.0 / (zi + zj);
        dphi_domega = -phi * r2 / kappa[i] * factor_z;
        dphi_dkappa = phi / kappa[i] * (-1.5 + (zi - 1.0) * factor_z);
      } else {
        const double gi = omega[i] * r2 + kappa[i];
        const double gj = omega[j] * r2 + kappa[j];
        const double dphi_dgi = -phi * (1.0 / gi + 1.0 / (gi + gj));
        dphi_domega = dphi_dgi * r2;
        dphi_dkappa = dphi_dgi;
      }
      const double dphi_drho = dphi_domega * domega_drho[i] + dphi_dkappa * dkappa_drho[i];
      const double dphi_dsigma = dphi_domega * domega_dsigma[i];
      sum_rho += factor * dphi_drho;
      sum_sigma += factor * dphi_dsigma;
    }
    if (point_derivative) {
      double logarithmic = 0.0;
      if (parameters.variant == Vv10Variant::rvv10) {
        const double ai = omega[i] / kappa[i];
        const double aj = omega[j] / kappa[j];
        const double zi = ai * r2 + 1.0;
        const double zj = aj * r2 + 1.0;
        logarithmic = ai / zi + aj / zj + (ai + aj) / (zi + zj);
      } else {
        const double gi = omega[i] * r2 + kappa[i];
        const double gj = omega[j] * r2 + kappa[j];
        logarithmic = omega[i] / gi + omega[j] / gj + (omega[i] + omega[j]) / (gi + gj);
      }
      const double dphi_dr2 = -phi * logarithmic;
      const double radial = -2.0 * factor * dphi_dr2;
      coordinate_sum[0] += radial * dx;
      coordinate_sum[1] += radial * dy;
      coordinate_sum[2] += radial * dz;
    }
  }
  const double scale = parameters.coefficient;
  energy_terms[i] = scale * weighted_density[i] * (beta + 0.5 * sum_phi);
  if (vrho) {
    vrho[i] = scale * (beta + sum_phi + density[i] * sum_rho);
    vsigma[i] = scale * density[i] * sum_sigma;
  }
  if (point_derivative) {
    point_derivative[3 * i] = scale * weighted_density[i] * coordinate_sum[0];
    point_derivative[3 * i + 1] = scale * weighted_density[i] * coordinate_sum[1];
    point_derivative[3 * i + 2] = scale * weighted_density[i] * coordinate_sum[2];
    weight_derivative[i] = scale * density[i] * (beta + sum_phi);
  }
  if (!isfinite(energy_terms[i]) || (vrho && (!isfinite(vrho[i]) || !isfinite(vsigma[i]))) ||
      (point_derivative &&
       (!isfinite(point_derivative[3 * i]) || !isfinite(point_derivative[3 * i + 1]) ||
        !isfinite(point_derivative[3 * i + 2]) || !isfinite(weight_derivative[i]))))
    atomicExch(failed, 1);
}

__global__ void reduce_energy_ordered_kernel(std::size_t npoint, const double* energy_terms,
                                             double* energy, int* failed) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double sum = 0.0;
  for (std::size_t i = 0; i < npoint; ++i) sum += energy_terms[i];
  if (!isfinite(sum)) atomicExch(failed, 1);
  *energy = sum;
}

__global__ void molecular_domain_kernel(std::size_t npoint, double threshold, const double* weights,
                                        const double* density, const double* gradient,
                                        double* effective_weights, double* effective_density,
                                        double* effective_gradient, int* failed) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < npoint;
       i += std::size_t(blockDim.x) * gridDim.x) {
    const double rho = density[i];
    const double gx = gradient[3 * i];
    const double gy = gradient[3 * i + 1];
    const double gz = gradient[3 * i + 2];
    const double weight = weights[i];
    const bool valid = isfinite(rho) && rho >= 0.0 && isfinite(gx) && isfinite(gy) &&
                       isfinite(gz) && isfinite(weight);
    if (!valid) atomicExch(failed, 1);
    const bool inactive = !valid || rho < threshold;
    effective_weights[i] = inactive ? 0.0 : weight;
    effective_density[i] = inactive ? 1.0 : rho;
    effective_gradient[3 * i] = inactive ? 0.0 : gx;
    effective_gradient[3 * i + 1] = inactive ? 0.0 : gy;
    effective_gradient[3 * i + 2] = inactive ? 0.0 : gz;
  }
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

Vv10CudaDeviceLayout vv10_cuda_device_layout(std::size_t point_count, std::size_t tile_points,
                                             bool features, bool geometry) {
  if (!point_count || !tile_points)
    throw std::invalid_argument("resident VV10 CUDA layout requires nonzero point/tile counts");
  const auto arrays = std::size_t{4} + (features ? 3u : 0u);
  const auto doubles =
      runtime::size_mul(arrays, point_count, "resident VV10 CUDA workspace extent overflow");
  return {point_count, std::min(tile_points, point_count),
          runtime::size_mul(doubles, sizeof(double), "resident VV10 CUDA workspace byte overflow"),
          features, geometry};
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
  const auto canonical = vv10_cuda_device_layout(layout.point_count, layout.tile_points,
                                                 layout.features, layout.geometry);
  if (layout.point_count != canonical.point_count || layout.tile_points != canonical.tile_points ||
      layout.workspace_bytes != canonical.workspace_bytes ||
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
  const auto expected_end =
      static_cast<double*>(workspace) + layout.workspace_bytes / sizeof(double);
  if (cursor != expected_end)
    throw std::logic_error("resident VV10 CUDA workspace partition mismatch");

  runtime::cuda_resource_check(cudaMemsetAsync(numerical_error, 0, sizeof(int), stream));
  constexpr unsigned threads = 128;
  const auto blocks = launch_blocks(npoint, threads);
  local_scales_kernel<<<blocks, threads, 0, stream>>>(
      npoint, parameters.b, parameters.c, weights, density, density_gradient, omega, kappa,
      domega_drho, domega_dsigma, dkappa_drho, weighted_density, layout.features, numerical_error);
  runtime::cuda_resource_check(cudaGetLastError());
  const double beta = std::pow(3.0 / (parameters.b * parameters.b), 0.75) / 32.0;
  // All point inputs, outputs and scales are already resident. Enqueue the
  // independent logical row tiles together instead of serializing a stream
  // into one/two-block kernels. Each row keeps its EXACT ordered j loop and
  // the final ordered energy reduction; no pair tensor or new scratch exists.
  const auto tile_blocks = launch_blocks(layout.tile_points, threads);
  const auto tiles = 1 + (npoint - 1) / layout.tile_points;
  const auto tiles_per_launch =
      static_cast<std::size_t>(std::numeric_limits<int>::max()) / tile_blocks;
  // Preserve a bounded launch fallback for layouts whose tile count exceeds
  // CUDA's x-grid limit. Normal resident molecular layouts need one launch.
  for (std::size_t first = 0; first < tiles; first += tiles_per_launch) {
    const auto count = std::min(tiles_per_launch, tiles - first);
    pair_kernel_ordered<<<static_cast<unsigned>(count * tile_blocks), threads, 0, stream>>>(
        first * layout.tile_points, layout.tile_points, tile_blocks, npoint, parameters, points_xyz,
        density, omega, kappa, domega_drho, domega_dsigma, dkappa_drho, weighted_density, beta,
        energy_terms, vrho, vsigma, point_derivative, weight_derivative, numerical_error);
    runtime::cuda_resource_check(cudaGetLastError());
  }
  reduce_energy_ordered_kernel<<<1, 1, 0, stream>>>(npoint, energy_terms, energy, numerical_error);
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

}  // namespace vibeqc::dft::nlc
