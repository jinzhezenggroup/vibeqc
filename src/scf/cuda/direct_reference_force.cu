#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_native_contraction.cuh"
#include "scf/cuda/direct_reference_force.hpp"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

__global__ void two_electron_force_kernel(DeviceBatch batch, const double* density,
                                          const std::uint8_t* active, double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t quartet_count = matrix_size * matrix_size;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coordinate_count = static_cast<std::size_t>(batch.total_atoms) * 3;
  if (element >= coordinate_count * quartet_count) return;
  const std::int64_t coordinate = static_cast<std::int64_t>(element / quartet_count);
  std::size_t local = element % quartet_count;
  const std::size_t l = local % n;
  local /= n;
  const std::size_t k = local % n;
  local /= n;
  const std::size_t j = local % n;
  const std::size_t i = local / n;
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const double coefficient = 0.5 * density[matrix_offset + matrix_index(i, j, n)] *
                                 density[matrix_offset + matrix_index(k, l, n)] -
                             0.25 * density[matrix_offset + matrix_index(i, k, n)] *
                                 density[matrix_offset + matrix_index(j, l, n)];
  if (coefficient == 0.0) return;
  const Dual derivative = contracted_eri<Dual>(
      batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
      static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), coordinate);
  atomicAdd(forces + coordinate, -coefficient * derivative.derivative);
}

__global__ void two_electron_uhf_force_kernel(DeviceBatch batch, const double* spin_density,
                                              const std::uint8_t* active, double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t quartet_count = matrix_size * matrix_size;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coordinate_count = static_cast<std::size_t>(batch.total_atoms) * 3;
  if (element >= coordinate_count * quartet_count) return;
  const std::int64_t coordinate = static_cast<std::int64_t>(element / quartet_count);
  std::size_t local = element % quartet_count;
  const std::size_t l = local % n;
  local /= n;
  const std::size_t k = local % n;
  local /= n;
  const std::size_t j = local % n;
  const std::size_t i = local / n;
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  const std::size_t alpha_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  const std::size_t ij = matrix_index(i, j, n);
  const std::size_t kl = matrix_index(k, l, n);
  const double alpha_ij = spin_density[alpha_offset + ij];
  const double beta_ij = spin_density[beta_offset + ij];
  const double alpha_kl = spin_density[alpha_offset + kl];
  const double beta_kl = spin_density[beta_offset + kl];
  const double total_ij = alpha_ij + beta_ij;
  const double total_kl = alpha_kl + beta_kl;
  const double coefficient = 0.5 * total_ij * total_kl -
                             0.5 * spin_density[alpha_offset + matrix_index(i, k, n)] *
                                 spin_density[alpha_offset + matrix_index(j, l, n)] -
                             0.5 * spin_density[beta_offset + matrix_index(i, k, n)] *
                                 spin_density[beta_offset + matrix_index(j, l, n)];
  if (coefficient == 0.0) return;
  const Dual derivative = contracted_eri<Dual>(
      batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
      static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), coordinate);
  atomicAdd(forces + coordinate, -coefficient * derivative.derivative);
}

__global__ void two_electron_force_direct_kernel(
    DeviceBatch batch, double screening_tolerance, const std::int32_t* pair_first,
    const std::int32_t* pair_second, std::size_t pair_count, const double* schwarz_bounds,
    const double* density, const std::uint8_t* active, double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t work_per_coordinate = matrix_size * pair_count;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coordinate_count = static_cast<std::size_t>(batch.total_atoms) * 3;
  if (element >= coordinate_count * work_per_coordinate) return;
  const std::int64_t coordinate = static_cast<std::int64_t>(element / work_per_coordinate);
  std::size_t local = element % work_per_coordinate;
  const std::size_t packed_kl = local % pair_count;
  local /= pair_count;
  const std::size_t j = local % n;
  const std::size_t i = local / n;
  const std::size_t k = static_cast<std::size_t>(pair_first[packed_kl]);
  const std::size_t l = static_cast<std::size_t>(pair_second[packed_kl]);
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const double pij = density[matrix_offset + matrix_index(i, j, n)];
  const double pkl = density[matrix_offset + matrix_index(k, l, n)];
  if (pij == 0.0 || pkl == 0.0) return;

  double energy_derivative = 0.0;
  const double coulomb_bound = schwarz_bounds[matrix_offset + matrix_index(i, j, n)] *
                               schwarz_bounds[matrix_offset + matrix_index(k, l, n)];
  if (coulomb_bound >= screening_tolerance) {
    const Dual derivative = contracted_eri<Dual>(
        batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
        static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), coordinate);
    // The packed density pair represents both (k,l) and (l,k) when k != l.
    const double coulomb_coefficient = k == l ? 0.5 * pij * pkl : pij * pkl;
    energy_derivative += coulomb_coefficient * derivative.derivative;
  }

  const double exchange_bound = schwarz_bounds[matrix_offset + matrix_index(i, k, n)] *
                                schwarz_bounds[matrix_offset + matrix_index(j, l, n)];
  if (exchange_bound >= screening_tolerance) {
    const Dual derivative = contracted_eri<Dual>(
        batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(k),
        static_cast<std::int32_t>(j), static_cast<std::int32_t>(l), coordinate);
    energy_derivative -= 0.25 * pij * pkl * derivative.derivative;
  }
  if (k != l) {
    // Packing (k,l) also represents the swapped density pair. Unlike the
    // Coulomb term, exchange maps it to a distinct integral permutation.
    const double transposed_exchange_bound = schwarz_bounds[matrix_offset + matrix_index(i, l, n)] *
                                             schwarz_bounds[matrix_offset + matrix_index(j, k, n)];
    if (transposed_exchange_bound >= screening_tolerance) {
      const Dual derivative = contracted_eri<Dual>(
          batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(l),
          static_cast<std::int32_t>(j), static_cast<std::int32_t>(k), coordinate);
      energy_derivative -= 0.25 * pij * pkl * derivative.derivative;
    }
  }
  if (energy_derivative != 0.0) {
    atomicAdd(forces + coordinate, -energy_derivative);
  }
}

__global__ void two_electron_uhf_force_direct_kernel(
    DeviceBatch batch, double screening_tolerance, const std::int32_t* pair_first,
    const std::int32_t* pair_second, std::size_t pair_count, const double* schwarz_bounds,
    const double* spin_density, const std::uint8_t* active, double* forces) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t work_per_coordinate = matrix_size * pair_count;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coordinate_count = static_cast<std::size_t>(batch.total_atoms) * 3;
  if (element >= coordinate_count * work_per_coordinate) return;
  const std::int64_t coordinate = static_cast<std::int64_t>(element / work_per_coordinate);
  std::size_t local = element % work_per_coordinate;
  const std::size_t packed_kl = local % pair_count;
  local /= pair_count;
  const std::size_t j = local % n;
  const std::size_t i = local / n;
  const std::size_t k = static_cast<std::size_t>(pair_first[packed_kl]);
  const std::size_t l = static_cast<std::size_t>(pair_second[packed_kl]);
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t alpha_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  const std::size_t ij = matrix_index(i, j, n);
  const std::size_t kl = matrix_index(k, l, n);
  const double alpha_ij = spin_density[alpha_offset + ij];
  const double beta_ij = spin_density[beta_offset + ij];
  const double alpha_kl = spin_density[alpha_offset + kl];
  const double beta_kl = spin_density[beta_offset + kl];
  const double total_ij = alpha_ij + beta_ij;
  const double total_kl = alpha_kl + beta_kl;

  double energy_derivative = 0.0;
  const double coulomb_bound =
      schwarz_bounds[physical_offset + ij] * schwarz_bounds[physical_offset + kl];
  if (total_ij != 0.0 && total_kl != 0.0 && coulomb_bound >= screening_tolerance) {
    const Dual derivative = contracted_eri<Dual>(
        batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
        static_cast<std::int32_t>(k), static_cast<std::int32_t>(l), coordinate);
    const double coefficient = k == l ? 0.5 * total_ij * total_kl : total_ij * total_kl;
    energy_derivative += coefficient * derivative.derivative;
  }

  const double exchange_pair = alpha_ij * alpha_kl + beta_ij * beta_kl;
  const double exchange_bound = schwarz_bounds[physical_offset + matrix_index(i, k, n)] *
                                schwarz_bounds[physical_offset + matrix_index(j, l, n)];
  if (exchange_pair != 0.0 && exchange_bound >= screening_tolerance) {
    const Dual derivative = contracted_eri<Dual>(
        batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(k),
        static_cast<std::int32_t>(j), static_cast<std::int32_t>(l), coordinate);
    energy_derivative -= 0.5 * exchange_pair * derivative.derivative;
  }
  if (k != l && exchange_pair != 0.0) {
    const double transposed_exchange_bound =
        schwarz_bounds[physical_offset + matrix_index(i, l, n)] *
        schwarz_bounds[physical_offset + matrix_index(j, k, n)];
    if (transposed_exchange_bound >= screening_tolerance) {
      const Dual derivative = contracted_eri<Dual>(
          batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(l),
          static_cast<std::int32_t>(j), static_cast<std::int32_t>(k), coordinate);
      energy_derivative -= 0.5 * exchange_pair * derivative.derivative;
    }
  }
  if (energy_derivative != 0.0) {
    atomicAdd(forces + coordinate, -energy_derivative);
  }
}

void launch_two_electron_force_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                      cudaStream_t stream, DeviceBatch batch, const double* density,
                                      const std::uint8_t* active, double* forces) {
  two_electron_force_kernel<<<grid, block, shared_bytes, stream>>>(batch, density, active, forces);
}

void launch_two_electron_uhf_force_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                          cudaStream_t stream, DeviceBatch batch,
                                          const double* spin_density, const std::uint8_t* active,
                                          double* forces) {
  two_electron_uhf_force_kernel<<<grid, block, shared_bytes, stream>>>(batch, spin_density, active,
                                                                       forces);
}

void launch_two_electron_force_direct_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    double screening_tolerance, const std::int32_t* pair_first, const std::int32_t* pair_second,
    std::size_t pair_count, const double* schwarz_bounds, const double* density,
    const std::uint8_t* active, double* forces) {
  two_electron_force_direct_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, screening_tolerance, pair_first, pair_second, pair_count, schwarz_bounds, density,
      active, forces);
}

void launch_two_electron_uhf_force_direct_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    double screening_tolerance, const std::int32_t* pair_first, const std::int32_t* pair_second,
    std::size_t pair_count, const double* schwarz_bounds, const double* spin_density,
    const std::uint8_t* active, double* forces) {
  two_electron_uhf_force_direct_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, screening_tolerance, pair_first, pair_second, pair_count, schwarz_bounds, spin_density,
      active, forces);
}

}  // namespace vibeqc::scf::cuda_execution
