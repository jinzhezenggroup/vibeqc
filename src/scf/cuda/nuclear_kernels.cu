#include <cmath>

#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/nuclear_kernels.hpp"
#include "scf/cuda/one_electron_export_kernels.hpp"

namespace vibeqc::scf::cuda_execution {

template <bool Derivative>
__global__ void build_cuda_nuclear_repulsion_kernel(DeviceBatch batch,
                                                    std::int64_t derivative_coordinate,
                                                    double* nuclear_repulsion) {
  const std::int32_t system =
      static_cast<std::int32_t>(static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch.batch_size) return;
  const std::int64_t system_derivative_coordinate =
      derivative_coordinate < 0 ? derivative_coordinate
                                : derivative_coordinate + batch.atom_offsets[system] * 3;
  if constexpr (Derivative) {
    Dual result{0.0, 0.0};
    for (std::int64_t first = batch.atom_offsets[system]; first < batch.atom_offsets[system + 1];
         ++first) {
      const Vec3<Dual> a = atom_position<Dual>(batch, first, system_derivative_coordinate);
      for (std::int64_t second = batch.atom_offsets[system]; second < first; ++second) {
        const Vec3<Dual> b = atom_position<Dual>(batch, second, system_derivative_coordinate);
        result = result +
                 static_cast<double>(batch.atomic_numbers[first] * batch.atomic_numbers[second]) /
                     qsqrt(distance_squared(a, b));
      }
    }
    nuclear_repulsion[system] = result.derivative;
  } else {
    double result = 0.0;
    for (std::int64_t first = batch.atom_offsets[system]; first < batch.atom_offsets[system + 1];
         ++first) {
      const Vec3<double> a = atom_position<double>(batch, first, -1);
      for (std::int64_t second = batch.atom_offsets[system]; second < first; ++second) {
        const Vec3<double> b = atom_position<double>(batch, second, -1);
        result += static_cast<double>(batch.atomic_numbers[first] * batch.atomic_numbers[second]) /
                  sqrt(distance_squared(a, b));
      }
    }
    nuclear_repulsion[system] = result;
  }
}

__global__ void build_nuclear_repulsion_kernel(DeviceBatch batch, double* nuclear_repulsion) {
  const std::int32_t system =
      static_cast<std::int32_t>(static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (system >= batch.batch_size) return;
  double result = 0.0;
  for (std::int64_t first = batch.atom_offsets[system]; first < batch.atom_offsets[system + 1];
       ++first) {
    const Vec3<double> a = atom_position<double>(batch, first, -1);
    for (std::int64_t second = batch.atom_offsets[system]; second < first; ++second) {
      const Vec3<double> b = atom_position<double>(batch, second, -1);
      result += static_cast<double>(batch.atomic_numbers[first] * batch.atomic_numbers[second]) /
                sqrt(distance_squared(a, b));
    }
  }
  nuclear_repulsion[system] = result;
}

__global__ void nuclear_force_kernel(DeviceBatch batch, const std::uint8_t* active,
                                     double* forces) {
  const std::int64_t coordinate = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (coordinate >= batch.total_atoms * 3) return;
  const std::int64_t atom = coordinate / 3;
  const std::int32_t system = batch.atom_systems[atom];
  if (active[system] == 0) return;
  Dual derivative{0.0, 0.0};
  for (std::int64_t first = batch.atom_offsets[system]; first < batch.atom_offsets[system + 1];
       ++first) {
    const Vec3<Dual> a = atom_position<Dual>(batch, first, coordinate);
    for (std::int64_t second = batch.atom_offsets[system]; second < first; ++second) {
      const Vec3<Dual> b = atom_position<Dual>(batch, second, coordinate);
      derivative = derivative +
                   static_cast<double>(batch.atomic_numbers[first] * batch.atomic_numbers[second]) /
                       qsqrt(distance_squared(a, b));
    }
  }
  forces[coordinate] = -derivative.derivative;
}

void launch_build_nuclear_repulsion_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                           cudaStream_t stream, DeviceBatch batch,
                                           double* nuclear_repulsion) {
  build_nuclear_repulsion_kernel<<<grid, block, shared_bytes, stream>>>(batch, nuclear_repulsion);
}

void launch_nuclear_force_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                 cudaStream_t stream, DeviceBatch batch, const std::uint8_t* active,
                                 double* forces) {
  nuclear_force_kernel<<<grid, block, shared_bytes, stream>>>(batch, active, forces);
}

void launch_build_cuda_nuclear_repulsion_kernel(bool derivative, dim3 grid, dim3 block,
                                                std::size_t shared_bytes, cudaStream_t stream,
                                                DeviceBatch batch,
                                                std::int64_t derivative_coordinate,
                                                double* nuclear_repulsion) {
  if (derivative) {
    build_cuda_nuclear_repulsion_kernel<true>
        <<<grid, block, shared_bytes, stream>>>(batch, derivative_coordinate, nuclear_repulsion);
  } else {
    build_cuda_nuclear_repulsion_kernel<false>
        <<<grid, block, shared_bytes, stream>>>(batch, derivative_coordinate, nuclear_repulsion);
  }
}

}  // namespace vibeqc::scf::cuda_execution
