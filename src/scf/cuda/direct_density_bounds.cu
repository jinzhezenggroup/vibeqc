#include <cmath>

#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_density_bounds.hpp"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/matrix_index.cuh"

namespace vibeqc::scf::cuda_execution {

/**
 * Reduce the current AO density to the shell-block magnitudes used by J/K.
 *
 * The direct quartet kernels scatter every ERI symmetry permutation, so a
 * shell quartet can contribute through its two Coulomb density blocks or any
 * of its four crossed exchange blocks. UHF alpha and beta exchange bounds
 * remain separate so the force gate never invents an opposite-spin product.
 * RHF stores its one physical density in the alpha field; the Fock gate
 * applies the existing one-half exchange factor when it consumes that field.
 */
template <bool Unrestricted>
__global__ void reduce_shell_pair_density_bounds_kernel(
    DeviceBatch batch, const double* density, const std::uint8_t* active,
    ShellPairDensityBounds* shell_pair_density_bounds) {
  extern __shared__ double block_maxima[];
  double* coulomb_maxima = block_maxima;
  double* exchange_alpha_maxima = block_maxima + blockDim.x;
  double* exchange_beta_maxima = block_maxima + 2 * blockDim.x;
  const std::size_t shell_pair = static_cast<std::size_t>(blockIdx.x);
  if (shell_pair >= static_cast<std::size_t>(batch.total_shell_pairs)) return;
  const std::int32_t system = batch.shell_pair_systems[shell_pair];
  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t physical_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t spin_offset = static_cast<std::size_t>(system) * 2 * matrix_size;
  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::size_t ao_pair_count = shell_ao_pair_count(batch, shell_pair);

  double local_coulomb = 0.0;
  double local_exchange_alpha = 0.0;
  double local_exchange_beta = 0.0;
  if (active == nullptr || active[system] != 0) {
    for (std::size_t ordinal = threadIdx.x; ordinal < ao_pair_count; ordinal += blockDim.x) {
      std::size_t first = 0;
      std::size_t second = 0;
      decode_shell_ao_pair(batch, shell_pair, ordinal, system_ao_begin, first, second);
      const std::size_t forward = matrix_index(first, second, n);
      const std::size_t reverse = matrix_index(second, first, n);
      if constexpr (Unrestricted) {
        const double alpha_forward = density[spin_offset + forward];
        const double beta_forward = density[spin_offset + matrix_size + forward];
        const double alpha_reverse = density[spin_offset + reverse];
        const double beta_reverse = density[spin_offset + matrix_size + reverse];
        local_coulomb = fmax(local_coulomb, fmax(fabs(alpha_forward + beta_forward),
                                                 fabs(alpha_reverse + beta_reverse)));
        local_exchange_alpha =
            fmax(local_exchange_alpha, fmax(fabs(alpha_forward), fabs(alpha_reverse)));
        local_exchange_beta =
            fmax(local_exchange_beta, fmax(fabs(beta_forward), fabs(beta_reverse)));
      } else {
        const double magnitude = fmax(fabs(density[physical_offset + forward]),
                                      fabs(density[physical_offset + reverse]));
        local_coulomb = fmax(local_coulomb, magnitude);
        local_exchange_alpha = fmax(local_exchange_alpha, magnitude);
      }
    }
  }

  coulomb_maxima[threadIdx.x] = local_coulomb;
  exchange_alpha_maxima[threadIdx.x] = local_exchange_alpha;
  exchange_beta_maxima[threadIdx.x] = local_exchange_beta;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      coulomb_maxima[threadIdx.x] =
          fmax(coulomb_maxima[threadIdx.x], coulomb_maxima[threadIdx.x + stride]);
      exchange_alpha_maxima[threadIdx.x] =
          fmax(exchange_alpha_maxima[threadIdx.x], exchange_alpha_maxima[threadIdx.x + stride]);
      exchange_beta_maxima[threadIdx.x] =
          fmax(exchange_beta_maxima[threadIdx.x], exchange_beta_maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    shell_pair_density_bounds[shell_pair] = {coulomb_maxima[0], exchange_alpha_maxima[0],
                                             exchange_beta_maxima[0]};
  }
}

/**
 * Reduce one permutation-contiguous shell-pair block to its Schwarz maximum.
 *
 * The permutation is refreshed whenever geometry changes so similarly sized
 * pairs share a coarse gate and class streams retain a monotonic Schwarz tail.
 */
__global__ void reduce_bounded_shell_pair_block_bounds_kernel(DeviceBatch batch,
                                                              const std::uint32_t* shell_pair_order,
                                                              const double* shell_pair_bounds,
                                                              double* shell_pair_block_bounds) {
  extern __shared__ double block_maxima[];
  const std::size_t block = static_cast<std::size_t>(blockIdx.x);
  if (block >= static_cast<std::size_t>(batch.total_shell_pair_blocks)) return;

  std::int32_t system = 0;
  while (system + 1 < batch.batch_size &&
         static_cast<std::size_t>(batch.system_shell_pair_block_offsets[system + 1]) <= block) {
    ++system;
  }
  const std::size_t local_block =
      block - static_cast<std::size_t>(batch.system_shell_pair_block_offsets[system]);
  const std::size_t pair_begin = static_cast<std::size_t>(batch.system_shell_pair_offsets[system]);
  const std::size_t pair_end =
      static_cast<std::size_t>(batch.system_shell_pair_offsets[system + 1]);
  const std::size_t ordered_begin =
      pair_begin + local_block * detail::kBoundedDirectShellPairBlockSize;
  const std::size_t ordered_end =
      min(pair_end, ordered_begin + detail::kBoundedDirectShellPairBlockSize);

  double local_maximum = 0.0;
  for (std::size_t ordered = ordered_begin + threadIdx.x; ordered < ordered_end;
       ordered += blockDim.x) {
    local_maximum = fmax(local_maximum, shell_pair_bounds[shell_pair_order[ordered]]);
  }
  block_maxima[threadIdx.x] = local_maximum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      block_maxima[threadIdx.x] =
          fmax(block_maxima[threadIdx.x], block_maxima[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    shell_pair_block_bounds[block] = block_maxima[0];
  }
}

/**
 * Reduce per-system density maxima for both block and class-level tails.
 *
 * The scalar maximum remains the conservative gate for arbitrary block-pair
 * products.  Generated streams know their fixed shell class, so retaining a
 * ten-entry pair-class maximum avoids using (for example) a large d/d density
 * to gate an s/s stream.
 */
__global__ void reduce_bounded_system_density_bounds_kernel(
    DeviceBatch batch, const ShellPairDensityBounds* shell_pair_density_bounds,
    double* system_density_bounds, double* system_pair_density_bounds) {
  extern __shared__ double block_maxima[];
  constexpr unsigned class_count = detail::kDirectShellPairClassCount;
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch.batch_size) return;
  const std::size_t pair_begin = static_cast<std::size_t>(batch.system_shell_pair_offsets[system]);
  const std::size_t pair_end =
      static_cast<std::size_t>(batch.system_shell_pair_offsets[system + 1]);
  for (unsigned pair_class = 0; pair_class < class_count; ++pair_class) {
    block_maxima[pair_class * blockDim.x + threadIdx.x] = 0.0;
  }
  for (std::size_t pair = pair_begin + threadIdx.x; pair < pair_end; pair += blockDim.x) {
    const ShellPairDensityBounds bound = shell_pair_density_bounds[pair];
    const std::int32_t first_shell = batch.shell_pair_first[pair];
    const std::int32_t second_shell = batch.shell_pair_second[pair];
    const unsigned pair_class = direct_shell_pair_class_cuda(batch.shell_angular[first_shell],
                                                             batch.shell_angular[second_shell]);
    block_maxima[pair_class * blockDim.x + threadIdx.x] =
        fmax(block_maxima[pair_class * blockDim.x + threadIdx.x],
             fmax(bound.coulomb, fmax(bound.exchange_alpha, bound.exchange_beta)));
  }
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
    if (threadIdx.x < stride) {
      for (unsigned pair_class = 0; pair_class < class_count; ++pair_class) {
        block_maxima[pair_class * blockDim.x + threadIdx.x] =
            fmax(block_maxima[pair_class * blockDim.x + threadIdx.x],
                 block_maxima[pair_class * blockDim.x + threadIdx.x + stride]);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    double overall_maximum = 0.0;
    for (unsigned pair_class = 0; pair_class < class_count; ++pair_class) {
      const double maximum = block_maxima[pair_class * blockDim.x];
      system_pair_density_bounds[static_cast<std::size_t>(system) * class_count + pair_class] =
          maximum;
      overall_maximum = fmax(overall_maximum, maximum);
    }
    system_density_bounds[system] = overall_maximum;
  }
}

void launch_reduce_shell_pair_density_bounds_kernel(
    bool unrestricted, dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    DeviceBatch batch, const double* density, const std::uint8_t* active,
    ShellPairDensityBounds* shell_pair_density_bounds) {
  if (unrestricted == true) {
    reduce_shell_pair_density_bounds_kernel<true>
        <<<grid, block, shared_bytes, stream>>>(batch, density, active, shell_pair_density_bounds);
  } else {
    reduce_shell_pair_density_bounds_kernel<false>
        <<<grid, block, shared_bytes, stream>>>(batch, density, active, shell_pair_density_bounds);
  }
}

void launch_reduce_bounded_shell_pair_block_bounds_kernel(dim3 grid, dim3 block,
                                                          std::size_t shared_bytes,
                                                          cudaStream_t stream, DeviceBatch batch,
                                                          const std::uint32_t* shell_pair_order,
                                                          const double* shell_pair_bounds,
                                                          double* shell_pair_block_bounds) {
  reduce_bounded_shell_pair_block_bounds_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, shell_pair_order, shell_pair_bounds, shell_pair_block_bounds);
}

void launch_reduce_bounded_system_density_bounds_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const ShellPairDensityBounds* shell_pair_density_bounds, double* system_density_bounds,
    double* system_pair_density_bounds) {
  reduce_bounded_system_density_bounds_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, shell_pair_density_bounds, system_density_bounds, system_pair_density_bounds);
}

}  // namespace vibeqc::scf::cuda_execution
