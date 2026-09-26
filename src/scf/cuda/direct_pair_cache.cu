#include <cmath>

#include "scf/cuda/direct_pair_cache.hpp"
#include "scf/cuda/gaussian_geometry.cuh"

namespace vibeqc::scf::cuda_execution {

/** Precompute geometry shared by every primitive quartet using a shell pair. */
__global__ void build_shell_primitive_pair_cache_kernel(DeviceBatch batch,
                                                        PrimitivePairData* shell_primitive_pairs) {
  const std::size_t shell_pair = static_cast<std::size_t>(blockIdx.x);
  if (shell_pair >= static_cast<std::size_t>(batch.total_shell_pairs)) return;
  const std::int32_t first_shell = batch.shell_pair_first[shell_pair];
  const std::int32_t second_shell = batch.shell_pair_second[shell_pair];
  const std::int64_t first_begin = batch.shell_primitive_offsets[first_shell];
  const std::int64_t second_begin = batch.shell_primitive_offsets[second_shell];
  const std::size_t first_count =
      static_cast<std::size_t>(batch.shell_primitive_offsets[first_shell + 1] - first_begin);
  const std::size_t second_count =
      static_cast<std::size_t>(batch.shell_primitive_offsets[second_shell + 1] - second_begin);
  const std::size_t pair_count = first_count * second_count;
  const Vec3<double> first = atom_position<double>(batch, batch.shell_atoms[first_shell], -1);
  const Vec3<double> second = atom_position<double>(batch, batch.shell_atoms[second_shell], -1);
  const double squared_distance = distance_squared(first, second);
  const std::size_t output_begin =
      static_cast<std::size_t>(batch.shell_pair_primitive_offsets[shell_pair]);
  for (std::size_t ordinal = threadIdx.x; ordinal < pair_count; ordinal += blockDim.x) {
    const std::int64_t first_primitive =
        first_begin + static_cast<std::int64_t>(ordinal / second_count);
    const std::int64_t second_primitive =
        second_begin + static_cast<std::int64_t>(ordinal % second_count);
    const double alpha = batch.primitive_exponents[first_primitive];
    const double beta = batch.primitive_exponents[second_primitive];
    const double exponent_sum = alpha + beta;
    const double reduced_exponent = alpha * beta / exponent_sum;
    shell_primitive_pairs[output_begin + ordinal] = {
        exponent_sum,
        reduced_exponent,
        product_center(alpha, first, beta, second),
        batch.primitive_coefficients[first_primitive] *
            batch.primitive_coefficients[second_primitive] *
            exp(-reduced_exponent * squared_distance),
        alpha / exponent_sum,
        beta / exponent_sum,
    };
  }
}

void launch_build_shell_primitive_pair_cache_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                    cudaStream_t stream, DeviceBatch batch,
                                                    PrimitivePairData* shell_primitive_pairs) {
  build_shell_primitive_pair_cache_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, shell_primitive_pairs);
}

}  // namespace vibeqc::scf::cuda_execution
