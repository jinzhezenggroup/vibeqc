#include <cmath>

#include "scf/cuda/direct_pair_cache.hpp"
#include "scf/cuda/gaussian_geometry.cuh"

namespace vibeqc::scf::cuda_execution {

/** Precompute geometry shared by every primitive quartet using a shell pair. */
__global__ void build_shell_primitive_pair_cache_kernel(
    std::int64_t total_shell_pairs, const std::int32_t* shell_pair_first,
    const std::int32_t* shell_pair_second, const std::int64_t* shell_primitive_offsets,
    const std::int32_t* shell_atoms, const double* positions,
    const std::int64_t* shell_pair_primitive_offsets, const double* primitive_exponents,
    const double* primitive_coefficients, PrimitivePairData* shell_primitive_pairs) {
  const std::size_t shell_pair = static_cast<std::size_t>(blockIdx.x);
  if (shell_pair >= static_cast<std::size_t>(total_shell_pairs)) return;
  const std::int32_t first_shell = shell_pair_first[shell_pair];
  const std::int32_t second_shell = shell_pair_second[shell_pair];
  const std::int64_t first_begin = shell_primitive_offsets[first_shell];
  const std::int64_t second_begin = shell_primitive_offsets[second_shell];
  const std::size_t first_count =
      static_cast<std::size_t>(shell_primitive_offsets[first_shell + 1] - first_begin);
  const std::size_t second_count =
      static_cast<std::size_t>(shell_primitive_offsets[second_shell + 1] - second_begin);
  const std::size_t pair_count = first_count * second_count;
  const std::int64_t first_atom = shell_atoms[first_shell];
  const std::int64_t second_atom = shell_atoms[second_shell];
  const std::int64_t first_position = 3 * first_atom;
  const std::int64_t second_position = 3 * second_atom;
  const Vec3<double> first = {positions[first_position], positions[first_position + 1],
                              positions[first_position + 2]};
  const Vec3<double> second = {positions[second_position], positions[second_position + 1],
                               positions[second_position + 2]};
  const double squared_distance = distance_squared(first, second);
  const std::size_t output_begin =
      static_cast<std::size_t>(shell_pair_primitive_offsets[shell_pair]);
  for (std::size_t ordinal = threadIdx.x; ordinal < pair_count; ordinal += blockDim.x) {
    const std::int64_t first_primitive =
        first_begin + static_cast<std::int64_t>(ordinal / second_count);
    const std::int64_t second_primitive =
        second_begin + static_cast<std::int64_t>(ordinal % second_count);
    const double alpha = primitive_exponents[first_primitive];
    const double beta = primitive_exponents[second_primitive];
    const double exponent_sum = alpha + beta;
    const double reduced_exponent = alpha * beta / exponent_sum;
    shell_primitive_pairs[output_begin + ordinal] = {
        exponent_sum,
        reduced_exponent,
        product_center(alpha, first, beta, second),
        primitive_coefficients[first_primitive] * primitive_coefficients[second_primitive] *
            exp(-reduced_exponent * squared_distance),
        alpha / exponent_sum,
        beta / exponent_sum,
    };
  }
}

void launch_build_shell_primitive_pair_cache_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                                    cudaStream_t stream, DeviceBatch batch,
                                                    PrimitivePairData* shell_primitive_pairs) {
  // Keep the public launcher contract in DeviceBatch form, but flatten the device-kernel
  // parameters. CuMetal's typed PTX lowering rejects aggregate DeviceBatch parameter loads,
  // while NVIDIA CUDA and the Metal translator both support these scalar/pointer fields.
  build_shell_primitive_pair_cache_kernel<<<grid, block, shared_bytes, stream>>>(
      batch.total_shell_pairs, batch.shell_pair_first, batch.shell_pair_second,
      batch.shell_primitive_offsets, batch.shell_atoms, batch.positions,
      batch.shell_pair_primitive_offsets, batch.primitive_exponents, batch.primitive_coefficients,
      shell_primitive_pairs);
}

}  // namespace vibeqc::scf::cuda_execution
