#include <cmath>

#include "scf/cuda/one_electron_export_kernels.hpp"
#include "scf/cuda/one_electron_native_contraction.cuh"

namespace vibeqc::scf::cuda_execution {

/** Evaluate first-coordinate one-electron response using the retained Dual path. */
__global__ void build_cuda_one_electron_derivatives_kernel(
    DeviceBatch batch, const std::int32_t* pair_first, const std::int32_t* pair_second,
    std::size_t pair_count, std::int64_t derivative_coordinate, double* overlap, double* hcore) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch.batch_size) * pair_count) {
    return;
  }
  const std::int32_t system = static_cast<std::int32_t>(element / pair_count);
  const std::size_t pair = element % pair_count;
  const std::size_t row = static_cast<std::size_t>(pair_first[pair]);
  const std::size_t column = static_cast<std::size_t>(pair_second[pair]);
  const std::int64_t system_derivative_coordinate =
      derivative_coordinate < 0 ? derivative_coordinate
                                : derivative_coordinate + batch.atom_offsets[system] * 3;
  const Dual overlap_value =
      contracted_overlap<Dual>(batch, system, static_cast<std::int32_t>(row),
                               static_cast<std::int32_t>(column), system_derivative_coordinate);
  const Dual hcore_value =
      contracted_hcore<Dual>(batch, system, static_cast<std::int32_t>(row),
                             static_cast<std::int32_t>(column), system_derivative_coordinate);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * n * n;
  overlap[matrix_offset + row * n + column] = overlap_value.derivative;
  hcore[matrix_offset + row * n + column] = hcore_value.derivative;
  if (row != column) {
    overlap[matrix_offset + column * n + row] = overlap_value.derivative;
    hcore[matrix_offset + column * n + row] = hcore_value.derivative;
  }
}

void launch_build_cuda_one_electron_derivatives_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    const std::int32_t* pair_first, const std::int32_t* pair_second, std::size_t pair_count,
    std::int64_t derivative_coordinate, double* overlap, double* hcore) {
  build_cuda_one_electron_derivatives_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, pair_first, pair_second, pair_count, derivative_coordinate, overlap, hcore);
}

}  // namespace vibeqc::scf::cuda_execution
