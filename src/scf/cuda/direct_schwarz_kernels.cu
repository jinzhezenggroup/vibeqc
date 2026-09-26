#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_native_source_contraction.cuh"
#include "scf/cuda/direct_queue_index.cuh"
#include "scf/cuda/direct_schwarz_kernels.hpp"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

__global__ void build_schwarz_bounds_packed_kernel(DeviceBatch batch, std::size_t pair_count,
                                                   double* schwarz_bounds) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch.batch_size) * pair_count) {
    return;
  }
  const std::int32_t system = static_cast<std::int32_t>(element / pair_count);
  const std::size_t pair = element % pair_count;
  std::size_t i = 0;
  std::size_t j = 0;
  decode_lower_triangle(pair, i, j);
  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * n * n;
  const double diagonal = contracted_eri_cartesian_source<double>(
      batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
      static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), -1);
  // fabs is conservative when roundoff makes a non-negative diagonal
  // slightly negative; it never converts that noise into a false zero.
  const double bound = sqrt(fabs(diagonal));
  schwarz_bounds[matrix_offset + matrix_index(i, j, n)] = bound;
  if (i != j) {
    schwarz_bounds[matrix_offset + matrix_index(j, i, n)] = bound;
  }
}

/** Atomically retain the maximum non-negative IEEE-754 double. */
__device__ void atomic_max_double(double* address, double value) {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (__longlong_as_double(old) < value) {
    const unsigned long long assumed = old;
    old = atomicCAS(bits, assumed, __double_as_longlong(value));
    if (old == assumed) return;
  }
}

/**
 * Build AO-pair Schwarz bounds and shell-pair maxima in one packed pass.
 *
 * AO-pair bounds are naturally a dense packed workload: a 768-AO system has
 * only 295,296 canonical AO pairs but 73,920 shell pairs.  Assigning one
 * 256-thread block to every shell pair leaves nearly all lanes idle for the
 * common s/p shells.  This kernel keeps the original packed grid and uses a
 * low-contention atomic maximum for the shell-pair reduction while each
 * contracted diagonal ERI is still in registers.
 */
__global__ void build_schwarz_and_shell_pair_bounds_packed_kernel(DeviceBatch batch,
                                                                  std::size_t pair_count,
                                                                  double* schwarz_bounds,
                                                                  double* shell_pair_bounds) {
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(batch.batch_size) * pair_count;
  if (element >= total) return;

  const std::int32_t system = static_cast<std::int32_t>(element / pair_count);
  const std::size_t pair = element % pair_count;
  std::size_t i = 0;
  std::size_t j = 0;
  decode_lower_triangle(pair, i, j);
  const std::size_t n = static_cast<std::size_t>(batch.direct_nbf);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * n * n;
  const double diagonal = contracted_eri_cartesian_source<double>(
      batch, system, static_cast<std::int32_t>(i), static_cast<std::int32_t>(j),
      static_cast<std::int32_t>(i), static_cast<std::int32_t>(j), -1);
  // fabs is conservative when roundoff makes a non-negative diagonal
  // slightly negative; it never converts that noise into a false zero.
  const double bound = sqrt(fabs(diagonal));
  schwarz_bounds[matrix_offset + matrix_index(i, j, n)] = bound;
  if (i != j) {
    schwarz_bounds[matrix_offset + matrix_index(j, i, n)] = bound;
  }

  const std::size_t system_ao_begin = static_cast<std::size_t>(system) * n;
  const std::int32_t first_shell = batch.direct_ao_shells[system_ao_begin + i];
  const std::int32_t second_shell = batch.direct_ao_shells[system_ao_begin + j];
  // The direct-AO-to-shell map is topology metadata.  Keep malformed or
  // stale entries from turning the unsigned local-index arithmetic below
  // into an arena write outside this system's shell-pair segment.
  const std::int64_t shell_begin = batch.system_shell_offsets[system];
  const std::int64_t shell_end = batch.system_shell_offsets[system + 1];
  if (first_shell < shell_begin || second_shell < shell_begin || first_shell >= shell_end ||
      second_shell >= shell_end) {
    return;
  }
  const std::size_t shell_pair = system_shell_pair_index(batch, system, first_shell, second_shell);
  const std::size_t shell_pair_begin =
      static_cast<std::size_t>(batch.system_shell_pair_offsets[system]);
  const std::size_t shell_pair_end =
      static_cast<std::size_t>(batch.system_shell_pair_offsets[system + 1]);
  if (shell_pair < shell_pair_begin || shell_pair >= shell_pair_end ||
      shell_pair >= static_cast<std::size_t>(batch.total_shell_pairs)) {
    return;
  }
  atomic_max_double(shell_pair_bounds + shell_pair, bound);
}

void launch_build_schwarz_bounds_packed_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                               cudaStream_t stream, DeviceBatch batch,
                                               std::size_t pair_count, double* schwarz_bounds) {
  build_schwarz_bounds_packed_kernel<<<grid, block, shared_bytes, stream>>>(batch, pair_count,
                                                                            schwarz_bounds);
}

void launch_build_schwarz_and_shell_pair_bounds_packed_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, DeviceBatch batch,
    std::size_t pair_count, double* schwarz_bounds, double* shell_pair_bounds) {
  build_schwarz_and_shell_pair_bounds_packed_kernel<<<grid, block, shared_bytes, stream>>>(
      batch, pair_count, schwarz_bounds, shell_pair_bounds);
}

}  // namespace vibeqc::scf::cuda_execution
