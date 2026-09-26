#include <cuda_runtime.h>

#include <cmath>

#include "scf/cuda/device_timer.cuh"
#include "scf/cuda/eigensolver_kernels.hpp"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda_batch.hpp"

namespace vibeqc::scf::cuda_execution {

/** Native Jacobi kernels and inactive-provider instrumentation; shared CUDA runtime, with no
 * molecular/provider dependency. */
namespace {
/** Allocate and initialize the record owned by this sequential Graph replay. */
__global__ void begin_inactive_eigensolver_profile_kernel(
    std::int32_t physical_batch_size, std::int32_t solver_batch_size, std::uint32_t family,
    bool provider_invoked, bool cublas_transformed_inactive, const std::uint8_t* physical_active,
    const std::uint8_t* solver_active, std::uint32_t capacity, std::uint32_t* count,
    DeviceInactiveEigensolverProfileEntry* entries) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const std::uint32_t index = atomicAdd(count, 1U);
  if (index >= capacity) return;
  std::uint32_t active_physical_count = 0;
  std::uint32_t active_solver_count = 0;
  for (std::int32_t system = 0; system < physical_batch_size; ++system) {
    active_physical_count += physical_active[system] != 0 ? 1U : 0U;
  }
  for (std::int32_t state = 0; state < solver_batch_size; ++state) {
    active_solver_count += solver_active[state] != 0 ? 1U : 0U;
  }
  const bool has_inactive = active_solver_count < static_cast<std::uint32_t>(solver_batch_size);
  DeviceInactiveEigensolverProfileEntry& entry = entries[index];
  entry.solver_start_nanoseconds = 0U;
  entry.solver_elapsed_nanoseconds = 0U;
  entry.iteration = index + 1U;
  entry.family = family;
  entry.physical_system_count = static_cast<std::uint32_t>(physical_batch_size);
  entry.solver_batch_count = static_cast<std::uint32_t>(solver_batch_size);
  entry.active_physical_count = active_physical_count;
  entry.active_solver_count = active_solver_count;
  entry.inactive_input_nonfinite_count = 0U;
  entry.inactive_submission_nonfinite_count = 0U;
  entry.inactive_info_nonzero_count = 0U;
  entry.inactive_touch_flags = has_inactive && cublas_transformed_inactive
                                   ? VIBEQC_EIGENSOLVER_INACTIVE_TOUCH_CUBLAS_TRANSFORM
                                   : 0U;
  entry.provider_invoked = provider_invoked ? 1U : 0U;
}

/**
 * Replace every inactive provider input with an identity matrix.
 *
 * cuSOLVER providers cannot consume the active mask. Identity substitution
 * guarantees finite, well-conditioned input without changing the fixed batch
 * size. The optional diagnostic counts non-finite values before replacement;
 * it is not part of the production fast path when profiling is disabled.
 */
__global__ void sanitize_inactive_solver_input_kernel(
    std::int32_t solver_batch_size, std::int32_t nbf, const std::uint8_t* solver_active,
    double* matrices, int* info, std::uint32_t profile_capacity, const std::uint32_t* profile_count,
    DeviceInactiveEigensolverProfileEntry* profile_entries) {
  const std::int32_t state = static_cast<std::int32_t>(blockIdx.x);
  if (state >= solver_batch_size) return;
  if (threadIdx.x == 0) info[state] = 0;
  if (solver_active[state] != 0) return;
  __shared__ unsigned matrix_nonfinite;
  __shared__ unsigned submission_nonfinite;
  if (threadIdx.x == 0) {
    matrix_nonfinite = 0U;
    submission_nonfinite = 0U;
  }
  __syncthreads();
  DeviceInactiveEigensolverProfileEntry* profile = nullptr;
  if (profile_entries != nullptr && profile_count != nullptr && *profile_count != 0U &&
      *profile_count <= profile_capacity) {
    profile = profile_entries + (*profile_count - 1U);
    if (threadIdx.x == 0) {
      atomicOr(&profile->inactive_touch_flags, VIBEQC_EIGENSOLVER_INACTIVE_TOUCH_IDENTITY_SANITIZE);
    }
  }
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t offset = static_cast<std::size_t>(state) * matrix_size;
  for (std::size_t element = threadIdx.x; element < matrix_size; element += blockDim.x) {
    const double input = matrices[offset + element];
    if (profile != nullptr && !isfinite(input)) {
      atomicExch(&matrix_nonfinite, 1U);
    }
    const std::size_t row = element % n;
    const std::size_t column = element / n;
    matrices[offset + element] = row == column ? 1.0 : 0.0;
  }
  __syncthreads();
  if (profile != nullptr && threadIdx.x == 0 && matrix_nonfinite != 0U) {
    atomicAdd(&profile->inactive_input_nonfinite_count, 1U);
  }
  if (profile != nullptr) {
    for (std::size_t element = threadIdx.x; element < matrix_size; element += blockDim.x) {
      if (!isfinite(matrices[offset + element])) {
        atomicExch(&submission_nonfinite, 1U);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0 && submission_nonfinite != 0U) {
      atomicAdd(&profile->inactive_submission_nonfinite_count, 1U);
    }
  }
}

__global__ void start_inactive_eigensolver_timer_kernel(
    std::uint32_t capacity, const std::uint32_t* count,
    DeviceInactiveEigensolverProfileEntry* entries) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || *count == 0U || *count > capacity) {
    return;
  }
  entries[*count - 1U].solver_start_nanoseconds = globaltimer_nanoseconds();
}

__global__ void finish_inactive_eigensolver_profile_kernel(
    std::int32_t solver_batch_size, const std::uint8_t* solver_active, const int* info,
    std::uint32_t capacity, const std::uint32_t* count,
    DeviceInactiveEigensolverProfileEntry* entries) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || *count == 0U || *count > capacity) {
    return;
  }
  DeviceInactiveEigensolverProfileEntry& entry = entries[*count - 1U];
  const std::uint64_t stop = globaltimer_nanoseconds();
  entry.solver_elapsed_nanoseconds = stop - entry.solver_start_nanoseconds;
  std::uint32_t inactive_info_nonzero_count = 0U;
  for (std::int32_t state = 0; state < solver_batch_size; ++state) {
    if (solver_active[state] == 0 && info[state] != 0) {
      ++inactive_info_nonzero_count;
    }
  }
  entry.inactive_info_nonzero_count = inactive_info_nonzero_count;
}

__global__ void symmetric_eigen_small_kernel(std::int32_t batch_size, std::int32_t nbf,
                                             double* matrices, double* eigenvalues, int* info,
                                             const std::uint8_t* active) {
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || threadIdx.x != 0) return;
  info[system] = 0;
  if (active != nullptr && active[system] == 0) return;
  if (nbf <= 0 || nbf > kSmallEigensolverLimit) {
    info[system] = -1;
    return;
  }
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t offset = static_cast<std::size_t>(system) * matrix_size;
  double matrix[kSmallEigensolverLimit * kSmallEigensolverLimit];
  double vectors[kSmallEigensolverLimit * kSmallEigensolverLimit];
  for (std::size_t column = 0; column < n; ++column) {
    for (std::size_t row = 0; row < n; ++row) {
      matrix[matrix_index(row, column, n)] = matrices[offset + matrix_index(row, column, n)];
      vectors[matrix_index(row, column, n)] = row == column ? 1.0 : 0.0;
    }
  }

  const std::size_t maximum_sweeps = 20 * matrix_size > 50 ? 20 * matrix_size : 50;
  for (std::size_t sweep = 0; sweep < maximum_sweeps; ++sweep) {
    std::size_t p = 0;
    std::size_t q = 0;
    double largest = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
      for (std::size_t column = row + 1; column < n; ++column) {
        const double candidate = fabs(matrix[matrix_index(row, column, n)]);
        if (candidate > largest) {
          largest = candidate;
          p = row;
          q = column;
        }
      }
    }
    if (largest < 1.0e-14) break;
    if (sweep + 1 == maximum_sweeps) info[system] = 1;

    const double app = matrix[matrix_index(p, p, n)];
    const double aqq = matrix[matrix_index(q, q, n)];
    const double apq = matrix[matrix_index(p, q, n)];
    const double angle = 0.5 * atan2(2.0 * apq, aqq - app);
    const double cosine = cos(angle);
    const double sine = sin(angle);
    for (std::size_t k = 0; k < n; ++k) {
      if (k == p || k == q) continue;
      const double mkp = matrix[matrix_index(k, p, n)];
      const double mkq = matrix[matrix_index(k, q, n)];
      matrix[matrix_index(k, p, n)] = matrix[matrix_index(p, k, n)] = cosine * mkp - sine * mkq;
      matrix[matrix_index(k, q, n)] = matrix[matrix_index(q, k, n)] = sine * mkp + cosine * mkq;
    }
    matrix[matrix_index(p, p, n)] =
        cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
    matrix[matrix_index(q, q, n)] =
        sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
    matrix[matrix_index(p, q, n)] = 0.0;
    matrix[matrix_index(q, p, n)] = 0.0;
    for (std::size_t row = 0; row < n; ++row) {
      const double vkp = vectors[matrix_index(row, p, n)];
      const double vkq = vectors[matrix_index(row, q, n)];
      vectors[matrix_index(row, p, n)] = cosine * vkp - sine * vkq;
      vectors[matrix_index(row, q, n)] = sine * vkp + cosine * vkq;
    }
  }

  // Stable selection sort keeps the same ascending eigenpair convention used
  // by the CPU oracle and cuSOLVER path.
  for (std::size_t column = 0; column < n; ++column) {
    std::size_t selected = column;
    for (std::size_t candidate = column + 1; candidate < n; ++candidate) {
      if (matrix[matrix_index(candidate, candidate, n)] <
          matrix[matrix_index(selected, selected, n)]) {
        selected = candidate;
      }
    }
    if (selected != column) {
      const double diagonal = matrix[matrix_index(column, column, n)];
      matrix[matrix_index(column, column, n)] = matrix[matrix_index(selected, selected, n)];
      matrix[matrix_index(selected, selected, n)] = diagonal;
      for (std::size_t row = 0; row < n; ++row) {
        const double swap = vectors[matrix_index(row, column, n)];
        vectors[matrix_index(row, column, n)] = vectors[matrix_index(row, selected, n)];
        vectors[matrix_index(row, selected, n)] = swap;
      }
    }
    eigenvalues[static_cast<std::size_t>(system) * n + column] =
        matrix[matrix_index(column, column, n)];
  }
  for (std::size_t element = 0; element < matrix_size; ++element) {
    matrices[offset + element] = vectors[element];
  }
}

/**
 * Graph-capture-safe Jacobi eigensolver for AO matrices above the provider's
 * small batched range.
 *
 * One block owns one physical or spin state. Threads cooperatively select the
 * largest off-diagonal element and apply its row/column rotation, while a
 * separate arena matrix retains eigenvectors. This keeps the device-tail SCF
 * loop intact for realistic named bases without cuSOLVER's capture-time host
 * synchronization or a fixed compile-time AO limit.
 */
__global__ void symmetric_eigen_graph_maximum_pivot_kernel(std::int32_t batch_size,
                                                           std::int32_t nbf, double* matrices,
                                                           double* eigenvectors,
                                                           double* eigenvalues, int* info,
                                                           const std::uint8_t* active) {
  static_assert(kGraphEigensolverThreads > 0 &&
                (kGraphEigensolverThreads & (kGraphEigensolverThreads - 1)) == 0);
  const std::int32_t state = static_cast<std::int32_t>(blockIdx.x);
  if (state >= batch_size) return;
  if (active != nullptr && active[state] == 0) {
    if (threadIdx.x == 0) info[state] = 0;
    return;
  }
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t matrix_offset = static_cast<std::size_t>(state) * matrix_size;
  const std::size_t eigenvalue_offset = static_cast<std::size_t>(state) * n;

  __shared__ double block_maximum[kGraphEigensolverThreads];
  __shared__ std::size_t block_index[kGraphEigensolverThreads];
  __shared__ std::size_t pivot_p;
  __shared__ std::size_t pivot_q;
  __shared__ double pivot_cosine;
  __shared__ double pivot_sine;
  __shared__ int converged;

  for (std::size_t element = threadIdx.x; element < matrix_size; element += blockDim.x) {
    const std::size_t row = element / n;
    const std::size_t column = element % n;
    eigenvectors[matrix_offset + element] = row == column ? 1.0 : 0.0;
  }
  if (threadIdx.x == 0) {
    info[state] = 1;
    converged = 0;
  }
  __syncthreads();

  const std::size_t maximum_rotations = 20 * matrix_size > 50 ? 20 * matrix_size : 50;
  for (std::size_t rotation = 0; rotation < maximum_rotations; ++rotation) {
    double local_maximum = 0.0;
    std::size_t local_index = 0;
    for (std::size_t element = threadIdx.x; element < matrix_size; element += blockDim.x) {
      const std::size_t row = element / n;
      const std::size_t column = element % n;
      if (row >= column) continue;
      const double candidate = fabs(matrices[matrix_offset + element]);
      if (candidate > local_maximum) {
        local_maximum = candidate;
        local_index = element;
      }
    }
    block_maximum[threadIdx.x] = local_maximum;
    block_index[threadIdx.x] = local_index;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride &&
          block_maximum[threadIdx.x + stride] > block_maximum[threadIdx.x]) {
        block_maximum[threadIdx.x] = block_maximum[threadIdx.x + stride];
        block_index[threadIdx.x] = block_index[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      if (block_maximum[0] < 1.0e-14) {
        converged = 1;
        info[state] = 0;
      } else {
        pivot_p = block_index[0] / n;
        pivot_q = block_index[0] % n;
        const double app = matrices[matrix_offset + matrix_index(pivot_p, pivot_p, n)];
        const double aqq = matrices[matrix_offset + matrix_index(pivot_q, pivot_q, n)];
        const double apq = matrices[matrix_offset + matrix_index(pivot_p, pivot_q, n)];
        const double angle = 0.5 * atan2(2.0 * apq, aqq - app);
        pivot_cosine = cos(angle);
        pivot_sine = sin(angle);
      }
    }
    __syncthreads();
    if (converged != 0) break;

    for (std::size_t k = threadIdx.x; k < n; k += blockDim.x) {
      if (k != pivot_p && k != pivot_q) {
        const double mkp = matrices[matrix_offset + matrix_index(k, pivot_p, n)];
        const double mkq = matrices[matrix_offset + matrix_index(k, pivot_q, n)];
        const double next_p = pivot_cosine * mkp - pivot_sine * mkq;
        const double next_q = pivot_sine * mkp + pivot_cosine * mkq;
        matrices[matrix_offset + matrix_index(k, pivot_p, n)] = next_p;
        matrices[matrix_offset + matrix_index(pivot_p, k, n)] = next_p;
        matrices[matrix_offset + matrix_index(k, pivot_q, n)] = next_q;
        matrices[matrix_offset + matrix_index(pivot_q, k, n)] = next_q;
      }
      const double vkp = eigenvectors[matrix_offset + matrix_index(k, pivot_p, n)];
      const double vkq = eigenvectors[matrix_offset + matrix_index(k, pivot_q, n)];
      eigenvectors[matrix_offset + matrix_index(k, pivot_p, n)] =
          pivot_cosine * vkp - pivot_sine * vkq;
      eigenvectors[matrix_offset + matrix_index(k, pivot_q, n)] =
          pivot_sine * vkp + pivot_cosine * vkq;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      const double app = matrices[matrix_offset + matrix_index(pivot_p, pivot_p, n)];
      const double aqq = matrices[matrix_offset + matrix_index(pivot_q, pivot_q, n)];
      const double apq = matrices[matrix_offset + matrix_index(pivot_p, pivot_q, n)];
      matrices[matrix_offset + matrix_index(pivot_p, pivot_p, n)] =
          pivot_cosine * pivot_cosine * app - 2.0 * pivot_sine * pivot_cosine * apq +
          pivot_sine * pivot_sine * aqq;
      matrices[matrix_offset + matrix_index(pivot_q, pivot_q, n)] =
          pivot_sine * pivot_sine * app + 2.0 * pivot_sine * pivot_cosine * apq +
          pivot_cosine * pivot_cosine * aqq;
      matrices[matrix_offset + matrix_index(pivot_p, pivot_q, n)] = 0.0;
      matrices[matrix_offset + matrix_index(pivot_q, pivot_p, n)] = 0.0;
    }
    __syncthreads();
  }

  // Stable selection sort preserves the ascending eigenpair convention of
  // both the CPU oracle and the two smaller CUDA solver paths.
  for (std::size_t column = 0; column < n; ++column) {
    if (threadIdx.x == 0) {
      std::size_t selected = column;
      for (std::size_t candidate = column + 1; candidate < n; ++candidate) {
        if (matrices[matrix_offset + matrix_index(candidate, candidate, n)] <
            matrices[matrix_offset + matrix_index(selected, selected, n)]) {
          selected = candidate;
        }
      }
      block_index[0] = selected;
      if (selected != column) {
        const double diagonal = matrices[matrix_offset + matrix_index(column, column, n)];
        matrices[matrix_offset + matrix_index(column, column, n)] =
            matrices[matrix_offset + matrix_index(selected, selected, n)];
        matrices[matrix_offset + matrix_index(selected, selected, n)] = diagonal;
      }
    }
    __syncthreads();
    const std::size_t selected = block_index[0];
    if (selected != column) {
      for (std::size_t row = threadIdx.x; row < n; row += blockDim.x) {
        const double swap = eigenvectors[matrix_offset + matrix_index(row, column, n)];
        eigenvectors[matrix_offset + matrix_index(row, column, n)] =
            eigenvectors[matrix_offset + matrix_index(row, selected, n)];
        eigenvectors[matrix_offset + matrix_index(row, selected, n)] = swap;
      }
    }
    __syncthreads();
  }
  for (std::size_t column = threadIdx.x; column < n; column += blockDim.x) {
    eigenvalues[eigenvalue_offset + column] =
        matrices[matrix_offset + matrix_index(column, column, n)];
  }
  __syncthreads();
  for (std::size_t element = threadIdx.x; element < matrix_size; element += blockDim.x) {
    matrices[matrix_offset + element] = eigenvectors[matrix_offset + element];
  }
}

}  // namespace

void launch_begin_inactive_eigensolver_profile_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::int32_t physical_batch_size, std::int32_t solver_batch_size, std::uint32_t family,
    bool provider_invoked, bool cublas_transformed_inactive, const std::uint8_t* physical_active,
    const std::uint8_t* solver_active, std::uint32_t capacity, std::uint32_t* count,
    DeviceInactiveEigensolverProfileEntry* entries) {
  begin_inactive_eigensolver_profile_kernel<<<grid, block, shared_bytes, stream>>>(
      physical_batch_size, solver_batch_size, family, provider_invoked, cublas_transformed_inactive,
      physical_active, solver_active, capacity, count, entries);
}

void launch_finish_inactive_eigensolver_profile_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::int32_t solver_batch_size, const std::uint8_t* solver_active, const int* info,
    std::uint32_t capacity, const std::uint32_t* count,
    DeviceInactiveEigensolverProfileEntry* entries) {
  finish_inactive_eigensolver_profile_kernel<<<grid, block, shared_bytes, stream>>>(
      solver_batch_size, solver_active, info, capacity, count, entries);
}

void launch_sanitize_inactive_solver_input_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
    std::int32_t solver_batch_size, std::int32_t nbf, const std::uint8_t* solver_active,
    double* matrices, int* info, std::uint32_t profile_capacity, const std::uint32_t* profile_count,
    DeviceInactiveEigensolverProfileEntry* profile_entries) {
  sanitize_inactive_solver_input_kernel<<<grid, block, shared_bytes, stream>>>(
      solver_batch_size, nbf, solver_active, matrices, info, profile_capacity, profile_count,
      profile_entries);
}

void launch_start_inactive_eigensolver_timer_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::uint32_t capacity,
    const std::uint32_t* count, DeviceInactiveEigensolverProfileEntry* entries) {
  start_inactive_eigensolver_timer_kernel<<<grid, block, shared_bytes, stream>>>(capacity, count,
                                                                                 entries);
}

void launch_symmetric_eigen_graph_maximum_pivot_kernel(dim3 grid, dim3 block,
                                                       std::size_t shared_bytes,
                                                       cudaStream_t stream, std::int32_t batch_size,
                                                       std::int32_t nbf, double* matrices,
                                                       double* eigenvectors, double* eigenvalues,
                                                       int* info, const std::uint8_t* active) {
  symmetric_eigen_graph_maximum_pivot_kernel<<<grid, block, shared_bytes, stream>>>(
      batch_size, nbf, matrices, eigenvectors, eigenvalues, info, active);
}

void launch_symmetric_eigen_small_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                         cudaStream_t stream, std::int32_t batch_size,
                                         std::int32_t nbf, double* matrices, double* eigenvalues,
                                         int* info, const std::uint8_t* active) {
  symmetric_eigen_small_kernel<<<grid, block, shared_bytes, stream>>>(batch_size, nbf, matrices,
                                                                      eigenvalues, info, active);
}

}  // namespace vibeqc::scf::cuda_execution
