#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "runtime/cuda_component_trace.hpp"
#include "scf/cuda/df_jk_internal.hpp"
#include "scf/cuda/df_jk_kernels.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"

namespace vibeqc::scf::cuda_df {
using runtime::cuda_trace::trace_call;
using runtime::cuda_trace::TraceOperation;

// Coulomb contraction preserves both passes over bounded source tiles.
vibeqc_status build_coulomb(CudaDensityFittingJkPlan& plan, const double* density,
                            std::string& detail) {
  TraceOperation trace(
      "ri_j", plan.stream,
      {plan.batch_size, plan.nbf, plan.naux, plan.integral_source != nullptr, plan.streamed});
  if (plan.streamed) {
    cudaError_t cuda_error = cudaMemsetAsync(
        plan.auxiliary_density, 0, plan.batch_size * plan.naux * sizeof(double), plan.stream);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemsetAsync(
          plan.coulomb, 0, plan.batch_size * plan.matrix_elements * sizeof(double), plan.stream);
    }
    if (cuda_error != cudaSuccess) {
      return cuda_failure(cuda_error, "zero streamed DF Coulomb buffers", detail);
    }
    if (plan.integral_source != nullptr) {
      // Source-backed replay regenerates each transformed tile on the same
      // stream.  No host raw tensor or pageable staging buffer is retained.
      const std::size_t pair_tile_capacity = plan.row_tile * plan.nbf;
      const std::size_t pair_tile = std::min(plan.ao_pair_tile, pair_tile_capacity);
      for (std::size_t system = 0; system < plan.batch_size; ++system) {
        for (std::size_t begin = 0; begin < plan.naux; begin += plan.auxiliary_tile) {
          const std::size_t count = std::min(plan.auxiliary_tile, plan.naux - begin);
          for (std::size_t pair_begin = 0; pair_begin < plan.matrix_elements;
               pair_begin += pair_tile) {
            const std::size_t pair_count = std::min(pair_tile, plan.matrix_elements - pair_begin);
            vibeqc_status source_status = generate_cuda_density_fitting_transformed_tile(
                plan.integral_source, system, pair_begin, pair_count, begin, count, -1,
                plan.inverse_square_roots + system * plan.naux * plan.naux,
                reinterpret_cast<void*>(plan.stream), plan.auxiliary_tile_values, detail);
            if (source_status != VIBEQC_STATUS_SUCCESS) return source_status;
            launch_accumulate_streamed_auxiliary_density_kernel(
                blocks_for(count), kThreads, 0, plan.stream, pair_count, count,
                plan.auxiliary_tile_values, density + system * plan.matrix_elements + pair_begin,
                plan.auxiliary_density + system * plan.naux + begin);
            cuda_error = cudaPeekAtLastError();
            if (cuda_error != cudaSuccess) {
              return cuda_failure(cuda_error, "source-backed DF auxiliary-density tile", detail);
            }
          }
          for (std::size_t pair_begin = 0; pair_begin < plan.matrix_elements;
               pair_begin += pair_tile) {
            const std::size_t pair_count = std::min(pair_tile, plan.matrix_elements - pair_begin);
            vibeqc_status source_status = generate_cuda_density_fitting_transformed_tile(
                plan.integral_source, system, pair_begin, pair_count, begin, count, -1,
                plan.inverse_square_roots + system * plan.naux * plan.naux,
                reinterpret_cast<void*>(plan.stream), plan.auxiliary_tile_values, detail);
            if (source_status != VIBEQC_STATUS_SUCCESS) return source_status;
            launch_build_streamed_coulomb_tile_kernel(
                blocks_for(pair_count), kThreads, 0, plan.stream, pair_count, count,
                plan.auxiliary_tile_values, plan.auxiliary_density + system * plan.naux + begin,
                plan.coulomb + system * plan.matrix_elements + pair_begin);
            cuda_error = cudaPeekAtLastError();
            if (cuda_error != cudaSuccess) {
              return cuda_failure(cuda_error, "source-backed DF Coulomb tile", detail);
            }
          }
        }
      }
      return VIBEQC_STATUS_SUCCESS;
    }
    // The staged device tile is sized from row_tile * nbf.  The planner's
    // AO-pair tile is a logical budget and may not be divisible by nbf, so
    // process at most the physically allocated row-major capacity per pass.
    const std::size_t pair_tile_capacity = plan.row_tile * plan.nbf;
    const std::size_t pair_tile = std::min(plan.ao_pair_tile, pair_tile_capacity);
    std::vector<double> host_tile;
    try {
      host_tile.resize(plan.auxiliary_tile * pair_tile_capacity);
    } catch (const std::bad_alloc&) {
      detail = "host allocation for streamed CUDA DF Coulomb tile failed";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
    for (std::size_t system = 0; system < plan.batch_size; ++system) {
      const double* raw =
          plan.streamed_raw_three_center.data() + system * plan.tensor_elements_per_system;
      const double* inverse =
          plan.streamed_inverse_square_roots.data() + system * plan.naux * plan.naux;
      const double* system_density = density + system * plan.matrix_elements;
      for (std::size_t begin = 0; begin < plan.naux; begin += plan.auxiliary_tile) {
        const std::size_t count = std::min(plan.auxiliary_tile, plan.naux - begin);
        // First accumulate the auxiliary density over bounded AO-pair tiles.
        for (std::size_t pair_begin = 0; pair_begin < plan.matrix_elements;
             pair_begin += pair_tile) {
          const std::size_t pair_count = std::min(pair_tile, plan.matrix_elements - pair_begin);
          for (std::size_t pair = 0; pair < pair_count; ++pair) {
            for (std::size_t auxiliary = 0; auxiliary < count; ++auxiliary) {
              double value = 0.0;
              for (std::size_t source = 0; source < plan.naux; ++source) {
                value += raw[(pair_begin + pair) * plan.naux + source] *
                         inverse[(begin + auxiliary) * plan.naux + source];
              }
              // Keep auxiliary contiguous within each pair so the same
              // staging layout is usable by the streamed contraction kernel.
              host_tile[pair * count + auxiliary] = value;
            }
          }
          const std::size_t tile_bytes = pair_count * count * sizeof(double);
          cuda_error = cudaMemcpyAsync(plan.auxiliary_tile_values, host_tile.data(), tile_bytes,
                                       cudaMemcpyHostToDevice, plan.stream);
          if (cuda_error != cudaSuccess) {
            return cuda_failure(cuda_error, "upload streamed DF Coulomb pair tile", detail);
          }
          launch_accumulate_streamed_auxiliary_density_kernel(
              blocks_for(count), kThreads, 0, plan.stream, pair_count, count,
              plan.auxiliary_tile_values, system_density + pair_begin,
              plan.auxiliary_density + system * plan.naux + begin);
          cuda_error = cudaPeekAtLastError();
          if (cuda_error != cudaSuccess) {
            return cuda_failure(cuda_error, "streamed DF auxiliary-density pair tile", detail);
          }
          cuda_error = cudaStreamSynchronize(plan.stream);
          if (cuda_error != cudaSuccess) {
            return cuda_failure(cuda_error, "finish streamed DF Coulomb pair tile", detail);
          }
        }

        // Revisit the same bounded tiles to form the AO Coulomb output.  AO
        // segments are disjoint, while auxiliary tiles accumulate into the
        // same output through the single stream (so no atomics are needed).
        for (std::size_t pair_begin = 0; pair_begin < plan.matrix_elements;
             pair_begin += pair_tile) {
          const std::size_t pair_count = std::min(pair_tile, plan.matrix_elements - pair_begin);
          for (std::size_t pair = 0; pair < pair_count; ++pair) {
            for (std::size_t auxiliary = 0; auxiliary < count; ++auxiliary) {
              double value = 0.0;
              for (std::size_t source = 0; source < plan.naux; ++source) {
                value += raw[(pair_begin + pair) * plan.naux + source] *
                         inverse[(begin + auxiliary) * plan.naux + source];
              }
              host_tile[pair * count + auxiliary] = value;
            }
          }
          const std::size_t tile_bytes = pair_count * count * sizeof(double);
          cuda_error = cudaMemcpyAsync(plan.auxiliary_tile_values, host_tile.data(), tile_bytes,
                                       cudaMemcpyHostToDevice, plan.stream);
          if (cuda_error != cudaSuccess) {
            return cuda_failure(cuda_error, "upload streamed DF Coulomb output tile", detail);
          }
          launch_build_streamed_coulomb_tile_kernel(
              blocks_for(pair_count), kThreads, 0, plan.stream, pair_count, count,
              plan.auxiliary_tile_values, plan.auxiliary_density + system * plan.naux + begin,
              plan.coulomb + system * plan.matrix_elements + pair_begin);
          cuda_error = cudaPeekAtLastError();
          if (cuda_error != cudaSuccess) {
            return cuda_failure(cuda_error, "streamed DF Coulomb output tile", detail);
          }
          cuda_error = cudaStreamSynchronize(plan.stream);
          if (cuda_error != cudaSuccess) {
            return cuda_failure(cuda_error, "finish streamed DF Coulomb output tile", detail);
          }
        }
      }
    }
    return VIBEQC_STATUS_SUCCESS;
  }

  const int batch_size = static_cast<int>(plan.batch_size);
  const int matrix_elements = static_cast<int>(plan.matrix_elements);
  const int naux = static_cast<int>(plan.naux);
  const long long tensor_stride = static_cast<long long>(plan.tensor_elements_per_system);
  const long long matrix_stride = static_cast<long long>(plan.matrix_elements);
  const long long auxiliary_stride = static_cast<long long>(plan.naux);
  const double one = 1.0;
  const double zero = 0.0;
  cublasStatus_t blas_status = trace_call("ri_j_gemm", plan.stream, [&] {
    return cublasDgemmStridedBatched(plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, naux, 1, matrix_elements,
                                     &one, plan.three_center, naux, tensor_stride, density,
                                     matrix_elements, matrix_stride, &zero, plan.auxiliary_density,
                                     naux, auxiliary_stride, batch_size);
  });
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return blas_failure(blas_status, "DF auxiliary-density contraction", detail);
  }
  blas_status = trace_call("ri_j_gemm", plan.stream, [&] {
    return cublasDgemmStridedBatched(plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, matrix_elements, 1, naux,
                                     &one, plan.three_center, naux, tensor_stride,
                                     plan.auxiliary_density, naux, auxiliary_stride, &zero,
                                     plan.coulomb, matrix_elements, matrix_stride, batch_size);
  });
  return blas_status == CUBLAS_STATUS_SUCCESS
             ? VIBEQC_STATUS_SUCCESS
             : blas_failure(blas_status, "DF Coulomb contraction", detail);
}

}  // namespace vibeqc::scf::cuda_df
