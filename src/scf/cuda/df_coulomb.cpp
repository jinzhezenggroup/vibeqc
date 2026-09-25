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
#include "runtime/df_progress_trace.hpp"
#include "scf/cuda/df_jk_internal.hpp"
#include "scf/cuda/df_jk_kernels.hpp"
#include "scf/cuda/df_metric_kernels.hpp"
#include "scf/cuda/df_packed_values.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"

namespace vibeqc::scf::cuda_df {
using runtime::cuda_trace::trace_call;
using runtime::cuda_trace::TraceOperation;

// Coulomb needs two bounded raw passes unless an admitted K traversal supplies its charge.
vibeqc_status build_coulomb(CudaDensityFittingJkPlan& plan, const double* density,
                            std::string& detail, bool raw_charge_ready) {
  plan.final_projection_token.reset();
  if (raw_charge_ready && (!plan.streamed || !plan.integral_source || plan.batch_size != 1 ||
                           plan.metric_full_rank.empty() || !plan.metric_full_rank[0] ||
                           plan.row_tile * plan.nbf * plan.auxiliary_tile < plan.naux)) {
    detail = "shared DF charge requires a full-rank, factor-first streamed source";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  TraceOperation trace(
      "ri_j", plan.stream,
      {plan.batch_size, plan.nbf, plan.naux, plan.integral_source != nullptr, plan.streamed});
  if (plan.value_storage.pairs == DfPairStorage::SymmetricLower) {
    const double one = 1, zero = 0;
    const auto a = static_cast<int>(plan.naux), pairs = static_cast<int>(plan.stored_pair_count);
    // The packed density dies after rho=B*d. Its same scratch is then the
    // packed J output, so no per-batch pair vectors or allocation are needed.
    auto* packed = plan.auxiliary_tile_values;
    for (std::size_t system = 0; system < plan.batch_size; ++system) {
      launch_pack_df_density(plan.stream, plan.nbf, density + system * plan.matrix_elements,
                             packed);
      auto error = cudaPeekAtLastError();
      if (error != cudaSuccess) return cuda_failure(error, "pack DF Coulomb density", detail);
      const auto* b = plan.three_center + system * plan.stored_tensor_elements_per_system;
      auto* charge = plan.auxiliary_density + system * plan.naux;
      auto status = trace_call("ri_j_gemm", plan.stream, [&] {
        return cublasDgemv(plan.blas, CUBLAS_OP_N, a, pairs, &one, b, a, packed, 1, &zero, charge,
                           1);
      });
      if (status != CUBLAS_STATUS_SUCCESS)
        return blas_failure(status, "packed DF charge contraction", detail);
      status = trace_call("ri_j_gemm", plan.stream, [&] {
        return cublasDgemv(plan.blas, CUBLAS_OP_T, a, pairs, &one, b, a, charge, 1, &zero, packed,
                           1);
      });
      if (status != CUBLAS_STATUS_SUCCESS)
        return blas_failure(status, "packed DF Coulomb contraction", detail);
      launch_scatter_df_pairs(plan.stream, plan.nbf, packed,
                              plan.coulomb + system * plan.matrix_elements);
      error = cudaPeekAtLastError();
      if (error != cudaSuccess) return cuda_failure(error, "scatter packed DF Coulomb", detail);
    }
    runtime::cuda_trace::trace_counter("value_packed_pairs", plan.stored_pair_count);
    runtime::cuda_trace::trace_counter("coulomb_contraction_flops",
                                       4 * plan.batch_size * plan.stored_pair_count * plan.naux);
    return VIBEQC_STATUS_SUCCESS;
  }
  if (plan.streamed) {
    cudaError_t cuda_error = cudaSuccess;
    if (!raw_charge_ready)
      cuda_error = cudaMemsetAsync(plan.auxiliary_density, 0,
                                   plan.batch_size * plan.naux * sizeof(double), plan.stream);
    if (cuda_error == cudaSuccess)
      cuda_error = cudaMemsetAsync(
          plan.coulomb, 0, plan.batch_size * plan.matrix_elements * sizeof(double), plan.stream);
    if (cuda_error != cudaSuccess) {
      return cuda_failure(cuda_error, "zero streamed DF Coulomb buffers", detail);
    }
    if (plan.integral_source != nullptr) {
      // J = A X X^T A^T D. Contract raw charge panels first, then apply
      // metric factors to vectors. J alone needs two raw-tensor passes;
      // admitted occupied K can provide the first pass without materializing
      // transformed three-center tiles just to reduce them into one vector.
      const auto pair_tile = std::min(plan.ao_pair_tile, plan.row_tile * plan.nbf);
      runtime::df_progress::number("planner_ao_pair_tile", plan.ao_pair_tile);
      runtime::df_progress::number("executed_ao_pairs", pair_tile);
      runtime::df_progress::number("executed_auxiliary_tile", plan.auxiliary_tile);
      runtime::df_progress::number("raw_tensor_passes", raw_charge_ready ? 1 : 2);
      const double one = 1.0, zero = 0.0;
      for (std::size_t system = 0; system < plan.batch_size; ++system) {
        const auto* inverse = plan.inverse_square_roots + system * plan.naux * plan.naux;
        auto* charge = plan.auxiliary_density + system * plan.naux;
        const bool factor_first = plan.metric_full_rank[system] &&
                                  plan.row_tile * plan.nbf * plan.auxiliary_tile >= plan.naux;
        // The small panel charge/potential borrows K scratch; only the first
        // auxiliary_count entries are live, within every resolved tile size.
        auto* panel_vector = plan.exchange_intermediate;
        if (!raw_charge_ready) {
          for (std::size_t begin = 0; begin < plan.naux; begin += plan.auxiliary_tile) {
            const auto count = std::min(plan.auxiliary_tile, plan.naux - begin);
            auto* raw_charge = factor_first ? charge + begin : panel_vector;
            cuda_error = cudaMemsetAsync(raw_charge, 0, count * sizeof(double), plan.stream);
            if (cuda_error != cudaSuccess)
              return cuda_failure(cuda_error, "zero raw DF panel charge", detail);
            for (std::size_t pair_begin = 0; pair_begin < plan.matrix_elements;
                 pair_begin += pair_tile) {
              const auto pairs = std::min(pair_tile, plan.matrix_elements - pair_begin);
              auto status = generate_cuda_density_fitting_raw_tile(
                  plan.integral_source, system, pair_begin, pairs, begin, count, -1,
                  reinterpret_cast<void*>(plan.stream), plan.auxiliary_tile_values, detail);
              if (status != VIBEQC_STATUS_SUCCESS) return status;
              launch_accumulate_streamed_auxiliary_density_kernel(
                  blocks_for(count), kThreads, 0, plan.stream, pairs, count,
                  plan.auxiliary_tile_values, density + system * plan.matrix_elements + pair_begin,
                  raw_charge);
              cuda_error = cudaPeekAtLastError();
              if (cuda_error != cudaSuccess)
                return cuda_failure(cuda_error, "raw DF panel charge contraction", detail);
            }
            if (factor_first) continue;
            const auto status = trace_call("ri_j_metric_gemm", plan.stream, [&] {
              return cublasDgemv(plan.blas, CUBLAS_OP_N, static_cast<int>(plan.naux),
                                 static_cast<int>(count), &one, inverse + begin * plan.naux,
                                 static_cast<int>(plan.naux), panel_vector, 1, &one, charge, 1);
            });
            if (status != CUBLAS_STATUS_SUCCESS)
              return blas_failure(status, "transform raw DF panel charge", detail);
          }
        }
        if (factor_first) {
          // Keep eigendirections until after division. Only two length-Naux
          // vectors are live, in the charge owner and existing K scratch;
          // source work stays at the same two complete raw-tensor passes.
          const auto a = static_cast<int>(plan.naux);
          const auto* q = plan.metric_eigenvectors + system * plan.naux * plan.naux;
          auto status = trace_call("ri_j_metric_eigen_projection", plan.stream, [&] {
            return cublasDgemv(plan.blas, CUBLAS_OP_T, a, a, &one, q, a, charge, 1, &zero,
                               panel_vector, 1);
          });
          if (status != CUBLAS_STATUS_SUCCESS)
            return blas_failure(status, "project raw DF charge eigendirections", detail);
          launch_scale_metric_projection(plan.stream, plan.naux, 1,
                                         plan.metric_eigenvalues + system * plan.naux, false,
                                         panel_vector);
          cuda_error = cudaPeekAtLastError();
          if (cuda_error != cudaSuccess)
            return cuda_failure(cuda_error, "scale raw DF charge eigendirections", detail);
          status = trace_call("ri_j_metric_scaled_rotation", plan.stream, [&] {
            return cublasDgemv(plan.blas, CUBLAS_OP_N, a, a, &one, q, a, panel_vector, 1, &zero,
                               charge, 1);
          });
          if (status != CUBLAS_STATUS_SUCCESS)
            return blas_failure(status, "rotate inverse-applied raw DF charge", detail);
          runtime::cuda_trace::trace_counter("streamed_coulomb_factor_first", 1);
          runtime::cuda_trace::trace_counter("streamed_coulomb_factor_gemvs", 2);
          runtime::cuda_trace::trace_counter("streamed_coulomb_factor_flops",
                                             4 * plan.naux * plan.naux);
        }
        for (std::size_t begin = 0; begin < plan.naux; begin += plan.auxiliary_tile) {
          const auto count = std::min(plan.auxiliary_tile, plan.naux - begin);
          if (!factor_first) {
            const auto status = trace_call("ri_j_metric_gemm", plan.stream, [&] {
              return cublasDgemv(plan.blas, CUBLAS_OP_N, static_cast<int>(count),
                                 static_cast<int>(plan.naux), &one, inverse + begin,
                                 static_cast<int>(plan.naux), charge, 1, &zero, panel_vector, 1);
            });
            if (status != CUBLAS_STATUS_SUCCESS)
              return blas_failure(status, "form raw DF panel potential", detail);
          }
          const auto* potential = factor_first ? charge + begin : panel_vector;
          for (std::size_t pair_begin = 0; pair_begin < plan.matrix_elements;
               pair_begin += pair_tile) {
            const auto pairs = std::min(pair_tile, plan.matrix_elements - pair_begin);
            auto source_status = generate_cuda_density_fitting_raw_tile(
                plan.integral_source, system, pair_begin, pairs, begin, count, -1,
                reinterpret_cast<void*>(plan.stream), plan.auxiliary_tile_values, detail);
            if (source_status != VIBEQC_STATUS_SUCCESS) return source_status;
            launch_build_streamed_coulomb_tile_kernel(
                blocks_for(pairs), kThreads, 0, plan.stream, pairs, count,
                plan.auxiliary_tile_values, potential,
                plan.coulomb + system * plan.matrix_elements + pair_begin);
            cuda_error = cudaPeekAtLastError();
            if (cuda_error != cudaSuccess)
              return cuda_failure(cuda_error, "raw DF Coulomb output contraction", detail);
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
