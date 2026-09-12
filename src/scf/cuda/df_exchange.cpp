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
#include "scf/cuda/df_generated_tiles.hpp"
#include "scf/cuda/df_jk_internal.hpp"
#include "scf/cuda/df_jk_kernels.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"

namespace vibeqc::scf::cuda_df {
using runtime::cuda_trace::trace_call;
using runtime::cuda_trace::TraceOperation;

// Exchange contraction preserves row/auxiliary tiling and cuBLAS layout.
vibeqc_status build_exchange(CudaDensityFittingJkPlan& plan, const double* density,
                             double* exchange, std::string& detail, bool density_is_column_major) {
  TraceOperation trace(
      "ri_k", plan.stream,
      {plan.batch_size, plan.nbf, plan.naux, plan.integral_source != nullptr, plan.streamed});
  const std::size_t output_elements = plan.batch_size * plan.matrix_elements;
  cudaError_t cuda_error =
      cudaMemsetAsync(exchange, 0, output_elements * sizeof(double), plan.stream);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "zero DF exchange output", detail);
  }

  const double one = 1.0;
  const double zero = 0.0;

  // The resident path keeps the original full-matrix batched GEMMs.  The
  // streamed path below uses bounded AO row blocks, so every device tile is
  // at most row_tile * nbf rather than auxiliary_tile * nbf^2.
  if (plan.streamed) {
    if (plan.integral_source != nullptr) {
      // Spend the same four tile buffers on full AO panels when one matrix
      // fits. Smaller auxiliary panels remove all repeated column generation;
      // exceptionally tight budgets retain the bounded row traversal below.
      const auto capacity = plan.row_tile * plan.nbf * plan.auxiliary_tile;
      const bool full_pairs = capacity >= plan.matrix_elements;
      const auto row_tile = full_pairs ? plan.nbf : plan.row_tile;
      const auto auxiliary_tile =
          full_pairs ? std::min(plan.auxiliary_tile, capacity / plan.matrix_elements)
                     : plan.auxiliary_tile;
      const auto pair_capacity = row_tile * plan.nbf;
      const auto row_tiles = (plan.nbf + row_tile - 1) / row_tile;
      for (std::size_t system = 0; system < plan.batch_size; ++system) {
        const double* system_density = density + system * plan.matrix_elements;
        const double* density_column_major = system_density;
        if (!density_is_column_major) {
          double* transposed_density =
              plan.exchange_density_column_major + system * plan.matrix_elements;
          density_column_major = transposed_density;
          launch_transpose_density_kernel(blocks_for(plan.matrix_elements), kThreads, 0,
                                          plan.stream, plan.nbf, system_density,
                                          transposed_density);
          cuda_error = cudaPeekAtLastError();
          if (cuda_error != cudaSuccess) {
            return cuda_failure(cuda_error, "transpose source-backed DF exchange density", detail);
          }
        }
        for (std::size_t auxiliary_begin = 0; auxiliary_begin < plan.naux;
             auxiliary_begin += auxiliary_tile) {
          const std::size_t auxiliary_count = std::min(auxiliary_tile, plan.naux - auxiliary_begin);
          for (std::size_t row_begin = 0; row_begin < plan.nbf; row_begin += row_tile) {
            const std::size_t row_count = std::min(row_tile, plan.nbf - row_begin);
            const std::size_t pair_count = row_count * plan.nbf;
            if (pair_count > pair_capacity) {
              return VIBEQC_STATUS_INTERNAL_ERROR;
            }
            vibeqc_status source_status = generate_metric_panel(
                plan, system, row_begin * plan.nbf, pair_count, auxiliary_begin, auxiliary_count,
                plan.auxiliary_tile_values, plan.exchange_tile_output, detail);
            if (source_status != VIBEQC_STATUS_SUCCESS) return source_status;
            launch_transpose_streamed_df_tile_kernel(
                blocks_for(pair_count * auxiliary_count), kThreads, 0, plan.stream, pair_count,
                auxiliary_count, plan.auxiliary_tile_values, plan.exchange_intermediate);
            cuda_error = cudaPeekAtLastError();
            if (cuda_error != cudaSuccess) {
              return cuda_failure(cuda_error, "transpose source-backed DF exchange row tile",
                                  detail);
            }
            cublasStatus_t blas_status = trace_call("ri_k_gemm", plan.stream, [&] {
              return cublasDgemmStridedBatched(
                  plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(plan.nbf),
                  static_cast<int>(row_count), static_cast<int>(plan.nbf), &one,
                  density_column_major, static_cast<int>(plan.nbf), 0, plan.exchange_intermediate,
                  static_cast<int>(plan.nbf), static_cast<long long>(pair_count), &zero,
                  plan.exchange_contributions, static_cast<int>(plan.nbf),
                  static_cast<long long>(pair_count), static_cast<int>(auxiliary_count));
            });
            if (blas_status != CUBLAS_STATUS_SUCCESS) {
              return blas_failure(blas_status, "source-backed DF exchange row GEMM", detail);
            }
            // Consume the row's own transposed values before another column
            // overwrites them. Output blocks are disjoint, so this visitation
            // order preserves each element's auxiliary accumulation order.
            for (std::size_t column_tile = 0; column_tile < row_tiles; ++column_tile) {
              const auto column_begin =
                  ((row_begin / row_tile + column_tile) % row_tiles) * row_tile;
              const auto column_count = std::min(row_tile, plan.nbf - column_begin);
              const auto column_pair_count = column_count * plan.nbf;
              if (column_tile != 0) {
                source_status = generate_metric_panel(
                    plan, system, column_begin * plan.nbf, column_pair_count, auxiliary_begin,
                    auxiliary_count, plan.auxiliary_tile_values, plan.exchange_tile_output, detail);
                if (source_status != VIBEQC_STATUS_SUCCESS) return source_status;
                launch_transpose_streamed_df_tile_kernel(
                    blocks_for(column_pair_count * auxiliary_count), kThreads, 0, plan.stream,
                    column_pair_count, auxiliary_count, plan.auxiliary_tile_values,
                    plan.exchange_intermediate);
                cuda_error = cudaPeekAtLastError();
                if (cuda_error != cudaSuccess)
                  return cuda_failure(cuda_error, "transpose source-backed DF exchange column tile",
                                      detail);
              } else {
                runtime::cuda_trace::trace_counter("transformed_tile_cache_hits", 1);
              }
              const std::size_t output_stride = row_count * column_count;
              blas_status = trace_call("ri_k_gemm", plan.stream, [&] {
                return cublasDgemmStridedBatched(
                    plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(row_count),
                    static_cast<int>(column_count), static_cast<int>(plan.nbf), &one,
                    plan.exchange_contributions, static_cast<int>(plan.nbf),
                    static_cast<long long>(pair_count), plan.exchange_intermediate,
                    static_cast<int>(plan.nbf), static_cast<long long>(column_pair_count), &zero,
                    plan.exchange_tile_output, static_cast<int>(row_count),
                    static_cast<long long>(output_stride), static_cast<int>(auxiliary_count));
              });
              if (blas_status != CUBLAS_STATUS_SUCCESS) {
                return blas_failure(blas_status, "source-backed DF exchange column GEMM", detail);
              }
              launch_reduce_exchange_row_tile_kernel(blocks_for(output_stride), kThreads, 0,
                                                     plan.stream, plan.nbf, row_begin, row_count,
                                                     column_begin, column_count, auxiliary_count,
                                                     system, plan.exchange_tile_output, exchange);
              cuda_error = cudaPeekAtLastError();
              if (cuda_error != cudaSuccess) {
                return cuda_failure(cuda_error, "reduce source-backed DF exchange tile", detail);
              }
            }
          }
        }
      }
      return VIBEQC_STATUS_SUCCESS;
    }
    const std::size_t pair_capacity = plan.row_tile * plan.nbf;
    std::vector<double> host_tile;
    try {
      host_tile.resize(plan.auxiliary_tile * pair_capacity);
    } catch (const std::bad_alloc&) {
      detail = "host allocation for streamed CUDA DF exchange tile failed";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }

    // cuBLAS consumes column-major matrices while public host densities and
    // streamed raw tensors are row-major.  Transpose host inputs explicitly;
    // device-resident SCF callers can opt out when their density is already
    // in cuBLAS layout.  This keeps rectangular row-block GEMMs independent
    // of density-symmetry assumptions.
    for (std::size_t system = 0; system < plan.batch_size; ++system) {
      const double* system_density = density + system * plan.matrix_elements;
      const double* density_column_major = system_density;
      if (!density_is_column_major) {
        density_column_major = plan.exchange_density_column_major + system * plan.matrix_elements;
        launch_transpose_density_kernel(
            blocks_for(plan.matrix_elements), kThreads, 0, plan.stream, plan.nbf, system_density,
            plan.exchange_density_column_major + system * plan.matrix_elements);
        cudaError_t transpose_error = cudaPeekAtLastError();
        if (transpose_error != cudaSuccess) {
          return cuda_failure(transpose_error, "transpose streamed DF exchange density", detail);
        }
      }

      const double* raw =
          plan.streamed_raw_three_center.data() + system * plan.tensor_elements_per_system;
      const double* inverse =
          plan.streamed_inverse_square_roots.data() + system * plan.naux * plan.naux;
      for (std::size_t auxiliary_begin = 0; auxiliary_begin < plan.naux;
           auxiliary_begin += plan.auxiliary_tile) {
        const std::size_t auxiliary_count =
            std::min(plan.auxiliary_tile, plan.naux - auxiliary_begin);

        for (std::size_t row_begin = 0; row_begin < plan.nbf; row_begin += plan.row_tile) {
          const std::size_t row_count = std::min(plan.row_tile, plan.nbf - row_begin);
          const std::size_t pair_count = row_count * plan.nbf;
          // Use the compact stride for this (possibly partial) row block.
          // The staging buffer is allocated for pair_capacity, but copying a
          // partial final block with that larger stride would include gaps
          // between auxiliary tiles and truncate the later tiles on device.
          const std::size_t row_stride = pair_count;

          // Transform this AO-row block for the active auxiliary tile.  The
          // host staging layout is [auxiliary][row][column] row-major; cuBLAS
          // interprets each block as the column-major transpose B_A^T.
          for (std::size_t auxiliary = 0; auxiliary < auxiliary_count; ++auxiliary) {
            for (std::size_t row = 0; row < row_count; ++row) {
              for (std::size_t column = 0; column < plan.nbf; ++column) {
                const std::size_t pair = (row_begin + row) * plan.nbf + column;
                double value = 0.0;
                for (std::size_t source = 0; source < plan.naux; ++source) {
                  value += raw[pair * plan.naux + source] *
                           inverse[(auxiliary_begin + auxiliary) * plan.naux + source];
                }
                host_tile[auxiliary * row_stride + row * plan.nbf + column] = value;
              }
            }
          }
          cuda_error = cudaMemcpyAsync(plan.auxiliary_tile_values, host_tile.data(),
                                       auxiliary_count * row_stride * sizeof(double),
                                       cudaMemcpyHostToDevice, plan.stream);
          if (cuda_error != cudaSuccess) {
            return cuda_failure(cuda_error, "upload streamed DF exchange row tile", detail);
          }

          // T_A = D^T * B_A^T, stored as nbf x row_count column-major.  The
          // transpose is B_A * D, which is the left factor of K_AB.
          cublasStatus_t blas_status = trace_call("ri_k_gemm", plan.stream, [&] {
            return cublasDgemmStridedBatched(
                plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(plan.nbf),
                static_cast<int>(row_count), static_cast<int>(plan.nbf), &one, density_column_major,
                static_cast<int>(plan.nbf), 0, plan.auxiliary_tile_values,
                static_cast<int>(plan.nbf), static_cast<long long>(row_stride), &zero,
                plan.exchange_contributions, static_cast<int>(plan.nbf),
                static_cast<long long>(row_stride), static_cast<int>(auxiliary_count));
          });
          if (blas_status != CUBLAS_STATUS_SUCCESS) {
            return blas_failure(blas_status, "streamed DF exchange row GEMM", detail);
          }

          for (std::size_t column_begin = 0; column_begin < plan.nbf;
               column_begin += plan.row_tile) {
            const std::size_t column_count = std::min(plan.row_tile, plan.nbf - column_begin);
            const std::size_t column_pair_count = column_count * plan.nbf;
            const std::size_t column_stride = column_pair_count;

            for (std::size_t auxiliary = 0; auxiliary < auxiliary_count; ++auxiliary) {
              for (std::size_t row = 0; row < column_count; ++row) {
                for (std::size_t column = 0; column < plan.nbf; ++column) {
                  const std::size_t pair = (column_begin + row) * plan.nbf + column;
                  double value = 0.0;
                  for (std::size_t source = 0; source < plan.naux; ++source) {
                    value += raw[pair * plan.naux + source] *
                             inverse[(auxiliary_begin + auxiliary) * plan.naux + source];
                  }
                  host_tile[auxiliary * column_stride + row * plan.nbf + column] = value;
                }
              }
            }
            cuda_error = cudaMemcpyAsync(plan.exchange_intermediate, host_tile.data(),
                                         auxiliary_count * column_stride * sizeof(double),
                                         cudaMemcpyHostToDevice, plan.stream);
            if (cuda_error != cudaSuccess) {
              return cuda_failure(cuda_error, "upload streamed DF exchange column tile", detail);
            }

            // K_AB = (B_A D) B_B^T.  T_A is stored as D^T B_A^T, so OP_T on
            // it gives B_A D; B_B is stored as B_B^T.  The result is emitted
            // as a column-major [row_count x column_count] tile and the
            // reduction kernel transposes that view while scattering.
            const std::size_t output_stride = row_count * column_count;
            blas_status = trace_call("ri_k_gemm", plan.stream, [&] {
              return cublasDgemmStridedBatched(
                  plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(row_count),
                  static_cast<int>(column_count), static_cast<int>(plan.nbf), &one,
                  plan.exchange_contributions, static_cast<int>(plan.nbf),
                  static_cast<long long>(row_stride), plan.exchange_intermediate,
                  static_cast<int>(plan.nbf), static_cast<long long>(column_stride), &zero,
                  plan.exchange_tile_output, static_cast<int>(row_count),
                  static_cast<long long>(output_stride), static_cast<int>(auxiliary_count));
            });
            if (blas_status != CUBLAS_STATUS_SUCCESS) {
              return blas_failure(blas_status, "streamed DF exchange column GEMM", detail);
            }
            launch_reduce_exchange_row_tile_kernel(blocks_for(output_stride), kThreads, 0,
                                                   plan.stream, plan.nbf, row_begin, row_count,
                                                   column_begin, column_count, auxiliary_count,
                                                   system, plan.exchange_tile_output, exchange);
            cuda_error = cudaPeekAtLastError();
            if (cuda_error != cudaSuccess) {
              return cuda_failure(cuda_error, "reduce streamed DF exchange tile", detail);
            }

            // Host staging and the two reusable device input buffers are
            // overwritten on the next column block; fence this block before
            // reusing them.  This is intentionally conservative and keeps
            // correctness independent of pageable-host-copy behavior.
            cuda_error = cudaStreamSynchronize(plan.stream);
            if (cuda_error != cudaSuccess) {
              return cuda_failure(cuda_error, "finish streamed DF exchange tile", detail);
            }
          }
        }
      }
    }
    return VIBEQC_STATUS_SUCCESS;
  }

  const int nbf = static_cast<int>(plan.nbf);
  const long long matrix_stride = static_cast<long long>(plan.matrix_elements);
  for (std::size_t system = 0; system < plan.batch_size; ++system) {
    const double* system_density = density + system * plan.matrix_elements;
    double* transposed_density = plan.exchange_density_column_major + system * plan.matrix_elements;
    const double* density_column_major = system_density;
    if (!density_is_column_major) {
      density_column_major = transposed_density;
      launch_transpose_density_kernel(blocks_for(plan.matrix_elements), kThreads, 0, plan.stream,
                                      plan.nbf, system_density, transposed_density);
      cuda_error = cudaPeekAtLastError();
      if (cuda_error != cudaSuccess) {
        return cuda_failure(cuda_error, "transpose CUDA DF exchange density", detail);
      }
    }
    for (std::size_t auxiliary_begin = 0; auxiliary_begin < plan.naux;
         auxiliary_begin += plan.auxiliary_tile) {
      const std::size_t auxiliary_count =
          std::min(plan.auxiliary_tile, plan.naux - auxiliary_begin);
      const std::size_t tile_elements = auxiliary_count * plan.matrix_elements;
      launch_gather_auxiliary_tile_kernel(
          blocks_for(tile_elements), kThreads, 0, plan.stream, plan.matrix_elements, plan.naux,
          system, auxiliary_begin, auxiliary_count, plan.three_center, plan.auxiliary_tile_values);
      cuda_error = cudaPeekAtLastError();
      if (cuda_error != cudaSuccess) {
        return cuda_failure(cuda_error, "gather DF exchange tile", detail);
      }

      const int tile_count = static_cast<int>(auxiliary_count);
      cublasStatus_t blas_status = trace_call("ri_k_gemm", plan.stream, [&] {
        return cublasDgemmStridedBatched(
            // The gathered AO-pair tile is B^T in cuBLAS layout.  Use D^T as
            // the first factor; the reduction below maps the column-major
            // result back to row-major public storage, yielding B D B^T.
            plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, nbf, nbf, nbf, &one, density_column_major, nbf, 0,
            plan.auxiliary_tile_values, nbf, matrix_stride, &zero, plan.exchange_intermediate, nbf,
            matrix_stride, tile_count);
      });
      if (blas_status != CUBLAS_STATUS_SUCCESS) {
        return blas_failure(blas_status, "DF exchange first GEMM", detail);
      }
      blas_status = trace_call("ri_k_gemm", plan.stream, [&] {
        return cublasDgemmStridedBatched(
            plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, nbf, nbf, nbf, &one, plan.auxiliary_tile_values,
            nbf, matrix_stride, plan.exchange_intermediate, nbf, matrix_stride, &zero,
            plan.exchange_contributions, nbf, matrix_stride, tile_count);
      });
      if (blas_status != CUBLAS_STATUS_SUCCESS) {
        return blas_failure(blas_status, "DF exchange second GEMM", detail);
      }
      launch_reduce_exchange_tile_kernel(blocks_for(plan.matrix_elements), kThreads, 0, plan.stream,
                                         plan.matrix_elements, auxiliary_count, system,
                                         plan.exchange_contributions, exchange);
      cuda_error = cudaPeekAtLastError();
      if (cuda_error != cudaSuccess) {
        return cuda_failure(cuda_error, "reduce DF exchange tile", detail);
      }
    }
  }
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::scf::cuda_df
