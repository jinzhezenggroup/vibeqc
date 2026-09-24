#include <math_constants.h>

#include <cmath>

#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/scf_diis_kernels.hpp"

namespace vibeqc::scf::cuda_execution {

namespace {
__device__ void decode_symmetric_pair(std::uint32_t pair, std::uint32_t width,
                                      std::uint32_t& row, std::uint32_t& column) {
  row = 0;
  auto row_width = width;
  while (pair >= row_width) {
    pair -= row_width;
    ++row;
    --row_width;
  }
  column = row + pair;
}

__global__ void diis_dot_partials_kernel(std::size_t vector_size, std::uint32_t history,
                                         const double* residual, const double* residual_history,
                                         const std::uint8_t* active, const std::uint32_t* counts,
                                         const std::uint32_t* heads, std::size_t parts,
                                         double* partials) {
  const auto system = blockIdx.z;
  if (!active[system]) return;
  __shared__ std::uint32_t pair_slots[2];
  if (threadIdx.x == 0) {
    decode_symmetric_pair(static_cast<std::uint32_t>(blockIdx.y), history, pair_slots[0],
                          pair_slots[1]);
  }
  __syncthreads();
  const auto row = pair_slots[0], column = pair_slots[1];
  const auto count = counts[system] < history ? counts[system] + 1 : history;
  // History retirement can leave a short live window anywhere in the ring.
  // Physical slot numbers therefore cannot be compared directly with count.
  const auto first = (heads[system] + 1 + history - count) % history;
  if (count < 2 || (row + history - first) % history >= count ||
      (column + history - first) % history >= count)
    return;
  // A not-yet-stored current residual logically occupies head. Other slots
  // remain read-only until this complete grid precedes the update kernel.
  const auto current = residual + system * vector_size;
  const auto base = residual_history + system * history * vector_size;
  const auto left = row == heads[system] ? current : base + row * vector_size;
  const auto right = column == heads[system] ? current : base + column * vector_size;
  const std::size_t begin = std::size_t{blockIdx.x} * 4096;
  const auto end = min(begin + 4096, vector_size);
  double value = 0;
  for (auto element = begin + threadIdx.x; element < end; element += blockDim.x)
    value += left[element] * right[element];
  for (unsigned delta = 16; delta; delta /= 2) value += __shfl_down_sync(0xffffffffU, value, delta);
  __shared__ double warps[8];
  if (threadIdx.x % 32 == 0) warps[threadIdx.x / 32] = value;
  __syncthreads();
  if (threadIdx.x < 32) {
    value = threadIdx.x < 8 ? warps[threadIdx.x] : 0;
    for (unsigned delta = 16; delta; delta /= 2)
      value += __shfl_down_sync(0xffffffffU, value, delta);
    if (threadIdx.x == 0) {
      const auto forward =
          ((static_cast<std::size_t>(system) * history + row) * history + column) * parts +
          blockIdx.x;
      partials[forward] = value;
      if (row != column) {
        const auto reverse =
            ((static_cast<std::size_t>(system) * history + column) * history + row) * parts +
            blockIdx.x;
        partials[reverse] = value;
      }
    }
  }
}
}  // namespace

void launch_diis_dot_partials(cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf,
                              std::int32_t spins, std::uint32_t history, const double* residual,
                              const double* residual_history, const std::uint8_t* active,
                              const std::uint32_t* counts, const std::uint32_t* heads,
                              std::size_t parts, double* partials) {
  const auto pair_blocks = history * (history + 1U) / 2U;
  diis_dot_partials_kernel<<<dim3(parts, pair_blocks, batch_size), 256, 0, stream>>>(
      static_cast<std::size_t>(nbf) * nbf * spins, history, residual, residual_history, active,
      counts, heads, parts, partials);
}

template <bool CooperativeDots>
__global__ void update_diis_kernel(std::int32_t batch_size, std::int32_t nbf,
                                   std::int32_t matrices_per_system, std::uint32_t history_capacity,
                                   const double* fock, const double* residual,
                                   const std::uint8_t* active, double* fock_history,
                                   double* residual_history, double* linear_system,
                                   double* coefficients, std::uint32_t* history_count,
                                   std::uint32_t* history_head, double* effective_fock,
                                   bool normalize_metric, const double* dot_partials,
                                   std::size_t parts) {
  constexpr bool cooperative_dots = CooperativeDots;
  // One warp owns one system.  History vectors and the O(N^2) residual-dot
  // products are distributed across lanes, while the small dense DIIS solve
  // remains in lane zero. The legacy instantiation preserves its dot order;
  // DF can instead consume deterministic parallel partials from free scratch.
  const std::int32_t system = static_cast<std::int32_t>(blockIdx.x);
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_size = static_cast<std::size_t>(nbf) * nbf;
  const std::size_t vector_size = matrix_size * static_cast<std::size_t>(matrices_per_system);
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * vector_size;
  if (history_capacity < 2) {
    for (std::size_t element = threadIdx.x; element < vector_size; element += blockDim.x) {
      effective_fock[matrix_offset + element] = fock[matrix_offset + element];
    }
    return;
  }

  const std::size_t history_stride = static_cast<std::size_t>(history_capacity) * vector_size;
  std::uint32_t slot = 0;
  if (threadIdx.x == 0) slot = history_head[system];
  slot = __shfl_sync(0xffffffffU, slot, 0);
  const std::size_t slot_offset = static_cast<std::size_t>(system) * history_stride +
                                  static_cast<std::size_t>(slot) * vector_size;
  for (std::size_t element = threadIdx.x; element < vector_size; element += blockDim.x) {
    fock_history[slot_offset + element] = fock[matrix_offset + element];
    residual_history[slot_offset + element] = residual[matrix_offset + element];
  }
  __syncwarp();
  std::uint32_t count = 0;
  if (threadIdx.x == 0) {
    count = history_count[system] < history_capacity ? history_count[system] + 1 : history_capacity;
    history_count[system] = count;
    history_head[system] = (slot + 1) % history_capacity;
  }
  count = __shfl_sync(0xffffffffU, count, 0);
  if (count < 2) {
    for (std::size_t element = threadIdx.x; element < vector_size; element += blockDim.x) {
      effective_fock[matrix_offset + element] = fock[matrix_offset + element];
    }
    return;
  }

  const std::size_t system_stride =
      static_cast<std::size_t>(history_capacity + 1) * (history_capacity + 1);
  double* matrix = linear_system + static_cast<std::size_t>(system) * system_stride;
  double* rhs = coefficients + static_cast<std::size_t>(system) * (history_capacity + 1);
  // Normalized KS DIIS retires the oldest dependent error and retries, as
  // CPU Diis does. Preserve chronological ring order without moving matrices.
  // Historical HF callers retain their unnormalized slot order and fallback.
  std::uint32_t first =
      normalize_metric ? (slot + 1 + history_capacity - count) % history_capacity : 0;
  int nonsingular = 1;
  for (;;) {
    const std::uint32_t dimension = count + 1;
    const std::size_t linear_elements = static_cast<std::size_t>(dimension) * dimension;
    for (std::size_t element = threadIdx.x; element < linear_elements; element += blockDim.x) {
      matrix[element] = 0.0;
    }
    for (std::uint32_t row = threadIdx.x; row < dimension; row += blockDim.x) {
      rhs[row] = row == count ? -1.0 : 0.0;
    }
    __syncwarp();
    const std::size_t dot_count = cooperative_dots
                                      ? static_cast<std::size_t>(count) * (count + 1U) / 2U
                                      : static_cast<std::size_t>(count) * count;
    // DF's compact solves have only a few history pairs on warm replays.
    // Assigning one lane per pair leaves nearly the whole warp idle while
    // each lane serially traverses nbf^2 values. The cooperative path also
    // exploits Gram symmetry and mirrors each upper-triangle result. The
    // small solve, common normalization and dependent-history retirement are
    // otherwise identical.
    for (std::size_t pair = cooperative_dots ? 0 : threadIdx.x; pair < dot_count;
         pair += cooperative_dots ? 1 : blockDim.x) {
      std::uint32_t row = 0, column = 0;
      if constexpr (CooperativeDots) {
        if (threadIdx.x == 0)
          decode_symmetric_pair(static_cast<std::uint32_t>(pair), count, row, column);
        row = __shfl_sync(0xffffffffU, row, 0);
        column = __shfl_sync(0xffffffffU, column, 0);
      } else {
        row = static_cast<std::uint32_t>(pair / count);
        column = static_cast<std::uint32_t>(pair % count);
      }
      const std::size_t row_offset =
          static_cast<std::size_t>(system) * history_stride +
          static_cast<std::size_t>((first + row) % history_capacity) * vector_size;
      const std::size_t column_offset =
          static_cast<std::size_t>(system) * history_stride +
          static_cast<std::size_t>((first + column) % history_capacity) * vector_size;
      double dot = 0.0;
      if (cooperative_dots && dot_partials) {
        const auto row_slot = (first + row) % history_capacity;
        const auto column_slot = (first + column) % history_capacity;
        const auto offset =
            ((static_cast<std::size_t>(system) * history_capacity + row_slot) * history_capacity +
             column_slot) *
            parts;
        for (std::size_t part = threadIdx.x; part < parts; part += blockDim.x)
          dot += dot_partials[offset + part];
      } else
        for (std::size_t element = cooperative_dots ? threadIdx.x : 0; element < vector_size;
             element += cooperative_dots ? blockDim.x : 1) {
          dot += residual_history[row_offset + element] * residual_history[column_offset + element];
        }
      if (cooperative_dots)
        for (unsigned delta = 16; delta; delta /= 2)
          dot += __shfl_down_sync(0xffffffffU, dot, delta);
      if (!cooperative_dots || threadIdx.x == 0) {
        matrix[static_cast<std::size_t>(row) * dimension + column] = dot;
        if (cooperative_dots && row != column)
          matrix[static_cast<std::size_t>(column) * dimension + row] = dot;
      }
    }
    __syncwarp();
    if (threadIdx.x == 0) {
      // A single common scale preserves the augmented DIIS solution. KS uses
      // this to avoid treating every residual below 1e-7 as an absolute-pivot
      // singularity. Existing HF callers retain the historical default path.
      if (normalize_metric) {
        double scale = 0.0;
        for (std::uint32_t row = 0; row < count; ++row)
          scale = fmax(scale, fabs(matrix[static_cast<std::size_t>(row) * dimension + row]));
        if (scale > 0.0 && isfinite(scale))
          for (std::uint32_t row = 0; row < count; ++row)
            for (std::uint32_t column = 0; column < count; ++column)
              matrix[static_cast<std::size_t>(row) * dimension + column] /= scale;
      }
      for (std::uint32_t row = 0; row < count; ++row) {
        matrix[static_cast<std::size_t>(row) * dimension + count] = -1.0;
        matrix[static_cast<std::size_t>(count) * dimension + row] = -1.0;
      }
    }
    __syncwarp();

    nonsingular = 1;
    if (threadIdx.x == 0) {
      for (std::uint32_t column = 0; column < dimension; ++column) {
        std::uint32_t pivot = column;
        for (std::uint32_t row = column + 1; row < dimension; ++row) {
          if (fabs(matrix[static_cast<std::size_t>(row) * dimension + column]) >
              fabs(matrix[static_cast<std::size_t>(pivot) * dimension + column])) {
            pivot = row;
          }
        }
        const double diagonal = matrix[static_cast<std::size_t>(pivot) * dimension + column];
        if (fabs(diagonal) < 1.0e-14) {
          nonsingular = 0;
          break;
        }
        if (pivot != column) {
          for (std::uint32_t item = 0; item < dimension; ++item) {
            const std::size_t first = static_cast<std::size_t>(column) * dimension + item;
            const std::size_t second = static_cast<std::size_t>(pivot) * dimension + item;
            const double swap = matrix[first];
            matrix[first] = matrix[second];
            matrix[second] = swap;
          }
          const double swap = rhs[column];
          rhs[column] = rhs[pivot];
          rhs[pivot] = swap;
        }
        const double scale = matrix[static_cast<std::size_t>(column) * dimension + column];
        for (std::uint32_t item = column; item < dimension; ++item) {
          matrix[static_cast<std::size_t>(column) * dimension + item] /= scale;
        }
        rhs[column] /= scale;
        for (std::uint32_t row = 0; row < dimension; ++row) {
          if (row == column) continue;
          const double factor = matrix[static_cast<std::size_t>(row) * dimension + column];
          for (std::uint32_t item = column; item < dimension; ++item) {
            matrix[static_cast<std::size_t>(row) * dimension + item] -=
                factor * matrix[static_cast<std::size_t>(column) * dimension + item];
          }
          rhs[row] -= factor * rhs[column];
        }
      }
    }
    __syncwarp();
    // The solve is lane-zero-only; broadcast its success flag before any lane
    // decides whether it should form the extrapolated Fock matrix.
    nonsingular = __shfl_sync(0xffffffffU, nonsingular, 0);

    if (nonsingular || !normalize_metric || count <= 2) break;
    --count;
    first = (first + 1) % history_capacity;
    if (threadIdx.x == 0) history_count[system] = count;
  }

  for (std::size_t element = threadIdx.x; element < vector_size; element += blockDim.x) {
    double value = fock[matrix_offset + element];
    if (nonsingular) {
      value = 0.0;
      for (std::uint32_t item = 0; item < count; ++item) {
        const std::size_t item_offset =
            static_cast<std::size_t>(system) * history_stride +
            static_cast<std::size_t>((first + item) % history_capacity) * vector_size;
        value += rhs[item] * fock_history[item_offset + element];
      }
    }
    effective_fock[matrix_offset + element] = value;
  }
}

void launch_update_diis_kernel(
    dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream, std::int32_t batch_size,
    std::int32_t nbf, std::int32_t matrices_per_system, std::uint32_t history_capacity,
    const double* fock, const double* residual, const std::uint8_t* active, double* fock_history,
    double* residual_history, double* linear_system, double* coefficients,
    std::uint32_t* history_count, std::uint32_t* history_head, double* effective_fock,
    bool normalize_metric, bool cooperative_dots, const double* dot_partials, std::size_t parts) {
  // Separate instantiations preserve the existing Direct/KS register and
  // instruction path; only the admitted DF policy uses collective dot work.
  const auto launch = [&]<bool CooperativeDots>() {
    update_diis_kernel<CooperativeDots><<<grid, block, shared_bytes, stream>>>(
        batch_size, nbf, matrices_per_system, history_capacity, fock, residual, active,
        fock_history, residual_history, linear_system, coefficients, history_count, history_head,
        effective_fock, normalize_metric, dot_partials, parts);
  };
  if (cooperative_dots)
    launch.template operator()<true>();
  else
    launch.template operator()<false>();
}

}  // namespace vibeqc::scf::cuda_execution
