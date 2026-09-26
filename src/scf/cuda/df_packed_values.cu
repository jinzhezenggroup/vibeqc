#include <algorithm>
#include <cmath>

#include "scf/cuda/df_packed_values.hpp"
#include "scf/df_value_storage.hpp"

namespace vibeqc::scf::cuda_df {
namespace {
__device__ DfSymmetricAoPair lower_pair(std::size_t mu, std::size_t nu) {
  const auto hi = mu > nu ? mu : nu, lo = mu > nu ? nu : mu;
  return {hi * (hi + 1) / 2 + lo};
}

__global__ void pack_density(std::size_t n, const double* density, double* packed) {
  const auto index = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (index >= n * n) return;
  const auto mu = index / n, nu = index % n;
  if (nu <= mu)
    packed[lower_pair(mu, nu).index] = density[index] + (mu == nu ? 0 : density[nu * n + mu]);
}

__global__ void scatter_pairs(std::size_t n, const double* packed, double* matrix) {
  const auto index = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (index < n * n) matrix[index] = packed[lower_pair(index / n, index % n).index];
}

__global__ void unpack_values(std::size_t n, std::size_t a, std::size_t row_begin, std::size_t rows,
                              std::size_t qbegin, std::size_t qcount, bool auxiliary_major,
                              const double* packed, double* output) {
  const auto elements = rows * n * qcount;
  for (auto k = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x; k < elements;
       k += std::size_t{gridDim.x} * blockDim.x) {
    const auto q = auxiliary_major ? k / (rows * n) : k % qcount;
    const auto pair = auxiliary_major ? k % (rows * n) : k / qcount;
    output[k] = packed[lower_pair(row_begin + pair / n, pair % n).index * a + qbegin + q];
  }
}

/** Logical expansion on load, not a second scientific equation. Each block
 * fixes mu and reuses 32-nu tiles across a 64-Q by 16-occupied rectangle.
 * Padded shared C columns avoid bank conflicts for canonical column-major C;
 * host compatibility C uses a coalesced row-major load into the same tile.
 */
template <bool ColumnMajor>
__global__ void project_packed(std::size_t n, std::size_t a, std::size_t rank, const double* packed,
                               const double* c, double* u) {
  constexpr int Q = 64, R = 16, K = 32;
  __shared__ double bs[K][Q], cs[R][K + 1];
  const int x = threadIdx.x, y = threadIdx.y, thread = y * Q + x;
  const auto mu = std::size_t{blockIdx.z}, q = std::size_t{blockIdx.x} * Q + x;
  const auto rb = std::size_t{blockIdx.y} * R;
  double value[4] = {};
  for (std::size_t begin = 0; begin < n; begin += K) {
    for (int element = thread; element < K * Q; element += Q * 4) {
      const auto nu = begin + element / Q, qq = std::size_t{blockIdx.x} * Q + element % Q;
      bs[element / Q][element % Q] =
          nu < n && qq < a ? packed[lower_pair(mu, nu).index * a + qq] : 0;
    }
    for (int element = thread; element < K * R; element += Q * 4) {
      const int k = ColumnMajor ? element % K : element / R;
      const int r = ColumnMajor ? element / K : element % R;
      const auto nu = begin + k, rr = rb + r;
      cs[r][k] = nu < n && rr < rank ? c[ColumnMajor ? nu + rr * n : nu * rank + rr] : 0;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const auto b = bs[k][x];
#pragma unroll
      for (int j = 0; j < 4; ++j) value[j] = fma(b, cs[y + j * 4][k], value[j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const auto r = rb + y + j * 4;
    if (q < a && r < rank) u[(mu * rank + r) * a + q] = value[j];
  }
}
}  // namespace

void launch_pack_df_density(cudaStream_t stream, std::size_t n, const double* density,
                            double* packed) {
  pack_density<<<(n * n + 255) / 256, 256, 0, stream>>>(n, density, packed);
}

void launch_scatter_df_pairs(cudaStream_t stream, std::size_t n, const double* packed,
                             double* matrix) {
  scatter_pairs<<<(n * n + 255) / 256, 256, 0, stream>>>(n, packed, matrix);
}

void launch_unpack_df_values(cudaStream_t stream, std::size_t n, std::size_t a,
                             std::size_t row_begin, std::size_t rows, std::size_t qbegin,
                             std::size_t qcount, bool auxiliary_major, const double* packed,
                             double* output) {
  if (!rows || !qcount) return;
  const auto blocks = std::min<std::size_t>(4096, (rows * n * qcount + 255) / 256);
  unpack_values<<<blocks, 256, 0, stream>>>(n, a, row_begin, rows, qbegin, qcount, auxiliary_major,
                                            packed, output);
}

void launch_project_packed_df(cudaStream_t stream, std::size_t n, std::size_t a, std::size_t rank,
                              bool column_major, const double* packed, const double* coefficients,
                              double* projection) {
  if (!rank) return;
  const dim3 grid((a + 63) / 64, (rank + 15) / 16, n), block(64, 4);
  if (column_major)
    project_packed<true><<<grid, block, 0, stream>>>(n, a, rank, packed, coefficients, projection);
  else
    project_packed<false><<<grid, block, 0, stream>>>(n, a, rank, packed, coefficients, projection);
}
}  // namespace vibeqc::scf::cuda_df
