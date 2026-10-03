#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "cc/df_triples.hpp"
#include "generated_df_occupied_triples_cuda.cuh"
#include "posthf/capacity.hpp"
#include "tensor/cuda_runtime.cuh"

namespace generativeqc::cc::triples {
namespace {
using posthf::checked_add;
using posthf::checked_mul;
using Clock = std::chrono::steady_clock;
constexpr std::size_t provider_allowance = 96ULL << 20;
constexpr std::size_t blas_workspace = 4ULL << 20;
std::size_t bytes(std::size_t n) { return checked_mul(n, sizeof(double)); }
std::size_t align256(std::size_t n) { return n % 256 ? checked_add(n, 256 - n % 256) : n; }
std::size_t reserve(std::size_t& cursor, std::size_t n) {
  cursor = align256(cursor);
  const auto offset = cursor;
  cursor = checked_add(cursor, n);
  return offset;
}

struct Layout {
  std::array<std::size_t, 9> sizes{}, inputs{};
  std::size_t panels{}, moments{}, partials{}, energies{}, energy{}, error{}, library{};
  std::size_t arena{}, total{}, panel_capacity{}, v3{}, tiles{};
  unsigned blocks{};
};

Layout layout(std::size_t o, std::size_t v, std::size_t q, std::size_t panels) {
  const auto oo = checked_mul(o, o), vv = checked_mul(v, v), ov = checked_mul(o, v);
  const auto ovv = checked_mul(o, vv);
  // Check every dimension and physical leading dimension before reading inputs
  // or allocating a provider. All subsequent pointer products fit these sizes.
  if (std::max({o, v, q, oo, vv, ov, ovv}) >
      static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("DF triples exceed BLAS indexing");
  Layout p;
  p.sizes = {checked_mul(q, ov),
             checked_mul(q, vv),
             checked_mul(ov, oo),
             checked_mul(ov, ov),
             ov,
             ov,
             checked_mul(oo, vv),
             o,
             v};
  p.v3 = checked_mul(v, vv);
  p.tiles = checked_mul(checked_mul(o, checked_add(o, 1)), checked_add(o, 2)) / 6;
  p.blocks = static_cast<unsigned>(std::min<std::size_t>(1 + (p.v3 - 1) / 256, 65535));
  p.panel_capacity = std::min(o, panels);
  // Reject unrepresentable complete work ledgers before any allocation or launch.
  (void)checked_mul(p.tiles, p.v3);
  const auto moment_terms =
      checked_mul(checked_mul(6, p.tiles), checked_add(checked_mul(p.v3, v), checked_mul(p.v3, o)));
  (void)checked_add(moment_terms, checked_mul(checked_mul(3, p.tiles), checked_mul(q, p.v3)));
  std::size_t cursor = 0;
  for (std::size_t x = 0; x < p.sizes.size(); ++x) p.inputs[x] = reserve(cursor, bytes(p.sizes[x]));
  p.panels = reserve(cursor, bytes(checked_mul(p.panel_capacity, p.v3)));
  p.moments = reserve(cursor, bytes(checked_mul(6, p.v3)));
  p.partials = reserve(cursor, bytes(p.blocks));
  p.energies = reserve(cursor, bytes(p.tiles));
  p.energy = reserve(cursor, sizeof(double));
  p.error = reserve(cursor, sizeof(int));
  p.library = reserve(cursor, blas_workspace);
  p.arena = align256(cursor);
  p.total = checked_add(p.arena, provider_allowance);
  return p;
}

// Deterministic generic reduction, shared by each tile and final publication.
__global__ void reduce(const double* values, std::size_t count, double* output, int* error) {
  double sum = 0.0;
  for (std::size_t x = threadIdx.x; x < count; x += blockDim.x) sum += values[x];
  __shared__ double sums[256];
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) *output = generativeqc_tensor::finite(sums[0], error, 3);
}
}  // namespace

DFCudaResult evaluate_df_cuda(std::size_t o, std::size_t v, std::size_t q, const double* bov,
                              const double* bvv, const double* ovoo, const double* ovov,
                              const double* fov, const double* t1, const double* t2,
                              const double* eps_o, const double* eps_v, double threshold,
                              std::size_t max_bytes, int device, std::size_t max_panel_buffers) {
  const auto started = Clock::now();
  if (!o || !v || !q || !max_bytes || device < 0 || !std::isfinite(threshold) || threshold <= 0 ||
      !max_panel_buffers || max_panel_buffers > 3)
    throw std::invalid_argument("invalid DF triples dimensions, threshold or panel limit");
  auto p = layout(o, v, q, max_panel_buffers);
  if (p.total > max_bytes) p = layout(o, v, q, 1);
  if (p.total > max_bytes) throw std::length_error("DF triples exceed numeric memory budget");
  const std::array<const double*, 9> host{bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v};
  for (std::size_t x = 0; x < host.size(); ++x) {
    if (!host[x]) throw std::invalid_argument("null DF triples input");
    if (!std::all_of(host[x], host[x] + p.sizes[x],
                     [](double value) { return std::isfinite(value); }))
      throw std::invalid_argument("nonfinite DF triples input");
  }
  for (std::size_t Q = 0; Q < q; ++Q)
    for (std::size_t a = 0; a < v; ++a)
      for (std::size_t b = 0; b < a; ++b)
        if (std::abs(bvv[(Q * v + a) * v + b] - bvv[(Q * v + b) * v + a]) > 1e-10)
          throw std::invalid_argument("DF triples require symmetric B_vv pairs");
  const auto max_occ = *std::max_element(eps_o, eps_o + o);
  const auto min_vir = *std::min_element(eps_v, eps_v + v);
  const double minimum = 3.0 * (min_vir - max_occ);
  if (!(max_occ < min_vir) || !std::isfinite(minimum) || minimum <= threshold)
    throw std::invalid_argument("noncanonical or near-zero DF triples denominator");

  // Borrowed D2H destinations outlive Context's exception-path stream drain.
  DFCudaResult result;
  int failed = 0;
  {
    runtime::CudaDeviceScope device_scope(device);
    generativeqc_tensor::Context context;
    cudaDeviceProp properties{};
    generativeqc_tensor::cuda_check(cudaGetDeviceProperties(&properties, device));
    context.prepare(device, properties.major, properties.minor, p.arena, p.error, p.library,
                    blas_workspace, provider_allowance, true);
    generated_df::Inputs in;
    const std::array<const double**, 9> fields{&in.bov, &in.bvv, &in.ovoo,  &in.ovov, &in.fov,
                                               &in.t1,  &in.t2,  &in.eps_o, &in.eps_v};
    for (std::size_t x = 0; x < host.size(); ++x) {
      *fields[x] = reinterpret_cast<const double*>(context.arena + p.inputs[x]);
      generativeqc_tensor::cuda_check(cudaMemcpyAsync(context.arena + p.inputs[x], host[x],
                                                      bytes(p.sizes[x]), cudaMemcpyHostToDevice,
                                                      context.stream));
      result.h2d_bytes = checked_add(result.h2d_bytes, bytes(p.sizes[x]));
    }
    generativeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    auto* panels = reinterpret_cast<double*>(context.arena + p.panels);
    auto* moments = reinterpret_cast<double*>(context.arena + p.moments);
    auto* partials = reinterpret_cast<double*>(context.arena + p.partials);
    auto* energies = reinterpret_cast<double*>(context.arena + p.energies);
    auto* energy = reinterpret_cast<double*>(context.arena + p.energy);
    auto gemm = [&](char ta, char tb, std::size_t m, std::size_t n, std::size_t k, double alpha,
                    const double* a, std::size_t lda, const double* b, std::size_t ldb, double beta,
                    double* c, std::size_t ldc) {
      generativeqc_tensor::blas_check(
          cublasDgemm(context.handle, ta == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T,
                      tb == 'N' ? CUBLAS_OP_N : CUBLAS_OP_T, static_cast<int>(m),
                      static_cast<int>(n), static_cast<int>(k), &alpha, a, static_cast<int>(lda), b,
                      static_cast<int>(ldb), &beta, c, static_cast<int>(ldc)));
      result.contraction_summands =
          checked_add(result.contraction_summands, checked_mul(checked_mul(m, n), k));
    };
    std::array<std::size_t, 3> identities;
    identities.fill(std::numeric_limits<std::size_t>::max());
    std::array<std::size_t, 3> ages{};
    std::size_t epoch = 0, tile = 0;
    auto panel_for = [&](std::size_t occupied) {
      std::size_t slot = 0;
      for (; slot < p.panel_capacity; ++slot)
        if (identities[slot] == occupied) break;
      if (slot == p.panel_capacity) {
        slot = static_cast<std::size_t>(
            std::min_element(ages.begin(), ages.begin() + p.panel_capacity) - ages.begin());
        generated_df::build_panel(o, v, q, occupied, in, panels + slot * p.v3, gemm);
        ++result.panel_gemms;
        identities[slot] = occupied;
      }
      ages[slot] = ++epoch;
      return panels + slot * p.v3;
    };
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j <= i; ++j)
        for (std::size_t k = 0; k <= j; ++k) {
          const std::array<std::size_t, 3> occupied{i, j, k};
          // Group W seeds by their integral-panel index. A single-panel fallback
          // consumes every dependent GEMM before that storage is reused. All
          // producer/consumer work is ordered on Context's one owned stream.
          for (std::size_t index = 0; index < occupied.size(); ++index) {
            if (std::find(occupied.begin(), occupied.begin() + index, occupied[index]) !=
                occupied.begin() + index)
              continue;
            const auto* panel = panel_for(occupied[index]);
            for (std::size_t permutation = 0; permutation < 6; ++permutation) {
              const auto* order = generated_df::permutations[permutation];
              if (occupied[order[0]] != occupied[index]) continue;
              generated_df::build_w(o, v, occupied[order[0]], occupied[order[1]],
                                    occupied[order[2]], in, panel, moments + permutation * p.v3,
                                    gemm);
              result.moment_gemms += 2;
            }
          }
          const double degeneracy = i == k ? 6.0 : (i == j || j == k ? 2.0 : 1.0);
          generated_df::energy_tile(o, v, i, j, k, degeneracy, threshold, in, moments, p.blocks,
                                    partials, context.error, context.stream);
          ++result.epilogue_kernels;
          reduce<<<1, 256, 0, context.stream>>>(partials, p.blocks, energies + tile, context.error);
          generativeqc_tensor::cuda_check(cudaGetLastError());
          ++result.reduction_kernels;
          ++tile;
        }
    if (tile != p.tiles) throw std::logic_error("DF triples occupied work mismatch");
    reduce<<<1, 256, 0, context.stream>>>(energies, p.tiles, energy, context.error);
    generativeqc_tensor::cuda_check(cudaGetLastError());
    ++result.reduction_kernels;
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&result.energy, energy, sizeof(double),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaMemcpyAsync(&failed, context.error, sizeof(int),
                                                    cudaMemcpyDeviceToHost, context.stream));
    generativeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    if (failed || !std::isfinite(result.energy))
      throw std::runtime_error("nonfinite or unsafe generated DF triples arithmetic");
    result.minimum_absolute_denominator = minimum;
    result.virtual_triples = checked_mul(checked_mul(v, checked_add(v, 1)), checked_add(v, 2)) / 6;
    result.occupied_tiles = p.tiles;
    result.epilogue_points = checked_mul(p.tiles, p.v3);
    result.workspace_bytes = p.total;
    result.arena_bytes = p.arena;
    result.provider_retained_bytes = context.metrics.provider_retained_bytes;
    result.panel_capacity = p.panel_capacity;
    result.d2h_bytes = sizeof(double) + sizeof(int);
  }  // Drain/destroy the stream, buffers and BLAS provider inside endpoint timing.
  result.seconds = std::chrono::duration<double>(Clock::now() - started).count();
  return result;
}
}  // namespace generativeqc::cc::triples
