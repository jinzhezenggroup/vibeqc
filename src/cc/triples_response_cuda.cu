#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "cc/triples_response_cuda.cuh"
#include "cc/triples_response_internal.hpp"
#include "tensor/cuda_runtime.cuh"

namespace generativeqc::cc {
namespace {
using generativeqc_tensor::cuda_check;

struct DeviceScope {
  int previous{-1};
  explicit DeviceScope(int device) {
    if (device < 0) throw std::invalid_argument("invalid triples response CUDA device");
    cuda_check(cudaGetDevice(&previous));
    cuda_check(cudaSetDevice(device));
  }
  ~DeviceScope() {
    if (previous >= 0) (void)cudaSetDevice(previous);
  }
};

/** Destroy before any borrowed host transfer buffer, including during unwinding.
 * Each completed page already fences its error flag; the destructor also drains
 * partial submissions when a later allocation, launch or transfer fails.
 */
struct DeviceStorage {
  DeviceScope scope;
  cudaStream_t stream{};
  unsigned char* base{};
  DeviceStorage(int device, std::size_t bytes) : scope(device) {
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    try {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&base), bytes));
    } catch (...) {
      (void)cudaStreamDestroy(stream);
      throw;
    }
  }
  ~DeviceStorage() {
    (void)cudaStreamSynchronize(stream);
    (void)cudaFree(base);
    (void)cudaStreamDestroy(stream);
  }
  DeviceStorage(const DeviceStorage&) = delete;
  DeviceStorage& operator=(const DeviceStorage&) = delete;
};

__global__ void accumulate_projected(double* target, const double* source, std::size_t count,
                                     std::size_t symmetric_side, int* error) {
  for (std::size_t flat = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; flat < count;
       flat += std::size_t(blockDim.x) * gridDim.x) {
    double value = source[flat];
    if (symmetric_side) {
      const auto matrix = symmetric_side * symmetric_side;
      const auto offset = flat % matrix;
      const auto mate =
          flat - offset + (offset % symmetric_side) * symmetric_side + offset / symmetric_side;
      value = __dmul_rn(0.5, __dadd_rn(value, source[mate]));
    }
    target[flat] = generativeqc_tensor::finite(__dadd_rn(target[flat], value), error, 10000);
  }
}
}  // namespace

TriplesResponseResult triples_response_cuda(const Problem& p, const SolverResult& cc,
                                            const std::vector<double>& eps_o,
                                            const std::vector<double>& eps_v, int device,
                                            const TriplesResponseOptions& options) {
  const auto denominator = detail::validate_triples_response(p, cc, eps_o, eps_v, options);
  auto layout = detail::triples_response_layout(p.nocc, p.nvir, options.batch_capacity, true);
  while (layout.numeric_bytes() > options.max_bytes && layout.q > 1)
    layout = detail::triples_response_layout(p.nocc, p.nvir, layout.q - 1, true);
  if (layout.numeric_bytes() > options.max_bytes)
    throw std::length_error("RCCSD(T) CUDA response exceeds the complete numeric memory budget");

  TriplesResponseResult result;
  result.minimum_absolute_denominator = denominator;
  result.arena_bytes = layout.arena * sizeof(double);
  result.numeric_capacity_bytes = layout.numeric_bytes();
  result.device_capacity_bytes = layout.device_bytes;
  const std::array<std::vector<double>*, 8> outputs{&result.ovvv,  &result.ovoo, &result.ovov,
                                                    &result.fov,   &result.t1,   &result.t2,
                                                    &result.eps_o, &result.eps_v};
  const std::array<const std::vector<double>*, 8> inputs{&p.ovvv, &p.ovoo, &p.ovov, &p.fov,
                                                         &cc.t1,  &cc.t2,  &eps_o,  &eps_v};
  for (std::size_t i = 0; i < outputs.size(); ++i) outputs[i]->resize(layout.sizes[i]);
  const auto q = layout.q;
  std::vector<std::int64_t> a_map(q), b_map(q), c_map(q);
  std::vector<double> active(q), weights(q, 1.0);
  const double seed = 1.0;
  int error = 0;
  {
    DeviceStorage storage(device, layout.device_bytes);
    generated::TriplesResponseCudaState state{};
    state.o = p.nocc;
    state.v = p.nvir;
    state.q = q;
    state.stream = storage.stream;
    std::size_t cursor = 0;
    auto reserve = [&](std::size_t bytes) {
      const auto next = generated::checked_add(cursor, bytes);
      if (next > layout.device_bytes) throw std::logic_error("triples CUDA layout overflow");
      auto* address = storage.base + cursor;
      cursor = next;
      return address;
    };
    auto upload = [&](const void* dst, const void* src, std::size_t bytes) {
      cuda_check(cudaMemcpyAsync(const_cast<void*>(dst), src, bytes, cudaMemcpyHostToDevice,
                                 storage.stream));
      result.host_to_device_bytes = generated::checked_add(result.host_to_device_bytes, bytes);
    };
    auto download = [&](void* dst, const void* src, std::size_t bytes) {
      cuda_check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, storage.stream));
      result.device_to_host_bytes = generated::checked_add(result.device_to_host_bytes, bytes);
    };
    const std::array<const double**, 8> fields{
        &state.inputs.ovvv, &state.inputs.ovoo, &state.inputs.ovov,  &state.inputs.fov,
        &state.inputs.t1,   &state.inputs.t2,   &state.inputs.eps_o, &state.inputs.eps_v};
    for (std::size_t i = 0; i < fields.size(); ++i) {
      const auto size = layout.sizes[i] * sizeof(double);
      *fields[i] = reinterpret_cast<double*>(reserve(size));
      upload(*fields[i], inputs[i]->data(), size);
    }
    std::array<double*, 8> accumulated{};
    for (std::size_t i = 0; i < accumulated.size(); ++i) {
      const auto size = layout.sizes[i] * sizeof(double);
      accumulated[i] = reinterpret_cast<double*>(reserve(size));
      cuda_check(cudaMemsetAsync(accumulated[i], 0, size, storage.stream));
    }
    state.arena = reinterpret_cast<double*>(reserve(result.arena_bytes));
    state.inputs.a_map = reinterpret_cast<std::int64_t*>(reserve(q * sizeof(std::int64_t)));
    state.inputs.b_map = reinterpret_cast<std::int64_t*>(reserve(q * sizeof(std::int64_t)));
    state.inputs.c_map = reinterpret_cast<std::int64_t*>(reserve(q * sizeof(std::int64_t)));
    state.inputs.active = reinterpret_cast<double*>(reserve(q * sizeof(double)));
    state.inputs.degeneracy = reinterpret_cast<double*>(reserve(q * sizeof(double)));
    state.inputs.bar_triples_energy = reinterpret_cast<double*>(reserve(sizeof(double)));
    state.error = reinterpret_cast<int*>(reserve(sizeof(int)));
    upload(state.inputs.bar_triples_energy, &seed, sizeof(seed));

    auto run_page = [&](std::size_t count) {
      if (!count) return;
      for (std::size_t lane = 0; lane < q; ++lane) {
        active[lane] = lane < count ? 1.0 : 0.0;
        if (lane >= count) weights[lane] = 1.0;
      }
      upload(state.inputs.a_map, a_map.data(), q * sizeof(std::int64_t));
      upload(state.inputs.b_map, b_map.data(), q * sizeof(std::int64_t));
      upload(state.inputs.c_map, c_map.data(), q * sizeof(std::int64_t));
      upload(state.inputs.active, active.data(), q * sizeof(double));
      upload(state.inputs.degeneracy, weights.data(), q * sizeof(double));
      const auto response = generated::run_triples_response_cuda(state);
      const std::array<const double*, 8> values{response.ovvv,  response.ovoo, response.ovov,
                                                response.fov,   response.t1,   response.t2,
                                                response.eps_o, response.eps_v};
      const std::array<std::size_t, 8> symmetry{p.nvir, p.nocc, p.nocc * p.nvir, 0, 0, 0, 0, 0};
      for (std::size_t i = 0; i < values.size(); ++i)
        accumulate_projected<<<generativeqc_tensor::blocks(layout.sizes[i], 256), 256, 0,
                               storage.stream>>>(accumulated[i], values[i], layout.sizes[i],
                                                 symmetry[i], state.error);
      cuda_check(cudaGetLastError());
      download(&error, state.error, sizeof(error));
      cuda_check(cudaStreamSynchronize(storage.stream));
      if (error)
        throw std::runtime_error("RCCSD(T) CUDA response failed at node " + std::to_string(error));
      ++result.pages;
      result.kernel_launches = generated::checked_add(
          result.kernel_launches, generated::triples_response_cuda_kernels_per_page() + 8);
    };
    std::size_t lane = 0;
    for (std::size_t a = 0; a < p.nvir; ++a)
      for (std::size_t b = 0; b <= a; ++b)
        for (std::size_t c = 0; c <= b; ++c) {
          a_map[lane] = static_cast<std::int64_t>(a);
          b_map[lane] = static_cast<std::int64_t>(b);
          c_map[lane] = static_cast<std::int64_t>(c);
          weights[lane] = a == c ? 6.0 : (a == b || b == c ? 2.0 : 1.0);
          if (++lane == q) {
            run_page(lane);
            lane = 0;
          }
        }
    run_page(lane);
    for (std::size_t i = 0; i < outputs.size(); ++i)
      download(outputs[i]->data(), accumulated[i], layout.sizes[i] * sizeof(double));
    cuda_check(cudaStreamSynchronize(storage.stream));
  }
  // Release device scratch before diagnostic strings can extend the phase peak.
  result.program_hash = generated::triples_response_program_hash;
  result.reason =
      "native CUDA runtime-indexed standard-(T) response with resident projected cotangents";
  return result;
}
}  // namespace generativeqc::cc
