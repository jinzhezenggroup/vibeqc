#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>

#include "cc/triples_fock_response.hpp"
#include "cc/triples_fock_response_cuda.cuh"
#include "cc/triples_response_internal.hpp"
#include "tensor/cuda_runtime.cuh"

namespace generativeqc::cc {
namespace {
using generated::checked_add;
using generativeqc_tensor::cuda_check;

struct DeviceStorage {
  int previous{-1};
  cudaStream_t stream{};
  unsigned char* base{};
  DeviceStorage(int device, std::size_t bytes) {
    if (device < 0) throw std::invalid_argument("invalid triples Fock CUDA device");
    cuda_check(cudaGetDevice(&previous));
    cuda_check(cudaSetDevice(device));
    try {
      cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&base), bytes));
    } catch (...) {
      if (stream) (void)cudaStreamDestroy(stream);
      (void)cudaSetDevice(previous);
      throw;
    }
  }
  ~DeviceStorage() {
    // Pending copies borrow host controls, so this owner must die before them.
    (void)cudaStreamSynchronize(stream);
    (void)cudaFree(base);
    (void)cudaStreamDestroy(stream);
    (void)cudaSetDevice(previous);
  }
  DeviceStorage(const DeviceStorage&) = delete;
  DeviceStorage& operator=(const DeviceStorage&) = delete;
};

__global__ void add_occupied(double* target, const double* values, std::size_t count, double weight,
                             int* error) {
  for (std::size_t k = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; k < count;
       k += std::size_t(blockDim.x) * gridDim.x)
    target[k] = generativeqc_tensor::finite(__dadd_rn(target[k], __dmul_rn(weight, values[k])),
                                            error, 20000);
}

__global__ void add_virtual(double* target, const double* values, std::size_t v, std::size_t q,
                            std::size_t left, std::size_t right, std::size_t nl, std::size_t nr,
                            double weight, int* error) {
  // Each page pair owns disjoint output blocks. Diagonal pairs are written once;
  // off-diagonal pairs also scatter their transpose without atomics or races.
  for (std::size_t k = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; k < nl * nr;
       k += std::size_t(blockDim.x) * gridDim.x) {
    const auto i = k / nr, j = k % nr;
    const double value = __dmul_rn(weight, values[i * q + j]);
    const auto index = (left + i) * v + right + j;
    target[index] = generativeqc_tensor::finite(__dadd_rn(target[index], value), error, 20001);
    if (left != right) {
      const auto mate = (right + j) * v + left + i;
      target[mate] = generativeqc_tensor::finite(__dadd_rn(target[mate], value), error, 20002);
    }
  }
}
}  // namespace

TriplesFockResponseResult triples_fock_response_cuda(const Problem& p, const SolverResult& cc,
                                                     const std::vector<double>& eo,
                                                     const std::vector<double>& ev, int device,
                                                     const TriplesResponseOptions& options) {
  (void)detail::validate_triples_response(p, cc, eo, ev, options);
  auto l = detail::triples_fock_response_layout(p.nocc, p.nvir, options.batch_capacity, true);
  while (l.numeric_bytes() > options.max_bytes && l.q > 1)
    l = detail::triples_fock_response_layout(p.nocc, p.nvir, l.q - 1, true);
  if (l.numeric_bytes() > options.max_bytes)
    throw std::length_error("triples Fock response exceeds complete CUDA memory budget");
  const auto o = p.nocc, v = p.nvir, q = l.q;
  TriplesFockResponseResult result;
  result.page_capacity = q;
  result.numeric_capacity_bytes = l.numeric_bytes();
  result.device_capacity_bytes = l.device_bytes;
  result.foo.resize(o * o);
  result.fvv.resize(v * v);
  std::vector<std::int64_t> amap(q), bmap(q), cmap(q);
  std::vector<double> active(q);
  int error = 0;
  {
    DeviceStorage storage(device, l.device_bytes);
    generated::TriplesFockCudaState state{};
    state.o = o;
    state.v = v;
    state.q = q;
    state.stream = storage.stream;
    std::size_t cursor = 0;
    auto reserve = [&](std::size_t bytes) {
      const auto next = checked_add(cursor, bytes);
      if (next > l.device_bytes) throw std::logic_error("triples Fock device layout overflow");
      auto* address = storage.base + cursor;
      cursor = next;
      return address;
    };
    auto upload = [&](const void* dst, const void* src, std::size_t bytes) {
      cuda_check(cudaMemcpyAsync(const_cast<void*>(dst), src, bytes, cudaMemcpyHostToDevice,
                                 storage.stream));
      result.host_to_device_bytes = checked_add(result.host_to_device_bytes, bytes);
    };
    auto download = [&](void* dst, const void* src, std::size_t bytes) {
      cuda_check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, storage.stream));
      result.device_to_host_bytes = checked_add(result.device_to_host_bytes, bytes);
    };
    auto copy = [&](double* dst, const double* src) {
      const auto bytes = l.vector_elements * sizeof(double);
      cuda_check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, storage.stream));
      result.device_copy_bytes = checked_add(result.device_copy_bytes, bytes);
    };
    const std::array<const std::vector<double>*, 8> inputs{&p.ovvv, &p.ovoo, &p.ovov, &p.fov,
                                                           &cc.t1,  &cc.t2,  &eo,     &ev};
    const std::array<const double**, 8> fields{
        &state.inputs.ovvv, &state.inputs.ovoo, &state.inputs.ovov,  &state.inputs.fov,
        &state.inputs.t1,   &state.inputs.t2,   &state.inputs.eps_o, &state.inputs.eps_v};
    for (std::size_t k = 0; k < inputs.size(); ++k) {
      const auto size = l.input_sizes[k] * sizeof(double);
      *fields[k] = reinterpret_cast<double*>(reserve(size));
      upload(*fields[k], inputs[k]->data(), size);
    }
    auto* foo = reinterpret_cast<double*>(reserve(o * o * sizeof(double)));
    auto* fvv = reinterpret_cast<double*>(reserve(v * v * sizeof(double)));
    cuda_check(cudaMemsetAsync(foo, 0, o * o * sizeof(double), storage.stream));
    cuda_check(cudaMemsetAsync(fvv, 0, v * v * sizeof(double), storage.stream));
    auto* xl = reinterpret_cast<double*>(reserve(l.vector_elements * sizeof(double)));
    auto* yl = reinterpret_cast<double*>(reserve(l.vector_elements * sizeof(double)));
    auto* xr = reinterpret_cast<double*>(reserve(l.vector_elements * sizeof(double)));
    auto* yr = reinterpret_cast<double*>(reserve(l.vector_elements * sizeof(double)));
    state.arena = reinterpret_cast<double*>(reserve(l.arena * sizeof(double)));
    state.inputs.a_map = reinterpret_cast<std::int64_t*>(reserve(q * sizeof(std::int64_t)));
    state.inputs.b_map = reinterpret_cast<std::int64_t*>(reserve(q * sizeof(std::int64_t)));
    state.inputs.c_map = reinterpret_cast<std::int64_t*>(reserve(q * sizeof(std::int64_t)));
    state.inputs.active = reinterpret_cast<double*>(reserve(q * sizeof(double)));
    state.error = reinterpret_cast<int*>(reserve(sizeof(int)));
    auto check = [&] {
      cuda_check(cudaGetLastError());
      download(&error, state.error, sizeof(error));
      cuda_check(cudaStreamSynchronize(storage.stream));
      if (error)
        throw std::runtime_error("triples Fock CUDA failed at node " + std::to_string(error));
    };
    auto vectors = [&](std::size_t start, double* x, double* y) {
      const auto count = std::min(q, v - start);
      for (std::size_t lane = 0; lane < q; ++lane) {
        amap[lane] = lane < count ? start + lane : 0;
        active[lane] = lane < count ? 1.0 : 0.0;
      }
      upload(state.inputs.a_map, amap.data(), q * sizeof(std::int64_t));
      upload(state.inputs.b_map, bmap.data(), q * sizeof(std::int64_t));
      upload(state.inputs.c_map, cmap.data(), q * sizeof(std::int64_t));
      upload(state.inputs.active, active.data(), q * sizeof(double));
      const auto out = generated::run_triples_resolvent_cuda(state);
      copy(x, out.x);
      copy(y, out.y);
      check();
      ++result.vector_pages;
      result.kernel_launches =
          checked_add(result.kernel_launches, generated::triples_resolvent_cuda_kernel_count());
      return count;
    };
    for (std::size_t b = 0; b < v; ++b)
      for (std::size_t c = 0; c <= b; ++c) {
        std::fill(bmap.begin(), bmap.end(), b);
        std::fill(cmap.begin(), cmap.end(), c);
        const double weight = b == c ? 1.0 : 2.0;
        ++result.pair_panels;
        for (std::size_t left = 0; left < v; left += q) {
          const auto nl = vectors(left, xl, yl);
          state.moments.x_left = xl;
          state.moments.y_left = yl;
          const auto occupied = generated::run_triples_oo_moment_cuda(state);
          add_occupied<<<generativeqc_tensor::blocks(o * o, 256), 256, 0, storage.stream>>>(
              foo, occupied.values, o * o, weight, state.error);
          check();
          ++result.occupied_moments;
          result.kernel_launches = checked_add(
              result.kernel_launches, generated::triples_oo_moment_cuda_kernel_count() + 1);
          for (std::size_t right = left; right < v; right += q) {
            const bool same = right == left;
            const auto nr = same ? nl : vectors(right, xr, yr);
            state.moments.x_right = same ? xl : xr;
            state.moments.y_right = same ? yl : yr;
            const auto virt = generated::run_triples_vv_moment_cuda(state);
            add_virtual<<<generativeqc_tensor::blocks(nl * nr, 256), 256, 0, storage.stream>>>(
                fvv, virt.values, v, q, left, right, nl, nr, weight, state.error);
            check();
            ++result.virtual_moments;
            result.kernel_launches = checked_add(
                result.kernel_launches, generated::triples_vv_moment_cuda_kernel_count() + 1);
          }
        }
      }
    download(result.foo.data(), foo, o * o * sizeof(double));
    download(result.fvv.data(), fvv, v * v * sizeof(double));
    cuda_check(cudaStreamSynchronize(storage.stream));
  }
  result.program_hash = generated::triples_resolvent_program_hash;
  return result;
}

}  // namespace generativeqc::cc
