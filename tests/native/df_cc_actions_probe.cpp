// Qualification harness for generated, one-auxiliary-slice DF CC actions.
// This is deliberately not a complete DF solver or a production owner.
#include <array>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "generated_df_ccsd_cpu.hpp"
#if defined(DF_PROBE_CUDA)
#include "generated_df_ccsd_cuda.cuh"
#include "tensor/cuda_error.hpp"
#endif

namespace dfcc = generativeqc::cc::generated::df;

int main() {
  try {
    std::size_t o{}, v{}, action{}, shortfall{};
    if (!(std::cin >> o >> v >> action >> shortfall) || !o || !v || action > 3)
      throw std::invalid_argument("invalid DF probe dimensions/action");
    const auto ov = dfcc::checked_mul(o, v), oovv = dfcc::checked_mul(ov, ov);
    const std::array<std::size_t, 8> sizes{ov, v * v, ov, oovv, ov, oovv, ov, oovv};
    std::array<std::vector<double>, 8> host;
    for (std::size_t index = 0; index < host.size(); ++index) {
      host[index].resize(sizes[index]);
      for (auto& value : host[index])
        if (!(std::cin >> value)) throw std::invalid_argument("truncated DF probe input");
    }
    dfcc::Inputs inputs{host[0].data(), host[1].data(), host[2].data(), host[3].data(),
                        host[4].data(), host[5].data(), host[6].data(), host[7].data()};
    const std::array<std::size_t, 4> cpu_sizes{
        dfcc::virtual_cpu_arena_elements(o, v), dfcc::amplitude_jvp_cpu_arena_elements(o, v),
        dfcc::amplitude_vjp_cpu_arena_elements(o, v), dfcc::factor_vjp_cpu_arena_elements(o, v)};
    std::size_t required = cpu_sizes[action];
    const double sentinel = 913.25;
    const double *first{}, *second{};
    const auto second_size = action == 3 ? v * v : oovv;
    std::vector<double> output(ov + second_size);
#if defined(DF_PROBE_CUDA)
    using generativeqc_tensor::cuda_check;
    if (shortfall == 1) throw std::invalid_argument("CUDA harness only accepts a full arena");
    const std::array<std::size_t, 4> cuda_sizes{
        dfcc::virtual_cuda_arena_elements(o, v), dfcc::amplitude_jvp_cuda_arena_elements(o, v),
        dfcc::amplitude_vjp_cuda_arena_elements(o, v), dfcc::factor_vjp_cuda_arena_elements(o, v)};
    required = cuda_sizes[action];
    // Each invocation is a fresh process. Keep ownership explicit so memcheck
    // can distinguish generated accesses from this small transfer harness.
    struct DeviceArrays {
      std::vector<void*> allocations;
      ~DeviceArrays() {
        for (auto* pointer : allocations) cudaFree(pointer);
      }
      void* allocate(std::size_t bytes) {
        void* pointer{};
        generativeqc_tensor::cuda_check(cudaMalloc(&pointer, bytes));
        allocations.push_back(pointer);
        return pointer;
      }
    } device;
    dfcc::CudaState state;
    std::array<const double**, 8> fields{&state.bov,
                                         &state.bvv,
                                         &state.t1,
                                         &state.t2,
                                         &state.d_t1,
                                         &state.d_t2,
                                         &state.bar_df_virtual_singles,
                                         &state.bar_df_virtual_doubles};
    for (std::size_t index = 0; index < host.size(); ++index) {
      auto* pointer = static_cast<double*>(device.allocate(host[index].size() * sizeof(double)));
      cuda_check(cudaMemcpy(pointer, host[index].data(), host[index].size() * sizeof(double),
                            cudaMemcpyHostToDevice));
      *fields[index] = pointer;
    }
    std::vector<double> canary(required + 2, sentinel);
    auto* allocation = static_cast<double*>(device.allocate(canary.size() * sizeof(double)));
    cuda_check(cudaMemcpy(allocation, canary.data(), canary.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
    state.o = o;
    state.v = v;
    state.response_arena = allocation + 1;
    state.error = static_cast<int*>(device.allocate(sizeof(int)));
    if (shortfall == 2) {
      // Composition must preserve an earlier failure even when this complete
      // finite action succeeds. The standalone entry still clears by contract.
      int seeded = 173;
      cuda_check(cudaMemcpy(state.error, &seeded, sizeof(int), cudaMemcpyHostToDevice));
      dfcc::run_virtual_accumulate_cuda(state);
      int observed{};
      cuda_check(cudaMemcpy(&observed, state.error, sizeof(int), cudaMemcpyDeviceToHost));
      if (observed != seeded) throw std::runtime_error("DF composition erased an earlier error");
      dfcc::run_virtual_cuda(state);
      cuda_check(cudaMemcpy(&observed, state.error, sizeof(int), cudaMemcpyDeviceToHost));
      if (observed) throw std::runtime_error("standalone DF action did not reset its error");
      std::cout << "sticky " << seeded << '\n';
      return 0;
    }
    if (action == 0) {
      auto result = dfcc::run_virtual_cuda(state);
      first = result.singles;
      second = result.doubles;
    }
    if (action == 1) {
      auto result = dfcc::run_amplitude_jvp_cuda(state);
      first = result.singles;
      second = result.doubles;
    }
    if (action == 2) {
      auto result = dfcc::run_amplitude_vjp_cuda(state);
      first = result.t1;
      second = result.t2;
    }
    if (action == 3) {
      auto result = dfcc::run_factor_vjp_cuda(state);
      first = result.bov;
      second = result.bvv;
    }
    cuda_check(cudaMemcpy(output.data(), first, ov * sizeof(double), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(output.data() + ov, second, second_size * sizeof(double),
                          cudaMemcpyDeviceToHost));
    int error{};
    cuda_check(cudaMemcpy(&error, state.error, sizeof(int), cudaMemcpyDeviceToHost));
    if (error) throw std::runtime_error("generated DF CUDA arithmetic failed");
    std::array<double, 2> guards{};
    cuda_check(cudaMemcpy(&guards[0], allocation, sizeof(double), cudaMemcpyDeviceToHost));
    cuda_check(
        cudaMemcpy(&guards[1], allocation + required + 1, sizeof(double), cudaMemcpyDeviceToHost));
    if (guards[0] != sentinel || guards[1] != sentinel)
      throw std::runtime_error("DF CUDA arena guard changed");
#else
    std::vector<double> arena(required + 2, sentinel);
    try {
      const auto available = shortfall ? required - 1 : required;
      if (action == 0) {
        auto result = dfcc::run_virtual_cpu(o, v, inputs, arena.data() + 1, available);
        first = result.singles;
        second = result.doubles;
      }
      if (action == 1) {
        auto result = dfcc::run_amplitude_jvp_cpu(o, v, inputs, arena.data() + 1, available);
        first = result.singles;
        second = result.doubles;
      }
      if (action == 2) {
        auto result = dfcc::run_amplitude_vjp_cpu(o, v, inputs, arena.data() + 1, available);
        first = result.t1;
        second = result.t2;
      }
      if (action == 3) {
        auto result = dfcc::run_factor_vjp_cpu(o, v, inputs, arena.data() + 1, available);
        first = result.bov;
        second = result.bvv;
      }
    } catch (const std::length_error&) {
      if (!shortfall) throw;
      for (const auto value : arena)
        if (value != sentinel) throw std::runtime_error("short DF arena performed numerical work");
      std::cout << "refused " << required << '\n';
      return 0;
    }
    if (shortfall) throw std::runtime_error("short DF arena was accepted");
    if (arena.front() != sentinel || arena.back() != sentinel)
      throw std::runtime_error("DF CPU arena guard changed");
    std::copy(first, first + ov, output.data());
    std::copy(second, second + second_size, output.data() + ov);
#endif
    std::cout << required << '\n' << std::setprecision(17);
    for (const auto value : output) std::cout << value << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
