#include "cc/solver.hpp"

#if VIBEQC_HAS_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cc/cuda_solver_support.cuh"
#include "cc/cuda_state.cuh"
#include "generated_rccsd_cpu.hpp"
#include "tensor/cuda_error.hpp"
#include "tensor/cuda_runtime.cuh"

namespace vibeqc::cc {
namespace {

using vibeqc_tensor::cuda_check;

std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("RCCSD CUDA size overflow");
  return a * b;
}
std::size_t checked_add(std::size_t a, std::size_t b) { return generated::checked_add(a, b); }
std::size_t align256(std::size_t x) {
  const auto rem = x % 256;
  return rem ? checked_add(x, 256 - rem) : x;
}

struct DeviceScope {
  int previous{};
  explicit DeviceScope(int device) {
    cuda_check(cudaGetDevice(&previous));
    cuda_check(cudaSetDevice(device));
  }
  ~DeviceScope() { cudaSetDevice(previous); }
};

__global__ void damped_advance(const double* current, const double* undamped, std::size_t count,
                               double factor, double* output, int* error) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(blockDim.x) * gridDim.x)
    output[i] = vibeqc_tensor::finite(
        __dadd_rn(current[i], __dmul_rn(factor, __dsub_rn(undamped[i], current[i]))), error, 0);
}

__global__ void gram_kernel(const double* errors, std::size_t elements, int history, double* gram) {
  const int pair = static_cast<int>(blockIdx.x);
  const int row = pair / history, col = pair % history;
  if (row > col) return;
  __shared__ double values[256];
  double sum = 0.0;
  for (std::size_t i = threadIdx.x; i < elements; i += blockDim.x)
    sum = __dadd_rn(sum, __dmul_rn(errors[std::size_t(row) * elements + i],
                                   errors[std::size_t(col) * elements + i]));
  values[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride)
      values[threadIdx.x] = __dadd_rn(values[threadIdx.x], values[threadIdx.x + stride]);
    __syncthreads();
  }
  if (!threadIdx.x) {
    gram[std::size_t(row) * history + col] = values[0];
    if (row != col) gram[std::size_t(col) * history + row] = values[0];
  }
}

struct Layout {
  std::array<std::size_t, 14> inputs{};
  std::size_t iteration{}, replay{}, last_t1{}, last_t2{}, vectors{}, errors{};
  std::size_t gram{}, system{}, coefficients{}, r1_partials{}, r2_partials{}, scalars{};
  std::size_t status{}, arithmetic{}, total{};
};

std::size_t reserve(Layout& layout, std::size_t& cursor, std::size_t bytes) {
  cursor = align256(cursor);
  const auto result = cursor;
  cursor = checked_add(cursor, bytes);
  return result;
}

struct Owner {
  DeviceScope scope;
  int device{};
  cudaStream_t stream{};
  unsigned char* base{};
  Layout layout;
  generated::CudaState state;
  double *last_t1{}, *last_t2{}, *vectors{}, *errors{}, *gram{}, *system{}, *coefficients{};
  double *r1_partials{}, *r2_partials{}, *scalars{};
  int *status{}, *arithmetic{};
  std::size_t n1{}, n2{}, elements{}, partial1{}, partial2{};
  unsigned history{}, restarts{};
  SolverDiagnostic diagnostic;

  Owner(const Problem& p, const SolverOptions& options, int ordinal)
      : scope(ordinal),
        device(ordinal),
        n1(checked_mul(p.nocc, p.nvir)),
        n2(checked_mul(checked_mul(p.nocc, p.nocc), checked_mul(p.nvir, p.nvir))),
        elements(checked_add(n1, n2)) {
    const std::array<const std::vector<double>*, 14> host = {
        &p.foo,  &p.fov,  &p.fvv,  &p.ovov, &p.ovvo, &p.oovv,       &p.ovvv,
        &p.ovoo, &p.oooo, &p.vvvv, &p.d1,   &p.d2,   &p.initial_t1, &p.initial_t2};
    std::size_t cursor = 0;
    for (std::size_t i = 0; i < host.size(); ++i)
      layout.inputs[i] = reserve(layout, cursor, checked_mul(host[i]->size(), sizeof(double)));
    layout.iteration =
        reserve(layout, cursor,
                checked_mul(generated::iteration_arena_elements(p.nocc, p.nvir), sizeof(double)));
    layout.replay =
        reserve(layout, cursor,
                checked_mul(generated::replay_arena_elements(p.nocc, p.nvir), sizeof(double)));
    layout.last_t1 = reserve(layout, cursor, checked_mul(n1, sizeof(double)));
    layout.last_t2 = reserve(layout, cursor, checked_mul(n2, sizeof(double)));
    layout.vectors = reserve(layout, cursor,
                             checked_mul(checked_mul(options.diis_size, elements), sizeof(double)));
    layout.errors = reserve(layout, cursor,
                            checked_mul(checked_mul(options.diis_size, elements), sizeof(double)));
    layout.gram =
        reserve(layout, cursor,
                checked_mul(checked_mul(options.diis_size, options.diis_size), sizeof(double)));
    layout.system = reserve(
        layout, cursor,
        checked_mul(checked_mul(options.diis_size + 1, options.diis_size + 1), sizeof(double)));
    layout.coefficients =
        reserve(layout, cursor, checked_mul(options.diis_size + 1, sizeof(double)));
    partial1 = std::min<std::size_t>((n1 + 255) / 256, 65535);
    partial2 = std::min<std::size_t>((n2 + 255) / 256, 65535);
    layout.r1_partials = reserve(layout, cursor, checked_mul(partial1, sizeof(double)));
    layout.r2_partials = reserve(layout, cursor, checked_mul(partial2, sizeof(double)));
    layout.scalars = reserve(layout, cursor, 2 * sizeof(double));
    layout.status = reserve(layout, cursor, sizeof(int));
    layout.arithmetic = reserve(layout, cursor, sizeof(int));
    layout.total = align256(cursor);

    // Final detached host amplitudes coexist with this resident device arena.
    const auto combined = checked_add(
        checked_add(p.reference_retained_bytes, checked_add(problem_host_bytes(p), layout.total)),
        checked_mul(elements, sizeof(double)));
    if (combined > options.max_bytes)
      throw std::length_error("RCCSD CUDA resident state exceeds correlation memory budget");
    try {
      cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&base), layout.total));

      std::array<double**, 14> fields = {
          &state.foo,  &state.fov,  &state.fvv,  &state.ovov, &state.ovvo, &state.oovv, &state.ovvv,
          &state.ovoo, &state.oooo, &state.vvvv, &state.d1,   &state.d2,   &state.t1,   &state.t2};
      for (std::size_t i = 0; i < host.size(); ++i) {
        *fields[i] = reinterpret_cast<double*>(base + layout.inputs[i]);
        const auto amount = host[i]->size() * sizeof(double);
        if (amount)
          cuda_check(
              cudaMemcpyAsync(*fields[i], host[i]->data(), amount, cudaMemcpyHostToDevice, stream));
        diagnostic.setup_h2d_bytes += amount;
      }
      state.o = p.nocc;
      state.v = p.nvir;
      state.stream = stream;
      state.iteration_arena = reinterpret_cast<double*>(base + layout.iteration);
      state.replay_arena = reinterpret_cast<double*>(base + layout.replay);
      state.error = reinterpret_cast<int*>(base + layout.arithmetic);
      last_t1 = reinterpret_cast<double*>(base + layout.last_t1);
      last_t2 = reinterpret_cast<double*>(base + layout.last_t2);
      vectors = reinterpret_cast<double*>(base + layout.vectors);
      errors = reinterpret_cast<double*>(base + layout.errors);
      gram = reinterpret_cast<double*>(base + layout.gram);
      system = reinterpret_cast<double*>(base + layout.system);
      coefficients = reinterpret_cast<double*>(base + layout.coefficients);
      r1_partials = reinterpret_cast<double*>(base + layout.r1_partials);
      r2_partials = reinterpret_cast<double*>(base + layout.r2_partials);
      scalars = reinterpret_cast<double*>(base + layout.scalars);
      status = reinterpret_cast<int*>(base + layout.status);
      arithmetic = reinterpret_cast<int*>(base + layout.arithmetic);
      cuda_check(cudaMemcpyAsync(last_t1, state.t1, n1 * sizeof(double), cudaMemcpyDeviceToDevice,
                                 stream));
      cuda_check(cudaMemcpyAsync(last_t2, state.t2, n2 * sizeof(double), cudaMemcpyDeviceToDevice,
                                 stream));
      cuda_check(cudaStreamSynchronize(stream));
      ++diagnostic.synchronizations;
      diagnostic.owned_device_bytes = layout.total;
      diagnostic.numeric_capacity_bytes = std::max(p.provider_peak_bytes, combined);
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~Owner() { cleanup(); }

  void cleanup() noexcept {
    if (stream) cudaStreamSynchronize(stream);
    if (base) cudaFree(base);
    if (stream) cudaStreamDestroy(stream);
    base = nullptr;
    stream = nullptr;
  }

  template <class Output>
  std::array<double, 3> read_status(const Output& out) {
    vibeqc::cc::residual_partials<<<vibeqc_tensor::blocks(static_cast<vibeqc_tensor::I>(n1), 256),
                                    256, 0, stream>>>(out.r1, static_cast<vibeqc_tensor::I>(n1),
                                                      r1_partials, state.error);
    vibeqc::cc::residual_partials<<<vibeqc_tensor::blocks(static_cast<vibeqc_tensor::I>(n2), 256),
                                    256, 0, stream>>>(out.r2, static_cast<vibeqc_tensor::I>(n2),
                                                      r2_partials, state.error);
    vibeqc::cc::residual_finish<<<1, 1, 0, stream>>>(r1_partials, static_cast<int>(partial1),
                                                     scalars);
    vibeqc::cc::residual_finish<<<1, 1, 0, stream>>>(r2_partials, static_cast<int>(partial2),
                                                     scalars + 1);
    cuda_check(cudaGetLastError());
    std::array<double, 3> host{};
    int error = 0;
    cuda_check(
        cudaMemcpyAsync(host.data(), out.energy, sizeof(double), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(host.data() + 1, scalars, 2 * sizeof(double), cudaMemcpyDeviceToHost,
                               stream));
    cuda_check(cudaMemcpyAsync(&error, state.error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    diagnostic.scalar_d2h_bytes += 3 * sizeof(double) + sizeof(int);
    ++diagnostic.synchronizations;
    if (error)
      throw std::runtime_error("nonfinite RCCSD generated CUDA tensor at node " +
                               std::to_string(std::abs(error)));
    return host;
  }

  void advance(const generated::DeviceIterationOutputs& out, double factor) {
    cuda_check(
        cudaMemcpyAsync(last_t1, state.t1, n1 * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    cuda_check(
        cudaMemcpyAsync(last_t2, state.t2, n2 * sizeof(double), cudaMemcpyDeviceToDevice, stream));
    cuda_check(cudaMemsetAsync(state.error, 0, sizeof(int), stream));
    damped_advance<<<vibeqc_tensor::blocks(static_cast<vibeqc_tensor::I>(n1), 256), 256, 0,
                     stream>>>(last_t1, out.next_t1, n1, factor, state.t1, state.error);
    damped_advance<<<vibeqc_tensor::blocks(static_cast<vibeqc_tensor::I>(n2), 256), 256, 0,
                     stream>>>(last_t2, out.next_t2, n2, factor, state.t2, state.error);
    cuda_check(cudaGetLastError());
  }

  void check_generated_error() {
    int host_error = 0;
    cuda_check(
        cudaMemcpyAsync(&host_error, state.error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    diagnostic.scalar_d2h_bytes += sizeof(int);
    ++diagnostic.synchronizations;
    if (host_error)
      throw std::runtime_error("nonfinite RCCSD generated CUDA tensor at node " +
                               std::to_string(std::abs(host_error)));
  }
};

void run_diis(Owner& s, const SolverOptions& options,
              const generated::DeviceIterationOutputs& trial) {
  if (!options.diis_size) return;
  int count = static_cast<int>(s.history);
  if (count == static_cast<int>(options.diis_size)) {
    vibeqc::cc::history_shift<<<
        vibeqc_tensor::blocks(static_cast<vibeqc_tensor::I>(s.elements), 256), 256, 0, s.stream>>>(
        s.vectors, static_cast<vibeqc_tensor::I>(s.elements), count);
    vibeqc::cc::history_shift<<<
        vibeqc_tensor::blocks(static_cast<vibeqc_tensor::I>(s.elements), 256), 256, 0, s.stream>>>(
        s.errors, static_cast<vibeqc_tensor::I>(s.elements), count);
    --count;
  }
  const int slot = count++;
  cuda_check(cudaMemcpyAsync(s.vectors + std::size_t(slot) * s.elements, s.state.t1,
                             s.n1 * sizeof(double), cudaMemcpyDeviceToDevice, s.stream));
  cuda_check(cudaMemcpyAsync(s.vectors + std::size_t(slot) * s.elements + s.n1, s.state.t2,
                             s.n2 * sizeof(double), cudaMemcpyDeviceToDevice, s.stream));
  cuda_check(cudaMemcpyAsync(s.errors + std::size_t(slot) * s.elements, trial.r1,
                             s.n1 * sizeof(double), cudaMemcpyDeviceToDevice, s.stream));
  cuda_check(cudaMemcpyAsync(s.errors + std::size_t(slot) * s.elements + s.n1, trial.r2,
                             s.n2 * sizeof(double), cudaMemcpyDeviceToDevice, s.stream));
  while (count > 1) {
    gram_kernel<<<count * count, 256, 0, s.stream>>>(s.errors, s.elements, count, s.gram);
    vibeqc::cc::diis_coefficients<<<1, 1, 0, s.stream>>>(s.gram, count, s.system, s.coefficients,
                                                         s.status);
    int host_status = 1;
    cuda_check(
        cudaMemcpyAsync(&host_status, s.status, sizeof(int), cudaMemcpyDeviceToHost, s.stream));
    cuda_check(cudaStreamSynchronize(s.stream));
    s.diagnostic.scalar_d2h_bytes += sizeof(int);
    ++s.diagnostic.synchronizations;
    if (host_status == 2) break;
    if (host_status == 0) {
      cuda_check(cudaMemsetAsync(s.arithmetic, 0, sizeof(int), s.stream));
      vibeqc::cc::diis_combine_slice<<<
          vibeqc_tensor::blocks(static_cast<vibeqc_tensor::I>(s.n1), 256), 256, 0, s.stream>>>(
          s.vectors, s.coefficients, static_cast<vibeqc_tensor::I>(s.elements), 0,
          static_cast<vibeqc_tensor::I>(s.n1), count, s.status, s.state.t1, s.arithmetic);
      vibeqc::cc::diis_combine_slice<<<
          vibeqc_tensor::blocks(static_cast<vibeqc_tensor::I>(s.n2), 256), 256, 0, s.stream>>>(
          s.vectors, s.coefficients, static_cast<vibeqc_tensor::I>(s.elements),
          static_cast<vibeqc_tensor::I>(s.n1), static_cast<vibeqc_tensor::I>(s.n2), count, s.status,
          s.state.t2, s.arithmetic);
      int arithmetic = 0;
      cuda_check(cudaMemcpyAsync(&arithmetic, s.arithmetic, sizeof(int), cudaMemcpyDeviceToHost,
                                 s.stream));
      cuda_check(cudaStreamSynchronize(s.stream));
      s.diagnostic.scalar_d2h_bytes += sizeof(int);
      ++s.diagnostic.synchronizations;
      if (arithmetic) throw std::runtime_error("nonfinite RCCSD CUDA DIIS extrapolation");
      break;
    }
    vibeqc::cc::history_shift<<<
        vibeqc_tensor::blocks(static_cast<vibeqc_tensor::I>(s.elements), 256), 256, 0, s.stream>>>(
        s.vectors, static_cast<vibeqc_tensor::I>(s.elements), count);
    vibeqc::cc::history_shift<<<
        vibeqc_tensor::blocks(static_cast<vibeqc_tensor::I>(s.elements), 256), 256, 0, s.stream>>>(
        s.errors, static_cast<vibeqc_tensor::I>(s.elements), count);
    --count;
    ++s.restarts;
  }
  s.history = static_cast<unsigned>(count);
  cuda_check(cudaGetLastError());
}

}  // namespace

SolverResult solve_cuda(const Problem& p, const SolverOptions& options, int device) {
  validate_problem(p);
  validate_options(options);
  Owner owner(p, options, device);
  SolverResult result;
  result.reason = "maximum RCCSD iterations reached";
  result.correlation_energy = std::numeric_limits<double>::quiet_NaN();
  result.total_energy = std::numeric_limits<double>::quiet_NaN();
  double previous = std::numeric_limits<double>::quiet_NaN();
  bool use_last = false;
  const auto started = std::chrono::steady_clock::now();

  for (unsigned iteration = 0; iteration <= options.max_iterations; ++iteration) {
    try {
      const auto output = generated::run_iteration_cuda(owner.state);
      const auto status = owner.read_status(output);
      const double delta = std::isfinite(previous) ? std::abs(status[0] - previous)
                                                   : std::numeric_limits<double>::infinity();
      result.correlation_energy = status[0];
      result.total_energy = p.reference_energy + status[0];
      owner.diagnostic.iterations = iteration + 1;
      owner.diagnostic.energy_change = delta;
      owner.diagnostic.r1_max = status[1];
      owner.diagnostic.r2_max = status[2];
      if (std::isfinite(previous) && delta <= options.energy_tolerance &&
          std::max(status[1], status[2]) <= options.residual_tolerance) {
        const auto replay = generated::run_replay_cuda(owner.state);
        const auto replay_status = owner.read_status(replay);
        owner.diagnostic.replay_r1_max = replay_status[1];
        owner.diagnostic.replay_r2_max = replay_status[2];
        if (std::max(replay_status[1], replay_status[2]) <= options.residual_tolerance &&
            std::abs(replay_status[0] - status[0]) <= options.energy_tolerance) {
          result.status = SolveStatus::Converged;
          result.reason = "energy change and freshly expanded physical R1/R2 passed on GPU";
          break;
        }
      }
      if (iteration == options.max_iterations) break;
      owner.advance(output, 1.0 - options.damping);
      owner.check_generated_error();
      const auto trial = generated::run_iteration_cuda(owner.state);
      owner.check_generated_error();
      run_diis(owner, options, trial);
      previous = status[0];
    } catch (const std::runtime_error& error) {
      const std::string message = error.what();
      if (message.find("nonfinite RCCSD") == std::string::npos) throw;
      result.status = SolveStatus::NumericalFailure;
      result.reason = message;
      use_last = true;
      break;
    }
  }
  owner.diagnostic.diis_restarts = owner.restarts;
  owner.diagnostic.tensor_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  result.diagnostic = owner.diagnostic;
  result.t1.resize(owner.n1);
  result.t2.resize(owner.n2);
  const double* final_t1 = use_last ? owner.last_t1 : owner.state.t1;
  const double* final_t2 = use_last ? owner.last_t2 : owner.state.t2;
  cuda_check(cudaMemcpyAsync(result.t1.data(), final_t1, owner.n1 * sizeof(double),
                             cudaMemcpyDeviceToHost, owner.stream));
  cuda_check(cudaMemcpyAsync(result.t2.data(), final_t2, owner.n2 * sizeof(double),
                             cudaMemcpyDeviceToHost, owner.stream));
  cuda_check(cudaStreamSynchronize(owner.stream));
  result.diagnostic.amplitude_d2h_bytes = (owner.n1 + owner.n2) * sizeof(double);
  ++result.diagnostic.synchronizations;
  return result;
}

}  // namespace vibeqc::cc

#endif
