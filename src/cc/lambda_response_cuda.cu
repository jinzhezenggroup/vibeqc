#include "cc/lambda_response.hpp"

#if VIBEQC_HAS_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "cc/cuda_solver_support.cuh"
#include "generated_rccsd_cpu.hpp"
#include "tensor/cuda_error.hpp"

namespace vibeqc::cc {
namespace {

using vibeqc_tensor::cuda_check;

std::size_t checked_add(std::size_t a, std::size_t b) { return generated::checked_add(a, b); }

std::size_t checked_mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("RCCSD CUDA Lambda size overflow");
  return a * b;
}

std::size_t bytes(std::size_t count) { return checked_mul(count, sizeof(double)); }

std::size_t align256(std::size_t value) {
  const auto remainder = value % 256;
  return remainder ? checked_add(value, 256 - remainder) : value;
}

struct DeviceScope {
  int previous{-1};
  explicit DeviceScope(int device) {
    cuda_check(cudaGetDevice(&previous));
    cuda_check(cudaSetDevice(device));
  }
  ~DeviceScope() {
    if (previous >= 0) (void)cudaSetDevice(previous);
  }
};

struct AmplitudeLayout {
  std::size_t o{}, v{}, n1{}, n2{};
  std::vector<std::size_t> representatives;
  std::vector<std::size_t> partners;
  std::vector<double> sqrt_weights;

  AmplitudeLayout(std::size_t occupied, std::size_t virtuals)
      : o(occupied),
        v(virtuals),
        n1(checked_mul(o, v)),
        n2(checked_mul(checked_mul(o, o), checked_mul(v, v))) {}

  std::size_t pair_count() const { return checked_add(n2, n1) / 2; }
  std::size_t dimension() const { return checked_add(n1, pair_count()); }

  void initialize() {
    sqrt_weights.assign(dimension(), 1.0);
    representatives.resize(pair_count());
    partners.resize(pair_count());
    std::size_t position = 0;
    for (std::size_t i = 0; i < o; ++i)
      for (std::size_t j = 0; j < o; ++j)
        for (std::size_t a = 0; a < v; ++a)
          for (std::size_t b = 0; b < v; ++b) {
            const auto flat = ((i * o + j) * v + a) * v + b;
            const auto mate = ((j * o + i) * v + b) * v + a;
            if (flat > mate) continue;
            representatives[position] = flat;
            partners[position] = mate;
            sqrt_weights[n1 + position] = flat == mate ? 1.0 : std::sqrt(2.0);
            ++position;
          }
  }

  void validate_dense(std::span<const double> one, std::span<const double> two) const {
    if (one.size() != n1 || two.size() != n2)
      throw std::invalid_argument("RCCSD CUDA Lambda amplitude shape mismatch");
    for (double value : one)
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite RCCSD Lambda singles");
    for (std::size_t k = 0; k < representatives.size(); ++k) {
      const double first = two[representatives[k]], second = two[partners[k]];
      if (!std::isfinite(first) || !std::isfinite(second) ||
          std::abs(first - second) > 1e-10 * (1.0 + std::max(std::abs(first), std::abs(second))))
        throw std::invalid_argument("RCCSD doubles violate simultaneous pair symmetry");
    }
  }

  void pack_weighted(std::span<const double> one, std::span<const double> two,
                     std::span<double> output) const {
    validate_dense(one, two);
    if (output.size() != dimension())
      throw std::invalid_argument("RCCSD CUDA Lambda packed output shape mismatch");
    std::copy(one.begin(), one.end(), output.begin());
    for (std::size_t k = 0; k < representatives.size(); ++k)
      output[n1 + k] = sqrt_weights[n1 + k] * two[representatives[k]];
  }

  void unpack_weighted(std::span<const double> packed, std::span<double> one,
                       std::span<double> two) const {
    if (packed.size() != dimension() || one.size() != n1 || two.size() != n2)
      throw std::invalid_argument("RCCSD CUDA Lambda unpack shape mismatch");
    std::fill(two.begin(), two.end(), 0.0);
    std::copy_n(packed.begin(), static_cast<std::ptrdiff_t>(n1), one.begin());
    for (std::size_t k = 0; k < representatives.size(); ++k) {
      const double value = packed[n1 + k] / sqrt_weights[n1 + k];
      two[representatives[k]] = value;
      two[partners[k]] = value;
    }
  }
};

struct DeviceLayout {
  std::array<std::size_t, 14> inputs{};
  std::size_t replay_arena{};
  std::size_t response_arena{};
  std::size_t energy_seed{};
  std::size_t residual_one{};
  std::size_t residual_two{};
  std::size_t error{};
  std::size_t total{};
};

std::size_t reserve(std::size_t& cursor, std::size_t amount) {
  cursor = align256(cursor);
  const auto offset = cursor;
  cursor = checked_add(cursor, amount);
  return offset;
}

int checked_device(int device) {
  if (device < 0) throw std::invalid_argument("RCCSD CUDA Lambda requires a valid device");
  return device;
}

class CudaLambdaActions {
 public:
  CudaLambdaActions(const Problem& p, const SolverResult& cc, const LambdaOptions& options,
                    int device, bool with_source)
      : scope_(checked_device(device)), layout_(p.nocc, p.nvir) {
    layout_.initialize();
    layout_.validate_dense(cc.t1, cc.t2);

    const std::array<const std::vector<double>*, 14> host = {
        &p.foo,  &p.fov,  &p.fvv,  &p.ovov, &p.ovvo, &p.oovv, &p.ovvv,
        &p.ovoo, &p.oooo, &p.vvvv, &p.d1,   &p.d2,   &cc.t1,  &cc.t2};
    std::size_t cursor = 0;
    for (std::size_t index = 0; index < host.size(); ++index)
      device_layout_.inputs[index] = reserve(cursor, bytes(host[index]->size()));
    device_layout_.replay_arena =
        reserve(cursor, bytes(generated::replay_arena_elements(p.nocc, p.nvir)));
    const auto response_elements =
        std::max({generated::lambda_rhs_arena_elements(p.nocc, p.nvir),
                  generated::lambda_transpose_arena_elements(p.nocc, p.nvir),
                  generated::lambda_independent_rhs_arena_elements(p.nocc, p.nvir),
                  generated::lambda_independent_transpose_arena_elements(p.nocc, p.nvir)});
    device_layout_.response_arena = reserve(cursor, bytes(response_elements));
    device_layout_.energy_seed = reserve(cursor, sizeof(double));
    device_layout_.residual_one = reserve(cursor, bytes(layout_.n1));
    device_layout_.residual_two = reserve(cursor, bytes(layout_.n2));
    device_layout_.error = reserve(cursor, sizeof(int));
    device_layout_.total = align256(cursor);

    numeric_capacity_bytes_ = lambda_cpu_numeric_capacity(p, cc, options, with_source);
    if (numeric_capacity_bytes_ > options.max_bytes || device_layout_.total > options.max_bytes)
      throw std::length_error("RCCSD CUDA Lambda exceeds host or device budget");

    try {
      cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&base_), device_layout_.total));
      std::array<double**, 14> fields = {&state_.foo,  &state_.fov,  &state_.fvv,  &state_.ovov,
                                         &state_.ovvo, &state_.oovv, &state_.ovvv, &state_.ovoo,
                                         &state_.oooo, &state_.vvvv, &state_.d1,   &state_.d2,
                                         &state_.t1,   &state_.t2};
      for (std::size_t index = 0; index < host.size(); ++index) {
        *fields[index] = reinterpret_cast<double*>(base_ + device_layout_.inputs[index]);
        const auto amount = bytes(host[index]->size());
        if (amount)
          cuda_check(cudaMemcpyAsync(*fields[index], host[index]->data(), amount,
                                     cudaMemcpyHostToDevice, stream_));
        h2d_bytes_ = checked_add(h2d_bytes_, amount);
      }
      state_.o = p.nocc;
      state_.v = p.nvir;
      state_.stream = stream_;
      state_.replay_arena = reinterpret_cast<double*>(base_ + device_layout_.replay_arena);
      state_.response_arena = reinterpret_cast<double*>(base_ + device_layout_.response_arena);
      state_.bar_correlation_energy = reinterpret_cast<double*>(base_ + device_layout_.energy_seed);
      state_.bar_singles_residual = reinterpret_cast<double*>(base_ + device_layout_.residual_one);
      state_.bar_doubles_residual = reinterpret_cast<double*>(base_ + device_layout_.residual_two);
      state_.error = reinterpret_cast<int*>(base_ + device_layout_.error);
      cuda_check(cudaStreamSynchronize(stream_));
      ++synchronizations_;
    } catch (const vibeqc_tensor::DeviceAllocationError&) {
      cleanup();
      throw std::bad_alloc();
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~CudaLambdaActions() { cleanup(); }

  CudaLambdaActions(const CudaLambdaActions&) = delete;
  CudaLambdaActions& operator=(const CudaLambdaActions&) = delete;

  const AmplitudeLayout& layout() const { return layout_; }
  std::size_t owned_device_bytes() const { return device_layout_.total; }
  std::size_t numeric_capacity_bytes() const { return numeric_capacity_bytes_; }
  std::size_t h2d_bytes() const { return h2d_bytes_; }
  std::size_t d2h_bytes() const { return d2h_bytes_; }
  std::size_t synchronizations() const { return synchronizations_; }

  void fresh_replay(double& energy, std::vector<double>& r1, std::vector<double>& r2) {
    const auto output = generated::run_replay_cuda(state_);
    r1.resize(layout_.n1);
    r2.resize(layout_.n2);
    int error = 0;
    cuda_check(
        cudaMemcpyAsync(&energy, output.energy, sizeof(double), cudaMemcpyDeviceToHost, stream_));
    cuda_check(
        cudaMemcpyAsync(r1.data(), output.r1, bytes(layout_.n1), cudaMemcpyDeviceToHost, stream_));
    cuda_check(
        cudaMemcpyAsync(r2.data(), output.r2, bytes(layout_.n2), cudaMemcpyDeviceToHost, stream_));
    cuda_check(cudaMemcpyAsync(&error, state_.error, sizeof(int), cudaMemcpyDeviceToHost, stream_));
    cuda_check(cudaStreamSynchronize(stream_));
    d2h_bytes_ = checked_add(
        d2h_bytes_, checked_add(sizeof(double) + sizeof(int), bytes(layout_.n1 + layout_.n2)));
    ++synchronizations_;
    check_error(error);
  }

  void rhs(bool independent, std::vector<double>& one, std::vector<double>& two) {
    const double seed = -1.0;
    cuda_check(cudaMemcpyAsync(state_.bar_correlation_energy, &seed, sizeof(double),
                               cudaMemcpyHostToDevice, stream_));
    h2d_bytes_ = checked_add(h2d_bytes_, sizeof(double));
    copy_output(independent ? generated::run_lambda_independent_rhs_cuda(state_)
                            : generated::run_lambda_rhs_cuda(state_),
                one, two);
  }

  void transpose(bool independent, std::span<const double> one, std::span<const double> two,
                 std::vector<double>& out_one, std::vector<double>& out_two) {
    if (one.size() != layout_.n1 || two.size() != layout_.n2)
      throw std::invalid_argument("RCCSD CUDA Lambda transpose seed shape mismatch");
    cuda_check(cudaMemcpyAsync(state_.bar_singles_residual, one.data(), bytes(layout_.n1),
                               cudaMemcpyHostToDevice, stream_));
    cuda_check(cudaMemcpyAsync(state_.bar_doubles_residual, two.data(), bytes(layout_.n2),
                               cudaMemcpyHostToDevice, stream_));
    h2d_bytes_ = checked_add(h2d_bytes_, bytes(layout_.n1 + layout_.n2));
    copy_output(independent ? generated::run_lambda_independent_transpose_cuda(state_)
                            : generated::run_lambda_transpose_cuda(state_),
                out_one, out_two);
  }

 private:
  void copy_output(const generated::DeviceLambdaOutputs& output, std::vector<double>& one,
                   std::vector<double>& two) {
    one.resize(layout_.n1);
    two.resize(layout_.n2);
    int error = 0;
    cuda_check(
        cudaMemcpyAsync(one.data(), output.t1, bytes(layout_.n1), cudaMemcpyDeviceToHost, stream_));
    cuda_check(
        cudaMemcpyAsync(two.data(), output.t2, bytes(layout_.n2), cudaMemcpyDeviceToHost, stream_));
    cuda_check(cudaMemcpyAsync(&error, state_.error, sizeof(int), cudaMemcpyDeviceToHost, stream_));
    cuda_check(cudaStreamSynchronize(stream_));
    d2h_bytes_ = checked_add(d2h_bytes_, checked_add(bytes(layout_.n1 + layout_.n2), sizeof(int)));
    ++synchronizations_;
    check_error(error);
  }

  static void check_error(int error) {
    if (error)
      throw std::runtime_error("nonfinite RCCSD generated CUDA Lambda tensor at node " +
                               std::to_string(std::abs(error)));
  }

  void cleanup() noexcept {
    if (stream_) (void)cudaStreamSynchronize(stream_);
    if (base_) (void)cudaFree(base_);
    if (stream_) (void)cudaStreamDestroy(stream_);
    base_ = nullptr;
    stream_ = nullptr;
  }

  DeviceScope scope_;
  AmplitudeLayout layout_;
  DeviceLayout device_layout_;
  cudaStream_t stream_{};
  unsigned char* base_{};
  generated::CudaState state_{};
  std::size_t numeric_capacity_bytes_{};
  std::size_t h2d_bytes_{};
  std::size_t d2h_bytes_{};
  std::size_t synchronizations_{};
};

double max_abs(std::span<const double> values) {
  double result = 0.0;
  for (double value : values) {
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite RCCSD CUDA Lambda residual");
    result = std::max(result, std::abs(value));
  }
  return result;
}

LambdaResult solve_impl(const Problem& p, const SolverResult& cc, std::span<const double> t1_source,
                        std::span<const double> t2_source, int device,
                        const LambdaOptions& options) {
  validate_problem(p);
  validate_lambda_options(options);
  if (!cc.converged())
    throw std::invalid_argument("RCCSD CUDA Lambda requires a converged CC result");
  if (!std::isfinite(cc.correlation_energy))
    throw std::invalid_argument("nonfinite RCCSD CUDA Lambda primal energy");

  const bool with_source = !t1_source.empty() || !t2_source.empty();
  AmplitudeLayout validation_layout(p.nocc, p.nvir);
  validation_layout.initialize();
  if (with_source &&
      (t1_source.size() != validation_layout.n1 || t2_source.size() != validation_layout.n2))
    throw std::invalid_argument("RCCSD CUDA Lambda energy-source shape mismatch");
  for (const auto source : {t1_source, t2_source})
    for (double value : source)
      if (!std::isfinite(value))
        throw std::invalid_argument("nonfinite RCCSD CUDA Lambda energy source");

  CudaLambdaActions owner(p, cc, options, device, with_source);
  const auto& layout = owner.layout();
  double replay_energy = 0.0;
  std::vector<double> dense_one, dense_two;
  owner.fresh_replay(replay_energy, dense_one, dense_two);
  const double replay_r1 = max_abs(dense_one);
  const double replay_r2 = max_abs(dense_two);
  if (std::max(replay_r1, replay_r2) > options.cc_tolerance ||
      std::abs(replay_energy - cc.correlation_energy) > 1e-10)
    throw std::runtime_error("RCCSD CUDA Lambda fresh primal replay gate failed");

  owner.rhs(false, dense_one, dense_two);
  std::vector<double> rhs(layout.dimension());
  layout.pack_weighted(dense_one, dense_two, rhs);

  std::vector<double> packed_source;
  if (with_source) {
    packed_source.resize(layout.dimension());
    std::copy(t1_source.begin(), t1_source.end(), packed_source.begin());
    for (std::size_t k = 0; k < layout.representatives.size(); ++k) {
      const auto first = layout.representatives[k];
      const auto second = layout.partners[k];
      const double projected =
          first == second ? t2_source[first] : 0.5 * (t2_source[first] + t2_source[second]);
      packed_source[layout.n1 + k] = layout.sqrt_weights[layout.n1 + k] * projected;
    }
    for (std::size_t index = 0; index < rhs.size(); ++index) rhs[index] -= packed_source[index];
  }

  std::vector<double> seed_one(layout.n1), seed_two(layout.n2), action_one, action_two;
  auto apply = [&](std::span<const double> input, std::span<double> output) {
    layout.unpack_weighted(input, seed_one, seed_two);
    owner.transpose(false, seed_one, seed_two, action_one, action_two);
    layout.pack_weighted(action_one, action_two, output);
  };

  const auto gmres_plan = response::prepare_gmres(layout.dimension(), options.gmres);
  auto solved = response::solve_gmres(gmres_plan, apply, rhs);
  if (!solved.converged()) throw std::runtime_error("RCCSD CUDA Lambda GMRES did not converge");

  layout.unpack_weighted(solved.solution, seed_one, seed_two);
  owner.transpose(true, seed_one, seed_two, action_one, action_two);
  std::vector<double> independent(layout.dimension());
  layout.pack_weighted(action_one, action_two, independent);

  owner.rhs(true, dense_one, dense_two);
  std::vector<double> independent_rhs(layout.dimension());
  layout.pack_weighted(dense_one, dense_two, independent_rhs);
  if (!packed_source.empty())
    for (std::size_t index = 0; index < independent_rhs.size(); ++index)
      independent_rhs[index] -= packed_source[index];
  for (std::size_t index = 0; index < independent.size(); ++index)
    independent[index] -= independent_rhs[index];

  const double independent_norm = response::stable_norm(independent);
  double independent_max = 0.0;
  for (std::size_t index = 0; index < independent.size(); ++index)
    independent_max =
        std::max(independent_max, std::abs(independent[index] / layout.sqrt_weights[index]));
  if (std::max({solved.residual_norm, independent_norm, independent_max}) >
      options.lambda_tolerance)
    throw std::runtime_error("RCCSD CUDA Lambda independent physical residual gate failed");

  LambdaResult result;
  result.lambda1.resize(layout.n1);
  result.lambda2.resize(layout.n2);
  layout.unpack_weighted(solved.solution, result.lambda1, result.lambda2);
  result.reason = "host GMRES with generated CUDA Lambda actions and physical residual passed";
  result.diagnostic.cc_r1_max = replay_r1;
  result.diagnostic.cc_r2_max = replay_r2;
  result.diagnostic.lambda_residual_norm = solved.residual_norm;
  result.diagnostic.independent_residual_norm = independent_norm;
  result.diagnostic.independent_residual_max = independent_max;
  result.diagnostic.iterations = solved.iterations;
  result.diagnostic.operator_actions = solved.operator_actions;
  result.diagnostic.numeric_capacity_bytes = owner.numeric_capacity_bytes();
  result.diagnostic.owned_device_bytes = owner.owned_device_bytes();
  result.diagnostic.h2d_bytes = owner.h2d_bytes();
  result.diagnostic.d2h_bytes = owner.d2h_bytes();
  result.diagnostic.synchronizations = owner.synchronizations();
  result.diagnostic.cuda_actions = true;
  result.diagnostic.shared_program_hash = generated::lambda_transpose_program_hash;
  result.diagnostic.independent_program_hash = generated::lambda_independent_transpose_program_hash;
  return result;
}

}  // namespace

LambdaResult solve_lambda_cuda(const Problem& problem, const SolverResult& cc_result, int device,
                               const LambdaOptions& options) {
  return solve_impl(problem, cc_result, {}, {}, device, options);
}

LambdaResult solve_lambda_cuda_with_energy_source(const Problem& problem,
                                                  const SolverResult& cc_result,
                                                  std::span<const double> t1_source,
                                                  std::span<const double> t2_source, int device,
                                                  const LambdaOptions& options) {
  return solve_impl(problem, cc_result, t1_source, t2_source, device, options);
}

}  // namespace vibeqc::cc

#endif
