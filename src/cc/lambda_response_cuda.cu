#include "cc/lambda_response.hpp"

#if VIBEQC_HAS_CUDA

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
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
                    int device, bool with_source, bool with_parameters)
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
    auto response_elements =
        std::max({generated::lambda_rhs_arena_elements(p.nocc, p.nvir),
                  generated::lambda_transpose_arena_elements(p.nocc, p.nvir),
                  generated::lambda_independent_rhs_arena_elements(p.nocc, p.nvir),
                  generated::lambda_independent_transpose_arena_elements(p.nocc, p.nvir)});
    if (with_parameters)
      response_elements =
          std::max({response_elements,
                    generated::parameter_foo_arena_elements(p.nocc, p.nvir),
                    generated::parameter_fov_arena_elements(p.nocc, p.nvir),
                    generated::parameter_fvv_arena_elements(p.nocc, p.nvir),
                    generated::parameter_ovov_arena_elements(p.nocc, p.nvir),
                    generated::parameter_ovvo_arena_elements(p.nocc, p.nvir),
                    generated::parameter_oovv_arena_elements(p.nocc, p.nvir),
                    generated::parameter_ovvv_arena_elements(p.nocc, p.nvir),
                    generated::parameter_ovoo_arena_elements(p.nocc, p.nvir),
                    generated::parameter_oooo_arena_elements(p.nocc, p.nvir),
                    generated::parameter_vvvv_arena_elements(p.nocc, p.nvir)});
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

  void set_parameter_seeds(std::span<const double> lambda1, std::span<const double> lambda2) {
    if (lambda1.size() != layout_.n1 || lambda2.size() != layout_.n2)
      throw std::invalid_argument("RCCSD CUDA parameter-response seed shape mismatch");
    const double energy_seed = 1.0;
    cuda_check(cudaMemcpyAsync(state_.bar_correlation_energy, &energy_seed, sizeof(double),
                               cudaMemcpyHostToDevice, stream_));
    cuda_check(cudaMemcpyAsync(state_.bar_singles_residual, lambda1.data(), bytes(layout_.n1),
                               cudaMemcpyHostToDevice, stream_));
    cuda_check(cudaMemcpyAsync(state_.bar_doubles_residual, lambda2.data(), bytes(layout_.n2),
                               cudaMemcpyHostToDevice, stream_));
    h2d_bytes_ = checked_add(
        h2d_bytes_, checked_add(sizeof(double), bytes(layout_.n1 + layout_.n2)));
  }

  using ParameterRunner = generated::DeviceParameterOutput (*)(generated::CudaState&);

  std::vector<double> parameter(ParameterRunner run, std::size_t count) {
    const auto output = run(state_);
    std::vector<double> values(count);
    int error = 0;
    cuda_check(
        cudaMemcpyAsync(values.data(), output.values, bytes(count), cudaMemcpyDeviceToHost, stream_));
    cuda_check(cudaMemcpyAsync(&error, state_.error, sizeof(int), cudaMemcpyDeviceToHost, stream_));
    cuda_check(cudaStreamSynchronize(stream_));
    d2h_bytes_ = checked_add(d2h_bytes_, checked_add(bytes(count), sizeof(int)));
    ++synchronizations_;
    check_error(error);
    return values;
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
                        const LambdaOptions& options,
                        CudaFixedOrbitalResponseResult* fixed_orbital) {
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

  CudaLambdaActions owner(p, cc, options, device, with_source, fixed_orbital != nullptr);
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
  if (fixed_orbital) {
    owner.set_parameter_seeds(result.lambda1, result.lambda2);
    fixed_orbital->foo =
        owner.parameter(generated::run_parameter_foo_cuda, checked_mul(p.nocc, p.nocc));
    fixed_orbital->fov =
        owner.parameter(generated::run_parameter_fov_cuda, checked_mul(p.nocc, p.nvir));
    fixed_orbital->fvv =
        owner.parameter(generated::run_parameter_fvv_cuda, checked_mul(p.nvir, p.nvir));
    const auto oovv = checked_mul(checked_mul(p.nocc, p.nocc), checked_mul(p.nvir, p.nvir));
    fixed_orbital->ovov = owner.parameter(generated::run_parameter_ovov_cuda, oovv);
    fixed_orbital->ovvo = owner.parameter(generated::run_parameter_ovvo_cuda, oovv);
    fixed_orbital->oovv = owner.parameter(generated::run_parameter_oovv_cuda, oovv);
    fixed_orbital->ovvv =
        owner.parameter(generated::run_parameter_ovvv_cuda,
                        checked_mul(p.nocc, checked_mul(p.nvir, checked_mul(p.nvir, p.nvir))));
    fixed_orbital->ovoo =
        owner.parameter(generated::run_parameter_ovoo_cuda,
                        checked_mul(checked_mul(p.nocc, p.nvir),
                                    checked_mul(p.nocc, p.nocc)));
    fixed_orbital->oooo =
        owner.parameter(generated::run_parameter_oooo_cuda,
                        checked_mul(checked_mul(p.nocc, p.nocc),
                                    checked_mul(p.nocc, p.nocc)));
    fixed_orbital->vvvv =
        owner.parameter(generated::run_parameter_vvvv_cuda,
                        checked_mul(checked_mul(p.nvir, p.nvir),
                                    checked_mul(p.nvir, p.nvir)));
  }
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
  return solve_impl(problem, cc_result, {}, {}, device, options, nullptr);
}

LambdaResult solve_lambda_cuda_with_energy_source(const Problem& problem,
                                                  const SolverResult& cc_result,
                                                  std::span<const double> t1_source,
                                                  std::span<const double> t2_source, int device,
                                                  const LambdaOptions& options) {
  return solve_impl(problem, cc_result, t1_source, t2_source, device, options, nullptr);
}

CudaFixedOrbitalResponseResult solve_lambda_parameter_response_cuda_with_energy_source(
    const Problem& problem, const SolverResult& cc_result, std::span<const double> t1_source,
    std::span<const double> t2_source, int device, const LambdaOptions& options) {
  CudaFixedOrbitalResponseResult result;
  result.lambda =
      solve_impl(problem, cc_result, t1_source, t2_source, device, options, &result);
  return result;
}


struct CudaHamiltonianResponseOwner::Impl {
  struct Layout {
    std::array<std::size_t, 4> raw{};
    std::array<std::size_t, 10> parameters{};
    std::size_t reference_seed{};
    std::size_t fock_seed{};
    std::size_t rotation_seed{};
    std::size_t response_arena{};
    std::size_t error{};
    std::size_t total{};
  };

  Impl(std::size_t occupied, std::size_t virtuals, CudaRawHamiltonianView raw, int device,
       std::size_t max_device_bytes)
      : scope(checked_device(device)), o(occupied), v(virtuals), n(checked_add(o, v)) {
    if (!o || !v || !max_device_bytes)
      throw std::invalid_argument("RCCSD CUDA Hamiltonian response requires nonzero dimensions and budget");
    n2 = checked_mul(n, n);
    n4 = checked_mul(n2, n2);
    ov = checked_mul(o, v);
    const std::array<std::span<const double>, 4> raw_values = {
        raw.density, raw.g, raw.h, raw.rotation};
    const std::array<std::size_t, 4> raw_sizes = {n2, n4, n2, n2};
    for (std::size_t index = 0; index < raw_values.size(); ++index)
      validate_values(raw_values[index], raw_sizes[index], "raw Hamiltonian");

    const std::array<std::size_t, 10> parameter_sizes = {
        checked_mul(o, o),
        ov,
        checked_mul(v, v),
        checked_mul(checked_mul(o, o), checked_mul(v, v)),
        checked_mul(checked_mul(o, o), checked_mul(v, v)),
        checked_mul(checked_mul(o, o), checked_mul(v, v)),
        checked_mul(o, checked_mul(v, checked_mul(v, v))),
        checked_mul(ov, checked_mul(o, o)),
        checked_mul(checked_mul(o, o), checked_mul(o, o)),
        checked_mul(checked_mul(v, v), checked_mul(v, v))};

    std::size_t cursor = 0;
    for (std::size_t index = 0; index < raw_sizes.size(); ++index)
      layout.raw[index] = reserve(cursor, bytes(raw_sizes[index]));
    for (std::size_t index = 0; index < parameter_sizes.size(); ++index)
      layout.parameters[index] = reserve(cursor, bytes(parameter_sizes[index]));
    layout.reference_seed = reserve(cursor, sizeof(double));
    layout.fock_seed = reserve(cursor, bytes(n2));
    layout.rotation_seed = reserve(cursor, bytes(n2));
    const auto response_elements =
        std::max({generated::hamiltonian_weights_arena_elements(o, v),
                  generated::fock_weights_arena_elements(o, v),
                  generated::orbital_jvp_arena_elements(o, v)});
    layout.response_arena = reserve(cursor, bytes(response_elements));
    layout.error = reserve(cursor, sizeof(int));
    layout.total = align256(cursor);
    if (layout.total > max_device_bytes)
      throw std::length_error("RCCSD CUDA Hamiltonian response exceeds device budget");

    try {
      cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&base), layout.total));
      std::array<double**, 4> raw_fields = {&state.density, &state.g, &state.h, &state.rotation};
      for (std::size_t index = 0; index < raw_values.size(); ++index) {
        *raw_fields[index] = pointer(layout.raw[index]);
        upload(raw_values[index], *raw_fields[index]);
      }
      std::array<double**, 10> parameter_fields = {
          &state.bar_foo,  &state.bar_fov,  &state.bar_fvv,  &state.bar_ovov, &state.bar_ovvo,
          &state.bar_oovv, &state.bar_ovvv, &state.bar_ovoo, &state.bar_oooo, &state.bar_vvvv};
      for (std::size_t index = 0; index < parameter_fields.size(); ++index)
        *parameter_fields[index] = pointer(layout.parameters[index]);
      state.bar_reference_electronic_energy = pointer(layout.reference_seed);
      state.bar_fock = pointer(layout.fock_seed);
      state.d_rotation = pointer(layout.rotation_seed);
      state.response_arena = pointer(layout.response_arena);
      state.error = reinterpret_cast<int*>(base + layout.error);
      state.o = o;
      state.v = v;
      state.stream = stream;
      cuda_check(cudaStreamSynchronize(stream));
      ++syncs;
    } catch (const vibeqc_tensor::DeviceAllocationError&) {
      cleanup();
      throw std::bad_alloc();
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~Impl() { cleanup(); }

  CudaHamiltonianResponseResult hamiltonian(CudaParameterResponseView parameters,
                                            double reference_seed) {
    if (!std::isfinite(reference_seed))
      throw std::invalid_argument("nonfinite RCCSD CUDA Hamiltonian reference seed");
    const std::array<std::span<const double>, 10> values = {
        parameters.foo,  parameters.fov,  parameters.fvv,  parameters.ovov, parameters.ovvo,
        parameters.oovv, parameters.ovvv, parameters.ovoo, parameters.oooo, parameters.vvvv};
    const std::array<std::size_t, 10> sizes = {
        checked_mul(o, o),
        ov,
        checked_mul(v, v),
        checked_mul(checked_mul(o, o), checked_mul(v, v)),
        checked_mul(checked_mul(o, o), checked_mul(v, v)),
        checked_mul(checked_mul(o, o), checked_mul(v, v)),
        checked_mul(o, checked_mul(v, checked_mul(v, v))),
        checked_mul(ov, checked_mul(o, o)),
        checked_mul(checked_mul(o, o), checked_mul(o, o)),
        checked_mul(checked_mul(v, v), checked_mul(v, v))};
    std::array<double*, 10> fields = {
        state.bar_foo,  state.bar_fov,  state.bar_fvv,  state.bar_ovov, state.bar_ovvo,
        state.bar_oovv, state.bar_ovvv, state.bar_ovoo, state.bar_oooo, state.bar_vvvv};
    for (std::size_t index = 0; index < values.size(); ++index) {
      validate_values(values[index], sizes[index], "parameter response");
      upload(values[index], fields[index]);
    }
    cuda_check(cudaMemcpyAsync(state.bar_reference_electronic_energy, &reference_seed,
                               sizeof(double), cudaMemcpyHostToDevice, stream));
    h2d = checked_add(h2d, sizeof(double));
    clear_error();
    return detach(generated::run_hamiltonian_weights_cuda(state));
  }

  CudaHamiltonianResponseResult fock(std::span<const double> bar_fock) {
    validate_values(bar_fock, n2, "Fock response");
    upload(bar_fock, state.bar_fock);
    clear_error();
    return detach(generated::run_fock_weights_cuda(state));
  }

  std::vector<double> orbital_jvp(std::span<const double> d_rotation) {
    validate_values(d_rotation, n2, "orbital JVP");
    upload(d_rotation, state.d_rotation);
    clear_error();
    const auto output = generated::run_orbital_jvp_cuda(state);
    std::vector<double> result(ov);
    int error = 0;
    cuda_check(cudaMemcpyAsync(result.data(), output.d_fov, bytes(ov), cudaMemcpyDeviceToHost,
                               stream));
    cuda_check(cudaMemcpyAsync(&error, state.error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    d2h = checked_add(d2h, checked_add(bytes(ov), sizeof(int)));
    ++syncs;
    check_error(error);
    return result;
  }

  std::size_t owned_device_bytes() const noexcept { return layout.total; }

  static void validate_values(std::span<const double> values, std::size_t expected,
                              const char* label) {
    if (values.size() != expected)
      throw std::invalid_argument(std::string("RCCSD CUDA ") + label + " shape mismatch");
    for (double value : values)
      if (!std::isfinite(value))
        throw std::invalid_argument(std::string("nonfinite RCCSD CUDA ") + label + " input");
  }

  double* pointer(std::size_t offset) { return reinterpret_cast<double*>(base + offset); }

  void upload(std::span<const double> values, double* target) {
    const auto amount = bytes(values.size());
    if (amount)
      cuda_check(cudaMemcpyAsync(target, values.data(), amount, cudaMemcpyHostToDevice, stream));
    h2d = checked_add(h2d, amount);
  }

  void clear_error() { cuda_check(cudaMemsetAsync(state.error, 0, sizeof(int), stream)); }

  CudaHamiltonianResponseResult detach(const generated::DeviceHamiltonianOutputs& output) {
    CudaHamiltonianResponseResult result;
    result.hcore.resize(n2);
    result.eri.resize(n4);
    result.overlap.resize(n2);
    result.rotation_gradient.resize(n2);
    result.stationarity.resize(n2);
    result.orbital_rhs.resize(ov);
    int error = 0;
    cuda_check(cudaMemcpyAsync(result.hcore.data(), output.hcore, bytes(n2),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(result.eri.data(), output.eri, bytes(n4),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(result.overlap.data(), output.overlap, bytes(n2),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(result.rotation_gradient.data(), output.rotation_gradient, bytes(n2),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(result.stationarity.data(), output.stationarity, bytes(n2),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(result.orbital_rhs.data(), output.orbital_rhs, bytes(ov),
                               cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaMemcpyAsync(&error, state.error, sizeof(int), cudaMemcpyDeviceToHost, stream));
    cuda_check(cudaStreamSynchronize(stream));
    const auto output_bytes =
        bytes(checked_add(n4, checked_add(checked_mul(4, n2), ov)));
    d2h = checked_add(d2h, checked_add(output_bytes, sizeof(int)));
    ++syncs;
    check_error(error);
    return result;
  }

  static void check_error(int error) {
    if (error)
      throw std::runtime_error("nonfinite RCCSD generated CUDA Hamiltonian tensor at node " +
                               std::to_string(std::abs(error)));
  }

  void cleanup() noexcept {
    if (stream) (void)cudaStreamSynchronize(stream);
    if (base) (void)cudaFree(base);
    if (stream) (void)cudaStreamDestroy(stream);
    base = nullptr;
    stream = nullptr;
  }

  DeviceScope scope;
  std::size_t o{}, v{}, n{}, n2{}, n4{}, ov{};
  Layout layout;
  cudaStream_t stream{};
  unsigned char* base{};
  generated::CudaState state{};
  std::size_t h2d{};
  std::size_t d2h{};
  std::size_t syncs{};
};

CudaHamiltonianResponseOwner::CudaHamiltonianResponseOwner(
    std::size_t nocc, std::size_t nvir, CudaRawHamiltonianView raw, int device,
    std::size_t max_device_bytes)
    : impl_(std::make_unique<Impl>(nocc, nvir, raw, device, max_device_bytes)) {}

CudaHamiltonianResponseOwner::~CudaHamiltonianResponseOwner() = default;

CudaHamiltonianResponseResult CudaHamiltonianResponseOwner::hamiltonian(
    CudaParameterResponseView parameters, double reference_seed) {
  return impl_->hamiltonian(parameters, reference_seed);
}

CudaHamiltonianResponseResult CudaHamiltonianResponseOwner::fock(
    std::span<const double> bar_fock) {
  return impl_->fock(bar_fock);
}

std::vector<double> CudaHamiltonianResponseOwner::orbital_jvp(
    std::span<const double> d_rotation) {
  return impl_->orbital_jvp(d_rotation);
}

std::size_t CudaHamiltonianResponseOwner::owned_device_bytes() const noexcept {
  return impl_->owned_device_bytes();
}

std::size_t CudaHamiltonianResponseOwner::h2d_bytes() const noexcept { return impl_->h2d; }

std::size_t CudaHamiltonianResponseOwner::d2h_bytes() const noexcept { return impl_->d2h; }

std::size_t CudaHamiltonianResponseOwner::synchronizations() const noexcept {
  return impl_->syncs;
}

}  // namespace vibeqc::cc

#endif
