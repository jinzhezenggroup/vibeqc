#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "cc/df_lambda.hpp"
#include "generated_df_ccsd_core_cpu.hpp"
#include "generated_df_ccsd_cuda.cuh"
#include "generated_df_lambda_cuda.cuh"
#include "posthf/capacity.hpp"
#include "runtime/cuda_resources.cuh"

namespace generativeqc::cc::detail {
namespace {
using generativeqc_tensor::cuda_check;
using posthf::checked_add;
using posthf::checked_mul;
namespace generated_core = generated::dfcore;
namespace generated_virtual = generated::df;
namespace generated_response = generated::dflambda;

std::size_t bytes(std::size_t n) { return checked_mul(n, sizeof(double)); }
std::size_t reserve(std::size_t& cursor, std::size_t amount) {
  if (cursor % 256) cursor = checked_add(cursor, 256 - cursor % 256);
  const auto offset = cursor;
  cursor = checked_add(cursor, amount);
  return offset;
}
struct Fence {
  cudaStream_t stream;
  ~Fence() {
    if (stream) (void)cudaStreamSynchronize(stream);
  }
  void complete() noexcept { stream = nullptr; }
};
struct Storage {
  cudaStream_t stream{};
  unsigned char* base{};
  ~Storage() {
    if (stream) (void)cudaStreamSynchronize(stream);
    if (base) (void)cudaFree(base);
    if (stream) (void)cudaStreamDestroy(stream);
  }
};

__global__ void accumulate(const double* source, std::size_t count, double* target, int* error) {
  for (std::size_t x = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; x < count;
       x += std::size_t(blockDim.x) * gridDim.x)
    target[x] = generativeqc_tensor::finite(target[x] + source[x], error, 1);
}

using Query = std::size_t (*)(std::size_t, std::size_t);
using Runner = generated::DeviceParameterOutput (*)(generated_response::CudaState&);
struct Parameter {
  std::string_view name;
  Query arena, work;
  std::size_t operations;
  Runner run;
};
#define GQC_DF_PARAMETER(name)                                    \
  Parameter {                                                     \
    #name, generated_response::parameter_##name##_arena_elements, \
        generated_response::parameter_##name##_contraction_terms, \
        generated_response::parameter_##name##_operations,        \
        generated_response::run_parameter_##name##_cuda           \
  }
const std::array parameters{GQC_DF_PARAMETER(foo),  GQC_DF_PARAMETER(fov),  GQC_DF_PARAMETER(fvv),
                            GQC_DF_PARAMETER(ovov), GQC_DF_PARAMETER(ovvo), GQC_DF_PARAMETER(oovv),
                            GQC_DF_PARAMETER(ovoo), GQC_DF_PARAMETER(oooo)};
#undef GQC_DF_PARAMETER
}  // namespace

struct DFLambdaActions::Impl {
  runtime::CudaDeviceScope device_scope;
  Storage storage;
  generated_response::CudaState state;
  generated_virtual::CudaState auxiliary;
  std::size_t q{}, n1{}, n2{}, vv{};
  const double *bov{}, *bvv{};
  double *sum1{}, *sum2{};
  LambdaDiagnostic metrics;
  bool parameters_admitted{};

  Impl(const Problem& p, const SolverResult& cc, const LambdaOptions& options, int device,
       bool with_source, bool with_parameters)
      : device_scope(device) {
    validate_problem(p, true);
    validate_lambda_options(options);
    if (!p.naux) throw std::invalid_argument("DF Lambda actions require auxiliary factors");
    const auto o = p.nocc, v = p.nvir;
    q = p.naux;
    n1 = checked_mul(o, v);
    n2 = checked_mul(n1, n1);
    vv = checked_mul(v, v);
    if (cc.t1.size() != n1 || cc.t2.size() != n2)
      throw std::invalid_argument("DF Lambda CC amplitude shape mismatch");
    const std::array<const std::vector<double>*, 16> host{
        &p.foo,  &p.fov,  &p.fvv, &p.ovov, &p.ovvo, &p.oovv, &p.ovvv,   &p.ovoo,
        &p.oooo, &p.vvvv, &p.d1,  &p.d2,   &cc.t1,  &cc.t2,  &p.df_bov, &p.df_bvv};
    std::array<std::size_t, 16> offsets{};
    std::size_t cursor = 0;
    for (std::size_t x = 0; x < host.size(); ++x)
      offsets[x] = reserve(cursor, bytes(host[x]->size()));
    std::size_t scratch = std::max({generated_core::replay_arena_elements(o, v),
                                    generated_response::rhs_arena_elements(o, v),
                                    generated_response::transpose_arena_elements(o, v),
                                    generated_response::independent_rhs_arena_elements(o, v),
                                    generated_response::independent_transpose_arena_elements(o, v),
                                    generated_virtual::virtual_cuda_arena_elements(o, v),
                                    generated_virtual::amplitude_vjp_cuda_arena_elements(o, v)});
    if (with_parameters) {
      for (const auto& item : parameters) scratch = std::max(scratch, item.arena(o, v));
      scratch = std::max(scratch, generated_virtual::factor_vjp_cuda_arena_elements(o, v));
    }
    const auto arena = reserve(cursor, bytes(scratch));
    const auto sums1 = reserve(cursor, bytes(n1)), sums2 = reserve(cursor, bytes(n2));
    const auto seed0 = reserve(cursor, sizeof(double)), seed1 = reserve(cursor, bytes(n1)),
               seed2 = reserve(cursor, bytes(n2));
    const auto error = reserve(cursor, sizeof(int));
    metrics.owned_device_bytes = cursor;
    // Conservative simultaneous host bound: both packed layouts (validation
    // and owner), dense seeds/actions/results, packed RHS/audits, GMRES storage,
    // optional external energy seeds and all detached parameter publications.
    const auto pairs = checked_add(n1, n2) / 2, dim = checked_add(n1, pairs),
               dense = checked_add(n1, n2);
    auto host_bytes = checked_add(p.reference_retained_bytes, problem_host_bytes(p));
    host_bytes = checked_add(host_bytes, bytes(checked_add(cc.t1.capacity(), cc.t2.capacity())));
    host_bytes = checked_add(host_bytes, bytes(checked_mul(6, checked_add(dense, dim))));
    host_bytes = checked_add(
        host_bytes, checked_mul(2, checked_add(bytes(dim), checked_mul(checked_mul(2, pairs),
                                                                       sizeof(std::size_t)))));
    host_bytes =
        checked_add(host_bytes, response::prepare_gmres(dim, options.gmres).workspace_bytes);
    if (with_source) host_bytes = checked_add(host_bytes, bytes(checked_add(dim, dense)));
    if (with_parameters) {
      std::size_t outputs = checked_add(p.df_bov.size(), p.df_bvv.size());
      for (const auto* values :
           {&p.foo, &p.fov, &p.fvv, &p.ovov, &p.ovvo, &p.oovv, &p.ovoo, &p.oooo})
        outputs = checked_add(outputs, values->size());
      host_bytes = checked_add(host_bytes, bytes(outputs));
    }
    metrics.numeric_capacity_bytes = checked_add(host_bytes, metrics.owned_device_bytes);
    if (metrics.numeric_capacity_bytes > options.max_bytes)
      throw std::length_error("DF Lambda complete numeric storage exceeds budget");
    parameters_admitted = with_parameters;
    cuda_check(cudaStreamCreateWithFlags(&storage.stream, cudaStreamNonBlocking));
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&storage.base), cursor));
    auto at = [&](std::size_t offset) { return reinterpret_cast<double*>(storage.base + offset); };
    const std::array<double**, 14> fields{
        &state.foo,  &state.fov,  &state.fvv,  &state.ovov, &state.ovvo, &state.oovv, &state.ovvv,
        &state.ovoo, &state.oooo, &state.vvvv, &state.d1,   &state.d2,   &state.t1,   &state.t2};
    for (std::size_t x = 0; x < host.size(); ++x) {
      if (x < fields.size()) *fields[x] = at(offsets[x]);
      if (host[x]->size()) upload(at(offsets[x]), host[x]->data(), host[x]->size());
    }
    bov = at(offsets[14]);
    bvv = at(offsets[15]);
    state.o = o;
    state.v = v;
    state.stream = storage.stream;
    state.response_arena = state.replay_arena = at(arena);
    state.error = reinterpret_cast<int*>(storage.base + error);
    state.bar_correlation_energy = at(seed0);
    state.bar_singles_residual = at(seed1);
    state.bar_doubles_residual = at(seed2);
    sum1 = at(sums1);
    sum2 = at(sums2);
    state.df_virtual_singles = sum1;
    state.df_virtual_doubles = sum2;
    auxiliary.o = o;
    auxiliary.v = v;
    auxiliary.stream = storage.stream;
    auxiliary.error = state.error;
    auxiliary.response_arena = state.response_arena;
    auxiliary.t1 = state.t1;
    auxiliary.t2 = state.t2;
    auxiliary.bar_df_virtual_singles = state.bar_singles_residual;
    auxiliary.bar_df_virtual_doubles = state.bar_doubles_residual;
    cuda_check(cudaStreamSynchronize(storage.stream));
    ++metrics.synchronizations;
    metrics.shared_program_hash = generated_response::operator_hash;
    metrics.independent_program_hash = generated_response::independent_operator_hash;
    metrics.cuda_actions = true;
  }

  void upload(double* target, const double* source, std::size_t count) {
    cuda_check(
        cudaMemcpyAsync(target, source, bytes(count), cudaMemcpyHostToDevice, storage.stream));
    metrics.h2d_bytes = checked_add(metrics.h2d_bytes, bytes(count));
  }
  void download(double* target, const double* source, std::size_t count) {
    cuda_check(
        cudaMemcpyAsync(target, source, bytes(count), cudaMemcpyDeviceToHost, storage.stream));
    metrics.d2h_bytes = checked_add(metrics.d2h_bytes, bytes(count));
  }
  void clear_error() { cuda_check(cudaMemsetAsync(state.error, 0, sizeof(int), storage.stream)); }
  void finish() {
    int error = 0;
    Fence fence{storage.stream};
    cuda_check(
        cudaMemcpyAsync(&error, state.error, sizeof(int), cudaMemcpyDeviceToHost, storage.stream));
    cuda_check(cudaStreamSynchronize(storage.stream));
    fence.complete();
    ++metrics.synchronizations;
    metrics.d2h_bytes = checked_add(metrics.d2h_bytes, sizeof(int));
    if (error) throw std::runtime_error("nonfinite native DF Lambda generated action");
  }
  void record(Query work, std::size_t operations) {
    metrics.df_contraction_terms =
        checked_add(metrics.df_contraction_terms, work(state.o, state.v));
    metrics.df_generated_kernels = checked_add(metrics.df_generated_kernels, operations);
  }
  void select(std::size_t index) {
    auxiliary.bov = bov + index * n1;
    auxiliary.bvv = bvv + index * vv;
    ++metrics.df_auxiliary_slices;
  }
  void add(const double* one, const double* two) {
    accumulate<<<generativeqc_tensor::blocks(n1, 256), 256, 0, storage.stream>>>(one, n1, sum1,
                                                                                 state.error);
    accumulate<<<generativeqc_tensor::blocks(n2, 256), 256, 0, storage.stream>>>(two, n2, sum2,
                                                                                 state.error);
    cuda_check(cudaGetLastError());
    metrics.df_generated_kernels = checked_add(metrics.df_generated_kernels, 2);
  }
  void output(const double* a, const double* b, std::vector<double>& one,
              std::vector<double>& two) {
    one.resize(n1);
    two.resize(n2);
    Fence fence{storage.stream};
    download(one.data(), a, n1);
    download(two.data(), b, n2);
    finish();
    fence.complete();
  }
};

DFLambdaActions::DFLambdaActions(const Problem& p, const SolverResult& cc,
                                 const LambdaOptions& options, int device, bool with_source,
                                 bool with_parameters)
    : impl_(std::make_unique<Impl>(p, cc, options, device, with_source, with_parameters)) {}
DFLambdaActions::~DFLambdaActions() = default;
const LambdaDiagnostic& DFLambdaActions::diagnostic() const noexcept { return impl_->metrics; }

void DFLambdaActions::replay(double& energy, std::vector<double>& r1, std::vector<double>& r2) {
  auto& s = *impl_;
  s.clear_error();
  cuda_check(cudaMemsetAsync(s.sum1, 0, bytes(s.n1), s.storage.stream));
  cuda_check(cudaMemsetAsync(s.sum2, 0, bytes(s.n2), s.storage.stream));
  for (std::size_t Q = 0; Q < s.q; ++Q) {
    s.select(Q);
    const auto out = generated_virtual::run_virtual_accumulate_cuda(s.auxiliary);
    s.add(out.singles, out.doubles);
    s.record(generated_response::virtual_virtual_contraction_terms,
             generated_virtual::virtual_cuda_operation_count);
  }
  const auto out = generated_core::run_replay_cuda(s.state);
  s.record(generated_response::replay_contraction_terms, generated_core::replay_operation_count);
  Fence fence{s.storage.stream};
  s.download(&energy, out.energy, 1);
  s.output(out.r1, out.r2, r1, r2);
  fence.complete();
}

void DFLambdaActions::rhs(bool independent, std::vector<double>& one, std::vector<double>& two) {
  auto& s = *impl_;
  const double seed = -1;
  Fence fence{s.storage.stream};
  s.clear_error();
  s.upload(s.state.bar_correlation_energy, &seed, 1);
  const auto out = independent ? generated_response::run_independent_rhs_cuda(s.state)
                               : generated_response::run_rhs_cuda(s.state);
  s.record(independent ? generated_response::independent_rhs_contraction_terms
                       : generated_response::rhs_contraction_terms,
           independent ? generated_response::independent_rhs_operations
                       : generated_response::rhs_operations);
  s.output(out.t1, out.t2, one, two);
  fence.complete();
}

void DFLambdaActions::transpose(bool independent, std::span<const double> one,
                                std::span<const double> two, std::vector<double>& out_one,
                                std::vector<double>& out_two) {
  auto& s = *impl_;
  Fence fence{s.storage.stream};
  if (one.size() != s.n1 || two.size() != s.n2)
    throw std::invalid_argument("DF Lambda transpose seed shape mismatch");
  s.clear_error();
  s.upload(s.state.bar_singles_residual, one.data(), s.n1);
  s.upload(s.state.bar_doubles_residual, two.data(), s.n2);
  const auto core = independent ? generated_response::run_independent_transpose_cuda(s.state)
                                : generated_response::run_transpose_cuda(s.state);
  s.record(independent ? generated_response::independent_transpose_contraction_terms
                       : generated_response::transpose_contraction_terms,
           independent ? generated_response::independent_transpose_operations
                       : generated_response::transpose_operations);
  // Consume borrowed core output before its scratch is reused for a Q action.
  cuda_check(
      cudaMemcpyAsync(s.sum1, core.t1, bytes(s.n1), cudaMemcpyDeviceToDevice, s.storage.stream));
  cuda_check(
      cudaMemcpyAsync(s.sum2, core.t2, bytes(s.n2), cudaMemcpyDeviceToDevice, s.storage.stream));
  for (std::size_t Q = 0; Q < s.q; ++Q) {
    s.select(Q);
    const auto out = generated_virtual::run_amplitude_vjp_accumulate_cuda(s.auxiliary);
    s.add(out.t1, out.t2);
    s.record(generated_response::virtual_amplitude_vjp_contraction_terms,
             generated_virtual::amplitude_vjp_cuda_operation_count);
  }
  s.output(s.sum1, s.sum2, out_one, out_two);
  fence.complete();
}

void DFLambdaActions::seeds(std::span<const double> one, std::span<const double> two) {
  auto& s = *impl_;
  const double energy = 1;
  Fence fence{s.storage.stream};
  if (!s.parameters_admitted || one.size() != s.n1 || two.size() != s.n2)
    throw std::invalid_argument("DF parameter response not admitted or seed shape mismatch");
  s.clear_error();
  s.upload(s.state.bar_correlation_energy, &energy, 1);
  s.upload(s.state.bar_singles_residual, one.data(), s.n1);
  s.upload(s.state.bar_doubles_residual, two.data(), s.n2);
  s.finish();
  fence.complete();
}

std::vector<double> DFLambdaActions::parameter(std::string_view name, std::size_t count) {
  auto& s = *impl_;
  if (!s.parameters_admitted) throw std::logic_error("DF parameter storage not admitted");
  const auto found = std::find_if(parameters.begin(), parameters.end(),
                                  [&](const auto& p) { return p.name == name; });
  if (found == parameters.end())
    throw std::invalid_argument(
        "DF parameter response cannot materialize a virtual integral block");
  const auto oo = checked_mul(s.state.o, s.state.o);
  const auto expected = name == "foo"    ? oo
                        : name == "fov"  ? s.n1
                        : name == "fvv"  ? s.vv
                        : name == "ovoo" ? checked_mul(s.n1, oo)
                        : name == "oooo" ? checked_mul(oo, oo)
                                         : s.n2;
  if (count != expected) throw std::invalid_argument("DF parameter response output shape mismatch");
  std::vector<double> values(count);
  Fence fence{s.storage.stream};
  s.clear_error();
  const auto out = found->run(s.state);
  s.record(found->work, found->operations);
  s.download(values.data(), out.values, count);
  s.finish();
  fence.complete();
  return values;
}

std::pair<std::vector<double>, std::vector<double>> DFLambdaActions::virtual_factors() {
  auto& s = *impl_;
  if (!s.parameters_admitted) throw std::logic_error("DF factor publication not admitted");
  std::pair<std::vector<double>, std::vector<double>> result{
      std::vector<double>(checked_mul(s.q, s.n1)), std::vector<double>(checked_mul(s.q, s.vv))};
  Fence fence{s.storage.stream};
  s.clear_error();
  for (std::size_t Q = 0; Q < s.q; ++Q) {
    s.select(Q);
    const auto out = generated_virtual::run_factor_vjp_accumulate_cuda(s.auxiliary);
    s.record(generated_response::virtual_factor_vjp_contraction_terms,
             generated_virtual::factor_vjp_cuda_operation_count);
    // The stream consumes each borrowed output before the next Q reuses it.
    s.download(result.first.data() + Q * s.n1, out.bov, s.n1);
    s.download(result.second.data() + Q * s.vv, out.bvv, s.vv);
  }
  s.finish();
  fence.complete();
  return result;
}
}  // namespace generativeqc::cc::detail
