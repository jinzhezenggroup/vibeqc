#include "scf/cuda/df_final_validation.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "runtime/cuda_component_trace.hpp"
#include "runtime/host_component_trace.hpp"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda/df_scf_state.hpp"
#include "scf/cuda/final_validation_kernels.hpp"

namespace vibeqc::scf::cuda_df {
namespace {
using reference::Matrix;
namespace trace = runtime::cuda_trace;
struct Workspace {
  int device{-1};
  double* storage{};
  ValidationPartial* partial{};
  std::size_t bytes{};
  unsigned blocks{};
  // Borrowed J/K are consumed synchronously by one tagged selection request.
  std::optional<solver::FinalStateIdentity> physical_identity;
  const Matrix* physical_hcore{};
  ~Workspace() {
    if (device >= 0) (void)cudaSetDevice(device);
    (void)runtime::resource_cuda_free(storage);
    (void)runtime::resource_cuda_free(partial);
  }
};
void check(cudaError_t status, const char* operation) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
void check(vibeqc_status status, const std::string& detail) {
  if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}
Workspace& prepare(CudaDensityFittingJkPlan& plan) {
  check(cudaSetDevice(plan.device_id), "select final-validation device");
  cudaStreamCaptureStatus capture{};
  check(cudaStreamIsCapturing(plan.stream, &capture), "inspect final-validation stream");
  if (capture != cudaStreamCaptureStatusNone)
    throw std::runtime_error("final validation requires an ordinary noncapturing stream");
  if (!plan.final_validation) {
    auto state = std::make_unique<Workspace>();
    state->device = plan.device_id;
    const auto allowance = df_final_validation_device_reservation(plan.nbf);
    const auto storage_bytes = (9 * plan.nbf * plan.nbf + plan.nbf) * sizeof(double);
    static_assert(sizeof(ValidationPartial) <= 128);
    state->blocks = validation_block_count(plan.nbf);
    const auto partial_bytes = (3 * state->blocks + 1) * sizeof(ValidationPartial);
    state->bytes = storage_bytes + partial_bytes;
    if (state->bytes > allowance)
      throw std::runtime_error("final validation exceeds its admitted workspace");
    std::string detail;
    check(allocate_device(reinterpret_cast<void**>(&state->storage), storage_bytes,
                          "allocate final-validation matrices", detail),
          detail);
    check(allocate_device(reinterpret_cast<void**>(&state->partial), partial_bytes,
                          "allocate final-validation reductions", detail),
          detail);
    plan.final_validation = state.release();
  }
  return *static_cast<Workspace*>(plan.final_validation);
}
/** All pageable input/output buffers outlive this guard. Drain on every
 * exceptional exit too, before publishing diagnostics or freeing staging. */
struct Drain {
  cudaStream_t stream;
  bool pending{true};
  ~Drain() {
    if (pending) {
      (void)cudaStreamSynchronize(stream);
      trace::trace_counter("explicit_synchronizations", 1);
    }
  }
  void finish() {
    check(cudaStreamSynchronize(stream), "finish final validation");
    trace::trace_counter("explicit_synchronizations", 1);
    pending = false;
  }
};
void upload(CudaDensityFittingJkPlan& plan, double* out, const Matrix& host) {
  check(cudaMemcpyAsync(out, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice,
                        plan.stream),
        "upload final-validation input");
  trace::trace_counter("host_to_device_bytes", host.size() * sizeof(double));
}
Matrix packed(const Matrix& a, std::size_t n) {
  if (a.size() != n * n) throw std::runtime_error("invalid final-validation matrix dimensions");
  Matrix out(a.size());
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) out[j * n + i] = a[i * n + j];
  return out;
}
void gemm(CudaDensityFittingJkPlan& plan, const double* a, const double* b, double* out,
          bool ta = false, bool tb = false, int rank = -1) {
  const int n = static_cast<int>(plan.nbf);
  const double one = 1, zero = 0;
  if (rank == 0) {
    check(cudaMemsetAsync(out, 0, plan.nbf * plan.nbf * sizeof(double), plan.stream),
          "zero empty occupied product");
    return;
  }
  const auto status =
      cublasDgemm(plan.blas, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, n, n,
                  rank < 0 ? n : rank, &one, a, n, b, n, &zero, out, n);
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("cuBLAS final-validation product failed: " + std::to_string(status));
}
ValidationInputs inputs(CudaDensityFittingJkPlan& plan, Workspace& w,
                        const solver::FinalFrameCandidate* frame, std::size_t spin, double weight) {
  const auto n = plan.nbf, count = n * n;
  ValidationInputs in;
  in.n = n;
  in.weight = weight;
  in.s = w.storage;
  in.h = w.storage + count;
  in.f = w.storage + 2 * count;
  in.d = w.storage + 3 * count;
  in.c = w.storage + 4 * count;
  in.values = w.storage + 9 * count;
  if (!frame) return in;
  in.occupied = frame->identity.occupied.at(spin);
  const auto& c = frame->spins.at(spin);
  if (!c.vectors.empty() || !c.values.empty()) return in;
  // Empty matrices are an explicit retained-frame request, never evidence
  // based on dimensions. Recheck every token field against the owner.
  CudaDfFinalStateToken current;
  std::string detail;
  const auto item = frame->identity.factor.reference - 1;
  check(cuda_density_fitting_final_state_token(&plan, item, current, detail), detail);
  if (current.identity != frame->identity || !frame->physical_origin ||
      frame->fock_density_generation != current.identity.factor.density_generation - 1)
    throw std::runtime_error("stale retained final-validation frame");
  const auto& state = *static_cast<const PersistentScfState*>(plan.persistent_scf_state);
  in.c = (spin ? state.d_final_beta_coefficients : state.d_final_alpha_coefficients) + item * count;
  in.values = (spin ? state.d_final_beta_values : state.d_final_alpha_values) + item * n;
  in.expected_density = in.d;
  in.d = (spin                 ? state.d_beta_density
          : state.unrestricted ? state.d_alpha_density
                               : state.d_density) +
         item * count;
  in.generation = (spin ? state.d_final_beta_generation : state.d_final_alpha_generation) + item;
  in.info = (spin ? state.d_final_beta_info : state.d_final_alpha_info) + item;
  in.expected_generation = current.identity.factor.density_generation;
  return in;
}
void physical_fock(CudaDensityFittingJkPlan& plan, Workspace& w,
                   const solver::FinalStateIdentity& id, std::size_t spin) {
  if (!w.physical_identity || *w.physical_identity != id ||
      id.factor.basis != plan.factor_basis_identity ||
      id.solve_epoch != plan.final_state_solve_epoch || !id.factor.reference ||
      id.factor.reference > plan.batch_size)
    throw std::runtime_error("physical Fock belongs to another validation request");
  const auto count = plan.matrix_elements;
  // The checked reference fits size_t; plan admission bounds the full batch
  // allocation, so this item's matrix offset fits too.
  const auto item = static_cast<std::size_t>(id.factor.reference - 1);
  const auto offset = item * count;
  launch_validation_fock(plan.stream, plan.nbf, w.storage + count, plan.coulomb + offset,
                         (spin ? plan.beta_exchange : plan.alpha_exchange) + offset,
                         id.model.spec.spin == FockSpin::Restricted ? .5 : 1,
                         w.storage + 2 * count);
}
std::vector<Matrix> materialize(CudaDensityFittingJkPlan& plan,
                                const solver::PhysicalFockFrame& fock) {
  auto& w = prepare(plan);
  if (!w.physical_hcore) throw std::runtime_error("missing physical Fock input");
  const auto n = plan.nbf, count = n * n;
  const auto h = packed(*w.physical_hcore, n);
  std::vector<Matrix> result(fock.spins.size(), Matrix(count));
  Drain drain{plan.stream};
  upload(plan, w.storage + count, h);
  for (std::size_t spin = 0; spin < result.size(); ++spin) {
    physical_fock(plan, w, fock.identity, spin);
    check(cudaMemcpyAsync(result[spin].data(), w.storage + 2 * count, count * sizeof(double),
                          cudaMemcpyDeviceToHost, plan.stream),
          "materialize physical Fock");
    trace::trace_counter("device_to_host_bytes", count * sizeof(double));
  }
  drain.finish();
  for (auto& f : result) f = packed(f, n);
  return result;
}
void eigen_products(CudaDensityFittingJkPlan& plan, Workspace& w, ValidationInputs in,
                    bool canonical) {
  const auto count = plan.nbf * plan.nbf;
  auto* a = w.storage + 5 * count;
  auto* b = a + count;
  auto* t = b + count;
  auto* u = t + count;
  gemm(plan, in.f, in.c, a);
  gemm(plan, in.s, in.c, b);
  gemm(plan, in.c, b, t, true);
  if (canonical) gemm(plan, in.c, a, u, true);
  launch_validation_eigen(plan.stream, in, a, b, t, canonical ? u : nullptr, w.partial);
}
bool eigen_diagnostic(const ValidationPartial& raw, solver::EigenFrameDiagnostic& out,
                      std::string& detail) {
  // Retained-owner corruption is a provider failure, as in detached snapshot
  // reads. It must not masquerade as a numerical candidate needing correction.
  if (raw.invalid & validation_input_failure)
    throw std::runtime_error(
        "CUDA final-state provider returned nonphysical, stale or mismatched data");
  const double scale = raw.norm_f * raw.norm_c + raw.norm_rhs;
  out.maximum_eigen_residual = raw.eigen;
  out.maximum_metric_error = raw.metric;
  out.scaled_eigen_residual = scale == 0 ? raw.norm_residual : raw.norm_residual / scale;
  if (raw.invalid || !std::isfinite(scale) || !std::isfinite(raw.norm_residual)) {
    detail = "nonfinite validation products, invalid eigen order or stale device frame";
    return false;
  }
  return true;
}
ValidationPartial download(CudaDensityFittingJkPlan& plan, Workspace& w, unsigned stages) {
  auto* device = w.partial + 3 * w.blocks;
  launch_validation_finish(plan.stream, w.partial, device, stages, w.blocks);
  check(cudaPeekAtLastError(), "launch final-validation reductions");
  ValidationPartial result{};
  Drain drain{plan.stream};
  check(cudaMemcpyAsync(&result, device, sizeof(result), cudaMemcpyDeviceToHost, plan.stream),
        "read compact final-validation diagnostics");
  trace::trace_counter("device_to_host_bytes", sizeof(result));
  drain.finish();
  return result;
}
bool products(CudaDensityFittingJkPlan& plan, const solver::FinalStateIdentity& current,
              const Matrix& overlap, const Matrix& hcore, double nuclear,
              const std::vector<Matrix>& density, const solver::PhysicalFockFrame& fock,
              const solver::FinalFrameCandidate& frame, const solver::FinalStateLimits& limits,
              solver::FinalStateDiagnostic& diagnostic, std::string& detail) {
  // Detached candidates are untrusted inputs: bad dimensions reject reuse and
  // permit the shared CPU/CUDA correction policy. Broken retained owners throw
  // from inputs(); malformed frames returned by a new solve fail its provider
  // check before projection in select_final_state().
  for (const auto& c : frame.spins) {
    if ((!c.vectors.empty() || !c.values.empty()) &&
        (c.vectors.size() != plan.nbf * plan.nbf || c.values.size() != plan.nbf)) {
      detail = "invalid final-validation orbital dimensions";
      return false;
    }
  }
  trace::TraceOperation trace(
      "final_state_validation", plan.stream,
      {1, plan.nbf, plan.naux, plan.integral_source != nullptr, plan.streamed});
  auto& w = prepare(plan);
  trace::trace_counter("workspace_bytes", w.bytes);
  const auto n = plan.nbf, count = n * n;
  const double weight = current.model.spec.spin == FockSpin::Restricted ? 2 : 1;
  // Pack even approximately symmetric inputs faithfully; no transpose-based
  // commutator shortcut assumes stronger symmetry than the shared contract.
  const auto s = packed(overlap, n), h = packed(hcore, n);
  Drain outer{plan.stream};
  upload(plan, w.storage, s);
  upload(plan, w.storage + count, h);
  diagnostic.energy = nuclear;
  diagnostic.eigenframes.resize(density.size());
  for (std::size_t spin = 0; spin < density.size(); ++spin) {
    auto in = inputs(plan, w, &frame, spin, weight);
    const bool device_fock = fock.spins[spin].empty();
    const auto f = device_fock ? Matrix{} : packed(fock.spins[spin], n);
    const auto d = packed(density[spin], n);
    const auto& orbitals = frame.spins[spin];
    const auto c = orbitals.vectors.empty() ? Matrix{} : packed(orbitals.vectors, n);
    if (!c.empty() && orbitals.values.size() != n) {
      detail = "invalid final-validation eigenvalue dimensions";
      return false;
    }
    Drain spin_drain{plan.stream};
    if (device_fock) {
      physical_fock(plan, w, current, spin);
      in.physical_fock = true;
    } else {
      upload(plan, w.storage + 2 * count, f);
    }
    upload(plan, w.storage + 3 * count, d);
    if (!c.empty()) {
      upload(plan, w.storage + 4 * count, c);
      upload(plan, w.storage + 9 * count, orbitals.values);
    }
    eigen_products(plan, w, in, limits.require_canonicality);
    auto* a = w.storage + 5 * count;
    auto* b = a + count;
    auto* t = b + count;
    auto* u = t + count;
    launch_validation_columns(plan.stream, n, in.occupied, in.c, nullptr, weight, b);
    gemm(plan, b, in.c, a, false, true, static_cast<int>(in.occupied));
    gemm(plan, in.d, in.s, b);
    gemm(plan, b, in.d, t);
    launch_validation_density(plan.stream, in, a, b, t, w.partial + w.blocks);
    gemm(plan, in.f, b, u);  // F(DS), reusing this request's checked DS.
    gemm(plan, in.s, in.d, a);
    gemm(plan, a, in.f, t);
    launch_validation_commutator(plan.stream, n, u, t, w.partial + 2 * w.blocks);
    const auto raw = download(plan, w, 3);
    spin_drain.pending = false;
    if (!eigen_diagnostic(raw, diagnostic.eigenframes[spin], detail)) return false;
    diagnostic.maximum_density_error = std::max(diagnostic.maximum_density_error, raw.density);
    diagnostic.maximum_canonical_error =
        std::max(diagnostic.maximum_canonical_error, raw.canonical);
    diagnostic.maximum_idempotency_error =
        std::max(diagnostic.maximum_idempotency_error, raw.idempotency);
    diagnostic.maximum_commutator = std::max(diagnostic.maximum_commutator, raw.commutator);
    diagnostic.density_rms = std::max(diagnostic.density_rms, raw.norm_density / n);
    diagnostic.maximum_trace_error =
        std::max(diagnostic.maximum_trace_error, std::abs(raw.electrons - weight * in.occupied));
    diagnostic.energy += raw.energy;
    if (!std::isfinite(raw.norm_density) || !std::isfinite(raw.electrons) ||
        !std::isfinite(raw.energy)) {
      detail = "nonfinite final-state reduction";
      return false;
    }
  }
  outer.pending = false;
  return true;
}
Matrix project(CudaDensityFittingJkPlan& plan, const reference::EigenResult& orbitals,
               std::size_t occupied, double weight, bool weighted,
               const solver::FinalFrameCandidate* frame = nullptr, std::size_t spin = 0) {
  const auto n = plan.nbf, count = n * n;
  if (occupied > n || (!orbitals.vectors.empty() && orbitals.values.size() != n))
    throw std::runtime_error("invalid final-state projection dimensions");
  auto& w = prepare(plan);
  auto in = inputs(plan, w, frame, spin, weight);
  const auto c = orbitals.vectors.empty() ? Matrix{} : packed(orbitals.vectors, n);
  Matrix result(count);
  Drain drain{plan.stream};
  if (!c.empty()) {
    upload(plan, w.storage + 4 * count, c);
    upload(plan, w.storage + 9 * count, orbitals.values);
  } else if (!frame) {
    throw std::runtime_error("missing projection coefficients");
  }
  auto* columns = w.storage + 5 * count;
  auto* output = columns + count;
  launch_validation_columns(plan.stream, n, occupied, in.c, weighted ? in.values : nullptr, weight,
                            columns);
  gemm(plan, columns, in.c, output, false, true, static_cast<int>(occupied));
  check(cudaPeekAtLastError(), "build final-state density");
  check(cudaMemcpyAsync(result.data(), output, count * sizeof(double), cudaMemcpyDeviceToHost,
                        plan.stream),
        "read final-state density for host consumer");
  trace::trace_counter("device_to_host_bytes", count * sizeof(double));
  drain.finish();
  return packed(result, n);
}
}  // namespace
void destroy_final_validation(void*& opaque) noexcept {
  delete static_cast<Workspace*>(opaque);
  opaque = nullptr;
}
}  // namespace vibeqc::scf::cuda_df

namespace vibeqc::scf {
solver::PhysicalFockFrame evaluate_cuda_density_fitting_final_fock(
    CudaDensityFittingJkPlan* plan, const solver::FinalStateIdentity& current,
    const std::vector<reference::Matrix>& density, const reference::Matrix& hcore) {
  using namespace cuda_df;
  if (!plan || current.factor.basis != plan->factor_basis_identity ||
      current.solve_epoch != plan->final_state_solve_epoch || !current.factor.reference ||
      current.factor.reference > plan->batch_size || density.size() != current.occupied.size() ||
      (density.size() != 1 && density.size() != 2) || hcore.size() != plan->matrix_elements)
    throw std::runtime_error("invalid physical device Fock request identity");
  const auto expected_model = resolve_fock_build(
      make_hf_fock_spec(density.size() == 1 ? FockSpin::Restricted : FockSpin::Unrestricted,
                        FockApproximation::DensityFitted),
      FockBackend::Cuda, 1e-12, plan->metric_relative_threshold);
  if (current.model != expected_model)
    throw std::runtime_error("physical device Fock model differs from the prepared HF provider");
  auto& w = prepare(*plan);
  w.physical_identity.reset();
  w.physical_hcore = nullptr;
  const auto item = static_cast<std::size_t>(current.factor.reference - 1);
  const auto count = plan->matrix_elements;
  for (const auto& d : density)
    if (d.size() != count || !finite_values(d))
      throw std::runtime_error("invalid physical device Fock density");
  trace::TraceOperation trace(
      "final_state_physical_fock", plan->stream,
      {1, plan->nbf, plan->naux, plan->integral_source != nullptr, plan->streamed});
  std::string detail;
  bool retained = false;
  if (density.size() == 1) {
    Matrix j, k;
    check(try_cuda_density_fitting_final_rhf_jk(plan, {1, current}, density[0], j, k, retained,
                                                detail, false),
          detail);
  }
  if (!retained) {
    runtime::host_trace::Region dense("final_state_dense_jk", plan->nbf);
    // Reuse the existing bounded item staging and device J/K operations;
    // zero neighbors just as the host item adapter does. No output matrix
    // crosses the host boundary merely to validate it.
    Drain drain{plan->stream};
    const auto bytes = plan->batch_size * count * sizeof(double);
    check(cudaMemsetAsync(plan->primary_density, 0, bytes, plan->stream), "clear item density");
    upload(*plan, plan->primary_density + item * count, density[0]);
    if (density.size() == 1) {
      check(execute_cuda_density_fitting_rhf_jk_device(plan, plan->primary_density, plan->coulomb,
                                                       plan->alpha_exchange, detail, {},
                                                       FockMatrixLayout::RowMajor),
            detail);
    } else {
      check(cudaMemsetAsync(plan->secondary_density, 0, bytes, plan->stream),
            "clear beta item density");
      upload(*plan, plan->secondary_density + item * count, density[1]);
      check(execute_cuda_density_fitting_uhf_jk_device(
                plan, plan->primary_density, plan->secondary_density, plan->coulomb,
                plan->alpha_exchange, plan->beta_exchange, detail, {}, FockMatrixLayout::RowMajor),
            detail);
    }
    drain.finish();
  }
  w.physical_identity = current;
  w.physical_hcore = &hcore;
  return {current, true, std::vector<Matrix>(density.size())};
}

solver::FinalStateOperations cuda_density_fitting_final_state_operations(
    CudaDensityFittingJkPlan* plan) {
  if (!plan) throw std::invalid_argument("missing CUDA final-state plan");
  solver::FinalStateOperations operations;
  operations.materialize_fock = [plan](const auto& fock) {
    return cuda_df::materialize(*plan, fock);
  };
  operations.products = [plan](const auto& id, const auto& s, const auto& h, double nuclear,
                               const auto& d, const auto& f, const auto& c, const auto& limits,
                               auto& diagnostic, auto& detail) {
    return cuda_df::products(*plan, id, s, h, nuclear, d, f, c, limits, diagnostic, detail);
  };
  operations.eigen = [plan](const auto& f, const auto& s, const auto& c, auto& diagnostic,
                            auto& detail) {
    return validate_cuda_density_fitting_eigen_frame(plan, f, &s, c.values, c.vectors, diagnostic,
                                                     detail);
  };
  operations.project = [plan](const auto& c, std::size_t occupied, double weight) {
    return cuda_df::project(*plan, c, occupied, weight, false);
  };
  operations.weighted = [plan](const auto& id, const auto& frame) {
    runtime::cuda_trace::TraceOperation trace(
        "final_state_weighted_density", plan->stream,
        {1, plan->nbf, plan->naux, plan->integral_source != nullptr, plan->streamed});
    std::vector<reference::Matrix> result;
    const double weight = id.model.spec.spin == FockSpin::Restricted ? 2 : 1;
    for (std::size_t spin = 0; spin < id.occupied.size(); ++spin)
      result.push_back(cuda_df::project(*plan, frame.spins[spin], id.occupied[spin], weight, true,
                                        &frame, spin));
    return result;
  };
  return operations;
}

bool validate_cuda_density_fitting_eigen_frame(CudaDensityFittingJkPlan* plan,
                                               const reference::Matrix& matrix,
                                               const reference::Matrix* overlap,
                                               const reference::Matrix& values,
                                               const reference::Matrix& coefficients,
                                               solver::EigenFrameDiagnostic& diagnostic,
                                               std::string& detail) {
  using namespace cuda_df;
  if (!plan || values.size() != plan->nbf || diagnostic.solver_info != 0) {
    detail = "invalid CUDA eigen validation request";
    return false;
  }
  trace::TraceOperation trace(
      "eigenframe_validation", plan->stream,
      {1, plan->nbf, plan->naux, plan->integral_source != nullptr, plan->streamed});
  auto& w = prepare(*plan);
  const auto n = plan->nbf, count = n * n;
  Matrix identity;
  if (!overlap) {
    identity.assign(count, 0);
    for (std::size_t i = 0; i < n; ++i) identity[i * n + i] = 1;
  }
  const auto f = packed(matrix, n), s = packed(overlap ? *overlap : identity, n),
             c = packed(coefficients, n);
  Drain drain{plan->stream};
  upload(*plan, w.storage, s);
  upload(*plan, w.storage + 2 * count, f);
  upload(*plan, w.storage + 4 * count, c);
  upload(*plan, w.storage + 9 * count, values);
  eigen_products(*plan, w, inputs(*plan, w, nullptr, 0, 0), false);
  const auto raw = download(*plan, w, 1);
  drain.pending = false;
  return eigen_diagnostic(raw, diagnostic, detail) &&
         solver::accept_eigen_frame(diagnostic, detail);
}
}  // namespace vibeqc::scf
