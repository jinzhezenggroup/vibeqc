#include "scf/cuda/df_scf_warm.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "runtime/cuda_component_trace.hpp"
#include "scf/cuda/df_final_validation.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda/df_scf_kernels.hpp"

namespace vibeqc::scf::cuda_df {
namespace {
bool enabled() {
  const auto* control = std::getenv("VIBEQC_DF_WARM_REUSE");
  return !control || std::strcmp(control, "0") != 0;
}
}  // namespace

std::shared_ptr<const RhfWarmState> find_rhf_warm_state(const CudaDensityFittingJkPlan* plan,
                                                        const std::vector<double>& density,
                                                        const std::vector<double>& hcore,
                                                        const std::vector<double>& overlap,
                                                        const std::vector<double>& orthogonalizer,
                                                        std::size_t occupied, double nuclear) {
  const auto* state =
      plan ? static_cast<const PersistentScfState*>(plan->persistent_scf_state) : nullptr;
  if (!enabled() || !state || state->unrestricted || !state->occupied_exchange ||
      plan->batch_size != 1 || state->diis_history < 2)
    return {};
  for (const auto& entry : {state->warm_current, state->warm_frozen}) {
    if (entry && entry->basis_identity == plan->factor_basis_identity &&
        entry->occupied == occupied && entry->nuclear == nuclear && entry->density == density &&
        entry->hcore == hcore && entry->overlap == overlap &&
        entry->orthogonalizer == orthogonalizer)
      return entry;
  }
  return {};
}
}  // namespace vibeqc::scf::cuda_df

namespace vibeqc::scf {
using namespace cuda_df;

bool cuda_density_fitting_rhf_warm_matches(const CudaDensityFittingJkPlan* plan,
                                           const reference::Matrix& density,
                                           const reference::Matrix& hcore,
                                           const reference::Matrix& overlap,
                                           const reference::Matrix& orthogonalizer,
                                           std::size_t occupied, double nuclear) {
  return bool(
      find_rhf_warm_state(plan, density, hcore, overlap, orthogonalizer, occupied, nuclear));
}

void prepare_cuda_density_fitting_rhf_warm_state(CudaDensityFittingJkPlan* plan,
                                                 const CudaDfFinalStateToken& token,
                                                 const reference::Matrix& density,
                                                 const reference::Matrix& hcore,
                                                 const reference::Matrix& overlap,
                                                 const reference::Matrix& orthogonalizer,
                                                 std::size_t occupied, double nuclear) {
  auto* state = plan ? static_cast<PersistentScfState*>(plan->persistent_scf_state) : nullptr;
  if (!state) return;
  state->warm_pending.reset();
  state->warm_pending_epoch = 0;
  if (!enabled() || state->unrestricted || !state->occupied_exchange || plan->batch_size != 1 ||
      state->diis_history < 2 || !occupied || occupied != state->alpha_factor_rank ||
      !final_occupied_fock_matches(*plan, token) || density.size() != plan->matrix_elements ||
      hcore.size() != density.size() || overlap.size() != density.size() ||
      orthogonalizer.size() != density.size() || !finite_values(density) || !finite_values(hcore) ||
      !finite_values(overlap) || !finite_values(orthogonalizer) || !std::isfinite(nuclear))
    return;
  // Bound both retained records plus the temporary detached full C/D/spectrum
  // download. Larger systems keep the ordinary seed; no device allocation is
  // added and cache retention cannot turn a successful endpoint into host OOM.
  const long double peak =
      sizeof(double) * (10.0L * plan->matrix_elements + 2.0L * plan->nbf * occupied + plan->nbf);
  if (peak > 64.0L * 1024 * 1024) return;
  try {
    CudaDfFinalStateSnapshot snapshot;
    std::string detail;
    auto status = read_cuda_density_fitting_final_state(plan, token, snapshot, detail);
    if (status == VIBEQC_STATUS_INVALID_ARGUMENT || status == VIBEQC_STATUS_OUT_OF_MEMORY) return;
    if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    if (snapshot.density.size() != 1 || snapshot.density[0] != density) return;
    auto candidate = std::make_shared<RhfWarmState>();
    candidate->token = token;
    candidate->basis_identity = plan->factor_basis_identity;
    candidate->occupied = occupied;
    candidate->nuclear = nuclear;
    candidate->density = density;
    candidate->hcore = hcore;
    candidate->overlap = overlap;
    candidate->orthogonalizer = orthogonalizer;
    candidate->factor.resize(plan->nbf * occupied);
    const auto& c = snapshot.candidate.spins[0].vectors;
    for (std::size_t column = 0; column < occupied; ++column)
      for (std::size_t row = 0; row < plan->nbf; ++row)
        candidate->factor[column * plan->nbf + row] = c[row * plan->nbf + column];

    runtime::cuda_trace::TraceOperation trace(
        "warm_state_energy_baseline", plan->stream,
        {1, plan->nbf, plan->naux, plan->integral_source != nullptr, plan->streamed});
    // Final validation still owns physical J/K here. Reassemble with the SCF
    // kernel so its FMA and reduction order match the next warm comparison;
    // copying the differently reduced public final energy would not suffice.
    auto error = cudaMemcpyAsync(state->d_hcore, hcore.data(), hcore.size() * sizeof(double),
                                 cudaMemcpyHostToDevice, plan->stream);
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(state->d_previous_energy, &candidate->nuclear, sizeof(double),
                              cudaMemcpyHostToDevice, plan->stream);
    if (error == cudaSuccess) {
      launch_assemble_rhf_fock_kernel(blocks_for(plan->matrix_elements), kThreads, 0, plan->stream,
                                      plan->matrix_elements, state->d_hcore, plan->coulomb,
                                      plan->alpha_exchange, state->d_next_density);
      launch_compute_device_energy_kernel(1, 32, 0, plan->stream, 1, plan->nbf, state->d_density,
                                          state->d_hcore, state->d_next_density,
                                          state->d_previous_energy, state->d_energy);
      error = cudaGetLastError();
    }
    if (error == cudaSuccess)
      error = cudaMemcpyAsync(&candidate->energy, state->d_energy, sizeof(double),
                              cudaMemcpyDeviceToHost, plan->stream);
    // Drain even on launch/copy failure before releasing pageable destinations.
    const auto drained = cudaStreamSynchronize(plan->stream);
    if (error == cudaSuccess) error = drained;
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    if (!std::isfinite(candidate->energy)) return;
    runtime::cuda_trace::trace_counter("extra_jk_builds", 0);
    runtime::cuda_trace::trace_counter("energy_reductions", 1);
    runtime::cuda_trace::trace_counter(
        "retained_host_bytes", (4 * plan->matrix_elements + plan->nbf * occupied) * sizeof(double));
    state->warm_pending = std::move(candidate);
    state->warm_pending_epoch = plan->final_state_solve_epoch;
  } catch (const std::bad_alloc&) {
    state->warm_pending.reset();
    state->warm_pending_epoch = 0;
  }
}

void commit_cuda_density_fitting_rhf_warm_state(CudaDensityFittingJkPlan* plan,
                                                const CudaDfFinalStateToken& token) noexcept {
  auto* state = plan ? static_cast<PersistentScfState*>(plan->persistent_scf_state) : nullptr;
  if (!state || !state->warm_pending ||
      state->warm_pending_epoch != plan->final_state_solve_epoch ||
      token != state->warm_pending->token ||
      token.identity.solve_epoch != plan->final_state_solve_epoch ||
      token.identity.factor.basis != plan->factor_basis_identity)
    return;
  state->warm_current = std::move(state->warm_pending);
  state->warm_frozen = std::move(state->warm_replay_seed);
  state->warm_pending_epoch = 0;
}
}  // namespace vibeqc::scf
