#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda/df_runtime.hpp"
#include "scf/cuda/df_scf_factor.hpp"
#include "scf/cuda/df_scf_kernels.hpp"
#include "scf/cuda/df_scf_library.hpp"

namespace vibeqc::scf {
using namespace cuda_df;

// Fixed-topology replay preserves provider calls, physical convergence tests,
// graph-capture fallback, per-item iteration limits and final density download.

vibeqc_status run_cuda_density_fitting_rhf_device_scf(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& hcore,
    const std::vector<double>& orthogonalizer, const std::vector<double>& initial_density,
    const std::vector<std::int32_t>& occupied, const std::vector<double>& nuclear_repulsion,
    unsigned max_iterations, double energy_tolerance, double density_tolerance,
    std::vector<double>& final_density, std::vector<CudaDensityFittingDeviceScfItem>& results,
    std::string& detail) {
  detail.clear();
  if (plan == nullptr || max_iterations == 0 || !(energy_tolerance > 0.0) ||
      !(density_tolerance > 0.0) || !std::isfinite(energy_tolerance) ||
      !std::isfinite(density_tolerance)) {
    detail = "CUDA DF device RHF SCF arguments are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t batch_size = plan->batch_size;
  const std::size_t matrix_elements = plan->matrix_elements;
  const std::size_t expected = batch_size * matrix_elements;
  if (hcore.size() != expected || orthogonalizer.size() != expected ||
      initial_density.size() != expected || occupied.size() != batch_size ||
      nuclear_repulsion.size() != batch_size || !finite_values(hcore) ||
      !finite_values(orthogonalizer) || !finite_values(initial_density) ||
      !finite_values(nuclear_repulsion)) {
    detail = "CUDA DF device RHF SCF buffers have invalid dimensions or values";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (std::int32_t value : occupied) {
    if (value < 0 || static_cast<std::size_t>(value) > plan->nbf) {
      detail = "CUDA DF device RHF occupation is invalid";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  }
  final_density.clear();
  results.assign(batch_size, {});
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  bool occupied_exchange = false;
  const auto policy_status = occupied_scf_policy(occupied_exchange, detail);
  if (policy_status != VIBEQC_STATUS_SUCCESS) return policy_status;
  PersistentScfState* state = static_cast<PersistentScfState*>(plan->persistent_scf_state);
  const bool compatible =
      state != nullptr && !state->unrestricted && state->device_id == plan->device_id &&
      state->batch_size == batch_size && state->nbf == plan->nbf && state->expected == expected &&
      state->occupied_exchange == occupied_exchange &&
      (!occupied_exchange || (state->factor_alpha_ranks == occupied &&
                              state->factor_beta_ranks == std::vector<std::int32_t>{}));
  if (!compatible) {
    destroy_persistent_scf_state(plan->persistent_scf_state);
    state = new (std::nothrow) PersistentScfState{};
    if (state == nullptr) {
      detail = "host allocation for persistent CUDA DF RHF state failed";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
    state->device_id = plan->device_id;
    state->batch_size = batch_size;
    state->nbf = plan->nbf;
    state->expected = expected;
    state->occupied_exchange = occupied_exchange;
    state->graph.device_id = plan->device_id;
    state->graph.stream = plan->stream;
    auto allocate = [&](void** pointer, std::size_t bytes,
                        const char* description) -> vibeqc_status {
      const vibeqc_status allocation = allocate_device(pointer, bytes, description, detail);
      if (allocation == VIBEQC_STATUS_SUCCESS) {
        try {
          state->allocations.push_back(*pointer);
        } catch (const std::bad_alloc&) {
          (void)runtime::resource_cuda_free(*pointer);
          *pointer = nullptr;
          detail = "host allocation failed for CUDA DF SCF state handles";
          return VIBEQC_STATUS_OUT_OF_MEMORY;
        }
      }
      return allocation;
    };
    vibeqc_status status = allocate(reinterpret_cast<void**>(&state->d_hcore),
                                    expected * sizeof(double), "allocate CUDA DF SCF Hcore");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_orthogonalizer),
                        expected * sizeof(double), "allocate CUDA DF SCF orthogonalizer");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_density), expected * sizeof(double),
                        "allocate CUDA DF SCF density");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_next_density), expected * sizeof(double),
                        "allocate CUDA DF SCF next density");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_fock), expected * sizeof(double),
                        "allocate CUDA DF SCF Fock");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_temporary), expected * sizeof(double),
                        "allocate CUDA DF SCF eigensolver temporary");
    if (status == VIBEQC_STATUS_SUCCESS)
      status =
          allocate(reinterpret_cast<void**>(&state->d_eigenvalues),
                   batch_size * plan->nbf * sizeof(double), "allocate CUDA DF SCF eigenvalues");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_occupied),
                        batch_size * sizeof(std::int32_t), "allocate CUDA DF SCF occupations");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_nuclear), batch_size * sizeof(double),
                        "allocate CUDA DF SCF nuclear energies");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_energy), batch_size * sizeof(double),
                        "allocate CUDA DF SCF energies");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_previous_energy),
                        batch_size * sizeof(double), "allocate CUDA DF SCF previous energies");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_energy_change),
                        batch_size * sizeof(double), "allocate CUDA DF SCF energy changes");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_density_rms),
                        batch_size * sizeof(double), "allocate CUDA DF SCF density RMS");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_active),
                        batch_size * sizeof(std::uint8_t), "allocate CUDA DF SCF active mask");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_converged),
                        batch_size * sizeof(std::uint8_t), "allocate CUDA DF SCF converged mask");
    if (status == VIBEQC_STATUS_SUCCESS)
      status =
          allocate(reinterpret_cast<void**>(&state->d_iterations),
                   batch_size * sizeof(std::uint32_t), "allocate CUDA DF SCF iteration counters");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_info), batch_size * sizeof(int),
                        "allocate CUDA DF SCF solver status");
    if (status != VIBEQC_STATUS_SUCCESS) {
      delete state;
      return status;
    }
    status = setup_device_solver(*plan, plan->nbf, batch_size, state->d_temporary,
                                 state->d_eigenvalues, state->solver, detail);
    if (status != VIBEQC_STATUS_SUCCESS) {
      delete state;
      return status;
    }
    if (occupied_exchange) {
      status = allocate_scf_factors(*plan, *state, occupied, std::vector<std::int32_t>{}, detail);
      if (status != VIBEQC_STATUS_SUCCESS) {
        delete state;
        return status;
      }
    }
    plan->persistent_scf_state = state;
  }
  double* d_hcore = state->d_hcore;
  double* d_orthogonalizer = state->d_orthogonalizer;
  double* d_density = state->d_density;
  double* d_next_density = state->d_next_density;
  double* d_fock = state->d_fock;
  double* d_temporary = state->d_temporary;
  double* d_eigenvalues = state->d_eigenvalues;
  std::int32_t* d_occupied = state->d_occupied;
  double* d_nuclear = state->d_nuclear;
  double* d_energy = state->d_energy;
  double* d_previous_energy = state->d_previous_energy;
  double* d_energy_change = state->d_energy_change;
  double* d_density_rms = state->d_density_rms;
  std::uint8_t* d_active = state->d_active;
  std::uint8_t* d_converged = state->d_converged;
  std::uint32_t* d_iterations = state->d_iterations;
  int* d_info = state->d_info;
  vibeqc_status status = VIBEQC_STATUS_SUCCESS;
  const std::size_t matrix_bytes = expected * sizeof(double);
  cuda_error =
      cudaMemcpyAsync(d_hcore, hcore.data(), matrix_bytes, cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess)
    cuda_error = cudaMemcpyAsync(d_orthogonalizer, orthogonalizer.data(), matrix_bytes,
                                 cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess)
    cuda_error = cudaMemcpyAsync(d_density, initial_density.data(), matrix_bytes,
                                 cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess)
    cuda_error = cudaMemcpyAsync(d_occupied, occupied.data(), batch_size * sizeof(std::int32_t),
                                 cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess)
    cuda_error = cudaMemcpyAsync(d_nuclear, nuclear_repulsion.data(), batch_size * sizeof(double),
                                 cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess)
    cuda_error = cudaMemsetAsync(d_converged, 0, batch_size * sizeof(std::uint8_t), plan->stream);
  if (cuda_error == cudaSuccess)
    cuda_error = cudaMemsetAsync(d_iterations, 0, batch_size * sizeof(std::uint32_t), plan->stream);
  if (cuda_error == cudaSuccess)
    cuda_error = cudaMemsetAsync(d_active, 1, batch_size * sizeof(std::uint8_t), plan->stream);
  if (cuda_error != cudaSuccess)
    return cuda_failure(cuda_error, "upload CUDA DF device RHF SCF state", detail);
  std::vector<double> initial_previous(batch_size, std::numeric_limits<double>::infinity());
  cuda_error = cudaMemcpyAsync(d_previous_energy, initial_previous.data(),
                               batch_size * sizeof(double), cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error != cudaSuccess)
    return cuda_failure(cuda_error, "initialize CUDA DF device RHF energy state", detail);
  std::vector<double> host_energy(batch_size), host_energy_change(batch_size),
      host_density_rms(batch_size);
  std::vector<std::uint8_t> host_converged(batch_size);
  std::vector<std::uint32_t> host_iterations(batch_size);
  std::vector<int> host_info(batch_size);
  const bool options_changed = state->max_iterations != max_iterations ||
                               state->energy_tolerance != energy_tolerance ||
                               state->density_tolerance != density_tolerance;
  if (options_changed) {
    state->graph.reset();
    state->graph.device_id = plan->device_id;
    state->graph.stream = plan->stream;
    state->graph_replay = false;
  }
  state->max_iterations = max_iterations;
  state->energy_tolerance = energy_tolerance;
  state->density_tolerance = density_tolerance;
  DeviceIterationGraph& iteration_graph = state->graph;
  const auto launch_iteration = [&](bool factors_ready, bool tail) -> vibeqc_status {
    vibeqc_status iteration_status =
        build_scf_occupied_jk(*plan, *state, d_density, nullptr, factors_ready, detail);
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    launch_assemble_rhf_fock_kernel(blocks_for(expected), kThreads, 0, plan->stream, expected,
                                    d_hcore, plan->coulomb, plan->alpha_exchange, d_fock);
    cudaError_t iteration_error = cudaPeekAtLastError();
    if (iteration_error != cudaSuccess) {
      return cuda_failure(iteration_error, "assemble CUDA DF device RHF Fock", detail);
    }
    launch_compute_device_energy_kernel(static_cast<unsigned>(batch_size), 32, 0, plan->stream,
                                        batch_size, plan->nbf, d_density, d_hcore, d_fock,
                                        d_nuclear, d_energy);
    iteration_status = scf_gemm(*plan, false, batch_size, plan->nbf, d_fock, d_orthogonalizer,
                                d_temporary, detail);
    if (iteration_status == VIBEQC_STATUS_SUCCESS) {
      iteration_status = scf_gemm(*plan, true, batch_size, plan->nbf, d_orthogonalizer, d_temporary,
                                  d_fock, detail);
    }
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    iteration_status = solve_device_batch(state->solver, plan->nbf, batch_size, d_fock,
                                          d_eigenvalues, d_info, detail);
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    iteration_status = scf_gemm(*plan, false, batch_size, plan->nbf, d_orthogonalizer, d_fock,
                                d_temporary, detail);
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    launch_build_device_density_kernel(blocks_for(expected), kThreads, 0, plan->stream, batch_size,
                                       plan->nbf, d_occupied, d_temporary, 2.0, d_next_density);
    if (occupied_exchange) store_scf_factor(*plan, *state, d_temporary, false);
    launch_update_device_convergence_kernel(
        static_cast<unsigned>(batch_size), 32, 0, plan->stream, batch_size, plan->nbf,
        energy_tolerance, density_tolerance, d_energy, d_previous_energy, d_next_density, d_density,
        d_active, d_converged, d_iterations, d_energy_change, d_density_rms);
    if (tail)
      launch_tail_cuda_density_fitting_scf_graph_kernel(1, 1, 0, plan->stream,
                                                        static_cast<std::int32_t>(batch_size),
                                                        max_iterations, d_active, d_iterations);
    iteration_error = cudaPeekAtLastError();
    return iteration_error == cudaSuccess
               ? VIBEQC_STATUS_SUCCESS
               : cuda_failure(iteration_error, "advance CUDA DF device RHF SCF", detail);
  };
  // Imported/warm D has no trustworthy C. Execute one dense iteration on
  // every invocation, then capture/replay only the canonical factor loop.
  // The seed has no tail launch and is downloaded once even at max_iterations=1.
  if (occupied_exchange) {
    status = reset_scf_factors(*plan, *state, detail);
    if (status == VIBEQC_STATUS_SUCCESS) status = launch_iteration(false, false);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
  }
  bool graph_replay = state->graph_replay;
  // Host-backed streamed tiles require pageable copies and fences, while a
  // source-backed plan generates every tile on-device and remains capture-safe.
  if (!graph_replay && (!plan->streamed || plan->integral_source != nullptr)) {
    // A previous capture may have produced a graph but failed during
    // instantiation/upload. Reset both handles before replacing them.
    iteration_graph.reset();
    cuda_error = cudaStreamSynchronize(plan->stream);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamBeginCapture(plan->stream, cudaStreamCaptureModeThreadLocal);
    }
    if (cuda_error == cudaSuccess) {
      status = launch_iteration(occupied_exchange, true);
      // Stream capture records the iteration graph; CUDA does not execute the
      // enclosed kernels until cudaGraphLaunch below. Consequently this setup
      // call consumes zero SCF iterations and the normal max_iterations loop
      // remains authoritative for convergence semantics.
      cudaGraph_t captured = nullptr;
      const cudaError_t end_error = cudaStreamEndCapture(plan->stream, &captured);
      if (status == VIBEQC_STATUS_SUCCESS && end_error == cudaSuccess && captured != nullptr) {
        iteration_graph.graph = captured;
        cuda_error = cudaGraphInstantiate(&iteration_graph.executable, iteration_graph.graph, 0U);
        if (cuda_error == cudaSuccess) {
          cuda_error = cudaGraphUpload(iteration_graph.executable, plan->stream);
        }
        if (cuda_error == cudaSuccess) {
          cuda_error = cudaStreamSynchronize(plan->stream);
          graph_replay = cuda_error == cudaSuccess;
          state->graph_replay = graph_replay;
        }
        if (!graph_replay) iteration_graph.reset();
      } else {
        if (captured != nullptr) {
          (void)cudaGraphDestroy(captured);
          iteration_graph.graph = nullptr;
        }
        iteration_graph.reset();
        cuda_error = end_error;
      }
    }
    // Graph capture is an optimization.  A provider/capture limitation falls
    // through to the same direct launch sequence without changing semantics.
    if (!graph_replay) {
      cuda_error = cudaSuccess;
    }
  }
  bool all_converged = false;
  const auto all_terminal = [&]() {
    for (std::size_t system = 0; system < batch_size; ++system) {
      if (host_converged[system] == 0 && host_iterations[system] < max_iterations) {
        return false;
      }
    }
    return true;
  };
  for (unsigned iteration = 0; iteration < max_iterations && !all_converged; ++iteration) {
    if (occupied_exchange && iteration == 0) {
      // The already-executed dense seed needs its convergence/limit readback.
    } else if (graph_replay) {
      cuda_error = cudaGraphLaunch(iteration_graph.executable, plan->stream);
      if (cuda_error != cudaSuccess) {
        return cuda_failure(cuda_error, "replay CUDA DF RHF SCF Graph", detail);
      }
    } else {
      // Host-driven fallback has no enclosing graph to tail-launch.
      status = launch_iteration(occupied_exchange, false);
      if (status != VIBEQC_STATUS_SUCCESS) return status;
    }
    cuda_error = cudaMemcpyAsync(host_energy.data(), d_energy, batch_size * sizeof(double),
                                 cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess)
      cuda_error =
          cudaMemcpyAsync(host_energy_change.data(), d_energy_change, batch_size * sizeof(double),
                          cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess)
      cuda_error =
          cudaMemcpyAsync(host_density_rms.data(), d_density_rms, batch_size * sizeof(double),
                          cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess)
      cuda_error =
          cudaMemcpyAsync(host_converged.data(), d_converged, batch_size * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess)
      cuda_error =
          cudaMemcpyAsync(host_iterations.data(), d_iterations, batch_size * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess)
      cuda_error = cudaMemcpyAsync(host_info.data(), d_info, batch_size * sizeof(int),
                                   cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(plan->stream);
    if (cuda_error != cudaSuccess)
      return cuda_failure(cuda_error, "read CUDA DF device RHF SCF records", detail);
    if (std::any_of(host_info.begin(), host_info.end(), [](int value) { return value != 0; })) {
      detail = "CUDA DF device RHF eigensolver did not converge";
      return VIBEQC_STATUS_CUDA_ERROR;
    }
    // A captured graph may tail-launch the iteration body repeatedly until
    // convergence or the device-side iteration limit.  Treat the limit as a
    // terminal host condition too; otherwise the outer replay loop would
    // launch an additional graph after the captured body already consumed all
    // permitted iterations.
    all_converged = all_terminal();
  }
  final_density.resize(expected);
  cuda_error = cudaMemcpyAsync(final_density.data(), d_density, matrix_bytes,
                               cudaMemcpyDeviceToHost, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(plan->stream);
  if (cuda_error != cudaSuccess)
    return cuda_failure(cuda_error, "read CUDA DF device RHF density", detail);
  if (!finite_values(final_density)) {
    detail = "CUDA DF device RHF SCF produced non-finite density";
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
  if (occupied_exchange) {
    status = verify_scf_factors(*plan, *state, host_iterations, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
  }
  for (std::size_t system = 0; system < batch_size; ++system) {
    auto& result = results[system];
    result.status = VIBEQC_STATUS_SUCCESS;
    result.converged = host_converged[system] != 0;
    result.iterations = host_iterations[system];
    result.energy = host_energy[system];
    result.energy_change = host_energy_change[system];
    result.density_rms = host_density_rms[system];
    if (!result.converged) result.status = VIBEQC_STATUS_SCF_NOT_CONVERGED;
  }
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::scf
