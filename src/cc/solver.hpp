#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace generativeqc::cc {

struct SolverOptions {
  unsigned max_iterations{100};
  unsigned diis_size{6};
  double energy_tolerance{1e-11};
  double residual_tolerance{1e-9};
  double denominator_threshold{1e-10};
  double damping{};
  double level_shift{};
  std::size_t max_bytes{256ULL << 20};
  // Internal DF scheduling control; dense/conventional paths are unaffected.
  // Admission retains the bounded original schedule when work or storage wins.
  bool df_auxiliary_reduction{true};
};

struct Problem {
  std::size_t nocc{}, nvir{};
  std::vector<double> foo, fov, fvv;
  std::vector<double> ovov, ovvo, oovv, ovvv, ovoo, oooo, vvvv;
  std::vector<double> d1, d2;
  std::vector<double> initial_t1, initial_t2;
  double reference_energy{};
  double minimum_absolute_denominator{};
  std::size_t reference_retained_bytes{};
  std::size_t provider_peak_bytes{};
  std::size_t provider_host_bytes{};
  // Correlation-only DF virtual representation. For naux > 0 these row-major
  // Q-major factors replace ovvv/vvvv, which must be empty. The retained small
  // blocks must come from the same fitted Hamiltonian; Fock/reference energy
  // retain the explicitly selected reference contract (conventional RHF here).
  std::size_t naux{};
  std::vector<double> df_bov, df_bvv;
  // Optional for supplied energy/Lambda inputs; required by the physical
  // retained-block pullback. Native molecular sources always publish Boo.
  std::vector<double> df_boo;
};

enum class SolveStatus { Converged, NotConverged, NumericalFailure };

struct SolverDiagnostic {
  unsigned iterations{};
  unsigned diis_restarts{};
  double energy_change{};
  double r1_max{}, r2_max{};
  double replay_r1_max{}, replay_r2_max{};
  std::size_t numeric_capacity_bytes{};
  std::size_t owned_device_bytes{};
  std::size_t setup_h2d_bytes{};
  std::size_t scalar_d2h_bytes{};
  std::size_t amplitude_d2h_bytes{};
  std::size_t synchronizations{};
  std::size_t iteration_graph_calls{};
  std::size_t replay_graph_calls{};
  std::size_t update_calls{};
  std::size_t generated_error_checks{};
  std::size_t diis_gram_calls{};
  std::size_t diis_coefficient_calls{};
  std::size_t diis_combine_calls{};
  // Complete auxiliary work, including trial evaluations and independent replay.
  std::size_t df_auxiliary_slices{};
  std::size_t df_virtual_operations{};
  std::size_t df_accumulation_calls{};
  std::size_t df_hoisted_evaluations{};
  std::size_t df_preparation_calls{};
  std::size_t df_contraction_terms{};
  double tensor_seconds{};
  double iteration_seconds{};
  double replay_seconds{};
  double update_seconds{};
  double diis_seconds{};
};

struct SolverResult {
  SolveStatus status{SolveStatus::NotConverged};
  std::string reason;
  double correlation_energy{};
  double total_energy{};
  std::vector<double> t1, t2;
  SolverDiagnostic diagnostic;
  [[nodiscard]] bool converged() const noexcept { return status == SolveStatus::Converged; }
};

// Shared admission defaults to the conventional representation. Only an owner
// that implements the full DF Q sum may opt in; Lambda/triples/force consumers
// must reject DF until their own factorized paths are implemented.
void validate_problem(const Problem& problem, bool allow_df_virtual = false);
void validate_options(const SolverOptions& options);
std::size_t problem_host_bytes(const Problem& problem);
SolverResult solve_cpu(const Problem& problem, const SolverOptions& options);
SolverResult solve_cuda(const Problem& problem, const SolverOptions& options, int device);

}  // namespace generativeqc::cc
