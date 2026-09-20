#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vibeqc::cc {

struct SolverOptions {
  unsigned max_iterations{100};
  unsigned diis_size{6};
  double energy_tolerance{1e-11};
  double residual_tolerance{1e-9};
  double denominator_threshold{1e-10};
  double damping{};
  double level_shift{};
  std::size_t max_bytes{256ULL << 20};
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
  double tensor_seconds{};
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

// Shared CPU/CUDA admission; rejects malformed data before any execution owner.
void validate_problem(const Problem& problem);
void validate_options(const SolverOptions& options);
std::size_t problem_host_bytes(const Problem& problem);
SolverResult solve_cpu(const Problem& problem, const SolverOptions& options);
SolverResult solve_cuda(const Problem& problem, const SolverOptions& options, int device);

}  // namespace vibeqc::cc
