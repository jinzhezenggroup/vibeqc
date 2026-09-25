#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "cc/solver.hpp"
#include "response/native_gmres.hpp"

namespace vibeqc::cc {

struct LambdaOptions {
  double cc_tolerance{1e-9};
  double lambda_tolerance{1e-9};
  std::size_t max_bytes{256ULL << 20};
  response::GmresOptions gmres = [] {
    response::GmresOptions options;
    options.relative_tolerance = 0.0;
    options.absolute_tolerance = 1e-11;
    return options;
  }();
};

struct LambdaDiagnostic {
  double cc_r1_max{};
  double cc_r2_max{};
  double lambda_residual_norm{};
  double independent_residual_norm{};
  double independent_residual_max{};
  std::size_t iterations{};
  std::size_t operator_actions{};
  std::size_t numeric_capacity_bytes{};
  std::size_t owned_device_bytes{};
  std::size_t h2d_bytes{};
  std::size_t d2h_bytes{};
  std::size_t synchronizations{};
  bool cuda_actions{};
  const char* shared_program_hash{};
  const char* independent_program_hash{};
};

struct LambdaResult {
  std::vector<double> lambda1;
  std::vector<double> lambda2;
  LambdaDiagnostic diagnostic;
  std::string reason;

  [[nodiscard]] bool converged() const noexcept { return !lambda1.empty() && !lambda2.empty(); }
};

void validate_lambda_options(const LambdaOptions& options);
/** Allocation-free total numeric bound, including borrowed CC/reference data.
 * Additional energy sources retain one packed projected RHS through the
 * independent residual check. Callers composing stages add their other live
 * owners separately rather than giving each stage the full endpoint budget.
 */
std::size_t lambda_cpu_numeric_capacity(const Problem& problem, const SolverResult& cc_result,
                                        const LambdaOptions& options, bool with_energy_source);
LambdaResult solve_lambda_cpu(const Problem& problem, const SolverResult& cc_result,
                              const LambdaOptions& options = {});
LambdaResult solve_lambda_cpu_with_energy_source(const Problem& problem,
                                                 const SolverResult& cc_result,
                                                 std::span<const double> t1_source,
                                                 std::span<const double> t2_source,
                                                 const LambdaOptions& options = {});

#if VIBEQC_HAS_CUDA
/** Solve RCCSD Lambda with generated RHS/J^T actions executed on CUDA.
 *
 * GMRES control and packed symmetry projection remain host-owned in this first
 * native residency slice. The generated scientific actions execute on the
 * selected CUDA device without a CPU response fallback.
 */
LambdaResult solve_lambda_cuda(const Problem& problem, const SolverResult& cc_result, int device,
                               const LambdaOptions& options = {});
LambdaResult solve_lambda_cuda_with_energy_source(const Problem& problem,
                                                  const SolverResult& cc_result,
                                                  std::span<const double> t1_source,
                                                  std::span<const double> t2_source, int device,
                                                  const LambdaOptions& options = {});

/** Corrected Lambda plus fixed-orbital RCCSD parameter VJPs from one CUDA state.
 *
 * The converged Problem/T1/T2 inputs are staged once. Lambda RHS/J^T actions and
 * all ten parameter VJPs then reuse that device state. Host GMRES control remains
 * unchanged; parameter outputs are detached to host for the later Hamiltonian
 * response owner.
 */
struct CudaFixedOrbitalResponseResult {
  LambdaResult lambda;
  std::vector<double> foo, fov, fvv, ovov, ovvo, oovv, ovvv, ovoo, oooo, vvvv;
};

CudaFixedOrbitalResponseResult solve_lambda_parameter_response_cuda_with_energy_source(
    const Problem& problem, const SolverResult& cc_result, std::span<const double> t1_source,
    std::span<const double> t2_source, int device, const LambdaOptions& options = {});

/** Borrowed host views for the generated Hamiltonian/Fock/orbital CUDA response programs. */
struct CudaRawHamiltonianView {
  std::span<const double> density, g, h, rotation;
};

struct CudaParameterResponseView {
  std::span<const double> foo, fov, fvv, ovov, ovvo, oovv, ovvv, ovoo, oooo, vvvv;
};

struct CudaHamiltonianResponseResult {
  std::vector<double> hcore, eri, overlap, rotation_gradient, stationarity, orbital_rhs;
};

/** Reusable native owner for post-Lambda Hamiltonian/Fock/orbital response TensorIR.
 *
 * Raw Hamiltonian inputs are staged once and retained on the selected device.
 * Scientific pullbacks/JVPs execute through the generated CUDA entry points;
 * callers explicitly decide when detached host weights are needed.
 */
class CudaHamiltonianResponseOwner {
 public:
  CudaHamiltonianResponseOwner(std::size_t nocc, std::size_t nvir,
                               CudaRawHamiltonianView raw, int device,
                               std::size_t max_device_bytes);
  ~CudaHamiltonianResponseOwner();

  CudaHamiltonianResponseOwner(const CudaHamiltonianResponseOwner&) = delete;
  CudaHamiltonianResponseOwner& operator=(const CudaHamiltonianResponseOwner&) = delete;

  CudaHamiltonianResponseResult hamiltonian(CudaParameterResponseView parameters,
                                            double reference_seed);
  CudaHamiltonianResponseResult fock(std::span<const double> bar_fock);
  std::vector<double> orbital_jvp(std::span<const double> d_rotation);

  [[nodiscard]] std::size_t owned_device_bytes() const noexcept;
  [[nodiscard]] std::size_t h2d_bytes() const noexcept;
  [[nodiscard]] std::size_t d2h_bytes() const noexcept;
  [[nodiscard]] std::size_t synchronizations() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace vibeqc::cc
