#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "cc/lambda_response.hpp"
#include "cc/solver.hpp"
#include "response/native_gmres.hpp"

namespace vibeqc::core {
struct System;
}
namespace vibeqc::scf {
struct PhysicalReference;
}

namespace vibeqc::cc {

/** Conservative simultaneous numeric peaks for the serialized force phases.
 * Every phase includes borrowed molecule/CC/reference inputs once and all earlier
 * outputs still retained by the force owner. Object headers and allocator
 * rounding follow the existing post-HF numeric-buffer convention.
 */
struct RccsdtForcePlan {
  std::size_t retained_input_bytes{};
  std::size_t triples_phase_bytes{}, lambda_phase_bytes{}, parameter_phase_bytes{};
  std::size_t raw_phase_bytes{}, response_phase_bytes{}, derivative_phase_bytes{};
  std::size_t peak_bytes{};
};

/** Plan/admit before allocating force buffers or invoking a generated kernel.
 * max_bytes is the complete endpoint allowance, including borrowed inputs.
 */
RccsdtForcePlan plan_rccsdt_force_cpu(const core::System& system,
                                      const scf::PhysicalReference& reference,
                                      const Problem& problem, const SolverResult& cc_result,
                                      std::size_t max_bytes);

struct RccsdtForceResult {
  std::vector<double> forces;
  LambdaDiagnostic lambda;
  response::GmresResult orbital_response;
  double independent_orbital_residual{};
  double orbital_stationarity{};
  double minimum_orbital_curvature{};
  double minimum_same_space_gap{};
  std::size_t triples_response_pages{};
  std::size_t numeric_capacity_bytes{};
  bool cuda_derivative{};
  std::size_t derivative_stage_budget_bytes{};
  std::string response_operator_hash;
};

/** Complete standard canonical closed-shell RCCSD(T) analytic force.
 *
 * This is the native/public closure of the already-qualified #155 graph.  It
 * composes generated CC/(T)/Hamiltonian adjoints and the shared RHF orbital
 * response, then reuses the generic conventional derivative consumer.  The
 * first public domain is deliberately bounded to conventional all-electron CPU
 * references with at most 12 AOs; CUDA, DF, frozen-core, ECP and open-shell
 * variants remain separate capabilities. max_bytes is the complete numeric
 * allowance, including the borrowed molecule, physical reference, CC problem/amplitudes
 * and occupied/virtual energy vectors, as in plan_rccsdt_force_cpu.
 */
RccsdtForceResult rccsdt_force_cpu(const core::System& system,
                                   const scf::PhysicalReference& reference, const Problem& problem,
                                   const SolverResult& cc_result, std::span<const double> eps_o,
                                   std::span<const double> eps_v, std::size_t max_bytes,
                                   double denominator_threshold = 1e-10);

/** Hybrid native qualification: keep the complete validated RCCSD(T) response
 * owner on CPU, but contract its final AO/nuclear derivative weights through
 * the existing generated CUDA conventional derivative consumer.
 *
 * max_bytes retains the complete host response allowance from
 * plan_rccsdt_force_cpu. derivative_stage_budget_bytes is a separate per-stage
 * CUDA allowance owned by conventional_derivative_cuda; it is not a combined
 * endpoint device-memory cap. This function does not widen public Calculator
 * CUDA-force capability.
 */
RccsdtForceResult rccsdt_force_cuda_derivative(
    const core::System& system, const scf::PhysicalReference& reference,
    const Problem& problem, const SolverResult& cc_result, std::span<const double> eps_o,
    std::span<const double> eps_v, std::size_t max_bytes, int device_id,
    std::size_t derivative_stage_budget_bytes, double denominator_threshold = 1e-10);

}  // namespace vibeqc::cc
