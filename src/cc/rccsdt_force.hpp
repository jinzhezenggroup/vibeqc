#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "cc/lambda_response.hpp"
#include "cc/solver.hpp"
#include "response/native_gmres.hpp"

namespace generativeqc::core {
struct System;
}
namespace generativeqc::hf {
struct PhysicalReference;
}
namespace generativeqc::integrals {
class ElectronInteractionSource;
}

namespace generativeqc::cc {

/** Public conventional CCSD(T) force qualification; CCSD keeps its own limit. */
inline constexpr std::size_t kRccsdtForceMaxAOs = 28;

/** Conservative simultaneous numeric peaks for the serialized force phases.
 * Every phase includes borrowed molecule/CC/reference inputs once and all earlier
 * outputs still retained by the force owner. Object headers and allocator
 * rounding follow the existing post-HF numeric-buffer convention.
 */
struct RccsdtForcePlan {
  std::size_t retained_input_bytes{};
  std::size_t triples_phase_bytes{}, lambda_phase_bytes{}, parameter_phase_bytes{};
  std::size_t raw_phase_bytes{}, response_phase_bytes{}, derivative_phase_bytes{};
  std::size_t raw_provider_budget_bytes{};
  // Selected within the complete endpoint budget; execution must use the
  // same basis-wide tile domain as admission, including partial final tiles.
  unsigned raw_provider_axis_tile{};
  std::size_t minimum_peak_bytes{};
  std::size_t peak_bytes{};
};

/** Plan/admit before allocating force buffers or invoking a generated kernel.
 * max_bytes is the complete endpoint allowance, including borrowed inputs.
 */
RccsdtForcePlan plan_rccsd_force_cpu(const core::System& system,
                                     const integrals::ElectronInteractionSource& source,
                                     const hf::PhysicalReference& reference, const Problem& problem,
                                     const SolverResult& cc_result, std::size_t max_bytes);
RccsdtForcePlan plan_rccsd_force_cpu(const core::System& system,
                                     const hf::PhysicalReference& reference, const Problem& problem,
                                     const SolverResult& cc_result, std::size_t max_bytes);
RccsdtForcePlan plan_rccsdt_force_cpu(const core::System& system,
                                      const integrals::ElectronInteractionSource& source,
                                      const hf::PhysicalReference& reference,
                                      const Problem& problem, const SolverResult& cc_result,
                                      std::size_t max_bytes);
RccsdtForcePlan plan_rccsdt_force_cpu(const core::System& system,
                                      const hf::PhysicalReference& reference,
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
  std::size_t raw_source_reads{}, raw_device_source_reads{}, raw_source_values{};
  std::size_t raw_transform_fmas{};
  double raw_source_seconds{}, raw_provider_seconds{};
  // Serialized native response phases, excluding the preceding energy solve.
  // Together with complete endpoint time these locate work amplification at
  // larger dimensions without treating a single faster phase as a speedup.
  double triples_seconds{}, lambda_parameter_seconds{}, orbital_seconds{}, derivative_seconds{};
  // Peak device capacity across the serialized triples and orbital owners.
  // Transfer and synchronization counters below describe the orbital owner;
  // triples transfers have their own completed-phase ledger fields.
  std::size_t response_owned_device_bytes{};
  std::size_t response_h2d_bytes{};
  std::size_t response_d2h_bytes{};
  std::size_t response_synchronizations{};
  bool cuda_response_actions{};
  std::string response_operator_hash;
};

/** Complete standard canonical closed-shell RCCSD(T) analytic force.
 *
 * This is the native/public closure of the already-qualified #155 graph.  It
 * composes generated CC/(T)/Hamiltonian adjoints and the shared RHF orbital
 * response, then reuses the generic conventional derivative consumer.  The
 * public domain is bounded to conventional all-electron references, through
 * 28 AOs for CCSD(T) and 12 AOs for CCSD. The CPU owner evaluates the complete
 * response and derivative on host. The CUDA owner runs generated Lambda,
 * parameter and Hamiltonian/orbital actions on device; GMRES control, triples
 * response and the final MO-to-AO weight pullback retain their host ownership.
 * The final nuclear derivative contraction uses the CUDA consumer. DF, frozen-core, ECP
 * and open-shell variants remain separate capabilities. max_bytes is the complete numeric
 * allowance, including the borrowed molecule, physical reference, CC problem/amplitudes
 * and occupied/virtual energy vectors, as in plan_rccsdt_force_cpu.
 */
RccsdtForceResult rccsd_force_cpu(const core::System& system,
                                  const integrals::ElectronInteractionSource& source,
                                  const hf::PhysicalReference& reference, const Problem& problem,
                                  const SolverResult& cc_result, std::span<const double> eps_o,
                                  std::span<const double> eps_v, std::size_t max_bytes);

RccsdtForceResult rccsd_force_cpu(const core::System& system,
                                  const hf::PhysicalReference& reference, const Problem& problem,
                                  const SolverResult& cc_result, std::span<const double> eps_o,
                                  std::span<const double> eps_v, std::size_t max_bytes);

RccsdtForceResult rccsdt_force_cpu(const core::System& system,
                                   const integrals::ElectronInteractionSource& source,
                                   const hf::PhysicalReference& reference, const Problem& problem,
                                   const SolverResult& cc_result, std::span<const double> eps_o,
                                   std::span<const double> eps_v, std::size_t max_bytes,
                                   double denominator_threshold = 1e-10);

RccsdtForceResult rccsdt_force_cpu(const core::System& system,
                                   const hf::PhysicalReference& reference, const Problem& problem,
                                   const SolverResult& cc_result, std::span<const double> eps_o,
                                   std::span<const double> eps_v, std::size_t max_bytes,
                                   double denominator_threshold = 1e-10);

/** Publish the qualified conventional RCCSD(T) force through a CUDA derivative consumer.
 *
 * Generated triples/Lambda actions plus Hamiltonian/Fock/orbital TensorIR execute on
 * CUDA in the promoted response path, while the physical Z/GMRES control flow
 * remains host-owned. device_id selects the CUDA response/derivative device;
 * derivative_stage_budget bounds each generated one-/two-electron derivative
 * consumer without authorizing a CPU fallback.
 */
RccsdtForceResult rccsd_force_cuda(const core::System& system,
                                   const integrals::ElectronInteractionSource& source,
                                   const hf::PhysicalReference& reference, const Problem& problem,
                                   const SolverResult& cc_result, std::span<const double> eps_o,
                                   std::span<const double> eps_v, std::size_t max_bytes,
                                   int device_id, std::size_t derivative_stage_budget);

RccsdtForceResult rccsd_force_cuda(const core::System& system,
                                   const hf::PhysicalReference& reference, const Problem& problem,
                                   const SolverResult& cc_result, std::span<const double> eps_o,
                                   std::span<const double> eps_v, std::size_t max_bytes,
                                   int device_id, std::size_t derivative_stage_budget);

RccsdtForceResult rccsdt_force_cuda(const core::System& system,
                                    const integrals::ElectronInteractionSource& source,
                                    const hf::PhysicalReference& reference, const Problem& problem,
                                    const SolverResult& cc_result, std::span<const double> eps_o,
                                    std::span<const double> eps_v, std::size_t max_bytes,
                                    int device_id, std::size_t derivative_stage_budget,
                                    double denominator_threshold = 1e-10);

RccsdtForceResult rccsdt_force_cuda(const core::System& system,
                                    const hf::PhysicalReference& reference, const Problem& problem,
                                    const SolverResult& cc_result, std::span<const double> eps_o,
                                    std::span<const double> eps_v, std::size_t max_bytes,
                                    int device_id, std::size_t derivative_stage_budget,
                                    double denominator_threshold = 1e-10);

}  // namespace generativeqc::cc
