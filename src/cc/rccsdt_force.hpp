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
  std::string response_operator_hash;
};

/** Complete standard canonical closed-shell RCCSD(T) analytic force.
 *
 * This is the native/public closure of the already-qualified #155 graph.  It
 * composes generated CC/(T)/Hamiltonian adjoints and the shared RHF orbital
 * response, then reuses the generic conventional derivative consumer.  The
 * first public domain is deliberately bounded to conventional all-electron CPU
 * references with at most 12 AOs; CUDA, DF, frozen-core, ECP and open-shell
 * variants remain separate capabilities.
 */
RccsdtForceResult rccsdt_force_cpu(const core::System& system,
                                   const scf::PhysicalReference& reference, const Problem& problem,
                                   const SolverResult& cc_result, std::span<const double> eps_o,
                                   std::span<const double> eps_v, std::size_t max_bytes,
                                   double denominator_threshold = 1e-10);

}  // namespace vibeqc::cc
