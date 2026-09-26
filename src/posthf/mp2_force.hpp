#pragma once

#include <cstddef>
#include <vector>

#include "response/native_gmres.hpp"

namespace vibeqc::posthf {
class RawSource;
}
namespace vibeqc::scf {
struct PhysicalReference;
}

namespace vibeqc::mp2 {

struct ConventionalForceResult {
  std::vector<double> forces;
  response::GmresResult response;
  double stationarity_residual{};
  std::size_t weighted_eri_shell_tiles{};
  std::size_t derivative_workspace_bytes{};
  std::size_t planned_endpoint_peak_bytes{};
  std::size_t measured_endpoint_peak_bytes{};
};

/** Complete conventional canonical RHF-MP2 analytic force on the CPU.
 *
 * The result is unpublished. The caller remains responsible for transactionally
 * copying it only after the full energy/force endpoint succeeds.
 */
ConventionalForceResult conventional_force_cpu(const scf::PhysicalReference& reference,
                                               const posthf::RawSource& source,
                                               std::size_t budget_bytes,
                                               double denominator_threshold,
                                               double same_space_threshold,
                                               const response::GmresOptions& response_options);

/** Complete conventional canonical RHF-MP2 analytic force on CUDA. */
ConventionalForceResult conventional_force_cuda(
    const scf::PhysicalReference& reference, const posthf::RawSource& source,
    std::size_t budget_bytes, double denominator_threshold, double same_space_threshold,
    const response::GmresOptions& response_options, int device_id);

/** Complete CPU RI-MP2 analytic force for the same fitted RHF Hamiltonian.
 *
 * The response uses the factorized MO provider, the metric reverse delegates
 * to the shared fixed-rank matrix-function VJP, and raw A/M cotangents are
 * consumed by the bounded generated DF derivative path.
 */
ConventionalForceResult density_fitted_force_cpu(
    const scf::PhysicalReference& reference, const posthf::RawSource& source,
    std::size_t budget_bytes, double denominator_threshold, double metric_relative_threshold,
    double same_space_threshold, const response::GmresOptions& response_options);

}  // namespace vibeqc::mp2
