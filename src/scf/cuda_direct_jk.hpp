#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "scf/fock_build.hpp"

namespace vibeqc::scf {
struct CudaDirectJkPlan;

/** Persistent source/scratch sizes for the generic direct provider. No molecular
 * four-index ERI or derivative tensor is retained. CUDA module/context and
 * compiler-managed recurrence stack storage are outside these buffer counts.
 */
struct CudaDirectJkDiagnostic {
  std::size_t batch_size{}, nbf{}, coordinates_per_item{};
  std::size_t device_bytes{}, host_bytes{}, host_preparation_bytes{};
  unsigned derivative_order{};
  double screening_tolerance{};
  const char* schedule{"generic-contracted-eri-public-ao"};
};

/** Bind normalized, homogeneous public AO dimensions and coordinate counts.
 * Each item retains its own shell/geometry metadata. A geometry or basis change
 * requires a new plan. A positive budget bounds explicit device allocations;
 * no host integrals are evaluated or uploaded by this preparation.
 */
vibeqc_status create_cuda_direct_jk_plan(int device_id, const std::vector<core::System>& systems,
                                         unsigned derivative_order, double screening_tolerance,
                                         std::size_t device_budget_bytes, CudaDirectJkPlan** output,
                                         CudaDirectJkDiagnostic& diagnostic, std::string& detail);
void destroy_cuda_direct_jk_plan(CudaDirectJkPlan* plan) noexcept;
/** Null handles return an empty diagnostic. */
CudaDirectJkDiagnostic cuda_direct_jk_plan_diagnostic(const CudaDirectJkPlan* plan) noexcept;

/** Return raw unscaled J/K in row-major [item,AO,AO] order. Densities use that
 * same layout; beta is empty for restricted spin. Absent outputs are empty.
 * Only density and requested raw matrices cross the host/device boundary.
 */
vibeqc_status execute_cuda_direct_jk(CudaDirectJkPlan* plan, FockBuildSpec spec,
                                     const std::vector<double>& density,
                                     const std::vector<double>& beta, std::vector<double>& coulomb,
                                     std::vector<double>& alpha_exchange,
                                     std::vector<double>& beta_exchange, std::string& detail);

/** Fixed-density two-electron gradient [item,coordinate], using the same
 * operators, coefficients and screened quartet set as the raw value provider.
 * Derivatives reuse the existing contracted-ERI Dual evaluator; nuclear,
 * one-electron and Pulay terms belong to the surrounding method.
 */
vibeqc_status execute_cuda_direct_energy_derivative(CudaDirectJkPlan* plan, FockBuildSpec spec,
                                                    const std::vector<double>& density,
                                                    const std::vector<double>& beta,
                                                    std::vector<double>& derivative,
                                                    std::string& detail);

/** Item-level counterparts preserve neighboring batch scratch and upload only
 * this item's densities. Returned arrays have no batch dimension. */
vibeqc_status execute_cuda_direct_jk_item(CudaDirectJkPlan* plan, std::size_t item,
                                          FockBuildSpec spec, const std::vector<double>& density,
                                          const std::vector<double>& beta,
                                          std::vector<double>& coulomb,
                                          std::vector<double>& alpha_exchange,
                                          std::vector<double>& beta_exchange, std::string& detail);
vibeqc_status execute_cuda_direct_energy_derivative_item(CudaDirectJkPlan* plan, std::size_t item,
                                                         FockBuildSpec spec,
                                                         const std::vector<double>& density,
                                                         const std::vector<double>& beta,
                                                         std::vector<double>& derivative,
                                                         std::string& detail);
}  // namespace vibeqc::scf
