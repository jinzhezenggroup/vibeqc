#ifndef VIBEQC_SCF_CUDA_DENSITY_FITTING_INTEGRALS_HPP
#define VIBEQC_SCF_CUDA_DENSITY_FITTING_INTEGRALS_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "integrals/s_integrals.hpp"

namespace vibeqc::scf {

/**
 * Generate Cartesian DF metric and three-center values/derivatives on CUDA.
 *
 * The returned tensor uses Cartesian AO ordering for both systems.  Public
 * spherical representations can be obtained with
 * `integrals::transform_density_fitting_integrals`, which deliberately keeps
 * the accelerator evaluator independent from the reference transformation.
 * include_derivatives=false omits both complete dM and dA arrays for a fused
 * response consumer; value arrays and the physical coordinate count remain.
 */
vibeqc_status build_cuda_density_fitting_integrals(int device_id,
                                                   const core::System& orbital_system,
                                                   const core::System& auxiliary_system,
                                                   integrals::DensityFittingIntegralData& output,
                                                   std::string& detail,
                                                   bool include_derivatives = true);

/**
 * Batched Cartesian DF generation for homogeneous orbital/auxiliary sizes.
 * Outputs are returned in input order; coordinate counts must match across
 * the batch so one derivative launch can serve every packed system.
 */
vibeqc_status build_cuda_density_fitting_integrals_batch(
    int device_id, const std::vector<core::System>& orbital_systems,
    const std::vector<core::System>& auxiliary_systems,
    std::vector<integrals::DensityFittingIntegralData>& outputs, std::string& detail,
    std::size_t output_budget_bytes = 0, bool include_derivatives = true);

/** Batched Cartesian overlap/Hcore and nuclear-repulsion generation.
 * include_derivatives=false omits only AO derivative matrices, retaining the
 * independent O(Natom) nuclear-repulsion response for a fused consumer. */
vibeqc_status build_cuda_one_electron_integrals_batch(int device_id,
                                                      const std::vector<core::System>& systems,
                                                      std::vector<integrals::IntegralData>& outputs,
                                                      std::string& detail,
                                                      bool include_derivatives = true);

/** Generate Cartesian one-electron values and optional first nuclear derivatives.
 * include_derivatives controls AO response matrices. Nuclear response remains
 * available to fused force consumers unless include_nuclear_derivatives=false;
 * energy-only callers disable both flags to omit all derivative work. */
vibeqc_status build_cuda_one_electron_integrals(int device_id, const core::System& system,
                                                integrals::IntegralData& output,
                                                std::string& detail,
                                                bool include_derivatives = true,
                                                bool include_nuclear_derivatives = true);

}  // namespace vibeqc::scf

#endif
