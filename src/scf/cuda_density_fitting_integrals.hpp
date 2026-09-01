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
 */
vibeqc_status build_cuda_density_fitting_integrals(int device_id,
                                                   const core::System& orbital_system,
                                                   const core::System& auxiliary_system,
                                                   integrals::DensityFittingIntegralData& output,
                                                   std::string& detail);

/**
 * Batched Cartesian DF generation for homogeneous orbital/auxiliary sizes.
 * Outputs are returned in input order; coordinate counts must match across
 * the batch so one derivative launch can serve every packed system.
 */
vibeqc_status build_cuda_density_fitting_integrals_batch(
    int device_id, const std::vector<core::System>& orbital_systems,
    const std::vector<core::System>& auxiliary_systems,
    std::vector<integrals::DensityFittingIntegralData>& outputs, std::string& detail,
    std::size_t output_budget_bytes = 0);

/** Batched Cartesian overlap/Hcore and nuclear-repulsion generation. */
vibeqc_status build_cuda_one_electron_integrals_batch(int device_id,
                                                      const std::vector<core::System>& systems,
                                                      std::vector<integrals::IntegralData>& outputs,
                                                      std::string& detail);

/** Generate Cartesian one-electron values and first nuclear derivatives. */
vibeqc_status build_cuda_one_electron_integrals(int device_id, const core::System& system,
                                                integrals::IntegralData& output,
                                                std::string& detail);

}  // namespace vibeqc::scf

#endif
