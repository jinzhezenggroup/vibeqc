#ifndef VIBEQC_SCF_MEAN_FIELD_HPP
#define VIBEQC_SCF_MEAN_FIELD_HPP

#include "core/types.hpp"
#include "scf/types.hpp"

#include <vector>

namespace vibeqc::scf {

/** Run closed-shell RHF and assemble its variational analytic gradient. */
ScfResult run_rhf(const core::System& system,
                  const ScfOptions& options,
                  const std::vector<double>* initial_density = nullptr);

/** Run spin-unrestricted HF and assemble its variational analytic gradient. */
ScfResult run_uhf(const core::System& system,
                  const ScfOptions& options,
                  const std::vector<double>* initial_density = nullptr);

/** Execute RHF through the native CUDA scientific path. */
ScfResult run_rhf_cuda(const core::System& system,
                       const ScfOptions& options,
                       int device_id,
                       const std::vector<double>* initial_density = nullptr);

/** Execute UHF through the native CUDA scientific path. */
ScfResult run_uhf_cuda(const core::System& system,
                       const ScfOptions& options,
                       int device_id,
                       const std::vector<double>* initial_density = nullptr);

}  // namespace vibeqc::scf

#endif
