#ifndef VIBEQC_SCF_CUDA_ONE_ELECTRON_GRADIENT_HPP
#define VIBEQC_SCF_CUDA_ONE_ELECTRON_GRADIENT_HPP

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "core/types.hpp"

namespace vibeqc::scf {

/** Explicit staging boundary of the standalone generic gradient operation. */
struct OneElectronGradientResources {
  std::size_t device_bytes{}, host_numeric_bytes{}, host_to_device_bytes{}, device_to_host_bytes{};
  std::size_t synchronous_uploads{}, stream_synchronizations{};
};

/** Contract arbitrary full public-AO S/T/V weights on CUDA with bounded storage.
 * Empty spans mean zero. Coefficients are already normalized by the system
 * validator; both Cartesian and real-spherical s/p/d/f AOs are supported.
 * Only metadata and weights cross H2D; only 3*Natom gradient scalars cross D2H.
 * Maximum_bytes bounds this operation's numeric host staging and device arena
 * separately, excluding caller-owned weights, system and existing HF plans.
 * Output is replaced only on success; it may alias a caller-owned weight span.
 * This synchronous host bridge is used by standalone DF and diagnostics;
 * prepared Direct HF invokes the same kernel on its existing owning stream.
 */
vibeqc_status execute_cuda_one_electron_gradient(int device_id, const core::System& system,
                                                 std::span<const double> overlap_weights,
                                                 std::span<const double> kinetic_weights,
                                                 std::span<const double> attraction_weights,
                                                 unsigned schedule, std::size_t maximum_bytes,
                                                 std::vector<double>& gradient, std::string& detail,
                                                 OneElectronGradientResources* resources = nullptr,
                                                 double overlap_scale = 1.0);

}  // namespace vibeqc::scf

#endif
