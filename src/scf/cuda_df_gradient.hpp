#ifndef VIBEQC_SCF_CUDA_DF_GRADIENT_HPP
#define VIBEQC_SCF_CUDA_DF_GRADIENT_HPP
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "scf/df_response_weights.hpp"
namespace vibeqc::scf {
struct CudaDensityFittingIntegralSource;
/** Owned numeric staging and explicit transfers, excluding caller weights/system data. */
struct DfGradientResources {
  std::size_t host_bytes{}, device_bytes{}, host_to_device_bytes{}, device_to_host_bytes{};
  std::size_t weight_tile_elements{}, tiles{}, uploads{}, stream_synchronizations{};
  std::size_t value_slices{}, auxiliary_weight_tile{};
};
/** Synchronous bounded bridge for fixed full A[mu,nu,P] and M[P,Q] weights.
 * Orbital/auxiliary bases share the same physical atom coordinates but may
 * attach their shells to distinct atoms. Missing weight spans mean zero.
 * Each domain is tiled independently; maximum_bytes bounds host numeric
 * staging and device allocations separately. Output changes only on success.
 */
vibeqc_status execute_cuda_df_gradient(int device, const core::System& orbital,
                                       const core::System& auxiliary, std::span<const double> bar_a,
                                       std::span<const double> bar_m, unsigned schedule,
                                       std::size_t maximum_bytes, std::size_t maximum_tile_elements,
                                       std::vector<double>& gradient, std::string& detail,
                                       DfGradientResources* resources = nullptr);
/** HF adapter using the same generic derivative consumer on the plan's stream.
 * Values are borrowed from raw_a when resident, or regenerated from source.
 * Host staging for HF weights/metric response remains explicit and bounded;
 * no coordinate-indexed A/M derivative tensors are formed or downloaded.
 * maximum_bytes bounds this bridge's numeric host/device scratch separately
 * from the caller's plan; an active resource ledger also enforces total device
 * ownership. Output is transactional and generated failures are returned.
 */
vibeqc_status execute_cuda_df_hf_gradient(
    int device, void* stream, CudaDensityFittingIntegralSource* source, std::size_t source_index,
    const core::System& orbital, const core::System& auxiliary, std::span<const double> raw_a,
    const std::vector<double>& metric, const std::vector<double>& inverse,
    std::span<const DensityFittingDensityResponse> terms, double relative_threshold,
    unsigned schedule, std::size_t maximum_bytes, std::size_t maximum_auxiliary_tile,
    std::vector<double>& gradient, std::string& detail, DfGradientResources* resources = nullptr);
}  // namespace vibeqc::scf
#endif
