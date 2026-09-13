#ifndef VIBEQC_SCF_CUDA_DF_GRADIENT_HPP
#define VIBEQC_SCF_CUDA_DF_GRADIENT_HPP
#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "runtime/strided_range.hpp"
#include "scf/df_response_weights.hpp"
namespace vibeqc::scf {
struct CudaDensityFittingIntegralSource;
/** Borrowed device factors for one fixed-geometry metric. Eigenvectors use
 * cuSOLVER column-major order; X=M^(-1/2) is symmetric. All pointers and the
 * retained cutoff belong to the same plan and outlive its owning stream work.
 */
struct CudaDfMetricView {
  const double* inverse_square_root{};
  const double* eigenvectors{};
  const double* eigenvalues{};
  double relative_threshold{};
};
/** Owned numeric staging and explicit transfers, excluding caller weights/system data. */
struct DfGradientResources {
  std::size_t host_bytes{}, device_bytes{}, host_to_device_bytes{}, device_to_host_bytes{};
  std::size_t weight_tile_elements{}, tiles{}, uploads{}, stream_synchronizations{};
  std::size_t value_slices{}, auxiliary_weight_tile{};
  /** Bulk tensors and response weights, separately from metadata/density/final results. */
  std::size_t tensor_host_to_device_bytes{}, tensor_device_to_host_bytes{};
  std::size_t response_host_to_device_bytes{}, density_host_to_device_bytes{};
  std::size_t recomputed_value_bytes{}, device_response_bytes{};
  bool device_response{};
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
/** Contract one caller-owned A/M weight tile into a detached partial gradient.
 * kind=0 maps
 * A[mu,nu,P], kind=1 maps M[P,Q]. The strided range addresses the
 * corresponding full row-major
 * tensor without padding a molecular weight.
 * This call owns only bounded metadata/device staging
 * and one O(Natom) result.
 */
vibeqc_status execute_cuda_df_gradient_tile(int device, const core::System& orbital,
                                            const core::System& auxiliary, unsigned kind,
                                            runtime::StridedRange range,
                                            std::span<const double> weights, unsigned schedule,
                                            std::size_t maximum_bytes,
                                            std::vector<double>& gradient, std::string& detail,
                                            DfGradientResources* resources = nullptr);
/** HF adapter using the same generic derivative consumer on the plan's stream.
 * Values are borrowed from raw_a when resident, or regenerated from source.
 * A source plus device_metric uses only device tensor/response contractions;
 * the compatibility raw-value adapter reports its bounded host staging.
 * No coordinate-indexed A/M derivative tensors are formed or downloaded.
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
    std::vector<double>& gradient, std::string& detail, DfGradientResources* resources = nullptr,
    const CudaDfMetricView* device_metric = nullptr);
}  // namespace vibeqc::scf
#endif
