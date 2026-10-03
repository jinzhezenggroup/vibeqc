#pragma once

#include <array>
#include <vector>

#include "hf/reference.hpp"
#include "integrals/electron_interaction_source.hpp"
#include "posthf/block_capacity_generated.hpp"
#include "posthf/raw_source.hpp"
#include "tensor/metrics.hpp"

namespace generativeqc::posthf {
using MOSlots = std::array<std::vector<std::size_t>, 4>;
inline constexpr std::size_t padded_mo = static_cast<std::size_t>(-1);

/** Candidate AO-tile extent, independent of the source's recurrence scratch.
 * Shell preserves existing consumers' small-tile admission contracts. Basis
 * permits cross-shell tiles; callers must select an admitted tile with plan()
 * or batch_bytes() before execution. Both use the same ordered AO source API.
 */
enum class AOTileDomain { Shell, Basis };

struct ProviderWork {
  std::size_t source_scans{};
  std::size_t source_reads{};
  std::size_t device_source_reads{};
  std::size_t source_values{};
  std::size_t transform_fmas{};
  std::size_t transform_stages{};
  std::size_t mo_blocks{};
  std::size_t cuda_transform_calls{};
  std::size_t cuda_batch_calls{};
  std::size_t h2d_bytes{};
  std::size_t d2h_bytes{};
  double source_seconds{};
  double provider_seconds{};
};

/** Common molecular-orbital two-electron block boundary used by response code.
 *
 * Providers may own conventional four-center or factorized RI data, but the
 * MP2 adjoint/Z-vector mathematics only consumes ordered chemist-notation
 * (pq|rs) blocks. A provider must never silently change the requested backend.
 */
class MOBlockProvider {
 public:
  virtual ~MOBlockProvider() = default;
  virtual std::vector<double> get(const MOSlots& slots, bool cuda = false, int device = 0,
                                  generativeqc_tensor::Metrics* metrics = nullptr) const = 0;
  virtual std::size_t provider_bytes() const noexcept = 0;
  virtual const hf::PhysicalReference& reference() const noexcept = 0;
};

/** Native consumer adapter of CG10's cyclic staged transformation. A private
 * all-zero coefficient column marks an energy tail, never a frozen orbital. */
class NativeBlockProvider final : public MOBlockProvider {
 public:
  NativeBlockProvider(const integrals::ElectronInteractionSource& source,
                      const hf::PhysicalReference& reference, std::size_t budget,
                      unsigned axis_tile = 2, AOTileDomain tile_domain = AOTileDomain::Shell);
  NumericBlockPlan plan(const std::array<std::size_t, 4>& shape, bool cuda = false) const;
  std::size_t batch_bytes(const std::array<std::size_t, 4>& shape, std::size_t requests,
                          bool cuda = false) const;
  std::size_t batch_capacity(const std::array<std::size_t, 4>& shape, bool cuda = false) const;
  std::vector<std::vector<double>> get_many(const std::vector<MOSlots>& requests, bool cuda = false,
                                            int device = 0,
                                            generativeqc_tensor::Metrics* metrics = nullptr,
                                            ProviderWork* work = nullptr) const;
  std::vector<double> get(const MOSlots& slots, bool cuda = false, int device = 0,
                          generativeqc_tensor::Metrics* metrics = nullptr) const override {
    return get(slots, cuda, device, metrics, nullptr);
  }
  std::vector<double> get(const MOSlots& slots, bool cuda, int device,
                          generativeqc_tensor::Metrics* metrics, ProviderWork* work) const;
  std::size_t source_bytes() const noexcept { return source_bytes_; }
  std::size_t reference_bytes() const noexcept { return reference_bytes_; }
  std::size_t provider_bytes() const noexcept override { return source_bytes_ + reference_bytes_; }
  const std::array<std::size_t, 4>& tile_shape() const noexcept { return tile_; }
  const hf::PhysicalReference& reference() const noexcept override { return ref_; }
  const integrals::ElectronInteractionSource& source() const noexcept { return source_; }

 private:
  std::size_t common_host_bytes() const;
  const integrals::ElectronInteractionSource& source_;
  const hf::PhysicalReference& ref_;
  std::size_t budget_, source_bytes_, reference_bytes_;
  std::array<std::size_t, 4> tile_;
};

/** CPU factorized two-electron provider for the RI-MP2 response path.
 *
 * The provider materializes the value-side A[mu,nu,P], transforms it once to
 * the canonical MO frame, and stores both A[p,q,P] and B[p,q,Q]=A* M^(-1/2).
 * The compiler stages the two orbital projections in O(N^3 Q) work; metric
 * whitening is O(N^2 Q^2). B storage holds the first projection temporarily.
 * Four-index blocks are reconstructed on demand as sum_Q B[p,q,Q] B[r,s,Q].
 * No nuclear derivative tensor is owned here; the reverse consumer publishes
 * A/M cotangents to the shared #143 derivative boundary.
 */
class DensityFittedBlockProvider final : public MOBlockProvider {
 public:
  DensityFittedBlockProvider(const RawSource& source, const hf::PhysicalReference& reference,
                             std::size_t budget, double relative_threshold = 1.0e-10);
  std::vector<double> get(const MOSlots& slots, bool cuda = false, int device = 0,
                          generativeqc_tensor::Metrics* metrics = nullptr) const override;
  std::size_t provider_bytes() const noexcept override { return provider_bytes_; }
  const hf::PhysicalReference& reference() const noexcept override { return ref_; }
  const RawSource& source() const noexcept { return source_; }
  std::size_t auxiliary_count() const noexcept { return naux_; }
  double relative_threshold() const noexcept { return relative_threshold_; }
  const std::vector<double>& metric() const noexcept { return metric_; }
  const std::vector<double>& inverse_square_root() const noexcept { return inverse_square_root_; }
  const std::vector<double>& transformed_three_center() const noexcept { return transformed_; }
  const std::vector<double>& whitened_three_center() const noexcept { return whitened_; }

 private:
  const RawSource& source_;
  const hf::PhysicalReference& ref_;
  std::size_t budget_{}, n_{}, naux_{}, provider_bytes_{};
  double relative_threshold_{};
  std::vector<double> metric_, inverse_square_root_, transformed_, whitened_;
};
}  // namespace generativeqc::posthf
