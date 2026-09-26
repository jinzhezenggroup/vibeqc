#pragma once

#include <array>
#include <vector>

#include "integrals/electron_interaction_source.hpp"
#include "posthf/block_capacity_generated.hpp"
#include "posthf/raw_source.hpp"
#include "scf/types.hpp"
#include "tensor/metrics.hpp"

namespace vibeqc::posthf {
using MOSlots = std::array<std::vector<std::size_t>, 4>;
inline constexpr std::size_t padded_mo = static_cast<std::size_t>(-1);

struct ProviderWork {
  std::size_t source_scans{};
  std::size_t source_reads{};
  std::size_t source_values{};
  std::size_t transform_fmas{};
  std::size_t mo_blocks{};
  std::size_t cuda_transform_calls{};
  std::size_t cuda_batch_calls{};
  std::size_t h2d_bytes{};
  std::size_t d2h_bytes{};
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
                                  vibeqc_tensor::Metrics* metrics = nullptr) const = 0;
  virtual std::size_t provider_bytes() const noexcept = 0;
  virtual const scf::PhysicalReference& reference() const noexcept = 0;
};

/** Native consumer adapter of CG10's cyclic staged transformation. A private
 * all-zero coefficient column marks an energy tail, never a frozen orbital. */
class NativeBlockProvider final : public MOBlockProvider {
 public:
  NativeBlockProvider(const integrals::ElectronInteractionSource& source,
                      const scf::PhysicalReference& reference, std::size_t budget,
                      unsigned axis_tile = 2);
  NumericBlockPlan plan(const std::array<std::size_t, 4>& shape, bool cuda = false) const;
  std::size_t batch_bytes(const std::array<std::size_t, 4>& shape, std::size_t requests,
                          bool cuda = false) const;
  std::size_t batch_capacity(const std::array<std::size_t, 4>& shape, bool cuda = false) const;
  std::vector<std::vector<double>> get_many(const std::vector<MOSlots>& requests, bool cuda = false,
                                            int device = 0,
                                            vibeqc_tensor::Metrics* metrics = nullptr,
                                            ProviderWork* work = nullptr) const;
  std::vector<double> get(const MOSlots& slots, bool cuda = false, int device = 0,
                          vibeqc_tensor::Metrics* metrics = nullptr) const override {
    return get(slots, cuda, device, metrics, nullptr);
  }
  std::vector<double> get(const MOSlots& slots, bool cuda, int device,
                          vibeqc_tensor::Metrics* metrics, ProviderWork* work) const;
  std::size_t source_bytes() const noexcept { return source_bytes_; }
  std::size_t reference_bytes() const noexcept { return reference_bytes_; }
  std::size_t provider_bytes() const noexcept override { return source_bytes_ + reference_bytes_; }
  const std::array<std::size_t, 4>& tile_shape() const noexcept { return tile_; }
  const scf::PhysicalReference& reference() const noexcept override { return ref_; }
  const integrals::ElectronInteractionSource& source() const noexcept { return source_; }

 private:
  std::size_t common_host_bytes() const;
  const integrals::ElectronInteractionSource& source_;
  const scf::PhysicalReference& ref_;
  std::size_t budget_, source_bytes_, reference_bytes_;
  std::array<std::size_t, 4> tile_;
};

/** CPU factorized two-electron provider for the RI-MP2 response path.
 *
 * The provider materializes the value-side A[mu,nu,P], transforms it once to
 * the canonical MO frame, and stores both A[P,p,q] and B[Q,p,q]=A* M^(-1/2).
 * Four-index blocks are reconstructed on demand as sum_Q B[Q,p,q] B[Q,r,s].
 * No nuclear derivative tensor is owned here; the reverse consumer publishes
 * A/M cotangents to the shared #143 derivative boundary.
 */
class DensityFittedBlockProvider final : public MOBlockProvider {
 public:
  DensityFittedBlockProvider(const RawSource& source, const scf::PhysicalReference& reference,
                             std::size_t budget, double relative_threshold = 1.0e-10);
  std::vector<double> get(const MOSlots& slots, bool cuda = false, int device = 0,
                          vibeqc_tensor::Metrics* metrics = nullptr) const override;
  std::size_t provider_bytes() const noexcept override { return provider_bytes_; }
  const scf::PhysicalReference& reference() const noexcept override { return ref_; }
  const RawSource& source() const noexcept { return source_; }
  std::size_t auxiliary_count() const noexcept { return naux_; }
  double relative_threshold() const noexcept { return relative_threshold_; }
  const std::vector<double>& metric() const noexcept { return metric_; }
  const std::vector<double>& inverse_square_root() const noexcept { return inverse_square_root_; }
  const std::vector<double>& transformed_three_center() const noexcept { return transformed_; }
  const std::vector<double>& whitened_three_center() const noexcept { return whitened_; }

 private:
  const RawSource& source_;
  const scf::PhysicalReference& ref_;
  std::size_t budget_{}, n_{}, naux_{}, provider_bytes_{};
  double relative_threshold_{};
  std::vector<double> metric_, inverse_square_root_, transformed_, whitened_;
};
}  // namespace vibeqc::posthf
