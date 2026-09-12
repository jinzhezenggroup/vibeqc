#pragma once

#include <array>
#include <vector>

#include "posthf/block_capacity_generated.hpp"
#include "posthf/raw_source.hpp"
#include "scf/types.hpp"
#include "tensor/metrics.hpp"

namespace vibeqc::posthf {
using MOSlots = std::array<std::vector<std::size_t>, 4>;
inline constexpr std::size_t padded_mo = static_cast<std::size_t>(-1);

/** Native consumer adapter of CG10's cyclic staged transformation. A private
 * all-zero coefficient column marks an energy tail, never a frozen orbital. */
class NativeBlockProvider {
 public:
  NativeBlockProvider(const RawSource& source, const scf::PhysicalReference& reference,
                      std::size_t budget, unsigned axis_tile = 2);
  NumericBlockPlan plan(const std::array<std::size_t, 4>& shape, bool cuda = false) const;
  std::vector<double> get(const MOSlots& slots, bool cuda = false, int device = 0,
                          vibeqc_tensor::Metrics* metrics = nullptr) const;
  std::size_t source_bytes() const { return source_bytes_; }
  std::size_t reference_bytes() const { return reference_bytes_; }
  const std::array<std::size_t, 4>& tile_shape() const { return tile_; }

 private:
  const RawSource& source_;
  const scf::PhysicalReference& ref_;
  std::size_t budget_, source_bytes_, reference_bytes_;
  std::array<std::size_t, 4> tile_;
};
}  // namespace vibeqc::posthf
