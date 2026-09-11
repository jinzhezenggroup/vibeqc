#pragma once

#include <array>
#include <memory>

#include "core/types.hpp"

namespace vibeqc::posthf {

/** Values-only public-AO tiles from the independent contracted CPU evaluator.
 *
 * The source owns normalized basis/geometry data. It never allocates molecular
 * four-index tensors or nuclear derivative jets. Offsets are global public-AO
 * indices; the Python CG02 adapter additionally checks shell-local requests.
 */
class RawSource {
 public:
  enum class Operator { overlap, hcore, eri, metric, three_center };
  RawSource(core::System orbital, const core::System* auxiliary = nullptr);
  ~RawSource();
  RawSource(const RawSource&) = delete;
  RawSource& operator=(const RawSource&) = delete;
  const core::System& orbital() const;
  const core::System& auxiliary() const;
  std::size_t nbf() const;
  std::size_t naux() const;
  /** Evaluate exactly the requested row-major tile into caller-owned storage.
   * Empty extents at valid endpoints are legal. Invalid ranges and arithmetic
   * overflow fail before writes. Recurrence scratch is independent of tile size.
   */
  void read(Operator op, const std::array<std::size_t, 4>& begin,
            const std::array<std::size_t, 4>& count, double* out, std::size_t elements) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace vibeqc::posthf
